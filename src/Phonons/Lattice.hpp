#ifndef LATTICE_HPP
#define LATTICE_HPP

#include <array>
#include <complex>
#include <vector>
#include "Operator/Operator.hpp"
#include "GridStructure/GridStructure.hpp"
#include "Parallel/Decomposition.hpp"
#include "Electrons/System.hpp"
#include "MeanField/MeanField.hpp"
#include "Model/Model.hpp"
#include "Phonons/PhononParameters.hpp"

namespace phonon {

/// @brief Coordinates of the lattice propagated with the electrons: for each phonon wavevector q and each
/// cartesian displacement mu = 3*atom + direction, the displacement u (bohr) and its time derivative (bohr/a.u.).
/// Complex, to hold the Fourier components at q != 0; at q = 0 they are real.
class Coordinates
{
    private:
        /// (0 = displacement or 1 = velocity, q point, mu)
        mdarray<std::complex<double>, 3> data_;

    public:
        Coordinates() = default;
        void initialize(const int num_qpoints__, const int num_modes__)
        {
            data_.initialize({2, num_qpoints__, num_modes__});
            data_.fill(0.);
        }

        std::complex<double>& displacement(const int iq__, const int mu__) { return data_(0, iq__, mu__); }
        const std::complex<double>& displacement(const int iq__, const int mu__) const { return data_(0, iq__, mu__); }
        std::complex<double>& velocity(const int iq__, const int mu__) { return data_(1, iq__, mu__); }
        const std::complex<double>& velocity(const int iq__, const int mu__) const { return data_(1, iq__, mu__); }

        int num_qpoints() const { return data_.get_Size(1); }
        int num_modes() const { return data_.get_Size(2); }

        /* what the time steppers need */
        auto begin() { return data_.begin(); }
        auto end() { return data_.end(); }
        auto begin() const { return data_.begin(); }
        auto end() const { return data_.end(); }
        void fill(const std::complex<double>& value__) { data_.fill(value__); }
};

/// @brief The lattice at one time in the normal modes lambda of the force constants of ph.x (see Lattice::project_on_modes)
struct ModeProjection
{
    /// Normal coordinate Q (bohr sqrt(m_e)) and its time derivative
    std::vector<double> Q, dQ;
    /// Force of the electrons on Q (Ha/(bohr sqrt(m_e))): Q'' = -omega^2 Q - F
    std::vector<double> F;
    /// Energy of the mode, 1/2 dQ^2 + 1/2 omega^2 Q^2 (Ha), and number of phonons E/omega (0 if |omega| < 1 cm^-1)
    std::vector<double> energy, population;
};

/// @brief Lattice coupled to the electrons at the mean-field (Ehrenfest) level, see src/Phonons/EHRENFEST.md.
///
/// The coordinates are the cartesian displacements of the atoms, @f$ \mu = (\kappa,\alpha) @f$. For q = 0:
/// @f[ M_\mu \ddot u_\mu = -\sum_\nu K_{\mu\nu} u_\nu - F_\mu - \frac{2M_\mu}{\tau}\dot u_\mu, \qquad
///     F_\mu = \frac{s}{N}\sum_{\mathbf k}\mathrm{Tr}\big[g^b_\mu(\mathbf k)\,(\rho(\mathbf k)-\rho_{BO}(\mathbf k))\big] @f]
/// with K the Born-Oppenheimer force constants of ph.x (Ha/bohr^2), which already contain the static screening of the
/// lattice by the electrons: only the excited part @f$ \rho-\rho_{BO}(u) @f$ pushes the atoms (see AdiabaticReference).
/// @f$ g^b_\mu = \partial H/\partial u_\mu @f$ is the bare electron-phonon coupling of EPW (Wannier gauge, Ha/bohr),
/// M the mass of the atom, s the spin degeneracy and tau the damping time. The electrons feel
/// - only g_b given: @f$ H = H_e + \sum_\mu u_\mu g^b_\mu + \Sigma[\rho-\rho_0] @f$, the mean field generates the screening;
/// - g_s given too: @f$ H = H_e + \sum_\mu u_\mu g^s_\mu + \Sigma[\rho-\rho_{BO}] @f$, the static screening is in g_s
///   and the mean field acts only on the excited part.
/// rho_BO is
/// - static: @f$ \rho_0 + \sum_\mu u_\mu\,\delta\rho_\mu @f$ with the static linear response of the electrons of the
///   model, so that the force is @f$ F[\rho-\rho_0] - \Pi u @f$, @f$ \Pi_{\mu\nu} = \frac{s}{N}\sum_k \mathrm{Tr}[g^b_\mu\,\delta\rho_\nu] @f$;
/// - dynamic: propagated by the Propagator with the same u(t) and no laser.
/// Host only.
class Lattice
{
    private:
        PhononParameters parameters_;
        int num_atoms_ = 0;
        int num_modes_ = 0;
        int num_bands_ = 0;
        /// Mass of the atom of each mode (electron masses)
        std::vector<double> mass_;
        /// Force constants of ph.x used in the dynamics, for each q point: (nq, mu, nu), Ha/bohr^2
        mdarray<std::complex<double>, 3> force_constants_;
        /// Frequencies of the modes from the force constants of ph.x (a.u.; negative if unstable)
        std::vector<double> frequencies_input_;
        /// Frequencies of the bare lattice K - Pi (static reference only, for the recap)
        std::vector<double> frequencies_bare_;
        /// Normal modes of the force constants of ph.x: omega^2 (a.u.) and real orthonormal eigenvectors
        /// e(mu, lambda) of D = M^{-1/2} K M^{-1/2}, for the projection of the dynamics on the modes
        std::vector<double> mode_omega2_;
        mdarray<double, 2> mode_vectors_;
        /// Largest difference between the Hamiltonian of EPW and the one of the tight-binding model (Ha), for the
        /// calculations of the bare and of the screened coupling
        double gauge_check_ = 0.;
        double gauge_check_screened_ = 0.;
        /// Correction of the acoustic sum rule (Ha/bohr^2)
        double asr_correction_ = 0.;
        /// Static response: iterations needed (largest over the modes) and final relative residual
        int response_iterations_ = 0;
        double response_error_ = 0.;
        /// Largest max|sum_atom g_(atom, alpha)| / max|g| removed from the coupling (acoustic_sum_rule)
        double translation_coupling_ = 0.;
        /// Check of the acoustic sum rule of Pi: largest |sum_atom' Pi(atom alpha, atom' beta)| (Ha/bohr^2), ~0
        double pi_asr_residual_ = 0.;
        /// With g_s: max|g_s - Sigma[chi0 g_s] - g_b| / max|g_b| (see screening_mismatch)
        double screening_mismatch_ = 0.;
        /// Static reference: electronic force constants Pi (mu, nu), Ha/bohr^2, and static response delta rho_mu per
        /// unit displacement (R components, Wannier gauge), rho_BO = rho0 + u.delta rho
        mdarray<std::complex<double>, 2> Pi_;
        std::vector<Operator<std::complex<double>>> response_;
        /// Bare electron-phonon coupling g_b at q = 0 on the grid of the simulation, one Operator per mode (Ha/bohr)
        std::vector<Operator<std::complex<double>>> coupling_;
        /// Screened coupling g_s, same layout; empty if not given
        std::vector<Operator<std::complex<double>>> coupling_screened_;
        /// Communicator for the sums over the R points
        const mpi::Communicator* comm_ = nullptr;

        /// Coupling of the EPW calculation in directory__ at q = 0 on the grid, in coupling__; checks that EPW and the
        /// tb model have the same Wannier functions (largest |H_EPW - H_tb| in gauge_check__)
        void read_coupling(const std::string& directory__, const std::string& epmatwp__,
                           const GridStructure& gridstructure__, const parallel::Decomposition& decomposition__,
                           Material& material__, std::vector<Operator<std::complex<double>>>& coupling__,
                           double& gauge_check__);
        /// g_(atom, alpha) -= M_atom/M_total sum_atom' g_(atom', alpha): a rigid translation does not couple to the
        /// electrons (unchanged along the displacements that keep the center of mass fixed); needs mass_.
        /// Returns max|sum_atom g| / max|g| before the correction
        double remove_translation_from_coupling(std::vector<Operator<std::complex<double>>>& coupling__) const;
        /// max|g_s - Sigma[chi0 g_s] - g_b| / max|g_b|: 0 if the screening of the model in g_s is the one of its mean field
        double screening_mismatch(const GridStructure& gridstructure__, const parallel::Decomposition& decomposition__,
                                  const electron::System& system__, electron::MeanField* meanfield__) const;
        /// Static linear response of the electrons of the model to the displacements at q = 0, and the resulting
        /// electronic force constants @f$ \Pi_{\mu\nu} = \frac{s}{N}\sum_k\mathrm{Tr}[g_\mu\,\delta\rho_\nu] @f$ (Ha/bohr^2).
        /// @f$ \delta\rho_\nu @f$ is the self-consistent solution of (Bloch gauge)
        /// @f$ \delta\rho_{nm} = \frac{f_n-f_m}{\varepsilon_n-\varepsilon_m}\,\big(g_\nu + \Sigma[\delta\rho]\big)_{nm} @f$,
        /// with the self energy of the mean field (one step without it; @f$ \chi_0 g^s_\nu @f$ with the screened coupling).
        /// Fills response_. Pi satisfies the acoustic sum rule if acoustic_sum_rule is set
        mdarray<std::complex<double>, 2> static_response(const GridStructure& gridstructure__,
                                                          const parallel::Decomposition& decomposition__,
                                                          const electron::System& system__,
                                                          electron::MeanField* meanfield__);
        /// Normal modes of the force constants K__: omega^2 in ascending order and eigenvectors e(mu, lambda)
        void normal_modes(const mdarray<std::complex<double>, 2>& K__, std::vector<double>& omega2__,
                          mdarray<double, 2>& vectors__) const;
        /// Frequencies of the modes (a.u.) for the force constants K__
        std::vector<double> frequencies(const mdarray<std::complex<double>, 2>& K__) const;

    public:
        Lattice() = default;
        /// Reads the EPW files and the dynamical matrix. It needs H0 of the material already on the k grid of the
        /// simulation (after System::initialize) and the eigenvectors in Operator::EigenVectors
        void initialize(const PhononParameters& parameters__, const GridStructure& gridstructure__,
                        const parallel::Decomposition& decomposition__, const electron::System& system__,
                        Material& material__, electron::MeanField* meanfield__);

        /// Allocates the coordinates and sets the initial displacement (lattice at rest)
        void initial_condition(Coordinates& x__) const;
        /// H__(R) += sum_mu u_mu g_mu(R), g_s if given, g_b otherwise (the R component of H__ must be up to date)
        void add_coupling(Operator<std::complex<double>>& H__, const Coordinates& x__) const;
        /// Static reference: R component of rho_bo__ = rho0__ + u.delta rho
        void adiabatic_density(Operator<std::complex<double>>& rho_bo__, const Coordinates& x__,
                               const Operator<std::complex<double>>& rho0__) const;
        /// Static reference: 1/2 u.Pi.u, energy of the adiabatic response (0 with the dynamic reference)
        double adiabatic_energy(const Coordinates& x__) const;
        /// Force on each mode at q = 0 (Ha/bohr) with g_b, from the R components of rho__ and of the reference
        /// reference__ (rho0, or rho_BO for the force used in the dynamics):
        /// @f$ F_\mu = s\sum_{\mathbf R}\sum_{mn} g_{\mu,mn}(\mathbf R)\,(\rho-\rho_{\rm ref})_{mn}(\mathbf R)^* @f$
        std::vector<double> force(const Operator<std::complex<double>>& rho__, const Operator<std::complex<double>>& reference__) const;
        /// Time derivative of the coordinates for the force F__
        void derivative(Coordinates& dx__, const Coordinates& x__, const std::vector<double>& F__) const;
        /// Kinetic and potential energy of the lattice per unit cell (Ha)
        double energy(const Coordinates& x__) const;
        /// Coordinates, force F__ (the one of force()) and energy of each normal mode of the force constants of ph.x
        ModeProjection project_on_modes(const Coordinates& x__, const std::vector<double>& F__) const;
        /// Frequency (a.u., negative if unstable) and eigenvector e(mu, lambda) of the normal mode lambda of ph.x
        double mode_frequency(const int lambda__) const { return frequencies_input_[lambda__]; }
        double mode_vector(const int mu__, const int lambda__) const { return mode_vectors_(mu__, lambda__); }

        void print_recap() const;

        const PhononParameters& parameters() const { return parameters_; }
        bool dynamic_reference() const { return parameters_.adiabatic_reference == AdiabaticReference::dynamic; }
        bool static_reference() const { return parameters_.adiabatic_reference == AdiabaticReference::static_response; }
        /// True if the electrons feel g_s and Sigma[rho - rho_BO]
        bool screened() const { return parameters_.screened(); }
        int num_modes() const { return num_modes_; }
        int num_qpoints() const { return int(parameters_.qpoints.size()); }
        const std::vector<double>& mass() const { return mass_; }
        /// Bare coupling g_b of the mode mu__ (force on the lattice)
        const Operator<std::complex<double>>& coupling(const int mu__) const { return coupling_[mu__]; }
        /// Coupling felt by the electrons: g_s if given, g_b otherwise
        const Operator<std::complex<double>>& electronic_coupling(const int mu__) const
        {
            return parameters_.screened() ? coupling_screened_[mu__] : coupling_[mu__];
        }
};

}

#endif
