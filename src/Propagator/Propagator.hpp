#ifndef PROPAGATOR_HPP
#define PROPAGATOR_HPP

#include "DESolver/DESolver.hpp"
#include "kGradient/kGradient.hpp"
#include "Laser/Laser.hpp"
#include "GridStructure/GridStructure.hpp"
#include "Electrons/System.hpp"
#include "Electrons/State.hpp"
#include "MeanField/MeanField.hpp"
#include "Phonons/Lattice.hpp"
#include "Propagator/PropagatorParameters.hpp"

/// @brief Time propagation of the electronic density matrix. It defines the equation of motion
/// @f[
/// \frac{\partial \rho}{\partial t} = -i [H_0 + H_{\text{eff}} + \boldsymbol{\varepsilon}(t)\cdot \Xi, \rho] +
/// \boldsymbol{\varepsilon}(t)\nabla_\textbf{k} \rho
/// @f]
/// and advances it in time with DESolver. With phonons (Ehrenfest dynamics) the Hamiltonian gets the term
/// @f$ \sum_\mu u_\mu g_\mu @f$ and the coordinates of the lattice are propagated together with the density matrix
/// (see phonon::Lattice). The class does not own the physical objects:
/// it only keeps pointers to them, so they must outlive the Propagator.
class Propagator : public CommutatorEquation
{
    private:
        PropagatorParameters parameters_;
        /// Immutable part of the electronic system (H0, r, DM0, ...)
        const electron::System* system_ = nullptr;
        /// Propagated variables (density matrix, H, auxiliary arrays)
        electron::State* state_ = nullptr;
        /// Effective Hamiltonian from the electron-electron interaction
        electron::MeanField* meanfield_ = nullptr;
        /// Lasers; not const because the vector potential is integrated in time
        SetOfLaser* lasers_ = nullptr;
        /// Lattice coupled to the electrons; nullptr without phonons
        const phonon::Lattice* lattice_ = nullptr;
        /// Force of the electrons on the lattice, computed with the last Hamiltonian
        std::vector<double> force_;
        /// Density matrix with respect to which the force is computed: rho0, or rho_BO in the dynamic reference
        const Operator<std::complex<double>>* reference_ = nullptr;
        /// Gradient in k, shared with System (used for the velocity)
        const kGradient* kgradient_ = nullptr;
        /// R grid centered in Gamma, used for the Peierls phase
        std::shared_ptr<MeshGrid> Rgrid_gamma_;
        /// Same as Rgrid_gamma_ (local part, cartesian), in a contiguous array for the gpu
        mdarray<double,2> bare_Rgrid_gamma_;

        /// Driver for the time propagation, it defines how we solve the differential equations
        DESolver<Operator<std::complex<double>>> desolver_;
        /// Driver for the time propagation of density matrix and lattice together (with phonons)
        DESolver<EhrenfestState> desolver_ehrenfest_;
        /// Equation of motion of density matrix and lattice, given to desolver_ehrenfest_
        std::unique_ptr<EquationOfMotion<EhrenfestState>> ehrenfest_equation_;
        /// Space where we calculate the commutator @f$ [H, \rho] @f$
        Space propagation_space_ = k;
        /// Space where we evaluate H=H0+E \cdot r
        Space ipa_hamiltonian_space_ = R;
        /// Processor where to run the heavy parts of the propagation
#ifdef EDUS_GPU
        Processor processor_ = device;
#else
        Processor processor_ = host;
#endif

        /// Initial condition: the equilibrium density matrix
        void initial_condition(Operator<std::complex<double>>& DM__);
        /// omega_max*dt over the stability limit of the time stepper (must be < 1)
        double stability_ratio_ = 0.;
        /// Stops if the time step is too large for the stability of the time stepper
        void check_stability();
        /// H_ of the state at time__ for the density matrix DM__ (with its R component up to date), in R:
        /// H0 + E.r + Sigma (+ sum_mu u_mu g_mu if lattice__ is given, also computing force_), with the Peierls
        /// phase if needed
        /// Without the field (with_field__ = false, for rho_BO) there are no laser, gradient term and Peierls phase,
        /// and the force is not computed.
        void build_hamiltonian(const double& time__, const Operator<std::complex<double>>& DM__,
                               const phonon::Coordinates* lattice__ = nullptr, const bool with_field__ = true);
        /// Time derivative of the density matrix, with the lattice at the coordinates lattice__ (if not nullptr),
        /// with or without the field
        void derivative_electrons(Operator<std::complex<double>>& Output__, const double& time__,
                                  const Operator<std::complex<double>>& Input__, const phonon::Coordinates* lattice__,
                                  const bool with_field__);
        std::string stepper_name() const;
        double stepper_stability_limit() const;

    public:
        Propagator() = default;
        /// DESolver keeps a pointer to this object (the equation of motion): it cannot be copied or moved
        Propagator(const Propagator&) = delete;
        Propagator& operator=(const Propagator&) = delete;

        /// Right hand side of the equation of motion, for DESolver
        void derivative(Operator<std::complex<double>>& Output__, const double& time__,
                        const Operator<std::complex<double>>& Input__) override;
        /// Right hand side of the equations of motion of density matrix and lattice (Ehrenfest dynamics)
        void derivative(EhrenfestState& Output__, const double& time__, const EhrenfestState& Input__);
        /// Hamiltonian of the commutator form (k component), for the Magnus time stepper
        void hamiltonian(Operator<std::complex<double>>& H__, const double& time__,
                         const Operator<std::complex<double>>& DM__) override;
        /// With the Peierls phase and without decay the equation is d(rho)/dt = -i[H, rho]
        bool commutator_form() const override { return parameters_.peierls && parameters_.decay <= 1.e-07; }
        /// H depends on rho through the mean field
        bool state_dependent() const override { return meanfield_->parameters().enabled; }

        void initialize(const PropagatorParameters& parameters__,
                        const GridStructure& gridstructure__,
                        const electron::System& system__,
                        electron::State& state__,
                        electron::MeanField& meanfield__,
                        SetOfLaser& lasers__,
                        const kGradient& kgradient__,
                        const phonon::Lattice* lattice__ = nullptr);

        /// Advances the density matrix of one time step
        void step();

        /// Sets the hamiltonian of the state to H0 + E(t) \cdot r (independent particles)
        void ipa_hamiltonian(const double& time__, const bool with_field__ = true);
        /// Multiplies O__(R) by exp(i*sign*A(t) \cdot R)
        void apply_peierls_phase(Operator<std::complex<double>>& O__, const double& time__, const int sign,
                                 const Processor& proc__=host);

        void initialize_device();
        void print_recap() const;

        /* getter methods */
        const PropagatorParameters& parameters() const { return parameters_; }
        const kGradient& kgradient() const { return *kgradient_; }
        double current_time() const { return lattice_ ? desolver_ehrenfest_.current_time() : desolver_.current_time(); }
        double time_step() const { return parameters_.desolver.dt; }
        /// Lattice coupled to the electrons, nullptr without phonons
        const phonon::Lattice* lattice() const { return lattice_; }
        Processor processor() const { return processor_; }
};

/// Functions to get the device code working
#ifdef EDUS_GPU
void ipa_hamiltonian_gpu( std::complex<double>* H__,
                                  const std::complex<double>* H0__,
                                  const std::complex<double>* x__,
                                  const std::complex<double>* y__,
                                  const std::complex<double>* z__,
                                  double* las0__,
                                  double* las1__,
                                  double* las2__,
                                  int N__
                                );


void apply_peierls_phase_gpu( std::complex<double>* O__,
                              std::complex<double>* Peierls_phase,
                              double* A0__,
                              double* A1__,
                              double* A2__,
                              double* Rvectors__,
                              int sign,
                              int N,
                              int nR
                            );
#endif

void ipa_hamiltonian_cpu( BlockMatrix<std::complex<double>>& H,
                                  const BlockMatrix<std::complex<double>>& H0,
                                  const BlockMatrix<std::complex<double>>& x,
                                  const BlockMatrix<std::complex<double>>& y,
                                  const BlockMatrix<std::complex<double>>& z,
                                  const Vector<double>& las);

void apply_peierls_phase_cpu( BlockMatrix<std::complex<double>>& OR__,
                              mdarray<std::complex<double>,1>& Peierls_phase,
                              const Coordinate& At,
                              const MeshGrid& Rgrid_gamma,
                              int sign);

#endif // PROPAGATOR_HPP
