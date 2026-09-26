#ifndef WANNIERWAVEFUNCTIONS_HPP
#define WANNIERWAVEFUNCTIONS_HPP

#include <array>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include "Json/json.hpp"
#include "Wannier/QEWavefunction.hpp"
#include "Wannier/WannierGauge.hpp"

/// Input of the QE -> Wannier rotation (see WannierWavefunctionsParameters::from_json for the keys)
struct WannierWavefunctionsParameters
{
    /// <outdir>/<prefix>.save, containing wfc<ik>.dat (and data-file-schema.xml for the atoms)
    std::string qe_save_dir;
    /// wannier90 seedname, with path: <seedname>_u.mat, <seedname>_u_dis.mat, <seedname>.eig
    std::string seedname;
    std::vector<int> exclude_bands;
    double dis_win_min = -std::numeric_limits<double>::max();
    double dis_win_max = std::numeric_limits<double>::max();
    /// 0 = no spin / noncollinear, 1 = up, 2 = down (LSDA: wfcup<ik>.dat / wfcdw<ik>.dat)
    int spin = 0;
    /// FFT grid of the unit cell for the real-space functions. 0 -> smallest grid containing all the G vectors
    std::array<int, 3> fft_grid{0, 0, 0};
    /// number of unit cells along each direction where w_n(r) is computed (as wannier_plot_supercell)
    std::array<int, 3> supercell{2, 2, 2};
    /// Wannier functions to compute in real space (1-based). Empty -> all
    std::vector<int> wannier_list;
    bool write_bloch  = true;
    bool real_space   = true;
    bool write_xsf    = true;
    /// multiply each w_n(r) by a global phase making it (as much as possible) real, as wannier90 does
    bool fix_phase    = true;
    std::string output_dir = "wannier_wfc";

    static WannierWavefunctionsParameters from_json(const nlohmann::json& in__);
};

/// @brief Rotates the Kohn-Sham wavefunctions of Quantum ESPRESSO to the Wannier gauge
/// obtained by wannier90 (u.mat, u_dis.mat):
/// @f[ \tilde\psi_{nk}(G) = \sum_m c_{mk}(G)\, [U^{dis}(k) U(k)]_{mn} @f]
/// and builds the Wannier functions on a supercell
/// @f[ w_n(r) = \frac{1}{N_k}\sum_k e^{ik\cdot r}\, \tilde u_{nk}(r), \qquad
///     \tilde u_{nk}(r) = \frac{1}{\sqrt{\Omega}}\sum_G \tilde\psi_{nk}(G) e^{iG\cdot r} @f]
/// normalized to 1 on the Born-von Karman supercell (norm-conserving pseudopotentials).
class WannierWavefunctions
{
    private:
        WannierWavefunctionsParameters p_;
        WannierGauge gauge_;
        /// QE file index (1-based) of each wannier90 k point
        std::vector<int> qe_index_;
        /// G0 such that k_QE = k_w90 + G0 (crystal coordinates)
        std::vector<std::array<int, 3>> G0_;
        std::array<std::array<double, 3>, 3> a_;  // bohr
        std::array<std::array<double, 3>, 3> b_;  // bohr^-1
        double omega_ = 0.;
        int npol_     = 1;
        int nbnd_qe_  = 0;
        std::array<int, 3> fft_grid_{0, 0, 0};
        std::vector<int> wannier_list_;  // 0-based
        /// real-space Wannier functions, [iw][ipol][supercell point]
        std::vector<std::vector<std::vector<std::complex<double>>>> w_;
        /// atoms (symbol, cartesian bohr) read from data-file-schema.xml, if present
        std::vector<std::pair<std::string, std::array<double, 3>>> atoms_;
        double max_orthonormality_error_ = 0.;

        void map_kpoints();
        void read_atoms();
        void accumulate_real_space(int ik__, const QEWavefunction& wf__);

    public:
        explicit WannierWavefunctions(const WannierWavefunctionsParameters& params__);

        const WannierGauge& gauge() const { return gauge_; }
        const std::array<int, 3>& fft_grid() const { return fft_grid_; }
        std::array<int, 3> supercell_grid() const
        {
            return {p_.supercell[0] * fft_grid_[0], p_.supercell[1] * fft_grid_[1], p_.supercell[2] * fft_grid_[2]};
        }
        /// fractional coordinate (in units of the unit cell) of the first supercell point along x
        int supercell_start(int x__) const { return -(p_.supercell[x__] / 2) * fft_grid_[x__]; }

        /// Bloch functions of the wannier90 k point ik (0-based) in the Wannier gauge.
        /// The returned object has nbnd = num_wann; xk and Miller indices refer to the wannier90 k point.
        QEWavefunction bloch_wannier_gauge(int ik__) const;

        /// Loop over k: rotation, (optional) output of the rotated Bloch functions and
        /// (optional) accumulation of the real-space Wannier functions.
        void run();

        /// real-space Wannier function iw (index in wannier_list), spin component ipol
        const std::vector<std::complex<double>>& wannier_function(int iw__, int ipol__ = 0) const { return w_[iw__][ipol__]; }
        const std::vector<int>& wannier_list() const { return wannier_list_; }

        /// norm, centre (angstrom) and spread (angstrom^2) of the real-space Wannier function iw
        void centre_and_spread(int iw__, double& norm__, std::array<double, 3>& centre__, double& spread__) const;

        void write_xsf(int iw__, const std::string& filename__) const;

        double max_orthonormality_error() const { return max_orthonormality_error_; }
};

#endif
