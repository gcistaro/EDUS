#ifndef WANNIERWAVEFUNCTIONS_HPP
#define WANNIERWAVEFUNCTIONS_HPP

#include <array>
#include <complex>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "Json/json.hpp"
#include "Wannier/QEWavefunction.hpp"
#include "Wannier/WannierGauge.hpp"
#include "core/mpi/Communicator.hpp"

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
    /// number of MPI ranks that jointly build w_n(r) for one k point, via a distributed FFT
    /// (SpFFT; requires EDUS built with -DEDUS_SPFFT=ON -DEDUS_MPI=ON) instead of the default
    /// (1) where a single rank holds the whole dense FFT box of its k point alone. Must divide
    /// the total number of MPI ranks.
    int fft_ranks_per_kpoint = 1;

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

        /// group of ranks that jointly build w_n(r) for one k point (see fft_ranks_per_kpoint).
        /// Different groups process different k points in parallel, like plain MPI ranks did
        /// before (fft_group_id_/num_fft_groups_ reduce to mpi_rank()/mpi_size() when
        /// fft_ranks_per_kpoint == 1, i.e. one rank per group).
        int fft_group_id_   = 0;
        int num_fft_groups_ = 1;
        /// non-null only if fft_ranks_per_kpoint > 1: the ranks of this rank's group
        std::unique_ptr<mpi::Communicator> fft_comm_;
        /// even split of the fft_grid_ z axis among fft_comm_ ranks: fft_z_offset_[r]/fft_z_len_[r]
        /// is the range of global z indices owned by rank r of fft_comm_ in accumulate_real_space
        std::vector<int> fft_z_offset_, fft_z_len_;

        void map_kpoints();
        void read_atoms();
        void setup_fft_groups();
        bool is_fft_group_leader() const { return !fft_comm_ || fft_comm_->rank() == 0; }
        void accumulate_real_space(int ik__, const QEWavefunction& wf__);
#if defined(EDUS_SPFFT) && defined(EDUS_MPI)
        /// accumulate_real_space's distributed path: fft_comm_ ranks jointly build w_n(r) for one
        /// k point via SpFFT, each rank inserting only the z-slab of the real-space box it owns.
        void accumulate_real_space_distributed(int ik__, const QEWavefunction& wf__, const std::complex<double>* ph0,
                                               const std::complex<double>* ph1, const std::complex<double>* ph2,
                                               double norm__);
#endif

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
