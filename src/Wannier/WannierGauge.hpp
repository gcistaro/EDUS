#ifndef WANNIERGAUGE_HPP
#define WANNIERGAUGE_HPP

#include <array>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include "mdContainers/mdContainers.hpp"

/// @brief Content of a wannier90 <seedname>_u.mat or <seedname>_u_dis.mat file
/// (written with write_u_matrices = .true.).
/// U(ik, i, j): i = row (band index, or index inside the outer window for u_dis),
///              j = column (Wannier index).
struct UMatrixFile
{
    int num_kpts  = 0;
    int num_wann  = 0;
    int num_rows  = 0;
    /// k points in crystal coordinates, (num_kpts, 3)
    mdarray<double, 2> kpt;
    /// (num_kpts, num_rows, num_wann)
    mdarray<std::complex<double>, 3> U;

    static UMatrixFile read(const std::string& filename__);
};

/// Read a wannier90 <seedname>.eig file. Returns eig(ik, ib) in eV, (num_kpts, num_bands).
mdarray<double, 2> read_w90_eig(const std::string& filename__, int& num_bands__, int& num_kpts__);

struct WannierGaugeParameters
{
    /// wannier90 seedname (files <seedname>_u.mat, <seedname>_u_dis.mat, <seedname>.eig)
    std::string seedname;
    /// bands excluded in wannier90 (exclude_bands in .win), 1-based QE band indices
    std::vector<int> exclude_bands;
    /// outer (disentanglement) window, eV. Defaults: all bands, as in wannier90.
    double dis_win_min = -std::numeric_limits<double>::max();
    double dis_win_max = std::numeric_limits<double>::max();
};

/// @brief Total gauge transformation from Bloch to Wannier gauge.
///
/// @f[ |\tilde\psi_{nk}\rangle = \sum_{m} |\psi_{mk}\rangle V_{mn}(k), \qquad
///     V(k) = U^{dis}(k)\, U(k) @f]
/// where m runs over the bands passed to wannier90 (i.e. QE bands without exclude_bands)
/// and U^{dis}(k) is expanded from the rows of the outer window to all the bands.
class WannierGauge
{
    private:
        int num_kpts_     = 0;
        int num_wann_     = 0;
        int num_bands_    = 0;
        bool disentangled_ = false;
        std::array<int, 3> mp_grid_{0, 0, 0};
        /// k points in crystal coordinates (num_kpts, 3), as in u.mat
        mdarray<double, 2> kpt_;
        /// V(ik, m, n) with m = wannier90 band (after exclude_bands), (num_kpts, num_bands, num_wann)
        mdarray<std::complex<double>, 3> V_;
        /// first band (0-based, wannier90 numbering) and number of bands in the outer window
        std::vector<int> win_first_;
        std::vector<int> ndimwin_;
        /// eigenvalues from .eig (num_kpts, num_bands), eV. Empty if the file is not available.
        mdarray<double, 2> eig_;
        bool has_eig_ = false;
        std::vector<int> exclude_bands_;
        /// k points where the last rows of u_dis inside the outer window are zero (wrong window?)
        int suspicious_window_kpoints_ = 0;

    public:
        WannierGauge() = default;
        explicit WannierGauge(const WannierGaugeParameters& params__);

        int num_kpts() const { return num_kpts_; }
        int num_wann() const { return num_wann_; }
        int num_bands() const { return num_bands_; }
        bool disentangled() const { return disentangled_; }
        bool has_eig() const { return has_eig_; }
        const std::array<int, 3>& mp_grid() const { return mp_grid_; }
        std::array<double, 3> kpt(int ik__) const { return {kpt_(ik__, 0), kpt_(ik__, 1), kpt_(ik__, 2)}; }
        const std::complex<double>& V(int ik__, int m__, int n__) const { return V_(ik__, m__, n__); }
        int ndimwin(int ik__) const { return ndimwin_[ik__]; }
        int win_first(int ik__) const { return win_first_[ik__]; }
        double eig(int ik__, int m__) const { return eig_(ik__, m__); }
        int suspicious_window_kpoints() const { return suspicious_window_kpoints_; }

        /// Map from QE band index (0-based, 0..nbnd_qe-1) to wannier90 band index, -1 if excluded
        std::vector<int> qe_to_w90_bands(int nbnd_qe__) const;

        /// V^T(k) expanded on all the QE bands: Vt(n, m_qe), (num_wann, nbnd_qe). Excluded bands -> 0.
        mdarray<std::complex<double>, 2> Vt_qe(int ik__, int nbnd_qe__) const;

        /// Hamiltonian in the Wannier gauge, H_W(k) = V^dagger diag(eps_k) V (eV), needs .eig
        mdarray<std::complex<double>, 2> Hamiltonian_k(int ik__) const;

        /// max_k || V^dagger V - 1 ||_max
        double unitarity_error() const;
};

#endif
