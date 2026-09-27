#ifndef QEWAVEFUNCTION_HPP
#define QEWAVEFUNCTION_HPP

#include <array>
#include <complex>
#include <string>
#include <vector>

#include "mdContainers/mdContainers.hpp"

/// @brief Kohn-Sham wavefunctions at one k point, as written by Quantum ESPRESSO
/// (pw.x, v6.4 or later) in <outdir>/<prefix>.save/wfc<ik>.dat.
///
/// File layout (Fortran unformatted, see Modules/io_base.f90::write_wfc):
///   ik, xk(3), ispin, gamma_only, scalef
///   ngw, igwx, npol, nbnd
///   b1(3), b2(3), b3(3)
///   mill(3, igwx)
///   evc(npol*igwx)            (one record per band, nbnd records)
///
/// xk and b_i are cartesian, in bohr^-1. The coefficients are normalized such that
/// sum_G |c_n(G)|^2 = 1 (norm-conserving PP), i.e.
/// @f$ \psi_{nk}(r) = \frac{1}{\sqrt{\Omega}} \sum_G c_{n}(G) e^{i(k+G)\cdot r} @f$.
/// For npol = 2 (noncollinear) each band stores first the spin-up then the spin-down
/// components, each of length igwx.
struct QEWavefunction
{
    int ik              = 0;
    int ispin           = 1;
    bool gamma_only     = false;
    double scalef       = 1.;
    int ngw             = 0;
    int igwx            = 0;
    int npol            = 1;
    int nbnd            = 0;
    /// k point, cartesian coordinates, bohr^-1
    std::array<double, 3> xk{0., 0., 0.};
    /// reciprocal lattice vectors b[i] (cartesian, bohr^-1)
    std::array<std::array<double, 3>, 3> b{};
    /// Miller indices, dimension (igwx, 3)
    mdarray<int, 2> miller;
    /// plane-wave coefficients, dimension (nbnd, npol*igwx)
    mdarray<std::complex<double>, 2> evc;

    /// What to read from a wfc file
    enum class Content { header, header_and_miller, all };

    /// Read a wfc file (by default everything; the header only is cheap and useful to scan the k points)
    static QEWavefunction read(const std::string& filename__, Content content__ = Content::all);

    /// Write in the same format (so that the output can be read by any QE-aware tool).
    void write(const std::string& filename__) const;

    /// For gamma_only files QE stores only half of the G vectors (c(-G) = c(G)^*).
    /// This adds the missing -G components and sets gamma_only = false.
    void expand_gamma();

    /// k point in crystal (fractional) coordinates of the reciprocal lattice.
    std::array<double, 3> xk_crystal() const;

    /// Direct lattice vectors a[i] (bohr), a_i . b_j = 2 pi delta_ij
    std::array<std::array<double, 3>, 3> direct_lattice() const;

    /// Largest |miller index| along each direction
    std::array<int, 3> max_miller() const;
};

/// Path of the (binary, ".dat") wfc file of the k point ik (1-based, as in QE).
/// spin = 0 for nspin=1/noncollinear, 1 (up) or 2 (down) for LSDA (wfcup<ik>.dat/wfcdw<ik>.dat)
/// Used to name the rotated Bloch functions written by WannierWavefunctions (always plain binary).
std::string qe_wfc_filename(const std::string& savedir__, int ik__, int spin__ = 0);

/// Path of the wfc file of the k point ik actually present on disk: QE writes either wfc<ik>.dat (plain binary)
/// or wfc<ik>.hdf5 (compiled with -D__HDF5), never both. Returns an empty string if neither exists.
std::string resolve_qe_wfc_file(const std::string& savedir__, int ik__, int spin__ = 0);

#endif
