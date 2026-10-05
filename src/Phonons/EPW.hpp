#ifndef EPW_HPP
#define EPW_HPP

#include <array>
#include <complex>
#include <string>
#include <vector>
#include "mdContainers/mdContainers.hpp"

namespace phonon {

/// @brief Data of an EPW calculation in the Wannier representation (files written by EPW with epwwrite = .true.).
/// Only what EDUS needs: the Wigner-Seitz vectors and degeneracies, the Hamiltonian (to check that the Wannier
/// functions are the ones of the tight-binding model) and, read separately, the electron-phonon matrix elements.
/// Conventions of EPW (checked on the source of EPW 6.1):
/// - O(R) = <m 0|O|n R>, O(k) = sum_R e^{i k.R} O(R) / ndegen(R): the same convention of wannier90 and EDUS;
/// - the phonon index of the matrix elements is the cartesian displacement mu = 3*(atom) + direction;
/// - epmatwp(nbnd, nbnd, nrr_k, nmodes, nrr_g) (Fortran order) is written as raw complex doubles, not divided by the
///   degeneracies; the matrix elements are in Ry/bohr, the Hamiltonian in Ry.
/// All the quantities stored here are converted to Hartree atomic units.
struct EPWData
{
    int num_bands = 0;
    int num_modes = 0;
    int num_atoms = 0;
    /// Number of Wigner-Seitz vectors for electrons (R_e), phonons and electron-phonon (R_g)
    int nrr_k = 0, nrr_q = 0, nrr_g = 0;
    /// Number of Wannier functions and of atoms if use_ws = .true. in EPW, 1 otherwise
    int dims = 1, dims2 = 1;
    /// Wigner-Seitz vectors in crystal coordinates
    std::vector<std::array<int, 3>> irvec_k, irvec_g;
    /// Degeneracies: ndegen_k(ir, iw, iw2) and ndegen_g(iw, ir, atom) (0-based, dims = 1 without use_ws)
    mdarray<int, 3> ndegen_k, ndegen_g;
    /// Hamiltonian in the Wannier gauge of EPW, H(R_e) divided by the degeneracies, (nrr_k, nb, nb), Hartree
    mdarray<std::complex<double>, 3> H;
    /// Fermi energy written by EPW (Hartree)
    double fermi_energy = 0.;
    /// Lattice vectors of EPW (rows, bohr), from crystal.fmt
    std::array<std::array<double, 3>, 3> lattice;
    /// Positions of the atoms (cartesian, bohr), from crystal.fmt
    std::vector<std::array<double, 3>> tau;
};

/// Reads epwdata.fmt, wigner.fmt and crystal.fmt in directory__
EPWData read_epw(const std::string& directory__);

/// Electron-phonon matrix elements at the phonon wavevector q__ (crystal coordinates), still in the Wannier
/// representation for the electrons:
/// @f[ g_\mu(R_e; q) = \frac{1}{\text{ndegen}_k}\sum_{R_g}\frac{e^{i 2\pi q\cdot R_g}}{\text{ndegen}_g}\, g_\mu(R_e,R_g) @f]
/// so that the coupling in the Bloch representation of the electrons is @f$ g_\mu(k,q) = \sum_{R_e} e^{ik\cdot R_e} g_\mu(R_e;q) @f$,
/// with rows at k+q and columns at k. Returned as (num_modes, nrr_k, nb, nb), Hartree/bohr.
/// Read by the rank 0 of MPI_COMM_WORLD and broadcast.
mdarray<std::complex<double>, 4> read_epmatwp(const std::string& filename__, const EPWData& epw__,
                                              const std::array<double, 3>& q__);

}

#endif
