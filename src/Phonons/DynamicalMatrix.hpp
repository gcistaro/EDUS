#ifndef DYNAMICAL_MATRIX_HPP
#define DYNAMICAL_MATRIX_HPP

#include <array>
#include <complex>
#include <string>
#include <vector>
#include "mdContainers/mdContainers.hpp"

namespace phonon {

/// @brief Force constants at one q point, read from a dynamical matrix file of ph.x (prefix.dynN)
struct DynamicalMatrix
{
    int num_atoms = 0;
    /// q point of the file (cartesian, 2pi/alat units)
    std::array<double, 3> q;
    /// Mass of each atom (atomic units, electron masses)
    std::vector<double> mass;
    /// Force constants C_{mu,nu}(q), mu = 3*atom + direction (Hartree/bohr^2)
    mdarray<std::complex<double>, 2> C;
};

/// @brief Interatomic force constants in real space, read from the file of q2r.x (prefix.fc)
struct ForceConstants
{
    int num_atoms = 0;
    /// Mass of each atom (electron masses)
    std::vector<double> mass;
    /// Size of the supercell of q2r: the q grid of ph.x
    std::array<int, 3> nr = {1, 1, 1};
    /// True if the file has the dielectric tensor and the Born effective charges (polar material): then C(R) is only the
    /// short-range part, and the long-range (non-analytic) part removed by q2r is not added back here
    bool polar = false;
    /// C_{mu,nu}(R), mu = 3*atom + direction, R = (m1, m2, m3) in units of the lattice vectors, 0 <= m_i < nr_i
    /// (periodic in the supercell of q2r), (m1, m2, m3, mu, nu), Ha/bohr^2. As in q2r/matdyn it couples the atom of mu in
    /// the cell R and the atom of nu in the cell 0, so that K(q) = sum_R C(R) e^{-iq.R} is the dynamical matrix of ph.x,
    /// for the displacements u_nu(R) = u_nu(q) e^{+iq.R}: (K(q) u)_mu = sum_R' C(atom mu in 0, atom nu in R') e^{iq.R'} u_nu
    mdarray<double, 5> C;
};

/// Reads the file of q2r.x (text format). Masses from Ry units, force constants from Ry/bohr^2 to Hartree units
ForceConstants read_force_constants(const std::string& filename__);

/// Acoustic sum rule in real space ("simple" of matdyn): C_{(atom alpha),(atom beta)}(R = 0) -= sum_{R, atom'}
/// C_{(atom alpha),(atom' beta)}(R), so that K(q = 0) satisfies it. Returns the largest correction (Hartree/bohr^2)
double impose_acoustic_sum_rule(ForceConstants& fc__);

/// Force constants at the wavevector q__ (crystal coordinates of the reciprocal lattice), as matdyn:
/// @f$ K_{\mu\nu}(q) = \sum_R C_{\mu\nu}(R)\,e^{-i2\pi q\cdot R} @f$. q__ must be on the q grid of q2r (q_i nr_i
/// integer), where the sum is exact without the Wigner-Seitz weights of matdyn; it throws otherwise.
/// Returned in a DynamicalMatrix (q in crystal coordinates)
DynamicalMatrix dynamical_matrix(const ForceConstants& fc__, const std::array<double, 3>& q__);

/// Reads the first dynamical matrix of the file (the one at q = 0 in the file of Gamma).
/// ph.x writes the force constants (not divided by the masses) in Ry/bohr^2 and the masses in Ry units (amu_ry):
/// both are converted to Hartree atomic units.
DynamicalMatrix read_dynamical_matrix(const std::string& filename__);

/// Imposes the acoustic sum rule at Gamma, sum_{atom'} C_{atom alpha, atom' beta} = 0, correcting the diagonal
/// blocks (the "simple" sum rule of matdyn). Returns the largest correction (Hartree/bohr^2).
double impose_acoustic_sum_rule(DynamicalMatrix& dyn__);

}

#endif
