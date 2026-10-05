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

/// Reads the first dynamical matrix of the file (the one at q = 0 in the file of Gamma).
/// ph.x writes the force constants (not divided by the masses) in Ry/bohr^2 and the masses in Ry units (amu_ry):
/// both are converted to Hartree atomic units.
DynamicalMatrix read_dynamical_matrix(const std::string& filename__);

/// Imposes the acoustic sum rule at Gamma, sum_{atom'} C_{atom alpha, atom' beta} = 0, correcting the diagonal
/// blocks (the "simple" sum rule of matdyn). Returns the largest correction (Hartree/bohr^2).
double impose_acoustic_sum_rule(DynamicalMatrix& dyn__);

}

#endif
