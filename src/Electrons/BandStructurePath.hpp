#ifndef BANDSTRUCTURE_PATH_HPP
#define BANDSTRUCTURE_PATH_HPP

#include <string>
#include <vector>
#include "Operator/Operator.hpp"

namespace electron
{
    /// @brief Interpolates the tight-binding Hamiltonian along a path in k space, diagonalizes it and writes
    /// BANDSTRUCTURE.txt (energies and weights on the wannier orbitals) and plotbands.gnu in directory__.
    /// @param H__ Hamiltonian in R space (e.g. the one read from the _tb file). Taken by value: the dft overwrites its k part
    /// @param kpath__ Vertices of the path, in crystal coordinates of the reciprocal lattice
    /// @param directory__ Where to write the files
    /// @param is_writer__ Only the rank with is_writer__ true writes the files
    void print_bandstructure_path(Operator<std::complex<double>> H__,
                                  const std::vector<std::vector<double>>& kpath__,
                                  const std::string& directory__,
                                  const bool is_writer__);
} // namespace electron

#endif
