#ifndef PDOS_HPP
#define PDOS_HPP

#include <string>
#include "Electrons/BandStructure.hpp"
#include "core/mpi/Communicator.hpp"

namespace electron
{
    /// @brief Density of states projected on the wannier orbitals, as a histogram in energy:
    /// @f[
    /// \text{pdos}_m(E) = \sum_{n\textbf{k}} |U_{mn}(\textbf{k})|^2 \, \theta(E \le \varepsilon_{n\textbf{k}} < E+\Delta E)
    /// @f]
    /// with @f$ \Delta E = 0.1 @f$ eV. The k points are distributed among the ranks of comm__, the result is
    /// reduced and written by its rank 0. First column: energy (a.u.), then one column per orbital.
    void print_pdos(const BandStructure& bandstructure__, const std::string& filename__,
                    const mpi::Communicator& comm__);
} // namespace electron

#endif
