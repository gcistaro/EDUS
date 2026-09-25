#include <fstream>
#include <limits>
#include <cmath>
#include "Electrons/Pdos.hpp"
#include "ConvertUnits.hpp"

namespace electron
{
    void print_pdos(const BandStructure& bandstructure__, const std::string& filename__,
                    const mpi::Communicator& comm__)
    {
        PROFILE("print_pdos");
        auto& energies = bandstructure__.energies();
        auto& U = bandstructure__.U();
        int nbnd = bandstructure__.num_bands();

        /* energy range over all the k points (of all the ranks) */
        double min_eig = std::numeric_limits<double>::max();
        double max_eig = std::numeric_limits<double>::lowest();
        for (int ik = 0; ik < bandstructure__.num_kpoints_local(); ++ik) {
            for (int ib = 0; ib < nbnd; ++ib) {
                min_eig = std::min(min_eig, energies[ik](ib));
                max_eig = std::max(max_eig, energies[ik](ib));
            }
        }
#ifdef EDUS_MPI
        MPI_Allreduce(MPI_IN_PLACE, &min_eig, 1, MPI_DOUBLE, MPI_MIN, comm__.communicator());
        MPI_Allreduce(MPI_IN_PLACE, &max_eig, 1, MPI_DOUBLE, MPI_MAX, comm__.communicator());
#endif
        auto E_resolution = Convert(0.1, ElectronVolt, AuEnergy);
        int num_bins = int( (max_eig - min_eig)/E_resolution ) + 1;

        /* histogram: since psi_{nk} = \sum_m U_{mn} \tilde{psi_{mk}} we have U_{mn} = <\tilde{psi_{mk}}|psi_{nk}> */
        std::vector<double> pdos(num_bins * nbnd, 0.);
        for (int ik = 0; ik < bandstructure__.num_kpoints_local(); ++ik) {
            for (int iwann = 0; iwann < nbnd; ++iwann) {
                for (int ib = 0; ib < nbnd; ++ib) {
                    auto ibin = int( ( energies[ik](ib) - min_eig )/E_resolution );
                    pdos[ibin * nbnd + iwann] += std::pow( std::abs( U[ik](iwann, ib) ), 2 );
                }
            }
        }
#ifdef EDUS_MPI
        MPI_Reduce( comm__.rank() == 0 ? MPI_IN_PLACE : pdos.data(), pdos.data(),
                    num_bins * nbnd, MPI_DOUBLE, MPI_SUM, 0, comm__.communicator() );
#endif
        if ( comm__.rank() != 0 ) {
            return;
        }
        std::ofstream os_pdos(filename__);
        os_pdos << "# energy(a.u.)   pdos on each wannier orbital\n";
        for (int ibin = 0; ibin < num_bins; ibin++) {
            os_pdos << min_eig + ibin * E_resolution << " ";
            for (int iwann = 0; iwann < nbnd; iwann++) {
                os_pdos << pdos[ibin * nbnd + iwann] << " ";
            }
            os_pdos << std::endl;
        }
    }
} // namespace electron
