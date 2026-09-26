#include <fstream>
#include <iomanip>
#include "Electrons/BandStructurePath.hpp"
#include "Electrons/BandStructure.hpp"
#include "ConvertUnits.hpp"

namespace electron
{
    void print_bandstructure_path(Operator<std::complex<double>> H__,
                                  const std::vector<std::vector<double>>& kpath__,
                                  const std::string& directory__,
                                  const bool is_writer__)
    {
        PROFILE("print_bandstructure_path");
        /* create the path of kpoints where to print the band structure */
        std::vector<Coordinate> vertices(kpath__.size());
        for( int ivertex = 0; ivertex < int(kpath__.size()); ++ivertex ) {
            auto& bare_k = kpath__[ivertex];
            vertices[ivertex] = Coordinate(bare_k[0], bare_k[1], bare_k[2], LatticeVectors(Space::k));
        }
        MeshGrid MeshGridPath(Space::k, vertices, 0.01);

        /* interpolate and diagonalize the hamiltonian on the path (each rank does all the points) */
        H__.dft(MeshGridPath.get_mesh(), +1, false);
        BandStructure bandstructure;
        bandstructure.initialize(H__.get_Operator_k());
        auto& energies = bandstructure.energies();
        auto& U = bandstructure.U();

        if( !is_writer__ ) {
            return;
        }

        /* print eigenvalues and weights on the wannier orbitals */
        std::ofstream Output(directory__ + "/BANDSTRUCTURE.txt");
        Output << "#k-number    energy(eV)\n";
        double min_eig = energies[0](0);
        double max_eig = energies[0](0);
        for(int ik=0; ik<int(energies.size()); ik++){
            for(int iband=0; iband<bandstructure.num_bands(); ++iband){
                min_eig = std::min(min_eig, energies[ik](iband));
                max_eig = std::max(max_eig, energies[ik](iband));
                Output << std::setw(6) << ik;
                Output << std::setw(15) << std::setprecision(6) << Convert(energies[ik](iband),AuEnergy,ElectronVolt);
                auto sum = 0.;
                for(int iwann=0; iwann<bandstructure.num_bands(); ++iwann) {
                    Output << std::setw(15) << std::setprecision(6) << std::pow(std::abs(U[ik](iwann,iband)),2);
                    sum += std::pow(std::abs(U[ik](iwann,iband)),2);
                }
                Output << std::setw(15) << std::setprecision(6)<< sum;
                Output << std::endl;
            }
        }

        /* gnuplot script, with vertical lines on the vertices of the path */
        std::ofstream Gnuplot(directory__ + "/plotbands.gnu");
        Gnuplot << "set xrange [ "<< -double(energies.size())/10. << " : " << 11./10.*double(energies.size()) << "]"<<std::endl;
        min_eig = Convert(min_eig, AuEnergy, ElectronVolt);
        max_eig = Convert(max_eig, AuEnergy, ElectronVolt);
        auto eig_range = max_eig - min_eig;
        Gnuplot << "set yrange [" << min_eig - eig_range/10. << ": " << max_eig + eig_range/10. << "]" << std::endl;
        for( int ivertex = 0; ivertex < int(vertices.size()); ++ivertex ) {
            int kpt = MeshGridPath.find(vertices[ivertex], ivertex);
            Gnuplot << "set arrow from  " << kpt << ", " << min_eig- eig_range/10. << " to " << kpt << ", " << max_eig+ eig_range/10. << " nohead" << std::endl;
        }
        Gnuplot << "plot \"BANDSTRUCTURE.txt\" w p" << std::endl;
        Gnuplot << "pause -1" << std::endl;
    }
} // namespace electron
