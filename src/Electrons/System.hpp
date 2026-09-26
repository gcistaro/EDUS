#ifndef SYSTEM_HPP
#define SYSTEM_HPP


#include "Operator/Operator.hpp"
#include "Model/Model.hpp"
#include "GridStructure/GridStructure.hpp"
#include "kGradient/kGradient.hpp"
#include "Electrons/BandStructure.hpp"
#include "Electrons/SystemParameters.hpp"
#include "core/mpi/Communicator.hpp"

namespace electron
{
    class System
    {
private:
            /// Parameters defining the equilibrium state
            SystemParameters parameters_;
            /// Number of bands in the system
            int num_bands_; 
            /// Number of k-points in the system
            int num_kpoints_;
            /// Wannier90 interface
            Material material_;
            /// KS Hamiltonian with EDUS grids
            Operator<std::complex<double>> H0_;
            /// Wannier r operator with EDUS grid
            std::array<Operator<std::complex<double>>, 3> r_;
            /// Velocity operator
            std::array<Operator<std::complex<double>>, 3> Velocity_;
            /// Volume (3D) or area (2D) of the unit cell, a.u.: the velocity operator is divided by it
            double cell_volume_ = 1.;
            /// Eigenvalues and eigenvectors of H0
            BandStructure bandstructure_;
            /// Wannier centers
            std::vector<Coordinate> wannier_centers_;
            /// Density matrix at equilibrium, in wannier gauge (valid both in k and R)
            Operator<std::complex<double>> DM0_;
            /// Internal function to allocate memory for the operators
            void allocate_memory(const GridStructure& gridstructure, const MPIindex<3>& mpindex);
            /// Internal function to load the operators from the material class
            void load_from_material(Material& material,const GridStructure& gridstructure);
            /// Internal function to compute the velocity operator
            void compute_velocity(const kGradient& kgradient, const Space& gradient_space, const std::array<int,3>& grid_size);
            /// Internal function to compute the Wannier centers
            void compute_wannier_centers(Material& material);
            /// Internal function to rigidly shift valence and conduction bands, to increase the gap
            void open_gap();
            /// Internal function to compute the equilibrium density matrix from the band occupations
            void compute_DM0(const std::vector<double>& occupations);
public: 
            System() = default;   
            void initialize(Material& material, 
                            const GridStructure& gridstructure, 
                            const MPIindex<3>& mpindex, 
                            const kGradient& kgradient, 
                            const Space& gradient_space, 
                            const SystemParameters& parameters);
            void print_recap() const;
            /// Prints H0 and r in the wannier90 _tb.dat format (only rank 0 of comm__ writes)
            void print_wannier(const std::string& filename__, const GridStructure& gridstructure__, 
                               const mpi::Communicator& comm__) const;
            /// Allocates and copies to device the operators used in time propagation
            void initialize_device();

            /* getter methods */
            const Operator<std::complex<double>>& H0() const {return H0_;};
            const std::array<Operator<std::complex<double>>, 3>& r() const {return r_;};
            const std::array<Operator<std::complex<double>>,3>& Velocity() const {return Velocity_;};
            double cell_volume() const {return cell_volume_;};
            const std::vector<Coordinate>& wannier_centers() const {return wannier_centers_;};
            const Operator<std::complex<double>>& DM0() const {return DM0_;};
            int num_bands() const { return num_bands_; }
            const SystemParameters& parameters() const { return parameters_; }
            int num_kpoints() const { return num_kpoints_; }
            const Material& material() const { return material_; }
            const BandStructure& bandstructure() const { return bandstructure_; }
    };    
} //end namespace electron


#endif