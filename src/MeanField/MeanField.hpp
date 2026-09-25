#ifndef MEANFIELD_HPP
#define MEANFIELD_HPP


#include "mdContainers/mdContainers.hpp"
#include "MeshGrid/MeshGrid.hpp"
#include "Operator/Operator.hpp"
#include "StreamFile.hpp"
#include "ModelCoulomb/ModelCoulomb.hpp"
#include "MeanField/MeanFieldParameters.hpp"
#include "GridStructure/GridStructure.hpp"
#include "Parallel/Decomposition.hpp"

namespace electron {

    /// @brief Class that takes care of the calculation of the effective Hamiltonian during time propagation, with the coulomb interaction matrix elements
    class MeanField 
    {
        private:
            /// Parameters of the mean-field calculation (method, interaction files, ...)
            MeanFieldParameters parameters_;
            /// Wrapper for bare interaction
            ModelCoulomb barecoulomb_;
            /// Wrapper for screened interaction
            ModelCoulomb screencoulomb_;
            /// True if the current rank propagates R=0 term
            bool HasOrigin_;
            /// Index of the origin R=0 inside the current rank
            int index_origin_local_ = -1;
            /// The effective Hartree potential, defined as: @f[ H_{nm} = \sum_\textbf{R} V_{nm}(\textbf{R}) = 
            /// \sum_\textbf{R} \langle n\textbf{0}m\textbf{R}|V(r-r')| n\textbf{0}m\textbf{R} \rangle @f]
            mdarray<std::complex<double>, 2> Hartree;
        public:
            MeanField(){};
            MeanField( const MeanFieldParameters& parameters__, 
                            const GridStructure& gridstructure__, 
                            const std::vector<Coordinate>& wannier_centers__, 
                            const parallel::Decomposition& decomposition__ );
            void initialize( const MeanFieldParameters& parameters__, 
                            const GridStructure& gridstructure__, 
                            const std::vector<Coordinate>& wannier_centers__, 
                            const parallel::Decomposition& decomposition__ );
            void EffectiveHamiltonian(Operator<std::complex<double>>& H__, const Operator<std::complex<double>>& DM__,
                                      const Operator<std::complex<double>>& DM0__, const bool& EraseH__);     
            
            void initialize_device();
            void print_recap() const;

            /* getter methods */
            const MeanFieldParameters& parameters() const { return parameters_; }
            const mdarray<std::complex<double>,3>& ScreenedPotential() const { return screencoulomb_.Potential_; }
    };

    void Hartree_interaction_gpu ( std::complex<double>* SigmaH, 
                                   const std::complex<double>* Hartree, 
                                   const std::complex<double>* DM, 
                                   const std::complex<double>* DM0, 
                                   int index, 
                                   int N);
}// namespace electron



#endif // MEANFIELD_HPP
