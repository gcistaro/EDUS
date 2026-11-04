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
            /// Hartree potential at the points q of the q grid (nq > 1): @f[ V_{nm}(\textbf{q}) = \sum_\textbf{R}
            /// e^{i\textbf{q}\cdot\textbf{R}} V_{nm}(\textbf{R}) @f], so that @f$ v_n(q) = \sum_m V_{nm}(q)\Delta n_m(q) @f$
            /// with @f$ n_m(q) = \rho_{mm}(\textbf{R}=0;q) @f$. hartree_q_(0) = Hartree
            mdarray<std::complex<double>, 3> hartree_q_;
            /// Self energy with the q components of the density matrix (H__ and DM__ with nq > 1)
            void self_energy_q(BlockMatrix<std::complex<double>>& HR__, const BlockMatrix<std::complex<double>>& DMR__,
                               const BlockMatrix<std::complex<double>>& DM0R__);
            /// Smallest eigenvalue of Hartree for the charge-neutral changes of the occupations (Ha), for the recap
            double hartree_min_eigenvalue_ = 0.;
            /// Number of groups of Wannier centers (atoms) with hartree_centers = atoms
            int hartree_groups_ = 0;
            /// Wannier centers grouped by atom, for the Hartree term (hartree_centers = atoms)
            std::vector<Coordinate> hartree_centers(const std::vector<Coordinate>& wannier_centers__);
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
            /// Adds to H__ the mean-field self energy @f$ \Sigma[\rho-\rho_0] @f$ (Hartree + screened exchange)
            void self_energy(Operator<std::complex<double>>& H__, const Operator<std::complex<double>>& DM__,
                             const Operator<std::complex<double>>& DM0__);
            
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
