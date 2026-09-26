#ifndef BANDSTRUCTURE_HPP
#define BANDSTRUCTURE_HPP

#include "Operator/BlockMatrix.hpp"
#include "core/profiler.hpp"

namespace electron
{
    /// @brief Eigenvalues and eigenvectors of H0 on the (local) k points:
    /// @f[
    /// H_0(\textbf{k}) U(\textbf{k}) = U(\textbf{k}) \varepsilon(\textbf{k})
    /// @f]
    /// U rotates from the bloch to the wannier gauge: @f$ O_{wannier} = U O_{bloch} U^\dagger @f$.
    /// Members are only set by initialize, so that U and U^\dagger are always consistent.
    class BandStructure
    {
        private:
            /// Eigenvalues, sorted in ascending order and grouped like: energies_[ik](ibnd)
            std::vector<mdarray<double,1>> energies_;
            /// Eigenvectors, one per column: U_[ik](iwann, ibnd)
            BlockMatrix<std::complex<double>> U_;
            /// Conjugate transpose of U_
            BlockMatrix<std::complex<double>> Udagger_;

        public:
            BandStructure() = default;
            /// Diagonalizes H0 in k space (only the k points local to this rank)
            void initialize(const BlockMatrix<std::complex<double>>& H0k__);
            /// Rigid shift of each band, the same for all k (eigenvectors are unchanged)
            void shift_energies(const std::vector<double>& shift__);

            /* getter methods */
            const std::vector<mdarray<double,1>>& energies() const { return energies_; }
            const BlockMatrix<std::complex<double>>& U() const { return U_; }
            const BlockMatrix<std::complex<double>>& Udagger() const { return Udagger_; }
            int num_bands() const { return U_.get_nrows(); }
            int num_kpoints_local() const { return U_.get_nblocks(); }
    };

    inline void BandStructure::initialize(const BlockMatrix<std::complex<double>>& H0k__)
    {
        PROFILE("BandStructure::initialize");
        H0k__.diagonalize(energies_, U_);

        Udagger_.initialize(k, U_.get_nblocks(), U_.get_nrows(), U_.get_ncols());
        #pragma omp parallel for
        for (int ik = 0; ik < Udagger_.get_nblocks(); ++ik) {
            for (int ir = 0; ir < Udagger_.get_nrows(); ++ir) {
                for (int ic = 0; ic < Udagger_.get_ncols(); ++ic) {
                    Udagger_[ik](ir, ic) = std::conj(U_[ik](ic, ir));
                }
            }
        }
    }
    inline void BandStructure::shift_energies(const std::vector<double>& shift__)
    {
        assert( int(shift__.size()) == num_bands() );
        for (auto& energies_k : energies_) {
            for (int ib = 0; ib < num_bands(); ++ib) {
                energies_k(ib) += shift__[ib];
            }
        }
    }
} // namespace electron

#endif // BANDSTRUCTURE_HPP
