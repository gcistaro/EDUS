#ifndef STATE_HPP
#define STATE_HPP

#include "Operator/Operator.hpp"
#include "Phonons/EhrenfestState.hpp"

namespace electron
{
    class State
    {
private: 
        /// Full Hamiltonian: @f$ H_ = H0 + H_{\text{eff}} +E(t)\cdot \Xi @f$, while the term
        /// with the gradient is treated separately
        Operator<std::complex<double>> H_;
        /// Propagated variables: density matrix of the system and, with phonons, coordinates of the lattice
        EhrenfestState variables_;
        // For printing stuff without touching the DensityMatrix we propagate
        Operator<std::complex<double>> aux_DM_;
        /// Phase acquired in the peierls transformation, to be calculated at each time step.
        mdarray<std::complex<double>,1> Peierls_phase_;

public: 
        State() = default;
        void initialize(const GridStructure& gridstructure__, 
                                  const int& num_bands__, 
                                  const MPIindex<3>& mpindex__,
                                  const MPIindex<3>& mpindex_q__,
                                  const bool& hamiltonian_with_q__ = false);
        /// Allocates and copies to device the variables used in time propagation
        void initialize_device();


        Operator<std::complex<double>>& H() {return H_;};
        Operator<std::complex<double>>& DensityMatrix() {return variables_.rho;};
        /// Coordinates of the lattice (not allocated without phonons)
        phonon::Coordinates& lattice() {return variables_.lattice;};
        /// Density matrix and lattice, propagated together in the Ehrenfest dynamics
        EhrenfestState& variables() {return variables_;};
        Operator<std::complex<double>>& aux_DM() {return aux_DM_;};
        mdarray<std::complex<double>,1>& Peierls_phase() {return Peierls_phase_;};
    };


    inline void State::initialize(const GridStructure& gridstructure__, 
                                  const int& num_bands__, 
                                  const MPIindex<3>& mpindex__,
                                  const MPIindex<3>& mpindex_q__,
                                  const bool& hamiltonian_with_q__)
    {
        auto num_bands = num_bands__;
        auto num_kpoints = gridstructure__.kgrid()->get_TotalSize();

        /* H_ has the q components only when the level of theory needs them (mean field with nq > 1: the self
           energy has the q components of rho); otherwise H(k,q) = delta_{q0} H(k) and H_ has nq = 1 */
        if( hamiltonian_with_q__ ) {
            H_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), num_bands, mpindex_q__, "H", gridstructure__.qgrid());
        }
        else {
            H_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), num_bands, mpindex__, "H");
        }
        variables_.rho.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), num_bands, mpindex_q__, "DensityMatrix", gridstructure__.qgrid());
        aux_DM_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), num_bands, mpindex_q__, "aux_DM", gridstructure__.qgrid());
        Peierls_phase_.initialize({mpindex__.get_nlocal()});
    }

    inline void State::initialize_device()
    {
        H_.initialize_device();
        variables_.rho.initialize_device();
        aux_DM_.initialize_device();
        Peierls_phase_.initialize_device();

        H_.transfer_to(Processor::device);
        variables_.rho.transfer_to(Processor::device);
        aux_DM_.transfer_to(Processor::device);
        Peierls_phase_.transfer_to(Processor::device);
    }
}

#endif // STATE_HPP
