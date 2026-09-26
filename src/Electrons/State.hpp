#ifndef STATE_HPP
#define STATE_HPP

#include "Operator/Operator.hpp"

namespace electron
{
    class State
    {
private: 
        /// Full Hamiltonian: @f$ H_ = H0 + H_{\text{eff}} +E(t)\cdot \Xi @f$, while the term
        /// with the gradient is treated separately
        Operator<std::complex<double>> H_;
        // Density matrix of the system
        Operator<std::complex<double>> DensityMatrix_;
        // For printing stuff without touching the DensityMatrix we propagate
        Operator<std::complex<double>> aux_DM_;
        /// Phase acquired in the peierls transformation, to be calculated at each time step.
        mdarray<std::complex<double>,1> Peierls_phase_;

public: 
        State() = default;
        void initialize(const GridStructure& gridstructure__, const int& num_bands__, const MPIindex<3>& mpindex);
        /// Allocates and copies to device the variables used in time propagation
        void initialize_device();


        Operator<std::complex<double>>& H() {return H_;};
        Operator<std::complex<double>>& DensityMatrix() {return DensityMatrix_;};
        Operator<std::complex<double>>& aux_DM() {return aux_DM_;};
        mdarray<std::complex<double>,1>& Peierls_phase() {return Peierls_phase_;};
    };


    inline void State::initialize(const GridStructure& gridstructure__, const int& num_bands__, const MPIindex<3>& mpindex)
    {
        auto num_bands = num_bands__;
        auto num_kpoints = gridstructure__.kgrid()->get_TotalSize();

        H_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), num_bands, mpindex, "H");
        DensityMatrix_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), num_bands, mpindex, "DensityMatrix");
        aux_DM_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), num_bands, mpindex, "aux_DM");
        Peierls_phase_.initialize({mpindex.get_nlocal()});
    }

    inline void State::initialize_device()
    {
        H_.initialize_device();
        DensityMatrix_.initialize_device();
        aux_DM_.initialize_device();
        Peierls_phase_.initialize_device();

        H_.transfer_to(Processor::device);
        DensityMatrix_.transfer_to(Processor::device);
        aux_DM_.transfer_to(Processor::device);
        Peierls_phase_.transfer_to(Processor::device);
    }
}

#endif // STATE_HPP
