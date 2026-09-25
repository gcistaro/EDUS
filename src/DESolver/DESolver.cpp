#include <complex>
#include "Operator/Operator.hpp"
#include "DESolver.hpp"

/// @brief Specialized allocation for Operator<std::complex<double>>. We initialize fft for aux_function so we allocate all the memory needed.
template<>
void DESolver<Operator<std::complex<double>>>::allocate_aux()
{
    for(auto& aux_f : aux_Function) {
        aux_f.initialize_fft(*function_);
        aux_f.lock_space(Space::k);
    }
}
