#include <complex>
#include "Operator/Operator.hpp"
#include "DESolver.hpp"

/// @brief Specialized allocation for Operator<std::complex<double>>. We initialize fft for the auxiliary array so we allocate all the memory needed.
template<>
void DESolver<Operator<std::complex<double>>>::allocate_aux(Operator<std::complex<double>>& aux__)
{
    aux__.initialize_fft(*function_);
    aux__.lock_space(Space::k);
}
