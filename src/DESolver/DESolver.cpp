#include <complex>
#include "Operator/Operator.hpp"
#include "DESolver/TimeStepper.hpp"

/// @brief Workspace for Operator<std::complex<double>>: we initialize the fft so that all the memory needed is allocated.
template<>
void make_workspace(Operator<std::complex<double>>& w__, const Operator<std::complex<double>>& y__)
{
    w__.initialize_fft(y__);
    w__.lock_space(Space::k);
}
