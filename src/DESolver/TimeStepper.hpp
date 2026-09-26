#ifndef TIME_STEPPER_HPP
#define TIME_STEPPER_HPP

#include <algorithm>
#include <complex>
#include <string>
#include <vector>
#include "Operator/Operator.hpp"
#include "LinearAlgebra/axpby.hpp"
#include "DESolver/EquationOfMotion.hpp"

/// @brief Allocates w__ with the same shape of y__, filled with 0
template<typename T>
void make_workspace(T& w__, const T& y__)
{
    w__ = y__;
    std::fill(w__.begin(), w__.end(), 0.);
}

/// For Operator the fft is initialized, so that all the memory needed is allocated
template<>
void make_workspace(Operator<std::complex<double>>& w__, const Operator<std::complex<double>>& y__);

/// @brief Algorithm that advances the solution of an EquationOfMotion by one time step.
/// Each algorithm owns the workspace it needs.
/// @tparam T Type of the state: it needs begin/end (for axpby), copy, fill and, on gpu, initialize_device
template<typename T>
class TimeStepper
{
    protected:
        /// Workspace arrays, with the shape of the state
        std::vector<T> work_;
        Processor processor_ = host;

        void allocate(const T& y__, const int n__)
        {
            work_.resize(n__);
            for (auto& w : work_) {
                make_workspace(w, y__);
            }
        }

    public:
        virtual ~TimeStepper() = default;

        /// Allocates the workspace for states with the shape of y__
        virtual void initialize(const T& y__) = 0;
        /// Advances y__ from t__ to t__ + dt__
        virtual void step(EquationOfMotion<T>& eom__, T& y__, const double& t__, const double& dt__) = 0;
        /// Largest |omega dt| for which the method is stable on @f$ \dot y = -i\omega y @f$
        /// (the eigenvalues of the equation of motion of the density matrix are on the imaginary axis)
        virtual double stability_limit() const = 0;
        virtual std::string name() const = 0;

        /// Moves the workspace on the device, where the propagation will run
        virtual void initialize_device()
        {
            processor_ = device;
            for (auto& w : work_) {
                w.initialize_device();
                w.set_processor(device);
                w.fill(0.);
            }
        }
};

#endif
