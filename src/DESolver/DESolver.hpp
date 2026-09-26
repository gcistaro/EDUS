#ifndef DESOLVER_HPP
#define DESOLVER_HPP

#include <memory>
#include <sstream>
#include <stdexcept>
#include "DESolver/DESolverParameters.hpp"
#include "DESolver/EquationOfMotion.hpp"
#include "DESolver/TimeStepper.hpp"
#include "DESolver/RungeKutta4.hpp"
#include "DESolver/AdamsBashforth.hpp"
#include "DESolver/Magnus.hpp"
#include <type_traits>

/// @brief Creates the time stepper required by the parameters
template<typename T>
std::unique_ptr<TimeStepper<T>> make_time_stepper(const DESolverParameters& parameters__)
{
    if( parameters__.solver == RK ) {
        if( parameters__.order != 4 ) {
            std::stringstream ss;
            ss << "Runge-Kutta of order " << parameters__.order << " is not available (use 4)\n";
            throw std::runtime_error(ss.str());
        }
        return std::make_unique<RungeKutta4<T>>();
    }
    if( parameters__.solver == MAGNUS ) {
        if constexpr (std::is_same_v<T, Operator<std::complex<double>>>) {
            if( parameters__.order != 4 ) {
                throw std::runtime_error("Magnus of order " + std::to_string(parameters__.order) + " is not available (use 4)\n");
            }
            return std::make_unique<CommutatorFreeMagnus4>();
        }
        throw std::runtime_error("The Magnus time stepper is available only for the density matrix (Operator)\n");
    }
    return std::make_unique<AdamsBashforth<T>>(parameters__.order);
}

/// @brief Driver for the numerical solution of the differential equation
/// @f[ \dot y = f(t,y), \qquad y(t_0) = y_0 @f]
/// It keeps the time and advances the state with the time stepper chosen in the parameters.
/// It does not own the state nor the equation: the caller sets the initial condition in the state
/// before initialize, and both must outlive the DESolver.
/// @tparam T Type of the state
template<typename T>
class DESolver
{
    private:
        DESolverParameters parameters_;
        T* y_ = nullptr;
        EquationOfMotion<T>* eom_ = nullptr;
        std::unique_ptr<TimeStepper<T>> stepper_;
        /// Number of steps done: the current time is initial_time + istep_*dt
        long istep_ = 0;

    public:
        DESolver() = default;

        /// Prepares the propagation of y__ (which already contains the initial condition) with eom__
        void initialize(T& y__, EquationOfMotion<T>& eom__, const DESolverParameters& parameters__)
        {
            parameters_ = parameters__;
            stepper_ = make_time_stepper<T>(parameters_);
            y_ = &y__;
            eom_ = &eom__;
            istep_ = 0;
            stepper_->initialize(y__);
        }

        /// Moves the workspace of the time stepper on the device
        void initialize_device() { stepper_->initialize_device(); }

        /// Advances the state of one time step
        void propagate()
        {
            stepper_->step(*eom_, *y_, current_time(), parameters_.dt);
            ++istep_;
        }

        /// Advances the state of nstep__ time steps
        void propagate(const int& nstep__)
        {
            for( int istep = 0; istep < nstep__; ++istep ) {
                propagate();
            }
        }

        /* getter methods */
        const T& function() const { return *y_; }
        T& function() { return *y_; }
        double current_time() const { return parameters_.initial_time + double(istep_) * parameters_.dt; }
        double time_step() const { return parameters_.dt; }
        const DESolverParameters& parameters() const { return parameters_; }
        const TimeStepper<T>& stepper() const { return *stepper_; }
};

#endif
