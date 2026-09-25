#ifndef DESOLVER_HPP
#define DESOLVER_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <sstream>
#include <stdexcept>
#include "Operator/Operator.hpp"
#include "LinearAlgebra/axpby.hpp"
#include "DESolver/DESolverParameters.hpp"

/// @brief This class is a driver for the numerical solution of the differential equation
/// @f[
/// \frac{dy}{dt} = f(t,y) \quad \text{ with } y(t_0) = y_0
/// @f]
/// Right now we can use Adams-Bashforth (4 and 5) and Runge-Kutta (4) but to add something
/// else it is sufficient to write a proper propagate method
/// @tparam T Object to propagate, can be a number or an iterable object
template<typename T>
class DESolver{
    public:
        using InitialCondition = std::function<void(T&)>;
        using SourceTerm = std::function<void(T&, const double&, const T&)>;

    private:
        DESolverParameters parameters_;
        /// Function at time t in propagation. NB: the class does not own the object, needs to be destroyed somewhere else
        T* function_ = nullptr;
        /// Time of function_, updated at each step
        double current_time_ = 0.;
        /// Defines the function at the initial time
        InitialCondition initial_condition_;
        /// Defines the derivative of the function, for the time propagation.
        /// @f$ \frac{\partial F}{\partial t} = g(t)@f$, this function gives as output @f$ g(t) @f$
        SourceTerm source_term_;
        /// @brief temporary arrays to store quantities needed in DE numerical methods
        std::array<T,5> aux_Function;
        /// Used to rotate indices in AB method, if not we need to do many copies
        std::array<int, 5> index = {0, 1, 2, 3, 4};
        /// Coefficients in AB method
        std::array<double,5> beta;
        Processor processor_ = host;

        void check_parameters() const;
        void allocate_aux();
        void initialize_beta();
        void propagate_RK();
        void propagate_AB();

    public:
        DESolver(){};

        DESolver(const DESolver& DEsolver__) = default;
        DESolver& operator=(const DESolver& DEsolver__) = default;

        DESolver(DESolver&& DEsolver__) = default;
        DESolver& operator=(DESolver&& DEsolver__) = default;

        void initialize(T& function__, const InitialCondition& initial_condition__,
                        const SourceTerm& source_term__, const DESolverParameters& parameters__);
        /// Allocates the auxiliary arrays on the device, where the propagation will run
        void initialize_device();
        /// Advances the function of one time step
        void propagate();
        /// Advances the function of nstep__ time steps
        void propagate(const int& nstep__);

        void set_aux_Function(const T& a, const T& b, const T& c, const T& d, const T& e){aux_Function[0] = a, aux_Function[1] = b, aux_Function[2] = c, aux_Function[3] = d; aux_Function[4] = e;}

        /* getter methods */
        const T& function() const { return *function_; }
        T& function() { return *function_; }
        double current_time() const { return current_time_; }
        double time_step() const { return parameters_.dt; }
        const DESolverParameters& parameters() const { return parameters_; }
};

/// @brief Initialize all the class variables and evaluates the initial condition
/// @tparam T Object to propagate, can be a number or an iterable object
/// @param function__ What will be propagated by DESolver
/// @param initial_condition__ Defines the function at the initial time
/// @param source_term__ Defines the derivative of the function, for the time propagation
/// @param parameters__ Solver, order, initial time and time step
template<typename T>
void DESolver<T>::initialize(T& function__, const InitialCondition& initial_condition__,
                             const SourceTerm& source_term__, const DESolverParameters& parameters__)
{
    parameters_ = parameters__;
    check_parameters();
    function_ = &function__;
    initial_condition_ = initial_condition__;
    source_term_ = source_term__;
    current_time_ = parameters_.initial_time;

    initial_condition_(*function_);
    allocate_aux();
    initialize_beta();
}

/// @brief Checks that the solver is implemented for the requested order
template<typename T>
void DESolver<T>::check_parameters() const
{
    bool implemented = ( parameters_.solver == RK && parameters_.order == 4 ) ||
                       ( parameters_.solver == AB && ( parameters_.order == 4 || parameters_.order == 5 ) );
    if( !implemented ) {
        std::stringstream ss;
        ss << "DESolver: order " << parameters_.order << " not implemented for "
           << ( parameters_.solver == RK ? "RK" : "AB" ) << " (available: RK 4, AB 4, AB 5)\n";
        throw std::runtime_error(ss.str());
    }
}

/// @brief Allocates the auxiliary arrays with the same shape of the function, filled with 0
template<typename T>
void DESolver<T>::allocate_aux()
{
    for( auto& aux_ : aux_Function) {
        aux_ = *function_;
        std::fill(aux_.begin(), aux_.end(), 0.);
    }
}

template<typename T>
void DESolver<T>::initialize_beta()
{
    if ( parameters_.solver != AB ) {
        return;
    }
    if( parameters_.order == 4) {
        beta = {55./24., -59./24., 37./24., -3./8., 0.};
    }
    else if( parameters_.order == 5) {
        beta = {1901./720., -2774./720.,
            2616./720., -1274./720., 251./720.};
    }
}

/// @brief Propagator from the RungeKutta method. For now only order=4 is defined.
/// The equations we solve is:
/// @f[
/// \frac{dy}{dt} = f(t,y) \quad \text{ with } y(t_0) = y_0
/// @f]
/// In RK4:
/// @f[ y(t_{n+1}) = y(t_n)+\frac{\Delta t}{6}\big(k_1+2k_2+2k_3+k_4\big) @f]
/// @f[ k_1=f(t_n,y(t_n))  @f]
/// @f[ k_2=f(t_n +\frac{\Delta t}{2},  y(t_n) +\frac{\Delta t}{2}k_1) @f]
/// @f[ k_3=f(t_n + \frac{\Delta t}{2},  y(t_n) +\frac{\Delta t}{2}k_2) @f]
/// @f[ k_4=f(t_n+\Delta t, y_n+\Delta t*k_3) @f]
/// f represents the source term of the differential equation
/// @tparam T
template<typename T>
void DESolver<T>::propagate_RK()
{
    auto& dt = parameters_.dt;
    /* k1, k2, k3 in RK method */
    auto& k = aux_Function[0];
    /* temporary values of function, second argument of source_term_ */
    auto& AuxiliaryFunction = aux_Function[1];
    /* sum up contributions from a step to next */
    auto& ReducingFunction = aux_Function[2];

    /* k1=f(tn,yn) */
    source_term_(k, current_time_, *function_);

    /* k2=f(tn+h/2,yn+h/2*k1) */
    axpby(AuxiliaryFunction, 1., *function_, dt/2., k, processor_);
    axpby(ReducingFunction, 1., *function_, dt/6., k, processor_);
    source_term_(k, current_time_+dt/2., AuxiliaryFunction);

    /* k3=f(tn+h/2, yn+h/2*k2) */
    axpby(AuxiliaryFunction, 1., *function_, dt/2., k, processor_);
    axpby(ReducingFunction, 1., ReducingFunction, dt/3., k, processor_);
    source_term_(k, current_time_+dt/2., AuxiliaryFunction);

    /* k4=f(tn+h,yn+h*k3) */
    axpby(AuxiliaryFunction, 1., *function_, dt, k, processor_);
    axpby(ReducingFunction, 1., ReducingFunction, dt/3., k, processor_);
    source_term_(k, current_time_+dt, AuxiliaryFunction);

    /* Compute final function */
    axpby(*function_, 1., ReducingFunction, dt/6., k, processor_);
    current_time_ += dt;
}

/// @brief  Propagator from the Adams-Bashforth method. For now only order=4,5 is defined.
/// The equations we solve is:
/// @f[
/// \frac{dy}{dt} = f(t,y) \quad \text{ with } y(t_0) = y_0
/// @f]
/// In AB:
/// @f[
/// y(t_n) = y(t_{n-1}) + \Delta_t \sum_{i=1}^{\text{order}-1} \beta_i f(t_{n-i}, y(t_{n-i}))
/// @f]
/// We store @f$ f(t_{n-i}, y(t_{n-i}))@ f$ in aux_Function. The index of n=i changes to avoid copies
/// of aux_function.
/// @tparam T
template<typename T>
void DESolver<T>::propagate_AB()
{
    auto& order = parameters_.order;
    /* get f(t(n-1), y(n-1)) */
    source_term_(aux_Function[index[0]], current_time_, *function_);

    for( int i = 0; i < order; ++i ) {
        // y_n = y_{n-i} + h*b_i*f(t_{n-i}, y_{n-i})
        axpby(*function_, 1., *function_, parameters_.dt*beta[i], aux_Function[index[i]], processor_);
    }

    //slice indices one step to the right
    int temp_index = index[order-1];
    for( int i = order-1; i > 0; --i ) {
        index[i] = index[i-1];
    }
    index[0] = temp_index;

    current_time_ += parameters_.dt;
}

/// @brief Driver for the propagation. Calls the correct propagator depending on the solver.
/// @tparam T
template<typename T>
void DESolver<T>::propagate()
{
    if (parameters_.solver == RK){
        propagate_RK();
    }
    else if (parameters_.solver == AB){
        propagate_AB();
    }
}

/// @brief Driver for the propagation of more steps
/// @tparam T
/// @param nstep__ Number of steps we want to propagate
template<typename T>
void DESolver<T>::propagate(const int& nstep__)
{
    for( int istep = 0; istep < nstep__; ++istep ) {
        propagate();
    }
}

template<typename T>
void DESolver<T>::initialize_device()
{
    processor_ = device;
    for( auto& ix : {0,1,2,3,4} ) {
        aux_Function[ix].initialize_device();
        aux_Function[ix].set_processor(device);
        aux_Function[ix].fill(0.);
    }
}

template<>
void DESolver<Operator<std::complex<double>>>::allocate_aux();

#endif
