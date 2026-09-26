#ifndef DESOLVER_HPP
#define DESOLVER_HPP

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include "Operator/Operator.hpp"
#include "LinearAlgebra/axpby.hpp"

/// @brief This class is a driver for the numerical solution of the differential equation. 
/// Right now we can use Adams-Bashforth (4 and 5) and Runge-Kutta (4) but to add something
/// else it is sufficient to write a proper Propagate method 
/// @tparam T Object to propagate, can be a number or an iterable object
template<typename T>
class DESolver{
    protected:
        /// Function at time t in propagation. NB: the class does not own the object, needs to be destroyed somewhere else
		T* Function;                  
        /// Initial point of the differential equation
		double InitialTime = 0.;
        /// Resolution we use for the propagation 
        double ResolutionTime = 0.;
        /// Time updated during propagation so it really gives the current step
        double CurrentTime = 0;
        /// (not used right now) final time where we stop the propagation
        double FinalTime;
        /// Standard function containing the definition of the Function at InitialTime (mandatory)
        std::function<void(T&)> EvaluateInitialCondition;
        /// Standard function defining the derivative of Function, for the time propagation.
        /// @f$ \frac{\partial F}{\partial t} = g(t)@f$, this function gives as output @f$ g(t) @f$ 
        std::function<void(T&, const double&, const T&)> EvaluateSourceFunction;
        /// @brief temporary arrays to store quantities needed in DE numerical methods 
        std::array<T,5> aux_Function;                       
        /// Used to rotate indices in AB method, if not we need to do many copies
        std::array<int, 5> index = {0, 1, 2, 3, 4};      
        /// Coefficients in AB method
        std::array<double,5> beta; 
        /// Can be RK (Runge-Kutta) or AB (Adams-Bashforth)
        SolverType type;
        /// Order of the method used
        int order;
        Processor processor_ = host;

        void Propagate_RK();
        void Propagate_AB();
    public:
        DESolver(){};

        DESolver(const DESolver& DEsolver__) = default;
        DESolver& operator=(const DESolver& DEsolver__) = default;

        DESolver(DESolver&& DEsolver__) = default;
        DESolver& operator=(DESolver&& DEsolver__) = default; 


        DESolver(T& Function_, const std::function<void(T&)>& EvaluateInitialCondition__, const std::function<void(T&, const double&, const T&)>& EvaluateSourceFunction__, SolverType type__, int order); 

        const T& get_Function() const; 
        T& get_Function(); 
        const double& get_CurrentTime() const {return CurrentTime;};
        const double& get_ResolutionTime() const {return ResolutionTime;};
        void set_InitialTime(const double& InitialTime_){InitialTime = InitialTime_;}
        void set_processor(const Processor& proc__){processor_ = proc__; if(processor_ == device) initialize_device();}
        void set_ResolutionTime(const double& ResolutionTime_){ResolutionTime = ResolutionTime_;}
        void initialize(T& Function_, const std::function<void(T&)>& EvaluateInitialCondition_, const std::function<void(T&, const double&, const T&)>& EvaluateSourceFunction_, SolverType type__, int order);
        void Propagate();
        void Propagate(const int& nstep__);
        void set_aux_Function(const T& a, const T& b, const T& c, const T& d, const T& e){aux_Function[0] = a, aux_Function[1] = b, aux_Function[2] = c, aux_Function[3] = d; aux_Function[4] = e;}
        void set_type(SolverType t){type = t;}
        void initialize_beta();
        SolverType get_type(){return type;}	
        void initialize_device();
};

/// @brief Initialize all the class variables
/// @tparam T Object to propagate, can be a number or an iterable object
/// @param Function__ What will be propagated by DEsolver 
/// @param EvaluateInitialCondition__ Standard function containing the definition of the Function at InitialTime
/// @param EvaluateSourceFunction__  Standard function defining the derivative of Function, for the time propagation.
/// @param type__ Can be RK (Runge-Kutta) or AB (Adams-Bashforth)
/// @param order__ Order of the method used
template<typename T>
void DESolver<T>::initialize(T& Function__, const std::function<void(T&)>& EvaluateInitialCondition__, 
    const std::function<void(T&, const double&, const T&)>& EvaluateSourceFunction__, SolverType type__, int order__)
{
    type = type__;
    order = order__;
    Function = &Function__;
    EvaluateInitialCondition = EvaluateInitialCondition__;
    EvaluateSourceFunction = EvaluateSourceFunction__;
    EvaluateInitialCondition(*Function);

    for( auto& aux_ : aux_Function) {
        aux_ = *Function;
        std::fill(aux_.begin(), aux_.end(), 0.);
    }
    initialize_beta();
}

template<typename T>
void DESolver<T>::initialize_beta()
{
    if ( type == AB ) {
        if( order == 4) {
            beta = {55./24., -59./24., 37./24., -3./8., 0.};
        }
        else if( order == 5) {
            beta = {1901./720., -2774./720., 
                2616./720., -1274./720., 251./720.};
        }
        else {
            std::stringstream ss;
            ss<< "order " << order << "not implemented for AB" << std::endl;
            throw std::runtime_error(ss.str());
        }
    }
}

/// @brief Triggers initialization of all the parameters
/// @tparam T Object to propagate, can be a number or an iterable object
/// @param Function__ What will be propagated by DEsolver 
/// @param EvaluateInitialCondition__ Standard function containing the definition of the Function at InitialTime
/// @param EvaluateSourceFunction__  Standard function defining the derivative of Function, for the time propagation.
/// @param type__ Can be RK (Runge-Kutta) or AB (Adams-Bashforth)
/// @param order__ Order of the method used
template<typename T>
DESolver<T>::DESolver(T& Function__, const std::function<void(T&)>& EvaluateInitialCondition__, 
                        const std::function<void(T&, const double&, const T&)>& EvaluateSourceFunction__, SolverType type__, int order__)
{
    initialize(Function__, EvaluateInitialCondition__, EvaluateSourceFunction__, type__, order__);
}

/// @brief Propagator from the RungeKutta method. For now only order=4 is defined. 
/// The equations we solve is: 
/// @f[
/// \frac{dy}{dt} = f(t,y) \quad \text{ with } y(t_0) = y_0  
/// In RK4: 
/// @f[ y(t_{n+1}) = y(t_n)+\frac{\Delta t}{6}\big(k_1+2k_2+2k_3+k_4\big) @f]
/// @f[ k_1=f(t_n,y(t_n))  @f]
/// @f[ k_2=f(t_n +\frac{\Delta t}{2},  y(t_n) +\frac{\Delta t}{2}k_1) @f]
/// @f[ k_3=f(t_n + \frac{\Delta t}{2},  y(t_n) +\frac{\Delta t}{2}k_2) @f]
/// @f[ k_4=f(t_n+\Delta t, y_n+\Delta t*k_3) @f]
/// f represents the source term of the differential equation
/// @tparam T 
template<typename T>
void DESolver<T>::Propagate_RK()
{
    assert(order == 4);

    /* k1, k2, k3 in RK method */
    auto& k = aux_Function[0];                  
    /* temporary values of function, second argument of EvaluateSourceFunction */
    auto& AuxiliaryFunction = aux_Function[1];  
    /* sum up contributions from a step to next */
    auto& ReducingFunction = aux_Function[2];   

    /* k1=f(tn,yn) */
    EvaluateSourceFunction(k, CurrentTime, *Function);

    /* k2=f(tn+h/2,yn+h/2*k1) */
    axpby(AuxiliaryFunction, 1., *Function, ResolutionTime/2., k, processor_);     
    axpby(ReducingFunction, 1., *Function, ResolutionTime/6., k, processor_);     
    EvaluateSourceFunction(k, CurrentTime+ResolutionTime/2., AuxiliaryFunction);

    /* k3=f(tn+h/2, yn+h/2*k2) */
    axpby(AuxiliaryFunction, 1., *Function, ResolutionTime/2., k, processor_);
    axpby(ReducingFunction, 1., ReducingFunction, ResolutionTime/3., k, processor_);   
    EvaluateSourceFunction(k,CurrentTime+ResolutionTime/2., AuxiliaryFunction); 

    /* k4=f(tn+h,yn+h*k3) */
    axpby(AuxiliaryFunction, 1., *Function, ResolutionTime, k, processor_);
    axpby(ReducingFunction, 1., ReducingFunction, ResolutionTime/3., k, processor_);  
    EvaluateSourceFunction(k, CurrentTime+ResolutionTime, AuxiliaryFunction); 

    /* Compute final function */
    axpby(*Function, 1., ReducingFunction, ResolutionTime/6., k, processor_);  
    CurrentTime += ResolutionTime;
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
void DESolver<T>::Propagate_AB()
{
    /* get f(t(n-1), y(n-1)) */
    EvaluateSourceFunction(aux_Function[index[0]], CurrentTime, *Function);

    for( int i = 0; i < order; ++i ) {
        // y_n = y_{n-i} + h*b_i*f(t_{n-i}, y_{n-i})
        axpby(*Function, 1., *Function, ResolutionTime*beta[i], aux_Function[index[i]], processor_);
    }

    //slice indices one step to the right
    int temp_index = index[order-1];
    for( int i = order-1; i > 0; --i ) {
        index[i] = index[i-1];
    }
    index[0] = temp_index;

    CurrentTime += ResolutionTime;
}

/// @brief Driver for the propagation. Calls the correct propagator depending on how we set it.
/// @tparam T 
template<typename T>
void DESolver<T>::Propagate(){
    if (type == RK){
        Propagate_RK();
    }
    else if (type == AB){
        Propagate_AB();
    }
}

/// @brief Driver for the propagation of more steps
/// @tparam T 
/// @param nstep__ Number of steps we want to propagate
template<typename T>
void DESolver<T>::Propagate(const int& nstep__){
    for( int istep = 0; istep < nstep__; ++istep ) {
        Propagate();
    }
}

/// @brief Getter for the Function of the class
/// @tparam T 
/// @return Function we are propagating
template<typename T>
const T& DESolver<T>::get_Function() const 
{
    return *Function;                         
}

template<typename T>
T& DESolver<T>::get_Function() 
{
    return *Function;
}

template<typename T>
void DESolver<T>::initialize_device()
{
    for( auto& ix : {0,1,2,3,4} ) {
        aux_Function[ix].initialize_device();    
        aux_Function[ix].set_processor(device);  
        aux_Function[ix].fill(0.);
    }
}

template<>
void DESolver<Operator<std::complex<double>>>::initialize(Operator<std::complex<double>>& Function_, 
                const std::function<void(Operator<std::complex<double>>&)>& EvaluateInitialCondition_, 
                const std::function<void(Operator<std::complex<double>>&, 
                const double&, const Operator<std::complex<double>>&)>& EvaluateSourceFunction_,
                SolverType type__, int order__);


#endif