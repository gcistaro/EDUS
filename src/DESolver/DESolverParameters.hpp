#ifndef DESOLVER_PARAMETERS_HPP
#define DESOLVER_PARAMETERS_HPP

#include "Constants.hpp"

struct DESolverParameters
{
    /// Algorithm used to solve the differential equation (Runge-Kutta or Adams-Bashforth)
    SolverType solver = SolverType::RK;
    /// Order of the solver: 4 for RK, 4 or 5 for AB
    int order = 4;
    /// Time where the propagation starts (a.u.)
    double initial_time = 0.;
    /// Time step (a.u.)
    double dt = 0.;
};

#endif
