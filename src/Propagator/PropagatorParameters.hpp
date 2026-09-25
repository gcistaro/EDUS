#ifndef PROPAGATOR_PARAMETERS_HPP
#define PROPAGATOR_PARAMETERS_HPP

#include "Constants.hpp"
#include "InputVariables/config.hpp"

struct PropagatorParameters
{
    /// Algorithm used to solve the differential equation (Runge-Kutta or Adams-Bashforth)
    SolverType solver = SolverType::RK;
    /// Order of the solver
    int order = 4;
    /// Initial time of the propagation (a.u.)
    double initial_time = 0.;
    /// Time step (a.u.)
    double dt = 0.;
    /// If true, the laser enters through the Peierls phase instead of the gradient term
    bool peierls = false;
    /// Decay time of the density matrix towards equilibrium (a.u.). Ignored if ~0
    double decay = 0.;
    /// Space where the gradient in k is evaluated
    Space gradient_space = R;
};

class PropagatorParametersFactory
{
public:
    /// Times are expected already converted in atomic units
    static PropagatorParameters create(const config_t& cfg)
    {
        PropagatorParameters p;
        p.solver = solver.at(cfg.solver());
        p.order = cfg.order();
        p.initial_time = cfg.initialtime();
        p.dt = cfg.dt();
        p.peierls = cfg.peierls();
        p.decay = cfg.decay();
        p.gradient_space = (cfg.gradient_space() == "R" ? Space::R : Space::k);
        return p;
    }
};

#endif
