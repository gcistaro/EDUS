#ifndef PROPAGATOR_PARAMETERS_HPP
#define PROPAGATOR_PARAMETERS_HPP

#include <stdexcept>
#include "Constants.hpp"
#include "DESolver/DESolverParameters.hpp"
#include "InputVariables/config.hpp"

struct PropagatorParameters
{
    /// Solver of the differential equation: algorithm, order, initial time and time step
    DESolverParameters desolver;
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
        p.desolver.solver = solver.at(cfg.solver());
        p.desolver.order = cfg.order();
        p.desolver.initial_time = cfg.initialtime();
        p.desolver.dt = cfg.dt();
        p.peierls = cfg.peierls();
        p.decay = cfg.decay();
        p.gradient_space = (cfg.gradient_space() == "R" ? Space::R : Space::k);
        /* with the Peierls phase the propagation corresponds to the gradient in R, while the velocity
           would use the finite differences in k: the current would not be the one of the dynamics */
        if( p.peierls && p.gradient_space == Space::k ) {
            throw std::runtime_error("gradient_space = \"k\" is not consistent with peierls = true (the default): "
                                     "set \"peierls\": false to use the gradient in k, or \"gradient_space\": \"R\"\n");
        }
        return p;
    }
};

#endif
