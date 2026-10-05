#ifndef EHRENFEST_STATE_HPP
#define EHRENFEST_STATE_HPP

#include <complex>
#include "Operator/Operator.hpp"
#include "DESolver/TimeStepper.hpp"
#include "Phonons/Lattice.hpp"

/// @brief Variables propagated in the Ehrenfest dynamics: the density matrix of the electrons and the coordinates
/// of the lattice. They are advanced together by the time stepper, so that at each intermediate stage the
/// Hamiltonian of the electrons sees the displacement of the lattice at the same time.
/// With the dynamic adiabatic reference it contains also rho_BO, the density matrix propagated with the same
/// displacements and without the laser (see phonon::AdiabaticReference).
/// The lattice is always on the host (few numbers).
struct EhrenfestState
{
    Operator<std::complex<double>> rho;
    phonon::Coordinates lattice;
    /// Adiabatic density matrix (allocated only if with_reference)
    Operator<std::complex<double>> rho_bo;
    bool with_reference = false;

    /* what the time steppers need */
    void initialize_device()
    {
        rho.initialize_device();
        if( with_reference ) {
            rho_bo.initialize_device();
        }
    }
    void set_processor(const Processor& proc__)
    {
        rho.set_processor(proc__);
        if( with_reference ) {
            rho_bo.set_processor(proc__);
        }
    }
    void fill(const std::complex<double>& value__)
    {
        rho.fill(value__);
        lattice.fill(value__);
        if( with_reference ) {
            rho_bo.fill(value__);
        }
    }
};

/* The time steppers (RK4, AB) are templates on the type of the state and only need, besides copy and fill:
   - axpby(out, a, x, b, y): out = a x + b y, used to build the intermediate stages (e.g. y + dt/2 k1);
   - make_workspace(w, y): allocates a workspace with the shape of y (the k values k1..k4 of RK4, the history of AB).
   For EhrenfestState each operation is applied to every component. The time stepper does not know that the state
   contains a density matrix and a lattice: this is what keeps them synchronized at every stage. */

/// out = a x + b y for each component (density matrix, lattice, rho_BO if present)
template <typename Scalar_T>
void axpby(EhrenfestState& Output__,
           const Scalar_T& FirstScalar__, const EhrenfestState& FirstAddend__,
           const Scalar_T& SecondScalar__, const EhrenfestState& SecondAddend__,
           const Processor& proc__=host)
{
    axpby(Output__.rho, FirstScalar__, FirstAddend__.rho, SecondScalar__, SecondAddend__.rho, proc__);
    axpby_cpu(Output__.lattice, FirstScalar__, FirstAddend__.lattice, SecondScalar__, SecondAddend__.lattice);
    if( Output__.with_reference ) {
        axpby(Output__.rho_bo, FirstScalar__, FirstAddend__.rho_bo, SecondScalar__, SecondAddend__.rho_bo, proc__);
    }
}

/// The density matrix gets the workspace of the Operators (fft initialized), the lattice a copy set to 0
template<>
inline void make_workspace(EhrenfestState& w__, const EhrenfestState& y__)
{
    make_workspace(w__.rho, y__.rho);
    w__.lattice = y__.lattice;
    w__.lattice.fill(0.);
    w__.with_reference = y__.with_reference;
    if( y__.with_reference ) {
        make_workspace(w__.rho_bo, y__.rho_bo);
    }
}

#endif
