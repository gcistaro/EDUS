#ifndef MAGNUS_HPP
#define MAGNUS_HPP

#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>
#include "Operator/Operator.hpp"
#include "DESolver/TimeStepper.hpp"

/// @brief Equation of motion in commutator form, @f$ \dot\rho = -i[H(t,\rho), \rho] @f$, block diagonal in k.
/// Besides the derivative (for explicit time steppers) it gives the Hamiltonian, used by the exponential
/// (Magnus) time steppers, which propagate @f$ \rho \to U\rho U^\dagger @f$ with U unitary.
class CommutatorEquation : public EquationOfMotion<Operator<std::complex<double>>>
{
    public:
        /// Computes in H__ (k component) the Hamiltonian at time t__ for the density matrix rho__
        virtual void hamiltonian(Operator<std::complex<double>>& H__, const double& t__,
                                 const Operator<std::complex<double>>& rho__) = 0;
        /// True if the equation of motion is really @f$ \dot\rho = -i[H,\rho] @f$ (no other terms)
        virtual bool commutator_form() const = 0;
        /// True if H depends on rho (mean field): the Hamiltonian at the Gauss points needs rho there
        virtual bool state_dependent() const = 0;
};

namespace magnus {

/// Coefficients of the commutator-free Magnus method of order 4 (Blanes and Moan, Appl. Numer. Math. 56, 1519 (2006))
const double c1 = 0.5 - std::sqrt(3.) / 6.;     // Gauss points in [0, 1]
const double c2 = 0.5 + std::sqrt(3.) / 6.;
const double a1 = (3. - 2. * std::sqrt(3.)) / 12.;
const double a2 = (3. + 2. * std::sqrt(3.)) / 12.;

/// U = exp(-i dt A) for each block, A hermitian (its hermitian part is used)
void exponential(BlockMatrix<std::complex<double>>& U__, const BlockMatrix<std::complex<double>>& A__, const double& dt__);

/// One step of the commutator-free Magnus method of order 4:
/// @f$ \rho \to U\rho U^\dagger @f$, @f$ U = e^{-i\Delta t(a_1H_1+a_2H_2)}\,e^{-i\Delta t(a_2H_1+a_1H_2)} @f$,
/// with H1, H2 the Hamiltonians at the Gauss points t + c1 dt, t + c2 dt.
void step(BlockMatrix<std::complex<double>>& rho__, const BlockMatrix<std::complex<double>>& H1__,
          const BlockMatrix<std::complex<double>>& H2__, const double& dt__);

/// out = -i[H, rho]
void derivative(BlockMatrix<std::complex<double>>& out__, const BlockMatrix<std::complex<double>>& H__,
                const BlockMatrix<std::complex<double>>& rho__);

}

/// @brief Commutator-free Magnus method of order 4 (CFM4) for @f$ \dot\rho = -i[H(t,\rho),\rho] @f$.
/// The propagator is unitary: stable for any time step (the limit is the accuracy, not the width of the bands),
/// and it conserves hermiticity, trace and purity of the density matrix exactly.
/// When H depends on rho (mean field), rho at the Gauss points is extrapolated with a cubic polynomial from the
/// last four steps (two Hamiltonians per step, order 4); the first three steps use a predictor-corrector with
/// Hermite interpolation. Host only.
class CommutatorFreeMagnus4 : public TimeStepper<Operator<std::complex<double>>>
{
    private:
        using Op = Operator<std::complex<double>>;
        using Block = BlockMatrix<std::complex<double>>;
        /// Hamiltonians at the two Gauss points
        std::array<Op, 2> H_;
        /// Density matrix at a Gauss point, argument of the Hamiltonian
        Op rho_stage_;
        /// k components of rho at the last four steps, for the extrapolation (ring buffer, newest_ is the last)
        std::vector<Block> history_;
        int newest_ = -1;
        /// Extrapolated k component of rho at a Gauss point
        Block stage_;
        long nstep_ = 0;

        CommutatorEquation& commutator_equation(EquationOfMotion<Op>& eom__) const;
        /// Hamiltonian at t__ for the density matrix given by its k component rho_k__, in H__
        void hamiltonian_at(CommutatorEquation& eq__, Op& H__, const double& t__, const Block& rho_k__);
        /// One step with a predictor and two correctors (Hermite interpolation of rho at the Gauss points)
        void step_predictor_corrector(CommutatorEquation& eq__, Op& rho__, const double& t__, const double& dt__);

    public:
        void initialize(const Op& y__) override;
        void step(EquationOfMotion<Op>& eom__, Op& y__, const double& t__, const double& dt__) override;
        void initialize_device() override;
        double stability_limit() const override { return 1.e300; }
        std::string name() const override { return "CFM4"; }
};

#endif
