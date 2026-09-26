#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include "mdContainers/mdContainers.hpp"
#include "DESolver/DESolver.hpp"

/* Test problem: y' = -i omega y, y(0) = 1, exact solution exp(-i omega t).
   Its eigenvalue is on the imaginary axis, like those of the equation of motion of the density matrix. */
using State = mdarray<std::complex<double>, 1>;

static FunctionEquation<State> oscillator(const double omega__)
{
    return FunctionEquation<State>([omega__](State& dy, const double&, const State& y) {
        dy[0] = std::complex<double>(0., -omega__) * y[0];
    });
}

/// Solves up to t_final with the given solver and returns y(t_final)
static std::complex<double> solve(const SolverType solver__, const int order__, const double omega__,
                                  const double dt__, const int nsteps__)
{
    State y({1});
    y[0] = 1.;
    auto eom = oscillator(omega__);
    DESolverParameters parameters;
    parameters.solver = solver__;
    parameters.order = order__;
    parameters.dt = dt__;
    DESolver<State> desolver;
    desolver.initialize(y, eom, parameters);
    desolver.propagate(nsteps__);
    return y[0];
}

/// Observed order of convergence at t = 10, halving the time step
static double observed_order(const SolverType solver__, const int order__)
{
    const double omega = 1., t_final = 10.;
    double error[2];
    for (int i : {0, 1}) {
        int nsteps = 200 * (1 << i);
        auto y = solve(solver__, order__, omega, t_final / nsteps, nsteps);
        error[i] = std::abs(y - std::exp(std::complex<double>(0., -omega * t_final)));
    }
    return std::log2(error[0] / error[1]);
}

TEST(TimeStepperTest, OrderOfConvergence)
{
    EXPECT_NEAR(observed_order(RK, 4), 4., 0.2);
    EXPECT_NEAR(observed_order(AB, 4), 4., 0.3);
    EXPECT_NEAR(observed_order(AB, 3), 3., 0.3);
}

/// |y| after many steps with omega*dt = fraction__ * stability limit
static double amplitude(const SolverType solver__, const int order__, const double fraction__)
{
    DESolverParameters parameters;
    parameters.solver = solver__;
    parameters.order = order__;
    double limit = make_time_stepper<State>(parameters)->stability_limit();
    return std::abs(solve(solver__, order__, 1., fraction__ * limit, 20000));
}

TEST(TimeStepperTest, StabilityLimit)
{
    for (auto [solver, order] : { std::pair{RK, 4}, std::pair{AB, 4}, std::pair{AB, 3} }) {
        EXPECT_LE(amplitude(solver, order, 0.95), 1.01) << "order " << order;
        /* above the limit the solution grows (it can even overflow to inf/nan) */
        double a = amplitude(solver, order, 1.05);
        EXPECT_TRUE(!std::isfinite(a) || a > 2.) << "order " << order << ": |y| = " << a;
    }
}

TEST(TimeStepperTest, UnavailableSolvers)
{
    DESolverParameters parameters;
    parameters.solver = AB;
    parameters.order = 5;
    EXPECT_THROW(make_time_stepper<State>(parameters), std::runtime_error);
    parameters.solver = RK;
    parameters.order = 3;
    EXPECT_THROW(make_time_stepper<State>(parameters), std::runtime_error);
}

TEST(TimeStepperTest, TimeAndInitialCondition)
{
    State y({1});
    y[0] = 2.;
    auto eom = oscillator(0.);
    DESolverParameters parameters;
    parameters.initial_time = -1.5;
    parameters.dt = 0.1;
    DESolver<State> desolver;
    desolver.initialize(y, eom, parameters);
    /* the initial condition is set by the caller and not modified by initialize */
    EXPECT_EQ(y[0], std::complex<double>(2.));
    desolver.propagate(15);
    EXPECT_DOUBLE_EQ(desolver.current_time(), -1.5 + 15 * 0.1);
    /* with omega = 0 the solution is constant */
    EXPECT_NEAR(std::abs(y[0] - 2.), 0., 1.e-14);
}

/* ------------------------------------------------------------------ Magnus (CFM4) */
#include "DESolver/Magnus.hpp"

/* two-level system with a time-dependent coupling: H(t) = [[e/2, g cos(w t)], [g cos(w t), -e/2]] */
static const double level_e = 1.0, level_g = 0.4, level_w = 0.7;

static void two_level_H(BlockMatrix<std::complex<double>>& H, const double t)
{
    H[0](0, 0) = level_e / 2.; H[0](1, 1) = -level_e / 2.;
    H[0](0, 1) = H[0](1, 0) = level_g * std::cos(level_w * t);
}

/// Reference solution: RK4 on d(rho)/dt = -i[H(t), rho] with a very small step
static std::array<std::complex<double>, 4> reference_two_level(const double t_final)
{
    const int n = 200000;
    const double h = t_final / n;
    auto H = [](double t) {
        std::array<std::complex<double>, 4> m = {level_e / 2., level_g * std::cos(level_w * t),
                                                 level_g * std::cos(level_w * t), -level_e / 2.};
        return m;
    };
    auto f = [&](double t, const std::array<std::complex<double>, 4>& r) {
        auto m = H(t);
        std::array<std::complex<double>, 4> out;
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
            std::complex<double> c = 0.;
            for (int l = 0; l < 2; ++l) c += m[2*i+l] * r[2*l+j] - r[2*i+l] * m[2*l+j];
            out[2*i+j] = std::complex<double>(0., -1.) * c;
        }
        return out;
    };
    std::array<std::complex<double>, 4> r = {1., 0., 0., 0.};
    auto axpy = [](const std::array<std::complex<double>, 4>& a, double s, const std::array<std::complex<double>, 4>& b) {
        std::array<std::complex<double>, 4> c; for (int i = 0; i < 4; ++i) c[i] = a[i] + s * b[i]; return c; };
    for (int i = 0; i < n; ++i) {
        double t = i * h;
        auto k1 = f(t, r), k2 = f(t + h/2, axpy(r, h/2, k1)), k3 = f(t + h/2, axpy(r, h/2, k2)), k4 = f(t + h, axpy(r, h, k3));
        for (int m = 0; m < 4; ++m) r[m] += h / 6. * (k1[m] + 2. * k2[m] + 2. * k3[m] + k4[m]);
    }
    return r;
}

/// CFM4 on the two-level system up to t_final with nsteps steps
static BlockMatrix<std::complex<double>> magnus_two_level(const double t_final, const int nsteps)
{
    BlockMatrix<std::complex<double>> rho(k, 1, 2, 2), H1(k, 1, 2, 2), H2(k, 1, 2, 2);
    rho.fill(0.); rho[0](0, 0) = 1.;
    const double h = t_final / nsteps;
    for (int i = 0; i < nsteps; ++i) {
        two_level_H(H1, i * h + magnus::c1 * h);
        two_level_H(H2, i * h + magnus::c2 * h);
        magnus::step(rho, H1, H2, h);
    }
    return rho;
}

TEST(MagnusTest, OrderOfConvergence)
{
    const double t_final = 10.;
    auto ref = reference_two_level(t_final);
    double error[2];
    for (int i : {0, 1}) {
        auto rho = magnus_two_level(t_final, 40 * (1 << i));
        error[i] = 0.;
        for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) {
            error[i] = std::max(error[i], std::abs(rho[0](a, b) - ref[2*a+b]));
        }
    }
    EXPECT_NEAR(std::log2(error[0] / error[1]), 4., 0.3);
}

TEST(MagnusTest, ConservationAndStability)
{
    /* dt = 5: omega dt = 5 for the level splitting, far beyond the limit of RK4 (2.83) */
    auto rho = magnus_two_level(1000., 200);
    auto trace = rho[0](0, 0) + rho[0](1, 1);
    std::complex<double> purity = 0.;
    for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) purity += rho[0](a, b) * rho[0](b, a);
    EXPECT_NEAR(std::abs(trace - 1.), 0., 1.e-12);
    EXPECT_NEAR(std::abs(purity - 1.), 0., 1.e-12);
    EXPECT_NEAR(std::abs(rho[0](0, 1) - std::conj(rho[0](1, 0))), 0., 1.e-12);
}
