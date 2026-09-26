#ifndef RUNGE_KUTTA4_HPP
#define RUNGE_KUTTA4_HPP

#include <cmath>
#include "DESolver/TimeStepper.hpp"

/// @brief Runge-Kutta method of order 4:
/// @f[ y(t_{n+1}) = y(t_n)+\frac{\Delta t}{6}\big(k_1+2k_2+2k_3+k_4\big) @f]
/// @f[ k_1=f(t_n,y(t_n)), \quad k_2=f(t_n +\tfrac{\Delta t}{2}, y(t_n) +\tfrac{\Delta t}{2}k_1) @f]
/// @f[ k_3=f(t_n + \tfrac{\Delta t}{2}, y(t_n) +\tfrac{\Delta t}{2}k_2), \quad k_4=f(t_n+\Delta t, y_n+\Delta t\,k_3) @f]
/// Four evaluations of f per step, three workspace arrays.
template<typename T>
class RungeKutta4 : public TimeStepper<T>
{
    public:
        void initialize(const T& y__) override
        {
            this->allocate(y__, 3);
        }

        void step(EquationOfMotion<T>& eom__, T& y__, const double& t__, const double& dt__) override
        {
            /* k1 in the array used afterwards for k2, k3, k4 */
            auto& k1 = this->work_[0];
            eom__.derivative(k1, t__, y__);
            step_from(eom__, y__, t__, dt__, k1);
        }

        /// One step when k1__ = f(t__, y__) is already known. k1__ can be the first workspace array:
        /// it is overwritten only after its use.
        void step_from(EquationOfMotion<T>& eom__, T& y__, const double& t__, const double& dt__, const T& k1__)
        {
            auto& proc = this->processor_;
            /* k2, k3, k4 */
            auto& k = this->work_[0];
            /* argument of f in the intermediate stages */
            auto& stage = this->work_[1];
            /* sum of the contributions to y(t+dt) */
            auto& sum = this->work_[2];

            /* k2=f(tn+h/2,yn+h/2*k1) */
            axpby(stage, 1., y__, dt__/2., k1__, proc);
            axpby(sum, 1., y__, dt__/6., k1__, proc);
            eom__.derivative(k, t__ + dt__/2., stage);

            /* k3=f(tn+h/2, yn+h/2*k2) */
            axpby(stage, 1., y__, dt__/2., k, proc);
            axpby(sum, 1., sum, dt__/3., k, proc);
            eom__.derivative(k, t__ + dt__/2., stage);

            /* k4=f(tn+h,yn+h*k3) */
            axpby(stage, 1., y__, dt__, k, proc);
            axpby(sum, 1., sum, dt__/3., k, proc);
            eom__.derivative(k, t__ + dt__, stage);

            /* y(tn+h) */
            axpby(y__, 1., sum, dt__/6., k, proc);
        }

        double stability_limit() const override { return 2. * std::sqrt(2.); }
        std::string name() const override { return "RK4"; }
};

#endif
