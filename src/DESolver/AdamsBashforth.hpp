#ifndef ADAMS_BASHFORTH_HPP
#define ADAMS_BASHFORTH_HPP

#include <array>
#include <memory>
#include <sstream>
#include <stdexcept>
#include "DESolver/RungeKutta4.hpp"

/// @brief Adams-Bashforth method of order 3 or 4:
/// @f[ y(t_{n+1}) = y(t_n) + \Delta t \sum_{i=0}^{\text{order}-1} \beta_i f(t_{n-i}, y(t_{n-i})) @f]
/// One evaluation of f per step and `order` workspace arrays with the previous values of f.
/// The first order-1 steps, where the previous values are not available, are done with RK4,
/// storing the values of f needed afterwards; the workspace of RK4 is then released.
/// Order 5 is not available: its stability region contains no segment of the imaginary axis (like orders
/// 1, 2 and 6), where the eigenvalues of the equation of motion of the density matrix are. The solution grows
/// at every step: slowly (~(omega dt)^6) for small omega dt, fast when a parasitic root leaves the unit circle
/// (omega dt > 0.23, 2.6% per step at omega dt = 0.237).
template<typename T>
class AdamsBashforth : public TimeStepper<T>
{
    private:
        int order_;
        std::array<double, 4> beta_;
        /// Position in work_ of f(t_{n-i}): rotated at each step instead of copying the arrays
        std::array<int, 4> index_ = {0, 1, 2, 3};
        /// Steps done, to know when the history is complete
        long nstep_ = 0;
        /// Runge-Kutta for the first steps
        std::unique_ptr<RungeKutta4<T>> start_;

    public:
        explicit AdamsBashforth(const int order__) : order_(order__)
        {
            if( order_ == 3 ) {
                beta_ = {23./12., -16./12., 5./12., 0.};
            }
            else if( order_ == 4 ) {
                beta_ = {55./24., -59./24., 37./24., -9./24.};
            }
            else {
                std::stringstream ss;
                ss << "Adams-Bashforth of order " << order_ << " is not available (use 3 or 4)";
                if( order_ == 5 ) {
                    ss << ": order 5 is unstable for any time step on the equation of motion of the density matrix"
                       << " (no segment of the imaginary axis in its stability region)";
                }
                throw std::runtime_error(ss.str() + "\n");
            }
        }

        void initialize(const T& y__) override
        {
            this->allocate(y__, order_);
            start_ = std::make_unique<RungeKutta4<T>>();
            start_->initialize(y__);
            nstep_ = 0;
        }

        void step(EquationOfMotion<T>& eom__, T& y__, const double& t__, const double& dt__) override
        {
            auto& f = this->work_;
            /* f(t_n, y_n) */
            eom__.derivative(f[index_[0]], t__, y__);

            if( nstep_ < order_ - 1 ) {
                start_->step_from(eom__, y__, t__, dt__, f[index_[0]]);
                /* after the last step with RK4 its workspace is not needed anymore */
                if( nstep_ == order_ - 2 ) {
                    start_.reset();
                }
            }
            else {
                for( int i = 0; i < order_; ++i ) {
                    axpby(y__, 1., y__, dt__ * beta_[i], f[index_[i]], this->processor_);
                }
            }

            /* f(t_n) becomes f(t_{n-1}) at the next step */
            int last = index_[order_ - 1];
            for( int i = order_ - 1; i > 0; --i ) {
                index_[i] = index_[i - 1];
            }
            index_[0] = last;
            ++nstep_;
        }

        void initialize_device() override
        {
            TimeStepper<T>::initialize_device();
            if( start_ ) {
                start_->initialize_device();
            }
        }

        double stability_limit() const override { return order_ == 3 ? 0.724 : 0.430; }
        std::string name() const override { return "AB" + std::to_string(order_); }
};

#endif
