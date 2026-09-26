#ifndef EQUATION_OF_MOTION_HPP
#define EQUATION_OF_MOTION_HPP

#include <functional>

/// @brief Differential equation @f$ \dot y = f(t, y) @f$ solved by DESolver.
/// @tparam T Type of the state y (an Operator, an mdarray, a composite state...)
template<typename T>
class EquationOfMotion
{
    public:
        virtual ~EquationOfMotion() = default;
        /// Computes dy__ = f(t__, y__)
        virtual void derivative(T& dy__, const double& t__, const T& y__) = 0;
};

/// @brief EquationOfMotion defined by a function, e.g. a lambda
template<typename T>
class FunctionEquation : public EquationOfMotion<T>
{
    public:
        using Function = std::function<void(T&, const double&, const T&)>;

        explicit FunctionEquation(const Function& f__) : f_(f__) {}
        void derivative(T& dy__, const double& t__, const T& y__) override { f_(dy__, t__, y__); }

    private:
        Function f_;
};

#endif
