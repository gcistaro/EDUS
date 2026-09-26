#include "Operator/Operator.hpp"
#include "DESolver/DESolver.hpp"
#include "Simulation/Simulation.hpp"

class GreenFunction : public Simulation
{
    private:
//        Operator<std::complex<double>> DM_;
//        Operator<std::complex<double>> H_; 
        Operator<std::complex<double>> Ut_; 
        Operator<std::complex<double>> Utime_; 
        Operator<std::complex<double>> Uptime_; 
        Operator<std::complex<double>> GR_;

        DESolver<Operator<std::complex<double>>> DEsolver_Ut_;
        /// Equation of motion of Ut, used by DEsolver_Ut_
        std::unique_ptr<FunctionEquation<Operator<std::complex<double>>>> equation_Ut_;
//        DESolver<Operator<std::complex<double>>>* DEsolver_DM_;

        int PrintResolution_;

    public:
        using Simulation::Simulation;
        void initialize();

        BlockMatrix<std::complex<double>>& GKBA(const double time__, const double ptime__);
        BlockMatrix<std::complex<double>>& GKBA_rel_ave(const double trel__, const double tave__);
        void Propagator(Operator<std::complex<double>>& Ut__, const double t__);
};

