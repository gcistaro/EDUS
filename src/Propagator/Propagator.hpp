#ifndef PROPAGATOR_HPP
#define PROPAGATOR_HPP

#include "DESolver/DESolver.hpp"
#include "kGradient/kGradient.hpp"
#include "Laser/Laser.hpp"
#include "GridStructure/GridStructure.hpp"
#include "Electrons/System.hpp"
#include "Electrons/State.hpp"
#include "MeanField/MeanField.hpp"
#include "Propagator/PropagatorParameters.hpp"

/// @brief Time propagation of the electronic density matrix. It defines the equation of motion
/// @f[
/// \frac{\partial \rho}{\partial t} = -i [H_0 + H_{\text{eff}} + \boldsymbol{\varepsilon}(t)\cdot \Xi, \rho] +
/// \boldsymbol{\varepsilon}(t)\nabla_\textbf{k} \rho
/// @f]
/// and advances it in time with DESolver. The class does not own the physical objects:
/// it only keeps pointers to them, so they must outlive the Propagator.
class Propagator
{
    private:
        PropagatorParameters parameters_;
        /// Immutable part of the electronic system (H0, r, DM0, ...)
        const electron::System* system_ = nullptr;
        /// Propagated variables (density matrix, H, auxiliary arrays)
        electron::State* state_ = nullptr;
        /// Effective Hamiltonian from the electron-electron interaction
        electron::MeanField* meanfield_ = nullptr;
        /// Lasers; not const because the vector potential is integrated in time
        SetOfLaser* lasers_ = nullptr;
        /// Gradient in k, shared with System (used for the velocity)
        const kGradient* kgradient_ = nullptr;
        /// R grid centered in Gamma, used for the Peierls phase
        std::shared_ptr<MeshGrid> Rgrid_gamma_;
        /// Same as Rgrid_gamma_ (local part, cartesian), in a contiguous array for the gpu
        mdarray<double,2> bare_Rgrid_gamma_;

        /// Driver for the time propagation, it defines how we solve the differential equations
        DESolver<Operator<std::complex<double>>> desolver_;
        /// Space where we calculate the commutator @f$ [H, \rho] @f$
        Space SpaceOfPropagation_ = k;
        /// Space where we evaluate H=H0+E \cdot r
        Space SpaceOfCalculateTDHamiltonian_ = R;
        /// Processor where to run the heavy parts of the propagation
#ifdef EDUS_GPU
        Processor processor_ = device;
#else
        Processor processor_ = host;
#endif

        /// Initial condition for DESolver: the equilibrium density matrix
        void InitialCondition(Operator<std::complex<double>>& DM__);
        /// Right hand side of the equation of motion, for DESolver
        void SourceTerm(Operator<std::complex<double>>& Output__, const double& time__,
                        const Operator<std::complex<double>>& Input__);

    public:
        Propagator() = default;
        /// DESolver keeps callbacks bound to this object: copying or moving it would leave them dangling
        Propagator(const Propagator&) = delete;
        Propagator& operator=(const Propagator&) = delete;

        void initialize(const PropagatorParameters& parameters__,
                        const GridStructure& gridstructure__,
                        const electron::System& system__,
                        electron::State& state__,
                        electron::MeanField& meanfield__,
                        SetOfLaser& lasers__,
                        const kGradient& kgradient__);

        /// Advances the density matrix of one time step
        void step();

        /// Calculates H_ = H0 + E(t) \cdot r in the state
        void Calculate_TDHamiltonian(const double& time__, const bool& erase_H__);
        /// Multiplies O__(R) by exp(i*sign*A(t) \cdot R)
        void Apply_Peierls_phase(Operator<std::complex<double>>& O__, const double& time__, const int sign,
                                 const Processor& proc__=host);

        void initialize_device();
        void print_recap() const;

        /* getter methods */
        const PropagatorParameters& parameters() const { return parameters_; }
        double current_time() const { return desolver_.get_CurrentTime(); }
        double resolution_time() const { return desolver_.get_ResolutionTime(); }
        Processor processor() const { return processor_; }
};

/// Functions to get the device code working
#ifdef EDUS_GPU
void Calculate_TDHamiltonian_gpu( std::complex<double>* H__,
                                  const std::complex<double>* H0__,
                                  const std::complex<double>* x__,
                                  const std::complex<double>* y__,
                                  const std::complex<double>* z__,
                                  double* las0__,
                                  double* las1__,
                                  double* las2__,
                                  int N__
                                );


void Apply_Peierls_phase_gpu( std::complex<double>* O__,
                              std::complex<double>* Peierls_phase,
                              double* A0__,
                              double* A1__,
                              double* A2__,
                              double* Rvectors__,
                              int sign,
                              int N,
                              int nR
                            );
#endif

void Calculate_TDHamiltonian_cpu( BlockMatrix<std::complex<double>>& H,
                                  const BlockMatrix<std::complex<double>>& H0,
                                  const BlockMatrix<std::complex<double>>& x,
                                  const BlockMatrix<std::complex<double>>& y,
                                  const BlockMatrix<std::complex<double>>& z,
                                  const Vector<double>& las);

void Apply_Peierls_phase_cpu( BlockMatrix<std::complex<double>>& OR__,
                              mdarray<std::complex<double>,1>& Peierls_phase,
                              const Coordinate& At,
                              const MeshGrid& Rgrid_gamma,
                              int sign);

#endif // PROPAGATOR_HPP
