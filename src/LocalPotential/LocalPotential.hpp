#ifndef LOCALPOTENTIAL_HPP
#define LOCALPOTENTIAL_HPP

#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <vector>
#include "Constants.hpp"
#include "Geometry/Coordinate.hpp"
#include "InputVariables/config.hpp"
#include "MeshGrid/MeshGrid.hpp"
#include "Operator/BlockMatrix.hpp"

namespace electron
{
    struct LocalPotentialParameters
    {
        bool enabled = false;
        /// wave vector, crystal coordinates of the reciprocal lattice
        std::array<double, 3> q = {0., 0., 0.};
        /// amplitude (a.u.)
        double amplitude = 0.;
        /// center and width of the Gaussian envelope (a.u.)
        double t0 = 0.;
        double sigma = 1.;
    };

    class LocalPotentialParametersFactory
    {
    public:
        /// amplitude, t0 and sigma are expected already converted in atomic units
        static LocalPotentialParameters create(const config_t& cfg__)
        {
            auto in = cfg__.local_potential();
            LocalPotentialParameters p;
            p.enabled = in.enabled();
            p.q = in.q();
            p.amplitude = in.amplitude();
            p.t0 = in.t0();
            p.sigma = in.sigma();
            return p;
        }
    };

    /// @brief External local potential at finite momentum, V(r,t) = V0 f(t) cos(q0.r), f(t) = exp(-(t-t0)^2/(2 sigma^2)).
    /// In the Wannier basis it is diagonal and local, V_n(R) = V0 f(t) cos(q0.(R + tau_n)). With
    /// X(R;q) = 1/N sum_R1 e^{-iqR1} X_{nR1,mR1+R}, only R = 0 and q = +-q0 survive:
    ///     V_nm(R;q) = delta_{R0} delta_nm V0 f(t)/2 [ delta_{q,q0} e^{+iq0.tau_n} + delta_{q,-q0} e^{-iq0.tau_n} ]
    /// so in k it does not depend on k. It is added to the q = +-q0 slices of the Hamiltonian, which must have
    /// the q components.
    class LocalPotential
    {
        private:
            LocalPotentialParameters parameters_;
            /// index of +q0 and -q0 in the q grid (the same point if q0 = G/2)
            int iq_plus_ = -1;
            int iq_minus_ = -1;
            /// e^{+iq0.tau_n}, e^{-iq0.tau_n}
            std::vector<std::complex<double>> phase_plus_, phase_minus_;

        public:
            LocalPotential() = default;

            void initialize(const LocalPotentialParameters& parameters__, const std::shared_ptr<MeshGrid>& qgrid__,
                            const std::vector<Coordinate>& wannier_centers__)
            {
                parameters_ = parameters__;
                if( !parameters_.enabled ) {
                    return;
                }
                if( parameters_.sigma <= 0. ) {
                    throw std::runtime_error("local_potential: sigma must be > 0\n");
                }
                const auto& q = parameters_.q;
                Coordinate q0(q[0], q[1], q[2], LatticeVectors(k));
                Coordinate mq0(-q[0], -q[1], -q[2], LatticeVectors(k));
                iq_plus_ = qgrid__->find(q0);
                iq_minus_ = qgrid__->find(mq0);
                if( iq_plus_ < 0 || iq_minus_ < 0 ) {
                    throw std::runtime_error("local_potential: q (and -q) must be points of the q grid\n");
                }
                if( iq_plus_ == 0 ) {
                    throw std::runtime_error("local_potential: q must be different from 0\n");
                }
                /* q0.tau_n with q0 and tau_n cartesian */
                auto q0c = q0.get("Cartesian");
                const int nb = int(wannier_centers__.size());
                phase_plus_.resize(nb);
                phase_minus_.resize(nb);
                for( int n = 0; n < nb; ++n ) {
                    auto tau = wannier_centers__[n].get("Cartesian");
                    const double qtau = q0c[0]*tau[0] + q0c[1]*tau[1] + q0c[2]*tau[2];
                    phase_plus_[n] = std::exp(+im*qtau);
                    phase_minus_[n] = std::exp(-im*qtau);
                }
            }

            bool enabled() const { return parameters_.enabled; }
            int iq_plus() const { return iq_plus_; }
            int iq_minus() const { return iq_minus_; }

            /// V0 f(t)
            double amplitude(const double time__) const
            {
                const double x = (time__ - parameters_.t0)/parameters_.sigma;
                return parameters_.amplitude*std::exp(-0.5*x*x);
            }

            /// V0 df/dt
            double amplitude_derivative(const double time__) const
            {
                return -(time__ - parameters_.t0)/(parameters_.sigma*parameters_.sigma)*amplitude(time__);
            }
            /// Energy of the electrons in the potential, per unit cell and spin channel, on the R = 0 of this rank:
            ///     <V> = 1/N sum_{k,q} Tr[V(k+q,-q) rho(k,q)] = sum_q sum_R sum_mn conj(V_mn(R;q)) rho_mn(R;q)
            /// (V(k+q,-q) = V(k,q)^dagger), only R = 0 and q = +-q0. With derivative__ = true the same with dV/dt, the
            /// power of the potential: d/dt (E + <V>) = <dV/dt>.
            /// @param rho__ density matrix in R with the q components
            double energy(const BlockMatrix<std::complex<double>>& rho__, const int iR0_local__, const double time__,
                          const bool derivative__ = false) const
            {
                if( !parameters_.enabled || iR0_local__ < 0 ) {
                    return 0.;
                }
                const double v = 0.5*( derivative__ ? amplitude_derivative(time__) : amplitude(time__) );
                std::complex<double> e = 0.;
                for( int n = 0; n < rho__.get_nrows(); ++n ) {
                    /* with q0 = G/2 both terms are in the same slice, as in add() */
                    e += std::conj(v*phase_plus_[n])*rho__(iR0_local__, iq_plus_, n, n);
                    e += std::conj(v*phase_minus_[n])*rho__(iR0_local__, iq_minus_, n, n);
                }
                return e.real();
            }
            /// H(R = 0; q = +-q0) += V0 f(t)/2 e^{+-iq0.tau_n} on the diagonal.
            /// @param H__ Hamiltonian in R with the q components
            /// @param iR0_local__ local index of R = 0 in H__, -1 if R = 0 is not on this rank
            void add(BlockMatrix<std::complex<double>>& H__, const int iR0_local__, const double time__) const
            {
                if( !parameters_.enabled || iR0_local__ < 0 ) {
                    return;
                }
                const double v = 0.5*amplitude(time__);
                for( int n = 0; n < H__.get_nrows(); ++n ) {
                    /* with q0 = G/2 the two terms go to the same slice: +=, not = */
                    H__(iR0_local__, iq_plus_, n, n) += v*phase_plus_[n];
                    H__(iR0_local__, iq_minus_, n, n) += v*phase_minus_[n];
                }
            }

            /// q indices where V is not zero, plus 0 (where H0 + E.r is): the only p of H(k,p) != 0 without mean field
            std::vector<int> active_q() const
            {
                std::vector<int> p = {0, iq_plus_};
                if( iq_minus_ != iq_plus_ ) {
                    p.push_back(iq_minus_);
                }
                return p;
            }
    };
}

#endif
