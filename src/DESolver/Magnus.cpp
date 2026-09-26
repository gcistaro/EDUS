#include <stdexcept>
#include <vector>
#include "DESolver/Magnus.hpp"
#include "core/profiler.hpp"

namespace magnus {

void exponential(BlockMatrix<std::complex<double>>& U__, const BlockMatrix<std::complex<double>>& A__, const double& dt__)
{
    PROFILE("magnus::exponential");
    const int n = A__.get_nrows();
    const std::complex<double>* A_data = A__.data();
    std::complex<double>* U_data = U__.data();
    #pragma omp parallel for schedule(static)
    for (int ik = 0; ik < A__.get_nblocks(); ++ik) {
        const std::complex<double>* a = A_data + std::size_t(ik) * n * n;
        std::complex<double>* u = U_data + std::size_t(ik) * n * n;
        /* hermitian part, to diagonalize with the hermitian eigensolver */
        Matrix<std::complex<double>> A(n, n), V;
        mdarray<double, 1> lambda;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A(i, j) = 0.5 * (a[i * n + j] + std::conj(a[j * n + i]));
            }
        }
        A.diagonalize(V, lambda);
        const std::complex<double>* v = &V(0, 0);
        /* U = V exp(-i dt lambda) V^dagger */
        std::vector<std::complex<double>> w(std::size_t(n) * n);
        for (int i = 0; i < n; ++i) {
            for (int l = 0; l < n; ++l) {
                w[i * n + l] = v[i * n + l] * std::exp(std::complex<double>(0., -dt__ * lambda(l)));
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::complex<double> sum = 0.;
                for (int l = 0; l < n; ++l) {
                    sum += w[i * n + l] * std::conj(v[j * n + l]);
                }
                u[i * n + j] = sum;
            }
        }
    }
}

void step(BlockMatrix<std::complex<double>>& rho__, const BlockMatrix<std::complex<double>>& H1__,
          const BlockMatrix<std::complex<double>>& H2__, const double& dt__)
{
    PROFILE("magnus::step");
    const int nk = rho__.get_nblocks(), n = rho__.get_nrows();
    const std::size_t nn = std::size_t(n) * n, total = std::size_t(nk) * nn;
    BlockMatrix<std::complex<double>> B(k, nk, n, n), U1(k, nk, n, n), U2(k, nk, n, n);
    const std::complex<double>* h1 = H1__.data();
    const std::complex<double>* h2 = H2__.data();
    std::complex<double>* b = B.data();
    /* the second exponential acts first */
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < total; ++i) {
        b[i] = a2 * h1[i] + a1 * h2[i];
    }
    exponential(U2, B, dt__);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < total; ++i) {
        b[i] = a1 * h1[i] + a2 * h2[i];
    }
    exponential(U1, B, dt__);
    /* rho <- U rho U^dagger, U = U1 U2 */
    std::complex<double>* r = rho__.data();
    const std::complex<double>* u1 = U1.data();
    const std::complex<double>* u2 = U2.data();
    #pragma omp parallel for schedule(static)
    for (int ik = 0; ik < nk; ++ik) {
        const std::size_t o = std::size_t(ik) * nn;
        std::vector<std::complex<double>> U(nn), T(nn);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            std::complex<double> sum = 0.;
            for (int l = 0; l < n; ++l) sum += u1[o + i * n + l] * u2[o + l * n + j];
            U[i * n + j] = sum;
        }
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            std::complex<double> sum = 0.;
            for (int l = 0; l < n; ++l) sum += U[i * n + l] * r[o + l * n + j];
            T[i * n + j] = sum;
        }
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            std::complex<double> sum = 0.;
            for (int l = 0; l < n; ++l) sum += T[i * n + l] * std::conj(U[j * n + l]);
            r[o + i * n + j] = sum;
        }
    }
}

void derivative(BlockMatrix<std::complex<double>>& out__, const BlockMatrix<std::complex<double>>& H__,
                const BlockMatrix<std::complex<double>>& rho__)
{
    const int n = rho__.get_nrows();
    const std::size_t nn = std::size_t(n) * n;
    const std::complex<double> mi(0., -1.);
    const std::complex<double>* h = H__.data();
    const std::complex<double>* r = rho__.data();
    std::complex<double>* out = out__.data();
    #pragma omp parallel for schedule(static)
    for (int ik = 0; ik < rho__.get_nblocks(); ++ik) {
        const std::size_t o = std::size_t(ik) * nn;
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            std::complex<double> c = 0.;
            for (int l = 0; l < n; ++l) c += h[o + i * n + l] * r[o + l * n + j] - r[o + i * n + l] * h[o + l * n + j];
            out[o + i * n + j] = mi * c;
        }
    }
}

}

CommutatorEquation& CommutatorFreeMagnus4::commutator_equation(EquationOfMotion<Op>& eom__) const
{
    auto eq = dynamic_cast<CommutatorEquation*>(&eom__);
    if( eq == nullptr || !eq->commutator_form() ) {
        throw std::runtime_error("The Magnus time stepper needs an equation of motion in the form d(rho)/dt = -i[H, rho]: "
                                 "in EDUS use \"peierls\": true and no \"decay\"\n");
    }
    return *eq;
}

void CommutatorFreeMagnus4::initialize(const Op& y__)
{
    for (auto& H : H_) {
        make_workspace(H, y__);
    }
    make_workspace(rho_stage_, y__);
    auto& yk = y__.get_Operator(Space::k);
    stage_.initialize(k, yk.get_nblocks(), yk.get_nrows(), yk.get_ncols());
    history_.clear();
    newest_ = -1;
    nstep_ = 0;
}

void CommutatorFreeMagnus4::initialize_device()
{
    throw std::runtime_error("The Magnus time stepper is not available on GPU yet: use \"solver\": \"RK\"\n");
}

void CommutatorFreeMagnus4::hamiltonian_at(CommutatorEquation& eq__, Op& H__, const double& t__, const Block& rho_k__)
{
    PROFILE("CFM4::hamiltonian");
    std::copy(rho_k__.begin(), rho_k__.end(), rho_stage_.get_Operator(Space::k).begin());
    rho_stage_.lock_space(Space::k);
    eq__.hamiltonian(H__, t__, rho_stage_);
}

void CommutatorFreeMagnus4::step_predictor_corrector(CommutatorEquation& eq__, Op& rho__, const double& t__, const double& dt__)
{
    auto& rho_n = rho__.get_Operator(Space::k);
    const Block rho0 = rho_n;
    const int nk = rho0.get_nblocks(), n = rho0.get_nrows();

    /* derivative at t_n */
    hamiltonian_at(eq__, H_[0], t__, rho0);
    Block drho0(k, nk, n, n), drho1(k, nk, n, n), stage(k, nk, n, n);
    magnus::derivative(drho0, H_[0].get_Operator(Space::k), rho0);

    /* predictor: rho at the Gauss points from the first order Taylor expansion */
    Block rho1 = rho0;
    const double c[2] = {magnus::c1, magnus::c2};
    auto gauss_hamiltonians_taylor = [&]() {
        for (int ig : {0, 1}) {
            for (int ik = 0; ik < nk; ++ik) for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
                stage[ik](i, j) = rho0[ik](i, j) + c[ig] * dt__ * drho0[ik](i, j);
            }
            hamiltonian_at(eq__, H_[ig], t__ + c[ig] * dt__, stage);
        }
    };
    gauss_hamiltonians_taylor();
    magnus::step(rho1, H_[0].get_Operator(Space::k), H_[1].get_Operator(Space::k), dt__);

    /* correctors: cubic Hermite interpolation between rho_n and rho_{n+1} at the Gauss points */
    for (int iteration = 0; iteration < 2; ++iteration) {
        hamiltonian_at(eq__, H_[0], t__ + dt__, rho1);
        magnus::derivative(drho1, H_[0].get_Operator(Space::k), rho1);
        for (int ig : {0, 1}) {
            double s = c[ig];
            double h00 = 2*s*s*s - 3*s*s + 1, h10 = s*s*s - 2*s*s + s, h01 = -2*s*s*s + 3*s*s, h11 = s*s*s - s*s;
            for (int ik = 0; ik < nk; ++ik) for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
                stage[ik](i, j) = h00 * rho0[ik](i, j) + h10 * dt__ * drho0[ik](i, j)
                                + h01 * rho1[ik](i, j) + h11 * dt__ * drho1[ik](i, j);
            }
            hamiltonian_at(eq__, H_[ig], t__ + s * dt__, stage);
        }
        rho1 = rho0;
        magnus::step(rho1, H_[0].get_Operator(Space::k), H_[1].get_Operator(Space::k), dt__);
    }
    std::copy(rho1.begin(), rho1.end(), rho_n.begin());
}

void CommutatorFreeMagnus4::step(EquationOfMotion<Op>& eom__, Op& y__, const double& t__, const double& dt__)
{
    PROFILE("CFM4::step");
    auto& eq = commutator_equation(eom__);
    /* the density matrix is propagated in k: make sure its k component is up to date */
    y__.go_to_k();
    auto& rho = y__.get_Operator(Space::k);
    const double c[2] = {magnus::c1, magnus::c2};

    if( !eq.state_dependent() ) {
        /* H depends only on time: exact order 4 with two Hamiltonians */
        for (int ig : {0, 1}) {
            hamiltonian_at(eq, H_[ig], t__ + c[ig] * dt__, rho);
        }
        magnus::step(rho, H_[0].get_Operator(Space::k), H_[1].get_Operator(Space::k), dt__);
    }
    else {
        /* store rho_n in the history: ring buffer of 4 steps, history_[(newest_ - m) mod 4] = rho_{n-m} */
        if( history_.size() < 4 ) {
            history_.push_back(rho);
            newest_ = int(history_.size()) - 1;
        }
        else {
            newest_ = (newest_ + 1) % 4;
            std::copy(rho.begin(), rho.end(), history_[newest_].begin());
        }
        if( nstep_ < 3 ) {
            step_predictor_corrector(eq, y__, t__, dt__);
        }
        else {
            /* cubic extrapolation of rho at the Gauss points from t_n, t_{n-1}, t_{n-2}, t_{n-3} */
            const std::size_t total = std::size_t(rho.get_nblocks()) * rho.get_nrows() * rho.get_ncols();
            const std::complex<double>* h[4];
            for (int m = 0; m < 4; ++m) {
                h[m] = history_[(newest_ - m + 4) % 4].data();
            }
            std::complex<double>* stage = stage_.data();
            for (int ig : {0, 1}) {
                double s = c[ig];
                double L[4] = {(s + 1) * (s + 2) * (s + 3) / 6., -s * (s + 2) * (s + 3) / 2.,
                               s * (s + 1) * (s + 3) / 2., -s * (s + 1) * (s + 2) / 6.};
                #pragma omp parallel for schedule(static)
                for (std::size_t i = 0; i < total; ++i) {
                    stage[i] = L[0] * h[0][i] + L[1] * h[1][i] + L[2] * h[2][i] + L[3] * h[3][i];
                }
                hamiltonian_at(eq, H_[ig], t__ + s * dt__, stage_);
            }
            magnus::step(rho, H_[0].get_Operator(Space::k), H_[1].get_Operator(Space::k), dt__);
        }
    }
    y__.lock_space(Space::k);
    ++nstep_;
}
