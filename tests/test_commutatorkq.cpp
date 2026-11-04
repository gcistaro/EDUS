#include <gtest/gtest.h>
#include <complex>
#include <random>
#include <vector>
#include "initialize.hpp"
#include "GridStructure/GridStructure.hpp"
#include "Operator/BlockMatrix.hpp"
#include "MPIindex/MPIindex.hpp"
#include "Parallel/HaloK.hpp"

/* hexagonal lattice (a = 1, c = 10) */
static void set_basis()
{
    static bool initialized = false;
    if (!initialized) {
        initialize();
        initialized = true;
    }
    Matrix<double> A(3, 3);
    A(0,0) = 1.;   A(0,1) = -0.5;               A(0,2) = 0.;
    A(1,0) = 0.;   A(1,1) = std::sqrt(3.) / 2.; A(1,2) = 0.;
    A(2,0) = 0.;   A(2,1) = 0.;                 A(2,2) = 10.;
    Coordinate::add_Basis(Basis(A), LatticeVectors(R));
    Coordinate::add_Basis(Basis(2. * pi * A.inverse().transpose()), LatticeVectors(k));
}

using cd = std::complex<double>;

static void fill_random(BlockMatrix<cd>& X__, std::mt19937& gen__)
{
    std::uniform_real_distribution<double> d(-1., 1.);
    for (int ik = 0; ik < X__.get_nk(); ++ik)
        for (int iq = 0; iq < X__.get_nq(); ++iq)
            for (int n = 0; n < X__.get_nrows(); ++n)
                for (int m = 0; m < X__.get_ncols(); ++m)
                    X__(ik, iq, n, m) = cd(d(gen__), d(gen__));
}

/* full matrix on the whole k grid: X_big[(k+q, n), (k, m)] = X_nm(k, q) */
static std::vector<cd> to_big(const BlockMatrix<cd>& X__, const mdarray<int,2>& ikq__)
{
    const int nk = X__.get_nk(), nq = X__.get_nq(), nb = X__.get_nrows(), N = nk*nb;
    std::vector<cd> big(N*N, 0.);
    for (int ik = 0; ik < nk; ++ik)
        for (int iq = 0; iq < nq; ++iq)
            for (int n = 0; n < nb; ++n)
                for (int m = 0; m < nb; ++m)
                    big[(ikq__(ik, iq)*nb + n)*N + ik*nb + m] = X__(ik, iq, n, m);
    return big;
}

/* [A, B] with A and B both with q components, against the commutator of the full matrices on the k grid */
TEST(CommutatorKQ, GeneralMatchesFullMatrix) {
    set_basis();
    std::array<int,3> ksize = {6, 6, 1};
    std::array<int,3> qsize = {3, 2, 1};
    GridStructure grids(ksize, qsize);
    const int nk = grids.kgrid()->get_TotalSize();
    const int nq = grids.qgrid()->get_TotalSize();
    const int nb = 2;
    const int N = nk*nb;
    auto& ikq = grids.ikq();
    auto& iqmp = grids.iqmp();

    BlockMatrix<cd> A(k, nk, nb, nb, 0, nq), B(k, nk, nb, nb, 0, nq), out(k, nk, nb, nb, 0, nq);
    std::mt19937 gen(7);
    fill_random(A, gen);
    fill_random(B, gen);
    fill_random(out, gen);
    BlockMatrix<cd> out0(k, nk, nb, nb, 0, nq);
    std::copy(out.begin(), out.end(), out0.begin());

    const cd s = -im;
    commutator_kq_general(out, s, A, B, ikq, iqmp);

    auto Abig = to_big(A, ikq);
    auto Bbig = to_big(B, ikq);
    std::vector<cd> C(N*N, 0.);
    for (int i = 0; i < N; ++i)
        for (int l = 0; l < N; ++l)
            for (int j = 0; j < N; ++j)
                C[i*N + j] += Abig[i*N + l]*Bbig[l*N + j] - Bbig[i*N + l]*Abig[l*N + j];

    /* the output adds to what was there before */
    for (int ik = 0; ik < nk; ++ik)
        for (int iq = 0; iq < nq; ++iq)
            for (int n = 0; n < nb; ++n)
                for (int m = 0; m < nb; ++m) {
                    cd ref = out0(ik, iq, n, m) + s*C[(ikq(ik, iq)*nb + n)*N + ik*nb + m];
                    EXPECT_NEAR(std::abs(out(ik, iq, n, m) - ref), 0., 1e-12)
                        << "ik=" << ik << " iq=" << iq << " n=" << n << " m=" << m;
                }
}

/* with A(k,p) = delta_{p0} H(k) the general commutator is the one of commutator_kq */
TEST(CommutatorKQ, GeneralWithDiagonalAEqualsKQ) {
    set_basis();
    std::array<int,3> ksize = {6, 6, 1};
    std::array<int,3> qsize = {3, 2, 1};
    GridStructure grids(ksize, qsize);
    const int nk = grids.kgrid()->get_TotalSize();
    const int nq = grids.qgrid()->get_TotalSize();
    const int nb = 3;
    auto& ikq = grids.ikq();
    auto& iqmp = grids.iqmp();

    BlockMatrix<cd> H(k, nk, nb, nb), Hq(k, nk, nb, nb, 0, nq), rho(k, nk, nb, nb, 0, nq);
    BlockMatrix<cd> out1(k, nk, nb, nb, 0, nq), out2(k, nk, nb, nb, 0, nq);
    std::mt19937 gen(11);
    fill_random(H, gen);
    fill_random(rho, gen);
    Hq.fill(0.);
    for (int ik = 0; ik < nk; ++ik)
        for (int n = 0; n < nb; ++n)
            for (int m = 0; m < nb; ++m)
                Hq(ik, 0, n, m) = H(ik, 0, n, m);

    out1.fill(0.);
    out2.fill(0.);
    commutator_kq(out1, -im, H, rho, true, host, ikq);
    commutator_kq_general(out2, -im, Hq, rho, ikq, iqmp);
    for (int ik = 0; ik < nk; ++ik)
        for (int iq = 0; iq < nq; ++iq)
            for (int n = 0; n < nb; ++n)
                for (int m = 0; m < nb; ++m)
                    EXPECT_NEAR(std::abs(out1(ik, iq, n, m) - out2(ik, iq, n, m)), 0., 1e-13);
}

/* value of the random test operators as a function of the GLOBAL indices: the same on every rank */
static cd value_global(int seed__, int ik__, int iq__, int n__, int m__)
{
    std::mt19937 gen(seed__ + 1000003*ik__ + 10007*iq__ + 101*n__ + m__);
    std::uniform_real_distribution<double> d(-1., 1.);
    return cd(d(gen), d(gen));
}

/* with the k points split among the ranks (as FFTW-MPI does), HaloK brings the k + p of other ranks:
   the commutators on the local k must be equal to the ones computed with all the k points */
TEST(CommutatorKQ, DistributedWithHaloK) {
    set_basis();
    std::array<int,3> ksize = {8, 4, 1};
    std::array<int,3> qsize = {4, 2, 1};
    GridStructure grids(ksize, qsize);
    const int nk = grids.kgrid()->get_TotalSize();
    const int nq = grids.qgrid()->get_TotalSize();
    const int nb = 2;
    auto& ikq = grids.ikq();
    auto& iqmp = grids.iqmp();

    MPIindex<3> mpindex_q;
    mpindex_q.initialize(ksize, nq*nb*nb);
    const int nlocal = mpindex_q.get_nlocal();
    const int offset = int(mpindex_q.get_LocalRange().first);

    parallel::HaloK halo;
    halo.initialize(ikq, mpindex_q);

    /* global operators (reference) and local slabs */
    BlockMatrix<cd> A(k, nk, nb, nb, 0, nq), B(k, nk, nb, nb, 0, nq), H(k, nk, nb, nb);
    BlockMatrix<cd> A_loc(k, nlocal, nb, nb, 0, nq), B_loc(k, nlocal, nb, nb, 0, nq), H_loc(k, nlocal, nb, nb);
    for (int ik = 0; ik < nk; ++ik)
        for (int iq = 0; iq < nq; ++iq)
            for (int n = 0; n < nb; ++n)
                for (int m = 0; m < nb; ++m) {
                    A(ik, iq, n, m) = value_global(1, ik, iq, n, m);
                    B(ik, iq, n, m) = value_global(2, ik, iq, n, m);
                    if (iq == 0) H(ik, 0, n, m) = value_global(3, ik, 0, n, m);
                }
    for (int ik = 0; ik < nlocal; ++ik)
        for (int iq = 0; iq < nq; ++iq)
            for (int n = 0; n < nb; ++n)
                for (int m = 0; m < nb; ++m) {
                    A_loc(ik, iq, n, m) = A(ik + offset, iq, n, m);
                    B_loc(ik, iq, n, m) = B(ik + offset, iq, n, m);
                    if (iq == 0) H_loc(ik, 0, n, m) = H(ik + offset, 0, n, m);
                }

    /* reference with all the k points */
    BlockMatrix<cd> ref_g(k, nk, nb, nb, 0, nq), ref_kq(k, nk, nb, nb, 0, nq);
    ref_g.fill(0.);
    commutator_kq_general(ref_g, -im, A, B, ikq, iqmp);
    commutator_kq(ref_kq, -im, H, B, true, host, ikq);

    /* distributed */
    BlockMatrix<cd> A_h(k, halo.n_halo(), nb, nb, 0, nq), B_h(k, halo.n_halo(), nb, nb, 0, nq), H_h(k, halo.n_halo(), nb, nb);
    halo.exchange(A_h, A_loc);
    halo.exchange(B_h, B_loc);
    halo.exchange(H_h, H_loc);
    BlockMatrix<cd> out_g(k, nlocal, nb, nb, 0, nq), out_kq(k, nlocal, nb, nb, 0, nq);
    out_g.fill(0.);
    commutator_kq_general(out_g, -im, A_h, B_h, ikq, iqmp, &halo.position(), halo.offset());
    commutator_kq(out_kq, -im, H_h, B_loc, true, host, ikq, &halo.position(), halo.offset());

    for (int ik = 0; ik < nlocal; ++ik)
        for (int iq = 0; iq < nq; ++iq)
            for (int n = 0; n < nb; ++n)
                for (int m = 0; m < nb; ++m) {
                    EXPECT_NEAR(std::abs(out_g(ik, iq, n, m) - ref_g(ik + offset, iq, n, m)), 0., 1e-13);
                    EXPECT_NEAR(std::abs(out_kq(ik, iq, n, m) - ref_kq(ik + offset, iq, n, m)), 0., 1e-13);
                }
}
