#include <gtest/gtest.h>
#include <complex>
#include <random>
#include "initialize.hpp"
#include "GlobalFunctions.hpp"
#include "GridStructure/GridStructure.hpp"
#include "MPIindex/MPIindex.hpp"
#include "Operator/Operator.hpp"

/* square lattice (a = 1, c = 10) */
static void set_basis()
{
    static bool initialized = false;
    if (!initialized) {
        initialize();
        initialized = true;
    }
    Matrix<double> A(3, 3);
    A(0,0) = 1.; A(0,1) = 0.; A(0,2) = 0.;
    A(1,0) = 0.; A(1,1) = 1.; A(1,2) = 0.;
    A(2,0) = 0.; A(2,1) = 0.; A(2,2) = 10.;
    Coordinate::add_Basis(Basis(A), LatticeVectors(R));
    Coordinate::add_Basis(Basis(2. * pi * A.inverse().transpose()), LatticeVectors(k));
}

/* fills Operator_k(ik, iq, n, m) with random numbers, goes to R and back to k and compares */
static void check_fft_roundtrip(Operator<std::complex<double>>& op__)
{
    auto& op_k = op__.get_Operator_k();
    int nk = op_k.get_nk();
    int nq = op_k.get_nq();
    int nb = op_k.get_nrows();

    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(-1., 1.);
    mdarray<std::complex<double>, 4> copy({nk, nq, nb, nb});
    for (int ik = 0; ik < nk; ++ik) {
        for (int iq = 0; iq < nq; ++iq) {
            for (int n = 0; n < nb; ++n) {
                for (int m = 0; m < nb; ++m) {
                    std::complex<double> value(dist(gen), dist(gen));
                    op_k(ik, iq, n, m) = value;
                    copy(ik, iq, n, m) = value;
                }
            }
        }
    }

    op__.lock_space(k);
    op__.go_to_R();
    op__.go_to_k();

    auto& op_k_after = op__.get_Operator_k();
    for (int ik = 0; ik < nk; ++ik) {
        for (int iq = 0; iq < nq; ++iq) {
            for (int n = 0; n < nb; ++n) {
                for (int m = 0; m < nb; ++m) {
                    EXPECT_NEAR(std::abs(op_k_after(ik, iq, n, m) - copy(ik, iq, n, m)), 0., 1e-12)
                        << "ik=" << ik << " iq=" << iq << " n=" << n << " m=" << m;
                }
            }
        }
    }
}

TEST(OperatorTest, FFTRoundtripNq3) {
    set_basis();
    std::array<int,3> ksize = {6, 6, 1};
    std::array<int,3> qsize = {3, 1, 1};
    int nb = 2;
    GridStructure grids(ksize, qsize);
    int nq = grids.qgrid()->get_TotalSize();
    ASSERT_EQ(nq, 3);

    MPIindex<3> mpindex_q;
    mpindex_q.initialize(ksize, nq * nb * nb);

    Operator<std::complex<double>> op;
    op.initialize_fft(grids.Rgrid(), grids.kgrid(), nb, mpindex_q, "test_nq3", grids.qgrid());

    EXPECT_EQ(op.get_nq(), 3);
    EXPECT_FALSE(op.is_k_diagonal());
    EXPECT_EQ(op.get_Operator_k().get_nk(), mpindex_q.get_nlocal());
    EXPECT_EQ(op.get_Operator_R().get_nq(), 3);

    check_fft_roundtrip(op);

    /* an operator allocated as a copy of op has the same nq and q grid */
    Operator<std::complex<double>> op_copy;
    op_copy.initialize_fft(op, "test_nq3_copy");
    EXPECT_EQ(op_copy.get_nq(), 3);
    EXPECT_FALSE(op_copy.is_k_diagonal());
    check_fft_roundtrip(op_copy);

    /* functions not yet generalized to nq > 1 must throw */
    EXPECT_THROW(op.is_hermitian(), std::runtime_error);
    EXPECT_THROW(op.go_to_bloch(), std::runtime_error);
}

TEST(OperatorTest, FFTRoundtripNq1) {
    set_basis();
    std::array<int,3> ksize = {6, 6, 1};
    std::array<int,3> qsize = {1, 1, 1};
    int nb = 2;
    GridStructure grids(ksize, qsize);

    MPIindex<3> mpindex_k;
    mpindex_k.initialize(ksize, nb * nb);

    Operator<std::complex<double>> op;
    op.initialize_fft(grids.Rgrid(), grids.kgrid(), nb, mpindex_k, "test_nq1");

    EXPECT_EQ(op.get_nq(), 1);
    EXPECT_TRUE(op.is_k_diagonal());
    check_fft_roundtrip(op);
}

TEST(OperatorTest, WrongHowmanyThrows) {
    set_basis();
    std::array<int,3> ksize = {6, 6, 1};
    std::array<int,3> qsize = {3, 1, 1};
    int nb = 2;
    GridStructure grids(ksize, qsize);

    /* mpindex with howmany = nb*nb, but q grid with nq = 3 */
    MPIindex<3> mpindex_k;
    mpindex_k.initialize(ksize, nb * nb);

    Operator<std::complex<double>> op;
    EXPECT_THROW(op.initialize_fft(grids.Rgrid(), grids.kgrid(), nb, mpindex_k, "test_wrong", grids.qgrid()),
                 std::runtime_error);
}

/* the FFT of each q slice of an nq = 3 operator must be equal to the FFT of an nq = 1 operator with the same data */
TEST(OperatorTest, EachQSliceEqualsNq1FFT) {
    set_basis();
    std::array<int,3> ksize = {6, 6, 1};
    std::array<int,3> qsize = {3, 1, 1};
    int nb = 2;
    GridStructure grids(ksize, qsize);
    int nq = grids.qgrid()->get_TotalSize();

    MPIindex<3> mpindex_q;
    mpindex_q.initialize(ksize, nq * nb * nb);
    MPIindex<3> mpindex_k;
    mpindex_k.initialize(ksize, nb * nb);

    Operator<std::complex<double>> op_q;
    op_q.initialize_fft(grids.Rgrid(), grids.kgrid(), nb, mpindex_q, "slice_nq3", grids.qgrid());
    std::vector<Operator<std::complex<double>>> op_1(nq);
    for (int iq = 0; iq < nq; ++iq) {
        op_1[iq].initialize_fft(grids.Rgrid(), grids.kgrid(), nb, mpindex_k, "slice_nq1");
    }

    auto& op_q_k = op_q.get_Operator_k();
    int nk = op_q_k.get_nk();
    ASSERT_EQ(nk, op_1[0].get_Operator_k().get_nk());
    std::mt19937 gen(54321);
    std::uniform_real_distribution<double> dist(-1., 1.);
    for (int ik = 0; ik < nk; ++ik) {
        for (int iq = 0; iq < nq; ++iq) {
            for (int n = 0; n < nb; ++n) {
                for (int m = 0; m < nb; ++m) {
                    std::complex<double> value(dist(gen), dist(gen));
                    op_q_k(ik, iq, n, m) = value;
                    op_1[iq].get_Operator_k()(ik, 0, n, m) = value;
                }
            }
        }
    }

    op_q.lock_space(k);
    op_q.go_to_R();
    for (int iq = 0; iq < nq; ++iq) {
        op_1[iq].lock_space(k);
        op_1[iq].go_to_R();
    }

    auto& op_q_R = op_q.get_Operator_R();
    for (int iR = 0; iR < nk; ++iR) {
        for (int iq = 0; iq < nq; ++iq) {
            auto& op_1_R = op_1[iq].get_Operator_R();
            for (int n = 0; n < nb; ++n) {
                for (int m = 0; m < nb; ++m) {
                    EXPECT_NEAR(std::abs(op_q_R(iR, iq, n, m) - op_1_R(iR, 0, n, m)), 0., 1e-12)
                        << "iR=" << iR << " iq=" << iq << " n=" << n << " m=" << m;
                }
            }
        }
    }
}
