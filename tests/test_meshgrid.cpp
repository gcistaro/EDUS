#include <gtest/gtest.h>
#include "MeshGrid/MeshGrid.hpp"
#include "GlobalFunctions.hpp"
#include "initialize.hpp"

/* hexagonal lattice (a = 1, c = 10): the reciprocal vectors are not orthogonal */
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

TEST(MeshGridTest, FindCubeGridPoints) {
    set_basis();
    MeshGrid kgrid(k, {12, 12, 1});
    for (int ik = 0; ik < kgrid.get_TotalSize(); ++ik) {
        EXPECT_EQ(kgrid.find(kgrid[ik]), ik);
    }
    /* points outside the first zone are reduced */
    EXPECT_EQ(kgrid.find(Coordinate(1. + 1./12., -3./12., 0., LatticeVectors(k))),
              kgrid.find(Coordinate(1./12., 9./12., 0., LatticeVectors(k))));
}

TEST(MeshGridTest, FindPointNotInCubeGrid) {
    set_basis();
    MeshGrid kgrid(k, {12, 12, 1});
    EXPECT_EQ(kgrid.find(Coordinate(1./24., 0., 0., LatticeVectors(k))), -1);
    EXPECT_EQ(kgrid.find(Coordinate(0., 1./5., 0., LatticeVectors(k))), -1);
    /* the grid has one point along z */
    EXPECT_EQ(kgrid.find(Coordinate(0., 0., 0.5, LatticeVectors(k))), -1);
}

TEST(MeshGridTest, CommensurateQGrid) {
    set_basis();
    MeshGrid kgrid(k, {12, 12, 1});
    MeshGrid qgrid(k, {3, 4, 1});
    /* every q of a commensurate grid is a k point, and k+q too */
    for (int iq = 0; iq < qgrid.get_TotalSize(); ++iq) {
        EXPECT_GE(kgrid.find(qgrid[iq]), 0);
        for (int ik = 0; ik < kgrid.get_TotalSize(); ++ik) {
            EXPECT_GE(kgrid.find(kgrid[ik] + qgrid[iq]), 0);
        }
    }
    /* q grid closed under difference: q_i - q_j is in the q grid */
    MeshGrid::Calculate_ConvolutionIndex(qgrid, qgrid, qgrid);
    auto& ci = MeshGrid::get_ConvolutionIndex(qgrid, qgrid, qgrid);
    for (int i = 0; i < qgrid.get_TotalSize(); ++i) {
        for (int j = 0; j < qgrid.get_TotalSize(); ++j) {
            ASSERT_GE(ci(i, j), 0);
            auto d = qgrid[i] - qgrid[j] - qgrid[ci(i, j)];
            auto dl = d.get(LatticeVectors(k));
            for (auto ix : {0, 1, 2}) {
                EXPECT_NEAR(dl[ix] - std::round(dl[ix]), 0., 1.e-10);
            }
        }
    }
    /* a k point that is not in the q grid */
    EXPECT_EQ(qgrid.find(kgrid[1]), -1);
}

TEST(MeshGridTest, PointsNearTheEdgeOfTheCell) {
    set_basis();
    MeshGrid kgrid(k, {12, 12, 1});
    int i0 = kgrid.find(Coordinate(0., 0., 0., LatticeVectors(k)));
    int i1 = kgrid.find(Coordinate(1./12., 0., 0., LatticeVectors(k)));
    /* rounding errors of k+q: just below 1 (or just above 0) is the point 0, never the next row */
    for (double eps : {1.e-12, 1.e-9, 5.e-8, 1.e-8}) {
        EXPECT_EQ(kgrid.find(Coordinate(1. - eps, 0., 0., LatticeVectors(k))), i0) << eps;
        EXPECT_EQ(kgrid.find(Coordinate(0., 1. - eps, 0., LatticeVectors(k))), i0) << eps;
        EXPECT_EQ(kgrid.find(Coordinate(-eps, -eps, 0., LatticeVectors(k))), i0) << eps;
        EXPECT_EQ(kgrid.find(Coordinate(1./12. + eps, 1. - eps, 0., LatticeVectors(k))), i1) << eps;
    }
}

TEST(MeshGridTest, RoundingErrorsInCartesian) {
    set_basis();
    MeshGrid kgrid(k, {12, 12, 1});
    MeshGrid qgrid(k, {3, 4, 1});
    /* k+q computed in cartesian coordinates and back, as in the code, plus a small noise */
    for (int iq = 0; iq < qgrid.get_TotalSize(); ++iq) {
        for (int ik = 0; ik < kgrid.get_TotalSize(); ++ik) {
            auto kq = kgrid[ik] + qgrid[iq];
            auto c = kq.get("Cartesian");
            Coordinate noisy(c[0] + 1.e-10, c[1] - 1.e-10, c[2]);
            auto l = kq.get(LatticeVectors(k));
            Coordinate exact(l[0], l[1], l[2], LatticeVectors(k));
            EXPECT_EQ(kgrid.find(noisy), kgrid.find(exact));
            EXPECT_GE(kgrid.find(noisy), 0);
        }
    }
}

TEST(MeshGridTest, FindRGrid) {
    set_basis();
    MeshGrid Rgrid(R, {6, 6, 1});
    for (int iR = 0; iR < Rgrid.get_TotalSize(); ++iR) {
        EXPECT_EQ(Rgrid.find(Rgrid[iR]), iR);
    }
    /* R is periodic with the size of the grid */
    EXPECT_EQ(Rgrid.find(Coordinate(-1., 7., 0., LatticeVectors(R))),
              Rgrid.find(Coordinate(5., 1., 0., LatticeVectors(R))));
    EXPECT_EQ(Rgrid.find(Coordinate(0.5, 0., 0., LatticeVectors(R))), -1);
}

TEST(MeshGridTest, ShiftedGridWithOnePointAlongZ) {
    set_basis();
    MeshGrid kgrid(k, {4, 4, 1}, 0.125);
    for (int ik = 0; ik < kgrid.get_TotalSize(); ++ik) {
        EXPECT_EQ(kgrid.find(kgrid[ik]), ik);
    }
    EXPECT_EQ(kgrid.find(Coordinate(0., 0., 0., LatticeVectors(k))), -1);
}
