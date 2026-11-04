#include <gtest/gtest.h>
#include "initialize.hpp"
#include "GridStructure/GridStructure.hpp"

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

TEST(GridStructureTest, KplusQmap) {
    set_basis();
    std::array<int,3> ksize = {6, 6, 1};
    std::array<int,3> qsize = {3, 2, 1};
    GridStructure gridstructure(ksize, qsize);
    auto& ikq = gridstructure.ikq();
    auto& kgrid = *(gridstructure.kgrid());
    auto& qgrid = *(gridstructure.qgrid());
    for (int ik = 0; ik < kgrid.get_TotalSize(); ++ik) {
        for(int iq = 0; iq < qgrid.get_TotalSize(); ++iq) {
            auto kq = kgrid[ik] + qgrid[iq];
            auto G = kgrid[ikq(ik,iq)] - kgrid[ik] - qgrid[iq];
            auto& Gc = G.get(LatticeVectors(k));
            for(auto& ix : {0,1,2}) {
                EXPECT_NEAR(Gc[ix] - std::round(Gc[ix]), 0., 1.e-10);
            }
        }
    }
}

TEST(GridStructureTest, QminusPmap) {
    set_basis();
    std::array<int,3> ksize = {6, 6, 1};
    std::array<int,3> qsize = {3, 2, 1};
    GridStructure gridstructure(ksize, qsize);
    auto& iqmp = gridstructure.iqmp();
    auto& qgrid = *(gridstructure.qgrid());
    for (int iq = 0; iq < qgrid.get_TotalSize(); ++iq) {
        for(int ip = 0; ip < qgrid.get_TotalSize(); ++ip) {
            /* q[iqmp(iq,ip)] = q[iq] - q[ip] up to a reciprocal lattice vector G */
            auto G = qgrid[iqmp(iq,ip)] - qgrid[iq] + qgrid[ip];
            auto& Gc = G.get(LatticeVectors(k));
            for(auto& ix : {0,1,2}) {
                EXPECT_NEAR(Gc[ix] - std::round(Gc[ix]), 0., 1.e-10);
            }
        }
        EXPECT_EQ(iqmp(iq, 0), iq);
        EXPECT_EQ(iqmp(iq, iq), 0);
    }
}
