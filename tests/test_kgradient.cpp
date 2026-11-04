#include <gtest/gtest.h>
#include <complex>
#include <random>
#include "initialize.hpp"
#include "GlobalFunctions.hpp"
#include "MeshGrid/MeshGrid.hpp"
#include "fftPair/fftPair.hpp"
#include "kGradient/kGradient.hpp"
#include "Operator/BlockMatrix.hpp"

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

/* in R the gradient is Derivative(R,q) = i*R.direction*Function(R,q), the same factor for every q */
TEST(kGradientTest, GradientInRWithNq3) {
    set_basis();
    MeshGrid kmesh(k, std::array<int,3>({6, 6, 1}));
    MeshGrid Rmesh = fftPair(kmesh);
    kGradient gradient(Rmesh);
    /* kGradient uses the Gamma centered R grid (R in the first Wigner-Seitz-like cell) */
    MeshGrid Rmesh_gamma = get_GammaCentered_grid(Rmesh);

    MPIindex<3> mpindex;
    mpindex.initialize(Rmesh.get_Size());
    int nR = mpindex.get_nlocal();
    int nq = 3;
    int nb = 2;

    BlockMatrix<std::complex<double>> function(R, nR, nb, nb, 0, nq);
    BlockMatrix<std::complex<double>> derivative(R, nR, nb, nb, 0, nq);
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> dist(-1., 1.);
    for (int iR = 0; iR < nR; ++iR) {
        for (int iq = 0; iq < nq; ++iq) {
            for (int n = 0; n < nb; ++n) {
                for (int m = 0; m < nb; ++m) {
                    function(iR, iq, n, m) = std::complex<double>(dist(gen), dist(gen));
                }
            }
        }
    }

    Coordinate direction(0.3, -0.7, 0., "Cartesian");
    gradient.Calculate(1. + 0.*im, derivative, function, direction, true);

    for (int iR = 0; iR < nR; ++iR) {
        auto iR_glob = mpindex.loc1D_to_glob1D(iR);
        auto factor = im*Rmesh_gamma[iR_glob].dot(direction);
        for (int iq = 0; iq < nq; ++iq) {
            for (int n = 0; n < nb; ++n) {
                for (int m = 0; m < nb; ++m) {
                    EXPECT_NEAR(std::abs(derivative(iR, iq, n, m) - factor*function(iR, iq, n, m)), 0., 1e-12)
                        << "iR=" << iR << " iq=" << iq << " n=" << n << " m=" << m;
                }
            }
        }
    }
}
