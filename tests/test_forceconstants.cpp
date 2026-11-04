#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include "Phonons/DynamicalMatrix.hpp"

/* Diatomic chain along x: atom A at 0 and B at a/2, springs k (Ha/bohr^2) only along x, A(0)-B(0) and A(0)-B(-1).
   The force constants of q2r couple the atom of the first index in the cell R and the one of the second in the cell 0:
     C_AB(R) = -k for R = 0, +1      (A in R bonded to B in 0)
     C_BA(R) = -k for R = 0, -1
     C_AA(0) = C_BB(0) = 2k          (sum rule)
   With u(R) = u(q) e^{iqR}, the force on A(0) is -k(2u_A - u_B - u_B e^{-iq}): K_AB(q) = -k (1 + e^{-i2pi q}). */
static std::string write_chain(const int nr__, const double k_ry__, const bool break_asr__)
{
    std::string name = "chain_test.fc";
    std::ofstream f(name);
    f << "  2    2  0  1.0 0.0 0.0 0.0 0.0 0.0\n";
    f << "  1.0 0.0 0.0\n  0.0 1.0 0.0\n  0.0 0.0 1.0\n";
    f << "  1  'A'  1000.0\n  2  'B'  2000.0\n";
    f << "  1  1  0.0 0.0 0.0\n  2  2  0.5 0.0 0.0\n";
    f << " F\n";
    f << "  " << nr__ << "   1   1\n";
    for (int j1 = 1; j1 <= 3; ++j1) {
        for (int j2 = 1; j2 <= 3; ++j2) {
            for (int na = 1; na <= 2; ++na) {
                for (int nb = 1; nb <= 2; ++nb) {
                    f << j1 << " " << j2 << " " << na << " " << nb << "\n";
                    for (int m1 = 1; m1 <= nr__; ++m1) {
                        double v = 0.;
                        const int R = m1 - 1;               // R = nr - 1 is the cell -1
                        if (j1 == 1 && j2 == 1) {
                            if (na == nb && R == 0) v = 2. * k_ry__ + (break_asr__ && na == 1 ? 0.1 * k_ry__ : 0.);
                            if (na == 1 && nb == 2 && (R == 0 || R == 1)) v = -k_ry__;
                            if (na == 2 && nb == 1 && (R == 0 || R == nr__ - 1)) v = -k_ry__;
                        }
                        f << m1 << " 1 1 " << v << "\n";
                    }
                }
            }
        }
    }
    return name;
}

TEST(ForceConstants, DiatomicChain)
{
    const int nr = 4;
    const double k_ry = 0.2, k = 0.5 * k_ry;
    auto name = write_chain(nr, k_ry, false);
    auto fc = phonon::read_force_constants(name);
    std::remove(name.c_str());
    EXPECT_EQ(fc.num_atoms, 2);
    EXPECT_FALSE(fc.polar);
    EXPECT_NEAR(fc.mass[0], 2000., 1.e-12);   // Ry mass unit = 2 m_e
    EXPECT_NEAR(fc.mass[1], 4000., 1.e-12);
    for (int iq = 0; iq < nr; ++iq) {
        const double q = double(iq) / nr;
        auto dyn = phonon::dynamical_matrix(fc, {q, 0., 0.});
        std::complex<double> KAB = -k * (1. + std::exp(std::complex<double>(0., -2. * M_PI * q)));
        EXPECT_NEAR(std::abs(dyn.C(0, 3) - KAB), 0., 1.e-12) << "q = " << q;
        EXPECT_NEAR(std::abs(dyn.C(3, 0) - std::conj(KAB)), 0., 1.e-12) << "hermitian, q = " << q;
        EXPECT_NEAR(std::abs(dyn.C(0, 0) - 2. * k), 0., 1.e-12);
        /* K(-q) = K(q)^* */
        auto dyn_minus = phonon::dynamical_matrix(fc, {-q, 0., 0.});
        EXPECT_NEAR(std::abs(dyn_minus.C(0, 3) - std::conj(dyn.C(0, 3))), 0., 1.e-12);
    }
    /* off the q grid of q2r */
    EXPECT_THROW(phonon::dynamical_matrix(fc, {0.1, 0., 0.}), std::runtime_error);
}

TEST(ForceConstants, AcousticSumRule)
{
    auto name = write_chain(4, 0.2, true);
    auto fc = phonon::read_force_constants(name);
    std::remove(name.c_str());
    double correction = phonon::impose_acoustic_sum_rule(fc);
    EXPECT_NEAR(correction, 0.5 * 0.1 * 0.2, 1.e-12);
    auto dyn = phonon::dynamical_matrix(fc, {0., 0., 0.});
    for (int mu = 0; mu < 6; ++mu) {
        for (int j = 0; j < 3; ++j) {
            std::complex<double> sum = dyn.C(mu, j) + dyn.C(mu, 3 + j);
            EXPECT_NEAR(std::abs(sum), 0., 1.e-12);
        }
    }
}
