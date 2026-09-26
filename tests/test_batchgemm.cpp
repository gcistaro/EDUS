#include <gtest/gtest.h>
#include "initialize.hpp"
#include "Operator/BlockMatrix.hpp"

TEST(BatchGEMM, Identity) {
    initialize();

    /* Allocate memory */
    int N = 3;
    BlockMatrix<std::complex<double>> A(k, 1,N,N);
    BlockMatrix<std::complex<double>> B(k, 1,N,N);
    BlockMatrix<std::complex<double>> C(k, 1,N,N);

    /* initialize matrices */
    for ( auto& b : B ) {
        b = 5.;
    }

    A.fill(0.);
    for(int i=0; i<N; i++) A(0,i,i) = 1.;

    /* transferring to device */
    A.initialize_device();       A.transfer_to(device); 
    B.initialize_device();       B.transfer_to(device);
    C.initialize_device();       C.transfer_to(device);

    /* do the multiplication C=0*C+A*B */
    multiply(C, std::complex<double>(1.), A, B, std::complex<double>(0.), device);
    C.transfer_to(host);
    for (const auto& c : C) {
        EXPECT_NEAR(c.real(), 5., 1e-12);
        EXPECT_NEAR(c.imag(), 0.0, 1e-12);
    }

}
