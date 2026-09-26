#include <gtest/gtest.h>
#include "initialize.hpp"
#include "mdContainers/mdContainers.hpp"
#include "LinearAlgebra/axpby.hpp"

TEST(AxpbyTest, InPlaceX)
{
    initialize();
#ifdef EDUS_GPU
    Processor processor = device;
#else
    Processor processor = host;
#endif
    /* allocate memory */
    const int N = 256;
    mdarray<std::complex<double>, 1> x({N}), y({N}), x_copy({N});
    x.initialize_device(); 
    y.initialize_device(); 

    /* initialize vectors */
    for (int i = 0; i < N; i++) {
        x[i] = {i, 3*i};
        y[i] = {2*i, -i};
        x_copy[i] = x[i];
    }

    std::complex<double> a(2.0, 0.0);
    std::complex<double> b(1.0, 0.0);

    /* do the operation */
    x.transfer_to(processor);
    y.transfer_to(processor);
    axpby(x, a, x, b, y);

    /* check result */
    for (int i = 0; i < N; i++) {
        auto result = a * x_copy[i] + b * y[i];
        EXPECT_NEAR(x[i].real(), result.real(), 1e-12);
        EXPECT_NEAR(x[i].imag(), result.imag(), 1e-12);
    }
    finalize();
}