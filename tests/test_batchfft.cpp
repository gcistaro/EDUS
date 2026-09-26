#include <gtest/gtest.h>
#include "initialize.hpp"
#include "fftPair/fftPair.hpp"
#include <complex>
#include <cmath>

TEST(FFTTest, BatchedImpulse) {
    initialize();

    int N = 8;
    int B = 3; // number of FFT

    mdarray<std::complex<double>,2> xt({N, B});
    mdarray<std::complex<double>,2> xw({N, B});

#ifdef EDUS_GPU
    Processor processor = device;
#else 
    Processor processor = host;
#endif

    xt.fill(0.);

    /* pulse with different positions in every batch */
    xt(0,0) = 1.; // batch 0
    xt(1,1) = 1.; // batch 1
    xt(2,2) = 1.; // batch 2

    xt.initialize_device(); xt.transfer_to(device);
    xw.initialize_device(); xw.transfer_to(device);

    FourierTransform fouriertransform;
    fouriertransform.initialize(xt, xw, {N}); // stessa dimensione FFT

    fouriertransform.fft(-1, processor);
    xw.transfer_to(host);

    // verifica
    for (int b = 0; b < B; b++) {
        for (int w = 0; w < N; w++) {
            std::cout << b << " " << w << std::endl;
            std::complex<double> expected = std::exp(-2.0*pi*im* double(w * b) / double(N)) / double(N);

            auto val = xw(w, b);
            EXPECT_NEAR(val.real(), expected.real(), 1e-12);
            EXPECT_NEAR(val.imag(), expected.imag(), 1e-12);
        }
    }
}