#include <gtest/gtest.h>
#include "initialize.hpp"
#include "fftPair/fftPair.hpp"

TEST(FFTTest, Impulse) {
    initialize(); 
    int N=8;
    mdarray<std::complex<double>,2> xt({N,1});
    mdarray<std::complex<double>,2> xw({N,1});
#ifdef EDUS_GPU
    Processor processor = device;
#else 
    Processor processor = host;
#endif
    xt.fill(0.);
    xt(0,0) = 1;

    xt.initialize_device();    xt.transfer_to(device);
    xw.initialize_device();    xw.transfer_to(device);
    
    FourierTransform fouriertransform; 
    fouriertransform.initialize(xt, xw, {8});
    
    fouriertransform.fft(-1, processor);
    xw.transfer_to(host);

    for (const auto& v : xw) {
        EXPECT_NEAR(v.real(), 1.0/double(N), 1e-12);
        EXPECT_NEAR(v.imag(), 0.0, 1e-12);
    }
}