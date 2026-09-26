#include <gtest/gtest.h>
#include "Operator/BlockMatrix.hpp"

TEST(BlockMatrixCopyTest, CopiesValuesCorrectly)
{
    /* allocate memory */
    int N= 10;
    int NN = 10;
    BlockMatrix<std::complex<double>> src(k, N, NN, NN);
    BlockMatrix<std::complex<double>> dst(k, N, NN, NN);
    src.initialize_device();
    dst.initialize_device();

#ifdef EDUS_GPU
    Processor processor = device;
#else
    Processor processor = host;
#endif
    /* Fill source */
    src.fill(2.);
    src.transfer_to(processor);

    /* Get the copy */
    copy(src, dst, processor);
    dst.transfer_to(host);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(x[i].real(), result.real(), 1e-16);
        EXPECT_NEAR(x[i].imag(), result.imag(), 1e-16);
    }
}