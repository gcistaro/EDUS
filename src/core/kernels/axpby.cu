
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <stdio.h>

__global__
void axpby_kernel(
    cuDoubleComplex* Output,
    const cuDoubleComplex a,
    const cuDoubleComplex* x,
    const cuDoubleComplex b,
    const cuDoubleComplex* y, 
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    cuDoubleComplex xi = x[i];
    cuDoubleComplex yi = y[i];

    cuDoubleComplex term1 = cuCmul(a, xi);
    cuDoubleComplex term2 = cuCmul(b, yi);

    Output[i] = cuCadd(term1, term2);
}




void axpby_gpu(std::complex<double>* Output, 
               const std::complex<double> a, 
               const std::complex<double>* x, 
               const std::complex<double> b, 
               const std::complex<double>* y,
               int N)
{
    int threads = 256;
    int blocks  = (N+threads-1)/threads;
    cuDoubleComplex a_c = make_cuDoubleComplex(a.real(), a.imag());
    cuDoubleComplex b_c = make_cuDoubleComplex(b.real(), b.imag());
    axpby_kernel<<<blocks, threads>>>(reinterpret_cast<cuDoubleComplex*>(Output),
                                      a_c,
                                      reinterpret_cast<const cuDoubleComplex*>(x),
                                      b_c,
                                      reinterpret_cast<const cuDoubleComplex*>(y),
                                      N);
}
