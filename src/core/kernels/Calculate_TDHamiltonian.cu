#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
__global__
void Calculate_TDHamiltonian_kernel(
    cuDoubleComplex* __restrict__ H,
    const cuDoubleComplex* __restrict__ H0,
    const cuDoubleComplex* __restrict__ x,
    const cuDoubleComplex* __restrict__ y,
    const cuDoubleComplex* __restrict__ z,
    double* l0,
    double* l1,
    double* l2,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    H[i] = cuCadd(H[i], cuCadd(H0[i],
           cuCadd(cuCmul(make_cuDoubleComplex(*l0, 0.0), x[i]),
           cuCadd(cuCmul(make_cuDoubleComplex(*l1, 0.0), y[i]),
                  cuCmul(make_cuDoubleComplex(*l2, 0.0), z[i])))));
}

void Calculate_TDHamiltonian_gpu( std::complex<double>* H, 
                                  const std::complex<double>* H0, 
                                  const std::complex<double>* x, 
                                  const std::complex<double>* y, 
                                  const std::complex<double>* z, 
                                  double* las0,
                                  double* las1,
                                  double* las2,
                                  int N
                                )
{
    int threads = 256;
    int blocks  = (N+threads-1)/threads;
    Calculate_TDHamiltonian_kernel<<<blocks, threads>>>
                                  ( reinterpret_cast<cuDoubleComplex*>(H), 
                                    reinterpret_cast<const cuDoubleComplex*>(H0), 
                                    reinterpret_cast<const cuDoubleComplex*>(x),
                                    reinterpret_cast<const cuDoubleComplex*>(y), 
                                    reinterpret_cast<const cuDoubleComplex*>(z),
                                    las0,
                                    las1,
                                    las2,
                                    N );

}


