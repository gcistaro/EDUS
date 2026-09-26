#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
__global__
void Hartree_interaction_kernel( 
    cuDoubleComplex* SigmaH, 
    const cuDoubleComplex* __restrict__ Hartree, 
    const cuDoubleComplex* __restrict__ DM, 
    const cuDoubleComplex* __restrict__ DM0, 
    int index, 
    int N)
{
    
}
                                    index,
                                    N );
Calculate_TDHamiltonian_kernel(
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



void Hartree_interaction_gpu ( std::complex<double>* SigmaH, 
                               const std::complex<double>* Hartree, 
                               const std::complex<double>* DM, 
                               const std::complex<double>* DM0, 
                               int index, 
                               int N)
{
    int threads = 256;
    int blocks  = (N+threads-1)/threads;
    Hartree_interaction_kernel<<<blocks, threads>>>
                                  ( reinterpret_cast<cuDoubleComplex*>(SigmaH), 
                                    reinterpret_cast<const cuDoubleComplex*>(Hartree), 
                                    reinterpret_cast<const cuDoubleComplex*>(DM),
                                    reinterpret_cast<const cuDoubleComplex*>(DM0), 
                                    index,
                                    N );
}
