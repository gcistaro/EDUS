#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>

__global__
void Apply_Peierls_phase_kernel(
    cuDoubleComplex* __restrict__ O,
    const cuDoubleComplex* __restrict__ Peierls_phase,
    int N, 
    int nblocks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 1D index
    if ( i >= N ) return; 
    int n2 = N/nblocks; //nrows*ncols
    int iR = i/n2; 
    int i2 = i%n2;

    O[n2*iR + i2] = cuCmul(Peierls_phase[iR],O[n2*iR+i2]);
}

__global__
void Compute_Peierls_phase_kernel(cuDoubleComplex* __restrict__ Peierls_phase,
                                  double* __restrict__ A0, 
                                  double* __restrict__ A1, 
                                  double* __restrict__ A2,
                                  double* __restrict__ Rvectors, 
                                  int sign,
                                  int nblocks)
{
    int iR = blockIdx.x * blockDim.x + threadIdx.x; // 1D index

    if( iR >= nblocks ) return;

    double dot = (*A0)*Rvectors[3*iR+0] +
                 (*A1)*Rvectors[3*iR+1] + 
                 (*A2)*Rvectors[3*iR+2]; 
                 
    cuDoubleComplex phase; 
    phase.x = cos( dot * double(sign) );
    phase.y = sin( dot * double(sign) );
    Peierls_phase[iR] = phase;
}


void Apply_Peierls_phase_gpu( std::complex<double>* O__, 
                              std::complex<double>* Peierls_phase,    
                              double* A0__, 
                              double* A1__, 
                              double* A2__,
                              double* Rvectors__, 
                              int sign, 
                              int N,
                              int nR
                            )
{
    int threads = 256;
    int blocks  = (nR+threads-1)/threads;
    Compute_Peierls_phase_kernel<<<blocks, threads>>>
                        ( reinterpret_cast<cuDoubleComplex*>(Peierls_phase),
                          A0__, 
                          A1__, 
                          A2__,
                          Rvectors__, 
                          sign,
                          nR);
    blocks = (N+threads-1)/threads;

    Apply_Peierls_phase_kernel<<<threads, blocks>>>
            (reinterpret_cast<cuDoubleComplex*>(O__),
             reinterpret_cast<const cuDoubleComplex*>(Peierls_phase),
             N, 
             nR);
}


