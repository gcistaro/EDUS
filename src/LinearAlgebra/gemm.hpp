#ifndef GEMM_HPP
#define GEMM_HPP

//define mkl_complex16 to avoid incompatibilities
#ifndef MKL_Complex16
    #define MKL_Complex16 std::complex<double>
#endif 

#include <type_traits>
#ifdef EDUS_MKL
    #include "mkl.h"
#else 
    #include <cblas.h>
#endif

template<typename U, typename V>
void gemm(int m, int n1, int k1, U alpha, const double* A, int k2, 
                   const double* B, int n2, V beta, double* C, int n3)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n1, k1, alpha, A, k2, B, n2, beta, C, n3);
}

template<typename U, typename V>
void gemm(int m, int n1, int k1, U alpha, const std::complex<double>* A, int k2, 
                   const std::complex<double>* B, int n2, V beta, std::complex<double>* C, int n3)
{
    void* alpha_ = &alpha;
    void* beta_ = &beta;

    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n1, k1, alpha_, A, k2, B, n2, beta_, C, n3);
}




#endif