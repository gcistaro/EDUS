#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <complex>
#include <memory>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <type_traits>

#ifdef EDUS_MKL
    #include "mkl.h"
#else 
    #include "lapacke.h"
#endif
#include "mdContainers/mdContainers.hpp"
#include "LinearAlgebra/gemm.hpp"
#include "core/profiler.hpp"

template<typename T>
class Vector;


template<class T>
class Matrix{
    private:
        mdarray<T,2> Values;
    public:
        Matrix() = default;
        Matrix(const mdarray<T,2>& Values_) : Values(Values_){};
        Matrix(mdarray<T,2>&& Values_) 
        {
                Values=std::move(Values_);
        };
        Matrix(const int& nrows, const int& ncols);
        void initialize(const int& nrows, const int& ncols);

        Matrix(T* Ptr, const std::array<int,2>& dims);
        
        Matrix(const Matrix& A) {this->Values = A.Values;};
        Matrix& operator=(const Matrix& m) {this->Values = m.Values; return *this;};
        
        Matrix(Matrix&& A) = default;
        Matrix& operator=(Matrix&& m) = default;

        const T& operator()(const int& n, const int& m) const;
        T& operator()(const int& n, const int& m);

        void fill(const T& filling_constant);
        Matrix<T> operator*(const Matrix<T>& B) const;
        Matrix<T> operator*(T Scalar) const;

        Vector<T> operator*(const Vector<T>& v) const;
        
        Matrix<T> operator+(const Matrix<T>& B) const;
        Matrix<T> operator-() const;
        Matrix<T> operator-(const Matrix<T>& B) const;
        Matrix<T>& operator+=(const Matrix<T>& B);
        Matrix<T>& operator-=(const Matrix<T>& B);

        template<typename U, typename V>
        friend Matrix<U> operator*(V Scalar, const Matrix<U> M);

        Matrix<T> inverse() const;
        Matrix<T> pseudoinv();
        Matrix<T> transpose() const;
        void LUdecompose(Matrix<T>& LU, lapack_int** pointer_to_ipiv) const;

        void diagonalize(Matrix<std::complex<double>>& Eigenvectors, mdarray<double,1>& EigenValues) const;
        void svd(Matrix<T>& u, Matrix<T>& vt, mdarray<T,1>& s);
        double norm() const;
	//Matrix<T> LUdecompose() const;
	T determinant() const;
        const T* data() const {return Values.data();};
        T* data() {return const_cast<T*>((static_cast<const Matrix<T>&>(*this)).Values.data());};
        //friend void multiply(Matrix<T>& OutputMatrix, const auto& Scalar1, const Matrix<T>& Matrix1, 
        //                                              const auto& Scalar2, const Matrix<T>& Matrix2);
        
        auto begin() const { return Values.begin(); }
        auto end() const {return Values.end(); } 	
        int get_nrows() const;
        int get_ncols() const;
        int get_TotalSize() const;
        void orthogonalize();
        bool is_hermitian() const; 
        
        friend class Vector<T>;
};

template<>
void Matrix<std::complex<double>>::diagonalize(Matrix<std::complex<double>>& EigenVectors, mdarray<double,1>& EigenValues) const;
template<>
Matrix<double> Matrix<double>::inverse() const;

template<typename T, typename T_>
void Matrix_gemm(Matrix<T>& OutputMatrix, const T_& alpha, const Matrix<T>& InputMatrix1, const Matrix<T>& InputMatrix2, const T_& beta);

        
//overloading writing matrix
template<class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& m);

template<class T>
auto max(const Matrix<T>& M);

template<typename T, typename U> 
Matrix<T> diag(const U& diag_entries__);

#include "Matrix_definitions.hpp"



#endif
