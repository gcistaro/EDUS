template<class T>
Matrix<T>::Matrix(const int& nrows, const int& ncols)
{
    this->initialize(nrows, ncols);
}

template<class T>
void Matrix<T>::initialize(const int& nrows, const int& ncols)
{
    //(*this).~Matrix<T>();
    assert(nrows > 0 && ncols > 0);
    this->Values.initialize({nrows, ncols});
}


template<class T>
Matrix<T>::Matrix(T* Ptr, const std::array<int,2>& Size_)
{
    Values.initialize(Ptr, Size_);
}

template<class T>
void Matrix<T>::fill(const T& filling_constant)
{
    Values.fill(filling_constant);
}

template<class T>
const T& Matrix<T>::operator()(const int& row, const int& col) const
{
    assert ( row >= 0 && col >= 0 && row < this->get_nrows() && col < this->get_ncols() );
    return this->Values(row, col);//row major
}


template<class T>
T& Matrix<T>::operator()(const int& row, const int& col) 
{
    return const_cast<T&>(static_cast<const Matrix<T>&> (*this)(row, col));//row major
}


template<typename T, typename T_>
void Matrix_gemm(Matrix<T>& OutputMatrix, const T_& alpha, const Matrix<T>& InputMatrix1, const Matrix<T>& InputMatrix2, const T_& beta)
{
    if(InputMatrix1.data() == nullptr || InputMatrix2.data() == nullptr){
        return;
    }
    
    //assert itis possible to do matrix multiplication
    assert( InputMatrix1.get_ncols() == InputMatrix2.get_nrows() );
    
    //initialize C if needed
    if( OutputMatrix.get_nrows() != InputMatrix1.get_nrows() ||
        OutputMatrix.get_ncols() != InputMatrix2.get_ncols())
    {
        OutputMatrix = Matrix<T>( InputMatrix1.get_nrows(), InputMatrix2.get_ncols() );
    }
    //get a notation that is consistent with cblas documentation
    auto m = InputMatrix1.get_nrows();
    auto n = InputMatrix2.get_ncols();
    auto k = InputMatrix1.get_ncols();
    //////////////////////////////////////////////////////
    gemm(m, n, k, alpha, &(InputMatrix1(0,0)), k, &(InputMatrix2(0,0)), n, beta, &(OutputMatrix(0,0)), n);
}




template<typename T>
T Matrix<T>::determinant() const
{
    //in LU decomposition, we can get advantage of the fact tha the determinant of a trinagular matrix 
    //is the product of the diagonal elements, which also reprensent the eigenvalues.
    assert((*this).get_nrows() == (*this).get_ncols());
    Matrix<T> LU;
    lapack_int* ipiv;
    this->LUdecompose(LU, &ipiv);

    T determinant = 1.;
    
    for(int i=0; i < (*this).get_nrows(); ++i){
        determinant *= LU(i,i);
    }
    delete[] ipiv;
    return determinant;
}


template<class T>
Matrix<T> Matrix<T>::transpose() const
{
    Matrix<T> transposeM(this->get_ncols(), this->get_nrows());
    for(int irow=0; irow<this->get_ncols(); irow++){
        for(int icol=0; icol<this->get_nrows(); icol++){
            transposeM(irow, icol) = (*this)(icol, irow);
        }
    }
    return transposeM;
}


template<class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& B) const
{
    Matrix<T> C;
    auto& A = static_cast<const Matrix<T>&>(*this);
    Matrix_gemm(C, 1., A, B, 0.);
    return C;
}

template<class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& B) const
{
    assert( this->get_nrows() == B.get_nrows() && this->get_ncols() == B.get_ncols() );
    Matrix<T> C(this->get_nrows(), this->get_ncols());
    for(int irow=0; irow<C.get_nrows(); ++irow){
        for(int icol=0; icol<C.get_ncols(); ++icol){
            C(irow, icol) = (*this)(irow, icol) + B(irow, icol); 
        }
    }
    return C;
}

template<class T>
Matrix<T> Matrix<T>::operator-() const
{
    Matrix<T> C(this->get_nrows(), this->get_ncols());
    for(int irow=0; irow<C.get_nrows(); ++irow){
        for(int icol=0; icol<C.get_ncols(); ++icol){
            C(irow, icol) = -(*this)(irow, icol);
        }
    }
    return C;
}

template<class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& B) const
{    
    return (*this) + (-B);
}

template<class T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& B) 
{   
    for(int irow=0; irow<this->get_nrows(); ++irow){
        for(int icol=0; icol<this->get_ncols(); ++icol){
            (*this)(irow, icol) += B(irow, icol);
        }
    }
    return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& B) 
{   
    return ( *this += (-B) );
}




template<class T>
Matrix<T> Matrix<T>::operator*(T Scalar) const
{
    Matrix<T> C(this->get_nrows(), this->get_ncols());
    for(int irow=0; irow<this->get_nrows(); irow++){
        for(int icol=0; icol<this->get_ncols(); icol++){
            C(irow,icol) = Scalar*(*this)(irow,icol);
        }
    }
    return C;
}

template<class T, class U>
Matrix<T> operator*(U Scalar, const Matrix<T> A)
{
    return A*T(Scalar);
}

template<class T>
Vector<T> Matrix<T>::operator*(const Vector<T>& v) const
{
    assert(v.get_NumberOfElements() == (*this).get_ncols());
    Vector<T> result(this->get_nrows());
    int m = this->get_nrows();
    int k = this->get_ncols();
    int n = 1;

    gemm(m, n, k, 1., &(*this)(0,0), k, &v(0), n, 0., &result(0), n);
    return result;

}




template<class T>
inline int Matrix<T>::get_nrows() const
{
    return this->Values.get_Size(0);
}


template<class T>
inline int Matrix<T>::get_ncols() const
{
    return this->Values.get_Size(1);
}

template<class T>
inline int Matrix<T>::get_TotalSize() const
{
    return this->Values.get_TotalSize();
}

template<typename T>
inline double Matrix<T>::norm() const
{
    double Norm=0;
    for(auto& val : (*this)){
        Norm += abs(val)*abs(val);
    }
    return Norm;
}

//overloading writing matrix
/*template<class T>
std::ostream& operator<<(std::ostream& os, Matrix<T> m)
{
    for(int irow=0; irow<m.get_nrows(); irow++){
        for(int icol=0; icol<m.get_ncols(); icol++){
            os << std::setprecision(5) << std::setw(10) << m(irow,icol); 
        }
        os << std::endl;
    }
    return os;
}*/


template<class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& m)
{
    for(int irow=0; irow<m.get_nrows(); irow++){
        for(int icol=0; icol<m.get_ncols(); icol++){
            os << std::setprecision(14) << std::setw(20) << m(irow,icol);
        }
        os << std::endl;
    }
    return os;
}

template<typename T>
auto max(const Matrix<T>& m)
{
    //double max = 0;
//    for(int iblock=0; iblock<m.get_nblocks(); ++iblock){
//        for(int irow=0; irow<m.get_nrows(); ++irow){
//            for(int icol=0; icol<m.get_ncols(); ++icol){
//                //HR(ci(iblock, 0), irow, icol) = H0R(iblock, irow, icol);
//                if(std::abs(m(iblock, irow, icol)) > max){
//                    max =std::abs( m(iblock, irow, icol));
//                }
//            }
//        }
//    }            
    return *std::max_element(m.begin(), m.end(), [&](const T& a1, const T& a2){ return std::abs(a1) < std::abs(a2);});
}

template<typename T, typename U> 
Matrix<T> diag(const U& diag_entries__)
{
    int n = diag_entries__.get_Size()[0];
    Matrix<T> D(n, n);
    D.fill(0.);
    for( int i=0; i<n; i++ ) {
        D(i,i) = diag_entries__(i);
    }
    return D;
}
