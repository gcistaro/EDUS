template<typename T>
Matrix<T> BlockMatrix<T>::EmptyMatrix = Matrix<T>();

template<typename T>
void BlockMatrix<T>::initialize(const Space& space, mdarray<T,3>& Values_)//, MeshGrid& meshgrid_)
{
    this->initialize(space, Values_.get_Size()[0], Values_.get_Size()[1], Values_.get_Size()[2]);
    std::copy(Values_.begin(), Values_.end(), Values.begin());
    //Values = std::move(Values_);
    //meshgrid = std::make_shared<MeshGrid>(meshgrid_);
}

template<typename T>
void BlockMatrix<T>::initialize_submatrix()
{
        submatrix.resize(Values.get_Size(0));

        for(auto iblock=0; iblock<Values.get_Size(0); iblock++){
            submatrix[iblock] =Matrix<T>(&(Values(iblock,0,0)),{Values.get_Size(1), Values.get_Size(2)});    
        }
}


template<typename T>
void BlockMatrix<T>::initialize(const Space& space__, const int& nblocks, const int& nrows, const int& ncols, const int& real_dims__)
{
    Values.initialize({nblocks, nrows, ncols}, real_dims__);
    space = space__;
    initialize_submatrix();
}

template<typename T>
void BlockMatrix<T>::test_submatrix()
{
    for(int ib=0; ib<this->get_nblocks(); ib++){
        assert( (*this)[ib].get_nrows() == this->get_nrows() );
        assert( (*this)[ib].get_ncols() == this->get_ncols() );

        for(int ir=0; ir<this->get_nrows(); ir++){
            for(int ic=0; ic<this->get_ncols(); ic++){
                assert( &((*this)(ib,ir,ic)) == &(this->submatrix[ib](ir,ic)) );
            }
        }
    }
}


template<typename T>
BlockMatrix<T>::BlockMatrix(const BlockMatrix<T>& A)
{
    *this = A;
}

template<typename T>
inline const T& BlockMatrix<T>::operator()(const int& iblock, const int& n, const int& m) const
{
    if(iblock == -1){
        return NullValue;
    }
    return this->Values(iblock, n, m);
}

template<typename T>
inline T& BlockMatrix<T>::operator()(const int& iblock, const int& n, const int& m)
{
    return (const_cast<T&>(const_cast<BlockMatrix<T> const&>(*this)(iblock, n, m)));
}

template<typename T>
inline T& BlockMatrix<T>::operator()(const int& i)
{
    return this->Values[i];
}

template<typename T>
inline const T& BlockMatrix<T>::operator()(const int& i) const
{
    return (const_cast<T&>(const_cast<BlockMatrix<T> const&>(*this)(i)));
}

template<typename T>
BlockMatrix<T>& BlockMatrix<T>::operator=(const BlockMatrix<T>& A)
{
    Values = A.Values;
    space = A.space;
    initialize_submatrix();
    meshgrid = A.meshgrid;
    return *this;
}

template<typename T>
BlockMatrix<T>::BlockMatrix(BlockMatrix<T>&& A)
{
    Values = std::move(A.Values);
    space = std::move(A.space);
    initialize_submatrix();
    meshgrid = std::move(A.meshgrid);
}


template<typename T>
BlockMatrix<T>& BlockMatrix<T>::operator=(BlockMatrix<T>&& A)
{
    Values = std::move(A.Values);
    space = std::move(A.space);
    initialize_submatrix();
    meshgrid = std::move(A.meshgrid);
    return *this;
}

template<typename T>
void BlockMatrix<T>::fill(const T& Scalar)
{
#ifdef EDUS_TIMERS
    PROFILE("BlockMatrix::fill");
#endif
    Values.fill(Scalar);
// ==     std::fill(this->Values.begin(), this->Values.end(), Scalar);
}

template<class T>
void copy(const BlockMatrix<T>& ToCopy__, BlockMatrix<T>& ToBeCopied__, const Processor& proc__)
{
    copy(ToCopy__.Values, ToBeCopied__.Values, proc__);
}


template<typename T>
Matrix<T>& BlockMatrix<T>::operator[](const int& iblock)
{
    return const_cast<Matrix<T>&>(const_cast<const BlockMatrix<T>&>(*this)[iblock]);
}

template<typename T>
const Matrix<T>& BlockMatrix<T>::operator[](const int& iblock) const
{
    if(iblock < 0){
        return EmptyMatrix;
    }
    return submatrix[iblock];
}

template<typename T>
Matrix<T>& BlockMatrix<T>::operator[](const Coordinate& Point)
{
    return const_cast<Matrix<T>&>(const_cast<const BlockMatrix<T>&>(*this)[Point]);
}

template<typename T>
const Matrix<T>& BlockMatrix<T>::operator[](const Coordinate& Point) const
{
    int iblock = (*meshgrid).find(Point);
    return (*this)[iblock];
}


template<typename T>
void multiply(BlockMatrix<T>& Output, T Scalar, const BlockMatrix<T>& Input1, const BlockMatrix<T>& Input2, const Processor& proc__ )
{
    multiply(Output, Scalar, Input1, Input2, T(0.), proc__);
}

template<typename T>
void multiply(BlockMatrix<T>& Output, T Scalar, const BlockMatrix<T>& Input1, const BlockMatrix<T>& Input2, T Scalar2, const Processor& proc__ )
{
    assert(Output.get_nblocks() == Input1.get_nblocks() && Input1.get_nblocks() == Input2.get_nblocks());
#ifdef EDUS_GPU
    if(proc__ == device) {
        const_cast<BlockMatrix<std::complex<double>>&>(Input1).transfer_to(device);
        const_cast<BlockMatrix<std::complex<double>>&>(Input2).transfer_to(device);
        static auto stride = Output.get_nblocks();
        static auto m = Output.get_nrows();
        static auto n = Output.get_ncols();
        static auto k = Input1.get_ncols();
         cublasZgemmStridedBatched(
             cublas_handle,
             CUBLAS_OP_N, CUBLAS_OP_N, 
             n, m, k,
             reinterpret_cast<const cuDoubleComplex*>(&Scalar),             
             reinterpret_cast<const cuDoubleComplex*>(Input2.data(device)),
             n,                 
             k * n,             
             reinterpret_cast<const cuDoubleComplex*>(Input1.data(device)),
             k,                    
             m * k,                
             reinterpret_cast<const cuDoubleComplex*>(&Scalar2),
             reinterpret_cast<cuDoubleComplex*>(Output.data(device)),
             n,                    
             m * n,                
             stride);
        Output.transfer_to(host);
///== debug        std::cout <<"multiply max GPU: "<< *max(Output) << std::endl;
        Output.set_processor(device);
        return;
    }
#endif
#ifdef EDUS_BATCHGEMM
    static auto stride = Output.get_nblocks();
    static auto m = Output.get_nrows();
    static auto n = Output.get_ncols();
    static auto k = Input1.get_ncols();
    assert( k == Input1.get_ncols() );
    assert( n == Input2.get_ncols() );
    assert( m == Input1.get_nrows() );
    cblas_zgemm_batch_strided( CblasRowMajor, CblasNoTrans, CblasNoTrans,
    m, n, k, &Scalar, &Input1(0,0,0), k, m*k, &Input2(0,0,0), n, k*n, &Scalar2,
    &Output(0,0,0), n, m*n, stride);
#else 
    #pragma omp parallel for schedule(static)
    for(int iblock=0; iblock<Output.get_nblocks(); iblock++){
        Matrix_gemm(Output[iblock], Scalar, Input1[iblock], Input2[iblock], Scalar2);
    }
///== debug    std::cout <<"multiply max: "<< *max(Output) << std::endl;
#endif
}


template<typename T, typename U>
void convolution(BlockMatrix<T>& Output, U Scalar, const BlockMatrix<T>& Input1, const BlockMatrix<T>& Input2 )
{
    assert(Output.get_nblocks()!=0);

    //Output.fill(0.);
    auto& ci = MeshGrid::ConvolutionIndex[{Output.meshgrid->get_id(), Input1.meshgrid->get_id(), Input2.meshgrid->get_id()}];

    if(ci.get_Size(0) == 0 ){
        MeshGrid::Calculate_ConvolutionIndex(*(Output.meshgrid), *(Input1.meshgrid), *(Input2.meshgrid));

    }
    PROFILE_START("BlockMatrix::convolution");

    #pragma omp parallel for schedule(static)
    for(int iblock_o=0; iblock_o<Output.get_nblocks(); iblock_o++){
        for(int iblock_i2=0; iblock_i2<Input2.get_nblocks(); iblock_i2++){
            auto& iblock_i1 = ci(iblock_o, iblock_i2);
            Matrix_gemm(Output[iblock_o], Scalar+im*0., Input1[iblock_i1], Input2[iblock_i2], 1.+im*0.);
        }
    }
    PROFILE_STOP("BlockMatrix::convolution");
}

template<typename T_, typename U>
void commutator(BlockMatrix<T_>& Output, U Scalar, const BlockMatrix<T_>& Input1, const BlockMatrix<T_>& Input2, const bool& Erase_Output, const Processor& proc__)
{
#ifdef EDUS_TIMERS
    PROFILE("Commutator");
#endif
    assert( Output.get_space() == Input1.get_space() );
    assert( Input2.get_space() == Input1.get_space() );

    switch( Output.get_space() ){
        case(R):
        {
            convolution(Output, Scalar, Input1, Input2);
            convolution(Output, -Scalar, Input2, Input1);
            break;
        }
        case(k):
        {
            multiply(Output, Scalar, Input1, Input2, double(!Erase_Output) + im*0., proc__);
            multiply(Output, -Scalar, Input2, Input1, 1.+im*0., proc__);
            break;
        }
    }
}


template<typename T>
void BlockMatrix<T>::diagonalize(std::vector<mdarray<double,1>>& Eigenvalues,
                 BlockMatrix<std::complex<double>>& Eigenvectors) const
{
    //assert(space_ == k);
    Eigenvectors.initialize(k, this->get_nblocks(), this->get_nrows(), this->get_ncols());
    Eigenvalues.resize(this->get_nblocks());
    #pragma omp parallel for
    for(int ik=0; ik<this->get_nblocks(); ik++){
        Eigenvalues[ik].initialize({this->get_nrows()});
        ((*this)[ik]).diagonalize(Eigenvectors[ik], Eigenvalues[ik]);
    }
}

template<typename T>
bool BlockMatrix<T>::is_hermitian()
{
    assert(space == k);
    bool hermitian = true;
    for(int ik=0; ik<this->get_nblocks(); ++ik) {
        for( int irow=0; irow<this->get_nrows(); irow++ ) {
            for(int icol=irow; icol < this->get_nrows(); ++icol ) {
                hermitian = hermitian && ( std::abs( (*this)[ik](irow, icol) - std::conj((*this)[ik](icol, irow)) ) < 1.e-15) ;
            }
        }
    }
    return hermitian;
}

template<typename T>
void BlockMatrix<T>::make_hermitian()
{
    PROFILE("make_hermitian");
    switch(space) 
    {
        case(Space::k) :
        {
            #pragma omp parallel for
            for(int ik=0; ik<this->get_nblocks(); ++ik) {
                for( int irow=0; irow<this->get_nrows(); irow++ ) {
                    for(int icol=irow; icol < this->get_nrows(); ++icol ) {
                        auto term = ((*this)(ik,irow,icol) + std::conj((*this)(ik,icol,irow)))/2.;
                        (*this)(ik,irow,icol) = term;
                        (*this)(ik,icol,irow) = std::conj(term);
                    }
                }
            }
            break;
        }
        case(Space::R) :
        {
            throw std::runtime_error("make_hermitian from R not implemented");
            break;
        }
    }
}

template<typename T>
void BlockMatrix<T>::make_dagger()
{
    PROFILE("make_dagger");
    switch(space) 
    {
        case(Space::k) :
        {
            #pragma omp parallel for
            for(int ik=0; ik<this->get_nblocks(); ++ik) {
                for( int irow=0; irow<this->get_nrows(); irow++ ) {
                    for(int icol=0; icol < this->get_ncols(); ++icol ) {
                        auto term = (*this)(ik,icol,irow);
                        (*this)(ik,icol,irow) = std::conj( (*this)(ik,irow,icol) );
                        (*this)(ik,icol,irow) = std::conj( term );
                    }
                }
            }
            break;
        }
        case(Space::R) :
        {
            throw std::runtime_error("make_dagger not implemented from R space");
        }
    }
}

template<typename T>
void BlockMatrix<T>::make_antihermitian()
{
    #pragma omp parallel for
    for(int ik=0; ik<this->get_nblocks(); ++ik) {
        for( int irow=0; irow<this->get_nrows(); irow++ ) {
            for(int icol=irow; icol < this->get_nrows(); ++icol ) {
                auto term = ((*this)(ik,irow,icol) - std::conj((*this)(ik,icol,irow)))/2.;
                (*this)(ik,irow,icol) = term;
                (*this)(ik,icol,irow) = -std::conj(term);
            }
        }
    }

}


template<typename T>
void BlockMatrix<T>::cut(const double& threshold__)
{
    #pragma omp parallel for
    for(int ik=0; ik<this->get_nblocks(); ++ik) {
        for( int irow=0; irow<this->get_nrows(); irow++ ) {
            for(int icol=irow; icol < this->get_nrows(); ++icol ) {
                (*this)(ik,irow,icol) = ( std::abs( (*this)(ik,irow,icol) ) > threshold ? (*this)(ik,irow,icol) : 0.);
            }
        }
    }
}


template<typename T>
void BlockMatrix<T>::write_h5(const std::string& name__, const std::string& node__, const std::string& label__)
{
#ifdef EDUS_HDF5
    PROFILE("write_h5");

    HDF5_tree fout(name__, hdf5_access_t::read_write);
    fout[node__].create_node(label__);
#ifdef EDUS_HDF5PARALLEL    
    fout[node__][label__].create_node(kpool_comm->rank());
    fout[node__][label__][kpool_comm->rank()].write("local", 
    reinterpret_cast<double*>(this->data()), (this->get_TotalSize() * 2) );
    mpi::Communicator::world().barrier();
#else
#ifdef EDUS_MPI
    MPI_Request req;
    static BlockMatrix<T> aux_(this->space, this->get_nblocks(), this->get_nrows(), this->get_ncols());
    kpool_comm->isend(&((*this)(0,0,0)), 0, this->get_TotalSize(),req);
    if( kpool_comm->rank() == 0 ) {
        for(int ik_rank = 0; ik_rank < kpool_comm->size(); ++ik_rank) {
            kpool_comm->receive(&(aux_(0,0,0)), ik_rank, this->get_TotalSize());
            fout[node__][label__].create_node(ik_rank);
            fout[node__][label__][ik_rank].write("local", 
                reinterpret_cast<double*>(aux_.data()), (this->get_TotalSize() * 2) );
        }
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
#else //EDUS_MPI
    auto& aux_ = *this;
    fout[node__][label__].create_node(0);
    fout[node__][label__][0].write("local", 
        reinterpret_cast<double*>(aux_.data()), (this->get_TotalSize() * 2) );
#endif
#endif //EDUS_HDF5PARALLEL
#endif //EDUS_HDF5
}

template<typename T>
void BlockMatrix<T>::load(const std::string& name__, const int& node__, const std::string& label__)
{
#ifdef EDUS_HDF5
    PROFILE("load");

    HDF5_tree fout(name__, hdf5_access_t::read_write);

    MPI_Request req;
    static BlockMatrix<T> aux_(this->space, this->get_nblocks(), this->get_nrows(), this->get_ncols());
#ifdef EDUS_MPI
    if( kpool_comm->rank() == 0 ) {
        for(int ik_rank = 0; ik_rank < kpool_comm->size(); ++ik_rank) {
            fout[node__][label__][ik_rank].read("local", 
                reinterpret_cast<double*>(aux_.data()), (this->get_TotalSize() * 2) );
            kpool_comm->isend(&(aux_(0,0,0)), ik_rank, this->get_TotalSize(), req);
        }
    }
    kpool_comm->receive(&((*this)(0,0,0)), 0, this->get_TotalSize());
    MPI_Wait(&req, MPI_STATUS_IGNORE);
#else
    auto& aux = *this;
    fout[node__][label__][0].read("local", 
        reinterpret_cast<double*>(aux_.data()), (this->get_TotalSize() * 2) );

#endif 
#endif //EDUS_HDF5
}

template<typename T>
void BlockMatrix<T>::initialize_device()
{
    Values.initialize_device();
}

template<typename T>
void BlockMatrix<T>::transfer_to(const Processor& proc__)
{
    Values.transfer_to(proc__);
}

template<typename T>
auto max(const BlockMatrix<T>& m)
{
    return std::max_element(m.begin(), m.end(), [&](const T& a1, const T& a2){ return std::abs(a1) < std::abs(a2);});
}

template<class T>
std::ostream& operator<<(std::ostream& os, const BlockMatrix<T>& m)
{
    for(int i=0; i<m.get_nblocks(); i++){
        os << i << " " << (*((const_cast<BlockMatrix<T>&>(m)).get_MeshGrid()))[i].get(LatticeVectors(m.get_space())) << m[i] << std::endl;
    }
    return os;
}

template<class T>
template<typename U>
void BlockMatrix<T>::Divide(const U& Divisors__)
{
    #pragma omp parallel for
    for( int iblock=0; iblock<this->get_nblocks(); iblock++ ) {
        for( int irow = 0; irow < this->get_nrows(); irow++ ) {
            for( int icol = 0; icol < this->get_ncols(); icol++ ) {
                (*this)(iblock,irow,icol) /= Divisors__[iblock];
            }
        }
    }
}
