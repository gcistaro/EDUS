template<typename T>
Matrix<T> BlockMatrix<T>::empty_matrix_ = Matrix<T>();

template<typename T>
void BlockMatrix<T>::initialize(const Space& space__, mdarray<T,4>& values__)//, MeshGrid& meshgrid_)
{
    this->initialize(space__, values__.get_Size()[0], 
                            values__.get_Size()[2],
                            values__.get_Size()[3], 
                            0, 
                            values__.get_Size()[1] 
    );
    std::copy(values__.begin(), values__.end(), values_.begin());
    //values_ = std::move(Values_);
    //meshgrid_ = std::make_shared<MeshGrid>(meshgrid_);
}

template<typename T>
void BlockMatrix<T>::initialize(const Space& space__, mdarray<T,3>& values__)
{
    this->initialize(space__, values__.get_Size(0), values__.get_Size(1), values__.get_Size(2), 0, 1);
    std::copy(values__.begin(), values__.end(), values_.begin());
}

template<typename T>
void BlockMatrix<T>::initialize_submatrix()
{
        submatrix_.resize(get_nblocks());

        for(int ik=0; ik<get_nk(); ik++){
            for(int iq=0; iq<get_nq(); iq++){
                submatrix_[ik*get_nq()+iq] = Matrix<T>(&(values_(ik,iq,0,0)),{get_nrows(), get_ncols()});
            }
        }
}


template<typename T>
void BlockMatrix<T>::initialize(const Space& space__, int nk__, 
                                                      int nrows__, 
                                                      int ncols__, 
                                                      int real_dims__,
                                                      int nq__)
{
    values_.initialize({nk__, nq__, nrows__, ncols__}, real_dims__);
    space_ = space__;
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
                assert( &((*this)(ib,ir,ic)) == &(this->submatrix_[ib](ir,ic)) );
            }
        }
    }
}


template<typename T>
BlockMatrix<T>::BlockMatrix(const BlockMatrix<T>& a__)
{
    *this = a__;
}

template<typename T>
inline const T& BlockMatrix<T>::operator()(const int& iblock__, const int& n__, const int& m__) const
{
    if(iblock__ == -1){
        return null_value_;
    }
    /* flat block index iblock = ik*nq + iq */
    return this->values_.data()[(iblock__*get_nrows() + n__)*get_ncols() + m__];
}

template<typename T>
inline T& BlockMatrix<T>::operator()(const int& iblock__, const int& n__, const int& m__)
{
    return (const_cast<T&>(const_cast<BlockMatrix<T> const&>(*this)(iblock__, n__, m__)));
}

template<typename T>
inline const T& BlockMatrix<T>::operator()(const int& ik__, const int& iq__, const int& n__, const int& m__) const
{
    return this->values_(ik__, iq__, n__, m__);
}

template<typename T>
inline T& BlockMatrix<T>::operator()(const int& ik__, const int& iq__, const int& n__, const int& m__)
{
    return (const_cast<T&>(const_cast<BlockMatrix<T> const&>(*this)(ik__, iq__, n__, m__)));
}

template<typename T>
inline T& BlockMatrix<T>::operator()(const int& i__)
{
    return this->values_[i__];
}

template<typename T>
inline const T& BlockMatrix<T>::operator()(const int& i__) const
{
    return (const_cast<T&>(const_cast<BlockMatrix<T> const&>(*this)(i__)));
}

template<typename T>
BlockMatrix<T>& BlockMatrix<T>::operator=(const BlockMatrix<T>& a__)
{
    values_ = a__.values_;
    space_ = a__.space_;
    initialize_submatrix();
    meshgrid_ = a__.meshgrid_;
    return *this;
}

template<typename T>
BlockMatrix<T>::BlockMatrix(BlockMatrix<T>&& a__)
{
    values_ = std::move(a__.values_);
    space_ = std::move(a__.space_);
    initialize_submatrix();
    meshgrid_ = std::move(a__.meshgrid_);
}


template<typename T>
BlockMatrix<T>& BlockMatrix<T>::operator=(BlockMatrix<T>&& a__)
{
    values_ = std::move(a__.values_);
    space_ = std::move(a__.space_);
    initialize_submatrix();
    meshgrid_ = std::move(a__.meshgrid_);
    return *this;
}

template<typename T>
void BlockMatrix<T>::fill(const T& scalar__)
{
#ifdef EDUS_TIMERS
    PROFILE("BlockMatrix::fill");
#endif
    values_.fill(scalar__);
// ==     std::fill(this->values_.begin(), this->values_.end(), Scalar);
}

template<class T>
void copy(const BlockMatrix<T>& to_copy__, BlockMatrix<T>& to_be_copied__, const Processor& proc__)
{
    copy(to_copy__.values_, to_be_copied__.values_, proc__);
}


template<typename T>
Matrix<T>& BlockMatrix<T>::operator[](const int& iblock__)
{
    return const_cast<Matrix<T>&>(const_cast<const BlockMatrix<T>&>(*this)[iblock__]);
}

template<typename T>
const Matrix<T>& BlockMatrix<T>::operator[](const int& iblock__) const
{
    if(iblock__ < 0){
        return empty_matrix_;
    }
    return submatrix_[iblock__];
}

template<typename T>
Matrix<T>& BlockMatrix<T>::operator[](const Coordinate& point__)
{
    return const_cast<Matrix<T>&>(const_cast<const BlockMatrix<T>&>(*this)[point__]);
}

template<typename T>
const Matrix<T>& BlockMatrix<T>::operator[](const Coordinate& point__) const
{
    if( this->get_nq() > 1 ) {
        throw std::runtime_error("BlockMatrix::operator[](Coordinate): not implemented for nq > 1");
    }
    int iblock = (*meshgrid_).find(point__);
    return (*this)[iblock];
}


template<typename T>
void multiply(BlockMatrix<T>& output__, T scalar__, const BlockMatrix<T>& input1__, const BlockMatrix<T>& input2__, const Processor& proc__ )
{
    multiply(output__, scalar__, input1__, input2__, T(0.), proc__);
}

template<typename T>
void multiply(BlockMatrix<T>& output__, T scalar__, const BlockMatrix<T>& input1__, const BlockMatrix<T>& input2__, T scalar2__, const Processor& proc__ )
{
    assert(output__.get_nblocks() == input1__.get_nblocks() && input1__.get_nblocks() == input2__.get_nblocks());
#ifdef EDUS_GPU
    if(proc__ == device) {
        const_cast<BlockMatrix<std::complex<double>>&>(input1__).transfer_to(device);
        const_cast<BlockMatrix<std::complex<double>>&>(input2__).transfer_to(device);
        auto stride = output__.get_nblocks();
        auto m = output__.get_nrows();
        auto n = output__.get_ncols();
        auto k = input1__.get_ncols();
         cublasZgemmStridedBatched(
             cublas_handle,
             CUBLAS_OP_N, CUBLAS_OP_N, 
             n, m, k,
             reinterpret_cast<const cuDoubleComplex*>(&scalar__),             
             reinterpret_cast<const cuDoubleComplex*>(input2__.data(device)),
             n,                 
             k * n,             
             reinterpret_cast<const cuDoubleComplex*>(input1__.data(device)),
             k,                    
             m * k,                
             reinterpret_cast<const cuDoubleComplex*>(&scalar2__),
             reinterpret_cast<cuDoubleComplex*>(output__.data(device)),
             n,                    
             m * n,                
             stride);
        output__.transfer_to(host);
///== debug        std::cout <<"multiply max GPU: "<< *max(Output) << std::endl;
        output__.set_processor(device);
        return;
    }
#endif
#ifdef EDUS_BATCHGEMM
    auto stride = output__.get_nblocks();
    auto m = output__.get_nrows();
    auto n = output__.get_ncols();
    auto k = input1__.get_ncols();
    assert( k == input1__.get_ncols() );
    assert( n == input2__.get_ncols() );
    assert( m == input1__.get_nrows() );
    cblas_zgemm_batch_strided( CblasRowMajor, CblasNoTrans, CblasNoTrans,
    m, n, k, &scalar__, &input1__(0,0,0), k, m*k, &input2__(0,0,0), n, k*n, &scalar2__,
    &output__(0,0,0), n, m*n, stride);
#else 
    #pragma omp parallel for schedule(static)
    for(int iblock=0; iblock<output__.get_nblocks(); iblock++){
        Matrix_gemm(output__[iblock], scalar__, input1__[iblock], input2__[iblock], scalar2__);
    }
///== debug    std::cout <<"multiply max: "<< *max(Output) << std::endl;
#endif
}


template<typename T, typename U>
void convolution(BlockMatrix<T>& output__, U scalar__, const BlockMatrix<T>& input1__, const BlockMatrix<T>& input2__ )
{
    if( output__.get_nq() > 1 ) {
        throw std::runtime_error("BlockMatrix::convolution: not implemented for nq > 1");
    }
    assert(output__.get_nblocks()!=0);

    //Output.fill(0.);
    auto& ci = MeshGrid::ConvolutionIndex[{output__.meshgrid_->get_id(), input1__.meshgrid_->get_id(), input2__.meshgrid_->get_id()}];

    if(ci.get_Size(0) == 0 ){
        MeshGrid::Calculate_ConvolutionIndex(*(output__.meshgrid_), *(input1__.meshgrid_), *(input2__.meshgrid_));

    }
    PROFILE_START("BlockMatrix::convolution");

    #pragma omp parallel for schedule(static)
    for(int iblock_o=0; iblock_o<output__.get_nblocks(); iblock_o++){
        for(int iblock_i2=0; iblock_i2<input2__.get_nblocks(); iblock_i2++){
            auto& iblock_i1 = ci(iblock_o, iblock_i2);
            Matrix_gemm(output__[iblock_o], scalar__+im*0., input1__[iblock_i1], input2__[iblock_i2], 1.+im*0.);
        }
    }
    PROFILE_STOP("BlockMatrix::convolution");
}

template<typename T_, typename U>
void commutator(BlockMatrix<T_>& output__, U scalar__, const BlockMatrix<T_>& input1__, const BlockMatrix<T_>& input2__, const bool& erase_output__, const Processor& proc__)
{
#ifdef EDUS_TIMERS
    PROFILE("Commutator");
#endif
    assert( output__.get_space() == input1__.get_space() );
    assert( input2__.get_space() == input1__.get_space() );

    switch( output__.get_space() ){
        case(R):
        {
            convolution(output__, scalar__, input1__, input2__);
            convolution(output__, -scalar__, input2__, input1__);
            break;
        }
        case(k):
        {
            multiply(output__, scalar__, input1__, input2__, double(!erase_output__) + im*0., proc__);
            multiply(output__, -scalar__, input2__, input1__, 1.+im*0., proc__);
            break;
        }
    }
}


template<typename T>
void BlockMatrix<T>::diagonalize(std::vector<mdarray<double,1>>& eigenvalues__,
                 BlockMatrix<std::complex<double>>& eigenvectors__) const
{
    if( this->get_nq() > 1 ) {
        throw std::runtime_error("BlockMatrix::diagonalize: not implemented for nq > 1");
    }
    //assert(space_ == k);
    eigenvectors__.initialize(k, this->get_nblocks(), this->get_nrows(), this->get_ncols());
    eigenvalues__.resize(this->get_nblocks());
    #pragma omp parallel for
    for(int ik=0; ik<this->get_nblocks(); ik++){
        eigenvalues__[ik].initialize({this->get_nrows()});
        ((*this)[ik]).diagonalize(eigenvectors__[ik], eigenvalues__[ik]);
    }
}

template<typename T>
bool BlockMatrix<T>::is_hermitian()
{
    if( this->get_nq() > 1 ) {
        throw std::runtime_error("BlockMatrix::is_hermitian: not implemented for nq > 1");
    }
    assert(space_ == k);
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
    if( this->get_nq() > 1 ) {
        throw std::runtime_error("BlockMatrix::make_hermitian: not implemented for nq > 1");
    }
    PROFILE("make_hermitian");
    switch(space_) 
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
    if( this->get_nq() > 1 ) {
        throw std::runtime_error("BlockMatrix::make_dagger: not implemented for nq > 1");
    }
    PROFILE("make_dagger");
    switch(space_) 
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
                (*this)(ik,irow,icol) = ( std::abs( (*this)(ik,irow,icol) ) > threshold__ ? (*this)(ik,irow,icol) : 0.);
            }
        }
    }
}


template<typename T>
void BlockMatrix<T>::write_h5(const std::string& name__, const std::string& node__, const std::string& label__, 
                              const mpi::Communicator& comm__)
{
#ifdef EDUS_HDF5
    PROFILE("write_h5");

    HDF5_tree fout(name__, hdf5_access_t::read_write);
    fout[node__].create_node(label__);
#ifdef EDUS_HDF5PARALLEL    
    fout[node__][label__].create_node(comm__.rank());
    fout[node__][label__][comm__.rank()].write("local", 
    reinterpret_cast<double*>(this->data()), (this->get_TotalSize() * 2) );
    mpi::Communicator::world().barrier();
#else
#ifdef EDUS_MPI
    MPI_Request req;
    BlockMatrix<T> aux_(this->space_, this->get_nk(), this->get_nrows(), this->get_ncols(), 0, this->get_nq());
    comm__.isend(&((*this)(0,0,0)), 0, this->get_TotalSize(),req);
    if( comm__.rank() == 0 ) {
        for(int ik_rank = 0; ik_rank < comm__.size(); ++ik_rank) {
            comm__.receive(&(aux_(0,0,0)), ik_rank, this->get_TotalSize());
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
void BlockMatrix<T>::load(const std::string& name__, const int& node__, const std::string& label__, 
                          const mpi::Communicator& comm__)
{
#ifdef EDUS_HDF5
    PROFILE("load");

    HDF5_tree fout(name__, hdf5_access_t::read_write);

    MPI_Request req;
    BlockMatrix<T> aux_(this->space_, this->get_nk(), this->get_nrows(), this->get_ncols(), 0, this->get_nq());
#ifdef EDUS_MPI
    if( comm__.rank() == 0 ) {
        for(int ik_rank = 0; ik_rank < comm__.size(); ++ik_rank) {
            fout[node__][label__][ik_rank].read("local", 
                reinterpret_cast<double*>(aux_.data()), (this->get_TotalSize() * 2) );
            comm__.isend(&(aux_(0,0,0)), ik_rank, this->get_TotalSize(), req);
        }
    }
    comm__.receive(&((*this)(0,0,0)), 0, this->get_TotalSize());
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
    values_.initialize_device();
}

template<typename T>
void BlockMatrix<T>::transfer_to(const Processor& proc__)
{
    values_.transfer_to(proc__);
}

template<typename T>
auto max(const BlockMatrix<T>& m__)
{
    return std::max_element(m__.begin(), m__.end(), [&](const T& a1, const T& a2){ return std::abs(a1) < std::abs(a2);});
}

template<class T>
std::ostream& operator<<(std::ostream& os__, const BlockMatrix<T>& m__)
{
    for(int i=0; i<m__.get_nblocks(); i++){
        os__ << i << " " << (*((const_cast<BlockMatrix<T>&>(m__)).get_MeshGrid()))[i].get(LatticeVectors(m__.get_space())) << m__[i] << std::endl;
    }
    return os__;
}

template<class T>
template<typename U>
void BlockMatrix<T>::Divide(const U& divisors__)
{
    #pragma omp parallel for
    for( int iblock=0; iblock<this->get_nblocks(); iblock++ ) {
        for( int irow = 0; irow < this->get_nrows(); irow++ ) {
            for( int icol = 0; icol < this->get_ncols(); icol++ ) {
                (*this)(iblock,irow,icol) /= divisors__[iblock];
            }
        }
    }
}
