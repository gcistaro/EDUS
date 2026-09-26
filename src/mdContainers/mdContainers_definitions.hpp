template<typename T, size_t dim> 
mdarray<T,dim>::mdarray(const mdarray<T,dim>& ToBeCopied)
{
    *this = ToBeCopied;
}

//copy assignment 
template<typename T, size_t dim> 
mdarray<T,dim>& mdarray<T,dim>::operator=(const mdarray<T,dim>& ToBeCopied)
{
    if( this->Size != ToBeCopied.Size || Ptr == nullptr){
        this->Ptr = new T[ToBeCopied.TotalSize];
        this->Size = ToBeCopied.Size;
        this->TotalSize = ToBeCopied.TotalSize;
        this->Offset = ToBeCopied.Offset;        
    }
    std::copy(ToBeCopied.begin(), ToBeCopied.end(), this->begin());
    this->multindex = ToBeCopied.multindex;
    return *this;
}

//move constructor
template<typename T, size_t dim> 
mdarray<T,dim>::mdarray(mdarray<T,dim>&& ToBeMoved)
{
    this->Ptr = std::move(ToBeMoved.Ptr);
    this->multindex = std::move(ToBeMoved.multindex);
    this->Size = std::move(ToBeMoved.Size);
    this->TotalSize = std::move(ToBeMoved.TotalSize);
    if(this->TotalSize>0)    
    this->Offset = std::move(ToBeMoved.Offset);
    ToBeMoved.Ptr=nullptr;
    NotDestruct = ToBeMoved.NotDestruct;
}


//move assignment 
template<typename T, size_t dim> 
mdarray<T,dim>& mdarray<T,dim>::operator=(mdarray<T,dim>&& ToBeMoved)
{
    this->Ptr = std::move(ToBeMoved.Ptr);
    this->multindex = std::move(ToBeMoved.multindex);
    this->Size = std::move(ToBeMoved.Size);
    this->TotalSize = std::move(ToBeMoved.TotalSize);
    if(this->TotalSize>0)    
    this->Offset = std::move(ToBeMoved.Offset);
    ToBeMoved.Ptr=nullptr;
    NotDestruct = ToBeMoved.NotDestruct;
    return *this;
}


template<typename T, size_t dim> 
mdarray<T,dim>::mdarray(const std::array<int,dim>& Size_, const int& real_dims__) 
{
    this->initialize(Size_, real_dims__);

}

template<typename T, size_t dim> 
mdarray<T,dim>::mdarray(T* Ptr_, const std::array<int,dim>& Size_, const int& real_dims__)
{
    this->initialize(Ptr_, Size_, real_dims__);
}

template<typename T, size_t dim> 
void mdarray<T,dim>::TotalSizeAndOffset()
{
    TotalSize = 1;
    for(size_t idim=0; idim<dim; idim++){
        TotalSize *= Size[idim]; 
    }
    Offset[dim-1] = 1;
    for(int idim=int(dim)-2; idim>=0; idim--){
        Offset[idim] = Size[idim+1]*Offset[idim+1];
    }
}

template<typename T, size_t dim> 
void mdarray<T,dim>::initialize(const std::array<int,dim>& Size_, const int& real_dims__)
{
    Size = Size_;
    this->multindex.initialize(Size_);
    TotalSizeAndOffset();
    real_dims = (real_dims__ == 0 ? TotalSize : real_dims__);
    if( real_dims < TotalSize ) {
        std::runtime_error("ERROR in mdarray::initialize. You are trying to initialize dimensions that are lower than the totalsize\n");
    }
    //allocate contiguous memory ot totalsize
    this->Ptr = new T[real_dims];
    std::fill((*this).begin(), (*this).end(), 0.);
}

template<typename T, size_t dim> 
void mdarray<T,dim>::initialize(T* Ptr_, const std::array<int,dim>& Size_, const int& real_dims__)
{
    Ptr = Ptr_; 
    Size = Size_; 
    NotDestruct=true;    //The Ptr_ must be destructed by the other owners. care!
    this->multindex.initialize(Size_); 
    TotalSizeAndOffset();
    real_dims = (real_dims__ == 0 ? TotalSize : real_dims__);
    if( real_dims < TotalSize ) {
        std::runtime_error("ERROR in mdarray::initialize. You are trying to initialize dimensions that are lower than the totalsize\n");
    }
}

template <typename T, size_t dim>
void mdarray<T,dim>::initialize_device()
{
    /* we allocate the memory on gpu, but we the array is still on CPU if not explicitly transferred */
#ifdef EDUS_GPU
    if( !initialized_device ) {
        cudaMalloc((void**)&Ptr_device, TotalSize*sizeof(T));
        processor_ = device; 
        fill(0.);
        processor_ = host;
        initialized_device = true;
    } 
#endif
}

template<typename T, size_t dim> 
void mdarray<T,dim>::initialize_device(T* Ptr_device_)
{
#ifdef EDUS_GPU
    Ptr_device = Ptr_device_;
    initialized_device = true;
#endif
}


template <typename T, size_t dim>
void mdarray<T,dim>::transfer_to(const Processor& proc__)
{
#ifdef __DEBUG
    std::cout << "transfer from " << (processor_==host ? "host" : "device") << " to " << (proc__==host ? "host" : "device")<< std::endl;
#endif
#ifdef EDUS_GPU
    if( !initialized_device ) {
        throw std::runtime_error("Trying to transfer memory but device memory is not allocated!\n");
    }
    /* Transfer the array from host to device or viceversa */
    if ( this->processor_ == proc__ ) {
        return;
    }
    auto& sender   = ( proc__ == device   ? Ptr                    : Ptr_device            );
    auto& receiver = ( proc__ == device   ? Ptr_device             : Ptr                   );
    auto  protocol = ( proc__ == device   ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);
    cudaMemcpy(receiver, sender, TotalSize * sizeof(T), protocol);

    processor_ = proc__;
#endif 
}

template <typename T, size_t dim>
void mdarray<T,dim>::fill(const T& FillingValue)
{
#ifdef EDUS_GPU
    if ( processor_ == device ) {
        if ( std::abs( FillingValue ) > 1.e-15 ) {
            throw std::runtime_error("Trying to fill a GPU array with a value != 0. Still not implemented.\n");
            exit(1);
        }
        cudaMemset(Ptr_device, 0, TotalSize * sizeof(T));
    } else 
#endif 
    std::fill((*this).begin(), (*this).end(), FillingValue);
}

template <typename T, size_t dim>
template <typename... Args>
inline int mdarray<T,dim>::oneDindex(Args... args) const
{
    return this->multindex.oneDindex(args...);
}


template <typename T, size_t dim>
inline std::vector<int> mdarray<T,dim>::nDindex(const int& oneDindex) const
{
    return this->multindex.nDindex(oneDindex);
}


template <typename T, size_t dim>
template <typename... Args>
inline T const& mdarray<T,dim>::operator()(Args... args) const 
{
    return Ptr[oneDindex(args...)]; 
}


template <typename T, size_t dim>
template <typename... Args>
inline T& mdarray<T,dim>::operator()(Args... args)
{
    return (const_cast<T&>(const_cast<mdarray<T,dim> const&>(*this)(args...)));
}


template <typename T, size_t dim>
inline T const& mdarray<T,dim>::operator[](const int& oneDindex) const
{
    assert(oneDindex>=0 && oneDindex < TotalSize);
    return Ptr[oneDindex]; 
}


template <typename T, size_t dim>
inline T& mdarray<T,dim>::operator[](const int& oneDindex) 
{
    return (const_cast<T&>(const_cast<mdarray<T,dim> const&>(*this)[oneDindex]));
}


template <typename T, size_t dim>
inline int mdarray<T,dim>::get_Size(const int& index) const
{
    return Size[index];
}

template <typename T, size_t dim>
mdarray<T,dim>::~mdarray()
{
    if(this->Ptr != nullptr && !NotDestruct){
        delete[] Ptr;
#ifdef EDUS_GPU
        if( initialized_device ) cudaFree(Ptr_device);
#endif
    }
    Ptr=nullptr;
}

template<typename T, size_t dim> //requires ( dim>0 && dim<7 )
std::ostream& operator<<(std::ostream& os, const mdarray<T,dim>& mdarray_)
{
    for(auto& number : mdarray_){
        os << number;
        os << std::endl;
    }
    return os;
}

template<typename T, size_t dim> 
void copy(const mdarray<T,dim>& ToCopy, mdarray<T,dim>& ToBeCopied, const Processor& proc__)
{
    assert(ToCopy.get_TotalSize() == ToBeCopied.get_TotalSize());
    if( proc__ == device ) {
        assert(ToCopy.initialized_device);
        assert(ToBeCopied.initialized_device);
    }
    switch(proc__)
    {
        case host: 
        {
            std::copy(ToCopy.begin(),
                      ToCopy.end(),
                      ToBeCopied.begin());   
            break;
        }
        case device:
        {
#ifdef EDUS_GPU
            cudaMemcpy(ToBeCopied.Ptr_device, 
                       ToCopy.Ptr_device, 
                       ToCopy.TotalSize * sizeof(T), 
                       cudaMemcpyDeviceToDevice);
#endif
            break;
        }
        default:
            break;
    }
}
