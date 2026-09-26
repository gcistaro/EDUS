#ifndef MDCONTAINERS_HPP
#define MDCONTAINERS_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <memory>
#include <complex>
#include <iterator> // For std::forward_iterator_tag
#include <cstddef>  // For std::ptrdiff_t
#include <algorithm>

#include "MultiIndex/MultiIndex.hpp"
#ifdef EDUS_GPU
#include <cuda_runtime.h>
#endif

enum Processor {host, device};

template<typename T>
class BlockMatrix;

template<typename T, size_t dim> //requires ( dim>0 && dim<7 )
class mdarray
{
    private:
        /// container for data on CPU
        T* Ptr=nullptr;
        /// container for data on GPU
        T* Ptr_device=nullptr;
        /// Dimension of mdarray on each of dim
        std::array<int, dim> Size{0};
        /// TotalDimension as multiplication of Size 
        int TotalSize=0;
        /// Offset over each dimension to link 1D index to nD index
        std::array<int, dim> Offset{0};
        /// multiindex to get the 1D<->nD link-
        MultiIndex<dim> multindex;
        /// Check if array needs to be deleted when out-of-scope
        bool NotDestruct = false;
        /// initialize TotalSize and Offset
        void TotalSizeAndOffset();
        /// In general different, because for some libraries we need additional memory (i.e. fftw)
        int real_dims;
        /// Keeps track of where the object lives
        Processor processor_=host;
        /// check if the GPU array is initialized
        bool initialized_device=false;
    public:
        mdarray() = default;
        mdarray(const mdarray<T,dim>& ToBeCopied);
        mdarray<T,dim>& operator=(const mdarray<T,dim>& ToBeCopied);
        
        mdarray(mdarray<T,dim>&& ToBeMoved);
        mdarray<T,dim>& operator=(mdarray<T,dim>&& ToBeMoved);

        mdarray(const std::array<int,dim>& Size_, const int& real_dims__=0);
        void initialize(const std::array<int,dim>& Size_, const int& real_dims__=0);

        mdarray(T* Ptr_, const std::array<int,dim>& Size_, const int& real_dims__=0);
        void initialize(T* Ptr_, const std::array<int,dim>& Size_, const int& real_dims__=0);
        void initialize_device(T* Ptr_device_);

        void initialize_device();
        void transfer_to ( const Processor& );

        void fill(const T& FillingValue);

        struct Iterator
        {
            //iterator tags
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = T;
            using pointer           = value_type*; 
            using reference         = value_type&;  

            //iterator constructor
            Iterator(pointer ptr) : m_ptr(ptr) {}
        
            auto data() { return m_ptr;}
            //iterator overloadings
            reference operator*() const { return *m_ptr; };
            pointer operator->() { return m_ptr; };
            // Prefix increment
            Iterator& operator++() { m_ptr++; return *this; };  
            // Postfix increment
            Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; };
            Iterator operator+=(int rhs) { pointer tmp = this->data(); tmp += rhs; return Iterator(tmp);}
            friend bool operator== (const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
            friend bool operator!= (const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };     
            //difference
            int operator-(const Iterator& it2) const {return this->m_ptr - it2.m_ptr;}        
            Iterator operator+(const Iterator& it2) {return Iterator(this->m_ptr + it2.m_ptr);}        
            Iterator operator+(const int& i) {return Iterator(this->m_ptr + i);}    
            bool operator<(const Iterator& it__) const {return *(*this) < *it__; }    
            bool operator>(const Iterator& it__) const {return *(*this) > *it__; }    
        private:
            pointer m_ptr;
        };

        Iterator begin() const{ return Iterator(Ptr); }
        Iterator end() const{ return Iterator(Ptr+TotalSize); } // TotalSize is out of bounds        
        size_t size() const { return TotalSize; }
        const T* data(const Processor& proc__=host) const {return (proc__ == host ? Ptr : Ptr_device);};
        T* data(const Processor& proc__=host) {return (proc__ == host ? Ptr : Ptr_device);};
                
        template <typename... Args>
        inline int oneDindex(Args... args) const;
        
        inline std::vector<int> nDindex(const int& oneDindex) const;

        template <typename... Args>
        inline T const& operator()(Args... args) const;

        template <typename... Args>
        inline T& operator()(Args... args);
        
        inline T const& operator[](const int& oneDindex) const;
        inline T& operator[](const int& oneDindex);        


        inline int get_Size(const int& index) const;
        inline auto get_Size() const {return Size;};
        inline auto get_TotalSize() const {return TotalSize;};
        inline bool on(const Processor& proc__) const {return ( proc__ == processor_ ? true : false );};
        inline void set_processor(const Processor& proc__) { processor_ = proc__;}
        ~mdarray();

        template<typename T_, size_t dim_>
        friend std::ostream& operator<<(std::ostream&, const mdarray<T_,dim_>& mdarray_); 
        template<typename T_, size_t dim_>
        friend void copy(const mdarray<T_,dim_>& ToCopy, mdarray<T_,dim_>& ToBeCopied, const Processor& proc__);

        friend class BlockMatrix<T>; 
};

#include "mdContainers_definitions.hpp"


#endif