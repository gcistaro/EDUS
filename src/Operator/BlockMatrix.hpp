#ifndef BLOCKMATRIX_HPP
#define BLOCKMATRIX_HPP

#include <complex>
#include <memory>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <typeinfo>

#include "mdContainers/mdContainers.hpp"
#include "MeshGrid/MeshGrid.hpp"
#include "Geometry/Matrix.hpp"
#include "core/mpi/Communicator.hpp"
#ifdef EDUS_HDF5
#include "core/hdf5/hdf5_tree.hpp"
#endif
#include "GlobalFunctions.hpp"
#include "initialize.hpp"
template<class T>
class Operator;
//Our sparse matrices are block matrices. 
//we save them in a contiguous array of size (#blocks, #rows, #cols)
//if the blocks have different shape, #rows and #cols must be the maximum. 
//in the current rudimental implementation, the blocks are all the same and 
//to map matrix elements (nk, n'k') to indices of the 3d array Values
//we have an auxiliary vector. The matrix element is non zero only if
//its 1d index is between SubMatrixPointer.first and SubMatrixPointer.first+nrows*ncols
//for one SubMatrixPointer. 
//we need two different styles
//1 - O(nk, nk') = O(n,n')(k) delta(k-k')
//2 - O(nR, nR') = O(n,n')(R-R') 
template<typename T=std::complex<double>>
class BlockMatrix{
    private:
        std::vector<Matrix<T>> submatrix;
        std::shared_ptr<MeshGrid> meshgrid;    
        static Matrix<T> EmptyMatrix;    
        Space space;
        T NullValue = 0;
        void initialize_submatrix();
    public:

        friend class Operator<T>;

        mdarray<T,3> Values; //first index -> block, others -> matrix
        BlockMatrix() : Values(mdarray<T,3>()){};
        BlockMatrix(const Space& space__, const int& nblocks, const int& nrows, const int& ncols, const int& real_dims__=0)
            {this->initialize(space__, nblocks, nrows, ncols, real_dims__);}
        void initialize(const Space& space__, mdarray<T,3>& Values);
        void initialize(const Space& space__, const int& nblocks, const int& nrows, const int& ncols, const int& real_dims__=0);
        BlockMatrix(const BlockMatrix<T>& A);
        BlockMatrix<T>& operator=(const BlockMatrix<T>& m);
        
        BlockMatrix(BlockMatrix<T>&& A);
        BlockMatrix<T>& operator=(BlockMatrix<T>&& m);

        const T& operator()(const int& nk1, const int& mk2) const; //full index
        T& operator()(const int& nk1, const int& mk2);             //full index

        inline const T& operator()(const int& iblock, const int& n, const int& m) const; //block-like index
        inline T& operator()(const int& iblock, const int& n, const int& m);             //block-like index
        
        inline const T& operator()(const int& i) const;
        inline T& operator()(const int& i) ;

        void fill(const T& Scalar);

        Matrix<T>& operator[](const int& iblock);
        const Matrix<T>& operator[](const int& iblock) const;

        Matrix<T>& operator[](const Coordinate& Point);
        const Matrix<T>& operator[](const Coordinate& Point) const;

        BlockMatrix<T> operator*(const BlockMatrix<T>& B);

        const T* data(const Processor& proc__=host) const {return Values.data(proc__);};
        T* data(const Processor& proc__=host) 
            {return const_cast<T*>((const_cast<const BlockMatrix<T>&>(*this)).Values.data(proc__));};
        //friend void multiply(Matrix<T>& OutputMatrix, const auto& Scalar1, const Matrix<T>& Matrix1, 
        //                                              const auto& Scalar2, const Matrix<T>& Matrix2);

        typedef typename mdarray<T,3>::Iterator  iterator;
        auto begin() const { return Values.begin(); }
        auto end() const {return Values.end(); } 	
        int get_nblocks() const{return Values.get_Size(0);};
        int get_nrows() const{return Values.get_Size(1);};
        int get_ncols() const{return Values.get_Size(2);};
        int get_TotalSize() const {return Values.get_TotalSize();};
        std::shared_ptr<MeshGrid>& get_MeshGrid(){return this->meshgrid;};
        const std::shared_ptr<MeshGrid>& get_MeshGrid() const {return this->meshgrid;} ;

        void set_MeshGrid(const MeshGrid& meshgrid_)
        {
            this->meshgrid = std::make_shared<MeshGrid>(meshgrid_);
        };

        bool has_meshgrid()
        {
            return (meshgrid != nullptr);
        }
        
        template<typename T_, typename U>
        friend void convolution(BlockMatrix<T_>& Output, U Scalar, const BlockMatrix<T_>& Input1, const BlockMatrix<T_>& Input2 );
        template<typename T_, typename U>
        friend void commutator(BlockMatrix<T_>& Output, U Scalar, const BlockMatrix<T_>& Input1, const BlockMatrix<T_>& Input2, 
                                const bool& Erase_Output, const Processor& proc__);


        void diagonalize(std::vector<mdarray<double,1>>& Eigenvalues,
                         BlockMatrix<std::complex<double>>& Eigenvectors) const;


        void test_submatrix();

        auto get_space() const{return space; };
        auto get_space() {return space; };
        void set_space(const Space& space__) {space = space__;};
        //void set_space(const Space& space__) const {space = space__;};
        bool is_hermitian();
        void make_hermitian();
        void make_antihermitian();
        void make_dagger();

        void cut(const double& threshold__);
        void write_h5(const std::string& name__, const std::string& node__, const std::string& label__="");
        void load    (const std::string& name__, const int& node__, const std::string& label__="");
        void identity()
        {
            this->fill(0.);
            for ( auto& sub_ : submatrix ) {
                for (int ibnd = 0; ibnd < sub_.get_nrows(); ++ibnd ) {
                    sub_(ibnd, ibnd) = 1.;
                }
            }
        }

        void set_processor(const Processor& proc__) { Values.processor_ = proc__;};
        void initialize_device();
        void transfer_to(const Processor& );

        template<typename U>
        void Divide(const U& Divisors__);
};

template<class T>
void copy(const BlockMatrix<T>& ToCopy__, BlockMatrix<T>& ToBeCopied__, const Processor& proc__);

//overloading writing matrix
template<class T>
std::ostream& operator<<(std::ostream& os, const BlockMatrix<T>& m);

template<typename T>
auto max(const BlockMatrix<T>& m);

template<typename T>
void multiply(BlockMatrix<T>& Output, T Scalar, const BlockMatrix<T>& Input1, const BlockMatrix<T>& Input2, const Processor& proc__=host );

template<typename T>
void multiply(BlockMatrix<T>& Output, T Scalar, const BlockMatrix<T>& Input1, const BlockMatrix<T>& Input2, T Scalar2, const Processor& proc__=host );



template<typename T, typename U>
void commutator(BlockMatrix<T>& Output, U Scalar, const BlockMatrix<T>& Input1, const BlockMatrix<T>& Input2, 
                                const bool& Erase_Output = true, const Processor& proc__=host);

#include "BlockMatrix_definitions.hpp"

#endif
