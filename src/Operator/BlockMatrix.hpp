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
        std::vector<Matrix<T>> submatrix_;
        std::shared_ptr<MeshGrid> meshgrid_;    
        static Matrix<T> empty_matrix_;    
        Space space_;
        T null_value_ = 0;
        void initialize_submatrix();
    public:

        friend class Operator<T>;

        mdarray<T,4> values_; //first index -> k, second index -> q, others -> matrix
        BlockMatrix() : values_(mdarray<T,4>()){};
        BlockMatrix(const Space& space__, int nk__, 
                                          int nrows__, 
                                          int ncols__, 
                                          int real_dims__=0,
                                          int nq__=1)
        {this->initialize(space__, nk__, nrows__, ncols__, real_dims__, nq__);}
        void initialize(const Space& space__, mdarray<T,4>& values__);
        void initialize(const Space& space__, mdarray<T,3>& values__);
        void initialize(const Space& space__, int nk__, 
                                              int nrows__, 
                                              int ncols__, 
                                              int real_dims__=0,
                                              int nq__=1); 
        BlockMatrix(const BlockMatrix<T>& a__);
        BlockMatrix<T>& operator=(const BlockMatrix<T>& a__);
        
        BlockMatrix(BlockMatrix<T>&& a__);
        BlockMatrix<T>& operator=(BlockMatrix<T>&& a__);

        const T& operator()(const int& nk1__, const int& mk2__) const; //full index
        T& operator()(const int& nk1__, const int& mk2__);             //full index

        inline const T& operator()(const int& iblock__, const int& n__, const int& m__) const; //block-like index
        inline T& operator()(const int& iblock__, const int& n__, const int& m__);             //block-like index

        inline const T& operator()(const int& ik__, const int& iq__, const int& n__, const int& m__) const; //(k,q)-like index
        inline T& operator()(const int& ik__, const int& iq__, const int& n__, const int& m__);             //(k,q)-like index
        
        inline const T& operator()(const int& i__) const;
        inline T& operator()(const int& i__) ;

        void fill(const T& scalar__);

        Matrix<T>& operator[](const int& iblock__);
        const Matrix<T>& operator[](const int& iblock__) const;

        Matrix<T>& operator[](const Coordinate& point__);
        const Matrix<T>& operator[](const Coordinate& point__) const;

        BlockMatrix<T> operator*(const BlockMatrix<T>& b__);

        const T* data(const Processor& proc__=host) const {return values_.data(proc__);};
        T* data(const Processor& proc__=host) 
            {return const_cast<T*>((const_cast<const BlockMatrix<T>&>(*this)).values_.data(proc__));};
        //friend void multiply(Matrix<T>& OutputMatrix, const auto& Scalar1, const Matrix<T>& Matrix1, 
        //                                              const auto& Scalar2, const Matrix<T>& Matrix2);

        typedef typename mdarray<T,4>::Iterator  iterator;
        auto begin() const { return values_.begin(); }
        auto end() const {return values_.end(); } 	
        int get_nk() const{return values_.get_Size(0);};
        int get_nq() const{return values_.get_Size(1);};
        int get_nblocks() const{return get_nk()*get_nq();};
        int get_nrows() const{return values_.get_Size(2);};
        int get_ncols() const{return values_.get_Size(3);};
        int get_TotalSize() const {return values_.get_TotalSize();};
        std::shared_ptr<MeshGrid>& get_MeshGrid(){return this->meshgrid_;};
        const std::shared_ptr<MeshGrid>& get_MeshGrid() const {return this->meshgrid_;} ;

        void set_MeshGrid(const MeshGrid& meshgrid__)
        {
            this->meshgrid_ = std::make_shared<MeshGrid>(meshgrid__);
        };

        bool has_meshgrid()
        {
            return (meshgrid_ != nullptr);
        }
        
        template<typename T_, typename U>
        friend void convolution(BlockMatrix<T_>& output__, U scalar__, const BlockMatrix<T_>& input1__, const BlockMatrix<T_>& input2__ );
        template<typename T_, typename U>
        friend void commutator(BlockMatrix<T_>& output__, U scalar__, const BlockMatrix<T_>& input1__, const BlockMatrix<T_>& input2__, 
                                const bool& erase_output__, const Processor& proc__);


        void diagonalize(std::vector<mdarray<double,1>>& eigenvalues__,
                         BlockMatrix<std::complex<double>>& eigenvectors__) const;


        void test_submatrix();

        auto get_space() const{return space_; };
        auto get_space() {return space_; };
        void set_space(const Space& space__) {space_ = space__;};
        //void set_space(const Space& space__) const {space_ = space__;};
        bool is_hermitian();
        void make_hermitian();
        void make_antihermitian();
        void make_dagger();

        void cut(const double& threshold__);
        /// Writes the matrix in the h5 file; the blocks of all the ranks of comm__ are gathered on its rank 0
        void write_h5(const std::string& name__, const std::string& node__, const std::string& label__, 
                      const mpi::Communicator& comm__);
        /// Reads the matrix from the h5 file; rank 0 of comm__ reads and sends the blocks to the other ranks
        void load    (const std::string& name__, const int& node__, const std::string& label__, 
                      const mpi::Communicator& comm__);
        void identity()
        {
            this->fill(0.);
            for ( auto& sub_ : submatrix_ ) {
                for (int ibnd = 0; ibnd < sub_.get_nrows(); ++ibnd ) {
                    sub_(ibnd, ibnd) = 1.;
                }
            }
        }

        void set_processor(const Processor& proc__) { values_.processor_ = proc__;};
        void initialize_device();
        void transfer_to(const Processor& );

        template<typename U>
        void Divide(const U& divisors__);
};

template<class T>
void copy(const BlockMatrix<T>& to_copy__, BlockMatrix<T>& to_be_copied__, const Processor& proc__);

//overloading writing matrix
template<class T>
std::ostream& operator<<(std::ostream& os__, const BlockMatrix<T>& m__);

template<typename T>
auto max(const BlockMatrix<T>& m__);

template<typename T>
void multiply(BlockMatrix<T>& output__, T scalar__, const BlockMatrix<T>& input1__, const BlockMatrix<T>& input2__, const Processor& proc__=host );

template<typename T>
void multiply(BlockMatrix<T>& output__, T scalar__, const BlockMatrix<T>& input1__, const BlockMatrix<T>& input2__, T scalar2__, const Processor& proc__=host );



template<typename T, typename U>
void commutator(BlockMatrix<T>& output__, U scalar__, const BlockMatrix<T>& input1__, const BlockMatrix<T>& input2__, 
                                const bool& erase_output__ = true, const Processor& proc__=host);

#include "BlockMatrix_definitions.hpp"

#endif
