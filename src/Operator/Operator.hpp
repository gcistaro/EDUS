#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include "cassert"
#include "Operator/BlockMatrix.hpp"
#include "fftPair/fftPair.hpp"
#include "MPIindex/MPIindex.hpp"
#include <vector>

/*********************************************************************************************
 *   numbering:
 *   [0   1   2    3 ]
 *   [.   4   5    6 ]
 *   [.   .   7    8 ]
 *   [.   .   .    9 ]
 *
 *   ......
 *   Total number of elements before row m:
 *   b + (b-1) + ... + (b-m-1) = \sum_{i=0}^{m-1} (b-i) = m*b-(m-1)*(m)/2  .... VALID FOR m>1
 *   This is the starting index of row m.
 *
 *   The final index is
 *   StartingIndex + #elements in row = m*b-(m-1)*m/2 + (b-m-1) 
 *
*/ 

class BandIndex
{
    private:
        int NumberOfBands;
        int oneDNumberOfBands;
        std::vector< std::pair<int,int> > RowIndexBoundary;

    public:
        int StartingIndex(const int& row_)
    {
        int StartingIndex = (row_ == 0) ? 0 
                                           :row_*NumberOfBands - row_*(row_-1)/2;
        return StartingIndex;
    };

    int RowIndex(const int& oneDindex)
    {
        auto RowIndexIterator_ = std::find_if(RowIndexBoundary.begin(), RowIndexBoundary.end(),
                                             [&](const auto& RowPair){return oneDindex >= RowPair.first && 
                                                                             oneDindex <= RowPair.second;
                                                                      });
        return RowIndexIterator_ - RowIndexBoundary.begin();
    };

    BandIndex(){};
    BandIndex(const int& NumberOfBands__)
    {
        initialize(NumberOfBands__);
    };

    void initialize(const int& NumberOfBands__)
    {
        NumberOfBands = NumberOfBands__;
        oneDNumberOfBands = StartingIndex(NumberOfBands);
        RowIndexBoundary.resize(NumberOfBands);
        for(int i=0; i<NumberOfBands; ++i){
            RowIndexBoundary[i].first = StartingIndex(i);
            RowIndexBoundary[i].second = StartingIndex(i) + (NumberOfBands - i) - 1;
        }

    }
    int oneDindex(const int& bnd1, const int& bnd2)
    {
        assert(bnd1 <= bnd2);
        assert(bnd1 <= NumberOfBands && bnd2 <= NumberOfBands);

        return RowIndexBoundary[bnd1].first + (bnd2-bnd1);
    };

    std::pair<int, int> twoDband(const int& oneDindex__)
    {
        assert(oneDindex__ < oneDNumberOfBands);
        std::pair<int, int> twoDband_;
        twoDband_.first = RowIndex(oneDindex__);
        twoDband_.second = oneDindex__ - RowIndexBoundary[twoDband_.first].first + twoDband_.first;
        return twoDband_;
    };

    int& get_oneDNumberOfBands(){ return oneDNumberOfBands; };
};



template < typename T=std::complex<double> >
class Operator
{
    private:
	    BlockMatrix<T> Operator_k;
        BlockMatrix<T> Operator_R;

        mdarray<std::complex<double>, 2> FTfriendly_Operator_R;
        mdarray<std::complex<double>, 2> FTfriendly_Operator_k;
        std::shared_ptr<MeshGrid> FT_meshgrid_k;
        std::shared_ptr<MeshGrid> FT_meshgrid_R;

        //static BandIndex bandindex;
        static MultiIndex<2> bandindex;
        BandGauge bandgauge;
        Space space;

        bool locked_bandgauge = false;
        bool locked_space = false;
        bool initialized_dft = false;
        bool initialized_fft = false;
        FourierTransform ft_;
        Processor processor_ = host;

        std::string tagname = "";


    public:
        static Space SpaceOfPropagation;
        static BlockMatrix<T> temp_k;
        friend class Simulation;
        static std::shared_ptr<MeshGrid> MeshGrid_Null;
        static BlockMatrix<T> EigenVectors;
        static BlockMatrix<T> EigenVectors_dagger;
        static MPIindex<3> mpindex;
        Operator() : Operator_k(BlockMatrix<T>()), Operator_R(BlockMatrix<T>()){Operator_k.set_space(k); Operator_R.set_space(R);};


        Operator(const Operator<T>& Op_)
        {
            (*this) = Op_;
        };

        Operator<T>& operator=(const Operator<T>& Op_) 
        {
            this->Operator_k = Op_.get_Operator(k);
            this->Operator_R = Op_.get_Operator(R);
            this->FTfriendly_Operator_R = Op_.FTfriendly_Operator_R;
            this->FTfriendly_Operator_k = Op_.FTfriendly_Operator_k;
            this->bandgauge = Op_.bandgauge;
            this->space = Op_.space;
            this->SpaceOfPropagation = Op_.SpaceOfPropagation;
            this->locked_bandgauge = Op_.locked_bandgauge;
            this->locked_space = Op_.locked_space;
            //this->ft_ = Op_.ft_; //Warning!! Never copy ft!! is deleted anyway
            return *this;
        }
        
        Operator(Operator<T>&& Op_)  = default;
        Operator<T>& operator=(Operator<T>&& Op_) = default;

        BlockMatrix<T>& get_Operator_k()
        {
            return const_cast<BlockMatrix<T>&>(static_cast<const Operator<T>&>(*this).get_Operator_k());
        };       
        
        const BlockMatrix<T>& get_Operator_k() const
        {
            return Operator_k;
        };      
        
        BlockMatrix<T>& get_Operator_R()
        {
            return const_cast<BlockMatrix<T>&>(static_cast<const Operator<T>&>(*this).get_Operator_R());
        };      
        
        const BlockMatrix<T>& get_Operator_R() const
        {
            return Operator_R;
        };

        BlockMatrix<T>& get_Operator(const Space& space__)
        {
            return const_cast<BlockMatrix<T>&>(static_cast<const Operator<T>&>(*this).get_Operator(space__));
        };       
        
        const BlockMatrix<T>& get_Operator(const Space& space__) const
        {
            auto& Operator_to_return = space__ == k ? Operator_k : Operator_R;
            return Operator_to_return;
        };      

        void set_SpaceOfPropagation(const Space& space__)
        {
            SpaceOfPropagation = space__;
        };

        auto get_SpaceOfPropagation()
        {
            return SpaceOfPropagation;
        };


        void initialize_fft(const MeshGrid& MG, const int& nbnd, const std::string& tagname_="")
        {
            tagname = tagname_;
            //for now this is the only case implemented. it will be more general.
            //we enter in this function only once
            if(initialized_fft){
                return;
            }
            initialized_fft = true;
#ifdef EDUS_MPI
            Operator_R = BlockMatrix<std::complex<double>>(R,mpindex.get_nlocal(), nbnd, nbnd, 
                                                            mpindex.get_RecommendedAllocate_fftw());
#else
            Operator_R = BlockMatrix<std::complex<double>>(R,mpindex.get_nlocal(), nbnd, nbnd, 
                                                            mpindex.get_RecommendedAllocate_fftw());
#endif
            switch(MG.get_space())
            {
                case R:
                {
                    auto& Rgrid = Operator_R.get_MeshGrid();
                    Rgrid = std::make_shared<MeshGrid>(MG);
                    FT_meshgrid_k = std::make_shared<MeshGrid>(fftPair(MG));
                    FT_meshgrid_R = std::make_shared<MeshGrid>(fftPair(*FT_meshgrid_k));
                    break;
                }
                case k:
                {
                    //FT_meshgrid_R = std::make_shared<MeshGrid<R>>(fftPair<R,k>(MG));
                    //FT_meshgrid_k = std::make_shared<MeshGrid<k>>(fftPair<R,k>(*FT_meshgrid_R));
                    throw std::runtime_error("Initialize fft from k Not yet implemented!\n");
                }
            }
            Operator_k = BlockMatrix<std::complex<double>>(k,mpindex.get_nlocal(), nbnd, nbnd, 
                                                           mpindex.get_RecommendedAllocate_fftw());
            auto& kgrid = Operator_k.get_MeshGrid();
            kgrid = FT_meshgrid_k;
            //bandindex.initialize(nbnd);
            bandindex.initialize({nbnd, nbnd});

            //bandindex FTfriendly_Operator_k = mdarray<std::complex<double>, 2>({nbnd*(nbnd+1)/2, mpindex.get_RecommendedAllocate_fftw()});
            //bandindex FTfriendly_Operator_R = mdarray<std::complex<double>, 2>({nbnd*(nbnd+1)/2, mpindex.get_RecommendedAllocate_fftw()});
            auto dim_k = mpindex.get_nlocal() > 0 ? mpindex.get_nlocal() : 1;//to avoid that we get a 0 allocated pointer
//==#ifdef EDUS_MPI
            FTfriendly_Operator_k = mdarray<std::complex<double>, 2>( Operator_k.data(), {dim_k,nbnd*nbnd} );
            FTfriendly_Operator_R = mdarray<std::complex<double>, 2>( Operator_R.data(), {dim_k,nbnd*nbnd} );
//==#else
//==            FTfriendly_Operator_k = mdarray<std::complex<double>, 2>({nbnd*nbnd, mpindex.get_nlocal()});
//==            FTfriendly_Operator_R = mdarray<std::complex<double>, 2>({nbnd*nbnd, mpindex.get_nlocal()});
//==#endif
            //use convolution index for shuffle index.
            std::vector<int> Dimensions(3);
            for(int ix=0; ix<3; ix++){
                Dimensions[ix] = FT_meshgrid_k->get_Size()[ix];
            }
            ft_.initialize(FTfriendly_Operator_k, FTfriendly_Operator_R, Dimensions, tagname);
            //shuffle_to_fft();
            this->space = R;
            locked_space = true;
        };

        void initialize_fft(const Operator& Op__, const std::string& tagname__="")
        {            
            auto nbnd = Op__.get_Operator_R().get_nrows();
            auto& MG = Op__.get_Operator_R().get_MeshGrid();
            initialize_fft(*MG, nbnd, tagname__);
        };

        void dft(const std::vector<Coordinate>& path, const int& sign, const bool& UseMPI=true)
        {
            initialize_dft();
#ifdef EDUS_MPI
            std::vector<Coordinate> path_local;
            auto& LocalRange = mpindex.get_LocalRange();
            if(UseMPI) {
                for ( int index_local = 0; index_local < LocalRange.second-LocalRange.first+1; index_local++ ) {
                    path_local.push_back( path[ mpindex.loc1D_to_glob1D( index_local )] );
                }
            }
            else{
                path_local = path;
            }
            execute_dft(path_local, sign);
            Operator_k.initialize(k, path_local.size(), Operator_R.get_nrows(), Operator_R.get_ncols());
            Operator_k.set_MeshGrid(MeshGrid(k, path));
            shuffle_to_RK_dft();
#else                        
            execute_dft(path, sign);
            Operator_k.initialize(k, FTfriendly_Operator_k.get_Size(1), Operator_R.get_nrows(), Operator_R.get_ncols());
            Operator_k.set_MeshGrid(MeshGrid(k, path));
            shuffle_to_RK();
#endif
        }


        void print_Rdecay(const std::string& str__, const std::vector<Coordinate>& rwann__)
        {
            std::stringstream rank;
#ifdef EDUS_MPI
            rank << str__ << mpi::Communicator::world().rank() << ".txt";
#else
            rank << str__ << "0.txt";
#endif
            std::ofstream os;
            os.open(rank.str());
            auto Rgamma_centered = get_GammaCentered_grid(*this->get_Operator_R().get_MeshGrid());
            for( int iR_loc=0; iR_loc< this->get_Operator_R().get_nblocks(); ++iR_loc ){
                //os << DensityMatrix.get_Operator_R()[i] << std::endl;
                auto iR_glob = this->mpindex.loc1D_to_glob1D(iR_loc);
                for( int ialpha = 0; ialpha < this->get_Operator_R().get_nrows(); ++ialpha ) {
                    for( int ibeta = 0; ibeta < this->get_Operator_R().get_nrows(); ++ibeta ) {
                        os << (rwann__[ialpha] + Rgamma_centered[iR_glob] -rwann__[ibeta]).norm() ;
                        os << this->get_Operator_R()[iR_loc](ialpha, ibeta) << std::endl;
                    }
                }
            }
            os.close();

        }



        void set_space(const Space& space__)
        {
            space = space__;
        }

        MeshGrid& get_FT_meshgrid_k() 
        {
            return *FT_meshgrid_k;
        }

/*
 * "initialize_dft()"
 * The following function is used to reshuffle the operator in such a way that R is the first index
 * and can be easily used for doing a fft. In the same time, we put as R mesh of the dft the one of 
 * Operator_R.
*/
        void initialize_dft()
        {
            PROFILE("initialize_dft");
            //we enter in this function only once
            if(initialized_dft){
                return;
            }
            initialized_dft = true;
            auto nbnd = Operator_R.get_nrows();
            //bandindex.initialize(nbnd);
            bandindex.initialize({nbnd, nbnd});
            //initialize input of fft
            //bandindex FTfriendly_Operator_R = mdarray<std::complex<double>, 2>({nbnd*(nbnd+1)/2, Operator_R.get_nblocks()});
            FTfriendly_Operator_R = mdarray<std::complex<double>, 2>({nbnd*nbnd, Operator_R.get_nblocks()});
            for(int iR=0; iR<Operator_R.get_nblocks(); iR++){
                for(int ibnd1=0; ibnd1<nbnd; ++ibnd1){
                    //bandindex for(int ibnd2=ibnd1; ibnd2<nbnd; ++ibnd2){
                    for(int ibnd2=0; ibnd2<nbnd; ++ibnd2){
                        FTfriendly_Operator_R(static_cast<int>(bandindex.oneDindex(ibnd1, ibnd2)), iR) = 
                                                                            Operator_R(iR, ibnd1, ibnd2);
                    }
                }
            }
            std::vector< std::vector<double> > Mesh_FT;
            auto MeshR_GammaCentered =  get_GammaCentered_grid(
                    *Operator_R.get_MeshGrid());
            auto& mesh_operator = MeshR_GammaCentered.get_mesh();

            Mesh_FT.resize(mesh_operator.size());        

            for(int im=0; im < int( mesh_operator.size() ); im++){
                Mesh_FT[im].resize(3);
                Mesh_FT[im][0] = mesh_operator[im].get(LatticeVectors(R))[0];
                Mesh_FT[im][1] = mesh_operator[im].get(LatticeVectors(R))[1];
                Mesh_FT[im][2] = mesh_operator[im].get(LatticeVectors(R))[2];
            }
            ft_.initialize(FTfriendly_Operator_R, Mesh_FT);
        }


        void execute_dft(const std::vector<Coordinate>& path, const int& sign)
        {
            std::vector< std::vector<double>> path_bare(path.size());
            for(int i=0; i < int( path.size() ); ++i){
                path_bare[i].resize(3);
                path_bare[i][0] = path[i].get(LatticeVectors(k))[0];
                path_bare[i][1] = path[i].get(LatticeVectors(k))[1];
                path_bare[i][2] = path[i].get(LatticeVectors(k))[2];
            }
            FTfriendly_Operator_k = ft_.dft(path_bare, +1);
        }


        void shuffle_to_RK_dft()
        {
            PROFILE("shuffle_to_RK_dft")
            #pragma omp parallel for
            for(int ik=0; ik<Operator_k.get_nblocks(); ++ik){
                for(int ibnd1=0; ibnd1<Operator_k.get_nrows(); ++ibnd1){
                    //bandindex for(int ibnd2=ibnd1; ibnd2<Operator_k.get_nrows(); ++ibnd2){ 
                    for(int ibnd2=0; ibnd2<Operator_k.get_nrows(); ++ibnd2){ 
                        Operator_k(ik, ibnd1, ibnd2) = FTfriendly_Operator_k(static_cast<int>(bandindex.oneDindex(ibnd1,ibnd2)),ik);            
                        //bandindex Operator_k(ik, ibnd2, ibnd1) = std::conj(Operator_k(ik,ibnd1, ibnd2));
        		    }
                }
            }
        }

        
        void shuffle_to_RK()
        {
#ifndef EDUS_MPI
            shuffle_to_RK_dft();
#endif
        }

        void shuffle_to_fft_R()
        {
//== #ifndef EDUS_MPI
//== #ifdef EDUS_TIMERS
//==             PROFILE("Operator::shuffle_to_fft_R");
//== #endif
//==             //auto ci = MeshGrid::get_ConvolutionIndex(*Operator_R.get_MeshGrid() , *FT_meshgrid_R, *MeshGrid_Null);
//== 
//==             #pragma omp parallel for
//==             for(int iR=0; iR<Operator_R.get_nblocks(); iR++){
//==                 //assert(ci(iR,0) != -1);
//==                 for(int ibnd1=0; ibnd1<Operator_R.get_nrows(); ++ibnd1){
//==                     // bandindex for(int ibnd2=ibnd1; ibnd2<Operator_R.get_ncols(); ++ibnd2){
//==                     for(int ibnd2=0; ibnd2<Operator_R.get_ncols(); ++ibnd2){
//==                     //    FTfriendly_Operator_R(static_cast<int>(bandindex.oneDindex(ibnd1, ibnd2)), ci(iR,0)) = Operator_R(iR, ibnd1, ibnd2);
//==                         FTfriendly_Operator_R(static_cast<int>(bandindex.oneDindex(ibnd1, ibnd2)), iR) = Operator_R(iR, ibnd1, ibnd2);
//==                     }
//==                 }
//==             }
//== #endif
        }

        void shuffle_to_fft_k()
        {
//== #ifndef EDUS_MPI
//== #ifdef EDUS_TIMERS
//==             PROFILE("Operator::shuffle_to_fft_k");
//== #endif
//==             #pragma omp parallel for
//==             for(int ik=0; ik<Operator_k.get_nblocks(); ik++){
//==                 for(int ibnd1=0; ibnd1<Operator_k.get_nrows(); ++ibnd1){
//==                     // bandindex for(int ibnd2=ibnd1; ibnd2<Operator_R.get_ncols(); ++ibnd2){
//==                     for(int ibnd2=0; ibnd2<Operator_k.get_ncols(); ++ibnd2){
//==                         FTfriendly_Operator_k(static_cast<int>(bandindex.oneDindex(ibnd1, ibnd2)), ik) = Operator_k(ik, ibnd1, ibnd2);
//==                     }
//==                 }
//==             }
//== #endif
        }

        void shuffle_from_fft_R()
        {
//== #ifndef EDUS_MPI
//== #ifdef EDUS_TIMERS
//==             PROFILE("Operator::shuffle_from_fft_R");
//== #endif
//==             //auto ci = MeshGrid::get_ConvolutionIndex(*Operator_R.get_MeshGrid(), *FT_meshgrid_R, *MeshGrid_Null);
//==             // bandindex auto ciminus = MeshGrid::get_ConvolutionIndex(*MeshGrid_Null, *Operator_R.get_MeshGrid(), *Operator_R.get_MeshGrid());
//== 
//==             #pragma omp parallel for 
//==             for(int iR=0; iR<Operator_R.get_nblocks(); iR++){
//==                 //assert(ci(iR,0) != -1);
//==                 for(int ibnd1=0; ibnd1<Operator_R.get_nrows(); ++ibnd1){
//==                     // bandindex for(int ibnd2=ibnd1; ibnd2<Operator_R.get_ncols(); ++ibnd2){
//==                     for(int ibnd2=0; ibnd2<Operator_R.get_ncols(); ++ibnd2){
//==                         //Operator_R(iR, ibnd1, ibnd2) = FTfriendly_Operator_R(static_cast<int>(bandindex.oneDindex(ibnd1, ibnd2)), ci(iR,0));
//==                         Operator_R(iR, ibnd1, ibnd2) = FTfriendly_Operator_R(static_cast<int>(bandindex.oneDindex(ibnd1, ibnd2)), iR);
//==                     }
//==                 }
//==             }
//== #endif
        }

        void shuffle_from_fft_k()
        {
//== #ifndef EDUS_MPI
//== #ifdef EDUS_TIMERS
//==             PROFILE("Operator::shuffle_from_fft_k");
//== #endif
//==             #pragma omp parallel for 
//==             for(int ik=0; ik<Operator_k.get_nblocks(); ik++){
//==                 for(int ibnd1=0; ibnd1<Operator_k.get_nrows(); ++ibnd1){
//==                     // bandindex for(int ibnd2=ibnd1; ibnd2<Operator_k.get_ncols(); ++ibnd2){
//==                     for(int ibnd2=0; ibnd2<Operator_k.get_ncols(); ++ibnd2){
//==                         Operator_k(ik, ibnd1, ibnd2) = 
//==                             FTfriendly_Operator_k(static_cast<int>(bandindex.oneDindex(ibnd1, ibnd2)), ik);
//==                         //Operator_k(ik, ibnd2, ibnd1) = conj(Operator_k(ik, ibnd1, ibnd2));
//==                     }
//==                 }
//==             }
//== #endif            
        }

        void go_to(const Space& space__) 
        {
            switch (space__) 
            {
                case ( k ) :
                {
                    go_to_k();
                    break;
                }
                case ( R ) :
                {
                    go_to_R();
                    break;
                }
                default :
                    break;
            }
        }

        void go_to(const BandGauge& bandgauge__) 
        {
            switch (bandgauge__) 
            {
                case ( wannier ) :
                {
                    go_to_wannier();
                    break;
                }
                case ( bloch ) :
                {
                    go_to_bloch();
                    break;
                }
                default :
                    break;
            }
        }

        void go_to_wannier()
        {
#ifdef EDUS_TIMERS
            PROFILE("Operator::go_to_wannier");
#endif
            assert(locked_bandgauge);
            if (bandgauge == wannier){
                return;
            }
            go_to_k();
            auto temp_k = Operator_k;
            temp_k.fill(0); 

            multiply(temp_k, std::complex<double>(1.), EigenVectors, Operator_k);
            Operator_k.fill(0);
            multiply(Operator_k, std::complex<double>(1.), temp_k, EigenVectors_dagger);
            bandgauge = wannier;
        };

        void go_to_bloch()
        {
            // O_{bloch} = U^\dagger O_{wannier} U
#ifdef EDUS_TIMERS
            PROFILE("Operator::go_to_bloch");
#endif
            assert(locked_bandgauge);
            if(bandgauge == bloch){
                return;
            }
            go_to_k();
            auto temp_k = Operator_k;
            temp_k.fill(0); 
            multiply(temp_k, std::complex<double>(1.), EigenVectors_dagger, Operator_k);
            Operator_k.fill(0);
            multiply(Operator_k, std::complex<double>(1.), temp_k, EigenVectors);
            bandgauge = bloch;
        };

        void go_to_R(const bool& do_fft = true)
        {
#ifdef EDUS_TIMERS
            PROFILE("Operator::go_to_R");
#endif
            assert(locked_space && initialized_fft);
            if(space == R){
                return;
            }
            if( !do_fft )
            {
                space = R;
                return;
            }
            shuffle_to_fft_k();
            ft_.fft(-1, processor_);       
            shuffle_from_fft_R();  

            space = R;
        }

        void go_to_k(const bool& do_fft = true)
        {
#ifdef EDUS_TIMERS
            PROFILE("Operator::go_to_k");
#endif
            assert(locked_space && initialized_fft);
            if(space == k){
                return;
            }
            if( !do_fft )
            {
                space = k;
                return;
            }
            shuffle_to_fft_R();
            ft_.fft(+1, processor_);          
            shuffle_from_fft_k();  

            space = k;
        }

        void lock_gauge(const BandGauge& bandgauge__)
        {
            bandgauge = bandgauge__;
            locked_bandgauge = true;
        };

        void lock_space(const Space& space__)
        {
            space = space__;
            locked_space = true;
        };


        void initialize_dims(const Operator<T>& Allocated_Op)
        {
            this->Operator_k.initialize(k, Allocated_Op.get_Operator_k().get_nblocks(), 
                Allocated_Op.get_Operator_k().get_nrows(),  Allocated_Op.get_Operator_k().get_ncols() );
            this->Operator_R.initialize(k, Allocated_Op.get_Operator_R().get_nblocks(), 
                Allocated_Op.get_Operator_R().get_nrows(),  Allocated_Op.get_Operator_R().get_ncols() );
            FTfriendly_Operator_R.initialize(Allocated_Op.FTfriendly_Operator_R.get_Size());
            FTfriendly_Operator_k.initialize(Allocated_Op.FTfriendly_Operator_k.get_Size());
            FT_meshgrid_k = Allocated_Op.FT_meshgrid_k;
            FT_meshgrid_R = Allocated_Op.FT_meshgrid_R;
            this->Operator_k.set_MeshGrid(*FT_meshgrid_k);
            this->Operator_R.set_MeshGrid(*FT_meshgrid_R);
            SpaceOfPropagation = Allocated_Op.SpaceOfPropagation;            
        }

        bool is_hermitian()
        {
            bool is_hermitian = true; 
            //#pragma omp parallel for
            for( int ik=0; ik<Operator_k.get_nblocks(); ik++ ) {
                for(int irow=0; irow < Operator_k.get_nrows(); irow++) {
                    for (int icol=irow; icol< Operator_k.get_ncols(); icol++) {
                        //std::cout << ik << " " << irow << " " << icol << " " << std::abs( Operator_k( ik, irow, icol ) - std::conj( Operator_k( ik, icol, irow ) ) ) << std::endl;
                        if( std::abs( Operator_k( ik, irow, icol ) - std::conj( Operator_k( ik, icol, irow ) ) ) > 1.e-13 ) {
                            std::stringstream ss; 
                            ss << "Error while checking hermiticity of operator with tagname " << tagname << "\n."; 
                            ss << "ik = " << ik << " irow = " << irow << " icol = " << icol << " have Operator_k( ik, irow, icol ) = ";
                            ss << std::setw(25) << std::setprecision(15) << Operator_k( ik, irow, icol ).real();
                            ss << std::setw(25) << std::setprecision(15) << Operator_k( ik, irow, icol ).imag();
                            ss << " but std::conj( Operator_k( ik, icol, irow ) ) = ";
                            ss << std::setw(25) << std::setprecision(15) << std::conj( Operator_k( ik, icol, irow ) ).real();
                            ss << std::setw(25) << std::setprecision(15) << std::conj( Operator_k( ik, icol, irow ) ).imag();
                            ss << std::endl;
                            throw std::runtime_error(ss.str());
                            is_hermitian = false;
                        }
                    }
                }
            }
            return is_hermitian;
        }

        void initialize_device()
        {
            Operator_k.initialize_device();
            Operator_R.initialize_device();
            FTfriendly_Operator_k.initialize_device(Operator_k.data(device));
            FTfriendly_Operator_R.initialize_device(Operator_R.data(device));
        }

        void transfer_to(const Processor& proc__)
        {
            processor_ = proc__;
            Operator_k.transfer_to(proc__);
            Operator_R.transfer_to(proc__);
        }

        void set_processor(const Processor& proc__) 
        {
            processor_ = proc__; 
            Operator_k.set_processor(proc__); 
            Operator_R.set_processor(proc__);
            FTfriendly_Operator_k.set_processor(proc__);
            FTfriendly_Operator_R.set_processor(proc__);            
        }

        template<typename T_>
        void fill(const T_& value) 
        {
            Operator_k.fill(value);
            Operator_R.fill(value);
        }
};

        template <typename Scalar_T>
void axpby(Operator<std::complex<double>>& Output_, 
                    const Scalar_T& FirstScalar_, 
                    const Operator<std::complex<double>>& FirstAddend_, 
                    const Scalar_T& SecondScalar_, 
                    const Operator<std::complex<double>>& SecondAddend_,
                    const Processor& proc__=host)

{
#ifdef EDUS_TIMERS
    PROFILE("axpby");
#endif
    auto& SpaceOfPropagation = Operator<std::complex<double>>::SpaceOfPropagation;

    auto& Output       = Output_      .get_Operator( SpaceOfPropagation );
    auto& FirstAddend  = FirstAddend_ .get_Operator( SpaceOfPropagation );
    auto& SecondAddend = SecondAddend_.get_Operator( SpaceOfPropagation );

    axpby(Output, FirstScalar_, FirstAddend, SecondScalar_, SecondAddend, proc__);
}




// bandindex template < typename T>
// bandindex BandIndex Operator<T>::bandindex;
template < typename T>
MultiIndex<2> Operator<T>::bandindex;

template < typename T>
BlockMatrix<T> Operator<T>::EigenVectors;

template < typename T>
BlockMatrix<T> Operator<T>::EigenVectors_dagger;


template < typename T>
BlockMatrix<T> Operator<T>::temp_k;

template < typename T>
std::shared_ptr<MeshGrid> Operator<T>::MeshGrid_Null = std::make_shared<MeshGrid>(MeshGrid(k, std::vector<Coordinate>({Coordinate(0,0,0)})));

template < typename T>
MPIindex<3> Operator<T>::mpindex;

template < typename T>
Space Operator<T>::SpaceOfPropagation;

#endif
