//this function must be called at the beginning of any miniapp to allow for 
//initialization of EDUS important quantities, such as:
//- header printing with git infos
//- MPI_COMM_WORLD
//- fftw initialization
#ifdef EDUS_MKL
    #include "mkl.h"
#else 
    #include "lapacke.h"
#endif
//#include <thread>
#include <iomanip>
#include "initialize.hpp"
#include "GlobalFunctions.hpp"

#ifdef EDUS_MPI
std::unique_ptr<mpi::Communicator> kpool_comm = nullptr;
std::unique_ptr<mpi::Communicator> band_comm = nullptr;
int NumberKpools;
#endif

#ifdef EDUS_GPU
cublasHandle_t cublas_handle = nullptr;
#endif

/*
    Parallelization using grid / example with 4 kpools, 12 ranks

    rows are kpools, columns are bandpools. inside the grid, I put the rank on the world communicator.

                  bandpool
                0         1         2
           --|--------------------------
    kpool  0 |  0         4         8        
           1 |  1         5         9 
           2 |  2         6        10
           3 |  3         7        11
*/
void initialize()
{  
#ifdef EDUS_MKL_THREAD
    mkl_set_num_threads(omp_get_max_threads());
#endif
#ifdef EDUS_MPI
    mpi::Communicator::initialize(MPI_THREAD_FUNNELED);
    NumberKpools = mpi::Communicator::world().size();
    fftw_mpi_init();  
#endif 
#ifdef EDUS_GPU
    cublasCreate(&cublas_handle); 
#endif
#ifdef EDUS_FFTWTHREADS
    fftw_init_threads();
#endif
    print_header();
    output::title("PARALLELIZATION RECAP");  
    output::print("OpenMP  threads:   *", omp_get_max_threads());
#ifdef EDUS_MKL
    output::print("MKL  threads:      *", mkl_get_max_threads());
#endif 
#ifdef EDUS_FFTWTHREADS
    fftw_plan_with_nthreads(omp_get_max_threads());
    output::print("fftw  threads:     *", fftw_planner_nthreads());
    output::stars();
#endif
 #ifdef EDUS_MPI
 
     assert( mpi::Communicator::world().size()%NumberKpools == 0 );//for now i just implemented a rectangular MPI grid
     
     //create k point communicator
     kpool_comm = std::make_unique<mpi::Communicator>();
     kpool_comm->generate( mpi::Communicator::world(), mpi::Communicator::world().rank()/NumberKpools );
 
     //create band communicator
     band_comm = std::make_unique<mpi::Communicator>();
     band_comm->generate( mpi::Communicator::world(), mpi::Communicator::world().rank()%NumberKpools );
// == 
// ==     //recap of mpi ranks -- to be done only for high verbosity
// ==     if ( mpi::Communicator::world().rank() != 0 ) {
// ==         auto aux = mpi::Communicator::world().rank();
// ==         mpi::Communicator::world().send( &aux, 0 );
// ==         aux = kpool_comm.rank();
// ==         mpi::Communicator::world().send( &aux, 0 );
// ==         aux = kpool_comm.size();
// ==         mpi::Communicator::world().send( &aux, 0 );
// ==         aux = band_comm.rank();
// ==         mpi::Communicator::world().send( &aux, 0 );
// ==         aux = band_comm.size();
// ==         mpi::Communicator::world().send( &aux, 0 );
// ==     }
// ==     else {
         output::print("MPI world size:    *", mpi::Communicator::world().size());
// ==         output::title("MPI parallelization recap");
// ==         output::print("WORLD RANK: " , mpi::Communicator::world().rank(),  "/" , mpi::Communicator::world().size());
// ==         output::print("KPOOL RANK: " , kpool_comm.rank(),  "/" , kpool_comm.size());
// ==         output::print(" BAND RANK: " , band_comm.rank(),  "/" , band_comm.size());
// == 
// ==         for(int irank_ = 1; irank_ < mpi::Communicator::world().size(); ++irank_ ) {
// ==             int world_comm_rank = 10, kpool_comm_rank, kpool_comm_size, band_comm_rank, band_comm_size;
// ==             mpi::Communicator::world().receive( &world_comm_rank, irank_ );
// ==             mpi::Communicator::world().receive( &kpool_comm_rank, irank_ );
// ==             mpi::Communicator::world().receive( &kpool_comm_size, irank_ );
// ==             mpi::Communicator::world().receive( &band_comm_rank, irank_ );
// ==             mpi::Communicator::world().receive( &band_comm_size, irank_ );
// == 
// ==         //output::print("WORLD RANK: " , world_comm_rank,  "/" , mpi::Communicator::world().size());
// ==         //output::print("KPOOL RANK: " , kpool_comm_rank,  "/" , kpool_comm_size);
// ==         //output::print(" BAND RANK: " , band_comm_rank,  "/" , band_comm_size);
// ==         }
// ==     }
#endif
}



void finalize()
{  
#ifdef EDUS_MPI
    mpi::Communicator::finalize();
    output::print("MPI Communicator finalized!");
#endif
#ifdef EDUS_GPU
     if (cublas_handle) cublasDestroy(cublas_handle);
#endif
    time_t now = time(0);
    char* dt = ctime(&now);
    std::stringstream ss;
    ss << "Execution finished: " << dt << std::endl;
    auto end_str = ss.str();
    end_str.erase(std::remove(end_str.begin(), end_str.end(), '\n'), end_str.end());
    output::print(end_str);
}
