//this function must be called at the beginning of any miniapp to allow for 
//initialization of EDUS important quantities, such as:
//- header printing with git infos
//- MPI_COMM_WORLD
//- fftw initialization
#ifndef INITIALIZE_HPP
#define INITIALIZE_HPP

#include <sstream>
#include <fstream>
#include <memory>

#include "omp.h"
#include "core/print_header.hpp"
#include "core/mpi/Communicator.hpp"
#ifdef EDUS_MPI
#include "fftw3-mpi.h"
#endif
#ifdef EDUS_GPU
#include "cublas_v2.h"
#endif

#ifdef EDUS_MPI
extern std::unique_ptr<mpi::Communicator> kpool_comm;
extern std::unique_ptr<mpi::Communicator> band_comm;
extern int NumberKpools;
#endif

#ifdef EDUS_GPU
extern cublasHandle_t cublas_handle;
#endif

//== #define variable(x)  (#x)

void initialize();
void finalize();
#endif