cmake_minimum_required(VERSION 3.18)

project(EDUS VERSION 1.0)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_SHARED_LIBS ON)
set(FFTW_USE_STATIC_LIBS OFF)
set(BLA_STATIC OFF)

# =================================================================================
# Options
# =================================================================================
option(EDUS_PROFILE     "Profiling function timings"                           ON )
option(EDUS_MKL         "Enable MKL"                                           ON )
option(EDUS_MPI         "Enabling MPI"                                         ON )
option(EDUS_BATCHGEMM   "Enabling Batch Gemm for k matrix multiplication"      OFF)
option(EDUS_FFTWTHREADS "Enabling threads in fftw"                             OFF)                
option(EDUS_MKL_THREAD  "Enabling threads in mkl"                              OFF)                  
option(EDUS_HDF5        "Link hdf5 library for fast writing of big matrices"   OFF)
option(EDUS_GPU         "Compile with GPU support"                             OFF)
option(__DEBUG          ""                                                     OFF)

# =================================================================================
# Compile definitions
# =================================================================================
add_compile_definitions(
    $<$<BOOL:${EDUS_PROFILE}>:EDUS_PROFILE>
    $<$<BOOL:${EDUS_MKL}>:EDUS_MKL>
    $<$<BOOL:${EDUS_MPI}>:EDUS_MPI>
    $<$<BOOL:${EDUS_BATCHGEMM}>:EDUS_BATCHGEMM>
    $<$<BOOL:${EDUS_FFTWTHREADS}>:EDUS_FFTWTHREADS>
    $<$<BOOL:${EDUS_MKL_THREAD}>:EDUS_MKL_THREAD>
    $<$<BOOL:${EDUS_HDF5}>:EDUS_HDF5>
    $<$<BOOL:${EDUS_GPU}>:EDUS_GPU>
    $<$<BOOL:${__DEBUG}>:__DEBUG>
)

# =================================================================================
# Build type flags
# =================================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g -Wall)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()


# =================================================================================
# Custom module path
# =================================================================================
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")



# =================================================================================
# Git version + headers
# =================================================================================
find_program(GIT_EXE NAMES git)

string(TIMESTAMP EDUS_TIMESTAMP "%Y-%m-%d %H:%M:%S")
message("PROJECT_SOURCE_DIR:     ${PROJECT_SOURCE_DIR}")
if(DEFINED GIT_EXE AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(COMMAND git rev-parse HEAD
                  OUTPUT_VARIABLE EDUS_SHA
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND git describe --all
                  OUTPUT_VARIABLE EDUS_GIT_BRANCH
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "git hash ${EDUS_SHA}")
endif()

# =================================================================================
# Configure files
# =================================================================================
configure_file("${PROJECT_SOURCE_DIR}/src/core/githash.h.in" "${PROJECT_SOURCE_DIR}/src/core/githash.hpp" @ONLY)
configure_file("${PROJECT_SOURCE_DIR}/src/core/projectdir.h.in" "${PROJECT_SOURCE_DIR}/src/core/projectdir.hpp" @ONLY)
configure_file("${PROJECT_SOURCE_DIR}/ci-test/inputs/MoS2.json.in" "${PROJECT_SOURCE_DIR}/ci-test/inputs/MoS2.json" @ONLY)
configure_file("${PROJECT_SOURCE_DIR}/ci-test/inputs/input_NonEQ_Grad.json.in" "${PROJECT_SOURCE_DIR}/ci-test/inputs/input_NonEQ_Grad.json" @ONLY)
file(READ "${PROJECT_SOURCE_DIR}/src/InputVariables/input_schema.json" EDUS_INPUT_SCHEMA NEWLINE_CONSUME)
configure_file("${PROJECT_SOURCE_DIR}/src/InputVariables/input_schema.hpp.in"
               "${PROJECT_SOURCE_DIR}/src/InputVariables/input_schema.hpp"
               @ONLY)             

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)


# =================================================================================
# Source files
# =================================================================================
include_directories("${PROJECT_SOURCE_DIR}/src/")
set(_SOURCES src/initialize.cpp
             src/GlobalFunctions.cpp
             src/core/print_header.cpp
             src/ConvertUnits.cpp
             src/ostream.cpp
             src/StreamFile.cpp
             src/InputVariables/simulation_parameters.cpp
             src/Geometry/Matrix.cpp
             src/Geometry/Coordinate.cpp
             src/MeshGrid/MeshGrid.cpp
             src/Wannier/Wannier.cpp
             src/Wannier/ReadWannier.cpp
             src/Wannier/PrintWannier.cpp
             src/Laser/Laser.cpp
             src/fftPair/fftPair.cpp
             src/kGradient/kGradient.cpp
             src/DESolver/DESolver.cpp
             src/Simulation/Simulation.cpp
             src/core/profiler.cpp
             src/core/print_timing.cpp
             src/core/cmd_args/cmd_args.cpp
             src/core/rt_graph.cpp
             src/core/mpi/Communicator.cpp
             src/Coulomb/Coulomb.cpp
#             src/Screened_potential/Screened_potential.cpp
#             src/GreenFunction/GreenFunction.cpp
             src/ModelCoulomb/ModelCoulomb.cpp
             src/ModelCoulomb/Potentials.cpp
    )

if (EDUS_GPU)
    list(APPEND _SOURCES src/core/kernels/Calculate_TDHamiltonian.cu
                         src/core/kernels/Apply_Peierls_phase.cu
                         src/core/kernels/axpby.cu)
endif()

add_library(core SHARED ${_SOURCES} )


set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)

# =================================================================================
# CUDA
# =================================================================================

if(EDUS_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  
  set_target_properties(core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES 86                   
    #CUDA_SEPARABLE_COMPILATION ON
    #CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )  
  target_link_libraries(core PUBLIC CUDA::cudart CUDA::cufft CUDA::cublas)
endif()

# =================================================================================
# OpenMP
# =================================================================================
find_package(OpenMP)
set(OpenMP_USE_STATIC_LIBRARIES OFF)
target_link_libraries(core PUBLIC OpenMP::OpenMP_CXX)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# =================================================================================
# MPI
# =================================================================================
if(EDUS_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(core PUBLIC MPI::MPI_CXX)
endif()

# =================================================================================
# HDF5
# =================================================================================
if(EDUS_HDF5)
    set(HDF5_USE_STATIC_LIBRARIES OFF)
    set(HDF5_PREFER_PARALLEL OFF)
    find_package(HDF5 REQUIRED COMPONENTS C HL)
    message(STATUS "HDF5 LIBRARY: ${HDF5_LIBRARIES}")
    message(STATUS "HDF5 parallel: ${HDF5_IS_PARALLEL}")
    message(STATUS "HDF5 include directories: ${HDF5_INCLUDE_DIRS}")
    include_directories(${HDF5_INCLUDE_DIRS})
    target_link_libraries(core PUBLIC HDF5::HDF5)
endif()


# =================================================================================
# BLAS / LAPACK / MKL
# =================================================================================
if(EDUS_MKL)
  set(MKL_USE_STATIC_LIBRARIES OFF)
  set(MKL_ARCH intel64)

  if(EDUS_MKL_THREAD)
    set(MKL_THREADING intel_thread)
  else()
    set(MKL_THREADING sequential)
  endif()

  set(MKL_INTERFACE lp64)
  find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
  add_compile_options($<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  include_directories($<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(core PUBLIC $<LINK_ONLY:MKL::MKL>)
  message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")
else()
  set(BLA_VENDOR OpenBLAS)
  find_path(LAPACKE_INCLUDE_DIR
      NAMES lapacke.h
  )
  find_library(LAPACKE_LIBRARY
      NAMES lapacke
  )
  if(NOT LAPACKE_INCLUDE_DIR OR NOT LAPACKE_LIBRARY)
    message(FATAL_ERROR "LAPACKE not found. Install liblapacke-dev")
  endif()
  find_library(LAPACKE_LIBRARY lapacke)  
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  # Create an imported interface target for LAPACKE
  add_library(LAPACKE::LAPACKE UNKNOWN IMPORTED)
  set_target_properties(LAPACKE::LAPACKE PROPERTIES
    IMPORTED_LOCATION "${LAPACKE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LAPACKE_INCLUDE_DIR}"
  )

  # IMPORTANT: link only through targets
  target_link_libraries(core PUBLIC
    LAPACKE::LAPACKE
    BLAS::BLAS
    LAPACK::LAPACK
  )
endif()

# =================================================================================
# FFTW3
# =================================================================================
find_package(FFTW REQUIRED)
message(STATUS "FFTW_LIBRARIES: ${FFTW_LIBRARIES}")
message(STATUS "FFTW_INCLUDE_DIRS: ${FFTW_INCLUDE_DIRS}")

target_link_libraries(core PUBLIC ${FFTW_LIBRARIES} fftw3_mpi)
target_include_directories(core PUBLIC ${FFTW_INCLUDE_DIRS})





# add the executable
add_executable(EDUS src/EDUS.cpp)
target_link_libraries(EDUS PRIVATE core)


# Google test
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(test_fft tests/test_fft.cpp)
add_executable(test_batchfft tests/test_batchfft.cpp)
add_executable(test_batchgemm tests/test_batchgemm.cpp)
add_executable(test_axpby tests/test_axpby.cpp)
target_link_libraries(test_fft gtest gtest_main core)
target_link_libraries(test_batchfft gtest gtest_main core)
target_link_libraries(test_batchgemm gtest gtest_main core)
target_link_libraries(test_axpby gtest gtest_main core)
include(GoogleTest)
gtest_discover_tests(test_fft)
gtest_discover_tests(test_batchfft)
gtest_discover_tests(test_batchgemm)
gtest_discover_tests(test_axpby)

#add armadillo
#find_package(Armadillo REQUIRED)
#target_compile_definitions(EDUS PRIVATE ARMA_DONT_USE_WRAPPER)
#target_link_libraries(EDUS armadillo mkl_rt pthread m dl)


#speed executables
###add_executable(Matrix_mult_speed speed-apps/Matrix_multiplication.cpp)
###add_executable(BlockMatrix_mult_speed speed-apps/BlockMatrix_multiplication.cpp)
###add_executable(FourierTransform_speed speed-apps/FourierTransform.cpp)

#Executables for test various classes
#add_executable(Containers_test ci-test/test_mdarray.cpp) 
#add_executable(Matrix_test ci-test/test_matrix.cpp)
#add_executable(Eigen_test ci-test/test_eigen.cpp) 
#add_executable(DE_test ci-test/test_DEsolver.cpp)
#add_executable(DE2_test ci-test/test_DEsolver2.cpp)
#add_executable(fft-many_test ci-test/test_fft_many.cpp)
#add_executable(fft_test ci-test/test_fft_mpi.cpp)
#add_executable(Gradient_R_test ci-test/test_Gradient_R.cpp)
#add_executable(NonEQ ci-test/test_NEQ_H0.cpp)
#add_executable(NonEQ_Grad ci-test/test_NEQ_Grad.cpp)
#add_executable(Bandstructure_test ci-test/test_bandstructure.cpp)
#add_executable(Pseudoinv_test ci-test/test_pseudoinverse.cpp)
#add_executable(ScreenedPotential ci-test/test_Response.cpp)


install (TARGETS EDUS PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_BINDIR})
#build for ctest
enable_testing()
#add_test(NAME Containers_test COMMAND Containers_test)
#add_test(NAME Matrix_test COMMAND Matrix_test)
#add_test(NAME Eigen_test COMMAND Eigen_test)
#add_test(NAME DE_test COMMAND DE_test)
#add_test(NAME DE2_test COMMAND DE2_test)
#add_test(NAME Gradient_R_test COMMAND Gradient_R_test)
#add_test(NAME Bandstructure_test COMMAND 
#					bash -c "${CMAKE_CURRENT_BINARY_DIR}/Bandstructure_test ; \
#					python3 ${CMAKE_SOURCE_DIR}/ci-test/compare.py ${CMAKE_CURRENT_BINARY_DIR}/BANDSTRUCTURE.txt ${CMAKE_SOURCE_DIR}/ci-test/Reference/BANDSTRUCTURE.txt")
#add_test(NAME fft_test COMMAND fft_test)
#add_test(NAME NonEQ_H0 COMMAND NonEQ)
#add_test(NAME NonEQ_Grad COMMAND 
#            bash -c "mkdir ${CMAKE_SOURCE_DIR}/CTEST; 
#            rm -rf ${CMAKE_SOURCE_DIR}/CTEST/NonEQ_Grad/; \ 
#            mkdir ${CMAKE_SOURCE_DIR}/CTEST/NonEQ_Grad/; \ 
#            cd ${CMAKE_SOURCE_DIR}/CTEST/NonEQ_Grad/; \ 
#            ${CMAKE_CURRENT_BINARY_DIR}/NonEQ_Grad ${PROJECT_SOURCE_DIR}/ci-test/inputs/input_NonEQ_Grad.json"
#            )
#
add_test(NAME hBN_IPA COMMAND bash ${CMAKE_SOURCE_DIR}/ci-test/scripts/run.sh hBN_IPA)
add_test(NAME hBN_HSEX COMMAND bash ${CMAKE_SOURCE_DIR}/ci-test/scripts/run.sh hBN_HSEX)
############
############add_test(NAME MPI_hBN COMMAND
############          bash -c "mkdir ${CMAKE_SOURCE_DIR}/CTEST; 
############          rm -rf ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN/; \ 
############          mkdir ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN/; \ 
############          cd ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN/; \             
############          ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS
############          ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 10 ${CMAKE_CURRENT_BINARY_DIR}/EDUS  ${PROJECT_SOURCE_DIR}/ci-test/inputs/hBN.json | tee output; \
############          python3 ${CMAKE_SOURCE_DIR}/ci-test/compare.py ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN/Output/Population.txt ${PROJECT_SOURCE_DIR}/ci-test/outputs/hBN/Population.txt
############          python3 ${CMAKE_SOURCE_DIR}/ci-test/compare.py ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN/Output/Velocity.txt ${PROJECT_SOURCE_DIR}/ci-test/outputs/hBN/Velocity.txt;" 
############          )
############add_test(NAME MPI_hBN_Coulomb COMMAND 
############          bash -c "mkdir ${CMAKE_SOURCE_DIR}/CTEST; 
############          rm -rf ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN_coulomb/; \ 
############          mkdir ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN_coulomb/; \ 
############          cd ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN_coulomb/; \             
############          python3 ${PROJECT_SOURCE_DIR}/Postproces/RytovaKeldysh.py 40 40 1 ${PROJECT_SOURCE_DIR}/tb_models/hBN_gap7.25eV_a2.5A_tb.dat; \
############          python3 ${CMAKE_SOURCE_DIR}/ci-test/compare.py ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN_coulomb/RytovaKeldysh.txt ${PROJECT_SOURCE_DIR}/ci-test/outputs/hBN_coulomb/RytovaKeldysh.txt; \
############          ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 10 ${CMAKE_CURRENT_BINARY_DIR}/EDUS ${PROJECT_SOURCE_DIR}/ci-test/inputs/hBN_coulomb.json | tee output; \
############          python3 ${CMAKE_SOURCE_DIR}/ci-test/compare.py ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN_coulomb/HF.txt ${PROJECT_SOURCE_DIR}/ci-test/outputs/hBN_coulomb/HF.txt; \
############          python3 ${CMAKE_SOURCE_DIR}/ci-test/compare.py ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN_coulomb/Output/Population.txt ${PROJECT_SOURCE_DIR}/ci-test/outputs/hBN_coulomb/Population.txt; \
############          python3 ${CMAKE_SOURCE_DIR}/ci-test/compare.py ${CMAKE_SOURCE_DIR}/CTEST/MPI_hBN_coulomb/Output/Velocity.txt ${PROJECT_SOURCE_DIR}/ci-test/outputs/hBN_coulomb/Velocity.txt")
