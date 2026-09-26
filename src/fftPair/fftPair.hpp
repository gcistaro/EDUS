#ifndef FOURIERTRANSFORM_HPP
#define FOURIERTRANSFORM_HPP

#include <vector>
#include <memory>
#include <complex>

#ifdef EDUS_MPI
#include <fftw3-mpi.h>
#else 
#include <fftw3.h>
#endif
#ifdef EDUS_GPU
#include <cufft.h>
#endif 

#include "mdContainers/mdContainers.hpp"
#include "Constants.hpp"
//NOTE WARNING!!!
//dft: By construction, this class does not own the objects Array_x and Array_k

//Dimension of Array_x and Array_k: [howmany x TotalSize of FFt]
class FourierTransform
{
    private:
        mdarray<std::complex<double>, 2>* Array_x; // Only the local array
        mdarray<std::complex<double>, 2>* Array_k; // Only the local array
        std::vector< std::vector<double> > Mesh;
        std::vector< std::vector<double> > Mesh_fft;
        int dim;
        std::vector<int> Dimensions;
        int TotalSize;
        double SqrtTotalSize;
        int istride = 1;
        int ostride = 1;
        int idist;
        int odist;
        int* inembed = nullptr;
        int* onembed = nullptr;
        int howmany = 1;
#ifdef EDUS_GPU
        cufftHandle MyPlan_device;
#endif
        fftw_plan MyPlan_FWD;
        fftw_plan MyPlan_BWD;

        std::string tagname="";
        bool IsFFT = false;
        bool destruct = true;
    public:
        FourierTransform(){};
        FourierTransform(const FourierTransform& ) = delete;
        FourierTransform& operator=(const FourierTransform& ) = delete;
        FourierTransform(FourierTransform&& ) = default;
        FourierTransform& operator=( FourierTransform&& ) = default;

        FourierTransform(mdarray<std::complex<double>, 2>& Array_x__, 
                         mdarray<std::complex<double>, 2>& Array_k__, 
                         const std::vector<int>& Dimensions__);

        void initialize(mdarray<std::complex<double>, 2>& Array_x__, 
                        mdarray<std::complex<double>, 2>& Array_k__, 
                        const std::vector<int>& Dimensions__, const std::string tagname_="");
        void initialize(mdarray<std::complex<double>, 2>& Array_x__, 
                        const std::vector<std::vector<double>>& Mesh__);
        void fft(const int& sign, const Processor& proc__=host);
        std::complex<double> dft(const std::vector<double>& Point, const int& h, const int& sign); 
        mdarray<std::complex<double>, 2> dft(const std::vector<std::vector<double>>& ArrayOfPoints, const int& sign);
        inline const mdarray<std::complex<double>, 2>& get_Array_k() const { return (*Array_k);};
        inline const mdarray<std::complex<double>, 2>& get_Array_x() const { return (*Array_x);};
        void normalize(mdarray<std::complex<double>,2>& array__);

        ~FourierTransform();
};

#endif