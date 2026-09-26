#include "mdContainers/mdContainers.hpp"

void axpby_gpu(std::complex<double>* Output, 
               const std::complex<double> FirstScalar, 
               const std::complex<double>* FirstAddend, 
               const std::complex<double> SecondScalar, 
               const std::complex<double>* SecondAddend,
               int N);

template <typename T, typename Scalar_T>
void axpby_cpu(T& Output, const Scalar_T& FirstScalar, const T& FirstAddend, const Scalar_T& SecondScalar, const T& SecondAddend)
{
    #pragma omp parallel for
    for( int i=0; i<Output.end()-Output.begin(); ++i) {
        *(Output.begin()+i) = FirstScalar*(*(FirstAddend.begin()+i)) + SecondScalar*(*(SecondAddend.begin()+i));
    }
}

template <typename T, typename Scalar_T>
void axpby(T& Output, 
           const Scalar_T& FirstScalar, const T& FirstAddend, 
           const Scalar_T& SecondScalar, const T& SecondAddend,
           const Processor& proc__=host)
{
    assert(Output.end() - Output.begin() == FirstAddend.end() - FirstAddend.begin());
    assert(FirstAddend.end() - FirstAddend.begin() == SecondAddend.end() - SecondAddend.begin());
///== cout debug  std::cout << "from axpby " << Output.end()-Output.begin() << " " << FirstScalar << " " << SecondScalar << std::endl;

///== cout debug  std::cout << "from axpby processor:  " << (proc__ == host ? "host" : "device") << std::endl;
#ifdef EDUS_GPU
    if(proc__ == device) {
        axpby_gpu(Output.data(device), FirstScalar, FirstAddend.data(device), SecondScalar, SecondAddend.data(device),
                  Output.end()-Output.begin());
        Output.set_processor(Processor::device);
        return;
    }
#endif
    axpby_cpu(Output, FirstScalar, FirstAddend, SecondScalar, SecondAddend);
///== cout debug    std::cout << "axpby. max Output: " << *max(Output) << std::endl;
}
