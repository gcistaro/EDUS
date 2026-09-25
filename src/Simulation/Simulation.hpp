#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "ConvertUnits.hpp"
#include "Electrons/System.hpp"
#include "Electrons/State.hpp"
#include "Parallel/Decomposition.hpp"
#include "Laser/Laser.hpp"
#include "Laser/LaserFactory.hpp"
//#include "ostream.hpp"
#include "kGradient/kGradient.hpp"
#include "MeanField/MeanField.hpp"
#include "Propagator/Propagator.hpp"
#include "Output/OutputManager.hpp"
//#include "initialize.hpp"
//#include "Json/json.hpp"
#include "InputVariables/simulation_parameters.hpp"

void print_bandstructure(const std::vector<std::vector<double>>& bare_kpath, Operator<std::complex<double>> Hamiltonian);


/// @brief This class contains all the variables that are used in the simulations
class Simulation
{
    private:
        /// @brief Object containing all the parameters that needs to be set; they get read or 
        /// simply they get some default values
        std::shared_ptr<Simulation_parameters> ctx_;
        /// @brief Electron system, containing immutable operators in time propagation
        electron::System electrons_;
        /// @brief Electron state, containing variables that are propagated in time, like the density matrix
        electron::State electrons_state_;
        /// @brief Grid structure, containing the grids in k and R space
        GridStructure gridstructure_;
        /// @brief Decomposition of the grids, containing the MPI decomposition of the grids
        parallel::Decomposition decomposition_;
        /// @brief Interaction state, containing self-energies (el-el and el-ph)
        //TODO InteractionState interaction_state_;
        /// @brief Function to convert all the units in the input json to atomic units, directly usable by EDUS
        void convert_input_au();
        /// Self-energy up to HSEX
        electron::MeanField meanfield_;
        /// Object to deal with the gradient term. More details in the class
        kGradient kgradient_;
        /// Total laser acting on the system, as a sum over the single lasers
        SetOfLaser setoflaser_;
        /// Time propagation of the density matrix
        Propagator propagator_;
        /// Writes the outputs during the propagation
        OutputManager output_;
        /// Recap of the simulation: each component prints its own section
        void print_recap();

    public:
        
    /// Dynamical variables that are propagated in time, like the density matrix
//         
//         std::array<Operator<std::complex<double>>, 3> Velocity_;
//         /// Eigenvalues of the Hamiltonian, grouped like: Band_energies[ik](ibnd)
//         std::vector<mdarray<double,1>> Band_energies_;
//         
         Simulation(){};
         Simulation(std::shared_ptr<Simulation_parameters>& ctx__);
//         void print_grids();
//         void Calculate_Velocity();
         void Propagate();
//         void print_recap();
//         void PrintWannier();
//         int get_it(const double& time__) const;
//         double jacobian(const Matrix<double>& A__) const;
//         void OpenGap();
//         void pdos();
//         void Print_DeltaRho(const double& it__);
//         
// 
//         template <typename Scalar_T>
//         friend void axpby(Operator<std::complex<double>>& Output__, 
//                     const Scalar_T& FirstScalar__, 
//                     const Operator<std::complex<double>>& FirstAddend__, 
//                     const Scalar_T& SecondScalar__, 
//                     const Operator<std::complex<double>>& SecondAddend__);
// 
};

// == template<typename T>
// == std::complex<double> Trace(BlockMatrix<T>& O1__, BlockMatrix<T>& O2__)
// == {
// ==     //calculate trace of product of two operators in R space as \sum_(n,R) (O1(R)O2(-R))_(nn)
// ==     auto& ci = MeshGrid::ConvolutionIndex[{Operator<std::complex<double>>::MeshGrid_Null->get_id(), 
// ==                                               O1__.get_MeshGrid()->get_id(), 
// ==                                               O2__.get_MeshGrid()->get_id()}];
// ==     if(ci.get_Size(0) == 0 ){
// ==         MeshGrid::Calculate_ConvolutionIndex(*(Operator<std::complex<double>>::MeshGrid_Null), 
// ==                                                  *(O1__.get_MeshGrid()), 
// ==                                                  *(O2__.get_MeshGrid()));
// ==     }
// == 
// ==     std::complex<double> Trace = 0.;
// ==     for(int iblock=0; iblock<O1__.get_nblocks(); ++iblock){
// ==         for(int irow=0; irow<O1__.get_nrows(); irow++){
// ==             for(int icol=0; icol<O1__.get_ncols(); icol++){   
// ==                 Trace += O1__[ci(0,iblock)](irow, icol)*O2__[iblock](icol, irow);
// ==             }
// ==         }
// ==     }
// ==     return Trace;
// == }

template<typename T>
std::vector<std::complex<double>> TraceK(BlockMatrix<T>& O__, mpi::Communicator& kcomm)
{
    //calculate trace over k P(n) = \sum_k O(k)_{nn}
    std::vector<std::complex<double>> TraceK(O__.get_nrows(), 0.);
    for(int iblock=0; iblock<O__.get_nblocks(); ++iblock){
        for(int ibnd=0; ibnd<O__.get_nrows(); ++ibnd){
            TraceK[ibnd] += O__[iblock](ibnd, ibnd);
        }
    }
#ifdef EDUS_MPI
    std::vector<std::complex<double>> TraceK_reduced(O__.get_nrows(), 0.);
    kcomm.reduce(&TraceK[0], &TraceK_reduced[0], TraceK.size(), MPI_SUM, 0);
    return TraceK_reduced;
#else
    return TraceK;
#endif
}


#endif
