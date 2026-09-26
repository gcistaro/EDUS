#ifndef OUTPUT_MANAGER_HPP
#define OUTPUT_MANAGER_HPP

#include <fstream>
#include "Json/json.hpp"
#include "GridStructure/GridStructure.hpp"
#include "Parallel/Decomposition.hpp"
#include "Electrons/System.hpp"
#include "Electrons/State.hpp"
#include "MeanField/MeanField.hpp"
#include "Propagator/Propagator.hpp"
#include "Laser/Laser.hpp"
#include "Output/OutputParameters.hpp"

/// @brief Decides when to print and writes all the outputs of the time propagation:
/// - txt files: Time, Laser, Laser_A, Population (bloch and wannier gauge), Velocity
/// - h5 file (if compiled with EDUS_HDF5): matrices selected in "toprint"
/// Only rank 0 writes the txt files; the quantities distributed among ranks are reduced to it.
/// The class does not own the physical objects, it only keeps pointers to them.
class OutputManager
{
    private:
        OutputParameters parameters_;
        const electron::System* system_ = nullptr;
        electron::State* state_ = nullptr;
        electron::MeanField* meanfield_ = nullptr;
        /// Needed to undo the Peierls phase and to compute H(t) for the h5 file
        Propagator* propagator_ = nullptr;
        /// Not const because the vector potential is integrated in time
        SetOfLaser* lasers_ = nullptr;
        const parallel::Decomposition* decomposition_ = nullptr;

        /// True for the rank that writes the txt files
        bool is_writer_ = false;
        /// True if the current rank has R=0 in its local R points
        bool has_origin_ = false;
        /// Local index of R=0, if has_origin_
        int index_origin_local_ = -1;
        /// Index of the next node in the h5 file
        int index_h5_ = 0;
        /// Workspace for the velocity
        BlockMatrix<std::complex<double>> temp_;

        std::ofstream os_time_;
        std::ofstream os_laser_;
        std::ofstream os_vectorpot_;
        std::ofstream os_pop_;
        std::ofstream os_pop_wannier_;
        std::ofstream os_velocity_;

        std::string path(const std::string& filename__) const { return parameters_.directory + "/" + filename__; }
        /// True if at time__ we print (use_sparse__=true: resolution depends on whether a laser is on)
        bool is_print_step(const double& time__, const bool& use_sparse__);
        /// Copies the density matrix in aux_DM, removing the Peierls phase if needed
        void copy_DM_to_aux(const double& time__);
        void print_population(const double& time__, const BandGauge& bandgauge__);
        void print_velocity(const double& time__);
        void initialize_h5(nlohmann::json dict__, const GridStructure& gridstructure__);
        void write_h5(const double& time__);

    public:
        OutputManager() = default;

        /// Creates the output directory, opens the files and prints the equilibrium density matrix
        void initialize(const OutputParameters& parameters__,
                        const nlohmann::json& input_dict__,
                        const GridStructure& gridstructure__,
                        const parallel::Decomposition& decomposition__,
                        const electron::System& system__,
                        electron::State& state__,
                        electron::MeanField& meanfield__,
                        Propagator& propagator__,
                        SetOfLaser& lasers__);

        void print_recap() const;
        /// Writes all the outputs at time__, if it is a printing step. To be called before each time step
        void write(const double& time__);
};

#endif // OUTPUT_MANAGER_HPP
