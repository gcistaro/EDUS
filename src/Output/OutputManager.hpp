#ifndef OUTPUT_MANAGER_HPP
#define OUTPUT_MANAGER_HPP

#include <array>
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
/// - txt files: Time, Laser, Laser_A, Population (bloch and wannier gauge), Velocity, Energy and, with phonons, Lattice
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
        /// Workspace for the energy balance: self energy and its gradient
        Operator<std::complex<double>> sigma_;
        Operator<std::complex<double>> grad_sigma_;
        /// rho_BO of the static adiabatic reference (with phonons), for the force on the lattice
        Operator<std::complex<double>> rho_bo_;
        /// Workspaces of EnergyTerms.txt (with phonons): Sigma[rho_ref - rho0] (R and k), and in k the Hamiltonian,
        /// d(rho)/dt and u.(g_e - g_b)
        Operator<std::complex<double>> sigma_ref_;
        BlockMatrix<std::complex<double>> hk_, rhodot_, eta_g_;
        /// Integrals in time (3-point rule, like W) of the powers that break the energy balance, see EnergyTerms.txt,
        /// and their values at the last two print steps
        std::array<double, 4> power_integral_{}, power_last_{}, power_second_last_{};
        /// Energy balance: work done by the field up to the last print step, power at the last two print steps
        /// (for the 3-point rule) and time of the last one, number of print steps done
        double work_ = 0.;
        double last_power_ = 0.;
        double second_last_power_ = 0.;
        double last_time_ = 0.;
        int num_energy_steps_ = 0;
        double initial_energy_ = 0.;

        std::ofstream os_time_;
        std::ofstream os_laser_;
        std::ofstream os_vectorpot_;
        std::ofstream os_pop_;
        std::ofstream os_pop_wannier_;
        std::ofstream os_velocity_;
        std::ofstream os_velocity_sigma_;
        std::ofstream os_energy_;
        std::ofstream os_lattice_;
        std::ofstream os_modes_;
        std::ofstream os_force_check_;
        std::ofstream os_energy_terms_;

        std::string path(const std::string& filename__) const { return parameters_.directory + "/" + filename__; }
        /// True if at time__ we print (use_sparse__=true: resolution depends on whether a laser is on)
        bool is_print_step(const double& time__, const bool& use_sparse__);
        /// Copies the density matrix in aux_DM, removing the Peierls phase if needed
        void copy_DM_to_aux(const double& time__);
        void print_population(const double& time__, const BandGauge& bandgauge__);
        /// Velocity (with the self energy and the lattice) in Velocity.txt, energy balance in Energy.txt and
        /// coordinates, forces and energy of the lattice in Lattice.txt
        void print_velocity_energy(const double& time__);
        void initialize_h5(nlohmann::json dict__, const GridStructure& gridstructure__);
        void write_h5(const double& time__);

    public:
        /// Variables integrated in time by the energy balance (and index of the h5 file), saved in the checkpoint
        struct RestartState
        {
            double work, last_power, second_last_power, last_time, initial_energy;
            int num_energy_steps, index_h5;
            std::array<double, 4> power_integral, power_last, power_second_last;
        };
        RestartState restart_state() const
        {
            return { work_, last_power_, second_last_power_, last_time_, initial_energy_, num_energy_steps_, index_h5_,
                     power_integral_, power_last_, power_second_last_ };
        }
        void set_restart_state(const RestartState& s__)
        {
            work_ = s__.work; last_power_ = s__.last_power; second_last_power_ = s__.second_last_power;
            last_time_ = s__.last_time; initial_energy_ = s__.initial_energy; num_energy_steps_ = s__.num_energy_steps;
            index_h5_ = s__.index_h5; power_integral_ = s__.power_integral; power_last_ = s__.power_last;
            power_second_last_ = s__.power_second_last;
        }
        /// Output directory
        const std::string& directory() const { return parameters_.directory; }

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
