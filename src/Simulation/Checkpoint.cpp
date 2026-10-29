#include <filesystem>
#include <fstream>
#include <iomanip>
#include "Simulation/Simulation.hpp"

/* Checkpoint of the time propagation, for a restart with RK (single step: no history of the time stepper).
   The state at the end of a run is:
   - the propagated variables: density matrix (k and R components, with their space and gauge), coordinates of the
     lattice and, with the dynamic reference, rho_BO; these are distributed among the ranks, so each rank writes its
     own blocks in rank_<r>.bin and the restart needs the same number of ranks;
   - the time, the vector potential of each laser (integrated in time from the last call) and the integrals of the
     energy balance, in header.json (rank 0).
   The static rho_BO, the mean field and the force are rebuilt from these at each stage. */

namespace {

std::string rank_file(const std::string& directory__, const int rank__)
{
    return directory__ + "/rank_" + std::to_string(rank__) + ".bin";
}

void write_block(std::ofstream& os__, const BlockMatrix<std::complex<double>>& M__)
{
    long size = M__.get_TotalSize();
    os__.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os__.write(reinterpret_cast<const char*>(M__.data()), size * sizeof(std::complex<double>));
}

void read_block(std::ifstream& is__, BlockMatrix<std::complex<double>>& M__)
{
    long size;
    is__.read(reinterpret_cast<char*>(&size), sizeof(size));
    if( size != M__.get_TotalSize() ) {
        throw std::runtime_error("Checkpoint: wrong size of a block (" + std::to_string(size) + " instead of " +
                                 std::to_string(M__.get_TotalSize()) + "): same grid, bands and MPI ranks needed\n");
    }
    is__.read(reinterpret_cast<char*>(M__.data()), size * sizeof(std::complex<double>));
}

void write_operator(std::ofstream& os__, const Operator<std::complex<double>>& O__)
{
    int flags[2] = { int(O__.get_space()), int(O__.get_bandgauge()) };
    os__.write(reinterpret_cast<const char*>(flags), sizeof(flags));
    write_block(os__, O__.get_Operator(k));
    write_block(os__, O__.get_Operator(R));
}

} // namespace

void Simulation::write_checkpoint(const std::string& directory__)
{
    PROFILE("Simulation::write_checkpoint");
    if( ctx_->cfg().solver() != "RK" ) {
        output::print("WARNING: checkpoint only with RK (AB needs its history): not written");
        return;
    }
    auto& comm = mpi::Communicator::world();
    if( comm.rank() == 0 ) {
        std::filesystem::create_directories(directory__);
    }
    comm.barrier();

    auto& y = electrons_state_.variables();
    y.rho.transfer_to(host);
    {
        std::ofstream os(rank_file(directory__, comm.rank()), std::ios::binary);
        write_operator(os, y.rho);
        if( y.with_reference ) {
            write_operator(os, y.rho_bo);
        }
    }

    if( comm.rank() == 0 ) {
        nlohmann::json header;
        header["time"] = propagator_.current_time();
        header["time_fs"] = Convert(propagator_.current_time(), AuTime, FemtoSeconds);
        header["dt"] = propagator_.time_step();
        header["initial_time"] = ctx_->cfg().initialtime();
        header["step"] = propagator_.current_step();
        header["grid"] = ctx_->cfg().grid();
        header["num_ranks"] = comm.size();
        header["with_reference"] = y.with_reference;
        if( propagator_.lattice() ) {
            std::vector<double> lattice;
            for( auto& x : y.lattice ) {
                lattice.push_back(x.real());
                lattice.push_back(x.imag());
            }
            header["lattice"] = lattice;
            header["lattice_shape"] = { y.lattice.num_qpoints(), y.lattice.num_modes() };
        }
        auto lasers = nlohmann::json::array();
        for( int ilaser = 0; ilaser < setoflaser_.size(); ++ilaser ) {
            auto& A = setoflaser_[ilaser].get_A().get("Cartesian");
            lasers.push_back({ { "A", { A[0], A[1], A[2] } },
                               { "previous_time", setoflaser_[ilaser].get_previous_Time() } });
        }
        header["lasers"] = lasers;
        auto s = output_.restart_state();
        header["energy"] = { { "work", s.work }, { "last_power", s.last_power },
                             { "second_last_power", s.second_last_power }, { "last_time", s.last_time },
                             { "initial_energy", s.initial_energy }, { "num_energy_steps", s.num_energy_steps },
                             { "index_h5", s.index_h5 }, { "power_integral", s.power_integral },
                             { "power_last", s.power_last }, { "power_second_last", s.power_second_last } };
        std::ofstream os(directory__ + "/header.json");
        os << std::setprecision(17) << header.dump(2) << std::endl;
    }
    comm.barrier();
    output::print("-> checkpoint written in ", directory__, " at t = ",
                  Convert(propagator_.current_time(), AuTime, FemtoSeconds), " fs");
}

long Simulation::read_checkpoint_step(const std::string& directory__)
{
    std::ifstream is(directory__ + "/header.json");
    if( !is ) {
        throw std::runtime_error("Checkpoint: cannot open " + directory__ + "/header.json\n");
    }
    auto header = nlohmann::json::parse(is);
    if( header["num_ranks"].get<int>() != mpi::Communicator::world().size() ) {
        throw std::runtime_error("Checkpoint: written with " + std::to_string(header["num_ranks"].get<int>()) +
                                 " MPI ranks, restart with the same number\n");
    }
    if( std::abs(header["dt"].get<double>() - ctx_->cfg().dt()) > 1.e-12 ) {
        throw std::runtime_error("Checkpoint: dt differs from the input\n");
    }
    if( header["grid"].get<std::array<int, 3>>() != ctx_->cfg().grid() ) {
        throw std::runtime_error("Checkpoint: grid differs from the input\n");
    }
    if( ctx_->cfg().solver() != "RK" ) {
        throw std::runtime_error("Checkpoint: restart only with RK\n");
    }
    if( std::abs(header["initial_time"].get<double>() - ctx_->cfg().initialtime()) > 1.e-12 ) {
        throw std::runtime_error("Checkpoint: initialtime differs from the input (keep the one of the first run)\n");
    }
    return header["step"].get<long>();
}

void Simulation::restore_checkpoint(const std::string& directory__)
{
    PROFILE("Simulation::restore_checkpoint");
    auto& comm = mpi::Communicator::world();
    std::ifstream ih(directory__ + "/header.json");
    auto header = nlohmann::json::parse(ih);
    auto& y = electrons_state_.variables();
    if( header["with_reference"].get<bool>() != y.with_reference ) {
        throw std::runtime_error("Checkpoint: adiabatic reference (static/dynamic) differs from the input\n");
    }
    if( header.contains("lattice") != (propagator_.lattice() != nullptr) ) {
        throw std::runtime_error("Checkpoint: phonons enabled in only one of checkpoint and input\n");
    }

    auto read_operator = [](std::ifstream& is__, Operator<std::complex<double>>& O__)
    {
        int flags[2];
        is__.read(reinterpret_cast<char*>(flags), sizeof(flags));
        read_block(is__, O__.get_Operator(k));
        read_block(is__, O__.get_Operator(R));
        O__.space = Space(flags[0]);
        O__.bandgauge = BandGauge(flags[1]);
    };
    std::ifstream is(rank_file(directory__, comm.rank()), std::ios::binary);
    if( !is ) {
        throw std::runtime_error("Checkpoint: cannot open " + rank_file(directory__, comm.rank()) + "\n");
    }
    read_operator(is, y.rho);
    if( y.with_reference ) {
        read_operator(is, y.rho_bo);
    }

    if( propagator_.lattice() ) {
        auto shape = header["lattice_shape"].get<std::vector<int>>();
        if( shape[0] != y.lattice.num_qpoints() || shape[1] != y.lattice.num_modes() ) {
            throw std::runtime_error("Checkpoint: number of phonon modes differs from the input\n");
        }
        auto lattice = header["lattice"].get<std::vector<double>>();
        int i = 0;
        for( auto& x : y.lattice ) {
            x = std::complex<double>(lattice[2 * i], lattice[2 * i + 1]);
            ++i;
        }
    }

    auto& lasers = header["lasers"];
    if( int(lasers.size()) != setoflaser_.size() ) {
        throw std::runtime_error("Checkpoint: number of lasers differs from the input\n");
    }
    for( int ilaser = 0; ilaser < setoflaser_.size(); ++ilaser ) {
        auto A = lasers[ilaser]["A"].get<std::vector<double>>();
        Coordinate A_cart(A[0], A[1], A[2]);
        setoflaser_[ilaser].set_A_state(A_cart, lasers[ilaser]["previous_time"].get<double>());
    }

    auto& e = header["energy"];
    OutputManager::RestartState s;
    s.work = e["work"]; s.last_power = e["last_power"]; s.second_last_power = e["second_last_power"];
    s.last_time = e["last_time"]; s.initial_energy = e["initial_energy"];
    s.num_energy_steps = e["num_energy_steps"]; s.index_h5 = e["index_h5"];
    s.power_integral = e["power_integral"].get<std::array<double, 4>>();
    s.power_last = e["power_last"].get<std::array<double, 4>>();
    s.power_second_last = e["power_second_last"].get<std::array<double, 4>>();
    output_.set_restart_state(s);
    propagator_.set_step(header["step"].get<long>());
    output::print("-> restarting from ", directory__, " at t = ", header["time_fs"].get<double>(), " fs");
}
