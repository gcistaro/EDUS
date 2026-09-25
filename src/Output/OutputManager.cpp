#include <filesystem>
#include "Output/OutputManager.hpp"
#include "GlobalFunctions.hpp"

void OutputManager::initialize(const OutputParameters& parameters__,
                               const nlohmann::json& input_dict__,
                               const GridStructure& gridstructure__,
                               const parallel::Decomposition& decomposition__,
                               const electron::System& system__,
                               electron::State& state__,
                               electron::MeanField& meanfield__,
                               Propagator& propagator__,
                               SetOfLaser& lasers__)
{
    PROFILE("OutputManager::initialize");
    parameters_    = parameters__;
    decomposition_ = &decomposition__;
    system_        = &system__;
    state_         = &state__;
    meanfield_     = &meanfield__;
    propagator_    = &propagator__;
    lasers_        = &lasers__;

    is_writer_ = ( mpi::Communicator::world().rank() == 0 );

    /* define the index of Rgrid where (0,0,0) is */
    int index_origin_global = gridstructure__.Rgrid()->find(Coordinate(0,0,0));
    has_origin_ = decomposition_->mpindex().is_local(index_origin_global);
    if( has_origin_ ) {
        index_origin_local_ = decomposition_->mpindex().glob1D_to_loc1D(index_origin_global);
    }

    auto& DMk = state_->DensityMatrix().get_Operator_k();
    temp_.initialize(k, DMk.get_nblocks(), DMk.get_nrows(), DMk.get_ncols());

    std::filesystem::create_directories(parameters_.directory);
    mpi::Communicator::world().barrier();

    /* print equilibrium density matrix in real space (one file per rank) */
    auto& aux = state_->aux_DM();
    auto& DM0R = system_->DM0().get_Operator(R);
    std::copy(DM0R.begin(), DM0R.end(), aux.get_Operator(R).begin());
    aux.lock_space(R);
    aux.print_Rdecay(path("DM"), system_->wannier_centers());

    if( is_writer_ ) {
        os_pop_.open(path("Population.txt"));
        os_pop_wannier_.open(path("Population_wannier.txt"));
        os_laser_.open(path("Laser.txt"));
        os_vectorpot_.open(path("Laser_A.txt"));
        os_time_.open(path("Time.txt"));
        os_velocity_.open(path("Velocity.txt"));
    }

#ifdef EDUS_HDF5
    initialize_h5(input_dict__, gridstructure__);
#endif
}

void OutputManager::print_recap() const
{
    auto to_str = [](bool b) { return std::string(b ? "True" : "False"); };
    output::title("OUTPUT");
    output::print("Directory                *", std::string(8, ' '), parameters_.directory);
    output::print("PrintResolution          *", parameters_.printresolution);
    output::print("PrintResolution(pulse)   *", parameters_.printresolution_pulse);
    output::print("toprint-> DMk_wannier    *", std::string(8, ' '), to_str(parameters_.print_DMk_wannier));
    output::print("toprint-> DMk_bloch      *", std::string(8, ' '), to_str(parameters_.print_DMk_bloch));
    output::print("toprint-> fullH          *", std::string(8, ' '), to_str(parameters_.print_fullH));
    output::print("toprint-> SelfEnergy     *", std::string(8, ' '), to_str(parameters_.print_SelfEnergy));
    output::stars();
}

void OutputManager::write(const double& time__)
{
    if ( is_print_step(time__, false) ) {
        state_->DensityMatrix().transfer_to(host);
        if( is_writer_ ) {
            os_time_ << time__ << std::endl;
            os_laser_ << (*lasers_)(time__).get("Cartesian");
            os_vectorpot_ << lasers_->VectorPotential(time__).get("Cartesian");
        }
        print_population(time__, BandGauge::bloch);
        print_population(time__, BandGauge::wannier);
        print_velocity(time__);
    }
#ifdef EDUS_HDF5
    if ( is_print_step(time__, true) ) {
        write_h5(time__);
    }
#endif
}

/// @brief Defines whether or not at the current time step time__ we print the txt files and the matrices in hdf5
/// @param time__ Current time in the simulation
/// @param use_sparse__ If false, we always use printresolution_pulse; if true we use it only when a laser is on,
/// otherwise printresolution. Mainly needed for the printing of big matrices when the laser is off
bool OutputManager::is_print_step(const double& time__, const bool& use_sparse__)
{
    int printresolution = parameters_.printresolution_pulse;
    if( use_sparse__ ) {
        printresolution = parameters_.printresolution;
        for (int ilaser = 0; ilaser < lasers_->size(); ++ilaser) {
            if (time__ > (*lasers_)[ilaser].get_InitialTime() + 1.e-07 && time__ < (*lasers_)[ilaser].get_FinalTime() + 1.e-07) {
                printresolution = parameters_.printresolution_pulse;
                break;
            }
        }
    }
    return (int(round(time__ / propagator_->resolution_time())) % printresolution == 0);
}

void OutputManager::copy_DM_to_aux(const double& time__)
{
    auto& DM = state_->DensityMatrix();
    auto& aux = state_->aux_DM();
    aux.set_processor(host);
    std::copy(DM.get_Operator(DM.get_space()).begin(),
              DM.get_Operator(DM.get_space()).end(),
              aux.get_Operator(DM.get_space()).begin());
    aux.lock_space(DM.get_space());
    aux.lock_gauge(DM.get_bandgauge());

    if( propagator_->parameters().peierls ) {
        propagator_->apply_peierls_phase(aux, time__, -1);
    }
}

/// @brief Prints the population for each band (bloch gauge) or orbital (wannier gauge):
/// @f[
/// P_n(t) = \frac{1}{N}\sum_{\textbf{k}} \rho_{nn}(\textbf{k}) = \rho_{nn}(\textbf{R}=0)
/// @f]
/// For filled bands in the bloch gauge we print the population of holes, @f$ 1-P_n(t) @f$
void OutputManager::print_population(const double& time__, const BandGauge& bandgauge__)
{
    copy_DM_to_aux(time__);
    auto& aux = state_->aux_DM();
    aux.go_to(bandgauge__);
    aux.go_to_R();

    int num_bands = aux.get_Operator_R().get_nrows();
    std::vector<double> population(num_bands, 0.);
    if( has_origin_ ) {
        for (int ibnd = 0; ibnd < num_bands; ibnd++) {
            population[ibnd] = aux.get_Operator_R()(index_origin_local_, ibnd, ibnd).real();
        }
    }
#ifdef EDUS_MPI
    /* only one rank has R=0: the sum brings its values to rank 0 */
    auto& comm = decomposition_->kpool_comm();
    MPI_Reduce( comm.rank() == 0 ? MPI_IN_PLACE : population.data(), population.data(),
                num_bands, MPI_DOUBLE, MPI_SUM, 0, comm.communicator() );
#endif

    if( is_writer_ ) {
        auto& os = (bandgauge__ == wannier) ? os_pop_wannier_ : os_pop_;
        for (int ibnd = 0; ibnd < num_bands; ibnd++) {
            if( bandgauge__ == bloch && ibnd < parameters_.filledbands ) {
                os << std::setw(30) << std::setprecision(14) << 1. - population[ibnd];
            }
            else {
                os << std::setw(30) << std::setprecision(14) << population[ibnd];
            }
            os << " ";
        }
        os << std::endl;
    }
    aux.set_processor(propagator_->processor());
}

/// @brief Calculates the mean value of the velocity operator and prints it in Velocity.txt.
/// The equation implemented is:
/// @f[
/// \textbf{v}(t) = \frac{1}{N_k}\sum_{\textbf{k}} \text{Tr}(\rho(\textbf{k}) \textbf{v}(\textbf{k}))
/// @f]
/// Everything is conveniently done in k space.
void OutputManager::print_velocity(const double& time__)
{
    copy_DM_to_aux(time__);
    auto& aux = state_->aux_DM();
    aux.go_to(Space::k);
    auto& DMK = aux.get_Operator(Space::k);

    std::array<std::complex<double>, 3> v = { 0., 0., 0. };
    for (auto ix : { 0, 1, 2 }) {
        temp_.fill(0.);
        multiply(temp_, 1. + im * 0., system_->Velocity()[ix].get_Operator_k(), DMK);
        for (int ik = 0; ik < temp_.get_nblocks(); ++ik) {
            for (int ibnd = 0; ibnd < temp_.get_nrows(); ++ibnd) {
                v[ix] += temp_[ik](ibnd, ibnd);
            }
        }
    }
#ifdef EDUS_MPI
    auto& comm = decomposition_->kpool_comm();
    MPI_Reduce( comm.rank() == 0 ? MPI_IN_PLACE : v.data(), v.data(),
                3, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, 0, comm.communicator() );
#endif
    for (auto ix : { 0, 1, 2 }) {
        v[ix] /= DMK.get_MeshGrid()->get_TotalSize();
    }

    if( is_writer_ ) {
        for (auto ix : { 0, 1, 2 }) {
            os_velocity_ << std::setw(20) << std::setprecision(8) << v[ix].real();
            os_velocity_ << std::setw(20) << std::setprecision(8) << v[ix].imag();
        }
        os_velocity_ << std::endl;
    }
    aux.set_processor(propagator_->processor());
}

#ifdef EDUS_HDF5
void OutputManager::initialize_h5(nlohmann::json dict__, const GridStructure& gridstructure__)
{
    /* add to the input some information on the system */
    auto& kgrid = *gridstructure__.kgrid();
    dict__["num_bands"] = system_->num_bands();
    dict__["num_kpoints"] = kgrid.get_TotalSize();
    mdarray<double, 2> bare_k({ kgrid.get_TotalSize(), 3 });
    for (int ik = 0; ik < kgrid.get_TotalSize(); ++ik) {
        for (auto& ix : { 0, 1, 2 }) {
            bare_k(ik, ix) = kgrid[ik].get(LatticeVectors(k))[ix];
        }
    }
    dict__["kpoints"] = bare_k;
    dict__["A"] = Coordinate::get_Basis(LatticeVectors(R)).get_M();
    dict__["B"] = Coordinate::get_Basis(LatticeVectors(k)).get_M();
    dict__["comm_size"] = decomposition_->kpool_comm().size();
    /* each laser is given in input through frequency or wavelength, in any units: 
       we store both in a.u., as the times in the file */
    for (int ilaser = 0; ilaser < lasers_->size(); ++ilaser) {
        auto& laser_dict = dict__["lasers"][ilaser];
        laser_dict["frequency"] = (*lasers_)[ilaser].get_Omega();
        laser_dict["frequency_units"] = "auenergy";
        laser_dict["wavelength"] = (*lasers_)[ilaser].get_Lambda();
        laser_dict["wavelength_units"] = "aulength";
    }

    std::string name = path("output.h5");
    if (!file_exists(name)) {
        HDF5_tree(name, hdf5_access_t::truncate);
    }
    HDF5_tree fout(name, hdf5_access_t::truncate);
    dump_json_in_h5(dict__, name);
    fout.create_node(nodename::fullH);
    fout.create_node(nodename::DMk);
    fout.create_node(nodename::DMk_bloch);
    fout.create_node(nodename::SelfEnergy);
    fout.create_node(nodename::time_au);
    mpi::Communicator::world().barrier();
}

void OutputManager::write_h5(const double& time__)
{
    auto& DM = state_->DensityMatrix();
    auto& H = state_->H();

    std::string name = path("output.h5");
    std::stringstream node;
    node << index_h5_;
    index_h5_++;

    HDF5_tree fout(name, hdf5_access_t::read_write);

    if( parameters_.print_DMk_wannier ) {
        DM.go_to_k(false);
        DM.get_Operator_k().write_h5(name, nodename::DMk, node.str(), decomposition_->kpool_comm());
    }
    if( parameters_.print_DMk_bloch ) {
        DM.go_to_k(false);
        DM.go_to_bloch();
        DM.get_Operator_k().write_h5(name, nodename::DMk_bloch, node.str(), decomposition_->kpool_comm());
        DM.go_to_wannier();
    }
    if( parameters_.print_SelfEnergy ) {
        DM.go_to_R(true);
        H.go_to_R(false);
        H.get_Operator_R().fill(0.);
        meanfield_->self_energy(H, DM, system_->DM0());
        H.go_to_k(true);
        H.get_Operator_k().write_h5(name, nodename::SelfEnergy, node.str(), decomposition_->kpool_comm());
        DM.go_to_k(false);
    }
    if( parameters_.print_fullH ) {
        /* H is computed in R: bring it to k before writing */
        propagator_->ipa_hamiltonian(time__);
        H.go_to_k();
        H.get_Operator_k().write_h5(name, nodename::fullH, node.str(), decomposition_->kpool_comm());
    }
    fout[nodename::time_au].write(node.str(), time__);
}
#endif
