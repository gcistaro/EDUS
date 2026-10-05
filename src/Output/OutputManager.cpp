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
    /* workspace for the energy balance */
    sigma_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), DMk.get_nrows(), decomposition_->mpindex(), "Sigma_energy");
    grad_sigma_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), DMk.get_nrows(), decomposition_->mpindex(), "grad_Sigma_energy");

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
        os_energy_.open(path("Energy.txt"));
        auto* lattice = propagator_->lattice();
        os_energy_ << "# Energy balance of the electrons, Hartree per unit cell (one spin channel)\n"
                   << "# E_band = Tr[H0 (rho-rho0)], E_MF = 1/2 Tr[Sigma[rho-rho0] (rho-rho0)], E = E_band + E_MF"
                   << (lattice ? " + E_ph\n# E_ph = (E_lattice + E_coupling)/s: lattice and coupling (see Lattice.txt), divided by the spin degeneracy s\n" : "\n")
                   << "# (changes with respect to equilibrium, computed from rho-rho0 to keep all the digits)\n"
                   << "# W = work of the field = -int E(t).v(t) dt, v = <grad_k(H0+Sigma) + i[H0+Sigma, r]>\n"
                   << "# P = -E.v is the power; W is integrated with the trapezoidal rule (error O(dt^2))\n"
                   << "# Without decay, E(t) - E(0) = W(t) up to the error of the time integration of W\n"
                   << "#        time (a.u.)       E_band-E_band(0)                  E_MF              E - E(0)                     P                     W          E - E(0) - W\n";
        if( lattice ) {
            int nmodes = lattice->num_modes();
            os_lattice_.open(path("Lattice.txt"));
            bool dynamic = lattice->dynamic_reference();
            os_lattice_ << "# Lattice at q = 0, atomic units, per unit cell (all spin channels), adiabatic reference: "
                        << phonon::to_string(lattice->parameters().adiabatic_reference) << "\n"
                        << "# mu = 3*atom + direction (cartesian), u = displacement (bohr), du/dt (bohr/a.u.),\n"
                        << "# F = force of the electrons used in the dynamics, s/N sum_k Tr[g_mu (rho - " << (dynamic ? "rho_BO" : "rho0")
                        << ")] (Ha/bohr): the atoms feel -F\n"
                        << "# E_lattice = kinetic + harmonic energy (force constants used in the dynamics)\n"
                        << (dynamic ? "# E_coupling = u.F[rho - rho0] - E[rho_BO], E[rho_BO] = energy of rho_BO including its coupling\n"
                                    : "# E_coupling = u.F\n")
                        << "# columns: time, E_lattice, E_coupling, u_mu (" << nmodes << "), du_mu/dt (" << nmodes << "), F_mu ("
                        << nmodes << ")\n";

            os_modes_.open(path("Phonon_modes.txt"));
            os_modes_ << "# Lattice at q = 0 in the normal modes lambda of the dynamical matrix of ph.x, atomic units per unit cell\n"
                      << "# Q = sum_mu e(mu,lambda) sqrt(M_mu) u_mu (bohr sqrt(m_e)), dQ/dt, F = sum_mu e(mu,lambda) F_mu/sqrt(M_mu)\n"
                      << "# (F_mu of Lattice.txt: Q'' = -omega^2 Q - F), E = 1/2 dQ^2 + 1/2 omega^2 Q^2 (Ha),\n"
                      << "# N = E/omega = |<b>|^2, number of phonons of the coherent (classical) lattice (0 for |omega| < 1 cm^-1)\n"
                      << "# In a degenerate subspace only the sum over the modes is meaningful\n"
                      << "# modes: lambda, frequency (cm^-1), eigenvector e(mu, lambda) for mu = 3*atom + direction\n";
            for( int lambda = 0; lambda < nmodes; ++lambda ) {
                os_modes_ << "# " << std::setw(4) << lambda << std::setw(14) << std::setprecision(6) << std::fixed
                          << lattice->mode_frequency(lambda) * 219474.6313705;
                for( int mu = 0; mu < nmodes; ++mu ) {
                    os_modes_ << std::setw(11) << std::setprecision(6) << lattice->mode_vector(mu, lambda);
                }
                os_modes_ << "\n";
            }
            os_modes_ << std::defaultfloat
                      << "# columns: time, Q (" << nmodes << "), dQ/dt (" << nmodes << "), F (" << nmodes << "), E ("
                      << nmodes << "), N (" << nmodes << ")\n";
        }
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
        print_velocity_energy(time__);
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
    return (int(round(time__ / propagator_->time_step())) % printresolution == 0);
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

/// @brief Sum over the local k points of Tr[A(k) B(k)]
static std::complex<double> trace_product(const BlockMatrix<std::complex<double>>& A__, const BlockMatrix<std::complex<double>>& B__)
{
    double re = 0., im_ = 0.;
    #pragma omp parallel for reduction(+:re,im_)
    for (int ik = 0; ik < A__.get_nblocks(); ++ik) {
        for (int i = 0; i < A__.get_nrows(); ++i) {
            for (int j = 0; j < A__.get_ncols(); ++j) {
                auto x = A__(ik, i, j) * B__(ik, j, i);
                re += x.real();
                im_ += x.imag();
            }
        }
    }
    return std::complex<double>(re, im_);
}

/// @brief Sum over the local k points of Tr[A(k) (B(k) - C(k))], with the difference taken element by element:
/// the energy absorbed can be many orders of magnitude smaller than Tr[A B], and Tr[A B] - Tr[A C] would lose its digits
static std::complex<double> trace_product_difference(const BlockMatrix<std::complex<double>>& A__,
                                                      const BlockMatrix<std::complex<double>>& B__,
                                                      const BlockMatrix<std::complex<double>>& C__)
{
    double re = 0., im_ = 0.;
    #pragma omp parallel for reduction(+:re,im_)
    for (int ik = 0; ik < A__.get_nblocks(); ++ik) {
        for (int i = 0; i < A__.get_nrows(); ++i) {
            for (int j = 0; j < A__.get_ncols(); ++j) {
                auto x = A__(ik, i, j) * (B__(ik, j, i) - C__(ik, j, i));
                re += x.real();
                im_ += x.imag();
            }
        }
    }
    return std::complex<double>(re, im_);
}

/// @brief Prints the mean value of the velocity (Velocity.txt) and the energy balance of the electrons (Energy.txt).
///
/// The velocity is the one generated by the Hamiltonian that propagates the density matrix, including the
/// nonlocal self energy of the mean field:
/// @f[
/// \mathbf v(t) = \frac{1}{N_k \Omega}\sum_{\mathbf k} \mathrm{Tr}\Big[\big(\nabla_k(H_0+\Sigma) + i[H_0+\Sigma, \mathbf r]\big)\rho(\mathbf k)\Big]
/// @f]
/// (@f$\Omega@f$ volume or area of the unit cell). With the equation of motion of EDUS (length gauge)
/// @f[ \dot\rho = \mathbf E\cdot\nabla_k\rho - i[H_0 + \mathbf E\cdot\mathbf r + \Sigma[\rho-\rho_0], \rho] @f]
/// the energy @f$ E = \mathrm{Tr}[H_0\Delta\rho] + \frac12 \mathrm{Tr}[\Sigma[\Delta\rho]\Delta\rho] @f$ changes only because of the
/// work of the field: @f$ \dot E = -\Omega\,\mathbf E(t)\cdot\mathbf v(t) @f$. The work is integrated with the trapezoidal
/// rule over the print steps.
void OutputManager::print_velocity_energy(const double& time__)
{
    PROFILE("OutputManager::print_velocity_energy");
    /* with the Peierls phase, copy_DM_to_aux gives back the density matrix of the length gauge: the Peierls
       propagation is the same equation with the gradient in R (i R.E), so velocity and balance are the same */
    copy_DM_to_aux(time__);
    auto& aux = state_->aux_DM();
    aux.go_to_k();
    auto& rho = aux.get_Operator(Space::k);
    auto& rho0 = system_->DM0().get_Operator(Space::k);

    /* local sums: 0-2 H0 part of the velocity (operator of System, already divided by the cell volume),
       3-5 Sigma part of the velocity (not divided), 6 Tr[H0 (rho-rho0)], 7 Tr[Sigma (rho-rho0)],
       8-9 the same two for rho_BO (dynamic adiabatic reference) */
    std::array<std::complex<double>, 10> local;
    local.fill(0.);
    for (auto ix : { 0, 1, 2 }) {
        temp_.fill(0.);
        multiply(temp_, 1. + im * 0., system_->Velocity()[ix].get_Operator_k(), rho);
        for (int ik = 0; ik < temp_.get_nblocks(); ++ik) {
            for (int ibnd = 0; ibnd < temp_.get_nrows(); ++ibnd) {
                local[ix] += temp_[ik](ibnd, ibnd);
            }
        }
    }
    local[6] = trace_product_difference(system_->H0().get_Operator(Space::k), rho, rho0);

    /* forces on the lattice: of rho - rho0 (force_full), of rho_BO - rho0 and of rho - rho_BO (dynamic reference) */
    auto* lattice = propagator_->lattice();
    const bool dynamic = lattice && lattice->dynamic_reference();
    std::vector<double> force_full, force_bo, force_excited;
    if( lattice ) {
        aux.go_to_R();
        force_full = lattice->force(aux, system_->DM0());
    }
    if( dynamic ) {
        auto& rho_bo = state_->variables().rho_bo;
        rho_bo.lock_space(Space::k);
        rho_bo.go_to_R();
        force_bo = lattice->force(rho_bo, system_->DM0());
        force_excited = lattice->force(aux, rho_bo);
        local[8] = trace_product_difference(system_->H0().get_Operator(Space::k), rho_bo.get_Operator(Space::k), rho0);
    }
    if( meanfield_->parameters().enabled || lattice ) {
        /* self energy Sigma[rho-rho0] + sum_mu u_mu g_mu, computed in R like in the propagation */
        sigma_.get_Operator(R).fill(0.);
        sigma_.lock_space(R);
        aux.go_to_R();
        if( meanfield_->parameters().enabled ) {
            meanfield_->self_energy(sigma_, aux, system_->DM0());
        }
        if( lattice ) {
            lattice->add_coupling(sigma_, state_->lattice());
        }
        sigma_.go_to_k();
        auto& sigma_k = sigma_.get_Operator(Space::k);
        local[7] = trace_product_difference(sigma_k, rho, rho0);

        auto& kgradient = propagator_->kgradient();
        auto gradient_space = propagator_->parameters().gradient_space;
        std::array<Coordinate, 3> direction;
        direction[0].initialize(1, 0, 0);
        direction[1].initialize(0, 1, 0);
        direction[2].initialize(0, 0, 1);
        for (auto ix : { 0, 1, 2 }) {
            /* i[Sigma, r] */
            commutator(temp_, im, sigma_k, system_->r()[ix].get_Operator(Space::k));
            local[3 + ix] = trace_product(temp_, rho);
            /* grad_k Sigma */
            grad_sigma_.lock_space(gradient_space);
            kgradient.Calculate(1. + 0. * im, grad_sigma_.get_Operator(gradient_space),
                                sigma_.get_Operator(gradient_space), direction[ix], true);
            grad_sigma_.go_to_k();
            local[3 + ix] += trace_product(grad_sigma_.get_Operator(Space::k), rho);
        }
    }
    /* mean-field energy of rho_BO */
    if( dynamic && meanfield_->parameters().enabled ) {
        auto& rho_bo = state_->variables().rho_bo;
        sigma_.get_Operator(R).fill(0.);
        sigma_.lock_space(R);
        meanfield_->self_energy(sigma_, rho_bo, system_->DM0());
        sigma_.go_to_k();
        local[9] = trace_product_difference(sigma_.get_Operator(Space::k), rho_bo.get_Operator(Space::k), rho0);
    }
    if( dynamic ) {
        /* the propagation aligns the R component again from k */
        state_->variables().rho_bo.lock_space(Space::k);
    }
#ifdef EDUS_MPI
    auto& comm = decomposition_->kpool_comm();
    MPI_Reduce( comm.rank() == 0 ? MPI_IN_PLACE : local.data(), local.data(),
                int(local.size()), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, 0, comm.communicator() );
#endif
    aux.set_processor(propagator_->processor());
    if( !is_writer_ ) {
        return;
    }
    double num_k = rho.get_MeshGrid()->get_TotalSize();
    auto volume = system_->cell_volume();

    /* velocity, normalized as the operator of System */
    std::array<std::complex<double>, 3> v;
    for (auto ix : { 0, 1, 2 }) {
        v[ix] = local[ix] / num_k + local[3 + ix] / (num_k * volume);
        os_velocity_ << std::setw(20) << std::setprecision(8) << v[ix].real();
        os_velocity_ << std::setw(20) << std::setprecision(8) << v[ix].imag();
    }
    os_velocity_ << std::endl;

    /* energy balance, per unit cell: changes with respect to the equilibrium density matrix rho0 */
    auto field = (*lasers_)(time__).get("Cartesian");
    double power = 0.;
    for (auto ix : { 0, 1, 2 }) {
        power -= field[ix] * v[ix].real() * volume;
    }
    double energy_band = local[6].real() / num_k;
    double energy_mf = 0.5 * local[7].real() / num_k;
    double energy = energy_band + energy_mf;
    if( lattice ) {
        /* Energy balance with the lattice. The total energy per unit cell (all spin channels) is
               s E_electrons + u.F + E_lattice,     E_electrons = E_band + E_MF  (one spin channel, as in Energy.txt)
           where u.F = s/N sum_k Tr[u.g (rho - rho0)] is the energy of the coupling. Energy.txt is per spin channel,
           so the lattice terms are divided by s: E = E_band + E_MF + (E_lattice + E_coupling)/s.
           sigma_ was built as Sigma + u.g (for the velocity), so local[7] = Tr[(Sigma + u.g)(rho - rho0)] contains
           also Tr[u.g (rho - rho0)] = N u.F/s: it is removed from E_MF here */
        auto& x = state_->lattice();
        double s = lattice->parameters().spin_degeneracy;
        auto dot_u = [&](const std::vector<double>& F__) {
            double sum = 0.;
            for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
                sum += x.displacement(0, mu).real() * F__[mu];
            }
            return sum;
        };
        double coupling = dot_u(force_full);
        energy_mf -= 0.5 * coupling / s;
        /* dynamic reference: the lattice is pushed by rho - rho_BO with K_BO, and the conserved energy is
           E[rho] - E[rho_BO] + E_lattice, with the energy of rho_BO
           E[rho_BO] = s (Tr[H0 (rho_BO-rho0)] + 1/2 Tr[Sigma (rho_BO-rho0)])/N + u.F[rho_BO].
           It is included in E_coupling: E_coupling = u.F[rho] - E[rho_BO] */
        if( dynamic ) {
            coupling -= s * (local[8].real() + 0.5 * local[9].real()) / num_k + dot_u(force_bo);
        }
        /* Lattice.txt has the force that pushes the atoms in the dynamics */
        auto& force = dynamic ? force_excited : force_full;
        double energy_lattice = lattice->energy(x);
        energy = energy_band + energy_mf + (energy_lattice + coupling) / s;

        os_lattice_ << std::setw(22) << std::setprecision(12) << time__
                    << std::setw(22) << std::setprecision(12) << energy_lattice
                    << std::setw(22) << std::setprecision(12) << coupling;
        for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
            os_lattice_ << std::setw(22) << std::setprecision(12) << x.displacement(0, mu).real();
        }
        for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
            os_lattice_ << std::setw(22) << std::setprecision(12) << x.velocity(0, mu).real();
        }
        for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
            os_lattice_ << std::setw(22) << std::setprecision(12) << force[mu];
        }
        os_lattice_ << std::endl;

        /* the same in the normal modes of ph.x */
        auto modes = lattice->project_on_modes(x, force);
        os_modes_ << std::setw(22) << std::setprecision(12) << time__;
        for( auto* v : {&modes.Q, &modes.dQ, &modes.F, &modes.energy, &modes.population} ) {
            for( auto value : *v ) {
                os_modes_ << std::setw(22) << std::setprecision(12) << value;
            }
        }
        os_modes_ << std::endl;
    }

    if( first_energy_step_ ) {
        first_energy_step_ = false;
        initial_energy_ = energy;
        work_ = 0.;
    }
    else {
        work_ += 0.5 * (last_power_ + power) * (time__ - last_time_);
    }
    last_power_ = power;
    last_time_ = time__;

    for (auto x : { time__, energy_band, energy_mf, energy - initial_energy_, power, work_, energy - initial_energy_ - work_ }) {
        os_energy_ << std::setw(22) << std::setprecision(12) << x;
    }
    os_energy_ << std::endl;
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
