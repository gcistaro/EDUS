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
    nq_ = state_->DensityMatrix().get_nq();

    /* define the index of Rgrid where (0,0,0) is */
    int index_origin_global = gridstructure__.Rgrid()->find(Coordinate(0,0,0));
    has_origin_ = decomposition_->mpindex().is_local(index_origin_global);
    if( has_origin_ ) {
        index_origin_local_ = decomposition_->mpindex().glob1D_to_loc1D(index_origin_global);
    }

    auto& DMk = state_->DensityMatrix().get_Operator_k();
    temp_.initialize(k, DMk.get_nk(), DMk.get_nrows(), DMk.get_ncols());
    if( nq_ > 1 ) {
        rho_q0_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), DMk.get_nrows(), decomposition_->mpindex(), "rho_q0");
        /* velocity with q: workspaces with the q grid of the density matrix, r at the halo k points (fixed) */
        const int nb = DMk.get_nrows();
        auto& halo = propagator_->halo_k();
        h_q_.initialize_fft(state_->aux_DM(), "H_velocity");
        for( int ix : {0, 1, 2} ) {
            v_q_[ix].initialize_fft(state_->aux_DM(), "v_q");
            v_q_halo_[ix] = BlockMatrix<std::complex<double>>(Space::k, halo.n_halo(), nb, nb, 0, nq_);
            r_halo_[ix] = BlockMatrix<std::complex<double>>(Space::k, halo.n_halo(), nb, nb);
            halo.exchange(r_halo_[ix], system_->r()[ix].get_Operator(Space::k));
        }
        ikq_ = &gridstructure__.ikq();
        iqmp_ = &gridstructure__.iqmp();
        auto& qgrid = *gridstructure__.qgrid();
        phase_tau_.initialize({nq_, nb});
        /* q and q + G are the same point of the grid but not the same Fourier component of the density: j(q) is
           the one at the smallest q, crystal coordinates folded in (-1/2, 1/2] */
        for( int iq = 0; iq < nq_; ++iq ) {
            auto qc = qgrid[iq].get(LatticeVectors(k));
            Coordinate folded(qc[0] - std::ceil(qc[0] - 0.5), qc[1] - std::ceil(qc[1] - 0.5), qc[2] - std::ceil(qc[2] - 0.5),
                              LatticeVectors(k));
            auto q = folded.get("Cartesian");
            for( int n = 0; n < nb; ++n ) {
                auto tau = system_->wannier_centers()[n].get("Cartesian");
                phase_tau_(iq, n) = std::exp(-im*(q[0]*tau[0] + q[1]*tau[1] + q[2]*tau[2]));
            }
        }
    }
    /* workspace for the energy balance */
    sigma_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), DMk.get_nrows(), decomposition_->mpindex(), "Sigma_energy");
    grad_sigma_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), DMk.get_nrows(), decomposition_->mpindex(), "grad_Sigma_energy");
    if( propagator_->lattice() && propagator_->lattice()->static_reference() ) {
        rho_bo_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), DMk.get_nrows(), decomposition_->mpindex(), "rho_BO_output");
    }
    if( propagator_->lattice() ) {
        sigma_ref_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), DMk.get_nrows(), decomposition_->mpindex(), "Sigma_ref_energy");
        for( auto* m : {&hk_, &rhodot_, &eta_g_} ) {
            m->initialize(k, DMk.get_nblocks(), DMk.get_nrows(), DMk.get_ncols());
        }
    }

    std::filesystem::create_directories(parameters_.directory);
    mpi::Communicator::world().barrier();

    /* print equilibrium density matrix in real space (one file per rank) */
    if( parameters_.print_DM_R ) {
        /* with nq > 1 aux_DM has the q components: DM0 (q = 0) goes in rho_q0_, which has nq = 1 */
        auto& aux = ( nq_ > 1 ? rho_q0_ : state_->aux_DM() );
        auto& DM0R = system_->DM0().get_Operator(R);
        std::copy(DM0R.begin(), DM0R.end(), aux.get_Operator(R).begin());
        aux.lock_space(R);
        aux.print_Rdecay(path("DM"), system_->wannier_centers());
    }

    if( is_writer_ ) {
        os_pop_.open(path("Population.txt"));
        os_pop_wannier_.open(path("Population_wannier.txt"));
        os_laser_.open(path("Laser.txt"));
        os_vectorpot_.open(path("Laser_A.txt"));
        os_time_.open(path("Time.txt"));
        os_velocity_.open(path("Velocity.txt"));
        if( parameters_.write_velocity_sigma ) {
            os_velocity_sigma_.open(path("Velocity_sigma.txt"));
        }
        if( nq_ > 1 ) {
            os_density_q_.open(path("DensityQ.txt"));
            os_density_q_ << "# n_m(q,t) = rho_mm(R=0; q) = 1/N sum_k rho_mm(k,q), Wannier gauge (q = 0: Population_wannier.txt)\n"
                          << "# q points (crystal coordinates of the reciprocal lattice):\n";
            auto& qgrid = *gridstructure__.qgrid();
            for( int iq = 0; iq < nq_; ++iq ) {
                auto& qc = qgrid[iq].get(LatticeVectors(k));
                os_density_q_ << "#   iq = " << iq << ": " << qc[0] << " " << qc[1] << " " << qc[2] << "\n";
            }
            os_density_q_ << "# columns: time, then for iq = 0..nq-1 and band m: Re n_m(q), Im n_m(q)\n";
        }
        if( nq_ > 1 ) {
            os_current_q_.open(path("CurrentQ.txt"));
            os_current_q_ << "# j(q,t) = 1/(N Omega) <1/2 {v, e^{-iq.r}}>, v = i[H0 + Sigma + V, r], e^{-iq.r} = diag(e^{-iq.(R+tau_n)}) (corrente_q_finito.pdf)\n"
                          << "# q = 0 is Velocity.txt. Continuity: dn(q)/dt = -i Omega q.j(q) (1 + O(q^2)), n(q) = sum_m e^{-iq.tau_m} n_m(q) of DensityQ.txt, Omega = cell volume\n"
                          << "# q points: the same of DensityQ.txt, folded in (-1/2, 1/2] (crystal coordinates) for e^{-iq.tau}\n"
                          << "# columns: time, then for iq = 0..nq-1 and direction x, y, z: Re j(q), Im j(q)\n";
        }
        if( propagator_->local_potential().enabled() ) {
            auto& lp = propagator_->local_potential();
            os_local_potential_.open(path("LocalPotential.txt"));
            os_local_potential_ << "# External local potential V(r,t) = V0 f(t) cos(q0.r), f(t) = exp(-(t-t0)^2/(2 sigma^2))\n"
                                << "# q0 = q point " << lp.iq_plus() << " of DensityQ.txt (-q0: " << lp.iq_minus() << ")\n"
                                << "# columns: time (a.u.), V0 f(t) (Hartree)\n";
        }
        if( nq_ == 1 ) {
            os_energy_.open(path("Energy.txt"));
        }
        auto* lattice = propagator_->lattice();
        os_energy_ << "# Energy balance of the electrons, Hartree per unit cell (one spin channel)\n"
                   << "# E_band = Tr[H0 (rho-rho0)], E_MF = 1/2 Tr[Sigma[rho-rho0] (rho-rho0)], E = E_band + E_MF"
                   << (lattice ? " + E_ph\n# E_ph = (E_lattice + E_coupling)/s: lattice and coupling (see Lattice.txt), divided by the spin degeneracy s\n" : "\n")
                   << "# (changes with respect to equilibrium, computed from rho-rho0 to keep all the digits)\n"
                   << "# W = work of the field = -int E(t).v(t) dt, v = <grad_k(H0+Sigma) + i[H0+Sigma, r]>\n"
                   << "# P = -E.v is the power; W is integrated with the 3-point rule (quadratic through three consecutive print steps, error O(dt^3);\n"
                   << "#   trapezoidal rule at the first print step)\n"
                   << "# Without decay, E(t) - E(0) = W(t) up to the error of the time integration of W\n"
                   << "#        time (a.u.)       E_band-E_band(0)                  E_MF              E - E(0)                     P                     W          E - E(0) - W\n";
        if( lattice ) {
            int nmodes = lattice->num_modes();
            os_lattice_.open(path("Lattice.txt"));
            if( nq_ > 1 ) {
                os_lattice_q_.open(path("LatticeQ.txt"));
                os_lattice_q_ << "# Displacements u_mu(q) (bohr) of all the q points of the grid, u(R) = sum_q u(q) e^{iq.R}\n"
                              << "# columns: time (a.u.), then for iq = 0..nq-1 and mu = 0.." << nmodes - 1 << ": Re u_mu(q), Im u_mu(q)\n";
            }
            os_force_check_.open(path("Force_check.txt"));
            os_force_check_ << "# Check of the force on the lattice: F_R = s sum_R sum_mn g_mn(R) conj(rho - rho_BO)_mn(R) (force used in the\n"
                            << "# dynamics, Lattice.txt) against F_k = s/N sum_k Tr[g(k) (rho - rho_BO)(k)], same density matrices (Ha/bohr)\n"
                            << "#        time (a.u.)       max_mu |F_R - F_k|          max_mu |F_R|\n";
            bool dynamic = lattice->dynamic_reference();
            os_lattice_ << "# Lattice at q = 0, atomic units, per unit cell (all spin channels), adiabatic reference: "
                        << phonon::to_string(lattice->parameters().adiabatic_reference)
                        << (lattice->screened() ? ", electrons with g_s and Sigma[rho - rho_BO]" : ", electrons with g_b and Sigma[rho - rho0]")
                        << (lattice->adiabatic_bare() ? ", rho_BO with g_b and Sigma[rho_BO - rho0]" : "") << "\n"
                        << "# mu = 3*atom + direction (cartesian), u = displacement (bohr), du/dt (bohr/a.u.),\n"
                        << "# F = force of the electrons used in the dynamics, s/N sum_k Tr[g_b,mu (rho - rho_BO)]"
                        << (dynamic ? "" : " = F[rho - rho0] - Pi u") << " (Ha/bohr): the atoms feel -F\n"
                        << "# E_lattice = kinetic + harmonic energy (force constants of ph.x)\n"
                        << (dynamic ? "# E_coupling = u.F[rho - rho0] - E[rho_BO], E[rho_BO] = energy of rho_BO including its coupling\n"
                                    : "# E_coupling = u.F[rho - rho0] - 1/2 u.Pi.u\n")
                        << (lattice->screened() ? "# (with g_s the energy is the one of the bare model, conserved only if g_b = g_s - Sigma[chi0 g_s])\n" : "")
                        << "# columns: time, E_lattice, E_coupling, u_mu (" << nmodes << "), du_mu/dt (" << nmodes << "), F_mu ("
                        << nmodes << ")\n";

            os_energy_terms_.open(path("EnergyTerms.txt"));
            os_energy_terms_ << "# Terms of the energy of Energy.txt and powers that change it, Hartree (per unit time) per unit cell, one spin channel\n"
                             << "# E = E_band + E_MF + E_int + T + U - E_ad, with\n"
                             << "#   E_band = Tr[H0 (rho-rho0)], E_MF = 1/2 Tr[Sigma[rho-rho0] (rho-rho0)] (bare model, always rho0)\n"
                             << "#   E_int = u.F[rho-rho0]/s, F[X] = s/N sum_k Tr[g_b X]; T = 1/2 M v^2/s, U = 1/2 u.K_BO.u/s\n"
                             << "#   E_ad = 1/2 u.F[rho_BO-rho0]/s (static) or E_band[rho_BO] + E_MF[rho_BO] + u.F[rho_BO-rho0]/s (dynamic, also\n"
                             << "#   printed separately); Tr = 1/N sum_k Tr\n"
                             << "# Without decay and damping dE/dt = P_field + P_g + P_sigma + P_BO + P_Pi (derived from the equations of motion):\n"
                             << "#   P_g     = -Tr[u.(g_e - g_b) drho/dt]      g_e = coupling felt by the electrons (g_s if given)\n"
                             << "#   P_sigma = +Tr[Sigma[rho_ref - rho0] drho/dt] rho_ref = rho0 (only g_b) or rho_BO (g_s): the mean field of rho is Sigma[rho-rho_ref]\n"
                             << "#   P_BO    = +Tr[eta drho_BO/dt], eta = u.(g_s - g_b) - Sigma[rho_BO - rho0] (dynamic with g_s and adiabatic_coupling = screened)\n"
                             << "#   P_Pi    = du/dt.Pi_A.u/s, Pi_A antisymmetric part of Pi (static with g_s)\n"
                             << "#   all zero with only g_b. drho/dt = E.grad_k rho - i[H0 + E.r + Sigma[rho-rho_ref] + u.g_e, rho]\n"
                             << "#   I_x = time integral of P_x (3-point rule, like W); residual = E - E(0) - W - I_g - I_sigma - I_BO - I_Pi ~ 0\n"
                             << "# columns: 1 time, 2 E_band, 3 E_MF, 4 E_int, 5 T, 6 U, 7 E_ad, 8 E_band[rho_BO], 9 E_MF[rho_BO], 10 u.F[rho_BO-rho0]/s,\n"
                             << "#   11 E-E(0), 12 W, 13 P_g, 14 P_sigma, 15 P_BO, 16 P_Pi, 17 I_g, 18 I_sigma, 19 I_BO, 20 I_Pi, 21 residual\n";

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
    /* with nq > 1 the Bloch gauge of rho(k,q) needs U(k+q): eigenvectors at the halo k points (fixed) */
    if( nq_ > 1 && parameters_.print_DMk_bloch ) {
        auto& halo = propagator_->halo_k();
        auto& U = *Operator<std::complex<double>>::EigenVectors;
        U_halo_ = BlockMatrix<std::complex<double>>(Space::k, halo.n_halo(), U.get_nrows(), U.get_ncols());
        halo.exchange(U_halo_, U);
        dm_bloch_q_ = BlockMatrix<std::complex<double>>(Space::k, DMk.get_nk(), DMk.get_nrows(), DMk.get_ncols(), 0, nq_);
    }
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
    output::print("toprint-> DM_R           *", std::string(8, ' '), to_str(parameters_.print_DM_R));
    output::print("write_velocity_sigma     *", std::string(8, ' '), to_str(parameters_.write_velocity_sigma));
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
        if( is_writer_ && propagator_->local_potential().enabled() ) {
            os_local_potential_ << std::setw(20) << std::setprecision(10) << time__ << " "
                                << std::setw(24) << std::setprecision(14) << propagator_->local_potential().amplitude(time__)
                                << std::endl;
        }
        /* with nq > 1: n(q,t), populations and velocity of the q = 0 slice; energy not generalized yet */
        if( nq_ > 1 ) {
            print_density_q(time__);
            print_population(time__, BandGauge::bloch);
            print_population(time__, BandGauge::wannier);
            print_current_q(time__);
            print_velocity_energy(time__);
        }
        else {
            print_population(time__, BandGauge::bloch);
            print_population(time__, BandGauge::wannier);
            print_velocity_energy(time__);
        }
    }
#ifdef EDUS_HDF5
    bool print_h5 = ( parameters_.printresolution_h5 > 0
                      ? int(round(time__ / propagator_->time_step())) % parameters_.printresolution_h5 == 0
                      : is_print_step(time__, true) );
    if ( print_h5 ) {
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

/// @brief With nq = 1 the density matrix without the Peierls phase (aux_DM). With nq > 1 its q = 0 slice in rho_q0_:
/// rho_nm(k, q=0) = <c^dag_{m,k} c_{n,k}> is the density matrix of the occupations (populations, velocity), while
/// q != 0 are coherences between k and k+q (density modulations, DensityQ.txt)
Operator<std::complex<double>>& OutputManager::density_matrix_q0(const double& time__)
{
    copy_DM_to_aux(time__);
    auto& aux = state_->aux_DM();
    if( nq_ == 1 ) {
        return aux;
    }
    aux.go_to_k();
    auto& rho = aux.get_Operator(Space::k);
    auto& rho0 = rho_q0_.get_Operator(Space::k);
    const int iq0 = 0; // q = 0 is the first point of the q grid
    #pragma omp parallel for collapse(3)
    for( int ik = 0; ik < rho.get_nk(); ++ik ) {
        for( int n = 0; n < rho.get_nrows(); ++n ) {
            for( int m = 0; m < rho.get_ncols(); ++m ) {
                rho0(ik, n, m) = rho(ik, iq0, n, m);
            }
        }
    }
    rho_q0_.lock_space(Space::k);
    rho_q0_.lock_gauge(aux.get_bandgauge());
    return rho_q0_;
}

/// @brief Prints the population for each band (bloch gauge) or orbital (wannier gauge):
/// @f[
/// P_n(t) = \frac{1}{N}\sum_{\textbf{k}} \rho_{nn}(\textbf{k}) = \rho_{nn}(\textbf{R}=0)
/// @f]
/// For filled bands in the bloch gauge we print the population of holes, @f$ 1-P_n(t) @f$
void OutputManager::print_population(const double& time__, const BandGauge& bandgauge__)
{
    auto& aux = density_matrix_q0(time__);
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

void OutputManager::print_density_q(const double& time__)
{
    copy_DM_to_aux(time__);
    auto& aux = state_->aux_DM();
    aux.go_to_R();

    auto& auxR = aux.get_Operator_R();
    int num_bands = auxR.get_nrows();
    /* real and imaginary part of n_m(q), index 2*(iq*num_bands + m) */
    std::vector<double> density(2 * nq_ * num_bands, 0.);
    if( has_origin_ ) {
        for( int iq = 0; iq < nq_; ++iq ) {
            for( int ibnd = 0; ibnd < num_bands; ++ibnd ) {
                auto value = auxR(index_origin_local_, iq, ibnd, ibnd);
                density[2 * (iq * num_bands + ibnd)]     = value.real();
                density[2 * (iq * num_bands + ibnd) + 1] = value.imag();
            }
        }
    }
#ifdef EDUS_MPI
    /* only one rank has R=0: the sum brings its values to rank 0 */
    auto& comm = decomposition_->kpool_comm();
    MPI_Reduce( comm.rank() == 0 ? MPI_IN_PLACE : density.data(), density.data(),
                int(density.size()), MPI_DOUBLE, MPI_SUM, 0, comm.communicator() );
#endif

    if( is_writer_ ) {
        os_density_q_ << std::setw(20) << std::setprecision(10) << time__;
        for( auto value : density ) {
            os_density_q_ << " " << std::setw(24) << std::setprecision(14) << value;
        }
        os_density_q_ << std::endl;
    }
    aux.set_processor(propagator_->processor());
}

/// @brief nq > 1: velocity with the q components and current at q (corrente_q_finito.pdf).
/// H(q) = H0 + Sigma[rho - rho0](q) + V(+-q0), without the laser (E.r commutes with r), in the length gauge.
///     v(k,q) = ( grad_k H(k,q) + i[H(k,q) r(k) - r(k+q) H(k,q)] )/Omega
///     j(q) = 1/(2N) sum_k sum_p Tr[ ( v(k+p-q, q-p) D_q + D_q v(k+p, q-p) ) rho(k,p) ],   D_q = diag(e^{-iq.tau_n})
/// written in CurrentQ.txt (per unit volume, like Velocity.txt). v at k+p and k+p-q comes from the other ranks (halo).
/// Also computes the terms p != 0 of the macroscopic velocity, sum_k sum_{p != 0} Tr[v(k+p,-p) rho(k,p)], added to
/// Velocity.txt by print_velocity_energy (the term p = 0 is computed there, as with nq = 1).
void OutputManager::print_current_q(const double& time__)
{
    PROFILE("OutputManager::print_current_q");
    /* rho without the Peierls phase, aligned in R (self energy) and k */
    copy_DM_to_aux(time__);
    auto& aux = state_->aux_DM();
    aux.go_to_R();
    aux.go_to_k();
    auto& rho = aux.get_Operator(Space::k);

    /* H(R;q) = H0 (q = 0) + Sigma(q) + V(+-q0) */
    auto& HR = h_q_.get_Operator(R);
    auto& H0R = system_->H0().get_Operator(R);
    HR.fill(0.);
    const int iq0 = 0;
    const int nb = HR.get_nrows();
    #pragma omp parallel for collapse(3)
    for( int iR = 0; iR < H0R.get_nk(); ++iR ) {
        for( int n = 0; n < nb; ++n ) {
            for( int m = 0; m < nb; ++m ) {
                HR(iR, iq0, n, m) = H0R(iR, 0, n, m);
            }
        }
    }
    h_q_.lock_space(R);
    if( meanfield_->parameters().enabled ) {
        meanfield_->self_energy(h_q_, aux, system_->DM0());
    }
    propagator_->local_potential().add(HR, has_origin_ ? index_origin_local_ : -1, time__);

    /* grad_k H(k,q): i R H(R;q) for every q */
    std::array<Coordinate, 3> direction;
    direction[0].initialize(1, 0, 0);
    direction[1].initialize(0, 1, 0);
    direction[2].initialize(0, 0, 1);
    for( int ix : {0, 1, 2} ) {
        v_q_[ix].get_Operator(R).fill(0.);
        propagator_->kgradient().Calculate(1. + 0. * im, v_q_[ix].get_Operator(R), HR, direction[ix], true);
        v_q_[ix].lock_space(R);
        v_q_[ix].go_to_k();
    }
    h_q_.go_to_k();
    auto& Hk = h_q_.get_Operator(Space::k);

    /* + i[H(k,q) r(k) - r(k+q) H(k,q)], then / Omega; v at the halo k points */
    auto& halo = propagator_->halo_k();
    auto& pos = halo.position();
    const int offset = halo.offset();
    const int nk = Hk.get_nk();
    const double volume = system_->cell_volume();
    const std::complex<double> one = 1.;
    for( int ix : {0, 1, 2} ) {
        auto& v = v_q_[ix].get_Operator(Space::k);
        auto& r = system_->r()[ix].get_Operator(Space::k);
        #pragma omp parallel for collapse(2)
        for( int ik = 0; ik < nk; ++ik ) {
            for( int iq = 0; iq < nq_; ++iq ) {
                const int iblock = ik*nq_ + iq;
                const int ikq_h = pos[(*ikq_)(ik + offset, iq)];
                Matrix_gemm(v[iblock],  im, Hk[iblock], r[ik], one);
                Matrix_gemm(v[iblock], -im, r_halo_[ix][ikq_h], Hk[iblock], one);
                for( int n = 0; n < nb; ++n ) {
                    for( int m = 0; m < nb; ++m ) {
                        v(ik, iq, n, m) /= volume;
                    }
                }
            }
        }
        halo.exchange(v_q_halo_[ix], v);
    }

    /* local sums: j(q) (index (iq*3 + ix)), and the terms p != 0 of the velocity */
    std::vector<std::complex<double>> local(3*nq_ + 3, 0.);
    for( int ix : {0, 1, 2} ) {
        auto& vh = v_q_halo_[ix];
        for( int iq = 0; iq < nq_; ++iq ) {
            double re = 0., im_ = 0.;
            #pragma omp parallel for collapse(2) reduction(+:re,im_)
            for( int ik = 0; ik < nk; ++ik ) {
                for( int ip = 0; ip < nq_; ++ip ) {
                    const int ik_glob = ik + offset;
                    const int iqp = (*iqmp_)(iq, ip);                          // q - p
                    const int i1 = pos[(*ikq_)(ik_glob, (*iqmp_)(ip, iq))];     // k + p - q
                    const int i2 = pos[(*ikq_)(ik_glob, ip)];                   // k + p
                    std::complex<double> sum = 0.;
                    for( int n = 0; n < nb; ++n ) {
                        for( int m = 0; m < nb; ++m ) {
                            sum += ( vh(i1, iqp, n, m)*phase_tau_(iq, m) + phase_tau_(iq, n)*vh(i2, iqp, n, m) )
                                   * rho(ik, ip, m, n);
                        }
                    }
                    re += 0.5*sum.real();
                    im_ += 0.5*sum.imag();
                }
            }
            local[iq*3 + ix] = std::complex<double>(re, im_);
        }
        /* velocity: sum_{p != 0} Tr[v(k+p, -p) rho(k,p)] */
        double re = 0., im_ = 0.;
        #pragma omp parallel for collapse(2) reduction(+:re,im_)
        for( int ik = 0; ik < nk; ++ik ) {
            for( int ip = 1; ip < nq_; ++ip ) {
                const int i2 = pos[(*ikq_)(ik + offset, ip)];
                const int imp = (*iqmp_)(iq0, ip);                              // -p
                std::complex<double> sum = 0.;
                for( int n = 0; n < nb; ++n ) {
                    for( int m = 0; m < nb; ++m ) {
                        sum += vh(i2, imp, n, m)*rho(ik, ip, m, n);
                    }
                }
                re += sum.real();
                im_ += sum.imag();
            }
        }
        velocity_q_terms_[ix] = std::complex<double>(re, im_);
    }
#ifdef EDUS_MPI
    auto& comm = decomposition_->kpool_comm();
    MPI_Reduce( comm.rank() == 0 ? MPI_IN_PLACE : local.data(), local.data(),
                int(local.size()), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, 0, comm.communicator() );
#endif
    aux.set_processor(propagator_->processor());
    if( is_writer_ ) {
        const double num_k = rho.get_MeshGrid()->get_TotalSize();
        os_current_q_ << std::setw(20) << std::setprecision(10) << time__;
        for( int i = 0; i < 3*nq_; ++i ) {
            os_current_q_ << " " << std::setw(24) << std::setprecision(14) << local[i].real() / num_k
                          << " " << std::setw(24) << std::setprecision(14) << local[i].imag() / num_k;
        }
        os_current_q_ << std::endl;
    }
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

/// @brief Integral in time with the 3-point rule of print_velocity_energy (the same used for the work W): step__ is the
/// index of the print step just done (0 = first). The integral on [t_{i-1}, t_i] of the quadratic through
/// P_{i-2}, P_{i-1}, P_i is dt/12 (-P_{i-2} + 8 P_{i-1} + 5 P_i); trapezoidal rule at the first interval, replaced at the
/// second one by the integral of the quadratic through P_0, P_1, P_2
static void integrate_3pt(double& integral__, const double second_last__, const double last__, const double power__,
                          const double dt__, const int step__)
{
    if( step__ == 0 ) {
        integral__ = 0.;
    }
    else if( step__ == 1 ) {
        integral__ += 0.5 * (last__ + power__) * dt__;
    }
    else if( step__ == 2 ) {
        integral__ = dt__ / 12. * (5. * second_last__ + 8. * last__ - power__)
                   + dt__ / 12. * (-second_last__ + 8. * last__ + 5. * power__);
    }
    else {
        integral__ += dt__ / 12. * (-second_last__ + 8. * last__ + 5. * power__);
    }
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
/// work of the field: @f$ \dot E = -\Omega\,\mathbf E(t)\cdot\mathbf v(t) @f$. The work is integrated over the print steps
/// with the 3-point rule: on each interval, the integral of the quadratic through the last three print steps.
void OutputManager::print_velocity_energy(const double& time__)
{
    PROFILE("OutputManager::print_velocity_energy");
    /* with the Peierls phase, copy_DM_to_aux gives back the density matrix of the length gauge: the Peierls
       propagation is the same equation with the gradient in R (i R.E), so velocity and balance are the same */
    /* with nq > 1 the q = 0 slice: Tr[v rho] = sum_k sum_p Tr[v(k+p,-p) rho(k,p)], and v has only q = 0 if H has
       only q = 0 (H0 + E.r, Sigma at q = 0). The terms p != 0 of Sigma(q), V(+-q0) (second order) are not included yet */
    auto& aux = density_matrix_q0(time__);
    aux.go_to_k();
    auto& rho = aux.get_Operator(Space::k);
    auto& rho0 = system_->DM0().get_Operator(Space::k);

    /* local sums: 0-2 H0 part of the velocity (operator of System, already divided by the cell volume),
       3-5 Sigma part of the velocity (not divided), 6 Tr[H0 (rho-rho0)], 7 Tr[Sigma (rho-rho0)],
       8-9 the same two for rho_BO (dynamic adiabatic reference), 10-12 powers of EnergyTerms.txt */
    std::array<std::complex<double>, 13> local;
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
    /* nq > 1: terms p != 0 of the velocity (print_current_q), already divided by the cell volume like local[0-2] */
    if( nq_ > 1 ) {
        for( int ix : {0, 1, 2} ) {
            local[ix] += velocity_q_terms_[ix];
        }
    }

    /* forces on the lattice (with g_b): of rho - rho0 (force_full), of rho - rho_BO (force_excited, the one of the
       dynamics) and of rho_BO - rho0 (dynamic reference) */
    auto* lattice = propagator_->lattice();
    const bool dynamic = lattice && lattice->dynamic_reference();
    const bool screened = lattice && lattice->screened();
    std::vector<double> force_full, force_bo, force_excited;
    const Operator<std::complex<double>>* rho_bo = nullptr;
    if( lattice ) {
        aux.go_to_R();
        force_full = phonon::Lattice::force_gamma(lattice->force(aux, system_->DM0()));
        if( dynamic ) {
            auto& rho_bo_dynamic = state_->variables().rho_bo;
            rho_bo_dynamic.lock_space(Space::k);
            rho_bo_dynamic.go_to_R();
            force_bo = phonon::Lattice::force_gamma(lattice->force(rho_bo_dynamic, system_->DM0()));
            local[8] = trace_product_difference(system_->H0().get_Operator(Space::k), rho_bo_dynamic.get_Operator(Space::k), rho0);
            rho_bo = &rho_bo_dynamic;
        }
        else {
            lattice->adiabatic_density(rho_bo_, state_->lattice(), system_->DM0());
            rho_bo = &rho_bo_;
        }
        force_excited = phonon::Lattice::force_gamma(lattice->force(aux, *rho_bo));
        /* check of the R formula: the same force directly in k */
        {
            auto& ref = const_cast<Operator<std::complex<double>>&>(*rho_bo);
            if( !dynamic ) {
                rho_bo_.lock_space(R);
                rho_bo_.go_to_k();
            }
            ref.lock_space(Space::k);
            auto force_check = phonon::Lattice::force_gamma(lattice->force_k(aux, ref));
            double diff = 0., norm = 0.;
            for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
                diff = std::max(diff, std::abs(force_excited[mu] - force_check[mu]));
                norm = std::max(norm, std::abs(force_excited[mu]));
            }
            os_force_check_ << std::setw(22) << std::setprecision(12) << time__
                            << std::setw(22) << std::setprecision(12) << diff
                            << std::setw(22) << std::setprecision(12) << norm << std::endl;
        }
    }
    if( meanfield_->parameters().enabled || lattice ) {
        /* self energy + sum_mu u_mu g_mu, computed in R like in the propagation: Sigma[rho-rho0] + u.g_b, or
           Sigma[rho-rho_BO] + u.g_s with the screened coupling */
        sigma_.get_Operator(R).fill(0.);
        sigma_.lock_space(R);
        aux.go_to_R();
        if( meanfield_->parameters().enabled ) {
            meanfield_->self_energy(sigma_, aux, screened ? *rho_bo : system_->DM0());
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
        /* powers that change the energy of the bare model (EnergyTerms.txt). The electrons feel
           H = H_bm + E.r + eta, H_bm = H0 + u.g_b + Sigma[rho - rho0], eta = u.(g_e - g_b) - Sigma[rho_ref - rho0]:
           d/dt E = W - Tr[eta drho/dt] (+ the terms of rho_BO and Pi, below). Only with g_s eta is not zero */
        if( lattice && screened ) {
            auto& x = state_->lattice();
            /* drho/dt = E.grad_k rho - i[H0 + E.r + sigma, rho], sigma_k = Sigma[rho - rho_ref] + u.g_e */
            auto field = (*lasers_)(time__);
            auto E = field.get("Cartesian");
            auto& H0k = system_->H0().get_Operator(Space::k);
            for( int i = 0; i < hk_.get_TotalSize(); ++i ) {
                *(hk_.begin() + i) = *(H0k.begin() + i) + *(sigma_k.begin() + i);
            }
            for( auto ix : { 0, 1, 2 } ) {
                auto& r = system_->r()[ix].get_Operator(Space::k);
                for( int i = 0; i < hk_.get_TotalSize(); ++i ) {
                    *(hk_.begin() + i) += E[ix] * *(r.begin() + i);
                }
            }
            commutator(rhodot_, -im, hk_, rho);
            if( std::abs(E[0]) + std::abs(E[1]) + std::abs(E[2]) > 0. ) {
                grad_sigma_.lock_space(gradient_space);
                kgradient.Calculate(1. + 0. * im, grad_sigma_.get_Operator(gradient_space),
                                    aux.get_Operator(gradient_space), field, true);
                grad_sigma_.go_to_k();
                auto& grad = grad_sigma_.get_Operator(Space::k);
                for( int i = 0; i < rhodot_.get_TotalSize(); ++i ) {
                    *(rhodot_.begin() + i) += *(grad.begin() + i);
                }
            }
            /* eta_g = u.(g_e - g_b), in k */
            eta_g_.fill(0.);
            for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
                double u = x.displacement(0, mu).real();
                auto& ge = lattice->electronic_coupling(mu).get_Operator(Space::k);
                auto& gb = lattice->coupling(mu).get_Operator(Space::k);
                for( int i = 0; i < eta_g_.get_TotalSize(); ++i ) {
                    *(eta_g_.begin() + i) += u * (*(ge.begin() + i) - *(gb.begin() + i));
                }
            }
            /* Sigma[rho_ref - rho0], rho_ref = rho_BO */
            sigma_ref_.get_Operator(R).fill(0.);
            sigma_ref_.lock_space(R);
            if( meanfield_->parameters().enabled ) {
                meanfield_->self_energy(sigma_ref_, *rho_bo, system_->DM0());
            }
            sigma_ref_.go_to_k();
            auto& sigma_ref_k = sigma_ref_.get_Operator(Space::k);
            local[10] = -trace_product(eta_g_, rhodot_);
            local[11] = trace_product(sigma_ref_k, rhodot_);
            /* dynamic reference propagated with H0 + u.g_s (adiabatic_coupling = screened): it also feels eta,
               d/dt E += Tr[eta drho_BO/dt], drho_BO/dt = -i[H0 + u.g_s, rho_BO] */
            if( dynamic && !lattice->adiabatic_bare() ) {
                std::copy(H0k.begin(), H0k.end(), hk_.begin());
                for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
                    double u = x.displacement(0, mu).real();
                    auto& gs = lattice->electronic_coupling(mu).get_Operator(Space::k);
                    for( int i = 0; i < hk_.get_TotalSize(); ++i ) {
                        *(hk_.begin() + i) += u * *(gs.begin() + i);
                    }
                }
                commutator(rhodot_, -im, hk_, rho_bo->get_Operator(Space::k));
                for( int i = 0; i < eta_g_.get_TotalSize(); ++i ) {
                    *(eta_g_.begin() + i) -= *(sigma_ref_k.begin() + i);
                }
                local[12] = trace_product(eta_g_, rhodot_);
            }
        }
        /* with the screened coupling the energy is the one of the bare model: E_MF with Sigma[rho - rho0] only */
        if( screened ) {
            local[7] = 0.;
            if( meanfield_->parameters().enabled ) {
                sigma_.get_Operator(R).fill(0.);
                sigma_.lock_space(R);
                meanfield_->self_energy(sigma_, aux, system_->DM0());
                sigma_.go_to_k();
                local[7] = trace_product_difference(sigma_.get_Operator(Space::k), rho, rho0);
            }
        }
    }
    /* mean-field energy of rho_BO */
    if( dynamic && meanfield_->parameters().enabled ) {
        sigma_.get_Operator(R).fill(0.);
        sigma_.lock_space(R);
        meanfield_->self_energy(sigma_, *rho_bo, system_->DM0());
        sigma_.go_to_k();
        local[9] = trace_product_difference(sigma_.get_Operator(Space::k), rho_bo->get_Operator(Space::k), rho0);
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
    /* mean-field part of the velocity, (i[Sigma, r] + grad_k Sigma)/V, already in Velocity.txt */
    if( parameters_.write_velocity_sigma ) {
        for (auto ix : { 0, 1, 2 }) {
            auto vs = local[3 + ix] / (num_k * volume);
            os_velocity_sigma_ << std::setw(20) << std::setprecision(8) << vs.real();
            os_velocity_sigma_ << std::setw(20) << std::setprecision(8) << vs.imag();
        }
        os_velocity_sigma_ << std::endl;
    }

    /* energy balance, per unit cell: changes with respect to the equilibrium density matrix rho0 (not yet with nq > 1).
       With phonons and nq > 1, provisionally: Lattice.txt and Modes.txt with q = 0 only, the lattice energy of all q and,
       as coupling, u(0).F[rho - rho0](0) (no rho_BO term). The full output at q is in a later step */
    if( nq_ > 1 ) {
        if( lattice ) {
            auto& x = state_->lattice();
            double coupling = 0.;
            for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
                coupling += x.displacement(0, mu).real() * force_full[mu];
            }
            os_lattice_ << std::setw(22) << std::setprecision(12) << time__
                        << std::setw(22) << std::setprecision(12) << lattice->energy(x)
                        << std::setw(22) << std::setprecision(12) << coupling;
            for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
                os_lattice_ << std::setw(22) << std::setprecision(12) << x.displacement(0, mu).real();
            }
            for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
                os_lattice_ << std::setw(22) << std::setprecision(12) << x.velocity(0, mu).real();
            }
            for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
                os_lattice_ << std::setw(22) << std::setprecision(12) << force_excited[mu];
            }
            os_lattice_ << std::endl;
            os_lattice_q_ << std::setw(22) << std::setprecision(12) << time__;
            for( int iq = 0; iq < x.num_qpoints(); ++iq ) {
                for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
                    os_lattice_q_ << std::setw(22) << std::setprecision(12) << x.displacement(iq, mu).real()
                                  << std::setw(22) << std::setprecision(12) << x.displacement(iq, mu).imag();
                }
            }
            os_lattice_q_ << std::endl;
        }
        return;
    }
    auto field = (*lasers_)(time__).get("Cartesian");
    double power = 0.;
    for (auto ix : { 0, 1, 2 }) {
        power -= field[ix] * v[ix].real() * volume;
    }
    double energy_band = local[6].real() / num_k;
    double energy_mf = 0.5 * local[7].real() / num_k;
    double energy = energy_band + energy_mf;
    /* EnergyTerms.txt: columns 2-10 (terms of the energy) and the powers P_g, P_sigma, P_BO, P_Pi */
    std::array<double, 9> terms{};
    std::array<double, 4> powers{};
    if( lattice ) {
        /* Energy balance with the lattice. The total energy per unit cell (all spin channels) is
               s E_electrons + u.F + E_lattice,     E_electrons = E_band + E_MF  (one spin channel, as in Energy.txt)
           where u.F = s/N sum_k Tr[u.g (rho - rho0)] is the energy of the coupling. Energy.txt is per spin channel,
           so the lattice terms are divided by s: E = E_band + E_MF + (E_lattice + E_coupling)/s.
           sigma_ was built as Sigma + u.g (for the velocity), so local[7] = Tr[(Sigma + u.g)(rho - rho0)] contains
           also Tr[u.g (rho - rho0)] = N u.F/s: it is removed from E_MF here (with the screened coupling local[7] was
           computed again with Sigma[rho - rho0] only).
           The adiabatic response, already in the force constants of ph.x, is removed from E_coupling: 1/2 u.Pi.u with
           the static reference, E[rho_BO] with the dynamic one */
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
        if( !screened ) {
            energy_mf -= 0.5 * coupling / s;
        }
        /* dynamic reference: the lattice is pushed by rho - rho_BO with K_BO, and the conserved energy is
           E[rho] - E[rho_BO] + E_lattice, with the energy of rho_BO
           E[rho_BO] = s (Tr[H0 (rho_BO-rho0)] + 1/2 Tr[Sigma (rho_BO-rho0)])/N + u.F[rho_BO].
           It is included in E_coupling: E_coupling = u.F[rho] - E[rho_BO] */
        if( dynamic ) {
            coupling -= s * (local[8].real() + 0.5 * local[9].real()) / num_k + dot_u(force_bo);
        }
        else {
            coupling -= lattice->adiabatic_energy(x);
        }
        /* Lattice.txt has the force that pushes the atoms in the dynamics */
        auto& force = force_excited;
        double energy_lattice = lattice->energy(x);
        energy = energy_band + energy_mf + (energy_lattice + coupling) / s;

        double kinetic = 0.;
        for( int mu = 0; mu < lattice->num_modes(); ++mu ) {
            kinetic += 0.5 * lattice->mass()[mu] * std::norm(x.velocity(0, mu));
        }
        double interaction = dot_u(force_full);
        terms = { energy_band, energy_mf, interaction / s, kinetic / s, (energy_lattice - kinetic) / s,
                  (interaction - coupling) / s, 0., 0., 0. };
        if( dynamic ) {
            terms[6] = local[8].real() / num_k;
            terms[7] = 0.5 * local[9].real() / num_k;
            terms[8] = dot_u(force_bo) / s;
        }
        powers = { local[10].real() / num_k, local[11].real() / num_k, local[12].real() / num_k,
                   lattice->antisymmetric_power(x) / s };

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

    /* work of the field with the 3-point rule (the same as ci-test/check_energy.py): the integral on
       [t_{i-1}, t_i] of the quadratic through P_{i-2}, P_{i-1}, P_i is dt/12 (-P_{i-2} + 8 P_{i-1} + 5 P_i).
       At the first interval only two points are known and W(t_1) is written with the trapezoidal rule; at the
       second one W(t_1) is replaced by the integral of the quadratic through P_0, P_1, P_2 */
    double dt = time__ - last_time_;
    if( num_energy_steps_ == 0 ) {
        initial_energy_ = energy;
        work_ = 0.;
    }
    else if( num_energy_steps_ == 1 ) {
        work_ += 0.5 * (last_power_ + power) * dt;
    }
    else if( num_energy_steps_ == 2 ) {
        work_ = dt / 12. * (5. * second_last_power_ + 8. * last_power_ - power)
              + dt / 12. * (-second_last_power_ + 8. * last_power_ + 5. * power);
    }
    else {
        work_ += dt / 12. * (-second_last_power_ + 8. * last_power_ + 5. * power);
    }
    ++num_energy_steps_;
    second_last_power_ = last_power_;
    last_power_ = power;
    last_time_ = time__;

    if( lattice ) {
        double residual = energy - initial_energy_ - work_;
        for( int i = 0; i < 4; ++i ) {
            integrate_3pt(power_integral_[i], power_second_last_[i], power_last_[i], powers[i], dt, num_energy_steps_ - 1);
            power_second_last_[i] = power_last_[i];
            power_last_[i] = powers[i];
            residual -= power_integral_[i];
        }
        os_energy_terms_ << std::setw(22) << std::setprecision(12) << time__;
        for( auto value : terms ) {
            os_energy_terms_ << std::setw(22) << std::setprecision(12) << value;
        }
        for( auto value : { energy - initial_energy_, work_ } ) {
            os_energy_terms_ << std::setw(22) << std::setprecision(12) << value;
        }
        for( auto* a : { &powers, &power_integral_ } ) {
            for( auto value : *a ) {
                os_energy_terms_ << std::setw(22) << std::setprecision(12) << value;
            }
        }
        os_energy_terms_ << std::setw(22) << std::setprecision(12) << residual << std::endl;
    }

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
    /* q grid: with nq > 1 the blocks of each rank are ordered (k, q, n, m), X_nm(k,q) = <n,k+q|X|m,k> */
    auto& qgrid = *gridstructure__.qgrid();
    dict__["num_qpoints"] = qgrid.get_TotalSize();
    mdarray<double, 2> bare_q({ qgrid.get_TotalSize(), 3 });
    for (int iq = 0; iq < qgrid.get_TotalSize(); ++iq) {
        for (auto& ix : { 0, 1, 2 }) {
            bare_q(iq, ix) = qgrid[iq].get(LatticeVectors(k))[ix];
        }
    }
    dict__["qpoints"] = bare_q;
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
    if( parameters_.print_DMk_bloch && nq_ > 1 ) {
        /* rho^bloch(k,q) = U(k+q)^dagger rho(k,q) U(k), U(k+q) from the halo */
        DM.go_to_k(false);
        auto& rho = DM.get_Operator_k();
        auto& U = *Operator<std::complex<double>>::EigenVectors;
        auto& halo = propagator_->halo_k();
        auto& pos = halo.position();
        const int nb = rho.get_nrows();
        dm_bloch_q_.fill(0.);
        #pragma omp parallel for collapse(2)
        for( int ik = 0; ik < rho.get_nk(); ++ik ) {
            for( int iq = 0; iq < nq_; ++iq ) {
                const auto& Ukq = U_halo_[pos[(*ikq_)(ik + halo.offset(), iq)]];
                for( int i = 0; i < nb; ++i ) {
                    for( int j = 0; j < nb; ++j ) {
                        std::complex<double> sum = 0.;
                        for( int n = 0; n < nb; ++n ) {
                            for( int m = 0; m < nb; ++m ) {
                                sum += std::conj(Ukq(n, i)) * rho(ik, iq, n, m) * U(ik, 0, m, j);
                            }
                        }
                        dm_bloch_q_(ik, iq, i, j) = sum;
                    }
                }
            }
        }
        dm_bloch_q_.write_h5(name, nodename::DMk_bloch, node.str(), decomposition_->kpool_comm());
    }
    else if( parameters_.print_DMk_bloch ) {
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
