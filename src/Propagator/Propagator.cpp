#include "ConvertUnits.hpp"
#include "Propagator/Propagator.hpp"

/// @brief Sets the pointers to the physical objects, sets the density matrix to the equilibrium one
/// and initializes DESolver.
void Propagator::initialize(const PropagatorParameters& parameters__,
                            const GridStructure& gridstructure__,
                            const electron::System& system__,
                            electron::State& state__,
                            electron::MeanField& meanfield__,
                            SetOfLaser& lasers__,
                            const kGradient& kgradient__)
{
    PROFILE("Propagator::initialize");
    parameters_ = parameters__;
    system_     = &system__;
    state_      = &state__;
    meanfield_  = &meanfield__;
    lasers_     = &lasers__;
    kgradient_  = &kgradient__;
    Rgrid_gamma_ = gridstructure__.Rgrid_GammaCentered();

    /* R vectors in a contiguous array, used on gpu for the Peierls phase */
    auto nR_local = Rgrid_gamma_->mpindex.get_nlocal();
    bare_Rgrid_gamma_.initialize({nR_local, 3});
    for( int iR_loc = 0; iR_loc < nR_local; iR_loc++ ) {
        int iR_glob = Rgrid_gamma_->mpindex.loc1D_to_glob1D(iR_loc);
        for( auto ix : {0, 1, 2} ) {
            bare_Rgrid_gamma_(iR_loc, ix) = (*Rgrid_gamma_)[iR_glob].get("Cartesian")[ix];
        }
    }

    initial_condition(state_->DensityMatrix());
    desolver_.initialize(state_->DensityMatrix(), *this, parameters_.desolver);
    check_stability();
}

/// @brief The eigenvalues of the equation of motion are -i(e_n(k) - e_m(k)): the time step must keep
/// omega_max*dt inside the stability region of the time stepper, with omega_max the largest difference
/// of band energies at the same k. The mean field and the field modify the spectrum a little:
/// a margin is kept, with a warning close to the limit.
void Propagator::check_stability()
{
    double omega_max = 0.;
    for( auto& energies : system_->bandstructure().energies() ) {
        double e_min = energies(0), e_max = energies(0);
        for( int ib = 0; ib < energies.get_Size(0); ++ib ) {
            e_min = std::min(e_min, energies(ib));
            e_max = std::max(e_max, energies(ib));
        }
        omega_max = std::max(omega_max, e_max - e_min);
    }
#ifdef EDUS_MPI
    MPI_Allreduce(MPI_IN_PLACE, &omega_max, 1, MPI_DOUBLE, MPI_MAX, mpi::Communicator::world().communicator());
#endif
    auto& stepper = desolver_.stepper();
    double dt = parameters_.desolver.dt;
    double ratio = omega_max * dt / stepper.stability_limit();
    stability_ratio_ = ratio;
    if( ratio > 1. ) {
        std::stringstream ss;
        ss << "The time step is too large for " << stepper.name() << ": omega_max*dt = " << omega_max * dt
           << ", stable only below " << stepper.stability_limit() << " (omega_max = "
           << Convert(omega_max, AuEnergy, ElectronVolt) << " eV, largest difference of band energies). "
           << "Use dt < " << 0.8 * stepper.stability_limit() / omega_max << " a.u. = "
           << Convert(0.8 * stepper.stability_limit() / omega_max, AuTime, FemtoSeconds) << " fs"
           << (stepper.name() == "RK4" ? "" : " or the solver RK") << "\n";
        throw std::runtime_error(ss.str());
    }
    if( ratio > 0.8 ) {
        output::print("WARNING: omega_max*dt is ", ratio * 100., " % of the stability limit of ", stepper.name(),
                      ": the mean field can move it beyond, consider a smaller dt");
    }
}

void Propagator::step()
{
    state_->DensityMatrix().set_processor(processor_);
    desolver_.propagate();
}

void Propagator::print_recap() const
{
    output::title("PROPAGATOR");
    auto& desolver = parameters_.desolver;
    output::print("Solver                   *", std::string(8, ' '), desolver_.stepper().name());
    output::print("omega_max*dt / limit     *", stability_ratio_);
    output::print("Resolution time          *", desolver.dt, " a.u.", Convert(desolver.dt, AuTime, FemtoSeconds), " fs");
    output::print("Initial time             *", desolver.initial_time, " a.u.",
                                                Convert(desolver.initial_time, AuTime, FemtoSeconds), " fs");
    output::print("Space for gradient       *", std::string(8, ' '), (parameters_.gradient_space == R ? "R" : "k"));
    output::print("Peierls                  *", std::string(8, ' '), (parameters_.peierls ? "True" : "False"));
    output::print("Decay                    *", parameters_.decay, " a.u.", Convert(parameters_.decay, AuTime, FemtoSeconds), " fs");
    output::stars();
}

void Propagator::initialize_device()
{
    bare_Rgrid_gamma_.initialize_device();
    bare_Rgrid_gamma_.transfer_to(Processor::device);
    desolver_.initialize_device();
}

/// @brief Standard function defining the initial density matrix.
/// The initial density matrix is the equilibrium one, computed in electron::System:
/// @f[
/// \rho_{nn}(\textbf{k}) = f_n \quad \text{in the bloch gauge}
/// @f]
/// already rotated to the wannier gauge, where the equations are propagated.
/// @param DM__ The Operator where we want to store the initial density matrix
void Propagator::initial_condition(Operator<std::complex<double>>& DM__)
{
    PROFILE("Propagator::initial_condition");
    auto& DM0k = system_->DM0().get_Operator(Space::k);
    std::copy(DM0k.begin(), DM0k.end(), DM__.get_Operator_k().begin());

    DM__.lock_gauge(wannier);
    DM__.lock_space(k);

    if(propagation_space_ == R) DM__.go_to_R();
}

/// @brief This standard function contains what equals the derivative in time of the density matrix.
/// From the Schrodinger equation, we know:
/// @f[
/// \frac{\partial \rho}{\partial t} = -i [H_0 + H_{\text{eff}} + \boldsymbol{\varepsilon}(t)\cdot \Xi, \rho] +
/// \boldsymbol{\varepsilon}(t)\nabla_\textbf{k} \rho
/// @f]
/// @param Output__ We store here @f$ \frac{\partial \rho}{\partial t} @f$
/// @param time__ Current time of the simulation, to calculate the time dependent hamiltonian and the laser
/// @param Input__ Input density matrix, to be used as the density matrix on the RHS of the equation
void Propagator::derivative(Operator<std::complex<double>>& Output__, const double& time__,
                            const Operator<std::complex<double>>& Input__)
{
    auto& H_ = state_->H();
    auto& aux_DM_ = state_->aux_DM();
    auto gradient_space = parameters_.gradient_space;

    /* next line aligns k and R components of input (DM_{i-1}) */
    const_cast<Operator<std::complex<double>>&>(Input__).go_to_R(true);
    const_cast<Operator<std::complex<double>>&>(Input__).set_processor(processor_);

    Output__.get_Operator(gradient_space).fill(0.);

    /* Gradient term:    Output +=   (E.Nabla) * Input */
    Output__.lock_space(gradient_space);
    if (!parameters_.peierls) {
        kgradient_->Calculate(1.+0.*im, Output__.get_Operator(gradient_space),
                        Input__.get_Operator(gradient_space),
                        (*lasers_)(time__), false);
    }

    build_hamiltonian(time__, Input__);

    /* Output__ += -i * [ H_, Input__ ] */
    Output__.go_to_k();
    H_.go_to_k();
    const_cast<Operator<std::complex<double>>&>(Input__).lock_space(Space::k); //this is already updated, no need to ft
    auto& Output = Output__.get_Operator(propagation_space_);
    auto& Input = Input__.get_Operator(propagation_space_);
    auto& H = H_.get_Operator(propagation_space_);

    commutator(Output, -im, H, Input, false, processor_);

    /* decay towards equilibrium: d(rho)/dt += -(rho - rho0)/decay.
       In the Peierls gauge the propagated density matrix is phase(+1)*rho, so the equilibrium one is
       phase(+1)*rho0: the phase is applied to rho0 (in R) and everything stays in k, like without decay */
    if( parameters_.decay > 1.e-07 ) {
        const BlockMatrix<std::complex<double>>* DM0k = &system_->DM0().get_Operator(Space::k);
        if(parameters_.peierls) {
            copy(system_->DM0().get_Operator(R), aux_DM_.get_Operator(R), processor_);
            aux_DM_.lock_space(R);
            apply_peierls_phase(aux_DM_, time__, +1, processor_);
            aux_DM_.go_to_k();
            DM0k = &aux_DM_.get_Operator(Space::k);
        }
        #pragma omp parallel for
        for(int ik=0; ik<Input.get_nblocks(); ik++) {
            for(int irow=0; irow < Input.get_nrows(); irow++ ) {
                for(int icol=0; icol < Input.get_ncols(); icol++) {
                    Output(ik,irow,icol) -= ( Input(ik,irow,icol)-(*DM0k)(ik,irow,icol) )/parameters_.decay;
                }
            }
        }
    }
    //TODO: these two lines come from the old code, they look like a leftover of gpu debugging
    Output__.get_Operator(k).transfer_to(host);
    Output__.get_Operator(k).set_processor(device);
}

void Propagator::build_hamiltonian(const double& time__, const Operator<std::complex<double>>& DM__)
{
    auto& H_ = state_->H();
    auto& aux_DM_ = state_->aux_DM();

    /* IPA Hamiltonian H_ = H0_ + E \cdot r*/
    ipa_hamiltonian(time__);

    /* Coulomb interaction H_ += \Sigma^H[\rho] + \Sigma^{SEX}[\rho] */
    H_.go_to_R();

    if(parameters_.peierls) {
        copy(DM__.get_Operator(R), aux_DM_.get_Operator(R), processor_);
        aux_DM_.lock_space(R);
        apply_peierls_phase(aux_DM_, time__, -1, processor_);
    }
    auto& DM = parameters_.peierls ? aux_DM_ : DM__;
    meanfield_->self_energy(H_, DM, system_->DM0());

    /* Peierls transformation H_(R) = H_(R)*exp(+i*A(t) \cdot R) */
    if(parameters_.peierls) {
        apply_peierls_phase(H_, time__, +1, processor_);
    }
}

/// @brief Hamiltonian of the equation in commutator form, d(rho)/dt = -i[H, rho] (Peierls gauge, no decay):
/// the same H used by derivative, in k.
void Propagator::hamiltonian(Operator<std::complex<double>>& H__, const double& time__,
                             const Operator<std::complex<double>>& DM__)
{
    /* align the R component of the density matrix, needed by the mean field */
    const_cast<Operator<std::complex<double>>&>(DM__).go_to_R(true);
    build_hamiltonian(time__, DM__);
    auto& H_ = state_->H();
    H_.go_to_k();
    auto& Hk = H_.get_Operator(Space::k);
    std::copy(Hk.begin(), Hk.end(), H__.get_Operator(Space::k).begin());
    H__.lock_space(Space::k);
}

void ipa_hamiltonian_cpu( BlockMatrix<std::complex<double>>& H,
                                  const BlockMatrix<std::complex<double>>& H0,
                                  const BlockMatrix<std::complex<double>>& x,
                                  const BlockMatrix<std::complex<double>>& y,
                                  const BlockMatrix<std::complex<double>>& z,
                                  const Vector<double>& las)
{
#pragma omp parallel for schedule(static) collapse(3)
    for (int iblock = 0; iblock < H0.get_nblocks(); ++iblock) {
        for (int irow = 0; irow < H0.get_nrows(); ++irow) {
            for (int icol = 0; icol < H0.get_ncols(); ++icol) {
                H(iblock, irow, icol) += H0(iblock, irow, icol)
                    + las[0] * x(iblock, irow, icol)
                    + las[1] * y(iblock, irow, icol)
                    + las[2] * z(iblock, irow, icol);
            }
        }
    }
}

/// @brief Function to calculate the time dependent Hamiltonian (one-body) as a sum of H0 and the
/// interaction with the laser:
/// @f[
/// H_{\text{1B}}(\textbf{k}) = H_0(\textbf{k})+ \boldsymbol{\varepsilon}(t)\cdot \Xi(\textbf{k})
///@f]
/// The previous content of the hamiltonian of the state is overwritten.
/// @param time__ Time on which we want to calculate the hamiltonian, used for the laser
void Propagator::ipa_hamiltonian(const double& time__)
{
#ifdef EDUS_TIMERS
    PROFILE("Propagator::ipa_hamiltonian");
#endif
    //--------------------get aliases for nested variables--------------------------------
    auto& H_ = state_->H();
    auto& H = H_.get_Operator(ipa_hamiltonian_space_);
    auto& H0 = system_->H0().get_Operator(ipa_hamiltonian_space_);
    auto& x = system_->r()[0].get_Operator(ipa_hamiltonian_space_);
    auto& y = system_->r()[1].get_Operator(ipa_hamiltonian_space_);
    auto& z = system_->r()[2].get_Operator(ipa_hamiltonian_space_);

    auto las  = (*lasers_)(time__).get("Cartesian");
    H.fill(0.);
    H_.lock_space(ipa_hamiltonian_space_);
#ifdef EDUS_GPU
    if ( processor_ == device ) {
        las.initialize_device();
        las.transfer_to(device);
        ipa_hamiltonian_gpu(H.data(device),
                                    H0.data(device),
                                    x.data(device),
                                    y.data(device),
                                    z.data(device),
                                    las.data(device),
                                    las.data(device)+1,
                                    las.data(device)+2,
                                    H.get_TotalSize()
                                );
        H.set_processor(Processor::device);
        return;
    }
#endif
    ipa_hamiltonian_cpu(H, H0, x, y, z, las);
}

void Propagator::apply_peierls_phase(Operator<std::complex<double>>& O__, const double& time__, const int sign,
                                     const Processor& proc__)
{
    O__.go_to_R();

    /* Calculate Peierls phase on the grid */
    auto At = lasers_->VectorPotential(time__);
#ifdef EDUS_GPU
    if ( proc__ == device ) {
        auto At_cart     = At.get("Cartesian");
        At_cart.initialize_device();
        At_cart.transfer_to(processor_);
        apply_peierls_phase_gpu(    O__.get_Operator(R).data(device),
                                    state_->Peierls_phase().data(device),
                                    At_cart.data(device),
                                    At_cart.data(device)+1,
                                    At_cart.data(device)+2,
                                    bare_Rgrid_gamma_.data(device),
                                    sign,
                                    O__.get_Operator(R).get_TotalSize(),
                                    O__.get_Operator(R).get_nblocks()
                                );
        O__.set_processor(Processor::device);
        return;
    }
#endif
    apply_peierls_phase_cpu(O__.get_Operator(R),
                            state_->Peierls_phase(),
                            At,
                            *Rgrid_gamma_,
                            sign);
}

void apply_peierls_phase_cpu( BlockMatrix<std::complex<double>>& OR__,
                              mdarray<std::complex<double>,1>& Peierls_phase,
                              const Coordinate& At,
                              const MeshGrid& Rgrid_gamma,
                              int sign)
{
#pragma omp parallel for schedule(static)
    for (int iR_loc = 0; iR_loc < OR__.get_nblocks(); ++iR_loc) {
        int iR_glob = Rgrid_gamma.mpindex.loc1D_to_glob1D(iR_loc);
        Peierls_phase(iR_loc) = std::exp(im*double(sign)*At.dot(Rgrid_gamma[iR_glob]));
    }

#pragma omp parallel for schedule(static) collapse(3)
    for (int iblock = 0; iblock < OR__.get_nblocks(); ++iblock) {
        for (int irow = 0; irow < OR__.get_nrows(); ++irow) {
            for (int icol = 0; icol < OR__.get_ncols(); ++icol) {
                OR__(iblock, irow, icol) *= Peierls_phase(iblock);
            }
        }
    }
}
