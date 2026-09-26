/// @brief This standard function contains what equals the derivative in time of the density matrix. 
/// From the Schrodinger equation, we know:
/// @f[
/// \frac{\partial \rho}{\partial t} = -i [H_0 + H_{\text{eff}} + \boldsymbol{\varepsilon}(t)\cdot \Xi, \rho] + 
/// \boldsymbol{\varepsilon}(t)\nabla_\textbf{k} \rho 
/// @f]
/// @param Output__ We store here @f$ \frac{\partial \rho}{\partial t} @f$
/// @param time__ Current time of the simulation, to calculate the time dependent hamiltonian and the laser
/// @param Input__ Input density matrix, to be used as the density matrix on the RHS of the equation
std::function<void(Operator<std::complex<double>>&, double const&, Operator<std::complex<double>> const&)> 
SourceTerm = 
[&](Operator<std::complex<double>>& Output__, const double& time__, const Operator<std::complex<double>>& Input__)
{
///cout debug    std::cout << "SOURCE TERM " << std::endl;

    /* next line aligns k and R components of input (DM_{i-1}) */
    const_cast<Operator<std::complex<double>>&>(Input__).go_to_R(true);

    const_cast<Operator<std::complex<double>>&>(Input__).transfer_to(host);
///cout debug    std::cout << "inputR_: " << *max(Input__.get_Operator(R)) << std::endl;
///cout debug    std::cout << "inputk_: " << *max(Input__.get_Operator(k)) << std::endl;
    const_cast<Operator<std::complex<double>>&>(Input__).set_processor(processor_);
    Output__.get_Operator(SpaceOfPropagation_Gradient_).fill(0.);
    
    /* Gradient term:    Output +=   (E.Nabla) * Input */ 
    Output__.lock_space(SpaceOfPropagation_Gradient_);
    if (!ctx_->cfg().peierls()) {
        kgradient_.Calculate(1.+0.*im, Output__.get_Operator(SpaceOfPropagation_Gradient_), 
                        Input__.get_Operator(SpaceOfPropagation_Gradient_), 
                        setoflaser_(time__), false);
    }

    /* IPA Hamiltonian H_ = H0_ + E \cdot r*/
    Calculate_TDHamiltonian(time__, true);

    /* Coulomb interaction H_ += \Sigma^H[\rho] + \Sigma^{SEX}[\rho] */
    H_.go_to_R();

    H_.transfer_to(host);
///cout debug    std::cout << "H_: " << *max(H_.get_Operator(R)) << std::endl;
    H_.set_processor(processor_);

    if(ctx_->cfg().peierls()) {
        copy(Input__.get_Operator(R), aux_DM_.get_Operator(R), processor_);
    aux_DM_.transfer_to(host);
///cout debug    std::cout << "aux_dm: " << *max(aux_DM_.get_Operator(R)) << std::endl;
    aux_DM_.set_processor(processor_);
        aux_DM_.lock_space(R);    
        Apply_Peierls_phase(aux_DM_, time__, -1, processor_);
    }
    aux_DM_.transfer_to(host);
///cout debug    std::cout << "aux_dm: " << *max(aux_DM_.get_Operator(R)) << std::endl;
    aux_DM_.set_processor(processor_);
    
    auto& DM = ctx_->cfg().peierls() ? aux_DM_ : Input__;
    coulomb_.EffectiveHamiltonian( H_, DM, false); 
    
    /* Peierls transformation H_(R) = H_(R)*exp(+i*A(t) \cdot R) */
    if(ctx_->cfg().peierls()) {
        Apply_Peierls_phase(H_, time__, +1, processor_);
    }
    H_.transfer_to(host);
///== cout debug    std::cout << "H_: " << *max(H_.get_Operator(R)) << std::endl;
    H_.set_processor(processor_);
    
    /* Output__ += -i * [ H_, Input__ ] */
    Output__.go_to_k();
    H_.go_to_k();
    const_cast<Operator<std::complex<double>>&>(Input__).lock_space(Space::k); //this is already updated, no need to ft
//==    H_.transfer_to(host);
//==    std::ofstream HH("H.txt");
//==    HH << H_.get_Operator(k) << std::endl;
//==    H_.set_processor(processor_);
//==    const_cast<Operator<std::complex<double>>&>(Input__).transfer_to(host);
//==    HH.close(); HH.open("inputk.txt");
//==    HH << "inputk_: " << Input__.get_Operator(k) << std::endl;
//==    const_cast<Operator<std::complex<double>>&>(Input__).set_processor(processor_);


    Output__.transfer_to(host);
///cout debug    std::cout << "Output__: " << *max(Output__.get_Operator(k)) << std::endl;
    Output__.set_processor(processor_);
    auto& Output = Output__.get_Operator(SpaceOfPropagation_);
    auto& Input = Input__.get_Operator(SpaceOfPropagation_);
    auto& H = H_.get_Operator(SpaceOfPropagation_);
    
    commutator(Output, -im, H, Input, false, processor_);

    /* apply decay in time */
    if( ctx_->cfg().decay() > 1.e-07 ) {
        auto& DM0k = coulomb_.get_DM0().get_Operator(Space::k);

        if(ctx_->cfg().peierls()) {
            Apply_Peierls_phase(Output__, time__, -1, processor_);
        }

        Output__.go_to_k();
        #pragma omp parallel for
        for(int ik=0; ik<Input.get_nblocks(); ik++) {
            for(int irow=0; irow < Input.get_nrows(); irow++ ) {
                for(int icol=0; icol < Input.get_ncols(); icol++) {
                    Output(ik,irow,icol) -= ( DM.get_Operator(k)(ik,irow,icol)-DM0k(ik,irow,icol) )/ctx_->cfg().decay();
                }
            }
        }
        if(ctx_->cfg().peierls()) {
            Apply_Peierls_phase(Output__, time__, +1, processor_);
        }
    }
    Output__.get_Operator(k).transfer_to(host);
    Output__.get_Operator(k).set_processor(device);
};
