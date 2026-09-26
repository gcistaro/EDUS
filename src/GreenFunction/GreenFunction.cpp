#include "GreenFunction.hpp"

void GreenFunction::initialize()
{
#ifndef EDUS_HDF5
    throw std::runtime_error("Error in GreenFunction::initialize: Compiled without HDF5 support");
#endif
    /* allocate memory for operators */
//    DM_           .initialize_fft(DEsolver_DM_.function());
//    H_            .initialize_fft(DEsolver_DM_.function());
    Ut_           .initialize_fft(DEsolver_DM_.function());
    Utime_        .initialize_fft(DEsolver_DM_.function());
    Uptime_       .initialize_fft(DEsolver_DM_.function());
    GR_           .initialize_fft(DEsolver_DM_.function());
    
    /* initialize solver for Ut */
    std::function<void(Operator<std::complex<double>>&)> 
    InitialCondition_U = 
    [&](Operator<std::complex<double>>& U__)
    {
        /* the initial evolution operator is the identity */
        U__.get_Operator(Space::k).identity();
    };

    std::function<void(Operator<std::complex<double>>&, double const&, Operator<std::complex<double>> const&)> 
    SourceTerm_U = 
    [&](Operator<std::complex<double>>& Output__, const double& time__, const Operator<std::complex<double>>& Input__)
    {
        /* EOM:      i\dot{U} = H.U */
        /* read H from saved datas */
        auto CurrentTime = DEsolver_DM_.current_time();
        std::string name_ = "output.h5";
        auto node = get_it_sparse(CurrentTime); 
        H_.get_Operator(Space::k).load(name_, node, nodename::H0pCoulomb); 
        /* Output = -i H.U */
        multiply(Output__.get_Operator(Space::k), -im, 
                 H_.get_Operator(Space::k), 
                 Input__.get_Operator(Space::k));
    };
    DESolverParameters desolver_parameters;
    desolver_parameters.solver = AB;
    desolver_parameters.order = 5;
    DEsolver_Ut_.initialize( Ut_, InitialCondition_U, SourceTerm_U, desolver_parameters );
}

void GreenFunction::Propagator(Operator<std::complex<double>>& Ut__, const double t__)
{
    if ( ( DEsolver_Ut_.current_time() - t__ ) > 1.e-06 ) {
        //read it from file
    }
    else {
        auto nsteps = (t__ - DEsolver_Ut_.current_time())/double(DEsolver_Ut_.time_step() );
        DEsolver_Ut_.propagate( nsteps );
    }
}

/*
    We calculate 
    G<(t,t') = Theta(t-t')U(t)U^\dagger(t')G<(t',t') + Theta(t'-t) G<(t,t)U(t)U^\dagger(t')
    as formula 31 of Stefanucci, An essential Introduction to NEGF methods for Real-Time simulations
    Note that G<(t',t) = Theta(t'-t)U(t')U^\dagger(t)G<(t,t) + Theta(t-t') G<(t',t')U(t')U^\dagger(t)
                       = Theta(t'-t) ( G<^\dagger(t,t)U(t)U^\dagger(t') )^\dagger 
                         + Theta(t-t') ( U(t)U^\dagger(t')G<^\dagger(t',t') )\dagger
                       = G<(t,t')^\dagger

    We also do the change of variables: 
              trel   = t'-t      |_______     t' = tave + trel/2
                                 |_______>
              tave = (t+t')/2    |            t  = tave - trel/2

*/

BlockMatrix<std::complex<double>>& GreenFunction::GKBA_rel_ave(const double trel__, const double tave__)
{
    return GKBA(tave__+trel__/2., tave__-trel__/2.);
}

BlockMatrix<std::complex<double>>& GreenFunction::GKBA(const double time__, const double ptime__)
{
    auto Theta = [](const double& t, const double& pt) { return ( t - pt ) > 1.e-06;};
    if( !Theta(time__,ptime__) ) {
        BlockMatrix<std::complex<double>>* aux = &GKBA( ptime__,time__ );
        aux->make_dagger();
        return *aux;
    }

    /* DM_ <- rho(t') */
    std::string name_ = "output.h5";
    auto node = get_it_sparse(  ptime__ );
    DensityMatrix_.get_Operator(Space::k).load(name_, node, nodename::DMk); 

    if( std::abs(time__-ptime__) < 1.e-06 ) {
        return DensityMatrix_.get_Operator(Space::k);
    }

    /* here we calculate U(t)U^\dagger(t')G<(t',t') */

    /* Utime_ <- U(t) */
    Propagator(Utime_, time__);
    /* Uptime_ <- U(t') */
    Propagator(Uptime_, ptime__);

    /* GR_ = U(t)U^\dagger(t') */
    multiply( GR_.get_Operator(Space::k), 1.+0.*im, 
              Utime_.get_Operator(Space::k), Uptime_.get_Operator(Space::k));
    /* G< = GR_*DM */
    /* we use Utime_ array as it is not needed anymore */
    auto& Lesser = Utime_.get_Operator(Space::k);
    multiply( Lesser, 1.+0.*im, 
              GR_.get_Operator(Space::k), DensityMatrix_.get_Operator(Space::k) );
    return Lesser;
}