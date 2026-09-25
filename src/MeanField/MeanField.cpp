#include "MeanField/MeanField.hpp"
#include "ModelCoulomb/ModelCoulomb.hpp"
#include <filesystem> 


namespace electron {


/// @brief Triggers the initialization of the object
MeanField::MeanField( const MeanFieldParameters& parameters__, 
                      const GridStructure& gridstructure__, 
                      const std::vector<Coordinate>& wannier_centers__, 
                      const parallel::Decomposition& decomposition__ )
{
    initialize( parameters__, gridstructure__, wannier_centers__, decomposition__ );
}

/// @brief Initialize the objects of the class, mainly ModelCoulomb and the Hartree potential, defined as:
/// @f[ H_{nm} = \sum_\textbf{R} V_{nm}(\textbf{R}) = 
/// \sum_\textbf{R} \langle n\textbf{0}m\textbf{R}|V(r-r')| n\textbf{0}m\textbf{R} \rangle @f]
void MeanField::initialize( const MeanFieldParameters& parameters__, 
                            const GridStructure& gridstructure__, 
                            const std::vector<Coordinate>& wannier_centers__, 
                            const parallel::Decomposition& decomposition__ )
{
    parameters_ = parameters__;
    if( !parameters_.enabled ) {
        return;
    }

    barecoulomb_.set_coulomb_model("vcoul3d");
    barecoulomb_.set_epsilon(1.);
    barecoulomb_.initialize  (wannier_centers__, 
                              gridstructure__.Rgrid_GammaCentered(), 
                              parameters__.read_interaction, 
                              parameters__.bare_file, 
                              decomposition__.mpindex(), 
                              2);
    screencoulomb_.set_coulomb_model(parameters__.coulomb_model);
    screencoulomb_.set_epsilon(parameters__.epsilon);
    screencoulomb_.set_r0(parameters__.r0);
    screencoulomb_.initialize(wannier_centers__, 
                              gridstructure__.Rgrid_GammaCentered(), 
                              parameters__.read_interaction, 
                              parameters__.screen_file, 
                              decomposition__.mpindex(), 
                              1);
    output::print("Maximum and minimum (in norm) of bare and screened interaction:");
    auto max = *std::max_element( barecoulomb_.Potential_.begin(), barecoulomb_.Potential_.end(),
                          [] (std::complex<double> a, std::complex<double> b) { return std::real(a) < std::real(b); }); 
    auto min = *std::max_element( barecoulomb_.Potential_.begin(), barecoulomb_.Potential_.end(),
                          [] (std::complex<double> a, std::complex<double> b) { return std::real(a) > std::real(b); }); 
    output::print( "bare:    ", std::real(min), std::real(max));
    max = *std::max_element( screencoulomb_.Potential_.begin(), screencoulomb_.Potential_.end(),
                          [] (std::complex<double> a, std::complex<double> b) { return std::real(a) < std::real(b); }); 
    min = *std::max_element( screencoulomb_.Potential_.begin(), screencoulomb_.Potential_.end(),
                          [] (std::complex<double> a, std::complex<double> b) { return std::real(a) > std::real(b); }); 
    output::print( "screened: ", std::real(min), std::real(max));
    /* define the index of Rgrid where (0,0,0) is */
    int index_origin_global = gridstructure__.Rgrid()->find(Coordinate(0,0,0));
    HasOrigin_ = decomposition__.mpindex().is_local(index_origin_global);
    if( HasOrigin_ ) {
        index_origin_local_ = decomposition__.mpindex().glob1D_to_loc1D(index_origin_global);  
    }  

#ifdef EDUS_MPI
    /* get rank with origin in all the ranks */
    int HasOrigin_int = HasOrigin_ ? 1 : 0;
    output::print("has origin: ", (HasOrigin_ ? "true" : "false"));
    std::vector<int> rank_has_origin(decomposition__.kpool_comm().size());

    MPI_Allgather(&HasOrigin_int, 1, MPI_INT, rank_has_origin.data(), 1, MPI_INT, decomposition__.kpool_comm().communicator());    
    output::print("rank_has_origin[0]: ", (rank_has_origin[0] ? "true" : "false"));
    int root_origin = -1;
    for ( int irank = 0; irank < decomposition__.kpool_comm().size(); irank++ ) {
        if( rank_has_origin[irank] ) {
            if( root_origin != -1 ) {
                output::print("error in origin belonging.");
            }
            root_origin = irank;
        }
    }
    output::print("rank with origin: ", root_origin);
#else
    HasOrigin_ = 1;
#endif

    /* define matrix for Hartree potential */
    int num_bands = wannier_centers__.size();

    Hartree.initialize({num_bands, num_bands});
    Hartree.fill(0.);
    for ( int iR = 0; iR < barecoulomb_.Potential_.get_Size(0); ++iR ) {
        for( int irow = 0; irow < num_bands; ++irow ) {
            for( int icol = 0; icol < num_bands; ++icol ) {
                Hartree(irow, icol) += barecoulomb_.Potential_(iR, irow, icol);
            }
        }
    }

#ifdef EDUS_MPI
    /* reduce the elements of the Hartree potential */
    if (decomposition__.kpool_comm().rank() == root_origin) {
        // Root process: in-place reduction
        MPI_Reduce(MPI_IN_PLACE, Hartree.data(),
               num_bands * num_bands, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
               root_origin, decomposition__.kpool_comm().communicator());
    } else {
        // Non-root processes: send their local Hartree
        MPI_Reduce(Hartree.data(), nullptr,
               num_bands * num_bands, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
               root_origin, decomposition__.kpool_comm().communicator());
    }
#endif

    /* make it hermitian (it must be mathematically) but is not numerically */
    for( int irow = 0; irow < num_bands; ++irow ) {
        for( int icol = irow+1; icol < num_bands; ++icol ) {
            auto value = (Hartree(irow, icol) + Hartree(icol, irow))/2.;
            Hartree(irow, icol) = value;
            Hartree(icol, irow) = value;
        }
    }

    /* make W is hermitian. From tests this modifies only last R */
    auto& W = screencoulomb_.Potential_;
    Operator<std::complex<double>> Wop; 
    Wop.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), num_bands, decomposition__.mpindex(), "W");
    std::copy(W.begin(), W.end(), Wop.get_Operator_R().begin());
    Wop.go_to(k);
    Wop.get_Operator_k().make_hermitian();
    Wop.go_to(R);
    std::copy(Wop.get_Operator(R).begin(), Wop.get_Operator(R).end(), W.begin());

#ifdef __DEBUG
    std::ofstream osos("bare.txt"); 
    osos << barecoulomb_.Potential_ << std::endl; 
    osos.close();
    osos.open("screen.txt"); 
    osos << screencoulomb_.Potential_ << std::endl; 
    osos.close();
#endif
}




void Hartree_interaction_cpu(BlockMatrix<std::complex<double>>& HR__, 
                         const mdarray<std::complex<double>,2>& Hartree, 
                         const BlockMatrix<std::complex<double>>& DMR__, 
                         const BlockMatrix<std::complex<double>>& DM0R_, 
                         int index_origin_local_)
{
    #pragma omp parallel for
    for( int irow = 0; irow < HR__.get_nrows(); ++irow ) {
        for( int icol = 0; icol < HR__.get_ncols(); ++icol ) {
            HR__(index_origin_local_, irow, irow) += 
                    Hartree(irow, icol)*(DMR__(index_origin_local_, icol, icol) - DM0R_(index_origin_local_, icol, icol)); 
        }
    }
}

void Hartree_interaction(BlockMatrix<std::complex<double>>& HR__, 
                         const mdarray<std::complex<double>,2>& Hartree, 
                         const BlockMatrix<std::complex<double>>& DMR__, 
                         const BlockMatrix<std::complex<double>>& DM0R_, 
                         int index_origin_local)
{
#ifdef EDUS_GPU
    if ( processor_ == device ) {
        Hartree_interaction_gpu ( HR__.data(device), 
                                  Hartree.data(device), 
                                  DMR__.data(device),
                                  DM0R_.data(device), 
                                  index_origin_local, 
                                  HR__.end() - HR__.begin() );
        HR__.set_processor(device);
        return;
    }
#endif      
    Hartree_interaction_cpu(HR__, Hartree, DMR__, DM0R_, index_origin_local);
}

/// @brief Adds to H__ the mean-field self energy due to the Coulomb interaction.
/// The Coulomb interaction has two different terms: 
/// - The Hartree term 
/// @f[
///   H^{(\text{eff})}_{n'n}(\textbf{R}) = \delta_{nn'} \delta_{\bf R, 0} \Big(\sum_{m\bf R'} V_{nm}[\textbf{R}']\Big) \rho_{mm}(\textbf{0}) 
/// @f]
/// where the sum over R' is pre-computed and saved in the variable Hartree.
/// - The Fock term
/// @f[
///   H^{(\text{eff})}_{n'n}(\textbf{R}) = -W_{n'n}[\textbf{R}]\rho_{n'n}(\textbf{R}) 
/// @f]
/// Notice that, as we always supposed that the model Hamiltonian at equilibrium @f$ H_0 @f$ already contains the contribution
/// of the Coulomb interaction due to the ground state, to avoid double counting we need to define the effective Hamiltonian
/// over  @f$ \Delta \rho(t) = \rho(t)-\rho(t_0)  @f$
/// @param H__ Hamiltonian Operator, to which the self energy is added (its R component is used)
/// @param DM__ Density matrix at current time, that we need to use to calculate the effective Coulomb interaction
/// @param DM0__ Density matrix at equilibrium, with a valid R component
void MeanField::self_energy(Operator<std::complex<double>>& H__, const Operator<std::complex<double>>& DM__,
                            const Operator<std::complex<double>>& DM0__) 
{
    /* We calculate the Coulomb interaction in R space */
    auto& HR__ = H__.get_Operator(R);
    auto& DMR__ = DM__.get_Operator(R);
    auto& DM0R_ = DM0__.get_Operator(R);

    if ( !parameters_.enabled ) {
        return;
    }

    /* Hartree term */
    if( HasOrigin_  && (parameters_.method == MeanFieldMethod::RPA || parameters_.method == MeanFieldMethod::HSEX)) { // Only the rank with R=0 contributes to this term 
        Hartree_interaction(HR__, Hartree, DMR__, DM0R_, index_origin_local_);
    }

    /* Fock term */
    if (parameters_.method == MeanFieldMethod::HSEX) {
        auto& W = screencoulomb_.Potential_;
        #pragma omp parallel for
        for( int iblock = 0; iblock < HR__.get_nblocks(); ++iblock ) {
            for( int irow = 0; irow < HR__.get_nrows(); ++irow ) {
                for( int icol = 0; icol < HR__.get_ncols(); ++icol ) {
                    HR__( iblock, irow, icol ) -= 
                            W( iblock, irow, icol )*( DMR__( iblock, irow, icol ) - DM0R_( iblock, irow, icol ) );
                }
            }
        } 
    }

    //for( int iblock = 0; iblock < H_.get_nblocks(); ++iblock ) {
    //    for( int irow = 0; irow < H_.get_nrows(); ++irow ) {
    //        for( int icol = 0; icol < H_.get_ncols(); ++icol ) {
    //            H_( iblock, irow, icol ) += W( iblock, irow, icol, iblock1, ibnd1, ibnd2 )*DM( iblock1, ibnd1, ibnd2 );
    //        }
    //    }
    //} 
}    


void read_rk_py(mdarray<std::complex<double>,3>& RytovaKeldysh_TB, const std::string& filename)
{
    auto file = ReadFile(filename);
    auto index = 1;
    
    /* Read Rytova Keldysh (screened) potential produced with RytovaKeldysh.py */
    for( int iR = 0; iR < int(RytovaKeldysh_TB.get_Size(0)); ++iR ) {
        for( int irow = 0; irow < int(RytovaKeldysh_TB.get_Size(1)); ++irow ) {
            for( int icol = 0; icol < int(RytovaKeldysh_TB.get_Size(1)); ++icol ) {
                assert(file[index].size() == 1);
                RytovaKeldysh_TB( iR, irow, icol ) = std::atof( file[index][0].c_str() );
                index++;
            }
        }
    }
    assert(index == int(file.size()));
}

void MeanField::print_recap() const
{
    output::title("MEAN FIELD");
    output::print("Coulomb                  *", std::string(8, ' '), (parameters_.enabled ? "True" : "False"));
    if( parameters_.enabled ) {
        output::print("Method                   *", std::string(8, ' '), to_string(parameters_.method));
        output::print("Read Interaction         *", std::string(8, ' '), (parameters_.read_interaction ? "True" : "False"));
        if( parameters_.read_interaction ) {
            output::print("barecoulomb              *", std::string(8, ' '), parameters_.bare_file);
            output::print("screencoulomb            *", std::string(8, ' '), parameters_.screen_file);
        }
        else {
            output::print("coulomb model            *", std::string(8, ' '), parameters_.coulomb_model);
            output::print("epsilon                  *", parameters_.epsilon);
            auto& r0 = screencoulomb_.r0();
            output::print("r0x                      *", r0[0], " a.u.", Convert(r0[0], AuLength, Angstrom), " angstrom");
            output::print("r0y                      *", r0[1], " a.u.", Convert(r0[1], AuLength, Angstrom), " angstrom");
            output::print("r0z                      *", r0[2], " a.u.", Convert(r0[2], AuLength, Angstrom), " angstrom");
            output::print("r0_avg                   *", screencoulomb_.r0_avg(), " a.u.", 
                                                        Convert(screencoulomb_.r0_avg(), AuLength, Angstrom), " angstrom");
        }
    }
    output::stars();
}

void MeanField::initialize_device()
{
    Hartree.initialize_device();
    Hartree.transfer_to(device);
    screencoulomb_.Potential_.initialize_device();
    screencoulomb_.Potential_.transfer_to(device);
}


}// end namespace electron 