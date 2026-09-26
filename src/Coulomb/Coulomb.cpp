#include "Coulomb.hpp"
#include "ModelCoulomb/ModelCoulomb.hpp"
#include <filesystem> 

/// @brief Triggers the initialization of the object
/// @param nbnd__ Number of bands/wannier functions used in the simulation
/// @param Rgrid__ Grid in R space used in the simulation
/// @param r__ Position operator read from Wannier90, in a.u.
Coulomb::Coulomb(const int& nbnd__, const std::shared_ptr<MeshGrid>& Rgrid__, const std::array<Operator<std::complex<double>>,3>& r__)
{
    initialize(nbnd__, Rgrid__, r__);
}
/// @brief Initialize the objects of the class, mainly ModelCoulomb and the Hartree potential, defined as:
/// @f[ H_{nm} = \sum_\textbf{R} V_{nm}(\textbf{R}) = 
/// \sum_\textbf{R} \langle n\textbf{0}m\textbf{R}|V(r-r')| n\textbf{0}m\textbf{R} \rangle @f]
/// @param nbnd__ Number of bands/wannier functions used in the simulation
/// @param Rgrid__ Grid in R space used in the simulation
/// @param r__ Position operator read from Wannier90, in a.u.
void Coulomb::initialize(const int& nbnd, const std::shared_ptr<MeshGrid>& Rgrid__, const std::array<Operator<std::complex<double>>,3>& r__)
{
    if(!DoCoulomb_) {
        return;
    }
    Rgrid_ = Rgrid__;

    // == auto size_MG_global =  Rgrid__->get_TotalSize();
    // == auto size_MG_local = Rgrid__->get_LocalSize();
    // == HF = mdarray<std::complex<double>,3> ( { int( size_MG_local ), nbnd, nbnd } );
    // == std::filesystem::path cwd = std::filesystem::current_path() / "RytovaKeldysh.txt";
    // == read_rk_py( RytovaKeldysh_TB, cwd.str());
    
    barecoulomb_.set_coulomb_model("vcoul3d");
    barecoulomb_.set_epsilon(1.);
    barecoulomb_.initialize(r__, Rgrid__, read_interaction_, bare_file_path_, 2);
    screencoulomb_.initialize(r__, Rgrid__, read_interaction_, screen_file_path_, 1);

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
    int index_origin_global = Rgrid__->find(Coordinate(0,0,0));
    HasOrigin_ = Rgrid__->mpindex.is_local(index_origin_global);
    if( HasOrigin_ ) {
        index_origin_local_ = Rgrid__->mpindex.glob1D_to_loc1D(index_origin_global);  
    }  

#ifdef EDUS_MPI
    /* get rank with origin in all the ranks */
    int HasOrigin_int = HasOrigin_ ? 1 : 0;
    output::print("has origin: ", (HasOrigin_ ? "true" : "false"));
    std::vector<int> rank_has_origin(kpool_comm->size());

    MPI_Allgather(&HasOrigin_int, 1, MPI_INT, rank_has_origin.data(), 1, MPI_INT, kpool_comm->communicator());    
    output::print("rank_has_origin[0]: ", (rank_has_origin[0] ? "true" : "false"));
    int root_origin = -1;
    for ( int irank = 0; irank < kpool_comm->size(); irank++ ) {
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
    Hartree.initialize({nbnd, nbnd});
    Hartree.fill(0.);
    for ( int iR = 0; iR < barecoulomb_.Potential_.get_Size(0); ++iR ) {
        for( int irow = 0; irow < nbnd; ++irow ) {
            for( int icol = 0; icol < nbnd; ++icol ) {
                Hartree(irow, icol) += barecoulomb_.Potential_(iR, irow, icol);
            }
        }
    }

#ifdef EDUS_MPI
    /* reduce the elements of the Hartree potential */
    if (kpool_comm->rank() == root_origin) {
        // Root process: in-place reduction
        MPI_Reduce(MPI_IN_PLACE, Hartree.data(),
               nbnd * nbnd, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
               root_origin, kpool_comm->communicator());
    } else {
        // Non-root processes: send their local Hartree
        MPI_Reduce(Hartree.data(), nullptr,
               nbnd * nbnd, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
               root_origin, kpool_comm->communicator());
    }
#endif

    /* make it hermitian (it must be mathematically) but is not numerically */
    for( int irow = 0; irow < nbnd; ++irow ) {
        for( int icol = irow+1; icol < nbnd; ++icol ) {
            auto value = (Hartree(irow, icol) + Hartree(icol, irow))/2.;
            Hartree(irow, icol) = value;
            Hartree(icol, irow) = value;
        }
    }

    /* make W is hermitian. From tests this modifies only last R */
    auto& W = screencoulomb_.Potential_;
    Operator<std::complex<double>> Wop; 
    Wop.initialize_fft(*Rgrid_, nbnd);
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

void Coulomb::set_read_interaction(const bool& read_interaction__)
{
    read_interaction_ = read_interaction__;
}

void Coulomb::set_bare_file_path(const std::string& bare_file_path__)
{
    bare_file_path_ = bare_file_path__;
}

void Coulomb::set_screen_file_path(const std::string& screen_file_path__)
{
    screen_file_path_ = screen_file_path__;
}

/// @brief Setter for DM0 (Density Matrix of the ground state at Wannier gauge in R)
/// @param DM0__ Value to use in the setter
void Coulomb::set_DM0( const Operator<std::complex<double>>& DM0__ )
{
    DM0_ = DM0__;
}

/// @brief Getter for DM0
/// @return DM0 
Operator<std::complex<double>>& Coulomb::get_DM0()
{
    return DM0_;
}


/// @brief Setter for Docoulomb variable of the class 
/// @param DoCoulomb__ Value to use in the setter
void Coulomb::set_DoCoulomb(const bool& DoCoulomb__)
{
    DoCoulomb_ = DoCoulomb__;
}

/// @brief Set Epsilon (macroscopic dielectric constant)
/// @param Epsilon__ Value we want to use as epsilon
void Coulomb::set_epsilon(const double& Epsilon__)
{
    screencoulomb_.set_epsilon( Epsilon__ );
}

/// @brief Set r0 in RytovaKeldysh model, not used otherwise
/// @param r0__ Value we want to use as r0 (in a.u.)
void Coulomb::set_r0(const std::vector<double>& r0__)
{
    screencoulomb_.set_r0( r0__ );
}

/// @brief Set the coulomb model type
/// @param model_name__ String name of the model: "vcoul3d" or "rytova_keldysh"
void Coulomb::set_coulomb_model(const std::string& model_name__)
{
    screencoulomb_.set_coulomb_model( model_name__ );
}

/// @brief Getter for r0 of Rytova Keldysh
///@return r0 in Rytova Keldysh, in a.u.
std::array<double, 3>& Coulomb::get_r0()
{
    return screencoulomb_.get_r0();
}

/// @brief Getter for r0_avg of Rytova Keldysh
///@return r0_avg in Rytova Keldysh, in a.u.
double Coulomb::get_r0_avg()
{
    return screencoulomb_.get_r0_avg();
}

void Coulomb::set_method(const std::string& method__)
{
    if (method__ == "ipa"){
        method_ = ipa;
    }
    else if (method__ == "rpa"){
        method_ = rpa;
    }
    else if (method__ == "hsex"){
        method_ = hsex;
    }
    else 
    {
        std::stringstream ss;
        ss << "Method given in input " << method__ << "not recognized." << std::endl;
        throw std::runtime_error(ss.str());
    }
}

/// @brief Getter for DoCoulomb variable 
/// @return DoCoulomb variable
const bool& Coulomb::get_DoCoulomb() const
{
    return DoCoulomb_;
}

/// @brief Getter for DoCoulomb variable 
/// @return DoCoulomb variable
bool& Coulomb::get_DoCoulomb()
{
    return DoCoulomb_;
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

/// @brief This function calculates the effective Hamiltonian from the Coulomb interaction. 
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
/// @param H__ Hamiltonian Operator, in which we want to add the effective Hamiltonian of the Coulomb interaction
/// @param DM__ Density matrix at current time, that we need to use to calculate the effective Coulomb interaction
/// @param EraseH__ True if we want to erase the H__ matrix before feeding it with the effective Coulomb interaction
void Coulomb::EffectiveHamiltonian(Operator<std::complex<double>>& H__, const Operator<std::complex<double>>& DM__,
                                  const bool& EraseH__ ) 
{
    /* We calculate the Coulomb interaction in R space */
    auto& HR__ = H__.get_Operator(R);
    auto& DMR__ = DM__.get_Operator(R);
    auto& DM0R_ = DM0_.get_Operator(R);

    if( EraseH__ ) {
        HR__.fill(0.);
    }

    if ( !DoCoulomb_ ) {
        return;
    }

    /* Hartree term */
    if( HasOrigin_  && (method_ == rpa || method_ == hsex)) { // Only the rank with R=0 contributes to this term 
        Hartree_interaction(HR__, Hartree, DMR__, DM0R_, index_origin_local_);
    }

    /* Fock term */
    if (method_ == hsex) {
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

mdarray<std::complex<double>,3>& Coulomb::get_ScreenedPotential()
{
    return screencoulomb_.Potential_;
}

std::string Coulomb::get_method()
{
    std::string method;
    if ( method_ == ipa ) {
        method = "ipa";
    } else if ( method_ == rpa ) {
        method = "rpa";
    } else if ( method_ == hsex ) {
        method = "hsex";
    }
    return method;
}

bool& Coulomb::get_read_interaction()
{
    return read_interaction_;
}

void Coulomb::initialize_device()
{
    Hartree.initialize_device();
    Hartree.transfer_to(device);
    screencoulomb_.Potential_.initialize_device();
    screencoulomb_.Potential_.transfer_to(device);
    DM0_.initialize_device();
    DM0_.transfer_to(device);
}
