#include "ModelCoulomb.hpp"
#include "StreamFile.hpp"
#include "Potentials.hpp"
#include <sstream>
#include <cmath>

CoulombModel model(const std::string& model_name__)
{
    if (model_name__ == "vcoul3d") {
        return CoulombModel::VCOUL3D;
    } else if (model_name__ == "rytova_keldysh") {
        return CoulombModel::RYTOVA_KELDYSH;
    } else {
        std::stringstream ss;
        ss << "Unknown coulomb model: '" << model_name__ << "'. Available models: vcoul3d, rytova_keldysh";
        throw std::runtime_error(ss.str());
    }
    return VCOUL3D;
}

/// @brief Constructor that calls the initialization of the class 
/// @param r Position operator read from Wannier90, in a.u.
/// @param MasterRGrid Grid in R space used in the code, (N.B: it is different than the one in r because that one is read from _tb file)
/// @param read_interaction__ true if interaction is read from a file
/// @param file_path__ where to find the file to read the interaction if read_interaction__=true
/// @param spin_deg__ can be 1 or 2
ModelCoulomb::ModelCoulomb(const std::array<Operator<std::complex<double>>,3>& r__,
                           const std::shared_ptr<MeshGrid>& MasterRGrid__, 
                           const bool& read_interaction__,
                           const std::string& file_path__, 
                           const int spin_deg__)
{
    initialize(r__, MasterRGrid__, read_interaction__, file_path__, spin_deg__);
}

/// @brief Initialization of matrix elements calling back the right function
void ModelCoulomb::initialize_Potential(const std::vector<Coordinate>& wannier_centers__)
{
    auto size_MG_global =  Rgrid_->get_TotalSize();
    auto nbnd = wannier_centers__.size();
    auto size_MG_local = Rgrid_->get_LocalSize();
    Potential_.initialize({int(size_MG_local), int(nbnd), int(nbnd)});

    #pragma omp parallel for
    for( int iR_local = 0; iR_local < size_MG_local; ++iR_local ) {
        
        auto iR_global = int( Rgrid_->mpindex.loc1D_to_glob1D(iR_local) );
        auto& R = (*Rgrid_)[iR_global];

        for(int in=0; in<Potential_.get_Size(1); in++) {
            auto r = wannier_centers__[in] - R;
            for(int im=0; im<Potential_.get_Size(2); im++){
                r = r - wannier_centers__[im];
                Potential_(iR_local, in, im) = double(spin_deg_)*Potentials_wrapper(r);
            }
        }
    }
}

/// @brief Initializing the interactions using the values inside a file
/// @param file_path__ file where the interaction is read
/// @param nbnd__ number of bands in the simulation
void ModelCoulomb::initialize_Potential(const std::string& file_path__, const int nbnd__)
{
    auto size_MG_local = Rgrid_->get_LocalSize();
    Potential_ = mdarray<std::complex<double>,3> ( { int( size_MG_local ), nbnd__, nbnd__ } );
    auto potential_file = ReadFile(file_path__);

    // read from the kcw file the R vectors where the potential is computed
    std::vector<Coordinate> bare_Rmesh;
    std::array<double,3> Rpoint;
    for (int iline = 0; iline < potential_file.size(); iline++) {
        if (potential_file[iline].size() == 3) {
            for (auto& ix : {0, 1, 2}) {
                Rpoint[ix] = stoi(potential_file[iline][ix]);}
                Coordinate R(Rpoint[0], Rpoint[1], Rpoint[2], LatticeVectors(Space::R));
                bare_Rmesh.push_back(R);
        }
    }

    // find in the systems grid the R vectors from the kcw file read above
    MeshGrid R_MeshGrid;
    R_MeshGrid.initialize(Space::R, bare_Rmesh, 0.0);
    auto Rgrid_shifted = get_GammaCentered_grid(*Rgrid_);
    MeshGrid::Calculate_ConvolutionIndex(R_MeshGrid, Rgrid_shifted, *Operator<std::complex<double>>::MeshGrid_Null);
    auto& ci = MeshGrid::ConvolutionIndex[{R_MeshGrid.get_id(), Rgrid_shifted.get_id(), Operator<std::complex<double>>::MeshGrid_Null->get_id()}];

    // build the coulomb interaction matrix elements in the imported R vectors
    Potential_.fill(0.0);
    for (int iRCoulomb=0; iRCoulomb<R_MeshGrid.get_TotalSize(); iRCoulomb++) {
        if ( Rgrid_->mpindex.is_local( ci(iRCoulomb,0) ) ) {
            auto iR_local = Rgrid_->mpindex.glob1D_to_loc1D( ci(iRCoulomb,0) );
            for (int irow=0; irow<nbnd__; irow++) {
                for (int icol=0; icol<nbnd__; icol++) {
                    int iline = nbnd__*2*irow + 2*icol + (std::pow(nbnd__,2)*2+1)*iRCoulomb + 1;
                    Potential_(iR_local, irow, icol) = spin_deg_*Convert(std::atof(potential_file[iline][3].c_str()), Rydberg, AuEnergy);
                    //std::cout << ci(iRCoulomb,0) << " " << ScreenedPotential_(ci(iRCoulomb,0), irow, icol) << std::endl;
                    //std::cout << "iRCoulomb = " << iRCoulomb << " | irow = " << irow + 1 << " | icol = " << icol + 1 << " potential = " << potential_file[iline][3].c_str() << std::endl;
                    //std::cout << (*Rgrid_)[ci(iRCoulomb,0)] << std::endl;
                }
            }
        }
    }
    output::print("coulomb read from file.");
    output::print("#R vectors in the file: ", int(bare_Rmesh.size()));
}

/// @brief Initialization of the variables of the class
/// @param r Position operator read from Wannier90, in a.u.
/// @param MasterRGrid Grid in R space used in the code, (N.B: it is different than the one in r because that one is read from _tb file)
/// @param read_interaction__ true if interaction is read from a file
/// @param file_path__ where to find the file to read the interaction if read_interaction__=true
/// @param spin_deg__ can be 1 or 2
void ModelCoulomb::initialize(const std::array<Operator<std::complex<double>>,3>& r__,
                           const std::shared_ptr<MeshGrid>& MasterRGrid__, 
                           const bool& read_interaction__,
                           const std::string& file_path__, 
                           const int spin_deg__)
{    
    spin_deg_ = spin_deg__;

    /* define wannier_centers from <n0|r|n0> */
    auto index_origin = r__[0].get_Operator_R().get_MeshGrid()->find(Coordinate(0,0,0));
    auto& x0 = (r__[0].get_Operator_R())[index_origin];
    auto& y0 = (r__[1].get_Operator_R())[index_origin];
    auto& z0 = (r__[2].get_Operator_R())[index_origin];

    auto nbnd = r__[0].get_Operator_R().get_nrows();
    std::vector<Coordinate> wannier_centers(nbnd);
    
    for(int in=0; in<nbnd; in++) {
        wannier_centers[in] = Coordinate(x0(in,in).real(), y0(in,in).real(), z0(in,in).real());
    }
    output::print("wannier centers");
    for(int in=0; in<nbnd; in++) {
       output::print(wannier_centers[in].get("Cartesian")[0], wannier_centers[in].get("Cartesian")[1], wannier_centers[in].get("Cartesian")[2]);
    }

    /* find minimum (non-zero) distance between wannier centers */
    min_distance_ = Coordinate(100,0,0,"Cartesian");
    min_distance_norm_ = min_distance_.norm();
    for(int in=0; in<nbnd; in++) {
        for(int im=in+1; im<nbnd; ++im) {
            auto distance = wannier_centers[in]-wannier_centers[im];
            if( distance.norm() < min_distance_norm_ && distance.norm() > 1.e-05 ) {
                min_distance_ = distance;
                min_distance_norm_ = min_distance_.norm();
            }
        }
    }

    /* Define grid centered at 0 */
    Rgrid_ = std::make_shared<MeshGrid>(get_GammaCentered_grid(*MasterRGrid__));

    /* initialize potential matrix elements */
    if (read_interaction__) {
        initialize_Potential(file_path__, nbnd );
    }
    else {
        initialize_Potential(wannier_centers);
    }
}

/// @brief Calculates the potential (bare/screened) at a point r__, using one of the model implemented
/// @param r__ Point in space on which we want to evaluate the potential
/// @param IsBare__ true if we are evaluating the bare potential
/// @return Value of the potential at r__
std::complex<double> ModelCoulomb::Potentials_wrapper(const Coordinate& r__)
{
    std::complex<double> V;
    switch(coulomb_model_)
    {
        case VCOUL3D:
        {
            auto r_norm = std::max( r__.norm(), min_distance_norm_ );
            V = pot::Coulomb(r_norm, Parameters_.epsilon);
            break;
        }
        case RYTOVA_KELDYSH:
        {
            auto& rcart = r__.get("Cartesian");
            auto r_reduced = Coordinate(rcart[0]/Parameters_.r0[0], rcart[1]/Parameters_.r0[1], rcart[2]/Parameters_.r0[2]); 
            auto r_norm = r_reduced.norm();
            if( r__.norm() < min_distance_norm_ ) {
                r_reduced = Coordinate(min_distance_.get("Cartesian")[0]/Parameters_.r0[0], 
                                       min_distance_.get("Cartesian")[1]/Parameters_.r0[1], 
                                       min_distance_.get("Cartesian")[2]/Parameters_.r0[2]);   
                r_norm = r_reduced.norm();
            }
            V = pot::RytovaKeldysh(r_norm, Parameters_.epsilon, Parameters_.r0_avg);
            break;
        }
        default:
        {
            throw std::runtime_error("Unknown coulomb model!");
            break;
        }
    }
    return V;
}

/// @brief Set Epsilon (macroscopic dielectric constant)
/// @param Epsilon__ Value we want to use as epsilon
void ModelCoulomb::set_epsilon(const double& Epsilon__)
{
    Parameters_.epsilon =Epsilon__;
}

/// @brief Set r0 in RytovaKeldysh model, not used otherwise
/// @param r0__ Value we want to use as r0 (in a.u.)
void ModelCoulomb::set_r0(const std::vector<double>& r0__)
{
    assert( r0__.size() > 0 && r0__.size() <= 3 );

    Parameters_.r0[0] = r0__[0];

    switch( r0__.size() )
    {
        case 1:
            /* if only one value is given in input, the thre components are the same*/
            Parameters_.r0[1] = r0__[0];
            Parameters_.r0[2] = r0__[0];
            break;
        case 2:
            /* if two values are given in input, the third component is the avg of the first two */
            Parameters_.r0[1] = r0__[1];
            Parameters_.r0[2] = (r0__[0] + r0__[1] )/2.;
            break;
        case 3:
            /* All three components are set from the input */
            Parameters_.r0[1] = r0__[1];
            Parameters_.r0[2] = r0__[2];
            break;
        default: 
            break;
    }
    
    Parameters_.r0_avg = ( Parameters_.r0[0] + Parameters_.r0[1] + Parameters_.r0[2] )/ 3.;
}

/// @brief Getter for r0 of Rytova Keldysh
///@return r0 in Rytova Keldysh, in a.u.
std::array<double, 3>& ModelCoulomb::get_r0()
{
    return Parameters_.r0;
}

/// @brief Getter for r0_avg of Rytova Keldysh
///@return r0_avg in Rytova Keldysh, in a.u.
double ModelCoulomb::get_r0_avg()
{
    return Parameters_.r0_avg;
}

mdarray<std::complex<double>,3>& ModelCoulomb::get_Potential()
{
    return Potential_;
}

/// @brief Set the coulomb model type from a string name
/// @param model_name__ String name of the model: "vcoul3d" or "rytova_keldysh"
void ModelCoulomb::set_coulomb_model(const std::string& model_name__)
{
    coulomb_model_ = model(model_name__);
}

