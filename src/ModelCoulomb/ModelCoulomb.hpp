#include <math.h>
#include "Constants.hpp"
#include "ConvertUnits.hpp"
#include "Geometry/Coordinate.hpp"
#include "Operator/Operator.hpp"
#include "fftPair/fftPair.hpp"
#include <string>

#ifndef __MODELCOULOMB_HPP
#define __MODELCOULOMB_HPP


struct PotParameters
{
    std::array<double, 3> r0;
    double r0_avg;
    double epsilon;
};

enum CoulombModel {RYTOVA_KELDYSH, VCOUL3D};

CoulombModel model(const std::string& model__);

/// @brief Class for modelling matrix elements of Coulomb interaction in Point-Like approximation. We only store the non-zero elements
class ModelCoulomb
{
    private:
        /// Coulomb model type: RYTOVA_KELDYSH (2D screened) or VCOUL3D (3D bare/screened)
        CoulombModel coulomb_model_;
        /// Potential parameters for the interaction
        PotParameters Parameters_;
        /// The MeshGrid of the simulation in R space
        std::shared_ptr<MeshGrid> Rgrid_;
        /// minimum distance between two wannier centers, for saturations of bare and screened coulomb
        Coordinate min_distance_;
        /// Norm of the minimum distance
        double min_distance_norm_;
        /// Spin degeneracy to be used. For normal simulations, 2 for bare and 1 for screened
        int spin_deg_;
    public:
        ModelCoulomb(){};
        ModelCoulomb(const std::array<Operator<std::complex<double>>,3>& r__,
                     const std::shared_ptr<MeshGrid>& MasterRGrid__, 
                     const bool& read_interaction__,
                     const std::string& file_path__, 
                     const int spin_deg__);
        void initialize(const std::array<Operator<std::complex<double>>,3>& r__,
                        const std::shared_ptr<MeshGrid>& MasterRGrid__, 
                        const bool& read_interaction__,
                        const std::string& file_path__, 
                        const int spin_deg__);
        void initialize_Potential(const std::vector<Coordinate>& wannier_centers__);
        void initialize_Potential(const std::string& file_path__, const int nbnd__);
        std::complex<double> Potentials_wrapper(const Coordinate& r__);
        void set_epsilon(const double& Epsilon__);
        void set_r0(const std::vector<double>& r0__);
        void set_coulomb_model(const std::string& model_name__);
        std::array<double, 3>& get_r0();
        double get_r0_avg();
        /// The interaction: Potential_(iR, i, j)  = @f$ \langle i \textbf{0}, j \textbf{R}[iR] | V | i \textbf{0}, j \textbf{R}[iR] \rangle @f$
        mdarray<std::complex<double>,3> Potential_;
        mdarray<std::complex<double>,3>& get_Potential();
};

#endif