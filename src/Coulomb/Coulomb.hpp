
#include "mdContainers/mdContainers.hpp"
#include "MeshGrid/MeshGrid.hpp"
#include "Operator/Operator.hpp"
#include "StreamFile.hpp"
#include "ModelCoulomb/ModelCoulomb.hpp"

/// @brief Class that takes care of the calculation of the effective Hamiltonian during time propagation, with the coulomb interaction matrix elements
class Coulomb 
{
    private:
        /// Grid in R space used during the simulation
        std::shared_ptr<MeshGrid> Rgrid_;
        /// Density matrix in wannier gauge for the ground state
        Operator<std::complex<double>> DM0_;
        /// Boolean that is set to true if in the simulation we want to include the coulomb interaction in the Hamiltonian
        bool DoCoulomb_ = false;
        /// Wrapper for bare interaction
        ModelCoulomb barecoulomb_;
        /// Wrapper for screened interaction
        ModelCoulomb screencoulomb_;
        /// True if the current rank propagates R=0 term
        bool HasOrigin_;
        /// Index of the origin R=0 inside the current rank
        int index_origin_local_ = -1;
        /// The effective Hartree potential, defined as: @f[ H_{nm} = \sum_\textbf{R} V_{nm}(\textbf{R}) = 
        /// \sum_\textbf{R} \langle n\textbf{0}m\textbf{R}|V(r-r')| n\textbf{0}m\textbf{R} \rangle @f]
        mdarray<std::complex<double>, 2> Hartree;
        enum Method{ipa, rpa, hsex} method_;
        /// True if interactions are read from file. False if not.
        bool read_interaction_;
        /// Paths of the read interactions
        std::string bare_file_path_;
        std::string screen_file_path_;
    public:
        Coulomb(){};
        Coulomb(const int& nbnd, const std::shared_ptr<MeshGrid>& Rgrid__, const std::array<Operator<std::complex<double>>,3>& r);
        void initialize(const int& nbnd, const std::shared_ptr<MeshGrid>& Rgrid__, const std::array<Operator<std::complex<double>>,3>& r);
        void EffectiveHamiltonian(Operator<std::complex<double>>& H__, const Operator<std::complex<double>>& DM__,
                                  const bool& EraseH__);     
        void set_DM0( const Operator<std::complex<double>>& DM0__ );
        Operator<std::complex<double>>& get_DM0();
        void set_DoCoulomb(const bool& DoCoulomb__);
        void set_epsilon(const double& Epsilon__);
        void set_r0(const std::vector<double>& r0__);
        void set_coulomb_model(const std::string& model_name__);
        const bool& get_DoCoulomb() const;
        bool& get_DoCoulomb();
        std::string get_method();
        bool& get_read_interaction();
        void set_method(const std::string& method__);

        mdarray<std::complex<double>,3>& get_ScreenedPotential();
        std::array<double, 3>& get_r0();
        double get_r0_avg();
        void set_read_interaction(const bool& read_interaction__);
        void set_bare_file_path(const std::string& bare_file_path__);
        void set_screen_file_path(const std::string& screen_file_path__);
        
        void initialize_device();
};

void Hartree_interaction_gpu ( std::complex<double>* SigmaH, 
                               const std::complex<double>* Hartree, 
                               const std::complex<double>* DM, 
                               const std::complex<double>* DM0, 
                               int index, 
                               int N);


