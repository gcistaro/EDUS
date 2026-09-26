#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <complex>
#include <map>

constexpr std::complex<double> im(0.,1.);
constexpr double pi = 3.14159265358979323846;


//CODATA values
constexpr double Bohr_value            = 5.2917721090380*1.e-11;//m
constexpr double ElectronVolt_value    = 1.602176634*1.e-19; //J
constexpr double Rydberg_value         = 2.1798723611035*1.e-18; //J
constexpr double AtomicUnitOfTime      = 2.4188843265857*1.e-17; //s
constexpr double Epsilon0              = 8.8541878188*1.e-12; //F m^-1
constexpr double E0                    = 5.14220674763*1.e+11;//V m^-1 (Atomic units of electric field) 
constexpr double FineStructure         = 7.2973525693*1.e-03;//pure units

constexpr double SpeedOfLight_SI       = 299792458; // m s^-1 
constexpr double SpeedOfLight          = 1./FineStructure; // a.u. 
constexpr double AtomicUnitOfIntensity = .5*SpeedOfLight_SI*Epsilon0*E0*E0; //W/m2
constexpr double threshold = 1.e-08;

enum Space{k,R};
enum BandGauge{bloch, wannier};
enum SolverType{AB, RK, MAGNUS};
const std::map<std::string, SolverType> solver = {{"AB", SolverType::AB}, {"RK", SolverType::RK}, {"Magnus", SolverType::MAGNUS}};

//Global functions
std::string LatticeVectors(const Space& space);


namespace nodename {
    const std::string H0         = "H0";
    const std::string H0pCoulomb = "H0+Coulomb";
    const std::string fullH      = "fullH";
    const std::string DMk        = "DensityMatrix_k";
    const std::string DMk_bloch  = "DensityMatrix_k_bloch";
    const std::string SelfEnergy = "SelfEnergy";
    const std::string time_au    = "time_au";
    const std::string G_lesser   = "G_lesser";
};//end namespace nodename 


namespace output {
    constexpr int linesize = 125;
}
#endif
