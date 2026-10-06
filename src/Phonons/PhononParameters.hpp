#ifndef PHONON_PARAMETERS_HPP
#define PHONON_PARAMETERS_HPP

#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include "InputVariables/config.hpp"

namespace phonon {

/// @brief How the adiabatic (Born-Oppenheimer) response of the electrons to the lattice is treated.
/// The force constants of ph.x already contain the static screening of the electrons, which the propagated electrons
/// generate again: only the excited part of the density matrix, rho - rho_BO(u), pushes the atoms.
enum class AdiabaticReference
{
    /// rho_BO = rho0 + sum_mu u_mu delta rho_mu, with the static linear response delta rho_mu of the electrons of the
    /// model computed at the beginning. The force is F[rho - rho0] - Pi u: the frequencies are those of the
    /// non-adiabatic theory, M omega^2 = K + Pi(omega) - Pi(0) (default)
    static_response,
    /// rho_BO propagated in time with the same equation of rho, the same displacement u(t) and no laser. rho_BO is the
    /// whole response to u(t), not the adiabatic one: without laser the lattice oscillates exactly with K, so the
    /// non-adiabatic correction Pi(omega) - Pi(0) of the ground state is lost (small in insulators, (omega/gap)^2)
    dynamic
};

inline std::string to_string(AdiabaticReference reference__)
{
    switch (reference__) {
        case AdiabaticReference::static_response: return "static";
        case AdiabaticReference::dynamic:         return "dynamic";
    }
    return "unknown";
}

struct PhononParameters
{
    /// If true, the lattice is propagated together with the electrons (Ehrenfest dynamics)
    bool enabled = false;
    /// Directory of the EPW calculation of the bare coupling g_b, with epwdata.fmt, wigner.fmt and crystal.fmt
    std::string epw_directory;
    /// File with the electron-phonon matrix elements in the Wannier representation (prefix.epmatwp), bare coupling
    std::string epmatwp;
    /// Directory and file of the EPW calculation of the screened coupling g_s (optional). If given, the electrons
    /// feel u.g_s with the mean field Sigma[rho - rho_BO]; otherwise u.g_b with Sigma[rho - rho0].
    /// The force on the lattice always uses g_b
    std::string epw_directory_screened;
    std::string epmatwp_screened;
    /// Dynamical matrix of ph.x at Gamma (text format), for the masses and the force constants
    std::string dyn_file;
    /// Phonon wavevectors (crystal coordinates): for now only Gamma
    std::vector<std::array<double, 3>> qpoints = {{0., 0., 0.}};
    /// Spin degeneracy of the bands: the force on the ions is summed over the spin channels
    double spin_degeneracy = 2.;
    /// Decay time of the amplitude of the oscillations (a.u.), phenomenological. Ignored if ~0
    double damping_time = 0.;
    /// Factor multiplying the electron-phonon couplings (1 = couplings of EPW, 0 = no coupling)
    double coupling_scale = 1.;
    /// If true, the acoustic sum rule is imposed on the force constants, and the rigid translation is removed from the
    /// coupling (the electrons do not move the center of mass, and Pi satisfies the sum rule)
    bool acoustic_sum_rule = true;
    /// Adiabatic response of the electrons removed from the force on the lattice
    AdiabaticReference adiabatic_reference = AdiabaticReference::static_response;
    /// Static response (Anderson mixing): largest relative residual, and maximum number of iterations
    double response_tolerance = 1.e-10;
    int response_max_iterations = 500;
    /// Initial displacement of the atoms (a.u.), 3*atom + direction; empty = lattice at equilibrium
    std::vector<double> initial_displacement;

    /// True if the screened coupling is given (two calculations of EPW)
    bool screened() const { return !epw_directory_screened.empty(); }
};

class PhononParametersFactory
{
private:
    /// The only file *.epmatwp of directory__ (option__ is the name of the input variable, for the error)
    static std::string find_epmatwp(const std::string& directory__, const std::string& option__)
    {
        std::vector<std::string> found;
        if( std::filesystem::is_directory(directory__) ) {
            for( auto& entry : std::filesystem::directory_iterator(directory__) ) {
                if( entry.path().extension() == ".epmatwp" ) {
                    found.push_back(entry.path().filename().string());
                }
            }
        }
        if( found.size() != 1 ) {
            throw std::runtime_error("phonons/" + option__ + ": set the name of the file prefix.epmatwp ("
                                     + std::to_string(found.size()) + " files *.epmatwp in " + directory__ + ")\n");
        }
        return found[0];
    }

public:
    /// Times and lengths are expected already converted in atomic units
    static PhononParameters create(const config_t& cfg)
    {
        PhononParameters p;
        p.enabled = cfg.phonons().enabled();
        if( !p.enabled ) {
            return p;
        }
        p.epw_directory = cfg.phonons().epw_directory();
        p.epmatwp = cfg.phonons().epmatwp();
        p.dyn_file = cfg.phonons().dyn_file();
        p.spin_degeneracy = cfg.phonons().spin_degeneracy();
        p.damping_time = cfg.phonons().damping_time();
        p.coupling_scale = cfg.phonons().coupling_scale();
        p.epw_directory_screened = cfg.phonons().epw_directory_screened();
        p.epmatwp_screened = cfg.phonons().epmatwp_screened();
        p.acoustic_sum_rule = cfg.phonons().acoustic_sum_rule();
        auto reference = cfg.phonons().adiabatic_reference();
        if( reference == "static" ) {
            p.adiabatic_reference = AdiabaticReference::static_response;
        }
        else if( reference == "dynamic" ) {
            p.adiabatic_reference = AdiabaticReference::dynamic;
        }
        else {
            throw std::runtime_error("phonons/adiabatic_reference must be static or dynamic\n");
        }
        p.initial_displacement = cfg.phonons().initial_displacement();

        p.qpoints.clear();
        for( auto& q : cfg.phonons().qpoints() ) {
            if( q.size() != 3 ) {
                throw std::runtime_error("phonons/qpoints: each q point needs 3 crystal coordinates\n");
            }
            p.qpoints.push_back({q[0], q[1], q[2]});
        }
        if( p.qpoints.size() != 1 || std::abs(p.qpoints[0][0]) + std::abs(p.qpoints[0][1]) + std::abs(p.qpoints[0][2]) > 1.e-10 ) {
            throw std::runtime_error("phonons/qpoints: only q = Gamma is implemented. The lattice at q != 0 needs "
                                     "the coherences between different k points of the electrons (see src/Phonons/EHRENFEST.md)\n");
        }

        /* the name of the epmatwp file is prefix.epmatwp: if not given, look for the only one in the directory */
        if( p.epmatwp.empty() ) {
            p.epmatwp = find_epmatwp(p.epw_directory, "epmatwp");
        }
        if( !p.epw_directory_screened.empty() && p.epmatwp_screened.empty() ) {
            p.epmatwp_screened = find_epmatwp(p.epw_directory_screened, "epmatwp_screened");
        }
        if( p.dyn_file.empty() ) {
            throw std::runtime_error("phonons/dyn_file: the dynamical matrix at Gamma of ph.x is needed for the masses and "
                                     "the force constants\n");
        }
        if( p.spin_degeneracy <= 0. ) {
            throw std::runtime_error("phonons/spin_degeneracy must be positive\n");
        }
        return p;
    }
};

}

#endif
