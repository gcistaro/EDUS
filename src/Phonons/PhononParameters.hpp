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
/// generate again: only the excited part of the density matrix, rho - rho0 - Delta rho_BO(u), must push the atoms.
enum class AdiabaticReference
{
    /// the force of the whole rho - rho0 with the force constants of ph.x (the response is counted twice)
    none,
    /// Delta rho_BO = sum_mu u_mu delta rho_mu, with the static linear response delta rho_mu of the electrons of the
    /// model (mean field included) computed at the beginning: equivalent to the force constants K - Pi(0). The
    /// frequencies are those of the non-adiabatic theory, M omega^2 = K + Pi(omega) - Pi(0) (default)
    static_response,
    /// rho_BO propagated in time with the same equation of rho, the same displacement u(t) and no laser. rho_BO is the
    /// whole response to u(t), not the adiabatic one: without laser the lattice oscillates exactly with K, so the
    /// non-adiabatic correction Pi(omega) - Pi(0) of the ground state is lost (small in insulators, (omega/gap)^2)
    dynamic
};

inline std::string to_string(AdiabaticReference reference__)
{
    switch (reference__) {
        case AdiabaticReference::none:            return "none";
        case AdiabaticReference::static_response: return "static";
        case AdiabaticReference::dynamic:         return "dynamic";
    }
    return "unknown";
}

/// @brief Which electron-phonon coupling the files of EPW contain.
/// The mean field of EDUS (Hartree + SEX) generates during the propagation the screening of the electrons of the model,
/// so the coupling in the Hamiltonian must not contain it (see src/Phonons/EHRENFEST.md section 3.1).
enum class CouplingType
{
    /// standard EPW run (dvscf of ph.x): the screening of the model is removed at the beginning,
    /// g_b = g_s - Sigma[chi0 g_s] (with the mean field; without it g_s is used as it is)
    screened,
    /// the coupling of the files is used as it is (already without the screening of the model, or synthetic tests)
    bare
};

inline std::string to_string(CouplingType coupling__)
{
    switch (coupling__) {
        case CouplingType::screened: return "screened";
        case CouplingType::bare:     return "bare";
    }
    return "unknown";
}

struct PhononParameters
{
    /// If true, the lattice is propagated together with the electrons (Ehrenfest dynamics)
    bool enabled = false;
    /// Directory of the EPW calculation, with epwdata.fmt, wigner.fmt and crystal.fmt
    std::string epw_directory;
    /// File with the electron-phonon matrix elements in the Wannier representation (prefix.epmatwp)
    std::string epmatwp;
    /// Dynamical matrix of ph.x at Gamma (text format), for the masses and the force constants
    std::string dyn_file;
    /// Phonon wavevectors (crystal coordinates): for now only Gamma
    std::vector<std::array<double, 3>> qpoints = {{0., 0., 0.}};
    /// Spin degeneracy of the bands: the force on the ions is summed over the spin channels
    double spin_degeneracy = 2.;
    /// Decay time of the amplitude of the oscillations (a.u.), phenomenological. Ignored if ~0
    double damping_time = 0.;
    /// Coupling contained in the files of EPW
    CouplingType coupling = CouplingType::screened;
    /// Factor multiplying the electron-phonon coupling (1 = coupling of EPW, 0 = no coupling)
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
};

class PhononParametersFactory
{
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
        auto coupling = cfg.phonons().coupling();
        if( coupling == "screened" ) {
            p.coupling = CouplingType::screened;
        }
        else if( coupling == "bare" ) {
            p.coupling = CouplingType::bare;
        }
        else {
            throw std::runtime_error("phonons/coupling must be screened or bare\n");
        }
        p.acoustic_sum_rule = cfg.phonons().acoustic_sum_rule();
        auto reference = cfg.phonons().adiabatic_reference();
        if( reference == "none" ) {
            p.adiabatic_reference = AdiabaticReference::none;
        }
        else if( reference == "static" ) {
            p.adiabatic_reference = AdiabaticReference::static_response;
        }
        else if( reference == "dynamic" ) {
            p.adiabatic_reference = AdiabaticReference::dynamic;
        }
        else {
            throw std::runtime_error("phonons/adiabatic_reference must be none, static or dynamic\n");
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
            std::vector<std::string> found;
            if( std::filesystem::is_directory(p.epw_directory) ) {
                for( auto& entry : std::filesystem::directory_iterator(p.epw_directory) ) {
                    if( entry.path().extension() == ".epmatwp" ) {
                        found.push_back(entry.path().filename().string());
                    }
                }
            }
            if( found.size() != 1 ) {
                throw std::runtime_error("phonons/epmatwp: set the name of the file prefix.epmatwp (" + std::to_string(found.size())
                                         + " files *.epmatwp in " + p.epw_directory + ")\n");
            }
            p.epmatwp = found[0];
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
