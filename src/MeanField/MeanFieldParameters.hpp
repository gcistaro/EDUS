#ifndef MEANFIELD_PARAMETERS_HPP
#define MEANFIELD_PARAMETERS_HPP

#include <string>
#include <vector>
#include <stdexcept>
#include "InputVariables/config.hpp"

namespace electron
{

    enum class MeanFieldMethod
    {
        IPA,
        RPA,
        HSEX
    };
    
    inline std::string to_string(MeanFieldMethod method)
    {
        switch (method) {
            case MeanFieldMethod::IPA:  return "ipa";
            case MeanFieldMethod::RPA:  return "rpa";
            case MeanFieldMethod::HSEX: return "hsex";
        }
        return "unknown";
    }

    struct MeanFieldParameters
    {
        /// Enable/disable mean-field corrections
        bool enabled = false;
        /// Approximation to use
        MeanFieldMethod method = MeanFieldMethod::IPA;
        /// Read interaction from file
        bool read_interaction = false;
        /// Background dielectric constant
        double epsilon = 1.0;
        /// Screening lengths (a.u.)
        std::vector<double> r0 = {0.0, 0.0, 0.0};
        /// Model for the screened interaction (e.g. "vcoul3d", "rytovakeldysh")
        std::string coulomb_model;
        /// Bare Coulomb file
        std::string bare_file;
        /// Screened Coulomb file
        std::string screen_file;  
        /// If true, the Hartree term uses the Wannier centers grouped by atom (hartree_centers = atoms)
        bool hartree_on_atoms = false;
        /// The Hartree interaction is saturated below hartree_cutoff_factor * d_min
        double hartree_cutoff_factor = 1.;
        /// Distance below which two Wannier centers belong to the same atom (a.u.)
        double hartree_center_tolerance = 0.6 / 0.529177210903;
    };

class MeanFieldParametersFactory
{
public:

    static MeanFieldParameters
    create(const config_t& cfg)
    {
        MeanFieldParameters p;

        p.enabled = cfg.coulomb();
        p.read_interaction = cfg.read_interaction();
        p.epsilon = cfg.epsilon();
        p.r0 = cfg.r0();
        p.coulomb_model = cfg.coulomb_model();
        p.bare_file = cfg.bare_file();
        p.screen_file = cfg.screen_file();
        if (cfg.hartree_centers() == "atoms")
            p.hartree_on_atoms = true;
        else if (cfg.hartree_centers() != "wannier")
            throw std::runtime_error("hartree_centers must be wannier or atoms");
        p.hartree_center_tolerance = cfg.hartree_center_tolerance() / 0.529177210903;
        p.hartree_cutoff_factor = cfg.hartree_cutoff_factor();
        if (p.hartree_cutoff_factor <= 0.)
            throw std::runtime_error("hartree_cutoff_factor must be positive");
        if (cfg.method() == "ipa")
            p.method = MeanFieldMethod::IPA;
        else if (cfg.method() == "rpa")
            p.method = MeanFieldMethod::RPA;
        else if (cfg.method() == "hsex")
            p.method = MeanFieldMethod::HSEX;
        else
            throw std::runtime_error(
                "Unknown mean-field method");

        return p;
    }
};

} // namespace electron

#endif