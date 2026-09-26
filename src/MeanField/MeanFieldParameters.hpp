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