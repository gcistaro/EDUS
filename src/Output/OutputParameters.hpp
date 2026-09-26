#ifndef OUTPUT_PARAMETERS_HPP
#define OUTPUT_PARAMETERS_HPP

#include <string>
#include "InputVariables/config.hpp"

struct OutputParameters
{
    /// Print every printresolution steps when no laser is on (only for h5)
    int printresolution = 1;
    /// Print every printresolution_pulse steps when a laser is on (always used for txt files)
    int printresolution_pulse = 1;
    /// Number of filled bands: for them we print the population of holes, 1-rho_nn
    int filledbands = 0;
    /// Matrices to print in the h5 file
    bool print_DMk_wannier = false;
    bool print_DMk_bloch = false;
    bool print_SelfEnergy = false;
    bool print_fullH = false;
    /// Directory where all the output files are written
    std::string directory = "Output";
};

class OutputParametersFactory
{
public:
    static OutputParameters create(const config_t& cfg)
    {
        OutputParameters p;
        p.printresolution = cfg.printresolution();
        p.printresolution_pulse = ( cfg.printresolution_pulse() == 0 ? cfg.printresolution()
                                                                     : cfg.printresolution_pulse() );
        p.filledbands = cfg.filledbands();

        auto toprint = [&cfg](const std::string& key) {
            auto& dict = cfg.dict();
            return dict.contains("toprint") && dict.at("toprint").contains(key) && dict.at("toprint").at(key) == "true";
        };
        p.print_DMk_wannier = toprint("DMk_wannier");
        p.print_DMk_bloch = toprint("DMk_bloch");
        p.print_SelfEnergy = toprint("SelfEnergy");
        p.print_fullH = toprint("fullH");
        return p;
    }
};

#endif
