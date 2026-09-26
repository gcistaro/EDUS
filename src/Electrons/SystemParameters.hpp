#ifndef SYSTEM_PARAMETERS_HPP
#define SYSTEM_PARAMETERS_HPP

#include <vector>
#include <stdexcept>
#include "InputVariables/config.hpp"

namespace electron
{
    struct SystemParameters
    {
        /// Number of bands fully occupied at equilibrium
        int filledbands = 0;
        /// Occupation of each band at equilibrium
        std::vector<double> occupations;
        /// Energy (a.u.) added to the gap: valence bands are shifted by -opengap/2, conduction bands by +opengap/2
        double opengap = 0.;
    };

    class SystemParametersFactory
    {
    public:
        /// opengap is expected already converted in atomic units
        static SystemParameters create(const config_t& cfg, const int num_bands)
        {
            SystemParameters p;
            p.filledbands = cfg.filledbands();
            if ( p.filledbands > num_bands ) {
                throw std::runtime_error("filledbands is larger than the number of bands");
            }
            /* filled bands are fully occupied */
            p.occupations = std::vector<double>(num_bands, 0.);
            for (int ib = 0; ib < p.filledbands; ++ib) {
                p.occupations[ib] = 1.;
            }
            p.opengap = cfg.opengap();
            return p;
        }
    };
} // namespace electron

#endif
