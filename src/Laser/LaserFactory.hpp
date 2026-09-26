#ifndef LASER_FACTORY_HPP
#define LASER_FACTORY_HPP

#include <stdexcept>
#include "Laser/Laser.hpp"
#include "InputVariables/config.hpp"

class LaserFactory
{
public:
    /// Builds the set of lasers described in the input.
    /// cfg is not const only because config_t has no const accessor for the lasers: it is not modified.
    static SetOfLaser create(config_t& cfg)
    {
        SetOfLaser setoflaser;
        for (int ilaser = 0; ilaser < int(cfg.lasers().size()); ++ilaser) {
            auto currentdata = cfg.lasers(ilaser);
            Laser laser;

            laser.set_InitialTime(currentdata.t0(), unit(currentdata.t0_units()));
            laser.set_Intensity(currentdata.intensity(), unit(currentdata.intensity_units()));
            /* one between frequency and wavelength is given, the other is calculated by Laser */
            if ( is_frequency(currentdata) ) {
                laser.set_Omega(currentdata.frequency(), unit(currentdata.frequency_units()));
            } else {
                laser.set_Lambda(currentdata.wavelength(), unit(currentdata.wavelength_units()));
            }
            laser.set_NumberOfCycles(currentdata.cycles());
            laser.set_Phase(currentdata.phase());
            Coordinate pol(currentdata.polarization()[0], currentdata.polarization()[1],
                currentdata.polarization()[2]);
            pol = pol / pol.norm();
            laser.set_Polarization(pol);
            setoflaser.push_back(laser);
        }
        return setoflaser;
    }

private:
    /// Checks if the laser is defined in input through its frequency (true) or wavelength (false)
    template <typename laser_input_t>
    static bool is_frequency(laser_input_t& laser_input)
    {
        if ( std::abs(laser_input.frequency()) > 1.e-07 ) {
            return true;
        }
        if ( !laser_input.wavelength() ) {
            throw std::runtime_error("You must specify (nonzero) frequency *xor* wavelength!");
        }
        return false;
    }
};

#endif
