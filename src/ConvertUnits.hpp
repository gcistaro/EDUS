#ifndef CONVERTUNITS_HPP
#define CONVERTUNITS_HPP

#include <map>
#include <string>
#include <cassert>
#include <stdexcept>
#include "Constants.hpp"

enum  Type{LENGTH, TIME, ENERGY, INTENSITY, NullType};

struct Unit{
    Type type;
    double value;
    
    Unit(const Type& type_, const double& value_);
};

extern Unit Angstrom;
extern Unit NanoMeters;
extern Unit ElectronVolt;
extern Unit Joule;
extern Unit Wcm2;
extern Unit FemtoSeconds;
extern Unit Rydberg;

extern Unit AuIntensity;
extern Unit AuTime;
extern Unit AuLength;
extern Unit AuEnergy;


/// Name of the physical quantity of a unit type, for the error messages
std::string type_name(const Type& type__);

template<typename T>
T Convert(const T& ConvertableValue, const Unit& InputUnit, const Unit& OutputUnit)
{
    if( InputUnit.type != OutputUnit.type ) {
        throw std::runtime_error("Convert: a unit of " + type_name(InputUnit.type) + " cannot be converted to a unit of "
                                 + type_name(OutputUnit.type) + " (check the units in the input)\n");
    }
    return ConvertableValue/OutputUnit.value*InputUnit.value; 
}

//we need a concept with iterators!!
template<typename T>
void Convert_iterable(T& ConvertableTensor, const Unit& InputUnit, const Unit& OutputUnit)
{
    for(auto& ToConvert : ConvertableTensor){
        ToConvert = Convert(ToConvert, InputUnit, OutputUnit);
    }
}

/// Unit from its name in the input; throws if the name is unknown
Unit unit( const std::string& to_unit);

#endif