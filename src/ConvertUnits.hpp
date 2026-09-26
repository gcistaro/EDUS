#ifndef CONVERTUNITS_HPP
#define CONVERTUNITS_HPP

#include <map>
#include <string>
#include <cassert>
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


template<typename T>
T Convert(const T& ConvertableValue, const Unit& InputUnit, const Unit& OutputUnit)
{
    assert(InputUnit.type == OutputUnit.type);
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

Unit unit( const std::string& to_unit);

#endif