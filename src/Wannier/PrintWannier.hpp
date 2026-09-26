#ifndef PRINTWANNIER_HPP
#define PRINTWANNIER_HPP

#include <string> 
#include <fstream> 
#include <iomanip>
#include <complex>

#include "mdContainers/mdContainers.hpp"
namespace wann{
void print(const std::string& FileName, const int& NumberOfBands, const int& NumberOfRpoints, 
                   const mdarray<double,2>& UnitCell, const std::vector<int>& Degeneracy, const mdarray<double,2>& Rmesh, 
                   const mdarray<std::complex<double>,3>& H, const std::array<mdarray<std::complex<double>,3>, 3>& r);
} //end namespace wann 
#endif