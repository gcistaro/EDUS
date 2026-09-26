#include "Wannier/Wannier.hpp"
/// @brief This constructor sets all the variables of the class, reading them 
/// from the $FilenName_tb.dat file 
/// @param FileName seedname of wannier90
Wannier::Wannier(const std::string& FileName)
{
    PROFILE("Wannier");
    std::stringstream TotalFileName;
    TotalFileName << FileName << "_tb.dat";    
    ParseWannier(TotalFileName.str(), NumberOfBands, NumberOfRpoints,
                  UnitCell, Degeneracy, Rmesh, H, r);
}

/// @brief This method can be used for check of validity. After constructing the wannier
/// class, you can print the values in the _tb.dat format using this method.
/// @param FileName seedname of wannier90
void Wannier::Print(const std::string& FileName)
{
    wann::print(FileName, NumberOfBands, NumberOfRpoints, 
                UnitCell, Degeneracy, Rmesh, H, r);
}
