//each function reads part of wannier "_tb.dat" file
#ifndef READWANNIER_HPP
#define READWANNIER_HPP


#include <stdlib.h>//for atof and atoi

#include "Constants.hpp"
#include "mdContainers/mdContainers.hpp"

//each function reads part of wannier "_tb.dat" file

int ParseWannier_Degeneracies(const std::vector<std::vector<std::string>>::iterator& LineIterator_Begin, 
                              std::vector<int>& Degeneracy);

int ParseWannier_MatrixElement(const std::vector<std::vector<std::string>>::iterator& LineIterator, 
                                std::complex<double>* Matrix_, const int& NumberOfBands);

int ParseWannier_MatrixElement(const std::vector<std::vector<std::string>>::iterator& LineIterator, 
                                std::complex<double>* Matrix0, std::complex<double>* Matrix1, 
                                std::complex<double>* Matrix2, const int& NumberOfBands);

int ParseWannier_Matrix(const std::vector<std::vector<std::string>>::iterator LineIterator_begin, 
                        double* R, std::complex<double>* Matrix_, const int& NumberOfBands);

int ParseWannier_Matrix(const std::vector<std::vector<std::string>>::iterator LineIterator_begin, 
                        const std::vector<std::vector<std::string>>::iterator& LineIterator_end,
                        double* R, std::complex<double>* Matrix0, std::complex<double>* Matrix1, 
                        std::complex<double>* Matrix2, const int& NumberOfBands);

int ParseWannier_Hamiltonian(const std::vector<std::vector<std::string>>::iterator LineIterator_begin, 
                            mdarray<double,2>& Rmesh, mdarray<std::complex<double>, 3>& H, 
                            const int& NumberOfBands);

int ParseWannier_PositionOperator(const std::vector<std::vector<std::string>>::iterator& LineIterator_begin,
                 const std::vector<std::vector<std::string>>::iterator& LineIterator_end,
                 mdarray<double,2>& Rmesh, std::array<mdarray<std::complex<double>,3>, 3>& r, 
                 const int& NumberOfBands);

void ParseWannier(const std::string& FileNameTB, int& NumberOfBands, int& NumberOfRpoints,
                  mdarray<double,2>& UnitCell, std::vector<int>& Degeneracy, mdarray<double,2>& Rmesh, 
                  mdarray<std::complex<double>, 3>& H, std::array<mdarray<std::complex<double>,3>, 3>& r);
#endif
