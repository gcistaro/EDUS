#include "Constants.hpp"
#include <fstream>

#ifndef GLOBALFUNCTIONS_HPP
#define GLOBALFUNCTIONS_HPP

#include <iostream>
#ifdef EDUS_HDF5
#include "core/hdf5/hdf5_tree.hpp"
#endif
#include <string>
#include "Json/json.hpp"
#include "core/mpi/Communicator.hpp"

std::string LatticeVectors(const Space& space);
bool file_exists(std::string file_name);
void dump_json_in_h5( const nlohmann::json& data__, const std::string& name__ );

namespace output {

inline void print(std::stringstream& ss__, const std::string& toprint__)
{
    ss__ << toprint__;
}

template <typename T>
inline void print(std::stringstream& ss__, const T& toprint__)
{            
/* note: this function might give errors because in cppreference they claim nothing ensures that hascode are unique! */
    const std::type_info& int_info    = typeid(0);
    const std::type_info& double_info = typeid(0.0);
    const std::type_info& float_info  = typeid(0.0f);
    const std::type_info& string_info = typeid(std::string(""));
    const char* a;
    const std::type_info& char_info   = typeid(a);
    
    auto hashcode__ = typeid(toprint__).hash_code();
    if ( int_info.hash_code() == hashcode__ ) {
        ss__ << std::right << std::setw(10) << toprint__;
    }
    else if ( double_info.hash_code() == hashcode__ ) {
        ss__ << std::right << std::scientific << std::setw(18) << std::setprecision(4) << toprint__;
    }
    else if ( float_info.hash_code() == hashcode__ ) {
        ss__ << std::right << std::scientific << std::setw(18) << std::setprecision(4) << toprint__;
    }
    else if ( char_info.hash_code() == hashcode__ ) {
        ss__ << toprint__;
    }
    else if ( string_info.hash_code() == hashcode__ ) {
        ss__ << toprint__;
    }    
    else {
        std::runtime_error("Tryng to print a variable that I don't know\n");
    }
}


template <typename... Args>
void print(Args... args__)
{
#ifdef EDUS_MPI
    if(mpi::Communicator::world().rank() == 0)
#endif
    {
        std::stringstream ss; 
        ss << '*' << std::string(3, ' '); 
        (print( ss, args__ ), ...);
        int num_spaces = output::linesize - int(ss.str().length()) -1;
        auto spaces =  ( num_spaces > 0 ) ? std::string(num_spaces, ' ') : std::string(""); 
        ss << spaces << '*' << '\n'; 
        std::cout << ss.str();
    }
}

void stars();
void title(const std::string& str__);

}


#endif