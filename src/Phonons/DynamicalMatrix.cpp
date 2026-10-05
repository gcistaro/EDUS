#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "Phonons/DynamicalMatrix.hpp"

namespace phonon {

/* Format of the text dynamical matrix written by ph.x (prefix.dynN), e.g.
       Dynamical matrix file
       <title>
         ntyp  nat  ibrav  celldm(1..6)
       [Basis vectors + 3 lines, only if ibrav = 0]
         itype  'label'  mass            <- one line per type, mass in Rydberg units (amu_ry)
         iatom  itype  tau(1..3)         <- one line per atom
            Dynamical  Matrix in cartesian axes
            q = (  q1  q2  q3 )
         na  nb                          <- nat*nat blocks: 3 lines of 3 complex numbers (re im), Ry/bohr^2
   Only the first q of the file is read (for the file of Gamma, q = 0). The matrix is C(q), the force constants,
   not yet divided by the masses. */
DynamicalMatrix read_dynamical_matrix(const std::string& filename__)
{
    std::ifstream file(filename__);
    if( !file ) {
        throw std::runtime_error("Cannot open the dynamical matrix file " + filename__ + "\n");
    }
    auto error = [&](const std::string& what) {
        return std::runtime_error("Error reading the dynamical matrix " + filename__ + ": " + what + "\n");
    };
    std::string line;
    /* "Dynamical matrix file" and the title */
    std::getline(file, line);
    std::getline(file, line);
    std::getline(file, line);
    int ntyp, nat, ibrav;
    {
        std::stringstream ss(line);
        if( !(ss >> ntyp >> nat >> ibrav) ) {
            throw error("wrong header");
        }
    }
    if( ibrav == 0 ) {
        /* "Basis vectors" and three lines */
        for( int i = 0; i < 4; ++i ) {
            std::getline(file, line);
        }
    }
    /* types: index 'label' mass (Ry units) */
    std::vector<double> type_mass(ntyp);
    for( int it = 0; it < ntyp; ++it ) {
        std::getline(file, line);
        auto last_quote = line.rfind('\'');
        if( last_quote == std::string::npos ) {
            throw error("wrong line of the atomic types: " + line);
        }
        type_mass[it] = std::stod(line.substr(last_quote + 1));
    }
    DynamicalMatrix dyn;
    dyn.num_atoms = nat;
    dyn.mass.resize(nat);
    for( int ia = 0; ia < nat; ++ia ) {
        std::getline(file, line);
        std::stringstream ss(line);
        int index, type;
        ss >> index >> type;
        if( type < 1 || type > ntyp ) {
            throw error("wrong type of atom " + std::to_string(ia + 1));
        }
        /* Ry mass unit = 2 electron masses */
        dyn.mass[ia] = 2. * type_mass[type - 1];
    }
    /* skip to the first q point */
    while( std::getline(file, line) && line.find("q = (") == std::string::npos ) {
    }
    if( !file ) {
        throw error("no q point found");
    }
    {
        auto open = line.find('(');
        std::stringstream ss(line.substr(open + 1));
        ss >> dyn.q[0] >> dyn.q[1] >> dyn.q[2];
    }
    dyn.C.initialize({3 * nat, 3 * nat});
    for( int block = 0; block < nat * nat; ++block ) {
        int na, nb;
        do {
            std::getline(file, line);
        } while( file && line.find_first_not_of(" \t\r") == std::string::npos );
        std::stringstream ss(line);
        if( !(ss >> na >> nb) ) {
            throw error("wrong block header: " + line);
        }
        for( int i = 0; i < 3; ++i ) {
            std::getline(file, line);
            std::stringstream row(line);
            for( int j = 0; j < 3; ++j ) {
                double re, im;
                if( !(row >> re >> im) ) {
                    throw error("wrong line of the block " + std::to_string(na) + " " + std::to_string(nb));
                }
                /* Ry/bohr^2 -> Ha/bohr^2 */
                dyn.C(3 * (na - 1) + i, 3 * (nb - 1) + j) = 0.5 * std::complex<double>(re, im);
            }
        }
    }
    return dyn;
}

/* A rigid translation u_(atom, beta) = constant costs no energy, so sum_atom' C(atom alpha, atom' beta) = 0 for each
   atom, alpha and beta. The numerical error of ph.x violates it slightly; the "simple" correction of matdyn subtracts
   the whole sum from the diagonal block (atom, atom). */
double impose_acoustic_sum_rule(DynamicalMatrix& dyn__)
{
    double largest = 0.;
    const int nat = dyn__.num_atoms;
    for( int ia = 0; ia < nat; ++ia ) {
        for( int i = 0; i < 3; ++i ) {
            for( int j = 0; j < 3; ++j ) {
                std::complex<double> sum = 0.;
                for( int ib = 0; ib < nat; ++ib ) {
                    sum += dyn__.C(3 * ia + i, 3 * ib + j);
                }
                dyn__.C(3 * ia + i, 3 * ia + j) -= sum;
                largest = std::max(largest, std::abs(sum));
            }
        }
    }
    return largest;
}

}
