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

/* Format of the file of q2r.x (text, see q2r.f90 / do_q2r):
         ntyp  nat  ibrav  celldm(1..6)
       [3 lines with the lattice vectors, only if ibrav = 0]
         itype  'label'  mass            <- one line per type, mass in Rydberg units (amu_ry)
         iatom  itype  tau(1..3)         <- one line per atom
         lrigid                          <- T if polar: then 3 lines of epsilon and, per atom, its index and 3 lines of Z*
         nr1 nr2 nr3
       then for j1, j2 = 1..3 and na1, na2 = 1..nat (j1 outermost):
         j1 j2 na1 na2
         m1 m2 m3 C          <- nr1*nr2*nr3 lines, m1 fastest, Ry/bohr^2 */
ForceConstants read_force_constants(const std::string& filename__)
{
    std::ifstream file(filename__);
    if( !file ) {
        throw std::runtime_error("Cannot open the file of the force constants " + filename__ + "\n");
    }
    auto error = [&](const std::string& what) {
        return std::runtime_error("Error reading the force constants " + filename__ + ": " + what + "\n");
    };
    std::string line;
    int ntyp, nat, ibrav;
    {
        std::getline(file, line);
        std::stringstream ss(line);
        if( !(ss >> ntyp >> nat >> ibrav) ) {
            throw error("wrong header: " + line);
        }
    }
    if( ibrav == 0 ) {
        for( int i = 0; i < 3; ++i ) {
            std::getline(file, line);
        }
    }
    std::vector<double> type_mass(ntyp);
    for( int it = 0; it < ntyp; ++it ) {
        std::getline(file, line);
        auto last_quote = line.rfind('\'');
        if( last_quote == std::string::npos ) {
            throw error("wrong line of the atomic types: " + line);
        }
        type_mass[it] = std::stod(line.substr(last_quote + 1));
    }
    ForceConstants fc;
    fc.num_atoms = nat;
    fc.mass.resize(nat);
    for( int ia = 0; ia < nat; ++ia ) {
        std::getline(file, line);
        std::stringstream ss(line);
        int index, type;
        if( !(ss >> index >> type) || type < 1 || type > ntyp ) {
            throw error("wrong line of the atom " + std::to_string(ia + 1) + ": " + line);
        }
        /* Ry mass unit = 2 electron masses */
        fc.mass[ia] = 2. * type_mass[type - 1];
    }
    /* "lrigid" or, in recent versions of QE, "lrigid alpha" (the Ewald parameter of the long-range part) */
    std::string lrigid;
    std::getline(file, line);
    {
        std::stringstream ss(line);
        ss >> lrigid;
    }
    fc.polar = ( lrigid.find('T') != std::string::npos || lrigid.find('t') != std::string::npos );
    if( fc.polar ) {
        /* epsilon (9 numbers), then per atom its index and Z* (9 numbers): not used for now */
        double x;
        for( int i = 0; i < 9; ++i ) {
            file >> x;
        }
        for( int ia = 0; ia < nat; ++ia ) {
            int index;
            file >> index;
            for( int i = 0; i < 9; ++i ) {
                file >> x;
            }
        }
    }
    if( !(file >> fc.nr[0] >> fc.nr[1] >> fc.nr[2]) ) {
        throw error("cannot read nr1 nr2 nr3");
    }
    const int nmodes = 3 * nat;
    fc.C.initialize({fc.nr[0], fc.nr[1], fc.nr[2], nmodes, nmodes});
    fc.C.fill(0.);
    for( int block = 0; block < 9 * nat * nat; ++block ) {
        int j1, j2, na1, na2;
        if( !(file >> j1 >> j2 >> na1 >> na2) || j1 < 1 || j1 > 3 || j2 < 1 || j2 > 3 || na1 < 1 || na1 > nat
            || na2 < 1 || na2 > nat ) {
            throw error("wrong header of the block " + std::to_string(block + 1));
        }
        const int mu = 3 * (na1 - 1) + (j1 - 1);
        const int nu = 3 * (na2 - 1) + (j2 - 1);
        for( int n = 0; n < fc.nr[0] * fc.nr[1] * fc.nr[2]; ++n ) {
            int m1, m2, m3;
            double value;
            if( !(file >> m1 >> m2 >> m3 >> value) || m1 < 1 || m1 > fc.nr[0] || m2 < 1 || m2 > fc.nr[1] || m3 < 1
                || m3 > fc.nr[2] ) {
                throw error("wrong line in the block " + std::to_string(block + 1));
            }
            /* Ry/bohr^2 -> Ha/bohr^2 */
            fc.C(m1 - 1, m2 - 1, m3 - 1, mu, nu) = 0.5 * value;
        }
    }
    return fc;
}

double impose_acoustic_sum_rule(ForceConstants& fc__)
{
    double largest = 0.;
    const int nat = fc__.num_atoms;
    for( int ia = 0; ia < nat; ++ia ) {
        for( int i = 0; i < 3; ++i ) {
            for( int j = 0; j < 3; ++j ) {
                double sum = 0.;
                for( int m1 = 0; m1 < fc__.nr[0]; ++m1 ) {
                    for( int m2 = 0; m2 < fc__.nr[1]; ++m2 ) {
                        for( int m3 = 0; m3 < fc__.nr[2]; ++m3 ) {
                            for( int ib = 0; ib < nat; ++ib ) {
                                sum += fc__.C(m1, m2, m3, 3 * ia + i, 3 * ib + j);
                            }
                        }
                    }
                }
                fc__.C(0, 0, 0, 3 * ia + i, 3 * ia + j) -= sum;
                largest = std::max(largest, std::abs(sum));
            }
        }
    }
    return largest;
}

DynamicalMatrix dynamical_matrix(const ForceConstants& fc__, const std::array<double, 3>& q__)
{
    for( int i = 0; i < 3; ++i ) {
        const double x = q__[i] * fc__.nr[i];
        if( std::abs(x - std::round(x)) > 1.e-6 ) {
            std::stringstream ss;
            ss << "The q point (" << q__[0] << ", " << q__[1] << ", " << q__[2] << ") is not on the q grid of q2r ("
               << fc__.nr[0] << ", " << fc__.nr[1] << ", " << fc__.nr[2] << "): use a q grid of EDUS that divides it\n";
            throw std::runtime_error(ss.str());
        }
    }
    DynamicalMatrix dyn;
    dyn.num_atoms = fc__.num_atoms;
    dyn.mass = fc__.mass;
    dyn.q = q__;
    const int nmodes = 3 * fc__.num_atoms;
    dyn.C.initialize({nmodes, nmodes});
    dyn.C.fill(0.);
    for( int m1 = 0; m1 < fc__.nr[0]; ++m1 ) {
        for( int m2 = 0; m2 < fc__.nr[1]; ++m2 ) {
            for( int m3 = 0; m3 < fc__.nr[2]; ++m3 ) {
                const double arg = 2. * M_PI * (q__[0] * m1 + q__[1] * m2 + q__[2] * m3);
                const std::complex<double> phase(std::cos(arg), -std::sin(arg));
                for( int mu = 0; mu < nmodes; ++mu ) {
                    for( int nu = 0; nu < nmodes; ++nu ) {
                        dyn.C(mu, nu) += fc__.C(m1, m2, m3, mu, nu) * phase;
                    }
                }
            }
        }
    }
    return dyn;
}

}
