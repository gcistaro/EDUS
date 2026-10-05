#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "core/mpi/Communicator.hpp"
#include "Phonons/EPW.hpp"

/* Reader of the files written by EPW (Quantum ESPRESSO) with epwwrite = .true., which contain the electron-phonon
   problem in the Wannier representation:
   - epwdata.fmt: Fermi energy, dimensions, Born charges, dielectric tensor, H(R_e)   (text, Rydberg)
   - wigner.fmt:  Wigner-Seitz vectors R_e (electrons), R_q (phonons), R_g (electron-phonon) and degeneracies (text)
   - crystal.fmt: number of atoms, lattice vectors, positions, masses                 (text, alat units)
   - prefix.epmatwp: g(R_e, R_g) for each cartesian displacement                      (binary, Ry/bohr)
   The layouts were checked on the source of EPW 6.1 (io.f90, wannier.f90, wigner.f90) and on real files. */

namespace phonon {

/// Reads the numbers of a file written with Fortran list-directed output: complex numbers "(re,im)" become two
/// numbers, logicals are skipped by the caller, repeat counts "n*value" are expanded
class FortranTokens
{
    private:
        std::vector<std::string> tokens_;
        size_t next_ = 0;
        std::string filename_;

    public:
        /// Splits the whole file in tokens at construction: the files are small (the big one, epmatwp, is binary)
        explicit FortranTokens(const std::string& filename__) : filename_(filename__)
        {
            std::ifstream file(filename__);
            if( !file ) {
                throw std::runtime_error("Cannot open the EPW file " + filename__ + "\n");
            }
            std::string word;
            while( file >> word ) {
                /* "(1.0,2.0)" -> "1.0 2.0" */
                for( auto& c : word ) {
                    if( c == '(' || c == ')' || c == ',' ) {
                        c = ' ';
                    }
                }
                std::stringstream ss(word);
                std::string token;
                while( ss >> token ) {
                    /* Fortran list-directed output writes repeated values as "3*0.0" = three times 0.0 */
                    auto star = token.find('*');
                    if( star != std::string::npos ) {
                        int repeat = std::stoi(token.substr(0, star));
                        for( int i = 0; i < repeat; ++i ) {
                            tokens_.push_back(token.substr(star + 1));
                        }
                    }
                    else {
                        tokens_.push_back(token);
                    }
                }
            }
        }

        /// Next token as it is (used to skip values)
        std::string next()
        {
            if( next_ >= tokens_.size() ) {
                throw std::runtime_error("The EPW file " + filename_ + " ended before expected\n");
            }
            return tokens_[next_++];
        }
        double real()
        {
            auto t = next();
            /* Fortran double precision exponent */
            for( auto& c : t ) {
                if( c == 'D' || c == 'd' ) {
                    c = 'E';
                }
            }
            return std::stod(t);
        }
        int integer() { return std::stoi(next()); }
        std::complex<double> complex()
        {
            double re = real();
            return std::complex<double>(re, real());
        }
};

static const double ry_to_ha = 0.5;

EPWData read_epw(const std::string& directory__)
{
    /* the files are read in the same order in which EPW writes them: each value is taken from the stream, the
       values not needed by EDUS are read and discarded */
    EPWData epw;

    /* epwdata.fmt: Fermi energy, dimensions, Born charges and dielectric tensor, Hamiltonian */
    FortranTokens data(directory__ + "/epwdata.fmt");
    epw.fermi_energy = data.real() * ry_to_ha;
    epw.num_bands = data.integer();
    epw.nrr_k = data.integer();
    epw.num_modes = data.integer();
    epw.nrr_q = data.integer();
    epw.nrr_g = data.integer();
    if( epw.num_modes % 3 != 0 ) {
        throw std::runtime_error("epwdata.fmt: the number of phonon modes is not a multiple of 3\n");
    }
    epw.num_atoms = epw.num_modes / 3;
    /* zstar(3, 3, nat), epsi(3, 3) */
    for( int i = 0; i < 9 * epw.num_atoms + 9; ++i ) {
        data.real();
    }
    /* H(R_e), written by EPW with the loops (ibnd, jbnd, irk), irk innermost, not divided by the degeneracies */
    int nb = epw.num_bands;
    epw.H.initialize({epw.nrr_k, nb, nb});
    for( int ib = 0; ib < nb; ++ib ) {
        for( int jb = 0; jb < nb; ++jb ) {
            for( int ir = 0; ir < epw.nrr_k; ++ir ) {
                epw.H(ir, ib, jb) = data.complex() * ry_to_ha;
            }
        }
    }

    /* wigner.fmt: Wigner-Seitz vectors and degeneracies. A vector on the border of the Wigner-Seitz supercell is shared
       by ndegen cells, and a sum over R must weight it by 1/ndegen (like in wannier90). With use_ws = .true. EPW
       computes the degeneracies for each pair of Wannier functions (dims = nbnd) or of Wannier function and atom
       (dims2 = nat), since the Wigner-Seitz cell is then centered on the Wannier centers; otherwise dims = dims2 = 1 */
    FortranTokens wigner(directory__ + "/wigner.fmt");
    int nrr_k = wigner.integer(), nrr_q = wigner.integer(), nrr_g = wigner.integer();
    epw.dims = wigner.integer();
    epw.dims2 = wigner.integer();
    if( nrr_k != epw.nrr_k || nrr_q != epw.nrr_q || nrr_g != epw.nrr_g ) {
        throw std::runtime_error("wigner.fmt and epwdata.fmt have a different number of Wigner-Seitz vectors: "
                                 "they do not come from the same EPW run\n");
    }
    if( epw.dims != 1 && epw.dims != nb ) {
        throw std::runtime_error("wigner.fmt: unexpected dimension of the degeneracies\n");
    }
    epw.irvec_k.resize(nrr_k);
    epw.ndegen_k.initialize({nrr_k, epw.dims, epw.dims});
    for( int ir = 0; ir < nrr_k; ++ir ) {
        for( int ix : {0, 1, 2} ) {
            epw.irvec_k[ir][ix] = wigner.integer();
        }
        wigner.real();    /* length of the vector */
        for( int iw = 0; iw < epw.dims; ++iw ) {
            for( int iw2 = 0; iw2 < epw.dims; ++iw2 ) {
                epw.ndegen_k(ir, iw, iw2) = wigner.integer();
            }
        }
    }
    /* vectors of the phonons (force constants in real space): not used, skipped (3 components + length) */
    for( int ir = 0; ir < nrr_q; ++ir ) {
        for( int i = 0; i < 4; ++i ) {
            wigner.next();
        }
        for( int i = 0; i < epw.dims2 * epw.dims2; ++i ) {
            wigner.integer();
        }
    }
    /* vectors of the electron-phonon matrix elements, R_g (position of the displaced atom) */
    epw.irvec_g.resize(nrr_g);
    epw.ndegen_g.initialize({epw.dims, nrr_g, epw.dims2});
    for( int ir = 0; ir < nrr_g; ++ir ) {
        for( int ix : {0, 1, 2} ) {
            epw.irvec_g[ir][ix] = wigner.integer();
        }
        wigner.real();
        for( int iw = 0; iw < epw.dims; ++iw ) {
            for( int ia = 0; ia < epw.dims2; ++ia ) {
                epw.ndegen_g(iw, ir, ia) = wigner.integer();
            }
        }
    }

    /* the Hamiltonian is used divided by the degeneracies, like in EPW */
    for( int ir = 0; ir < nrr_k; ++ir ) {
        for( int ib = 0; ib < nb; ++ib ) {
            for( int jb = 0; jb < nb; ++jb ) {
                int d = epw.dims == 1 ? epw.ndegen_k(ir, 0, 0) : epw.ndegen_k(ir, ib, jb);
                epw.H(ir, ib, jb) = d > 0 ? epw.H(ir, ib, jb) / double(d) : 0.;
            }
        }
    }

    /* crystal.fmt: nat, nmodes, nelec and nbndskip, lattice vectors (alat units), reciprocal vectors,
       volume, alat, positions (alat units) */
    FortranTokens crystal(directory__ + "/crystal.fmt");
    if( crystal.integer() != epw.num_atoms || crystal.integer() != epw.num_modes ) {
        throw std::runtime_error("crystal.fmt and epwdata.fmt have a different number of atoms or modes\n");
    }
    /* nelec, nbndskip */
    crystal.real();
    crystal.real();
    /* lattice vectors at(:, i) in alat units, one vector after the other */
    std::array<std::array<double, 3>, 3> at;
    for( auto& a : at ) {
        for( auto& x : a ) {
            x = crystal.real();
        }
    }
    /* reciprocal vectors (not needed) and volume, then alat (bohr) */
    for( int i = 0; i < 9; ++i ) {
        crystal.real();
    }
    crystal.real();
    double alat = crystal.real();
    for( int i = 0; i < 3; ++i ) {
        for( int ix = 0; ix < 3; ++ix ) {
            epw.lattice[i][ix] = at[i][ix] * alat;
        }
    }
    /* positions of the atoms, alat units -> bohr (the masses and the rest of the file are not needed) */
    epw.tau.resize(epw.num_atoms);
    for( auto& t : epw.tau ) {
        for( auto& x : t ) {
            x = crystal.real() * alat;
        }
    }
    return epw;
}

/* epmatwp is a raw binary file (no Fortran record markers) with the complex array epmatwp(nbnd, nbnd, nrr_k, nmodes,
   nrr_g) in Fortran order: nrr_g consecutive blocks ("records") of nbnd*nbnd*nrr_k*nmodes numbers, one per R_g.
   The mixed representation g(R_e; q) = sum_{R_g} e^{i q.R_g} g(R_e, R_g) is built one block at a time, so the whole
   file (which can be large) is never in memory. */
mdarray<std::complex<double>, 4> read_epmatwp(const std::string& filename__, const EPWData& epw__,
                                              const std::array<double, 3>& q__)
{
    const int nb = epw__.num_bands, nrr_k = epw__.nrr_k, nmodes = epw__.num_modes;
    mdarray<std::complex<double>, 4> g({nmodes, nrr_k, nb, nb});
    g.fill(0.);
    const std::size_t record = std::size_t(nb) * nb * nrr_k * nmodes;
    auto& comm = mpi::Communicator::world();

    /* only rank 0 reads the file; an error there must stop all the ranks, which would otherwise wait forever in
       MPI_Bcast: the message of the error is sent to all of them, and each one throws it */
    std::string error_message;
    if( comm.rank() == 0 ) {
        try {
            std::ifstream file(filename__, std::ios::binary);
            if( !file ) {
                throw std::runtime_error("Cannot open the EPW file " + filename__ + "\n");
            }
            /* the size of the file must match the dimensions of epwdata.fmt: catches files of different runs, and files
               written in another layout (etf_mem != 0 writes one file per R_g or per mode) */
            file.seekg(0, std::ios::end);
            std::size_t expected = record * epw__.nrr_g * sizeof(std::complex<double>);
            if( std::size_t(file.tellg()) != expected ) {
                std::stringstream ss;
                ss << filename__ << " has " << file.tellg() << " bytes, expected " << expected
                   << " (nbnd = " << nb << ", nrr_k = " << nrr_k << ", nmodes = " << nmodes << ", nrr_g = " << epw__.nrr_g
                   << "): it does not come from the same EPW run of epwdata.fmt, or it was written with etf_mem != 0\n";
                throw std::runtime_error(ss.str());
            }
            file.seekg(0);

            std::vector<std::complex<double>> buffer(record);
            for( int irg = 0; irg < epw__.nrr_g; ++irg ) {
                file.read(reinterpret_cast<char*>(buffer.data()), record * sizeof(std::complex<double>));
                /* q and R_g in crystal coordinates: q.R = 2 pi sum_i q_i R_i */
                double qR = 2. * M_PI * (q__[0] * epw__.irvec_g[irg][0] + q__[1] * epw__.irvec_g[irg][1]
                                         + q__[2] * epw__.irvec_g[irg][2]);
                std::complex<double> phase(std::cos(qR), std::sin(qR));
                /* Fortran order: (ib, jb, ir, mode), ib fastest. Each term is divided by the degeneracy of R_g, which with
                   use_ws depends on the Wannier function ib and on the atom displaced by the mode */
                for( int mode = 0; mode < nmodes; ++mode ) {
                    int atom = mode / 3;
                    for( int ir = 0; ir < nrr_k; ++ir ) {
                        for( int jb = 0; jb < nb; ++jb ) {
                            for( int ib = 0; ib < nb; ++ib ) {
                                int d = epw__.dims == 1 ? epw__.ndegen_g(0, irg, 0) : epw__.ndegen_g(ib, irg, atom);
                                if( d > 0 ) {
                                    g(mode, ir, ib, jb) += phase * buffer[ib + nb * (jb + nb * (ir + std::size_t(nrr_k) * mode))]
                                                           / double(d);
                                }
                            }
                        }
                    }
                }
            }
            /* degeneracies of the electronic vectors R_e (so that g(k) = sum_{R_e} e^{ik.R_e} g(R_e), without weights,
               like H(R) in EDUS), Ry/bohr -> Ha/bohr */
            for( int mode = 0; mode < nmodes; ++mode ) {
                for( int ir = 0; ir < nrr_k; ++ir ) {
                    for( int ib = 0; ib < nb; ++ib ) {
                        for( int jb = 0; jb < nb; ++jb ) {
                            int d = epw__.dims == 1 ? epw__.ndegen_k(ir, 0, 0) : epw__.ndegen_k(ir, ib, jb);
                            g(mode, ir, ib, jb) = d > 0 ? g(mode, ir, ib, jb) * ry_to_ha / double(d) : 0.;
                        }
                    }
                }
            }
        }
        catch( const std::exception& e ) {
            error_message = e.what();
        }
    }
#ifdef EDUS_MPI
    int length = int(error_message.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, comm.communicator());
    error_message.resize(length);
    MPI_Bcast(error_message.data(), length, MPI_CHAR, 0, comm.communicator());
#endif
    if( !error_message.empty() ) {
        throw std::runtime_error(error_message);
    }
    /* the other ranks receive the result of rank 0 */
#ifdef EDUS_MPI
    MPI_Bcast(g.data(), int(g.get_TotalSize()), MPI_CXX_DOUBLE_COMPLEX, 0, comm.communicator());
#endif
    return g;
}

}
