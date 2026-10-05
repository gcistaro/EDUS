#include <algorithm>
#include <cmath>
#include <sstream>
#include "ConvertUnits.hpp"
#include "GlobalFunctions.hpp"
#include "Geometry/Matrix.hpp"
#include "Phonons/EPW.hpp"
#include "Phonons/DynamicalMatrix.hpp"
#include "Phonons/Lattice.hpp"

namespace phonon {

/// @brief Sets up everything the lattice needs during the propagation, in this order:
/// 1. the electron-phonon coupling g_mu of EPW, moved on the grid of the simulation (read_coupling);
/// 2. masses and force constants K at Gamma from the dynamical matrix of ph.x; then g without the rigid translation
///    (remove_translation_from_coupling) and without the screening of the electrons of the model if it is the
///    screened coupling of a standard EPW run (unscreen_coupling);
/// 3. with the static adiabatic reference, the static response Pi of the electrons, removed from K.
/// After this call the class is immutable: during the propagation it only computes H += u.g, forces and energies.
void Lattice::initialize(const PhononParameters& parameters__, const GridStructure& gridstructure__,
                         const parallel::Decomposition& decomposition__, const electron::System& system__,
                         Material& material__, electron::MeanField* meanfield__)
{
    PROFILE("phonon::Lattice::initialize");
    parameters_ = parameters__;
    /* the operators are distributed over the ranks of this communicator (R or k points): all the sums over R or k
       are local sums followed by an MPI_Allreduce on it */
    comm_ = &decomposition__.kpool_comm();
    num_bands_ = system__.num_bands();

    /* 1. coupling: sets also num_atoms_ and num_modes_ (= 3 * num_atoms_) */
    read_coupling(gridstructure__, decomposition__, material__);

    /* 2. masses and force constants at Gamma */
    auto dyn = read_dynamical_matrix(parameters_.dyn_file);
    if( dyn.num_atoms != num_atoms_ ) {
        std::stringstream ss;
        ss << "The dynamical matrix " << parameters_.dyn_file << " has " << dyn.num_atoms << " atoms, EPW "
           << num_atoms_ << "\n";
        throw std::runtime_error(ss.str());
    }
    if( std::abs(dyn.q[0]) + std::abs(dyn.q[1]) + std::abs(dyn.q[2]) > 1.e-8 ) {
        throw std::runtime_error("The first dynamical matrix of " + parameters_.dyn_file + " is not at q = Gamma\n");
    }
    /* a rigid translation of the crystal costs no energy: sum_atom' K(atom alpha, atom' beta) = 0. ph.x satisfies it only
       approximately, and the small error would give a spurious frequency to the acoustic modes */
    if( parameters_.acoustic_sum_rule ) {
        asr_correction_ = impose_acoustic_sum_rule(dyn);
    }
    /* each cartesian coordinate mu = 3*atom + direction moves the mass of its atom */
    mass_.resize(num_modes_);
    for( int mu = 0; mu < num_modes_; ++mu ) {
        mass_[mu] = dyn.mass[mu / 3];
    }
    /* the force constants must be hermitian (real symmetric at Gamma): remove the numerical noise of the file */
    mdarray<std::complex<double>, 2> K({num_modes_, num_modes_});
    for( int mu = 0; mu < num_modes_; ++mu ) {
        for( int nu = 0; nu < num_modes_; ++nu ) {
            K(mu, nu) = 0.5 * (dyn.C(mu, nu) + std::conj(dyn.C(nu, mu)));
        }
    }
    /* frequencies of ph.x, for the recap, and its normal modes, on which the dynamics is projected in the output.
       These are the physical (Born-Oppenheimer) phonons, whatever the adiabatic reference */
    frequencies_input_ = frequencies(K);
    normal_modes(K, mode_omega2_, mode_vectors_);

    /* the coupling without the rigid translation (it needs the masses), then without the screening of the model */
    if( parameters_.acoustic_sum_rule ) {
        remove_translation_from_coupling();
    }
    if( parameters_.coupling == CouplingType::screened ) {
        unscreen_coupling(gridstructure__, decomposition__, system__, meanfield__);
    }

    /* 3. static reference. K of ph.x is Born-Oppenheimer: K_BO = K0 + Pi, with Pi the static screening of the lattice
       by the electrons. The propagated electrons generate Pi again, so the dynamics must use the bare K0 = K_BO - Pi
       (equivalently: K_BO, and only the excited part of rho pushes the atoms, see EHRENFEST.md section 5).
       With the dynamic reference K_BO is used as it is: rho_BO takes care of it during the propagation */
    if( parameters_.adiabatic_reference == AdiabaticReference::static_response ) {
        auto Pi = static_response(gridstructure__, decomposition__, system__, meanfield__);
        for( int mu = 0; mu < num_modes_; ++mu ) {
            for( int nu = 0; nu < num_modes_; ++nu ) {
                K(mu, nu) -= Pi(mu, nu);
            }
        }
    }
    /* gtilde was needed only by static_response */
    gtilde_.clear();
    /* frequencies of the force constants actually used, only for the recap (e.g. K0 with the static reference) */
    frequencies_ = frequencies(K);

    /* one matrix per q point; for now only Gamma (index 0) */
    force_constants_.initialize({num_qpoints(), num_modes_, num_modes_});
    for( int mu = 0; mu < num_modes_; ++mu ) {
        for( int nu = 0; nu < num_modes_; ++nu ) {
            force_constants_(0, mu, nu) = K(mu, nu);
        }
    }

    if( !parameters_.initial_displacement.empty() && int(parameters_.initial_displacement.size()) != num_modes_ ) {
        std::stringstream ss;
        ss << "phonons/initial_displacement has " << parameters_.initial_displacement.size() << " values, expected "
           << num_modes_ << " (3 per atom)\n";
        throw std::runtime_error(ss.str());
    }
}

/// @brief Reads the EPW files and puts the coupling g_mu at q = 0 on the grid of the simulation.
///
/// EPW gives g_mu(R_e) on its own list of Wigner-Seitz vectors R_e (the same kind of list of a wannier90 _tb.dat
/// file), while EDUS works on the regular FFT grid of the simulation. The same path used for H0 by the tight-binding
/// model (System::load_from_material) is followed:
/// g(R_e) on the EPW list --dft--> g(k) on the k points of the simulation --fft--> g(R) on the FFT grid.
/// The Hamiltonian of EPW goes through the same path and is compared with H0: this checks that EPW and the
/// tight-binding model use the same Wannier functions (same order, same phases), otherwise g is in another gauge.
void Lattice::read_coupling(const GridStructure& gridstructure__, const parallel::Decomposition& decomposition__,
                            Material& material__)
{
    auto epw = read_epw(parameters_.epw_directory);
    if( epw.num_bands != num_bands_ ) {
        std::stringstream ss;
        ss << "EPW has " << epw.num_bands << " Wannier functions, the tight-binding model " << num_bands_
           << ": the electron-phonon coupling must be in the basis of the Wannier functions of the model\n";
        throw std::runtime_error(ss.str());
    }
    num_atoms_ = epw.num_atoms;
    num_modes_ = epw.num_modes;

    /* the R vectors of EPW are in crystal coordinates: the lattice vectors must be the ones of the model.
       A has the lattice vectors of the model as columns, epw.lattice as rows (both in bohr) */
    auto& A = Coordinate::get_Basis(LatticeVectors(R)).get_M();
    double lattice_difference = 0.;
    for( int i = 0; i < 3; ++i ) {
        for( int ix = 0; ix < 3; ++ix ) {
            lattice_difference = std::max(lattice_difference, std::abs(A(ix, i) - epw.lattice[i][ix]));
        }
    }
    if( lattice_difference > 1.e-3 ) {
        std::stringstream ss;
        ss << "The lattice vectors of EPW (crystal.fmt) and of the tight-binding model differ by " << lattice_difference
           << " bohr: the R vectors of EPW are in crystal coordinates and would be misplaced\n";
        throw std::runtime_error(ss.str());
    }

    /* list of the R vectors of EPW (crystal coordinates), as a MeshGrid like the one of the _tb.dat file */
    mdarray<double, 2> bare_R({epw.nrr_k, 3});
    for( int ir = 0; ir < epw.nrr_k; ++ir ) {
        for( int ix : {0, 1, 2} ) {
            bare_R(ir, ix) = epw.irvec_k[ir][ix];
        }
    }
    MeshGrid Rmesh_epw(R, bare_R, LatticeVectors(R));
    auto& kmesh = gridstructure__.kgrid()->get_mesh();

    /* O(R) of EPW (already divided by the degeneracies) -> O(k) = sum_R e^{ik.R} O(R) on the local k points of the
       simulation (a direct Fourier sum, dft, since the EPW list is not a regular grid) */
    auto to_k = [&](mdarray<std::complex<double>, 3>& values__) {
        Operator<std::complex<double>> O;
        O.get_Operator_R().initialize(R, values__);
        O.get_Operator_R().set_MeshGrid(Rmesh_epw);
        O.lock_gauge(wannier);
        O.lock_space(R);
        O.dft(kmesh, +1);
        return O;
    };

    /* same Wannier functions: the Hamiltonian of EPW must be the one of the model, at every k point.
       Only the largest difference is stored, and printed in the recap (with a warning if it is large) */
    {
        auto Hepw = to_k(epw.H);
        auto& Hk = Hepw.get_Operator(Space::k);
        auto& Htb = material__.H.get_Operator(Space::k);
        gauge_check_ = 0.;
        for( int ik = 0; ik < Hk.get_nblocks(); ++ik ) {
            for( int i = 0; i < num_bands_; ++i ) {
                for( int j = 0; j < num_bands_; ++j ) {
                    gauge_check_ = std::max(gauge_check_, std::abs(Hk(ik, i, j) - Htb(ik, i, j)));
                }
            }
        }
#ifdef EDUS_MPI
        MPI_Allreduce(MPI_IN_PLACE, &gauge_check_, 1, MPI_DOUBLE, MPI_MAX, comm_->communicator());
#endif
    }

    /* g_mu(R_e; q) for q = Gamma: (mode, R_e, band, band), Ha/bohr */
    auto g = read_epmatwp(parameters_.epw_directory + "/" + parameters_.epmatwp, epw, parameters_.qpoints[0]);
    coupling_.resize(num_modes_);
    mdarray<std::complex<double>, 3> values({epw.nrr_k, num_bands_, num_bands_});
    for( int mu = 0; mu < num_modes_; ++mu ) {
        /* coupling_scale allows to switch the coupling off (0) keeping everything else, e.g. for tests */
        for( int ir = 0; ir < epw.nrr_k; ++ir ) {
            for( int i = 0; i < num_bands_; ++i ) {
                for( int j = 0; j < num_bands_; ++j ) {
                    values(ir, i, j) = parameters_.coupling_scale * g(mu, ir, i, j);
                }
            }
        }
        /* EPW list of R -> k points of the simulation */
        auto gk = to_k(values);
        /* at q = 0 the coupling is hermitian: remove the error of the Wannier interpolation */
        gk.get_Operator(Space::k).make_hermitian();

        /* k points of the simulation -> FFT grid in R, where it is added to H during the propagation.
           Both components (k and R) of coupling_[mu] stay available afterwards */
        std::stringstream tag;
        tag << "g_" << mu;
        coupling_[mu].initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), num_bands_,
                                     decomposition__.mpindex(), tag.str());
        auto& src = gk.get_Operator(Space::k);
        std::copy(src.begin(), src.end(), coupling_[mu].get_Operator(Space::k).begin());
        coupling_[mu].lock_gauge(wannier);
        coupling_[mu].lock_space(Space::k);
        coupling_[mu].go_to_R();
    }
}

/// @brief Removes from the coupling the part that moves the whole crystal:
///     g_(atom, alpha) <- g_(atom, alpha) - M_atom/M_total sum_atom' g_(atom', alpha).
///
/// A rigid translation of the crystal does not change the electrons, but in the fixed basis of the Wannier functions
/// sum_atom g_(atom, alpha) is not zero: it is the matrix element of -grad V between Wannier functions that do not move
/// with the atoms (in DFPT the second-order term <d^2 V/du du> compensates it in the force constants, and it is
/// missing here). Without this correction the translation is coupled to the electrons: the electrons push the center
/// of mass, and the static response Pi does not satisfy the acoustic sum rule, so that the acoustic modes of K - Pi
/// get a finite frequency.
/// With the weights M_atom/M_total the coupling does not change along any displacement that keeps the center of mass
/// fixed (sum_atom M_atom u_atom = 0, e.g. the optical modes): u.g is the same before and after. After it,
/// sum_atom g_(atom, alpha) = 0, so the force on the center of mass is zero and Pi satisfies the acoustic sum rule.
/// The largest max|sum_atom g| / max|g| is stored for the recap.
void Lattice::remove_translation_from_coupling()
{
    double total_mass = 0.;
    for( int atom = 0; atom < num_atoms_; ++atom ) {
        total_mass += mass_[3 * atom];
    }
    auto& g0 = coupling_[0].get_Operator(Space::k);
    BlockMatrix<std::complex<double>> sum(k, g0.get_nblocks(), num_bands_, num_bands_);
    translation_coupling_ = 0.;
    for( int ix : {0, 1, 2} ) {
        sum.fill(0.);
        double scale = 0.;
        for( int atom = 0; atom < num_atoms_; ++atom ) {
            auto& g = coupling_[3 * atom + ix].get_Operator(Space::k);
            for( int i = 0; i < g.get_TotalSize(); ++i ) {
                *(sum.begin() + i) += *(g.begin() + i);
                scale = std::max(scale, std::abs(*(g.begin() + i)));
            }
        }
        double largest = 0.;
        for( int i = 0; i < sum.get_TotalSize(); ++i ) {
            largest = std::max(largest, std::abs(*(sum.begin() + i)));
        }
#ifdef EDUS_MPI
        MPI_Allreduce(MPI_IN_PLACE, &largest, 1, MPI_DOUBLE, MPI_MAX, comm_->communicator());
        MPI_Allreduce(MPI_IN_PLACE, &scale, 1, MPI_DOUBLE, MPI_MAX, comm_->communicator());
#endif
        if( scale > 0. ) {
            translation_coupling_ = std::max(translation_coupling_, largest / scale);
        }
        for( int atom = 0; atom < num_atoms_; ++atom ) {
            int mu = 3 * atom + ix;
            auto& g = coupling_[mu].get_Operator(Space::k);
            double weight = mass_[mu] / total_mass;
            for( int i = 0; i < g.get_TotalSize(); ++i ) {
                *(g.begin() + i) -= weight * *(sum.begin() + i);
            }
        }
    }
    for( int mu = 0; mu < num_modes_; ++mu ) {
        coupling_[mu].lock_gauge(wannier);
        coupling_[mu].lock_space(Space::k);
        coupling_[mu].go_to_R();
    }
}

namespace {

/// @brief Static linear response of the electrons of the model: the two ingredients chi0 and Sigma[delta rho],
/// on the local k points and in the Wannier gauge. Used by the static reference (Lattice::static_response) and by
/// the unscreening of the coupling of EPW (Lattice::unscreen_coupling).
class ModelResponse
{
    private:
        const BlockMatrix<std::complex<double>>& U_;
        const BlockMatrix<std::complex<double>>& Udagger_;
        const std::vector<mdarray<double, 1>>& energies_;
        const std::vector<double>& f_;
        const Operator<std::complex<double>>& DM0_;
        electron::MeanField* meanfield_;
        int num_bands_;
        /// rho0 + delta rho and Sigma in R, with the FFT (only with the mean field)
        Operator<std::complex<double>> dm_, sigma_;
        BlockMatrix<std::complex<double>> temp_;

    public:
        ModelResponse(const GridStructure& gridstructure__, const parallel::Decomposition& decomposition__,
                      const electron::System& system__, electron::MeanField* meanfield__, const int num_bands__)
            : U_(system__.bandstructure().U())
            , Udagger_(system__.bandstructure().Udagger())
            , energies_(system__.bandstructure().energies())
            , f_(system__.parameters().occupations)
            , DM0_(system__.DM0())
            , meanfield_(meanfield__ && meanfield__->parameters().enabled ? meanfield__ : nullptr)
            , num_bands_(num_bands__)
            , temp_(k, system__.bandstructure().U().get_nblocks(), num_bands__, num_bands__)
        {
            /* to compute Sigma[delta rho] the MeanField class needs a full density matrix rho0 + delta rho (it computes
               Sigma[rho - rho0]) in R, like in the propagation */
            if( meanfield_ ) {
                dm_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), num_bands_, decomposition__.mpindex(), "drho_static");
                sigma_.initialize_fft(gridstructure__.Rgrid(), gridstructure__.kgrid(), num_bands_, decomposition__.mpindex(), "sigma_static");
            }
        }

        bool interacting() const { return meanfield_ != nullptr; }
        int nk_local() const { return U_.get_nblocks(); }

        /// out = chi0 V, input and output in the Wannier gauge:
        /// 1. to the Bloch gauge, O_bloch = U^dagger O U;
        /// 2. multiply each element by (f_n - f_m)/(e_n - e_m); pairs with the same occupation (e.g. two valence bands)
        ///    do not respond, and degenerate pairs are skipped (they would need the intraband term of a metal);
        /// 3. back to the Wannier gauge, O = U O_bloch U^dagger
        void chi0(BlockMatrix<std::complex<double>>& out__, const BlockMatrix<std::complex<double>>& V__)
        {
            temp_.fill(0.);
            multiply(temp_, std::complex<double>(1.), Udagger_, V__);
            out__.fill(0.);
            multiply(out__, std::complex<double>(1.), temp_, U_);
            for( int ik = 0; ik < nk_local(); ++ik ) {
                for( int n = 0; n < num_bands_; ++n ) {
                    for( int m = 0; m < num_bands_; ++m ) {
                        double df = f_[n] - f_[m];
                        double de = energies_[ik](n) - energies_[ik](m);
                        out__(ik, n, m) *= ( std::abs(df) < 1.e-12 || std::abs(de) < 1.e-10 ) ? 0. : df / de;
                    }
                }
            }
            temp_.fill(0.);
            multiply(temp_, std::complex<double>(1.), U_, out__);
            out__.fill(0.);
            multiply(out__, std::complex<double>(1.), temp_, Udagger_);
        }

        /// out = Sigma[delta rho] (k components, Wannier gauge): rho0 + delta rho to R, self energy in R, back to k.
        /// Zero without the mean field
        void self_energy(BlockMatrix<std::complex<double>>& out__, const BlockMatrix<std::complex<double>>& drho__)
        {
            if( !meanfield_ ) {
                out__.fill(0.);
                return;
            }
            auto& dmk = dm_.get_Operator(Space::k);
            auto& dm0k = DM0_.get_Operator(Space::k);
            for( int i = 0; i < drho__.get_TotalSize(); ++i ) {
                *(dmk.begin() + i) = *(dm0k.begin() + i) + *(drho__.begin() + i);
            }
            dm_.lock_gauge(wannier);
            dm_.lock_space(Space::k);
            dm_.go_to_R();
            sigma_.get_Operator(R).fill(0.);
            sigma_.lock_gauge(wannier);
            sigma_.lock_space(R);
            meanfield_->self_energy(sigma_, dm_, DM0_);
            sigma_.go_to_k();
            auto& sigmak = sigma_.get_Operator(Space::k);
            std::copy(sigmak.begin(), sigmak.end(), out__.begin());
        }
};

/// max over the elements of |a| (b == nullptr) or of |a - b|, over all the ranks of comm__
double max_abs(const BlockMatrix<std::complex<double>>& a__, const BlockMatrix<std::complex<double>>* b__,
               const mpi::Communicator& comm__)
{
    double result = 0.;
    for( int i = 0; i < a__.get_TotalSize(); ++i ) {
        result = std::max(result, std::abs(*(a__.begin() + i) - (b__ ? *(b__->begin() + i) : 0.)));
    }
#ifdef EDUS_MPI
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_DOUBLE, MPI_MAX, comm__.communicator());
#endif
    return result;
}

/// <a, b> = sum over the elements of conj(a) b, over all the ranks of comm__
std::complex<double> dot(const BlockMatrix<std::complex<double>>& a__, const BlockMatrix<std::complex<double>>& b__,
                         const mpi::Communicator& comm__)
{
    std::complex<double> result = 0.;
    for( int i = 0; i < a__.get_TotalSize(); ++i ) {
        result += std::conj(*(a__.begin() + i)) * *(b__.begin() + i);
    }
#ifdef EDUS_MPI
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm__.communicator());
#endif
    return result;
}

/// Solves the small hermitian system A x = b (n x n, row major) by Gaussian elimination with partial pivoting.
/// A tiny multiple of the largest diagonal element is added to the diagonal: the Anderson history can be almost
/// linearly dependent
std::vector<std::complex<double>> solve_small(std::vector<std::complex<double>> A__, std::vector<std::complex<double>> b__)
{
    const int n = int(b__.size());
    double largest = 0.;
    for( int i = 0; i < n; ++i ) {
        largest = std::max(largest, std::abs(A__[i * n + i]));
    }
    for( int i = 0; i < n; ++i ) {
        A__[i * n + i] += 1.e-12 * largest;
    }
    for( int c = 0; c < n; ++c ) {
        int p = c;
        for( int r = c + 1; r < n; ++r ) {
            if( std::abs(A__[r * n + c]) > std::abs(A__[p * n + c]) ) {
                p = r;
            }
        }
        if( std::abs(A__[p * n + c]) == 0. ) {
            continue;
        }
        for( int j = 0; j < n; ++j ) {
            std::swap(A__[c * n + j], A__[p * n + j]);
        }
        std::swap(b__[c], b__[p]);
        for( int r = c + 1; r < n; ++r ) {
            auto f = A__[r * n + c] / A__[c * n + c];
            for( int j = c; j < n; ++j ) {
                A__[r * n + j] -= f * A__[c * n + j];
            }
            b__[r] -= f * b__[c];
        }
    }
    std::vector<std::complex<double>> x(n, 0.);
    for( int r = n - 1; r >= 0; --r ) {
        if( std::abs(A__[r * n + r]) == 0. ) {
            continue;
        }
        auto s = b__[r];
        for( int j = r + 1; j < n; ++j ) {
            s -= A__[r * n + j] * x[j];
        }
        x[r] = s / A__[r * n + r];
    }
    return x;
}

/// Eigenvalues w (ascending) and orthonormal eigenvectors V (columns, row major) of the real symmetric matrix A
/// (n x n, row major), by the cyclic Jacobi method. Real eigenvectors also in degenerate subspaces, which a complex
/// solver does not guarantee; the sign of each one is fixed by making its largest component positive
void symmetric_eigen(std::vector<double> A__, const int n__, std::vector<double>& w__, std::vector<double>& V__)
{
    V__.assign(n__ * n__, 0.);
    for( int i = 0; i < n__; ++i ) {
        V__[i * n__ + i] = 1.;
    }
    double norm2 = 0.;
    for( auto a : A__ ) {
        norm2 += a * a;
    }
    for( int sweep = 0; sweep < 100; ++sweep ) {
        double off = 0.;
        for( int p = 0; p < n__; ++p ) {
            for( int q = p + 1; q < n__; ++q ) {
                off += A__[p * n__ + q] * A__[p * n__ + q];
            }
        }
        if( off <= 1.e-32 * norm2 ) {
            break;
        }
        for( int p = 0; p < n__; ++p ) {
            for( int q = p + 1; q < n__; ++q ) {
                double apq = A__[p * n__ + q];
                if( apq == 0. ) {
                    continue;
                }
                /* rotation in the (p, q) plane that zeroes A(p, q): A <- J^T A J, V <- V J */
                double theta = (A__[q * n__ + q] - A__[p * n__ + p]) / (2. * apq);
                double t = (theta >= 0. ? 1. : -1.) / (std::abs(theta) + std::sqrt(theta * theta + 1.));
                double c = 1. / std::sqrt(t * t + 1.);
                double s = t * c;
                for( int k = 0; k < n__; ++k ) {
                    double akp = A__[k * n__ + p], akq = A__[k * n__ + q];
                    A__[k * n__ + p] = c * akp - s * akq;
                    A__[k * n__ + q] = s * akp + c * akq;
                }
                for( int k = 0; k < n__; ++k ) {
                    double apk = A__[p * n__ + k], aqk = A__[q * n__ + k];
                    A__[p * n__ + k] = c * apk - s * aqk;
                    A__[q * n__ + k] = s * apk + c * aqk;
                }
                for( int k = 0; k < n__; ++k ) {
                    double vkp = V__[k * n__ + p], vkq = V__[k * n__ + q];
                    V__[k * n__ + p] = c * vkp - s * vkq;
                    V__[k * n__ + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    /* ascending order, and the sign of each eigenvector */
    std::vector<int> order(n__);
    for( int i = 0; i < n__; ++i ) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return A__[a * n__ + a] < A__[b * n__ + b]; });
    std::vector<double> V(n__ * n__);
    w__.resize(n__);
    for( int j = 0; j < n__; ++j ) {
        int col = order[j];
        w__[j] = A__[col * n__ + col];
        int largest = 0;
        for( int i = 0; i < n__; ++i ) {
            if( std::abs(V__[i * n__ + col]) > std::abs(V__[largest * n__ + col]) + 1.e-12 ) {
                largest = i;
            }
        }
        double sign = V__[largest * n__ + col] < 0. ? -1. : 1.;
        for( int i = 0; i < n__; ++i ) {
            V[i * n__ + j] = sign * V__[i * n__ + col];
        }
    }
    V__ = V;
}

}

/// @brief Screened coupling of EPW -> coupling to use with the mean field of the model (coupling = screened).
///
/// EPW gives the screened coupling g_s = dV_ion + dV_Hxc: the electrons of DFT (all the bands) respond to the
/// displacement. EDUS generates again, during the propagation, the part of this screening due to the electrons of
/// the model through Sigma[delta rho], so the coupling in H must be screened by everything except the model
/// (the same idea of the constrained RPA and DFPT). In the static limit the electrons of the model respond with
/// delta rho = chi0 V, V = g_b + Sigma[delta rho] the total perturbation. Asking V = g_s gives, exactly and without
/// iterations (Sigma is linear in delta rho),
///     g_b = g_s - Sigma[chi0 g_s].
/// Then in the static limit the electrons feel exactly the coupling of DFPT. The approximation is that the response
/// of the model is the one of its mean field (Hartree + SEX) and not the Hxc of DFT. Without the mean field g_b = g_s.
/// The k and R components of coupling_ are replaced. gtilde = chi0 g_s (Eq. 19 of the notes on the Born-Oppenheimer
/// response) is the static response per unit displacement: it is kept in gtilde_ for static_response, which with
/// g_b does not need to solve the self-consistent problem again.
void Lattice::unscreen_coupling(const GridStructure& gridstructure__, const parallel::Decomposition& decomposition__,
                                const electron::System& system__, electron::MeanField* meanfield__)
{
    PROFILE("phonon::Lattice::unscreen_coupling");
    ModelResponse response(gridstructure__, decomposition__, system__, meanfield__, num_bands_);
    unscreening_change_ = 0.;
    if( !response.interacting() ) {
        return;
    }
    const int nk_local = response.nk_local();
    BlockMatrix<std::complex<double>> sigma(k, nk_local, num_bands_, num_bands_);
    gtilde_.clear();
    for( int mu = 0; mu < num_modes_; ++mu ) {
        /* g = g_s on input, g_b on output */
        auto& g = coupling_[mu].get_Operator(Space::k);
        gtilde_.emplace_back(k, nk_local, num_bands_, num_bands_);
        auto& gtilde = gtilde_.back();
        response.chi0(gtilde, g);
        response.self_energy(sigma, gtilde);
        /* relative size of the correction, for the recap */
        double scale = max_abs(g, nullptr, *comm_);
        if( scale > 0. ) {
            unscreening_change_ = std::max(unscreening_change_, max_abs(sigma, nullptr, *comm_) / scale);
        }
        for( int i = 0; i < g.get_TotalSize(); ++i ) {
            *(g.begin() + i) -= *(sigma.begin() + i);
        }
        coupling_[mu].lock_gauge(wannier);
        coupling_[mu].lock_space(Space::k);
        coupling_[mu].go_to_R();
    }
}

/// @brief Static linear response of the electrons to a displacement of the lattice, and the electronic force
/// constants Pi that it gives.
///
/// A static displacement u_nu adds u_nu g_nu to the Hamiltonian. The electrons respond with a change of the density
/// matrix delta rho_nu (per unit displacement) that, at first order and in the Bloch gauge, is
///     delta rho_nm = (f_n - f_m)/(e_n - e_m) V_nm = [chi0 V]_nm,
/// with f the occupations, e the band energies and V the total perturbation felt by the electrons. With the mean
/// field, delta rho changes the self energy, which is part of the perturbation: V = g_nu + Sigma[delta rho].
/// The equation delta rho = chi0 (g_nu + Sigma[delta rho]) is solved as follows:
/// - without the mean field a single step, V = g_nu, is exact;
/// - if the coupling was unscreened (unscreen_coupling), g_b + Sigma[gtilde] = g_s, so the solution is gtilde = chi0 g_s,
///   already computed: no iterations;
/// - otherwise by Anderson mixing of the fixed point delta rho -> chi0 (g_nu + Sigma[delta rho]). The problem is linear,
///   and Anderson mixing is then close to GMRES: it converges also when the plain iteration does not (strong mean field).
/// Pi is then the force on mu due to delta rho_nu,
///     Pi_{mu nu} = s/N sum_k Tr[g_mu delta rho_nu].
/// This is exactly the screening contained in the Born-Oppenheimer force constants of ph.x, but computed with the
/// electrons of the model, which are the ones that are propagated.
///
/// With the translation removed from g (remove_translation_from_coupling) Pi satisfies the acoustic sum rule: the
/// largest sum_atom' Pi(atom alpha, atom' beta) is stored for the recap as a check.
mdarray<std::complex<double>, 2> Lattice::static_response(const GridStructure& gridstructure__,
                                                           const parallel::Decomposition& decomposition__,
                                                           const electron::System& system__,
                                                           electron::MeanField* meanfield__)
{
    PROFILE("phonon::Lattice::static_response");
    ModelResponse response(gridstructure__, decomposition__, system__, meanfield__, num_bands_);
    const int nk_local = response.nk_local();
    const bool interacting = response.interacting();

    /* k components, Wannier gauge: V = perturbation, drho = current solution x, map = chi0 V = F(x),
       residual = F(x) - x */
    BlockMatrix<std::complex<double>> V(k, nk_local, num_bands_, num_bands_);
    BlockMatrix<std::complex<double>> sigma(k, nk_local, num_bands_, num_bands_);
    BlockMatrix<std::complex<double>> drho(k, nk_local, num_bands_, num_bands_);
    BlockMatrix<std::complex<double>> map(k, nk_local, num_bands_, num_bands_);
    BlockMatrix<std::complex<double>> residual(k, nk_local, num_bands_, num_bands_);
    BlockMatrix<std::complex<double>> drho_last(k, nk_local, num_bands_, num_bands_);
    BlockMatrix<std::complex<double>> residual_last(k, nk_local, num_bands_, num_bands_);
    /* Anderson history: differences of the solutions and of the residuals between consecutive iterations */
    const int history = 10;
    std::vector<BlockMatrix<std::complex<double>>> dx, dr;

    mdarray<std::complex<double>, 2> Pi({num_modes_, num_modes_});
    Pi.fill(0.);
    response_iterations_ = 0;
    response_error_ = 0.;
    /* one problem for each displacement nu (the response is linear: displacements are independent) */
    for( int nu = 0; nu < num_modes_; ++nu ) {
        auto& g = coupling_[nu].get_Operator(Space::k);
        int iteration = 0;
        double error = 0.;
        if( !gtilde_.empty() ) {
            /* unscreened coupling: the solution is gtilde, computed by unscreen_coupling */
            std::copy(gtilde_[nu].begin(), gtilde_[nu].end(), drho.begin());
        }
        else if( !interacting ) {
            /* V = g does not depend on delta rho: one step is the solution */
            response.chi0(drho, g);
            iteration = 1;
        }
        else {
            /* Anderson mixing, starting from delta rho = 0 */
            drho.fill(0.);
            dx.clear();
            dr.clear();
            while( true ) {
                ++iteration;
                /* F(x) = chi0 (g + Sigma[x]) */
                response.self_energy(sigma, drho);
                for( int i = 0; i < V.get_TotalSize(); ++i ) {
                    *(V.begin() + i) = *(g.begin() + i) + *(sigma.begin() + i);
                }
                response.chi0(map, V);
                for( int i = 0; i < residual.get_TotalSize(); ++i ) {
                    *(residual.begin() + i) = *(map.begin() + i) - *(drho.begin() + i);
                }
                /* relative residual max|F(x) - x| / max|F(x)|, over all the ranks */
                double scale = max_abs(map, nullptr, *comm_);
                error = scale > 0. ? max_abs(residual, nullptr, *comm_) / scale : 0.;
                if( error < parameters_.response_tolerance ) {
                    std::copy(map.begin(), map.end(), drho.begin());
                    break;
                }
                if( iteration >= parameters_.response_max_iterations ) {
                    std::stringstream ss;
                    ss << "The static response of the electrons to the displacement " << nu << " did not converge: "
                       << "relative residual " << error << " after " << iteration << " iterations\n";
                    throw std::runtime_error(ss.str());
                }
                /* update the history with the last step */
                if( iteration > 1 ) {
                    if( int(dx.size()) == history ) {
                        dx.erase(dx.begin());
                        dr.erase(dr.begin());
                    }
                    dx.emplace_back(k, nk_local, num_bands_, num_bands_);
                    dr.emplace_back(k, nk_local, num_bands_, num_bands_);
                    for( int i = 0; i < drho.get_TotalSize(); ++i ) {
                        *(dx.back().begin() + i) = *(drho.begin() + i) - *(drho_last.begin() + i);
                        *(dr.back().begin() + i) = *(residual.begin() + i) - *(residual_last.begin() + i);
                    }
                }
                std::copy(drho.begin(), drho.end(), drho_last.begin());
                std::copy(residual.begin(), residual.end(), residual_last.begin());
                /* gamma = argmin |r - sum_j gamma_j dr_j|, then x <- x + r - sum_j gamma_j (dx_j + dr_j) */
                const int n = int(dr.size());
                std::vector<std::complex<double>> A(n * n), b(n);
                for( int i = 0; i < n; ++i ) {
                    for( int j = 0; j < n; ++j ) {
                        A[i * n + j] = dot(dr[i], dr[j], *comm_);
                    }
                    b[i] = dot(dr[i], residual, *comm_);
                }
                auto gamma = solve_small(A, b);
                for( int i = 0; i < drho.get_TotalSize(); ++i ) {
                    auto step = *(residual.begin() + i);
                    for( int j = 0; j < n; ++j ) {
                        step -= gamma[j] * (*(dx[j].begin() + i) + *(dr[j].begin() + i));
                    }
                    *(drho.begin() + i) += step;
                }
            }
        }
        /* worst case over the displacements, for the recap */
        response_iterations_ = std::max(response_iterations_, iteration);
        response_error_ = std::max(response_error_, error);

        /* column nu of Pi: Pi(mu, nu) = s/N sum_k Tr[g_mu delta rho_nu] (local k points here, reduced below;
           s/N applied at the end) */
        for( int mu = 0; mu < num_modes_; ++mu ) {
            auto& gmu = coupling_[mu].get_Operator(Space::k);
            for( int ik = 0; ik < nk_local; ++ik ) {
                for( int n = 0; n < num_bands_; ++n ) {
                    for( int m = 0; m < num_bands_; ++m ) {
                        Pi(mu, nu) += gmu(ik, n, m) * drho(ik, m, n);
                    }
                }
            }
        }
    }
#ifdef EDUS_MPI
    MPI_Allreduce(MPI_IN_PLACE, Pi.data(), int(Pi.get_TotalSize()), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm_->communicator());
#endif
    double num_k = coupling_[0].get_Operator(Space::k).get_MeshGrid()->get_TotalSize();
    /* Pi is hermitian (real symmetric at Gamma) up to the convergence error: symmetrize, and apply s/N */
    mdarray<std::complex<double>, 2> result({num_modes_, num_modes_});
    for( int mu = 0; mu < num_modes_; ++mu ) {
        for( int nu = 0; nu < num_modes_; ++nu ) {
            result(mu, nu) = 0.5 * (Pi(mu, nu) + std::conj(Pi(nu, mu))) * parameters_.spin_degeneracy / num_k;
        }
    }
    /* acoustic sum rule of Pi, only as a check: with the translation removed from g (acoustic_sum_rule) the sums
       sum_atom' Pi(atom alpha, atom' beta) must vanish */
    pi_asr_residual_ = 0.;
    for( int atom = 0; atom < num_atoms_; ++atom ) {
        for( int a = 0; a < 3; ++a ) {
            for( int b = 0; b < 3; ++b ) {
                std::complex<double> sum = 0.;
                for( int other = 0; other < num_atoms_; ++other ) {
                    sum += result(3 * atom + a, 3 * other + b);
                }
                pi_asr_residual_ = std::max(pi_asr_residual_, std::abs(sum));
            }
        }
    }
    return result;
}

/// @brief Normal modes of the force constants K__ at q = 0.
/// M u'' = -K u -> with v = M^{1/2} u: v'' = -D v, D = M^{-1/2} K M^{-1/2}, whose eigenvalues are omega^2 and whose
/// eigenvectors e(mu, lambda) are the polarization vectors of ph.x. At Gamma D is real symmetric (the imaginary part
/// of K is numerical noise and is dropped), so the eigenvectors are real
void Lattice::normal_modes(const mdarray<std::complex<double>, 2>& K__, std::vector<double>& omega2__,
                           mdarray<double, 2>& vectors__) const
{
    std::vector<double> D(num_modes_ * num_modes_);
    for( int mu = 0; mu < num_modes_; ++mu ) {
        for( int nu = 0; nu < num_modes_; ++nu ) {
            D[mu * num_modes_ + nu] = K__(mu, nu).real() / std::sqrt(mass_[mu] * mass_[nu]);
        }
    }
    std::vector<double> V;
    symmetric_eigen(D, num_modes_, omega2__, V);
    vectors__.initialize({num_modes_, num_modes_});
    for( int mu = 0; mu < num_modes_; ++mu ) {
        for( int lambda = 0; lambda < num_modes_; ++lambda ) {
            vectors__(mu, lambda) = V[mu * num_modes_ + lambda];
        }
    }
}

/// @brief Frequencies of the normal modes for the force constants K__. Unstable modes (omega^2 < 0) are returned as
/// negative frequencies, like ph.x does.
std::vector<double> Lattice::frequencies(const mdarray<std::complex<double>, 2>& K__) const
{
    std::vector<double> omega2;
    mdarray<double, 2> vectors;
    normal_modes(K__, omega2, vectors);
    std::vector<double> omega(num_modes_);
    for( int i = 0; i < num_modes_; ++i ) {
        omega[i] = (omega2[i] >= 0. ? 1. : -1.) * std::sqrt(std::abs(omega2[i]));
    }
    return omega;
}

/// @brief The lattice in the normal modes lambda of ph.x (the physical, Born-Oppenheimer phonons at Gamma):
///     Q_lambda = sum_mu e(mu, lambda) sqrt(M_mu) u_mu,     F_lambda = sum_mu e(mu, lambda) F_mu / sqrt(M_mu),
/// so that without the electrons Q'' = -omega^2 Q, and E_lambda = 1/2 dQ^2 + 1/2 omega^2 Q^2.
/// The lattice of the Ehrenfest dynamics is classical, a coherent state of the phonons with
/// <b_lambda> = (omega Q + i dQ)/sqrt(2 omega): its number of phonons is |<b>|^2 = E_lambda/omega (hbar = 1).
/// With the adiabatic reference none or dynamic the dynamics uses these force constants and
/// sum_lambda E_lambda = E_lattice; with static it uses K - Pi, and sum_lambda E_lambda = E_lattice + 1/2 u.Pi.u, the
/// energy of the lattice with the adiabatic response of the electrons.
/// In a degenerate subspace (e.g. the E modes of hBN) the single modes are an arbitrary choice: only their sum is meaningful.
ModeProjection Lattice::project_on_modes(const Coordinates& x__, const std::vector<double>& F__) const
{
    /* below this frequency (acoustic modes) the number of phonons is not defined, and set to 0 */
    const double omega_min = 1. / 219474.6313705;
    ModeProjection p;
    for( auto* v : {&p.Q, &p.dQ, &p.F, &p.energy, &p.population} ) {
        v->assign(num_modes_, 0.);
    }
    for( int lambda = 0; lambda < num_modes_; ++lambda ) {
        for( int mu = 0; mu < num_modes_; ++mu ) {
            double e = mode_vectors_(mu, lambda);
            double sqrt_mass = std::sqrt(mass_[mu]);
            p.Q[lambda] += e * sqrt_mass * x__.displacement(0, mu).real();
            p.dQ[lambda] += e * sqrt_mass * x__.velocity(0, mu).real();
            p.F[lambda] += e * F__[mu] / sqrt_mass;
        }
        p.energy[lambda] = 0.5 * p.dQ[lambda] * p.dQ[lambda] + 0.5 * mode_omega2_[lambda] * p.Q[lambda] * p.Q[lambda];
        double omega = std::sqrt(std::abs(mode_omega2_[lambda]));
        p.population[lambda] = omega > omega_min ? p.energy[lambda] / omega : 0.;
    }
    return p;
}

/// @brief Lattice at t = 0: displaced by initial_displacement (or at equilibrium) and at rest
void Lattice::initial_condition(Coordinates& x__) const
{
    x__.initialize(num_qpoints(), num_modes_);
    for( int mu = 0; mu < int(parameters_.initial_displacement.size()); ++mu ) {
        x__.displacement(0, mu) = parameters_.initial_displacement[mu];
    }
}

/// @brief Electrons in the displaced lattice: H(R) += sum_mu u_mu g_mu(R), on the local R points.
/// It is called on the R component of the Hamiltonian, after H0 + E.r and the self energy and before the Peierls
/// phase, like any other lattice-periodic term of the Hamiltonian.
void Lattice::add_coupling(Operator<std::complex<double>>& H__, const Coordinates& x__) const
{
    auto& HR = H__.get_Operator(R);
    for( int mu = 0; mu < num_modes_; ++mu ) {
        /* at q = 0 the displacement is real (only its real part is used, to keep H hermitian) */
        double u = x__.displacement(0, mu).real();
        if( u == 0. ) {
            continue;
        }
        auto& gR = coupling_[mu].get_Operator(R);
        #pragma omp parallel for schedule(static)
        for( int iR = 0; iR < HR.get_nblocks(); ++iR ) {
            for( int i = 0; i < num_bands_; ++i ) {
                for( int j = 0; j < num_bands_; ++j ) {
                    HR(iR, i, j) += u * gR(iR, i, j);
                }
            }
        }
    }
}

/// @brief Force of the electrons on the lattice, F_mu = s/N sum_k Tr[g_mu(k) (rho(k) - rho_ref(k))].
///
/// It is computed in R, where rho is already available during the propagation (the self energy is computed in R
/// too), using Parseval with the convention O(k) = sum_R e^{ik.R} O(R):
///     1/N sum_k Tr[A(k) B(k)] = sum_R Tr[A(R) B(-R)] = sum_R sum_mn A_mn(R) conj(B_mn(R))   (B hermitian).
/// rho__ must be the physical density matrix (without the Peierls phase) with its R component up to date;
/// reference__ is rho0 (only the change of rho pushes the atoms, at equilibrium the geometry is relaxed) or rho_BO
/// (dynamic adiabatic reference). The atoms feel -F (see derivative).
std::vector<double> Lattice::force(const Operator<std::complex<double>>& rho__, const Operator<std::complex<double>>& reference__) const
{
    auto& rho = rho__.get_Operator(R);
    auto& rho0 = reference__.get_Operator(R);
    std::vector<double> F(num_modes_, 0.);
    for( int mu = 0; mu < num_modes_; ++mu ) {
        auto& gR = coupling_[mu].get_Operator(R);
        double sum = 0.;
        #pragma omp parallel for reduction(+:sum)
        for( int iR = 0; iR < gR.get_nblocks(); ++iR ) {
            for( int i = 0; i < num_bands_; ++i ) {
                for( int j = 0; j < num_bands_; ++j ) {
                    sum += (gR(iR, i, j) * std::conj(rho(iR, i, j) - rho0(iR, i, j))).real();
                }
            }
        }
        /* EDUS propagates one spin channel: the force of all the electrons has the spin degeneracy s */
        F[mu] = parameters_.spin_degeneracy * sum;
    }
    /* each rank has summed its R points. With acoustic_sum_rule the total force on the center of mass is zero here
       without corrections: the translation was removed from g (remove_translation_from_coupling) */
#ifdef EDUS_MPI
    MPI_Allreduce(MPI_IN_PLACE, F.data(), num_modes_, MPI_DOUBLE, MPI_SUM, comm_->communicator());
#endif
    return F;
}

/// @brief Right hand side of the equations of motion of the lattice, written as a first-order system:
///     du/dt = v,     dv/dt = -(K u + F)/M - (2/tau) v
/// Newton's equation M u'' = -K u - F with the harmonic force -K u, the force -F of the electrons and a
/// phenomenological friction that makes the amplitude decay as exp(-t/tau).
void Lattice::derivative(Coordinates& dx__, const Coordinates& x__, const std::vector<double>& F__) const
{
    double friction = parameters_.damping_time > 1.e-07 ? 2. / parameters_.damping_time : 0.;
    for( int iq = 0; iq < x__.num_qpoints(); ++iq ) {
        for( int mu = 0; mu < num_modes_; ++mu ) {
            /* harmonic force: (K u)_mu, coupling all the cartesian coordinates at the same q */
            std::complex<double> Ku = 0.;
            for( int nu = 0; nu < num_modes_; ++nu ) {
                Ku += force_constants_(iq, mu, nu) * x__.displacement(iq, nu);
            }
            /* only q = 0 is coupled to the electrons */
            double F = (iq == 0 ? F__[mu] : 0.);
            dx__.displacement(iq, mu) = x__.velocity(iq, mu);
            dx__.velocity(iq, mu) = -(Ku + F) / mass_[mu] - friction * x__.velocity(iq, mu);
        }
    }
}

/// @brief Energy of the lattice per unit cell: kinetic 1/2 M v^2 plus harmonic 1/2 u.K.u, with the force
/// constants used in the dynamics. The coupling energy u.F is computed by OutputManager, which has rho.
double Lattice::energy(const Coordinates& x__) const
{
    double energy = 0.;
    for( int iq = 0; iq < x__.num_qpoints(); ++iq ) {
        for( int mu = 0; mu < num_modes_; ++mu ) {
            energy += 0.5 * mass_[mu] * std::norm(x__.velocity(iq, mu));
            for( int nu = 0; nu < num_modes_; ++nu ) {
                energy += 0.5 * (std::conj(x__.displacement(iq, mu)) * force_constants_(iq, mu, nu) * x__.displacement(iq, nu)).real();
            }
        }
    }
    return energy;
}

/// @brief Recap in the output: files, options, checks (acoustic sum rule, static response, gauge of EPW) and the
/// frequencies at Gamma, in cm^-1, of the force constants used in the dynamics and of ph.x
void Lattice::print_recap() const
{
    auto to_str = [](bool b) { return std::string(b ? "True" : "False"); };
    const double ha_to_cm = 219474.6313705;
    output::title("PHONONS (EHRENFEST)");
    output::print("EPW directory            *", std::string(8, ' '), parameters_.epw_directory);
    output::print("epmatwp                  *", std::string(8, ' '), parameters_.epmatwp);
    output::print("Dynamical matrix         *", std::string(8, ' '), parameters_.dyn_file);
    output::print("Atoms, modes             *", num_atoms_, num_modes_);
    output::print("q points                 *", num_qpoints(), " (Gamma)");
    output::print("Spin degeneracy          *", parameters_.spin_degeneracy);
    output::print("Coupling of EPW          *", std::string(8, ' '), to_string(parameters_.coupling));
    output::print("Coupling scale           *", parameters_.coupling_scale);
    if( parameters_.coupling == CouplingType::screened ) {
        output::print("Unscreening: correction  *", unscreening_change_);
    }
    output::print("Damping time             *", parameters_.damping_time, " a.u.",
                  Convert(parameters_.damping_time, AuTime, FemtoSeconds), " fs");
    output::print("Acoustic sum rule        *", std::string(8, ' '), to_str(parameters_.acoustic_sum_rule));
    output::print("ASR correction           *", asr_correction_, " Ha/bohr^2");
    if( parameters_.acoustic_sum_rule ) {
        output::print("Translation in g removed *", translation_coupling_, " (max|sum_atom g| / max|g|)");
    }
    output::print("Adiabatic reference      *", std::string(8, ' '), to_string(parameters_.adiabatic_reference));
    if( parameters_.adiabatic_reference == AdiabaticReference::static_response ) {
        if( response_iterations_ == 0 ) {
            output::print("Static response          *", std::string(8, ' '), "chi0 g_s from the unscreening (exact)");
        }
        else {
            output::print("Static response: iter.   *", response_iterations_);
            output::print("Static response: resid.  *", response_error_);
        }
        output::print("ASR of Pi: max|sum|      *", pi_asr_residual_, " Ha/bohr^2");
    }
    output::print("max|H_EPW - H_tb|        *", gauge_check_, " Ha", Convert(gauge_check_, AuEnergy, ElectronVolt), " eV");
    if( gauge_check_ > 1.e-4 ) {
        output::print("WARNING: the Hamiltonian of EPW is not the one of the tight-binding model: the Wannier functions "
                      "(or their order and phases) are different and the coupling is in another gauge");
    }
    output::print("Frequencies at Gamma (cm^-1): of the force constants used in the dynamics (without the static response of the");
    output::print("electrons for the static reference), and of the dynamical matrix of ph.x:");
    for( int i = 0; i < num_modes_; ++i ) {
        output::print("   ", i, frequencies_[i] * ha_to_cm, frequencies_input_[i] * ha_to_cm);
    }
    output::stars();
}

}
