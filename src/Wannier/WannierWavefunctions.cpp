#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef EDUS_MPI
#include <fftw3-mpi.h>
#include "core/mpi/Communicator.hpp"
#else
#include <fftw3.h>
#endif

#include "Constants.hpp"
#include "LinearAlgebra/gemm.hpp"
#include "Wannier/WannierWavefunctions.hpp"

namespace {

constexpr double bohr_to_angstrom = Bohr_value * 1.e+10;

int mpi_rank()
{
#ifdef EDUS_MPI
    return mpi::Communicator::world().rank();
#else
    return 0;
#endif
}

int mpi_size()
{
#ifdef EDUS_MPI
    return mpi::Communicator::world().size();
#else
    return 1;
#endif
}

/// smallest n >= n_min whose only prime factors are 2, 3, 5, 7
int good_fft_size(int n_min__)
{
    for (int n = std::max(n_min__, 1);; ++n) {
        int m = n;
        for (int p : {2, 3, 5, 7}) {
            while (m % p == 0) {
                m /= p;
            }
        }
        if (m == 1) {
            return n;
        }
    }
}

inline int positive_mod(int a__, int n__)
{
    int r = a__ % n__;
    return r < 0 ? r + n__ : r;
}

} // namespace

WannierWavefunctionsParameters WannierWavefunctionsParameters::from_json(const nlohmann::json& in__)
{
    WannierWavefunctionsParameters p;
    p.qe_save_dir = in__.at("qe_save_dir").get<std::string>();
    p.seedname    = in__.at("seedname").get<std::string>();
    if (in__.contains("exclude_bands")) {
        p.exclude_bands = in__["exclude_bands"].get<std::vector<int>>();
    }
    if (in__.contains("dis_win_min")) {
        p.dis_win_min = in__["dis_win_min"].get<double>();
    }
    if (in__.contains("dis_win_max")) {
        p.dis_win_max = in__["dis_win_max"].get<double>();
    }
    if (in__.contains("spin")) {
        auto s = in__["spin"].get<std::string>();
        if (s == "none") {
            p.spin = 0;
        } else if (s == "up") {
            p.spin = 1;
        } else if (s == "down") {
            p.spin = 2;
        } else {
            throw std::runtime_error("spin must be one of none/up/down");
        }
    }
    if (in__.contains("fft_grid")) {
        auto v = in__["fft_grid"].get<std::vector<int>>();
        p.fft_grid = {v.at(0), v.at(1), v.at(2)};
    }
    if (in__.contains("supercell")) {
        auto v = in__["supercell"].get<std::vector<int>>();
        p.supercell = {v.at(0), v.at(1), v.at(2)};
    }
    if (in__.contains("wannier_list")) {
        p.wannier_list = in__["wannier_list"].get<std::vector<int>>();
    }
    p.write_bloch = in__.value("write_bloch", p.write_bloch);
    p.real_space  = in__.value("real_space", p.real_space);
    p.write_xsf   = in__.value("write_xsf", p.write_xsf);
    p.fix_phase   = in__.value("fix_phase", p.fix_phase);
    p.output_dir  = in__.value("output_dir", p.output_dir);
    return p;
}

WannierWavefunctions::WannierWavefunctions(const WannierWavefunctionsParameters& params__)
    : p_(params__)
{
    WannierGaugeParameters gp;
    gp.seedname      = p_.seedname;
    gp.exclude_bands = p_.exclude_bands;
    gp.dis_win_min   = p_.dis_win_min;
    gp.dis_win_max   = p_.dis_win_max;
    gauge_           = WannierGauge(gp);

    for (int x = 0; x < 3; ++x) {
        if (p_.supercell[x] < 1) {
            throw std::runtime_error("WannierWavefunctions: supercell must be >= 1");
        }
    }
    if (p_.wannier_list.empty()) {
        for (int n = 0; n < gauge_.num_wann(); ++n) {
            wannier_list_.push_back(n);
        }
    } else {
        for (int n : p_.wannier_list) {
            if (n < 1 || n > gauge_.num_wann()) {
                throw std::runtime_error("WannierWavefunctions: wannier_list out of range");
            }
            wannier_list_.push_back(n - 1);
        }
    }

    map_kpoints();
    read_atoms();

    if (gauge_.suspicious_window_kpoints() > 0 && mpi_rank() == 0) {
        std::cout << "WARNING: at " << gauge_.suspicious_window_kpoints()
                  << " k points the last rows of the outer window in " << p_.seedname
                  << "_u_dis.mat are zero.\n         Check that dis_win_min/dis_win_max are those of the .win file.\n";
    }
    for (int x = 0; x < 3; ++x) {
        if (p_.real_space && p_.supercell[x] > gauge_.mp_grid()[x] && mpi_rank() == 0) {
            std::cout << "WARNING: supercell larger than the k grid along direction " << x + 1
                      << ": w_n(r) is periodic on the Born-von Karman cell, periodic images will appear\n";
        }
    }
}

void WannierWavefunctions::map_kpoints()
{
    const int nk = gauge_.num_kpts();
    // read the headers of all the QE files (only the first 3 small records)
    std::vector<std::array<double, 3>> kqe;
    int max_miller[3] = {0, 0, 0};
    for (int ik = 1;; ++ik) {
        auto name = qe_wfc_filename(p_.qe_save_dir, ik, p_.spin);
        if (!std::filesystem::exists(name)) {
            break;
        }
        auto wf = QEWavefunction::read(name, QEWavefunction::Content::header);
        if (ik == 1) {
            b_       = wf.b;
            a_       = wf.direct_lattice();
            npol_    = wf.npol;
            nbnd_qe_ = wf.nbnd;
        } else if (wf.nbnd != nbnd_qe_ || wf.npol != npol_) {
            throw std::runtime_error("WannierWavefunctions: files in " + p_.qe_save_dir + " have different nbnd/npol");
        }
        kqe.push_back(wf.xk_crystal());
    }
    if (kqe.empty()) {
        throw std::runtime_error("WannierWavefunctions: no file " + qe_wfc_filename(p_.qe_save_dir, 1, p_.spin) +
                                 " (a QE compiled with HDF5 writes wfc<ik>.hdf5, not supported yet)");
    }
    // check early that the number of bands is consistent with wannier90
    gauge_.qe_to_w90_bands(nbnd_qe_);

    omega_ = std::abs(a_[0][0] * (a_[1][1] * a_[2][2] - a_[1][2] * a_[2][1]) -
                      a_[0][1] * (a_[1][0] * a_[2][2] - a_[1][2] * a_[2][0]) +
                      a_[0][2] * (a_[1][0] * a_[2][1] - a_[1][1] * a_[2][0]));

    qe_index_.assign(nk, -1);
    G0_.assign(nk, {0, 0, 0});
    for (int ik = 0; ik < nk; ++ik) {
        auto kw = gauge_.kpt(ik);
        // first try the same index (standard workflow with kmesh.pl), then all the others
        std::vector<int> candidates;
        if (ik < int(kqe.size())) {
            candidates.push_back(ik);
        }
        for (int jk = 0; jk < int(kqe.size()); ++jk) {
            if (jk != ik) {
                candidates.push_back(jk);
            }
        }
        for (int jk : candidates) {
            std::array<int, 3> G0;
            bool match = true;
            for (int x = 0; x < 3; ++x) {
                double d = kqe[jk][x] - kw[x];
                G0[x]    = int(std::lround(d));
                if (std::abs(d - G0[x]) > 1.e-5) {
                    match = false;
                }
            }
            if (match) {
                qe_index_[ik] = jk + 1;
                G0_[ik]       = G0;
                break;
            }
        }
        if (qe_index_[ik] < 0) {
            std::stringstream ss;
            ss << "WannierWavefunctions: k point " << ik + 1 << " (" << kw[0] << ", " << kw[1] << ", " << kw[2]
               << ") of " << p_.seedname << "_u.mat not found among the QE wavefunctions."
               << " The nscf calculation must contain the full wannier90 grid (nosym, noinv).";
            throw std::runtime_error(ss.str());
        }
    }

    // FFT grid: large enough to contain all the G vectors of all the k points (-> no aliasing)
    if (p_.fft_grid[0] * p_.fft_grid[1] * p_.fft_grid[2] > 0) {
        fft_grid_ = p_.fft_grid;
    } else if (p_.real_space) {
        for (int ik = 0; ik < nk; ++ik) {
            auto wf = QEWavefunction::read(qe_wfc_filename(p_.qe_save_dir, qe_index_[ik], p_.spin),
                                           QEWavefunction::Content::header_and_miller);
            // for gamma_only files -G is missing, but |m| is symmetric
            for (int ig = 0; ig < wf.igwx; ++ig) {
                for (int x = 0; x < 3; ++x) {
                    max_miller[x] = std::max(max_miller[x], std::abs(wf.miller(ig, x) + G0_[ik][x]));
                    if (wf.gamma_only) {
                        max_miller[x] = std::max(max_miller[x], std::abs(-wf.miller(ig, x) + G0_[ik][x]));
                    }
                }
            }
        }
        for (int x = 0; x < 3; ++x) {
            fft_grid_[x] = good_fft_size(2 * max_miller[x] + 1);
        }
    }
}

void WannierWavefunctions::read_atoms()
{
    // the save directory is <outdir>/<prefix>.save
    auto name = p_.qe_save_dir + "/data-file-schema.xml";
    std::ifstream f(name);
    if (!f.is_open()) {
        return;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string xml = ss.str();
    // use the last atomic_positions block (output structure)
    auto begin = xml.rfind("<atomic_positions>");
    auto end   = xml.find("</atomic_positions>", begin);
    if (begin == std::string::npos || end == std::string::npos) {
        return;
    }
    size_t pos = begin;
    while (true) {
        pos = xml.find("<atom ", pos);
        if (pos == std::string::npos || pos > end) {
            break;
        }
        auto name_begin = xml.find("name=\"", pos) + 6;
        auto name_end   = xml.find('"', name_begin);
        auto val_begin  = xml.find('>', pos) + 1;
        auto val_end    = xml.find("</atom>", val_begin);
        std::stringstream vs(xml.substr(val_begin, val_end - val_begin));
        std::array<double, 3> r;
        vs >> r[0] >> r[1] >> r[2];
        std::string symbol = xml.substr(name_begin, name_end - name_begin);
        // species labels like Fe1, Fe_up -> element symbol
        std::string element;
        for (char c : symbol) {
            if (!std::isalpha(static_cast<unsigned char>(c)) || element.size() == 2) {
                break;
            }
            element += element.empty() ? char(std::toupper(c)) : char(std::tolower(c));
        }
        atoms_.push_back({element, r});
        pos = val_end;
    }
}

QEWavefunction WannierWavefunctions::bloch_wannier_gauge(int ik__) const
{
    auto wf = QEWavefunction::read(qe_wfc_filename(p_.qe_save_dir, qe_index_[ik__], p_.spin));
    wf.expand_gamma();
    if (wf.nbnd != nbnd_qe_) {
        throw std::runtime_error("WannierWavefunctions: inconsistent nbnd in " +
                                 qe_wfc_filename(p_.qe_save_dir, qe_index_[ik__], p_.spin));
    }
    const int nw  = gauge_.num_wann();
    const int npw = wf.npol * wf.igwx;

    // psi~(n, G) = sum_m V(m, n) c(m, G)  ->  C = V^T evc
    auto Vt = gauge_.Vt_qe(ik__, wf.nbnd);
    QEWavefunction out;
    out.ik         = ik__ + 1;
    out.ispin      = wf.ispin;
    out.gamma_only = false;
    out.scalef     = wf.scalef;
    out.ngw        = wf.ngw;
    out.igwx       = wf.igwx;
    out.npol       = wf.npol;
    out.nbnd       = nw;
    out.b          = wf.b;
    out.evc.initialize({nw, npw});
    gemm(nw, npw, wf.nbnd, std::complex<double>(1.), Vt.data(), wf.nbnd, wf.evc.data(), npw,
         std::complex<double>(0.), out.evc.data(), npw);

    // k_QE = k_w90 + G0 : psi_{k_w90} has Miller indices shifted by +G0
    auto kw = gauge_.kpt(ik__);
    for (int x = 0; x < 3; ++x) {
        out.xk[x] = kw[0] * wf.b[0][x] + kw[1] * wf.b[1][x] + kw[2] * wf.b[2][x];
    }
    out.miller.initialize({wf.igwx, 3});
    for (int ig = 0; ig < wf.igwx; ++ig) {
        for (int x = 0; x < 3; ++x) {
            out.miller(ig, x) = wf.miller(ig, x) + G0_[ik__][x];
        }
    }
    return out;
}

void WannierWavefunctions::accumulate_real_space(int ik__, const QEWavefunction& wf__)
{
    const int N1 = fft_grid_[0], N2 = fft_grid_[1], N3 = fft_grid_[2];
    const int Nr = N1 * N2 * N3;
    auto S       = supercell_grid();
    int s0[3]    = {supercell_start(0), supercell_start(1), supercell_start(2)};
    auto kw      = gauge_.kpt(ik__);

    // 1D phases e^{i 2pi k_x i_x / N_x} on the supercell
    std::vector<std::complex<double>> ph[3];
    for (int x = 0; x < 3; ++x) {
        ph[x].resize(S[x]);
        for (int i = 0; i < S[x]; ++i) {
            ph[x][i] = std::exp(im * 2. * pi * kw[x] * double(s0[x] + i) / double(fft_grid_[x]));
        }
    }

    auto* buffer = reinterpret_cast<std::complex<double>*>(fftw_malloc(sizeof(fftw_complex) * Nr));
    auto plan    = fftw_plan_dft_3d(N1, N2, N3, reinterpret_cast<fftw_complex*>(buffer),
                                    reinterpret_cast<fftw_complex*>(buffer), FFTW_BACKWARD, FFTW_ESTIMATE);
    const double norm = 1. / std::sqrt(omega_);

    for (size_t iw = 0; iw < wannier_list_.size(); ++iw) {
        const int n = wannier_list_[iw];
        for (int ipol = 0; ipol < npol_; ++ipol) {
            std::fill(buffer, buffer + Nr, std::complex<double>(0.));
            // G vectors outside the grid are folded (equivalent to sampling on a coarser grid)
            for (int ig = 0; ig < wf__.igwx; ++ig) {
                int i1 = positive_mod(wf__.miller(ig, 0), N1);
                int i2 = positive_mod(wf__.miller(ig, 1), N2);
                int i3 = positive_mod(wf__.miller(ig, 2), N3);
                buffer[(i1 * N2 + i2) * N3 + i3] += wf__.evc(n, ipol * wf__.igwx + ig);
            }
            fftw_execute(plan); // u(r) = sum_G c(G) e^{iG.r}

            auto& w = w_[iw][ipol];
            #pragma omp parallel for
            for (int j1 = 0; j1 < S[0]; ++j1) {
                const int i1 = positive_mod(s0[0] + j1, N1);
                for (int j2 = 0; j2 < S[1]; ++j2) {
                    const int i2 = positive_mod(s0[1] + j2, N2);
                    const auto p12 = ph[0][j1] * ph[1][j2] * norm;
                    for (int j3 = 0; j3 < S[2]; ++j3) {
                        const int i3 = positive_mod(s0[2] + j3, N3);
                        w[(size_t(j1) * S[1] + j2) * S[2] + j3] += p12 * ph[2][j3] * buffer[(i1 * N2 + i2) * N3 + i3];
                    }
                }
            }
        }
    }
    fftw_destroy_plan(plan);
    fftw_free(buffer);
}

void WannierWavefunctions::run()
{
    const int nk   = gauge_.num_kpts();
    const int rank = mpi_rank();
    const int size = mpi_size();
    auto S         = supercell_grid();
    const size_t Nsuper = size_t(S[0]) * S[1] * S[2];

    if (rank == 0) {
        std::cout << "\n=== QE -> Wannier gauge ===\n"
                  << "  QE save dir        : " << p_.qe_save_dir << "\n"
                  << "  seedname           : " << p_.seedname << "\n"
                  << "  num_kpts           : " << nk << "  (mp_grid " << gauge_.mp_grid()[0] << " "
                  << gauge_.mp_grid()[1] << " " << gauge_.mp_grid()[2] << ")\n"
                  << "  num_wann           : " << gauge_.num_wann() << "\n"
                  << "  num_bands (w90/QE) : " << gauge_.num_bands() << " / " << nbnd_qe_ << "\n"
                  << "  disentanglement    : " << (gauge_.disentangled() ? "yes" : "no") << "\n"
                  << "  npol               : " << npol_ << "\n"
                  << "  max |V^+V - 1|     : " << std::scientific << std::setprecision(3)
                  << gauge_.unitarity_error() << std::defaultfloat << "\n";
        if (p_.real_space) {
            std::cout << "  FFT grid (cell)    : " << fft_grid_[0] << " " << fft_grid_[1] << " " << fft_grid_[2] << "\n"
                      << "  supercell          : " << p_.supercell[0] << " " << p_.supercell[1] << " "
                      << p_.supercell[2] << "\n";
        }
    }
    {
        // all the ranks write Bloch functions: error_code overload tolerates concurrent creation
        std::error_code ec;
        std::filesystem::create_directories(p_.output_dir, ec);
        if (!std::filesystem::is_directory(p_.output_dir)) {
            throw std::runtime_error("WannierWavefunctions: cannot create " + p_.output_dir);
        }
    }

    if (p_.real_space) {
        w_.assign(wannier_list_.size(),
                  std::vector<std::vector<std::complex<double>>>(npol_, std::vector<std::complex<double>>(Nsuper, 0.)));
    }

    max_orthonormality_error_ = 0.;
    for (int ik = rank; ik < nk; ik += size) {
        auto wf = bloch_wannier_gauge(ik);

        // orthonormality of the rotated Bloch functions (exact only for norm-conserving PP)
        const int npw = wf.npol * wf.igwx;
        for (int n1 = 0; n1 < wf.nbnd; ++n1) {
            for (int n2 = 0; n2 < wf.nbnd; ++n2) {
                std::complex<double> s = 0.;
                for (int ig = 0; ig < npw; ++ig) {
                    s += std::conj(wf.evc(n1, ig)) * wf.evc(n2, ig);
                }
                max_orthonormality_error_ = std::max(max_orthonormality_error_, std::abs(s - (n1 == n2 ? 1. : 0.)));
            }
        }

        if (p_.write_bloch) {
            wf.write(p_.output_dir + "/wfc" + std::to_string(ik + 1) + ".dat");
        }
        if (p_.real_space) {
            accumulate_real_space(ik, wf);
        }
    }

#ifdef EDUS_MPI
    {
        auto comm = mpi::Communicator::world().communicator();
        MPI_Allreduce(MPI_IN_PLACE, &max_orthonormality_error_, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (p_.real_space) {
            const size_t chunk = size_t(1) << 26;
            for (auto& wn : w_) {
                for (auto& wp : wn) {
                    for (size_t offset = 0; offset < wp.size(); offset += chunk) {
                        int count = int(std::min(chunk, wp.size() - offset));
                        MPI_Allreduce(MPI_IN_PLACE, wp.data() + offset, count, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
                    }
                }
            }
        }
    }
#endif

    if (rank == 0) {
        std::cout << "  max |<psi~_n|psi~_m> - delta| = " << std::scientific << std::setprecision(3)
                  << max_orthonormality_error_ << std::defaultfloat
                  << (max_orthonormality_error_ > 1.e-6
                          ? "  (large: ultrasoft/PAW pseudopotentials? then only <psi|S|psi> = 1)"
                          : "")
                  << "\n";
        if (p_.write_bloch) {
            std::cout << "  Bloch functions in the Wannier gauge written to " << p_.output_dir
                      << "/wfc<ik>.dat (QE format, ik = wannier90 k index)\n";
        }
    }
    if (!p_.real_space) {
        return;
    }

    for (size_t iw = 0; iw < wannier_list_.size(); ++iw) {
        for (auto& wp : w_[iw]) {
            for (auto& v : wp) {
                v /= double(nk);
            }
        }
        if (p_.fix_phase) {
            // same convention as wannier90: divide by the phase of the point of largest modulus
            size_t imax = 0;
            double vmax = 0.;
            int pmax    = 0;
            for (int ipol = 0; ipol < npol_; ++ipol) {
                for (size_t i = 0; i < Nsuper; ++i) {
                    if (std::norm(w_[iw][ipol][i]) > vmax) {
                        vmax = std::norm(w_[iw][ipol][i]);
                        imax = i;
                        pmax = ipol;
                    }
                }
            }
            auto phase = w_[iw][pmax][imax] / std::abs(w_[iw][pmax][imax]);
            for (auto& wp : w_[iw]) {
                for (auto& v : wp) {
                    v /= phase;
                }
            }
        }
    }

    if (rank == 0) {
        std::cout << "\n  Wannier functions in real space (supercell " << p_.supercell[0] << "x" << p_.supercell[1]
                  << "x" << p_.supercell[2] << ")\n"
                  << "      WF      norm       centre (Ang)                         spread (Ang^2)   max Im/Re\n";
        std::ofstream summary(p_.output_dir + "/wannier_centres.txt");
        summary << "# WF   norm   centre_x centre_y centre_z (Ang)   spread (Ang^2), from the real-space grid\n";
        for (size_t iw = 0; iw < wannier_list_.size(); ++iw) {
            double norm, spread;
            std::array<double, 3> c;
            centre_and_spread(int(iw), norm, c, spread);
            double ratmax = 0., wmax = 0.;
            if (npol_ == 1) {
                for (auto& v : w_[iw][0]) {
                    wmax = std::max(wmax, std::abs(v));
                }
                for (auto& v : w_[iw][0]) {
                    if (std::abs(v.real()) >= 0.01 * wmax) {
                        ratmax = std::max(ratmax, std::abs(v.imag()) / std::abs(v.real()));
                    }
                }
            }
            std::cout << std::fixed << std::setprecision(6) << "    " << std::setw(4) << wannier_list_[iw] + 1 << "  "
                      << std::setw(9) << norm << "  (" << std::setw(10) << c[0] << ", " << std::setw(10) << c[1]
                      << ", " << std::setw(10) << c[2] << " )  " << std::setw(12) << spread << "  " << std::setw(10)
                      << ratmax << std::defaultfloat << "\n";
            summary << std::setw(5) << wannier_list_[iw] + 1 << std::scientific << std::setprecision(10) << std::setw(20)
                    << norm << std::setw(20) << c[0] << std::setw(20) << c[1] << std::setw(20) << c[2] << std::setw(20)
                    << spread << std::defaultfloat << "\n";
            if (p_.write_xsf) {
                std::stringstream name;
                name << p_.output_dir << "/wannier_" << std::setw(5) << std::setfill('0') << wannier_list_[iw] + 1
                     << ".xsf";
                write_xsf(int(iw), name.str());
            }
        }
        std::cout << "  (norm = 1 and spreads comparable with wannier90 only if the supercell is large enough;\n"
                     "   centres/spreads from the real-space grid are approximate)\n";
    }
}

void WannierWavefunctions::centre_and_spread(int iw__, double& norm__, std::array<double, 3>& centre__,
                                             double& spread__) const
{
    auto S       = supercell_grid();
    int s0[3]    = {supercell_start(0), supercell_start(1), supercell_start(2)};
    const double dV = omega_ / (double(fft_grid_[0]) * fft_grid_[1] * fft_grid_[2]);
    double n = 0., r2 = 0.;
    std::array<double, 3> r1{0., 0., 0.};
    for (int j1 = 0; j1 < S[0]; ++j1) {
        for (int j2 = 0; j2 < S[1]; ++j2) {
            for (int j3 = 0; j3 < S[2]; ++j3) {
                size_t i   = (size_t(j1) * S[1] + j2) * S[2] + j3;
                double rho = 0.;
                for (int ipol = 0; ipol < npol_; ++ipol) {
                    rho += std::norm(w_[iw__][ipol][i]);
                }
                double f[3] = {double(s0[0] + j1) / fft_grid_[0], double(s0[1] + j2) / fft_grid_[1],
                               double(s0[2] + j3) / fft_grid_[2]};
                std::array<double, 3> r;
                for (int x = 0; x < 3; ++x) {
                    r[x] = f[0] * a_[0][x] + f[1] * a_[1][x] + f[2] * a_[2][x];
                }
                n += rho;
                for (int x = 0; x < 3; ++x) {
                    r1[x] += rho * r[x];
                }
                r2 += rho * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            }
        }
    }
    norm__   = n * dV;
    spread__ = r2 / n;
    for (int x = 0; x < 3; ++x) {
        centre__[x] = r1[x] / n;
        spread__ -= centre__[x] * centre__[x];
        centre__[x] *= bohr_to_angstrom;
    }
    spread__ *= bohr_to_angstrom * bohr_to_angstrom;
}

void WannierWavefunctions::write_xsf(int iw__, const std::string& filename__) const
{
    std::ofstream f(filename__);
    auto S    = supercell_grid();
    int s0[3] = {supercell_start(0), supercell_start(1), supercell_start(2)};
    f << "# Wannier function " << wannier_list_[iw__] + 1 << " from QE wavefunctions + " << p_.seedname
      << "_u.mat (EDUS)\n";
    f << "# " << (npol_ == 1 ? "real part" : "spinor: sqrt(|w_up|^2 + |w_dw|^2)") << ", units bohr^-3/2\n";
    f << "CRYSTAL\nPRIMVEC\n" << std::fixed << std::setprecision(7);
    for (int i = 0; i < 3; ++i) {
        f << std::setw(14) << a_[i][0] * bohr_to_angstrom << std::setw(14) << a_[i][1] * bohr_to_angstrom
          << std::setw(14) << a_[i][2] * bohr_to_angstrom << "\n";
    }
    f << "PRIMCOORD\n" << atoms_.size() << " 1\n";
    for (auto& at : atoms_) {
        f << std::setw(3) << at.first << std::setw(14) << at.second[0] * bohr_to_angstrom << std::setw(14)
          << at.second[1] * bohr_to_angstrom << std::setw(14) << at.second[2] * bohr_to_angstrom << "\n";
    }
    f << "\nBEGIN_BLOCK_DATAGRID_3D\n3D_field\nBEGIN_DATAGRID_3D_UNKNOWN\n";
    f << S[0] << " " << S[1] << " " << S[2] << "\n";
    for (int x = 0; x < 3; ++x) {
        double o = 0.;
        for (int i = 0; i < 3; ++i) {
            o += double(s0[i]) / fft_grid_[i] * a_[i][x];
        }
        f << std::setw(14) << o * bohr_to_angstrom;
    }
    f << "\n";
    for (int i = 0; i < 3; ++i) {
        double span = double(S[i] - 1) / fft_grid_[i];
        f << std::setw(14) << span * a_[i][0] * bohr_to_angstrom << std::setw(14) << span * a_[i][1] * bohr_to_angstrom
          << std::setw(14) << span * a_[i][2] * bohr_to_angstrom << "\n";
    }
    // XSF: first index runs fastest
    f << std::scientific << std::setprecision(5);
    int count = 0;
    for (int j3 = 0; j3 < S[2]; ++j3) {
        for (int j2 = 0; j2 < S[1]; ++j2) {
            for (int j1 = 0; j1 < S[0]; ++j1) {
                size_t i = (size_t(j1) * S[1] + j2) * S[2] + j3;
                double v;
                if (npol_ == 1) {
                    v = w_[iw__][0][i].real();
                } else {
                    v = std::sqrt(std::norm(w_[iw__][0][i]) + std::norm(w_[iw__][1][i]));
                }
                f << std::setw(13) << v;
                if (++count % 6 == 0) {
                    f << "\n";
                }
            }
        }
    }
    if (count % 6 != 0) {
        f << "\n";
    }
    f << "END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n";
}
