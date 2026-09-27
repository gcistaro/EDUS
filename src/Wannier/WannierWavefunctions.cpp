#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef EDUS_MPI
#include <fftw3-mpi.h>
#include "core/mpi/Communicator.hpp"
#else
#include <fftw3.h>
#endif

#if defined(EDUS_SPFFT) && defined(EDUS_MPI)
#include <spfft/spfft.hpp>
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
    p.fft_ranks_per_kpoint = in__.value("fft_ranks_per_kpoint", p.fft_ranks_per_kpoint);
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
    setup_fft_groups();
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
        auto name = resolve_qe_wfc_file(p_.qe_save_dir, ik, p_.spin);
        if (name.empty()) {
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
                                 " or " + p_.qe_save_dir + "/wfc1.hdf5 (checked for spin = " +
                                 std::to_string(p_.spin) + ")");
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
            auto wf = QEWavefunction::read(resolve_qe_wfc_file(p_.qe_save_dir, qe_index_[ik], p_.spin),
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

void WannierWavefunctions::setup_fft_groups()
{
    if (p_.fft_ranks_per_kpoint < 1) {
        throw std::runtime_error("WannierWavefunctions: fft_ranks_per_kpoint must be >= 1");
    }
#if !(defined(EDUS_SPFFT) && defined(EDUS_MPI))
    if (p_.fft_ranks_per_kpoint > 1) {
        throw std::runtime_error(
            "WannierWavefunctions: fft_ranks_per_kpoint > 1 requires EDUS built with "
            "-DEDUS_SPFFT=ON -DEDUS_MPI=ON (otherwise every rank would have to hold the "
            "full dense FFT box of a k point alone)");
    }
#endif
    if (mpi_size() % p_.fft_ranks_per_kpoint != 0) {
        throw std::runtime_error(
            "WannierWavefunctions: fft_ranks_per_kpoint must divide the total number of MPI ranks");
    }
    // groups of fft_ranks_per_kpoint consecutive ranks each own one k point at a time; different
    // groups process different k points in parallel, exactly as plain ranks did before (reduces to
    // fft_group_id_ == mpi_rank(), num_fft_groups_ == mpi_size() when fft_ranks_per_kpoint == 1).
    num_fft_groups_ = mpi_size() / p_.fft_ranks_per_kpoint;
    fft_group_id_   = mpi_rank() / p_.fft_ranks_per_kpoint;

#if defined(EDUS_SPFFT) && defined(EDUS_MPI)
    if (p_.fft_ranks_per_kpoint > 1) {
        fft_comm_ = std::make_unique<mpi::Communicator>();
        fft_comm_->generate(mpi::Communicator::world(), fft_group_id_);

        // Even slab split of the fft_grid_ z axis among the ranks of the group (the first
        // `remainder` ranks get one extra plane), fixed for the whole run: accumulate_real_space
        // only ever inserts into the z range owned by this rank into w_, so summing every rank's
        // contribution (done once, at the end of run(), exactly as before) reconstructs w_n(r).
        const int n3    = fft_grid_[2];
        const int gsize = fft_comm_->size();
        const int base  = n3 / gsize;
        const int rem    = n3 % gsize;
        fft_z_offset_.assign(gsize, 0);
        fft_z_len_.assign(gsize, 0);
        int z = 0;
        for (int r = 0; r < gsize; ++r) {
            fft_z_len_[r]    = base + (r < rem ? 1 : 0);
            fft_z_offset_[r] = z;
            z += fft_z_len_[r];
        }
    }
#endif
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
    auto wf = QEWavefunction::read(resolve_qe_wfc_file(p_.qe_save_dir, qe_index_[ik__], p_.spin));
    wf.expand_gamma();
    if (wf.nbnd != nbnd_qe_) {
        throw std::runtime_error("WannierWavefunctions: inconsistent nbnd in " +
                                 resolve_qe_wfc_file(p_.qe_save_dir, qe_index_[ik__], p_.spin));
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

#if defined(EDUS_SPFFT) && defined(EDUS_MPI)
void WannierWavefunctions::accumulate_real_space_distributed(int ik__, const QEWavefunction& wf__,
                                                              const std::complex<double>* ph0,
                                                              const std::complex<double>* ph1,
                                                              const std::complex<double>* ph2,
                                                              double norm__)
{
    const int N1 = fft_grid_[0], N2 = fft_grid_[1], N3 = fft_grid_[2];
    auto S       = supercell_grid();
    int s0[3]    = {supercell_start(0), supercell_start(1), supercell_start(2)};

    const int gsize = fft_comm_->size();
    const int grank = fft_comm_->rank();
    const int zOff  = fft_z_offset_[grank];
    const int zLen  = fft_z_len_[grank];

    // Columns (fixed x-y pairs) owned by this rank, round-robin on their linear index: every G
    // vector folding into an owned column is grouped here (>1 G's can fold to the same point when
    // fft_grid_ is smaller than 2*max_miller+1, exactly as the "+=" of the non-distributed path
    // below handles); this is recomputed per k point since the Miller indices differ per k, but is
    // the same for every band/spin component of this k, so it is built once here.
    struct LocalPoint {
        int i1, i2, i3;
        std::vector<int> igs;
    };
    std::vector<LocalPoint> local_points;
    {
        std::unordered_map<long long, int> index_of;
        for (int ig = 0; ig < wf__.igwx; ++ig) {
            int i1 = positive_mod(wf__.miller(ig, 0), N1);
            int i2 = positive_mod(wf__.miller(ig, 1), N2);
            int i3 = positive_mod(wf__.miller(ig, 2), N3);
            if ((i1 * N2 + i2) % gsize != grank) {
                continue;
            }
            long long key = (static_cast<long long>(i1) * N2 + i2) * N3 + i3;
            auto it        = index_of.find(key);
            if (it == index_of.end()) {
                index_of.emplace(key, int(local_points.size()));
                local_points.push_back({i1, i2, i3, {ig}});
            } else {
                local_points[it->second].igs.push_back(ig);
            }
        }
    }

    std::vector<int> local_indices;
    local_indices.reserve(3 * local_points.size());
    for (auto& p : local_points) {
        local_indices.push_back(p.i1);
        local_indices.push_back(p.i2);
        local_indices.push_back(p.i3);
    }
    const int num_local = int(local_points.size());

    spfft::Transform transform(/*maxNumThreads=*/1, fft_comm_->communicator(), SPFFT_EXCH_DEFAULT, SPFFT_PU_HOST,
                               SPFFT_TRANS_C2C, N1, N2, N3, zLen, num_local, SPFFT_INDEX_TRIPLETS,
                               local_indices.data());
    std::vector<std::complex<double>> local_freq(num_local);

    for (size_t iw = 0; iw < wannier_list_.size(); ++iw) {
        const int n = wannier_list_[iw];
        for (int ipol = 0; ipol < npol_; ++ipol) {
            for (int k = 0; k < num_local; ++k) {
                std::complex<double> c = 0.;
                for (int ig : local_points[k].igs) {
                    c += wf__.evc(n, ipol * wf__.igwx + ig);
                }
                local_freq[k] = c;
            }
            transform.backward(reinterpret_cast<const double*>(local_freq.data()), SPFFT_PU_HOST);
            auto* space = reinterpret_cast<std::complex<double>*>(transform.space_domain_data(SPFFT_PU_HOST));
            // SpFFT space-domain layout: (zLocal * N2 + i2) * N1 + i1, zLocal in [0, zLen)

            auto& w = w_[iw][ipol];
            #pragma omp parallel for
            for (int j1 = 0; j1 < S[0]; ++j1) {
                const int i1   = positive_mod(s0[0] + j1, N1);
                const auto ph1v = ph0[j1];
                for (int j2 = 0; j2 < S[1]; ++j2) {
                    const int i2  = positive_mod(s0[1] + j2, N2);
                    const auto p12 = ph1v * ph1[j2] * norm__;
                    for (int j3 = 0; j3 < S[2]; ++j3) {
                        const int i3 = positive_mod(s0[2] + j3, N3);
                        if (i3 < zOff || i3 >= zOff + zLen) {
                            continue; // owned by a different rank of this group
                        }
                        const int zLocal = i3 - zOff;
                        w[(size_t(j1) * S[1] + j2) * S[2] + j3] +=
                            p12 * ph2[j3] * space[(zLocal * N2 + i2) * N1 + i1];
                    }
                }
            }
        }
    }
}
#endif

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
    const double norm = 1. / std::sqrt(omega_);

#if defined(EDUS_SPFFT) && defined(EDUS_MPI)
    if (fft_comm_) {
        accumulate_real_space_distributed(ik__, wf__, ph[0].data(), ph[1].data(), ph[2].data(), norm);
        return;
    }
#endif

    auto* buffer = reinterpret_cast<std::complex<double>*>(fftw_malloc(sizeof(fftw_complex) * Nr));
    auto plan    = fftw_plan_dft_3d(N1, N2, N3, reinterpret_cast<fftw_complex*>(buffer),
                                    reinterpret_cast<fftw_complex*>(buffer), FFTW_BACKWARD, FFTW_ESTIMATE);

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
            if (p_.fft_ranks_per_kpoint > 1) {
                std::cout << "  fft_ranks_per_kpoint: " << p_.fft_ranks_per_kpoint << " (SpFFT, " << num_fft_groups_
                          << " k-point groups)\n";
            }
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
    // fft_group_id_/num_fft_groups_ reduce to rank/size when fft_ranks_per_kpoint == 1 (the
    // default): every rank is then its own group of size 1, exactly as before. With
    // fft_ranks_per_kpoint > 1, a whole group of ranks processes the same k point together
    // (each redundantly reads and rotates it; only the group leader writes/checks it once),
    // so that accumulate_real_space can distribute that k point's real-space FFT across the group.
    for (int ik = fft_group_id_; ik < nk; ik += num_fft_groups_) {
        auto wf = bloch_wannier_gauge(ik);

        if (is_fft_group_leader()) {
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
    auto S          = supercell_grid();
    int s0[3]       = {supercell_start(0), supercell_start(1), supercell_start(2)};
    const double dV = omega_ / (double(fft_grid_[0]) * fft_grid_[1] * fft_grid_[2]);
    // w_n(r) is built as (1/Nk) sum_k e^{ik.r} u_nk(r): with a finite k mesh it is exactly periodic
    // on the Born-von Karman supercell (mp_grid unit cells along each direction), regardless of how
    // large a chunk of it `supercell` samples. A plain (unwrapped) second moment is only correct as
    // long as the sampled box is small compared to that period; once `supercell` approaches or equals
    // mp_grid, density near one edge of the box is really close (through the periodic boundary) to
    // density near the opposite edge, and must be treated as such. Both passes below fold displacements
    // modulo mp_grid, so the result is insensitive to how large `supercell` is (as it should be).
    const auto& M = gauge_.mp_grid();

    // Pass 1: periodic ("circular") mean of the fractional coordinate along each direction, which
    // is well defined even when w_n(r) wraps around the supercell.
    double n = 0.;
    std::complex<double> z[3] = {0., 0., 0.};
    for (int j1 = 0; j1 < S[0]; ++j1) {
        const double f1 = double(s0[0] + j1) / fft_grid_[0];
        for (int j2 = 0; j2 < S[1]; ++j2) {
            const double f2 = double(s0[1] + j2) / fft_grid_[1];
            for (int j3 = 0; j3 < S[2]; ++j3) {
                const double f3 = double(s0[2] + j3) / fft_grid_[2];
                size_t i   = (size_t(j1) * S[1] + j2) * S[2] + j3;
                double rho = 0.;
                for (int ipol = 0; ipol < npol_; ++ipol) {
                    rho += std::norm(w_[iw__][ipol][i]);
                }
                n += rho;
                const double f[3] = {f1, f2, f3};
                for (int x = 0; x < 3; ++x) {
                    z[x] += rho * std::exp(im * 2. * pi * f[x] / double(M[x]));
                }
            }
        }
    }
    // preliminary centre, fractional coordinates (unit cell units), one representative per period
    double f0[3];
    for (int x = 0; x < 3; ++x) {
        f0[x] = double(M[x]) * std::arg(z[x]) / (2. * pi);
    }

    // Pass 2: second moment of the minimum-image displacement from f0 (each direction folded modulo
    // mp_grid), then the usual Var(r) = E[dr^2] - E[dr]^2 correction to refine the centre.
    std::array<double, 3> dr1{0., 0., 0.};
    double r2 = 0.;
    for (int j1 = 0; j1 < S[0]; ++j1) {
        double df1 = double(s0[0] + j1) / fft_grid_[0] - f0[0];
        df1 -= double(M[0]) * std::round(df1 / M[0]);
        for (int j2 = 0; j2 < S[1]; ++j2) {
            double df2 = double(s0[1] + j2) / fft_grid_[1] - f0[1];
            df2 -= double(M[1]) * std::round(df2 / M[1]);
            for (int j3 = 0; j3 < S[2]; ++j3) {
                double df3 = double(s0[2] + j3) / fft_grid_[2] - f0[2];
                df3 -= double(M[2]) * std::round(df3 / M[2]);
                size_t i   = (size_t(j1) * S[1] + j2) * S[2] + j3;
                double rho = 0.;
                for (int ipol = 0; ipol < npol_; ++ipol) {
                    rho += std::norm(w_[iw__][ipol][i]);
                }
                const double df[3] = {df1, df2, df3};
                std::array<double, 3> dr;
                for (int x = 0; x < 3; ++x) {
                    dr[x] = df[0] * a_[0][x] + df[1] * a_[1][x] + df[2] * a_[2][x];
                }
                for (int x = 0; x < 3; ++x) {
                    dr1[x] += rho * dr[x];
                }
                r2 += rho * (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
            }
        }
    }
    norm__   = n * dV;
    spread__ = r2 / n;
    std::array<double, 3> f0_cart{0., 0., 0.};
    for (int x = 0; x < 3; ++x) {
        f0_cart[x] = f0[0] * a_[0][x] + f0[1] * a_[1][x] + f0[2] * a_[2][x];
    }
    for (int x = 0; x < 3; ++x) {
        const double dr_mean = dr1[x] / n;
        spread__ -= dr_mean * dr_mean;
        centre__[x] = (f0_cart[x] + dr_mean) * bohr_to_angstrom;
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
