#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "Wannier/WannierGauge.hpp"

namespace {

bool file_exists(const std::string& name__)
{
    std::ifstream f(name__);
    return f.good();
}

} // namespace

UMatrixFile UMatrixFile::read(const std::string& filename__)
{
    std::ifstream f(filename__);
    if (!f.is_open()) {
        throw std::runtime_error("UMatrixFile::read: cannot open " + filename__);
    }
    UMatrixFile u;
    std::string header;
    std::getline(f, header);
    int num_wann_1, num_2;
    f >> u.num_kpts >> num_wann_1 >> num_2;
    if (!f) {
        throw std::runtime_error("UMatrixFile::read: cannot read dimensions in " + filename__);
    }
    // u.mat    : num_kpts num_wann num_wann
    // u_dis.mat: num_kpts num_wann num_bands
    u.num_wann = num_wann_1;
    u.num_rows = num_2;
    u.kpt.initialize({u.num_kpts, 3});
    u.U.initialize({u.num_kpts, u.num_rows, u.num_wann});
    for (int ik = 0; ik < u.num_kpts; ++ik) {
        f >> u.kpt(ik, 0) >> u.kpt(ik, 1) >> u.kpt(ik, 2);
        // written as ((u(i, j), i=1,num_rows), j=1,num_wann): row index runs fastest
        for (int j = 0; j < u.num_wann; ++j) {
            for (int i = 0; i < u.num_rows; ++i) {
                double re, im_;
                f >> re >> im_;
                u.U(ik, i, j) = std::complex<double>(re, im_);
            }
        }
        if (!f) {
            throw std::runtime_error("UMatrixFile::read: error while reading k point " + std::to_string(ik + 1) +
                                     " in " + filename__);
        }
    }
    return u;
}

mdarray<double, 2> read_w90_eig(const std::string& filename__, int& num_bands__, int& num_kpts__)
{
    std::ifstream f(filename__);
    if (!f.is_open()) {
        throw std::runtime_error("read_w90_eig: cannot open " + filename__);
    }
    std::vector<int> band, kpt;
    std::vector<double> e;
    int b, k;
    double en;
    while (f >> b >> k >> en) {
        band.push_back(b);
        kpt.push_back(k);
        e.push_back(en);
    }
    num_bands__ = *std::max_element(band.begin(), band.end());
    num_kpts__  = *std::max_element(kpt.begin(), kpt.end());
    if (size_t(num_bands__) * num_kpts__ != e.size()) {
        throw std::runtime_error("read_w90_eig: inconsistent number of lines in " + filename__);
    }
    mdarray<double, 2> eig({num_kpts__, num_bands__});
    for (size_t i = 0; i < e.size(); ++i) {
        eig(kpt[i] - 1, band[i] - 1) = e[i];
    }
    return eig;
}

WannierGauge::WannierGauge(const WannierGaugeParameters& params__)
    : exclude_bands_(params__.exclude_bands)
{
    auto u = UMatrixFile::read(params__.seedname + "_u.mat");
    if (u.num_rows != u.num_wann) {
        throw std::runtime_error("WannierGauge: " + params__.seedname + "_u.mat is not square");
    }
    num_kpts_ = u.num_kpts;
    num_wann_ = u.num_wann;
    kpt_      = u.kpt;

    const std::string udis_name = params__.seedname + "_u_dis.mat";
    disentangled_               = file_exists(udis_name);
    UMatrixFile udis;
    if (disentangled_) {
        udis = UMatrixFile::read(udis_name);
        if (udis.num_kpts != num_kpts_ || udis.num_wann != num_wann_) {
            throw std::runtime_error("WannierGauge: " + udis_name + " and _u.mat have different dimensions");
        }
        for (int ik = 0; ik < num_kpts_; ++ik) {
            for (int x = 0; x < 3; ++x) {
                if (std::abs(udis.kpt(ik, x) - kpt_(ik, x)) > 1.e-6) {
                    throw std::runtime_error("WannierGauge: k points in " + udis_name + " and _u.mat differ");
                }
            }
        }
        num_bands_ = udis.num_rows;
    } else {
        num_bands_ = num_wann_;
    }

    // eigenvalues (needed to locate the outer window when it does not contain all the bands)
    const std::string eig_name = params__.seedname + ".eig";
    if (file_exists(eig_name)) {
        int nb, nk;
        eig_ = read_w90_eig(eig_name, nb, nk);
        if (nb != num_bands_ || nk != num_kpts_) {
            throw std::runtime_error("WannierGauge: " + eig_name + " has " + std::to_string(nb) + " bands and " +
                                     std::to_string(nk) + " k points, expected " + std::to_string(num_bands_) +
                                     " and " + std::to_string(num_kpts_));
        }
        has_eig_ = true;
    }

    // outer window, as in wannier90 (dis_windows): bands with dis_win_min <= e <= dis_win_max
    win_first_.assign(num_kpts_, 0);
    ndimwin_.assign(num_kpts_, num_bands_);
    int suspicious_kpoints = 0;
    if (disentangled_) {
        for (int ik = 0; ik < num_kpts_; ++ik) {
            // number of rows of u_dis actually used: after ndimwin the rows are zero
            int last_nonzero = -1;
            for (int i = 0; i < num_bands_; ++i) {
                double norm = 0.;
                for (int j = 0; j < num_wann_; ++j) {
                    norm += std::norm(udis.U(ik, i, j));
                }
                if (norm > 1.e-12) {
                    last_nonzero = i;
                }
            }
            if (has_eig_) {
                int first = -1, count = 0;
                for (int m = 0; m < num_bands_; ++m) {
                    if (eig_(ik, m) >= params__.dis_win_min && eig_(ik, m) <= params__.dis_win_max) {
                        if (first < 0) {
                            first = m;
                        }
                        ++count;
                    }
                }
                if (count < num_wann_) {
                    throw std::runtime_error("WannierGauge: outer window contains less than num_wann bands at k point " +
                                             std::to_string(ik + 1));
                }
                if (last_nonzero >= count) {
                    throw std::runtime_error("WannierGauge: " + udis_name + " has non-zero rows outside the outer window at k point " +
                                             std::to_string(ik + 1) + ". Check dis_win_min/dis_win_max and exclude_bands.");
                }
                if (last_nonzero + 1 < count) {
                    // legitimate only if some band of the window is orthogonal to the disentangled subspace
                    ++suspicious_kpoints;
                }
                win_first_[ik] = first;
                ndimwin_[ik]   = count;
            } else if (last_nonzero != num_bands_ - 1) {
                throw std::runtime_error("WannierGauge: the outer window does not contain all the bands (k point " +
                                         std::to_string(ik + 1) + "): " + eig_name +
                                         " and dis_win_min/dis_win_max are needed to locate it.");
            }
        }
    }

    suspicious_window_kpoints_ = suspicious_kpoints;

    // V(k) = Udis(k) U(k), with Udis expanded to all the bands
    V_.initialize({num_kpts_, num_bands_, num_wann_});
    for (int ik = 0; ik < num_kpts_; ++ik) {
        if (!disentangled_) {
            for (int m = 0; m < num_bands_; ++m) {
                for (int n = 0; n < num_wann_; ++n) {
                    V_(ik, m, n) = u.U(ik, m, n);
                }
            }
            continue;
        }
        for (int i = 0; i < ndimwin_[ik]; ++i) {
            const int m = win_first_[ik] + i;
            for (int n = 0; n < num_wann_; ++n) {
                std::complex<double> v = 0.;
                for (int j = 0; j < num_wann_; ++j) {
                    v += udis.U(ik, i, j) * u.U(ik, j, n);
                }
                V_(ik, m, n) = v;
            }
        }
    }

    // Monkhorst-Pack grid from the k points (wannier90 needs a full, uniform grid)
    for (int x = 0; x < 3; ++x) {
        double kmin = 1.;
        for (int ik = 0; ik < num_kpts_; ++ik) {
            double f = kpt_(ik, x) - std::floor(kpt_(ik, x));
            f        = std::min(f, 1. - f);
            if (f > 1.e-6) {
                kmin = std::min(kmin, f);
            }
        }
        mp_grid_[x] = int(std::lround(1. / kmin));
    }
    if (mp_grid_[0] * mp_grid_[1] * mp_grid_[2] != num_kpts_) {
        throw std::runtime_error("WannierGauge: the k points in _u.mat do not form a uniform grid");
    }
}

std::vector<int> WannierGauge::qe_to_w90_bands(int nbnd_qe__) const
{
    std::vector<int> map(nbnd_qe__, -1);
    int counter = 0;
    for (int ib = 0; ib < nbnd_qe__; ++ib) {
        bool excluded = std::find(exclude_bands_.begin(), exclude_bands_.end(), ib + 1) != exclude_bands_.end();
        if (!excluded) {
            map[ib] = counter++;
        }
    }
    if (counter != num_bands_) {
        throw std::runtime_error("WannierGauge: QE has " + std::to_string(nbnd_qe__) + " bands, " +
                                 std::to_string(exclude_bands_.size()) + " excluded -> " + std::to_string(counter) +
                                 ", but wannier90 used num_bands = " + std::to_string(num_bands_) +
                                 ". Check exclude_bands.");
    }
    return map;
}

mdarray<std::complex<double>, 2> WannierGauge::Vt_qe(int ik__, int nbnd_qe__) const
{
    auto map = qe_to_w90_bands(nbnd_qe__);
    mdarray<std::complex<double>, 2> Vt({num_wann_, nbnd_qe__});
    for (int ib = 0; ib < nbnd_qe__; ++ib) {
        if (map[ib] < 0) {
            continue;
        }
        for (int n = 0; n < num_wann_; ++n) {
            Vt(n, ib) = V_(ik__, map[ib], n);
        }
    }
    return Vt;
}

mdarray<std::complex<double>, 2> WannierGauge::Hamiltonian_k(int ik__) const
{
    if (!has_eig_) {
        throw std::runtime_error("WannierGauge::Hamiltonian_k: .eig file not available");
    }
    mdarray<std::complex<double>, 2> H({num_wann_, num_wann_});
    for (int n1 = 0; n1 < num_wann_; ++n1) {
        for (int n2 = 0; n2 < num_wann_; ++n2) {
            std::complex<double> h = 0.;
            for (int m = 0; m < num_bands_; ++m) {
                h += std::conj(V_(ik__, m, n1)) * eig_(ik__, m) * V_(ik__, m, n2);
            }
            H(n1, n2) = h;
        }
    }
    return H;
}

double WannierGauge::unitarity_error() const
{
    double err = 0.;
    for (int ik = 0; ik < num_kpts_; ++ik) {
        for (int n1 = 0; n1 < num_wann_; ++n1) {
            for (int n2 = 0; n2 < num_wann_; ++n2) {
                std::complex<double> s = 0.;
                for (int m = 0; m < num_bands_; ++m) {
                    s += std::conj(V_(ik, m, n1)) * V_(ik, m, n2);
                }
                err = std::max(err, std::abs(s - (n1 == n2 ? 1. : 0.)));
            }
        }
    }
    return err;
}
