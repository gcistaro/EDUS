#include <cmath>
#include <sstream>
#include <stdexcept>

#include "Constants.hpp"
#include "Wannier/FortranBinary.hpp"
#include "Wannier/QEWavefunction.hpp"

std::string qe_wfc_filename(const std::string& savedir__, int ik__, int spin__)
{
    std::stringstream ss;
    ss << savedir__;
    if (!savedir__.empty() && savedir__.back() != '/') {
        ss << "/";
    }
    ss << "wfc";
    if (spin__ == 1) {
        ss << "up";
    } else if (spin__ == 2) {
        ss << "dw";
    }
    ss << ik__ << ".dat";
    return ss.str();
}

QEWavefunction QEWavefunction::read(const std::string& filename__, Content content__)
{
    QEWavefunction wf;
    FortranBinaryReader f(filename__);

    // ik, xk, ispin, gamma_only, scalef
    f.next_record();
    wf.ik = f.get<int32_t>();
    f.get(wf.xk.data(), 3);
    wf.ispin      = f.get<int32_t>();
    wf.gamma_only = f.get_logical();
    wf.scalef     = f.get<double>();

    // ngw, igwx, npol, nbnd
    f.next_record();
    wf.ngw  = f.get<int32_t>();
    wf.igwx = f.get<int32_t>();
    wf.npol = f.get<int32_t>();
    wf.nbnd = f.get<int32_t>();
    if (wf.igwx <= 0 || wf.nbnd <= 0 || (wf.npol != 1 && wf.npol != 2)) {
        throw std::runtime_error("QEWavefunction::read: inconsistent header in " + filename__);
    }

    // b1, b2, b3
    f.next_record();
    for (int i = 0; i < 3; ++i) {
        f.get(wf.b[i].data(), 3);
    }
    if (content__ == Content::header) {
        return wf;
    }

    // Miller indices, Fortran layout mill(3, igwx) -> our (igwx, 3) row major: same memory
    f.next_record();
    if (f.record_size() != sizeof(int32_t) * 3 * size_t(wf.igwx)) {
        throw std::runtime_error("QEWavefunction::read: wrong size of Miller indices record in " + filename__);
    }
    wf.miller.initialize({wf.igwx, 3});
    f.get(wf.miller.data(), 3 * size_t(wf.igwx));
    if (content__ == Content::header_and_miller) {
        return wf;
    }

    // coefficients, one record per band
    const int npw = wf.npol * wf.igwx;
    wf.evc.initialize({wf.nbnd, npw});
    for (int ib = 0; ib < wf.nbnd; ++ib) {
        f.next_record();
        if (f.record_size() != sizeof(std::complex<double>) * size_t(npw)) {
            throw std::runtime_error("QEWavefunction::read: wrong size of band record in " + filename__);
        }
        f.get(&wf.evc(ib, 0), size_t(npw));
    }
    return wf;
}

void QEWavefunction::write(const std::string& filename__) const
{
    FortranBinaryWriter f(filename__);
    f.put(int32_t(ik));
    f.put(xk.data(), 3);
    f.put(int32_t(ispin));
    f.put_logical(gamma_only);
    f.put(scalef);
    f.end_record();

    f.put(int32_t(ngw));
    f.put(int32_t(igwx));
    f.put(int32_t(npol));
    f.put(int32_t(nbnd));
    f.end_record();

    for (int i = 0; i < 3; ++i) {
        f.put(b[i].data(), 3);
    }
    f.end_record();

    f.put(miller.data(), 3 * size_t(igwx));
    f.end_record();

    for (int ib = 0; ib < nbnd; ++ib) {
        f.put(&evc(ib, 0), size_t(npol) * igwx);
        f.end_record();
    }
}

void QEWavefunction::expand_gamma()
{
    if (!gamma_only) {
        return;
    }
    // collect the G != 0 vectors
    std::vector<int> nonzero;
    for (int ig = 0; ig < igwx; ++ig) {
        if (miller(ig, 0) != 0 || miller(ig, 1) != 0 || miller(ig, 2) != 0) {
            nonzero.push_back(ig);
        }
    }
    const int igwx_new = igwx + int(nonzero.size());
    mdarray<int, 2> miller_new({igwx_new, 3});
    mdarray<std::complex<double>, 2> evc_new({nbnd, npol * igwx_new});
    for (int ig = 0; ig < igwx; ++ig) {
        for (int x = 0; x < 3; ++x) {
            miller_new(ig, x) = miller(ig, x);
        }
    }
    for (size_t j = 0; j < nonzero.size(); ++j) {
        for (int x = 0; x < 3; ++x) {
            miller_new(igwx + int(j), x) = -miller(nonzero[j], x);
        }
    }
    for (int ib = 0; ib < nbnd; ++ib) {
        for (int ipol = 0; ipol < npol; ++ipol) {
            for (int ig = 0; ig < igwx; ++ig) {
                evc_new(ib, ipol * igwx_new + ig) = evc(ib, ipol * igwx + ig);
            }
            for (size_t j = 0; j < nonzero.size(); ++j) {
                evc_new(ib, ipol * igwx_new + igwx + int(j)) = std::conj(evc(ib, ipol * igwx + nonzero[j]));
            }
        }
    }
    miller     = std::move(miller_new);
    evc        = std::move(evc_new);
    igwx       = igwx_new;
    ngw        = igwx_new;
    gamma_only = false;
}

std::array<std::array<double, 3>, 3> QEWavefunction::direct_lattice() const
{
    // A B^T = 2 pi I, with rows of A = a_i and rows of B = b_j  =>  A = 2 pi (B^T)^-1
    // (B^T)^-1 = (B^-1)^T ; B^-1 = adj(B)/det(B), columns of B^-1 are (b2 x b3, b3 x b1, b1 x b2)/det
    auto cross = [](const std::array<double, 3>& u, const std::array<double, 3>& v) {
        return std::array<double, 3>{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    };
    auto c0    = cross(b[1], b[2]);
    auto c1    = cross(b[2], b[0]);
    auto c2    = cross(b[0], b[1]);
    double det = b[0][0] * c0[0] + b[0][1] * c0[1] + b[0][2] * c0[2];
    std::array<std::array<double, 3>, 3> a;
    for (int x = 0; x < 3; ++x) {
        a[0][x] = 2. * pi * c0[x] / det;
        a[1][x] = 2. * pi * c1[x] / det;
        a[2][x] = 2. * pi * c2[x] / det;
    }
    return a;
}

std::array<double, 3> QEWavefunction::xk_crystal() const
{
    // k = sum_i k_i b_i  ->  k_i = k . a_i / (2 pi)
    auto a = direct_lattice();
    std::array<double, 3> kc;
    for (int i = 0; i < 3; ++i) {
        kc[i] = (xk[0] * a[i][0] + xk[1] * a[i][1] + xk[2] * a[i][2]) / (2. * pi);
    }
    return kc;
}

std::array<int, 3> QEWavefunction::max_miller() const
{
    std::array<int, 3> m{0, 0, 0};
    for (int ig = 0; ig < igwx; ++ig) {
        for (int x = 0; x < 3; ++x) {
            m[x] = std::max(m[x], std::abs(miller(ig, x)));
        }
    }
    return m;
}
