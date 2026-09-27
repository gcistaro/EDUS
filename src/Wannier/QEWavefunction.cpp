#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#ifdef EDUS_HDF5
#include <hdf5.h>
#endif

#include "Constants.hpp"
#include "Wannier/FortranBinary.hpp"
#include "Wannier/QEWavefunction.hpp"

namespace {

std::string qe_wfc_basename(const std::string& savedir__, int ik__, int spin__)
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
    ss << ik__;
    return ss.str();
}

#ifdef EDUS_HDF5
/// Reads one HDF5 attribute, matching the exact type wannier90/QE (Modules/qeh5_module.f90) uses to write it.
/// This mirrors the low-level calls of QE's own reader rather than the H5LT convenience API, since the array-typed
/// attributes below (a single element whose datatype is "array of 3 doubles", not a length-3 attribute) are a case
/// the generic H5LT calls are not guaranteed to unpack correctly.
herr_t hdf5_read_scalar_attr(hid_t loc__, const char* name__, hid_t mem_type__, void* value__)
{
    hid_t attr = H5Aopen_by_name(loc__, ".", name__, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        return -1;
    }
    herr_t status = H5Aread(attr, mem_type__, value__);
    H5Aclose(attr);
    return status;
}

/// Reads an attribute stored as a single element of array-of-3-doubles type (xk, bg1, bg2, bg3 in QE's HDF5 wfc
/// files), as opposed to a plain 3-element vector attribute.
herr_t hdf5_read_array3_attr(hid_t loc__, const char* name__, double value__[3])
{
    hid_t attr = H5Aopen_by_name(loc__, ".", name__, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        return -1;
    }
    hsize_t dims[1] = {3};
    hid_t arr_type  = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dims);
    herr_t status   = H5Aread(attr, arr_type, value__);
    H5Tclose(arr_type);
    H5Aclose(attr);
    return status;
}

herr_t hdf5_read_string_attr(hid_t loc__, const char* name__, std::string& value__)
{
    hid_t attr = H5Aopen_by_name(loc__, ".", name__, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        return -1;
    }
    hid_t file_type = H5Aget_type(attr);
    size_t nbytes    = H5Tget_size(file_type);
    std::vector<char> buf(nbytes + 1, '\0');
    herr_t status = H5Aread(attr, file_type, buf.data());
    H5Tclose(file_type);
    H5Aclose(attr);
    value__.assign(buf.data());
    return status;
}

/// Reads a wfc<ik>.hdf5 file (QE compiled with -D__HDF5), as written by Modules/io_base.f90::write_wfc.
/// File-root attributes: ik, xk (array-of-3-double), ispin, gamma_only (string), scale_factor, ngw, igwx, npol, nbnd.
/// Dataset "MillerIndices" (int, (igwx,3) in C order), with attributes bg1/bg2/bg3 (array-of-3-double).
/// Dataset "evc" (double, (nbnd, 2*npol*igwx) in C order: real/imag interleaved, contiguous per band).
QEWavefunction read_hdf5(const std::string& filename__, QEWavefunction::Content content__)
{
    QEWavefunction wf;
    hid_t file = H5Fopen(filename__.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("QEWavefunction::read: cannot open " + filename__);
    }
    auto fail = [&](const std::string& what__) {
        H5Fclose(file);
        throw std::runtime_error("QEWavefunction::read: cannot read attribute/dataset '" + what__ + "' in " +
                                 filename__ + " (unexpected HDF5 layout)");
    };
    int ik_int;
    if (hdf5_read_scalar_attr(file, "ik", H5T_NATIVE_INT, &ik_int) < 0) {
        fail("ik");
    }
    wf.ik = ik_int;
    if (hdf5_read_array3_attr(file, "xk", wf.xk.data()) < 0) {
        fail("xk");
    }
    if (hdf5_read_scalar_attr(file, "ispin", H5T_NATIVE_INT, &wf.ispin) < 0) {
        fail("ispin");
    }
    std::string gamma_str;
    if (hdf5_read_string_attr(file, "gamma_only", gamma_str) < 0) {
        fail("gamma_only");
    }
    wf.gamma_only = (gamma_str.find("TRUE") != std::string::npos || gamma_str.find("true") != std::string::npos);
    if (hdf5_read_scalar_attr(file, "scale_factor", H5T_NATIVE_DOUBLE, &wf.scalef) < 0) {
        fail("scale_factor");
    }
    int ngw_int, igwx_int, npol_int, nbnd_int;
    if (hdf5_read_scalar_attr(file, "ngw", H5T_NATIVE_INT, &ngw_int) < 0 ||
        hdf5_read_scalar_attr(file, "igwx", H5T_NATIVE_INT, &igwx_int) < 0 ||
        hdf5_read_scalar_attr(file, "npol", H5T_NATIVE_INT, &npol_int) < 0 ||
        hdf5_read_scalar_attr(file, "nbnd", H5T_NATIVE_INT, &nbnd_int) < 0) {
        fail("ngw/igwx/npol/nbnd");
    }
    wf.ngw = ngw_int; wf.igwx = igwx_int; wf.npol = npol_int; wf.nbnd = nbnd_int;
    if (wf.igwx <= 0 || wf.nbnd <= 0 || (wf.npol != 1 && wf.npol != 2)) {
        H5Fclose(file);
        throw std::runtime_error("QEWavefunction::read: inconsistent header in " + filename__);
    }

    hid_t mill_dset = H5Dopen2(file, "MillerIndices", H5P_DEFAULT);
    if (mill_dset < 0) {
        fail("MillerIndices");
    }
    if (hdf5_read_array3_attr(mill_dset, "bg1", wf.b[0].data()) < 0 ||
        hdf5_read_array3_attr(mill_dset, "bg2", wf.b[1].data()) < 0 ||
        hdf5_read_array3_attr(mill_dset, "bg3", wf.b[2].data()) < 0) {
        H5Dclose(mill_dset);
        fail("bg1/bg2/bg3");
    }
    if (content__ == QEWavefunction::Content::header) {
        H5Dclose(mill_dset);
        H5Fclose(file);
        return wf;
    }

    wf.miller.initialize({wf.igwx, 3});
    if (H5Dread(mill_dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, wf.miller.data()) < 0) {
        H5Dclose(mill_dset);
        fail("MillerIndices");
    }
    H5Dclose(mill_dset);
    if (content__ == QEWavefunction::Content::header_and_miller) {
        H5Fclose(file);
        return wf;
    }

    hid_t evc_dset = H5Dopen2(file, "evc", H5P_DEFAULT);
    if (evc_dset < 0) {
        fail("evc");
    }
    const int npw = wf.npol * wf.igwx;
    wf.evc.initialize({wf.nbnd, npw});
    // std::complex<double> has the same layout as two contiguous doubles (real, imag), matching how QE stores it
    if (H5Dread(evc_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, reinterpret_cast<double*>(wf.evc.data())) <
        0) {
        H5Dclose(evc_dset);
        fail("evc");
    }
    H5Dclose(evc_dset);
    H5Fclose(file);
    return wf;
}
#endif

} // namespace

std::string qe_wfc_filename(const std::string& savedir__, int ik__, int spin__)
{
    return qe_wfc_basename(savedir__, ik__, spin__) + ".dat";
}

std::string resolve_qe_wfc_file(const std::string& savedir__, int ik__, int spin__)
{
    auto base = qe_wfc_basename(savedir__, ik__, spin__);
    if (std::filesystem::exists(base + ".dat")) {
        return base + ".dat";
    }
    if (std::filesystem::exists(base + ".hdf5")) {
        return base + ".hdf5";
    }
    return {};
}

QEWavefunction QEWavefunction::read(const std::string& filename__, Content content__)
{
    if (filename__.size() >= 5 && filename__.compare(filename__.size() - 5, 5, ".hdf5") == 0) {
#ifdef EDUS_HDF5
        return read_hdf5(filename__, content__);
#else
        throw std::runtime_error(filename__ +
                                 " is an HDF5 wavefunction file, but EDUS was compiled without HDF5 support "
                                 "(cmake -DEDUS_HDF5=ON)");
#endif
    }
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
