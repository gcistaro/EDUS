#!/usr/bin/env python3
"""
Synthetic electron-phonon data in the format of EPW, for tests of the Ehrenfest dynamics of EDUS.

From a wannier90 tight-binding model (seedname_tb.dat) with one orbital per atom and two atoms (e.g. hBN):
- the hopping between different atoms depends on the length d of the bond as
      t(d) = t(d0) exp(-beta (d/d0 - 1)),      dt/dd = -beta t / d0
  so that the coupling to the cartesian displacement u of the atoms, g = dH/du, is
      dH_mn(R)/du(atom of n) = +dt/dd * d_hat,     dH_mn(R)/du(atom of m) = -dt/dd * d_hat
  with d = R + tau_n - tau_m the bond from orbital m in the cell 0 to orbital n in the cell R
  (only the nearest neighbours are modulated, the on-site energies are not);
- the force constants at Gamma are the ones of a spring between the two atoms,
      C = K (x) [[1, -1], [-1, 1]],   K = diag(k_xy, k_xy, k_z),
  with k chosen to give the requested frequencies of the optical modes at Gamma.

Written files (EPW conventions, Rydberg atomic units, see src/Phonons/EPW.hpp):
  epwdata.fmt, wigner.fmt, crystal.fmt, <prefix>.epmatwp, <prefix>.dyn1 (text dynamical matrix of ph.x)

Usage:
  python3 make_synthetic_epw.py tb_models/hBN_gap7.25eV_a2.5A_tb.dat outdir --prefix hbn \
          --masses 14.0067 10.811 --labels N B --beta 3.0 --omega-xy 1370 --omega-z 800
"""
import argparse
import os
import numpy as np

RY_EV = 13.605693122994           # eV
BOHR_ANG = 0.529177210903         # angstrom
AMU_ME = 1822.888486209           # electron masses
AMU_RY = AMU_ME / 2.              # Rydberg mass unit = 2 electron masses
HA_CM = 219474.6313705            # cm^-1


def read_tb(filename):
    """wannier90 seedname_tb.dat: lattice (rows, angstrom), R vectors, degeneracies, H(R) (eV), r(R) (angstrom)"""
    with open(filename) as f:
        lines = f.readlines()
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in (1, 2, 3)])
    tokens = " ".join(lines[4:]).split()
    pos = 0

    def take(n):
        nonlocal pos
        out = tokens[pos:pos + n]
        pos += n
        return out
    nw = int(take(1)[0])
    nr = int(take(1)[0])
    degen = np.array([int(x) for x in take(nr)])
    R = np.zeros((nr, 3), dtype=int)
    H = np.zeros((nr, nw, nw), dtype=complex)
    for ir in range(nr):
        R[ir] = [int(x) for x in take(3)]
        for _ in range(nw * nw):
            m, n, re, im = take(4)
            H[ir, int(m) - 1, int(n) - 1] = float(re) + 1j * float(im)
    r = np.zeros((nr, 3, nw, nw), dtype=complex)
    for ir in range(nr):
        Rr = [int(x) for x in take(3)]
        assert list(Rr) == list(R[ir]), "R vectors of H and r differ"
        for _ in range(nw * nw):
            v = take(8)
            m, n = int(v[0]) - 1, int(v[1]) - 1
            for ix in range(3):
                r[ir, ix, m, n] = float(v[2 + 2 * ix]) + 1j * float(v[3 + 2 * ix])
    return lattice, R, degen, H, r


def fortran_complex(z):
    return "({:.17E},{:.17E})".format(z.real, z.imag)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tb_file")
    p.add_argument("outdir")
    p.add_argument("--prefix", default="synthetic")
    p.add_argument("--masses", type=float, nargs="+", required=True, help="mass of each atom (amu), in the order of the orbitals")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--beta", type=float, default=3.0, help="d ln t / d ln d at the equilibrium bond length")
    p.add_argument("--omega-xy", type=float, default=1370., help="in-plane optical frequency at Gamma (cm^-1)")
    p.add_argument("--omega-z", type=float, default=800., help="out-of-plane optical frequency at Gamma (cm^-1)")
    p.add_argument("--fermi", type=float, default=0., help="Fermi energy written in epwdata.fmt (eV)")
    args = p.parse_args()

    lattice, R, degen, H, r = read_tb(args.tb_file)
    nr, nw = H.shape[0], H.shape[1]
    nat = nw
    if len(args.masses) != nat or len(args.labels) != nat:
        raise SystemExit("one orbital per atom is assumed: give %d masses and labels" % nat)
    if nat != 2:
        raise SystemExit("the force constants are written for two atoms")
    nmodes = 3 * nat

    # positions of the orbitals (= atoms): diagonal of r at R = 0 (angstrom)
    i0 = [i for i in range(nr) if not R[i].any()][0]
    tau = np.array([r[i0, :, n, n].real for n in range(nw)])

    # physical H(R) = H_file(R)/degen, bond lengths
    Hphys = H / degen[:, None, None]
    bonds = []
    for ir in range(nr):
        for m in range(nw):
            for n in range(nw):
                if m == n or abs(Hphys[ir, m, n]) < 1e-10:
                    continue
                d = R[ir] @ lattice + tau[n] - tau[m]
                bonds.append((ir, m, n, d))
    d0 = min(np.linalg.norm(b[3]) for b in bonds)

    # g_mu(R) = dH(R)/du_mu, eV/angstrom, mu = 3*atom + direction
    g = np.zeros((nmodes, nr, nw, nw), dtype=complex)
    for ir, m, n, d in bonds:
        dist = np.linalg.norm(d)
        if dist > 1.01 * d0:
            continue
        dtdd = -args.beta * Hphys[ir, m, n] / dist
        dhat = d / dist
        for ix in range(3):
            g[3 * n + ix, ir, m, n] += dtdd * dhat[ix]
            g[3 * m + ix, ir, m, n] -= dtdd * dhat[ix]
    g_ry_bohr = g / RY_EV * BOHR_ANG

    # force constants: spring between the atoms, omega^2 = k / reduced mass
    mass_me = np.array(args.masses) * AMU_ME
    mu_red = 1. / (1. / mass_me[0] + 1. / mass_me[1])
    k_ha = [(args.omega_xy / HA_CM) ** 2 * mu_red] * 2 + [(args.omega_z / HA_CM) ** 2 * mu_red]
    C = np.zeros((nmodes, nmodes))
    for a in range(nat):
        for b in range(nat):
            for ix in range(3):
                C[3 * a + ix, 3 * b + ix] = k_ha[ix] * (1. if a == b else -1.)
    C_ry = 2. * C

    os.makedirs(args.outdir, exist_ok=True)
    alat = np.linalg.norm(lattice[0]) / BOHR_ANG           # bohr
    at = lattice / BOHR_ANG / alat                           # alat units, rows = vectors
    bg = np.linalg.inv(at).T                                 # 2pi/alat units, rows = vectors
    omega = abs(np.linalg.det(lattice / BOHR_ANG))
    tau_alat = tau / BOHR_ANG / alat

    with open(os.path.join(args.outdir, "crystal.fmt"), "w") as f:
        f.write("%12d\n%12d\n" % (nat, nmodes))
        f.write("%22.15f %12d\n" % (0., 0))
        f.write(" ".join("%22.15f" % x for x in at.flatten()) + "\n")
        f.write(" ".join("%22.15f" % x for x in bg.flatten()) + "\n")
        f.write("%22.15f\n%22.15f\n" % (omega, alat))
        f.write(" ".join("%22.15f" % x for x in tau_alat.flatten()) + "\n")
        f.write(" ".join("%22.15f" % (m * AMU_RY) for m in args.masses) + "\n")
        f.write(" ".join("%12d" % (a + 1) for a in range(nat)) + "\n")

    with open(os.path.join(args.outdir, "epwdata.fmt"), "w") as f:
        f.write("%22.15f\n" % (args.fermi / RY_EV))
        f.write("%12d %12d %12d %12d %12d\n" % (nw, nr, nmodes, 1, 1))
        f.write(" ".join("%22.15f" % 0. for _ in range(9 * nat)) + "\n")     # Born charges
        f.write(" ".join("%22.15f" % x for x in np.eye(3).flatten()) + "\n")  # dielectric tensor
        # H(R) not divided by the degeneracies, Ry: loops ib, jb, ir
        for ib in range(nw):
            for jb in range(nw):
                for ir in range(nr):
                    f.write(fortran_complex(H[ir, ib, jb] / RY_EV) + "\n")

    with open(os.path.join(args.outdir, "wigner.fmt"), "w") as f:
        f.write("%d %d %d %d %d\n" % (nr, 1, 1, 1, 1))
        for ir in range(nr):
            length = np.linalg.norm(R[ir] @ lattice) / BOHR_ANG / alat
            f.write("%6d%6d%6d %22.15E\n%d\n" % (R[ir][0], R[ir][1], R[ir][2], length, degen[ir]))
        for _ in range(2):   # q vectors (phonons) and g vectors (electron-phonon): only R = 0
            f.write("%6d%6d%6d %22.15E\n%d\n" % (0, 0, 0, 0., 1))

    # epmatwp(nb, nb, nrr_k, nmodes, nrr_g), not divided by the degeneracies, Ry/bohr, Fortran order
    raw = g_ry_bohr * degen[None, :, None, None]             # (mode, ir, m, n)
    data = np.transpose(raw, (0, 1, 3, 2)).astype(np.complex128)   # C order (mode, ir, n, m): m fastest
    data.tofile(os.path.join(args.outdir, args.prefix + ".epmatwp"))

    with open(os.path.join(args.outdir, args.prefix + ".dyn1"), "w") as f:
        f.write("Dynamical matrix file\n")
        f.write("synthetic force constants from make_synthetic_epw.py\n")
        f.write("%3d %4d %2d %12.7f %12.7f %12.7f %12.7f %12.7f %12.7f\n" % (nat, nat, 0, alat, 0, 0, 0, 0, 0))
        f.write("Basis vectors\n")
        for v in at:
            f.write("%15.9f %15.9f %15.9f\n" % tuple(v))
        for a in range(nat):
            f.write("%12d  '%-3s'  %22.12f\n" % (a + 1, args.labels[a], args.masses[a] * AMU_RY))
        for a in range(nat):
            f.write("%5d %4d %17.10f %17.10f %17.10f\n" % (a + 1, a + 1, *tau_alat[a]))
        f.write("\n     Dynamical  Matrix in cartesian axes\n\n")
        f.write("     q = (    0.000000000   0.000000000   0.000000000 ) \n\n")
        for a in range(nat):
            for b in range(nat):
                f.write("%5d%5d\n" % (a + 1, b + 1))
                for i in range(3):
                    f.write("  ".join("%12.8f %12.8f" % (C_ry[3 * a + i, 3 * b + j], 0.) for j in range(3)) + "\n")

    print("d0 = %.6f angstrom, max|g| = %.4f eV/angstrom, k = %s Ha/bohr^2" % (d0, np.abs(g).max(), k_ha))
    print("files written in", args.outdir)


if __name__ == "__main__":
    main()
