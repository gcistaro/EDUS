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

With --fc-grid n1 n2 n3 the model is written at every q (phonons at finite q):
- the coupling keeps the cell of the displaced atom, g_mu(R_e, R_g) = dH_mn(R_e)/du_mu(R_g) with m in the cell 0
  and n in the cell R_e: the atom of n is displaced in R_g = R_e, the atom of m in R_g = 0 (EPW convention,
  g(k,q) = sum_{R_e,R_g} e^{ik.R_e} e^{iq.R_g} g(R_e,R_g));
- the force constants are springs on the nearest-neighbour bonds, longitudinal k_l and transverse k_t in plane
  (k_t = --transverse-ratio * k_l) and k_z out of plane, written as the real-space force constants of q2r.x
  (<prefix>.fc), with C(R) between the atom of the block in the cell R and the other atom in the cell 0. The
  springs give the acoustic sum rule exactly, and at Gamma the same optical frequencies of the spring of the
  Gamma-only model (k_l + k_t = 2/3 k_xy, k_z = 1/3 k_z,Gamma for the three bonds of hBN).

Written files (EPW conventions, Rydberg atomic units, see src/Phonons/EPW.hpp):
  epwdata.fmt, wigner.fmt, crystal.fmt, <prefix>.epmatwp, <prefix>.dyn1 (text dynamical matrix of ph.x),
  <prefix>.fc (force constants of q2r.x, only with --fc-grid)

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
    p.add_argument("--fc-grid", type=int, nargs=3, default=None,
                   help="write the model at every q: g(R_e, R_g) and the force constants of q2r on this grid")
    p.add_argument("--transverse-ratio", type=float, default=0.3, help="k_t / k_l of the in-plane springs (with --fc-grid)")
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

    # with --fc-grid: the same coupling keeping the cell of the displaced atom, g(mode, R_e, m, n, R_g)
    with_q = args.fc_grid is not None
    if with_q:
        rg_list = [(0, 0, 0)]
        for ir, m, n, d in bonds:
            if np.linalg.norm(d) <= 1.01 * d0 and tuple(R[ir]) not in rg_list:
                rg_list.append(tuple(R[ir]))
        g_q = np.zeros((nmodes, nr, nw, nw, len(rg_list)), dtype=complex)
        for ir, m, n, d in bonds:
            dist = np.linalg.norm(d)
            if dist > 1.01 * d0:
                continue
            dtdd = -args.beta * Hphys[ir, m, n] / dist
            dhat = d / dist
            irg_n = rg_list.index(tuple(R[ir]))
            for ix in range(3):
                g_q[3 * n + ix, ir, m, n, irg_n] += dtdd * dhat[ix]
                g_q[3 * m + ix, ir, m, n, 0] -= dtdd * dhat[ix]
        assert np.allclose(g_q.sum(axis=4), g), "sum over R_g must give the coupling at q = 0"
        g_q_ry_bohr = g_q / RY_EV * BOHR_ANG

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

    # with --fc-grid: springs on the nearest-neighbour bonds, C(R) of q2r (Ha/bohr^2), (m1, m2, m3, mu, nu)
    if with_q:
        nr1, nr2, nr3 = args.fc_grid
        kl = 2. / 3. * k_ha[0] / (1. + args.transverse_ratio)
        kt = args.transverse_ratio * kl
        kz = k_ha[2] / 3.
        Cq = np.zeros((nr1, nr2, nr3, nmodes, nmodes))
        lat_bohr = lattice / BOHR_ANG
        tau_bohr = tau / BOHR_ANG
        nbonds = 0
        for a in range(nat):
            for b in range(nat):
                if a == b:
                    continue
                for i1 in range(-2, 3):
                    for i2 in range(-2, 3):
                        Rv = np.array([i1, i2, 0])
                        d = Rv @ lat_bohr + tau_bohr[a] - tau_bohr[b]    # from b in the cell 0 to a in the cell R
                        if abs(np.linalg.norm(d) - d0 / BOHR_ANG) > 1e-3:
                            continue
                        nbonds += 1
                        e = d / np.linalg.norm(d)
                        ez = np.array([0., 0., 1.])
                        Phi = kl * np.outer(e, e) + kt * (np.eye(3) - np.outer(e, e) - np.outer(ez, ez)) + kz * np.outer(ez, ez)
                        # E = 1/2 (u_a(R) - u_b(0)) Phi (u_a(R) - u_b(0)): C_ab(R) = -Phi, C_aa(0) += Phi
                        # (each bond is found once from each of its atoms: C_aa(0) gets the Phi of all the bonds of a)
                        m = (i1 % nr1, i2 % nr2, 0)
                        Cq[m[0], m[1], m[2], 3 * a:3 * a + 3, 3 * b:3 * b + 3] -= Phi
                        Cq[0, 0, 0, 3 * a:3 * a + 3, 3 * a:3 * a + 3] += Phi
        # check: at Gamma the same force constants of the spring model
        C_gamma = Cq.sum(axis=(0, 1, 2))
        assert np.allclose(C_gamma, C, atol=1e-12 * np.abs(C).max()), "force constants at Gamma differ from the spring model"

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
        nrr_g = len(rg_list) if with_q else 1
        f.write("%12d %12d %12d %12d %12d\n" % (nw, nr, nmodes, nrr_g, nrr_g))
        f.write(" ".join("%22.15f" % 0. for _ in range(9 * nat)) + "\n")     # Born charges
        f.write(" ".join("%22.15f" % x for x in np.eye(3).flatten()) + "\n")  # dielectric tensor
        # H(R) not divided by the degeneracies, Ry: loops ib, jb, ir
        for ib in range(nw):
            for jb in range(nw):
                for ir in range(nr):
                    f.write(fortran_complex(H[ir, ib, jb] / RY_EV) + "\n")

    with open(os.path.join(args.outdir, "wigner.fmt"), "w") as f:
        rgs = rg_list if with_q else [(0, 0, 0)]
        f.write("%d %d %d %d %d\n" % (nr, len(rgs), len(rgs), 1, 1))
        for ir in range(nr):
            length = np.linalg.norm(R[ir] @ lattice) / BOHR_ANG / alat
            f.write("%6d%6d%6d %22.15E\n%d\n" % (R[ir][0], R[ir][1], R[ir][2], length, degen[ir]))
        for _ in range(2):   # q vectors (phonons) and g vectors (electron-phonon): R = 0 only, or the R_g list
            for rg in rgs:
                length = np.linalg.norm(np.array(rg) @ lattice) / BOHR_ANG / alat
                f.write("%6d%6d%6d %22.15E\n%d\n" % (rg[0], rg[1], rg[2], length, 1))

    # epmatwp(nb, nb, nrr_k, nmodes, nrr_g), not divided by the degeneracies, Ry/bohr, Fortran order
    if with_q:
        raw = g_q_ry_bohr * degen[None, :, None, None, None]                # (mode, ir, m, n, irg)
        data = np.transpose(raw, (4, 0, 1, 3, 2)).astype(np.complex128)     # C order (irg, mode, ir, n, m)
    else:
        raw = g_ry_bohr * degen[None, :, None, None]             # (mode, ir, m, n)
        data = np.transpose(raw, (0, 1, 3, 2)).astype(np.complex128)   # C order (mode, ir, n, m): m fastest
    data.tofile(os.path.join(args.outdir, args.prefix + ".epmatwp"))

    if with_q:
        # force constants of q2r.x (Ry/bohr^2), ibrav = 0, no long-range part
        with open(os.path.join(args.outdir, args.prefix + ".fc"), "w") as f:
            f.write("%3d %4d %2d %12.7f %12.7f %12.7f %12.7f %12.7f %12.7f\n" % (nat, nat, 0, alat, 0, 0, 0, 0, 0))
            for v in at:
                f.write("%15.9f %15.9f %15.9f\n" % tuple(v))
            for a in range(nat):
                f.write("%12d  '%-3s'  %22.12f\n" % (a + 1, args.labels[a], args.masses[a] * AMU_RY))
            for a in range(nat):
                f.write("%5d %4d %17.10f %17.10f %17.10f\n" % (a + 1, a + 1, *tau_alat[a]))
            f.write(" F\n")
            f.write("%4d %4d %4d\n" % (nr1, nr2, nr3))
            for j1 in range(3):
                for j2 in range(3):
                    for na1 in range(nat):
                        for na2 in range(nat):
                            f.write("%4d%4d%4d%4d\n" % (j1 + 1, j2 + 1, na1 + 1, na2 + 1))
                            for m3 in range(nr3):
                                for m2 in range(nr2):
                                    for m1 in range(nr1):
                                        f.write("%4d%4d%4d %20.11E\n" % (m1 + 1, m2 + 1, m3 + 1,
                                                                          2. * Cq[m1, m2, m3, 3 * na1 + j1, 3 * na2 + j2]))
        print("force constants: %d bonds, k_l = %.6g, k_t = %.6g, k_z = %.6g Ha/bohr^2, grid %s"
              % (nbonds, kl, kt, kz, args.fc_grid))

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
