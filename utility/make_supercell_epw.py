#!/usr/bin/env python3
"""
Supercell n1 x 1 x 1 at Gamma of a model at finite q, for the test of the phonons at q of EDUS (ci-test hBN_phonons_q).

A primitive model with the coupling at every q (tight-binding seedname_tb.dat, EPW files with g(R_e, R_g), force
constants C(R) of q2r.x, e.g. from make_synthetic_epw.py --fc-grid) is folded into the supercell A1 = n1 a1, A2 = a2,
A3 = a3, at Gamma only. The supercell at Gamma contains exactly the q = (j/n1, 0, 0) of the primitive cell, so that a
run of the supercell at nq = 1 must give the run of the primitive cell with qgrid [n1, 1, 1] (same electrons, same
lattice): this checks signs and normalization of g(k,q), F(q) and K(q) independently of the theory of EDUS at q.

Orbital/atom (n, s) of the supercell (n in the primitive cell s = 0..n1-1) -> index i = nw*s + n, position tau_n + s a1
(one orbital per atom, as in make_synthetic_epw.py):
  H'_{(m,s),(n,s')}(R')   = H_mn(R),  R = (n1 R'_1 + s' - s, R'_2, R'_3)
  r'                      = r_mn(R) + s a1 on the diagonal of R' = 0
  g'_{(kappa,t)}(R')      = sum of g_kappa(R_e = R, R_g) over the R_g of the cell t: R_g,1 = t - s (mod n1)
  C'_{(a,t),(b,t')}       = sum of C_ab(m) over the m of q2r with m_1 = t - t' (mod n1)

Usage:
  python3 make_supercell_epw.py primitive_tb.dat primitive_epw_dir primitive.fc outdir n1 [--prefix hbn]
Writes outdir/<prefix>_sc<n1>_tb.dat and outdir/{crystal.fmt, epwdata.fmt, wigner.fmt, <prefix>.epmatwp, <prefix>.dyn1}
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_synthetic_epw import read_tb, fortran_complex, RY_EV, BOHR_ANG, AMU_RY

import argparse
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("tb_file"); parser.add_argument("epw_dir"); parser.add_argument("fc_file"); parser.add_argument("outdir")
parser.add_argument("n1", type=int); parser.add_argument("--prefix", default="hbn")
args = parser.parse_args()
tb_file, epw_dir, fc_file, out, n1, prefix = args.tb_file, args.epw_dir, args.fc_file, args.outdir, args.n1, args.prefix
lattice, R, degen, H, r = read_tb(tb_file)
nr, nw = H.shape[0], H.shape[1]
Hp = H / degen[:, None, None]
rp = r / degen[:, None, None, None]

# --- primitive g(R_e, R_g) (Ry/bohr): epmatwp and wigner.fmt
wig = open(os.path.join(epw_dir, "wigner.fmt")).read().split()
nrr_k, nrr_q, nrr_g = int(wig[0]), int(wig[1]), int(wig[2])
pos = 5
Rk = []
for i in range(nrr_k):
    Rk.append(tuple(int(x) for x in wig[pos:pos + 3])); pos += 5
pos += 5 * nrr_q
Rg = []
for i in range(nrr_g):
    Rg.append(tuple(int(x) for x in wig[pos:pos + 3])); pos += 5
nmodes = 3 * nw
raw = np.fromfile(os.path.join(epw_dir, prefix + ".epmatwp"), dtype=np.complex128).reshape(nrr_g, nmodes, nrr_k, nw, nw)
g = np.transpose(raw, (1, 2, 4, 3, 0)) / degen[None, :, None, None, None]   # (mode, ir, m, n, irg), degen of Re (same list as tb)
assert [tuple(x) for x in R] == Rk

# --- primitive C(R) of q2r (Ha/bohr^2), C[m1, m2, m3, mu, nu]
lines = open(fc_file).read().split("\n")
hdr = lines[0].split(); ntyp, nat = int(hdr[0]), int(hdr[1])
i = 1 + 3 + ntyp + nat + 1
nrg = [int(x) for x in lines[i].split()]; i += 1
C = np.zeros((nrg[0], nrg[1], nrg[2], 3 * nat, 3 * nat))
while i < len(lines) and lines[i].strip():
    j1, j2, a1_, a2_ = [int(x) for x in lines[i].split()]; i += 1
    for _ in range(nrg[0] * nrg[1] * nrg[2]):
        m1, m2, m3, v = lines[i].split(); i += 1
        C[int(m1) - 1, int(m2) - 1, int(m3) - 1, 3 * (a1_ - 1) + j1 - 1, 3 * (a2_ - 1) + j2 - 1] = 0.5 * float(v)
assert nrg[0] % n1 == 0

# --- supercell
A = lattice.copy(); A[0] = n1 * lattice[0]
nws = nw * n1
idx = lambda n, s: nw * s + n
# H'(R') and r'(R'): <(m,s) 0| H |(n,s') R'> = H_mn(R), R = (n1 R'1 + s' - s, R'2, R'3)
Hs, rs = {}, {}
for ir in range(nr):
    for s in range(n1):
        for sp in range(n1):
            d1 = R[ir][0] - (sp - s)
            if d1 % n1:
                continue
            Rsc = (d1 // n1, R[ir][1], R[ir][2])
            Hs.setdefault(Rsc, np.zeros((nws, nws), complex))
            rs.setdefault(Rsc, np.zeros((3, nws, nws), complex))
            for m in range(nw):
                for n in range(nw):
                    Hs[Rsc][idx(m, s), idx(n, sp)] += Hp[ir, m, n]
                    rs[Rsc][:, idx(m, s), idx(n, sp)] += rp[ir, :, m, n]
for s in range(n1):   # r + s a1 on the diagonal of R' = 0
    for m in range(nw):
        rs[(0, 0, 0)][:, idx(m, s), idx(m, s)] += s * lattice[0]
Rlist = sorted(Hs.keys())
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "%s_sc%d_tb.dat" % (prefix, n1)), "w") as f:
    f.write("supercell %d x 1 x 1\n" % n1)
    for v in A:
        f.write("%22.15f %22.15f %22.15f\n" % tuple(v))
    f.write("%d\n%d\n" % (nws, len(Rlist)))
    for i0 in range(0, len(Rlist), 15):
        f.write(" ".join("%5d" % 1 for _ in Rlist[i0:i0 + 15]) + "\n")
    for Rsc in Rlist:
        f.write("\n%d %d %d\n" % Rsc)
        for n in range(nws):
            for m in range(nws):
                z = Hs[Rsc][m, n]
                f.write("%d %d %.15e %.15e\n" % (m + 1, n + 1, z.real, z.imag))
    for Rsc in Rlist:
        f.write("\n%d %d %d\n" % Rsc)
        for n in range(nws):
            for m in range(nws):
                z = rs[Rsc][:, m, n]
                f.write("%d %d %.15e %.15e %.15e %.15e %.15e %.15e\n" % (m + 1, n + 1, z[0].real, z[0].imag, z[1].real, z[1].imag, z[2].real, z[2].imag))

# g'(R'_e) at Gamma of the supercell: mode (kappa, t, alpha), sum over R_g = (n1 R'g1 + t - s, ...) of the atom of the cell t
nms = 3 * nws
gs = {Rsc: np.zeros((nms, nws, nws), complex) for Rsc in Rlist}
for ir in range(nr):
    for s in range(n1):
        for sp in range(n1):
            d1 = R[ir][0] - (sp - s)
            if d1 % n1:
                continue
            Rsc = (d1 // n1, R[ir][1], R[ir][2])
            for irg, rg in enumerate(Rg):
                t = (rg[0] + s) % n1      # cell of the displaced atom in the supercell
                for kappa in range(nw):
                    for a in range(3):
                        mu_p = 3 * kappa + a
                        mu_s = 3 * idx(kappa, t) + a
                        for m in range(nw):
                            for n in range(nw):
                                gs[Rsc][mu_s, idx(m, s), idx(n, sp)] += g[mu_p, ir, m, n, irg]

# C' at Gamma: C'_{(a,t),(b,t')} = sum_{m: m1 = t - t' mod n1-grid} C_ab(m)
Cs = np.zeros((nms, nms))
for t in range(n1):
    for tp in range(n1):
        for m1 in range(nrg[0]):
            if (m1 - (t - tp)) % n1:
                continue
            for a in range(nat):
                for b in range(nat):
                    Cs[3 * idx(a, t):3 * idx(a, t) + 3, 3 * idx(b, tp):3 * idx(b, tp) + 3] += C[m1, :, :, 3 * a:3 * a + 3, 3 * b:3 * b + 3].sum(axis=(0, 1))

# --- EPW files of the supercell (Gamma only)
cr = open(os.path.join(epw_dir, "crystal.fmt")).read().split("\n")
masses_ry = [float(x) for x in cr[8].split()]
lat_b = A / BOHR_ANG
alat = np.linalg.norm(lattice[0]) / BOHR_ANG
at = lat_b / alat
bg = np.linalg.inv(at).T
tau = np.array([rp[[i for i in range(nr) if not R[i].any()][0], :, n, n].real for n in range(nw)])
taus = np.array([(tau[n] + s * lattice[0]) for s in range(n1) for n in range(nw)])
mass_s = [masses_ry[n] for s in range(n1) for n in range(nw)]
ed = out
with open(os.path.join(ed, "crystal.fmt"), "w") as f:
    f.write("%12d\n%12d\n" % (nws, nms))
    f.write("%22.15f %12d\n" % (0., 0))
    f.write(" ".join("%22.15f" % x for x in at.flatten()) + "\n")
    f.write(" ".join("%22.15f" % x for x in bg.flatten()) + "\n")
    f.write("%22.15f\n%22.15f\n" % (abs(np.linalg.det(lat_b)), alat))
    f.write(" ".join("%22.15f" % x for x in (taus / BOHR_ANG / alat).flatten()) + "\n")
    f.write(" ".join("%22.15f" % m for m in mass_s) + "\n")
    f.write(" ".join("%12d" % (a + 1) for a in range(nws)) + "\n")
with open(os.path.join(ed, "epwdata.fmt"), "w") as f:
    f.write("%22.15f\n" % 0.)
    f.write("%12d %12d %12d %12d %12d\n" % (nws, len(Rlist), nms, 1, 1))
    f.write(" ".join("%22.15f" % 0. for _ in range(9 * nws)) + "\n")
    f.write(" ".join("%22.15f" % x for x in np.eye(3).flatten()) + "\n")
    for ib in range(nws):
        for jb in range(nws):
            for Rsc in Rlist:
                f.write(fortran_complex(Hs[Rsc][ib, jb] / RY_EV) + "\n")
with open(os.path.join(ed, "wigner.fmt"), "w") as f:
    f.write("%d %d %d %d %d\n" % (len(Rlist), 1, 1, 1, 1))
    for Rsc in Rlist:
        f.write("%6d%6d%6d %22.15E\n%d\n" % (Rsc[0], Rsc[1], Rsc[2], np.linalg.norm(np.array(Rsc) @ A) / BOHR_ANG / alat, 1))
    for _ in range(2):
        f.write("%6d%6d%6d %22.15E\n%d\n" % (0, 0, 0, 0., 1))
# g in Ha/bohr -> Ry/bohr, Fortran epmatwp(m, n, ir, mode)
graw = np.array([gs[Rsc] for Rsc in Rlist])             # (ir, mode, m, n), Ry/bohr as the primitive file
data = np.transpose(graw, (1, 0, 3, 2)).astype(np.complex128)
data.tofile(os.path.join(ed, prefix + ".epmatwp"))
with open(os.path.join(ed, prefix + ".dyn1"), "w") as f:
    f.write("Dynamical matrix file\nsupercell\n")
    f.write("%3d %4d %2d %12.7f %12.7f %12.7f %12.7f %12.7f %12.7f\n" % (nws, nws, 0, alat, 0, 0, 0, 0, 0))
    f.write("Basis vectors\n")
    for v in at:
        f.write("%15.9f %15.9f %15.9f\n" % tuple(v))
    for a in range(nws):
        f.write("%12d  'X%d'  %22.12f\n" % (a + 1, a, mass_s[a]))
    for a in range(nws):
        f.write("%5d %4d %17.10f %17.10f %17.10f\n" % (a + 1, a + 1, *(taus[a] / BOHR_ANG / alat)))
    f.write("\n     Dynamical  Matrix in cartesian axes\n\n")
    f.write("     q = (    0.000000000   0.000000000   0.000000000 ) \n\n")
    for a in range(nws):
        for b in range(nws):
            f.write("%5d%5d\n" % (a + 1, b + 1))
            for i in range(3):
                f.write("  ".join("%12.8f %12.8f" % (2. * Cs[3 * a + i, 3 * b + j], 0.) for j in range(3)) + "\n")
print("supercell: %d orbitals, %d R vectors, max|g'| = %.4f Ry/bohr" % (nws, len(Rlist), np.abs(graw).max()))
