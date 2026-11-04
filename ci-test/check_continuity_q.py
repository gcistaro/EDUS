"""
Continuity at finite q: dn(q)/dt = -i Omega q.j(q), from Output/DensityQ.txt and Output/CurrentQ.txt.

n(q) = sum_m e^{-iq.tau_m} n_m(q) is the density per cell (n_m of DensityQ.txt, tau_m the Wannier centers), j(q) the
current per unit volume of CurrentQ.txt, Omega the volume (area for a 2D grid) of the cell, dn/dt by finite differences.
In tight binding, with the position diagonal in the Wannier basis, the relation holds only to first order in q: the
relative error goes as q^2 (see corrente_q_finito.pdf), so the tolerance must be chosen for the q of the test.
Also checks that j(q = 0) is Velocity.txt. Only the q points with a density comparable to the largest are checked.

Usage: python3 check_continuity_q.py input.json Output [--tolerance 3e-2]
"""
import argparse
import json
import os
import sys
import numpy as np

ANGSTROM_TO_BOHR = 1. / 0.529177210903

parser = argparse.ArgumentParser(description="Continuity dn(q)/dt = -i Omega q.j(q)")
parser.add_argument("input")
parser.add_argument("output_dir")
parser.add_argument("--tolerance", type=float, default=3.e-2, help="maximum of |dn/dt + i Omega q.j| / max|dn/dt|")
args = parser.parse_args()

d = json.load(open(args.input))
tb_file = d["tb_file"] + "_tb.dat"
if not os.path.isabs(tb_file) and not os.path.exists(tb_file):
    tb_file = os.path.join(os.path.dirname(os.path.abspath(args.input)), tb_file)


def read_tb(filename):
    """lattice vectors (bohr, rows), R vectors and r(R) (bohr) of a wannier90 _tb.dat file"""
    lines = open(filename).read().split("\n")
    A = np.array([[float(x) for x in lines[i].split()] for i in (1, 2, 3)]) * ANGSTROM_TO_BOHR
    nw = int(lines[4].split()[0])
    nR = int(lines[5].split()[0])
    tokens = " ".join(lines[6:]).split()
    deg = np.array([int(x) for x in tokens[:nR]], dtype=float)
    p = nR
    Rs = np.zeros((nR, 3), dtype=int)
    for i in range(nR):
        Rs[i] = [int(x) for x in tokens[p:p + 3]]
        p += 3 + 4 * nw * nw
    r = np.zeros((nR, 3, nw, nw), dtype=complex)
    for i in range(nR):
        p += 3
        for _ in range(nw * nw):
            m, n = int(tokens[p]) - 1, int(tokens[p + 1]) - 1
            v = [float(x) for x in tokens[p + 2:p + 8]]
            r[i, :, m, n] = [v[0] + 1j * v[1], v[2] + 1j * v[3], v[4] + 1j * v[5]]
            p += 8
    return A, Rs, r * ANGSTROM_TO_BOHR / deg[:, None, None, None]


A, Rs, rR = read_tb(tb_file)
B = 2. * np.pi * np.linalg.inv(A).T
iR0 = np.where((Rs == 0).all(axis=1))[0][0]
tau = np.real(np.array([np.diag(rR[iR0, ix]) for ix in range(3)])).T
# volume of the cell as in EDUS: area of the periodic directions for a 2D grid
periodic = [ix for ix in range(3) if d["grid"][ix] > 1]
if len(periodic) == 3:
    volume = abs(np.linalg.det(A))
elif len(periodic) == 2:
    volume = np.linalg.norm(np.cross(A[periodic[0]], A[periodic[1]]))
else:
    volume = np.linalg.norm(A[periodic[0]])

qs = []
for line in open(f"{args.output_dir}/DensityQ.txt"):
    if not line.startswith("#"):
        break
    if line.startswith("#   iq ="):
        qs.append([float(x) for x in line.split(":")[1].split()])
qs = np.array(qs)
qs = qs - np.ceil(qs - 0.5)          # folded in (-1/2, 1/2], as in CurrentQ.txt
nq = len(qs)
D = np.loadtxt(f"{args.output_dir}/DensityQ.txt")
t = D[:, 0]
nb = (D.shape[1] - 1) // (2 * nq)
n = (D[:, 1::2] + 1j * D[:, 2::2]).reshape(len(t), nq, nb)
J = np.loadtxt(f"{args.output_dir}/CurrentQ.txt")
j = (J[:, 1::2] + 1j * J[:, 2::2]).reshape(len(t), nq, 3)
V = np.loadtxt(f"{args.output_dir}/Velocity.txt")

failures = 0
err0 = np.abs(j[:, 0, :] - (V[:, 0::2] + 1j * V[:, 1::2])).max()
scale0 = max(np.abs(V).max(), 1.e-30)
# Velocity.txt is written with 8 significant digits; when it is ~0 (IPA with the potential at q0 it is of second
# order) the two ways of computing it differ by the round-off of the larger terms, of the order of j(q != 0)
if err0 > 1.e-7 * (scale0 + np.abs(j).max()):
    print(f"[FAIL] continuity check: j(q=0) != Velocity.txt, max difference {err0:.3e}")
    failures += 1

dndt = np.zeros((len(t), nq), dtype=complex)
for iq in range(1, nq):
    q = qs[iq] @ B
    nphys = (n[:, iq, :] * np.exp(-1j * tau @ q)[None, :]).sum(axis=1)
    dndt[:, iq] = np.gradient(nphys, t)
largest = np.abs(dndt).max()
for iq in range(1, nq):
    scale = np.abs(dndt[:, iq]).max()
    if scale < 1.e-2 * largest:
        continue
    q = qs[iq] @ B
    err = np.abs(dndt[:, iq] + 1j * volume * (j[:, iq, :] @ q)).max() / scale
    status = "OK" if err < args.tolerance else "FAIL"
    print(f"[{status}] continuity at q = {qs[iq]}: max|dn/dt + i Omega q.j| / max|dn/dt| = {err:.3e} (tolerance {args.tolerance:.1e})")
    failures += err >= args.tolerance
sys.exit(1 if failures else 0)
