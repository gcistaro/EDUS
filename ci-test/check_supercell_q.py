"""
Phonons at q against the supercell (ci-test hBN_phonons_q): the run of the primitive cell with qgrid [n1, 1, 1] and the
run of the supercell n1 x 1 x 1 at nq = 1 (utility/make_supercell_epw.py) describe the same system, so they must give
- the same displacements: u_(kappa, s)(t) of the supercell = u_kappa(R = s a1, t) = sum_q u_kappa(q, t) e^{i 2 pi q_1 s},
- the same current density (Velocity.txt),
- the same energy per primitive cell (Energy.txt E - E(0), Lattice.txt E_lattice: the supercell has n1 cells).
This tests signs and normalization of g(k,q), of the force F(q) and of K(q) independently of the equations at q.

Usage: python3 check_supercell_q.py Output supercell/Output n1 [--tolerance 1e-6]
"""
import argparse
import os
import sys
import numpy as np

parser = argparse.ArgumentParser(description="Primitive cell at q against the supercell at Gamma")
parser.add_argument("primitive")
parser.add_argument("supercell")
parser.add_argument("n1", type=int)
parser.add_argument("--tolerance", type=float, default=1.e-6)
args = parser.parse_args()
n1 = args.n1
load = lambda d, f: np.loadtxt(os.path.join(d, f), ndmin=2)

Lp, Ls = load(args.primitive, "Lattice.txt"), load(args.supercell, "Lattice.txt")
nmodes = (Lp.shape[1] - 3) // 3
lq = load(args.primitive, "LatticeQ.txt")
nq = n1
off = 2 + 2 * nq * nmodes
u_q = (lq[:, off::2] + 1j * lq[:, off + 1::2]).reshape(len(lq), nq, nmodes)
# q points of the grid [n1, 1, 1]: iq -> (iq/n1, 0, 0)
u_R = np.array([sum(u_q[:, iq, :] * np.exp(2j * np.pi * iq / n1 * s) for iq in range(nq)) for s in range(n1)])
u_R = np.transpose(u_R.real, (1, 0, 2)).reshape(len(lq), n1 * nmodes)     # (t, s*nmodes + mu), as the supercell
u_s = Ls[:, 3:3 + n1 * nmodes]

checks = [
    ("displacements u(R)", np.abs(u_s - u_R).max(), np.abs(u_s - u_s[0]).max()),
    ("velocity", np.abs(load(args.primitive, "Velocity.txt") - load(args.supercell, "Velocity.txt")).max(),
     np.abs(load(args.supercell, "Velocity.txt")).max()),
    ("E - E(0) per cell", np.abs(n1 * load(args.primitive, "Energy.txt")[:, 3] - load(args.supercell, "Energy.txt")[:, 3]).max(),
     np.abs(load(args.supercell, "Energy.txt")[:, 3]).max()),
    ("E_lattice per cell", np.abs(n1 * Lp[:, 1] - Ls[:, 1]).max(), np.abs(Ls[:, 1]).max()),
]
failures = 0
for name, diff, scale in checks:
    rel = diff / max(scale, 1.e-300)
    status = "OK" if rel <= args.tolerance else "FAIL"
    failures += status == "FAIL"
    print(f"[{status}] supercell: {name}, max difference {diff:.2e} (relative {rel:.2e})")
sys.exit(1 if failures else 0)
