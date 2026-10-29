"""
Check of the force on the lattice from Output/Force_check.txt: the force of the dynamics, computed in R as
F_R = s sum_R sum_mn g_mn(R) conj(rho - rho_BO)_mn(R), must equal F_k = s/N sum_k Tr[g(k) (rho - rho_BO)(k)].

Usage: python3 check_force.py Output [--tolerance 1e-10] [--atol 1e-12]
  passes if max|F_R - F_k| <= tolerance * max|F_R| or <= atol (Ha/bohr): with the dynamic reference and no laser
  rho = rho_BO and the force is zero up to rounding
"""
import argparse
import os
import sys
import numpy as np

parser = argparse.ArgumentParser(description="force on the lattice: R formula against k formula")
parser.add_argument("output_dir")
parser.add_argument("--tolerance", type=float, default=1.e-10)
parser.add_argument("--atol", type=float, default=1.e-12)
args = parser.parse_args()

data = np.loadtxt(os.path.join(args.output_dir, "Force_check.txt"), ndmin=2)
diff, norm = data[:, 1].max(), data[:, 2].max()
message = f"max|F_R - F_k| = {diff:.2e}, max|F| = {norm:.2e} Ha/bohr"
if diff > args.tolerance * norm and diff > args.atol:
    print(f"[FAIL] force: {message}")
    sys.exit(1)
print(f"[OK] force: {message}")
