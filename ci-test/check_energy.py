"""
Checks the energy balance written by EDUS in Output/Energy.txt: without decay, the energy of the
electrons changes only because of the work of the field, E(t) - E(0) = W(t) = int_0^t P dt'.

The work is integrated here from the power P with a 3-point rule (error O(dt^3)), more accurate than
the trapezoidal rule used by EDUS, and the residual is compared with the maximum of |W|.

Usage: python3 check_energy.py Output/Energy.txt [--tolerance 1e-2]
"""
import argparse
import sys
import numpy as np

# columns of Energy.txt
TIME, E_BAND, E_MF, DELTA_E, POWER, WORK, RESIDUAL = range(7)


def cumulative_work(t, P):
    """int_{t_0}^{t_i} P dt for every i, with the quadratic through three consecutive points."""
    W = np.zeros_like(P)
    if len(P) < 3:
        W[1:] = np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(t))
        return W
    dt = np.diff(t)
    # first interval: quadratic through points 0, 1, 2
    W[1] = dt[0] / 12. * (5. * P[0] + 8. * P[1] - P[2])
    # interval [i, i+1]: quadratic through points i-1, i, i+1
    W[2:] = W[1] + np.cumsum(dt[1:] / 12. * (-P[:-2] + 8. * P[1:-1] + 5. * P[2:]))
    return W


parser = argparse.ArgumentParser(description="Check the energy balance of an EDUS run")
parser.add_argument("energy_file")
parser.add_argument("--tolerance", type=float, default=1.e-2, help="maximum of |E - E(0) - W| / max|W|")
args = parser.parse_args()

try:
    data = np.loadtxt(args.energy_file, ndmin=2)
except OSError as error:
    print(f"[FAIL] energy balance: {error}")
    sys.exit(1)

W = cumulative_work(data[:, TIME], data[:, POWER])
work = np.abs(W).max()
if work < 1.e-300:
    print("[OK] energy balance: no work done by the field")
    sys.exit(0)
ratio = np.abs(data[:, DELTA_E] - W).max() / work
if ratio > args.tolerance:
    print(f"[FAIL] energy balance: max|E - E(0) - W| / max|W| = {ratio:.2e} > {args.tolerance:.1e}")
    sys.exit(1)
print(f"[OK] energy balance: max|E - E(0) - W| / max|W| = {ratio:.2e}")
