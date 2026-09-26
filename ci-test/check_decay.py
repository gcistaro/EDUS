"""
Analytic check of the decay term d(rho)/dt += -(rho - rho0)/tau.

After the lasers are off (E = 0), in IPA the deviation from equilibrium evolves as
    rho(t) - rho0 = exp(-(t - t1)/tau) exp(-i H0 (t - t1)) (rho(t1) - rho0) exp(i H0 (t - t1)),
so the energy absorbed by the electrons, E(t) - E(0) = Tr[H0 (rho - rho0)], decays exactly as exp(-(t - t1)/tau).
t1 is the first printed time after the end of the pulses; tau is read from the input.

Usage: python3 check_decay.py input.json Output [--tolerance 1e-4]
"""
import argparse
import json
import sys
import numpy as np

AU_TIME_FS = 2.4188843265857e-2
TIME_UNITS = {"femtoseconds": 1. / AU_TIME_FS, "fs": 1. / AU_TIME_FS, "autime": 1.}

parser = argparse.ArgumentParser(description="Analytic check of the decay term")
parser.add_argument("input")
parser.add_argument("output_dir")
parser.add_argument("--tolerance", type=float, default=1.e-4, help="maximum of |E - analytic| / (E(t1) - E(0))")
args = parser.parse_args()

d = json.load(open(args.input))
tau = float(d["decay"]) * TIME_UNITS[d.get("decay_units", "femtoseconds").lower()]
if d.get("coulomb", False):
    print("[SKIP] decay check: valid only in IPA (the mean field makes the relaxation nonlinear)")
    sys.exit(0)

t = np.loadtxt(f"{args.output_dir}/Time.txt").reshape(-1)
field = np.loadtxt(f"{args.output_dir}/Laser.txt").reshape(len(t), -1)
dE = np.loadtxt(f"{args.output_dir}/Energy.txt")[:, 3]
on = np.where(np.abs(field).max(axis=1) > 0.)[0]
k1 = (on[-1] + 1) if len(on) else 0
if k1 >= len(t) - 5 or abs(dE[k1]) < 1.e-300:
    print("[FAIL] decay check: the run must continue for some time after the end of the pulses")
    sys.exit(1)

analytic = dE[k1] * np.exp(-(t[k1:] - t[k1]) / tau)
error = np.abs(dE[k1:] - analytic).max() / abs(dE[k1])
if error > args.tolerance:
    print(f"[FAIL] decay check: max|E - E(t1) exp(-(t-t1)/tau)| / E(t1) = {error:.2e} > {args.tolerance:.1e}")
    sys.exit(1)
print(f"[OK] decay check: max|E - E(t1) exp(-(t-t1)/tau)| / E(t1) = {error:.2e}")
