"""
Analytic checks of the Ehrenfest dynamics of the lattice (Output/Lattice.txt and Output/Energy.txt).

1. Without work of the field the total energy (electrons + lattice + coupling) is conserved:
   max|E - E(0)| must be small compared with the energy of the lattice (both per spin channel in Energy.txt).
   With the screened coupling (epw_directory_screened) and the mean field it is conserved only if
   g_b = g_s - Sigma[chi0 g_s], which synthetic data do not satisfy: then the drift is only printed.
2. Only the excited electrons, rho - rho_BO, push the atoms, and the force constants are those of ph.x: small oscillations started from "initial_displacement" must have a frequency of the
   dynamical matrix of ph.x (static limit; with "static" the non-adiabatic correction is of order (omega/gap)^2,
   with "dynamic" and no laser rho = rho_BO and the frequency is exact).

Usage: python3 check_phonons.py input.json Output [--energy-tolerance 1e-4] [--frequency-tolerance 1e-3]
"""
import argparse
import json
import os
import sys
import numpy as np

HA_CM = 219474.6313705


def dynamical_matrix_frequencies(filename):
    """Frequencies (cm^-1) at q = 0 from a text dynamical matrix of ph.x (Ry units), with the acoustic sum rule"""
    with open(filename) as f:
        lines = f.readlines()
    ntyp, nat, ibrav = [int(x) for x in lines[2].split()[:3]]
    i = 3 + (4 if ibrav == 0 else 0)
    type_mass = [float(lines[i + it].split("'")[-1]) for it in range(ntyp)]
    i += ntyp
    mass = [2. * type_mass[int(lines[i + a].split()[1]) - 1] for a in range(nat)]
    i += nat
    while "q = (" not in lines[i]:
        i += 1
    i += 1
    C = np.zeros((3 * nat, 3 * nat), dtype=complex)
    for _ in range(nat * nat):
        while not lines[i].strip():
            i += 1
        a, b = [int(x) - 1 for x in lines[i].split()[:2]]
        for r in range(3):
            v = [float(x) for x in lines[i + 1 + r].split()]
            for c in range(3):
                C[3 * a + r, 3 * b + c] = 0.5 * (v[2 * c] + 1j * v[2 * c + 1])
        i += 4
    for a in range(nat):
        for r in range(3):
            for c in range(3):
                C[3 * a + r, 3 * a + c] -= sum(C[3 * a + r, 3 * b + c] for b in range(nat))
    m = np.repeat(mass, 3)
    w2 = np.linalg.eigvalsh(0.5 * (C + C.conj().T) / np.sqrt(np.outer(m, m)))
    return np.sign(w2) * np.sqrt(np.abs(w2)) * HA_CM


def fitted_frequency(t, x, guess):
    """Frequency (a.u.) of x(t) = a cos(w t) + b sin(w t) + c, by least squares around guess"""
    def residual(w):
        A = np.vstack([np.cos(w * t), np.sin(w * t), np.ones_like(t)]).T
        coef = np.linalg.lstsq(A, x, rcond=None)[0]
        return np.sum((A @ coef - x) ** 2)
    ws = np.linspace(0.5 * guess, 1.5 * guess, 2001)
    w = ws[np.argmin([residual(w) for w in ws])]
    for width in (ws[1] - ws[0], 1.e-3 * w, 1.e-5 * w):
        ws = np.linspace(w - width, w + width, 201)
        w = ws[np.argmin([residual(w) for w in ws])]
    return w


parser = argparse.ArgumentParser(description="Analytic checks of the lattice dynamics of an EDUS run")
parser.add_argument("input")
parser.add_argument("output_dir")
parser.add_argument("--energy-tolerance", type=float, default=1.e-4)
parser.add_argument("--frequency-tolerance", type=float, default=1.e-3)
args = parser.parse_args()

with open(args.input) as f:
    inp = json.load(f)
phonons = inp.get("phonons", {})
lattice = np.loadtxt(os.path.join(args.output_dir, "Lattice.txt"), ndmin=2)
energy = np.loadtxt(os.path.join(args.output_dir, "Energy.txt"), ndmin=2)
failures = 0

t = lattice[:, 0]
nmodes = (lattice.shape[1] - 3) // 3
u = lattice[:, 3:3 + nmodes]
spin = phonons.get("spin_degeneracy", 2.)

# 1. conservation of the total energy without work of the field
if np.abs(energy[:, 5]).max() < 1.e-300:
    scale = np.abs(lattice[:, 1]).max() / spin
    drift = np.abs(energy[:, 3]).max()
    if scale < 1.e-300:
        print(f"[OK] lattice energy: no work and lattice at rest, max|E - E(0)| = {drift:.2e}")
    elif phonons.get("epw_directory_screened") and inp.get("coulomb", False):
        print(f"[INFO] lattice energy (screened coupling with mean field, not conserved): "
              f"max|E - E(0)| / max(E_lattice) = {drift / scale:.2e}")
    elif drift / scale > args.energy_tolerance:
        print(f"[FAIL] lattice energy: max|E - E(0)| / max(E_lattice) = {drift / scale:.2e} > {args.energy_tolerance:.1e}")
        failures += 1
    else:
        print(f"[OK] lattice energy: max|E - E(0)| / max(E_lattice) = {drift / scale:.2e}")

# 2. static limit
if phonons.get("initial_displacement"):
    dyn = phonons["dyn_file"]
    if not os.path.isabs(dyn):
        dyn = os.path.join(os.getcwd(), dyn)
    omega = dynamical_matrix_frequencies(dyn)
    # the displacement along the initial one, minus the motion of the center of mass (constant velocity 0)
    u0 = np.array(phonons["initial_displacement"], dtype=float)
    x = (u - u[0]) @ u0 / np.dot(u0, u0) + 1.
    optical = omega[omega > 1.]
    w = fitted_frequency(t, x, optical.mean() / HA_CM) * HA_CM
    expected = optical[np.argmin(np.abs(optical - w))]
    error = abs(w - expected) / expected
    if error > args.frequency_tolerance:
        print(f"[FAIL] static limit: frequency {w:.3f} cm^-1, dynamical matrix {expected:.3f} cm^-1 (error {error:.1e})")
        failures += 1
    else:
        print(f"[OK] static limit: frequency {w:.3f} cm^-1, dynamical matrix {expected:.3f} cm^-1 (error {error:.1e})")

# 3. projection on the normal modes of ph.x: the dynamics uses the force constants of ph.x, so the energies of the
#    modes add up to E_lattice
modes_file = os.path.join(args.output_dir, "Phonon_modes.txt")
if os.path.exists(modes_file):
    modes = np.loadtxt(modes_file, ndmin=2)
    energy_modes = modes[:, 1 + 3 * nmodes:1 + 4 * nmodes].sum(axis=1)
    scale = max(np.abs(lattice[:, 1]).max(), 1.e-300)
    error = np.abs(energy_modes - lattice[:, 1]).max() / scale
    if error > 1.e-8:
        print(f"[FAIL] normal modes: max|sum E_lambda - E_lattice| / max(E_lattice) = {error:.2e}")
        failures += 1
    else:
        print(f"[OK] normal modes: max|sum E_lambda - E_lattice| / max(E_lattice) = {error:.2e}")

sys.exit(1 if failures else 0)
