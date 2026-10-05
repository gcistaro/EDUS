"""
Symmetry checks of the coherent phonons of a 2D crystal with the horizontal mirror sigma_h (z -> -z), e.g. monolayer
MoS2 (D3h), from Output/Phonon_modes.txt and the positions of the EPW files (crystal.fmt in "epw_directory").

The modes of ph.x are classified by their parity under sigma_h and by their degeneracy:
- odd modes (E'' and A2'' in MoS2): a field in the plane cannot excite them, |Q| must stay at the numerical noise;
- even nondegenerate optical modes (A1'): totally symmetric, pushed by any population of excited carriers (DECP);
- even degenerate optical modes (E'): pushed only by an anisotropic distribution of carriers, i.e. a linear
  polarization; with a circular one the excited distribution keeps the C3 symmetry and E' gets no force.

Usage: python3 check_symmetry.py input.json Output [--dominant] [--circular] [--odd-tolerance 1e-6]
                                                    [--circular-tolerance 1e-3]
  --dominant   at the end of the run the energy of the A1'-like modes is the largest of all the modes
  --circular   at the end of the run E(E'-like) / E(A1'-like) < circular-tolerance
"""
import argparse
import json
import os
import sys
import numpy as np


def positions(epw_directory):
    """Number of atoms and positions (alat units) from crystal.fmt of EPW"""
    with open(os.path.join(epw_directory, "crystal.fmt")) as f:
        tokens = f.read().split()
    nat = int(tokens[0])
    # nat, nmodes, nelec, nbndskip, at (9), bg (9), omega, alat, tau (3 nat)
    start = 2 + 2 + 9 + 9 + 2
    return np.array([float(x) for x in tokens[start:start + 3 * nat]]).reshape(nat, 3)


def mirror_permutation(tau, tol=1.e-5):
    """Atom b = image of atom a under z -> -z (the same atom up to a lattice vector in the plane), None if absent"""
    perm = []
    for t in tau:
        image = t * np.array([1., 1., -1.])
        found = [b for b, s in enumerate(tau) if abs(s[2] - image[2]) < tol]
        # in the plane the image coincides with the atom itself: pick the one at the same in-plane position
        same = [b for b in found if np.linalg.norm(tau[b][:2] - t[:2]) < tol]
        if not same:
            return None
        perm.append(same[0])
    return perm


def modes_header(filename):
    """Frequencies (cm^-1) and eigenvectors e(mu, lambda) from the header of Phonon_modes.txt"""
    freqs, vectors = [], []
    with open(filename) as f:
        for line in f:
            if not line.startswith("#"):
                break
            fields = line[1:].split()
            if len(fields) > 2 and fields[0].isdigit():
                freqs.append(float(fields[1]))
                vectors.append([float(x) for x in fields[2:]])
    return np.array(freqs), np.array(vectors).T


parser = argparse.ArgumentParser(description="sigma_h and C3 checks of the coherent phonons of a 2D crystal")
parser.add_argument("input")
parser.add_argument("output_dir")
parser.add_argument("--dominant", action="store_true")
parser.add_argument("--circular", action="store_true")
parser.add_argument("--odd-tolerance", type=float, default=1.e-6)
parser.add_argument("--circular-tolerance", type=float, default=1.e-3)
args = parser.parse_args()

with open(args.input) as f:
    phonons = json.load(f)["phonons"]
epw_directory = phonons["epw_directory"]
if not os.path.isabs(epw_directory):
    epw_directory = os.path.join(os.getcwd(), epw_directory)

tau = positions(epw_directory)
perm = mirror_permutation(tau)
if perm is None:
    print("[FAIL] symmetry: the crystal has no horizontal mirror z -> -z")
    sys.exit(1)

modes_file = os.path.join(args.output_dir, "Phonon_modes.txt")
freqs, e = modes_header(modes_file)
nmodes = len(freqs)
# sigma_h on the cartesian displacements: u'(b) = diag(1, 1, -1) u(a) with b = perm[a]
S = np.zeros((nmodes, nmodes))
for a, b in enumerate(perm):
    for x, sign in enumerate((1., 1., -1.)):
        S[3 * b + x, 3 * a + x] = sign
parity = np.einsum("ml,mn,nl->l", e, S, e)
optical = freqs > 1.
degenerate = np.array([np.sum(np.abs(freqs - w) < 1.e-3 * max(abs(w), 1.)) > 1 for w in freqs])
odd = parity < -0.99
even_single = optical & (parity > 0.99) & ~degenerate
even_degenerate = optical & (parity > 0.99) & degenerate
if np.any(np.abs(np.abs(parity) - 1.) > 1.e-3):
    print(f"[FAIL] symmetry: modes without a definite parity under sigma_h: {parity}")
    sys.exit(1)

data = np.loadtxt(modes_file, ndmin=2)
Q = data[:, 1:1 + nmodes]
E = data[:, 1 + 3 * nmodes:1 + 4 * nmodes]
failures = 0

scale = max(np.abs(Q).max(), 1.e-300)
odd_max = np.abs(Q[:, odd]).max() / scale if odd.any() else 0.
label = ", ".join(f"{w:.1f}" for w in freqs[odd])
if odd_max > args.odd_tolerance:
    print(f"[FAIL] sigma_h: odd modes ({label} cm^-1) max|Q| / max|Q| = {odd_max:.2e} > {args.odd_tolerance:.1e}")
    failures += 1
else:
    print(f"[OK] sigma_h: odd modes ({label} cm^-1) max|Q| / max|Q| = {odd_max:.2e}")

E_single = E[-1, even_single].sum()
E_degenerate = E[-1, even_degenerate].sum()
if args.dominant:
    largest = freqs[np.argmax(E[-1])]
    if not even_single.any() or not even_single[np.argmax(E[-1])]:
        print(f"[FAIL] dominant mode: the largest energy is in the mode at {largest:.1f} cm^-1, "
              f"not in a totally symmetric one")
        failures += 1
    else:
        print(f"[OK] dominant mode: {largest:.1f} cm^-1 (totally symmetric), "
              f"E(degenerate even) / E(totally symmetric) = {E_degenerate / E_single:.2e}")
if args.circular:
    ratio = E_degenerate / max(E_single, 1.e-300)
    if ratio > args.circular_tolerance:
        print(f"[FAIL] circular pump: E(degenerate even) / E(totally symmetric) = {ratio:.2e} > "
              f"{args.circular_tolerance:.1e}")
        failures += 1
    else:
        print(f"[OK] circular pump: E(degenerate even) / E(totally symmetric) = {ratio:.2e}")

sys.exit(1 if failures else 0)
