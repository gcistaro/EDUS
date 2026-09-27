#!/usr/bin/env python3
"""Synthetic test for EDUS_wfc2wannier.

We choose Wannier functions analytically (normalized Gaussians w_n(r) = (pi s^2)^(-3/4) exp(-|r-tau_n|^2/(2 s^2)),
optionally times a spinor) and build Bloch functions in the QE convention
    psi_k(r) = sum_R e^{ik.R} w(r-R) = 1/sqrt(Omega) sum_G c(G) e^{i(k+G).r},  c(G) = w^(k+G)/sqrt(Omega).
The "Kohn-Sham" states written in wfc<ik>.dat are random unitary mixtures of them (plus excluded bands and
bands outside the disentanglement window), and <seed>_u.mat/<seed>_u_dis.mat/<seed>.eig contain the matrices
that bring them back to the Gaussians. EDUS_wfc2wannier must then recover exactly the analytic Bloch sums
and Wannier functions with the right norm, centre and spread (3 s^2/2).

Cases:
  plain  : 3x3x3 k grid, 2 WFs, 2 bands, some k points in another periodic image, k files in shuffled order
  dis    : 3x3x3 k grid, 6 QE bands, band 1 excluded, outer window moving from k to k, LSDA file names (wfcup)
  gamma  : Gamma only, gamma_only storage (half G sphere), real orthogonal mixing
  spinor : 2x2x2 k grid, noncollinear (npol = 2)

Usage: generate.py <directory>
"""
import json
import os
import struct
import sys

import numpy as np

BOHR_ANG = 0.52917721090380  # same CODATA value used by EDUS (Constants.hpp)
A0 = 10.26
LAT = A0 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])  # rows a_i, bohr
REC = 2 * np.pi * np.linalg.inv(LAT).T  # rows b_i, bohr^-1
OMEGA = abs(np.linalg.det(LAT))
SIGMA = 1.2  # default width (bohr); smaller for the cases where the Gaussians must fit in a small supercell


def rec(f, data):
    f.write(struct.pack("<i", len(data)))
    f.write(data)
    f.write(struct.pack("<i", len(data)))


def kgrid(n):
    return np.array([[i / n[0], j / n[1], k / n[2]] for i in range(n[0]) for j in range(n[1]) for k in range(n[2])])


def miller_sphere(kc, gamma_only=False, sigma=SIGMA):
    """Miller indices with |k+G| < QMAX (k in crystal coordinates), exp(-qmax^2 s^2/2) ~ 1e-14"""
    QMAX = 8.0 / sigma
    nmax = [int(QMAX * np.linalg.norm(LAT[i]) / (2 * np.pi)) + 2 for i in range(3)]
    r = [np.arange(-n, n + 1) for n in nmax]
    M = np.array(np.meshgrid(*r, indexing="ij")).reshape(3, -1).T
    q = (kc + M) @ REC
    M = M[np.linalg.norm(q, axis=1) < QMAX]
    if gamma_only:
        keep = (M[:, 0] > 0) | ((M[:, 0] == 0) & (M[:, 1] > 0)) | ((M[:, 0] == 0) & (M[:, 1] == 0) & (M[:, 2] >= 0))
        M = M[keep]
    rng = np.random.default_rng(len(M))
    return M[rng.permutation(len(M))]  # QE order is not the one of numpy


def gauss_coef(kc, M, tau_frac, sigma=SIGMA):
    q = (kc + M) @ REC
    tau = np.asarray(tau_frac) @ LAT
    return (4 * np.pi * sigma**2) ** 0.75 * np.exp(-0.5 * sigma**2 * np.sum(q * q, axis=1) - 1j * q @ tau) / np.sqrt(OMEGA)


def random_unitary(n, rng, real=False):
    x = rng.normal(size=(n, n)) + (0 if real else 1j * rng.normal(size=(n, n)))
    q, r = np.linalg.qr(x)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def write_wfc(fname, ik, kc_qe, M, evc, npol=1, gamma_only=False, ispin=1):
    """evc: (nbnd, npol*len(M))"""
    xk = kc_qe @ REC
    with open(fname, "wb") as f:
        rec(f, struct.pack("<i3diid", ik, *xk, ispin, int(gamma_only), 1.0))
        rec(f, struct.pack("<4i", len(M), len(M), npol, evc.shape[0]))
        rec(f, REC.astype("<f8").tobytes())
        rec(f, np.ascontiguousarray(M).astype("<i4").tobytes())
        for b in range(evc.shape[0]):
            rec(f, np.ascontiguousarray(evc[b]).astype("<c16").tobytes())


def write_umat(fname, kpts, U):
    """U: (nk, nrows, nwann)"""
    nk, nr, nw = U.shape
    with open(fname, "w") as f:
        f.write(" written on 26Sep2026 at 12:00:00\n")
        f.write(f"{nk:12d}{nw:12d}{nr:12d}\n")
        for k in range(nk):
            f.write("\n")
            f.write("%15.10f%+15.10f%+15.10f\n" % tuple(kpts[k]))
            for j in range(nw):
                for i in range(nr):
                    f.write("%15.10f%+15.10f\n" % (U[k, i, j].real, U[k, i, j].imag))


def write_xml(save, taus):
    with open(f"{save}/data-file-schema.xml", "w") as f:
        f.write("<qes:espresso>\n<input><atomic_structure><atomic_positions>\n")
        f.write('<atom name="X" index="1">9 9 9</atom>\n')  # input block must be ignored
        f.write("</atomic_positions></atomic_structure></input>\n<output><atomic_structure><atomic_positions>\n")
        for t in taus:
            r = np.asarray(t) @ LAT
            f.write(f'<atom name="Si1" index="1">{r[0]:.10f} {r[1]:.10f} {r[2]:.10f}</atom>\n')
        f.write("</atomic_positions></atomic_structure></output>\n</qes:espresso>\n")


def reference(case_dir, kpts, taus, spinors=None, sigma=SIGMA):
    """analytic Bloch sums (for the check) and expected centres"""
    ref = {"centres_ang": [(np.asarray(t) @ LAT * BOHR_ANG).tolist() for t in taus],
           "spread_ang2": 1.5 * sigma**2 * BOHR_ANG**2, "sigma": sigma, "lattice": LAT.tolist(),
           "taus": [list(t) for t in taus], "kpts": kpts.tolist(),
           "spinors": None if spinors is None else [[[s.real, s.imag] for s in sp] for sp in spinors]}
    json.dump(ref, open(f"{case_dir}/reference.json", "w"), indent=1)


def case_plain(d):
    rng = np.random.default_rng(1)
    taus = [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]
    kpts = kgrid([3, 3, 3])
    save = f"{d}/si.save"
    os.makedirs(save, exist_ok=True)
    order = rng.permutation(len(kpts))  # QE file index of each k point
    U = np.zeros((len(kpts), 2, 2), complex)
    for ik, kc in enumerate(kpts):
        G0 = np.array([1, -1, 0]) if ik % 4 == 1 else np.zeros(3, int)  # k_QE = k + G0
        M = miller_sphere(kc + G0)
        phi = np.array([gauss_coef(kc + G0, M, t) for t in taus])
        A = random_unitary(2, rng)
        psi = A.T @ phi  # psi_m = sum_j phi_j A_jm
        U[ik] = A.conj().T
        write_wfc(f"{save}/wfc{order[ik] + 1}.dat", order[ik] + 1, kc + G0, M, psi)
    write_umat(f"{d}/si_u.mat", kpts, U)
    write_xml(save, taus)
    reference(d, kpts, taus)
    json.dump({"qe_save_dir": "si.save", "seedname": "si", "supercell": [3, 3, 3], "output_dir": "out"},
              open(f"{d}/input.json", "w"), indent=1)


def case_dis(d):
    rng = np.random.default_rng(2)
    taus = [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]
    other = [(0.1, 0.2, 0.3), (0.6, 0.1, 0.4)]           # also inside the window
    excluded = (0.5, 0.5, 0.5)                           # QE band 1, excluded
    outside = (0.3, 0.7, 0.2)                            # w90 band outside the outer window
    kpts = kgrid([3, 3, 3])
    save = f"{d}/si.save"
    os.makedirs(save, exist_ok=True)
    nk = len(kpts)
    U, Udis = np.zeros((nk, 2, 2), complex), np.zeros((nk, 5, 2), complex)
    eig = np.zeros((nk, 5))
    for ik, kc in enumerate(kpts):
        M = miller_sphere(kc)
        Phi = np.array([gauss_coef(kc, M, t) for t in taus + other])  # 4 states in the window
        W = random_unitary(4, rng)
        Q = random_unitary(2, rng)
        s = ik % 2  # first band of the window (w90 numbering)
        win = W.T @ Phi
        out = gauss_coef(kc, M, outside)
        w90 = np.zeros((5, len(M)), complex)
        w90[s:s + 4] = win
        w90[4 if s == 0 else 0] = out
        e_win = np.array([-3.0, -1.0, 1.0, 3.0]) + 0.3 * np.sin(2 * np.pi * kc[0])
        eig[ik, s:s + 4] = e_win
        eig[ik, 4 if s == 0 else 0] = 10.0 if s == 0 else -8.0
        qe = np.vstack([gauss_coef(kc, M, excluded)[None, :], w90])
        write_wfc(f"{save}/wfcup{ik + 1}.dat", ik + 1, kc, M, qe)
        Udis[ik, :4, :] = W.conj().T[:, :2] @ Q
        U[ik] = Q.conj().T
    write_umat(f"{d}/si_u.mat", kpts, U)
    write_umat(f"{d}/si_u_dis.mat", kpts, Udis)
    with open(f"{d}/si.eig", "w") as f:
        for ik in range(nk):
            for b in range(5):
                f.write(f"{b + 1:5d}{ik + 1:5d}{eig[ik, b]:18.12f}\n")
    write_xml(save, taus)
    reference(d, kpts, taus)
    json.dump({"qe_save_dir": "si.save", "seedname": "si", "exclude_bands": [1], "dis_win_min": -5.0,
               "dis_win_max": 5.0, "spin": "up", "supercell": [3, 3, 3], "output_dir": "out"},
              open(f"{d}/input.json", "w"), indent=1)


def case_gamma(d):
    rng = np.random.default_rng(3)
    taus = [(0.45, 0.45, 0.45), (0.55, 0.6, 0.5)]
    kpts = kgrid([1, 1, 1])
    save = f"{d}/si.save"
    os.makedirs(save, exist_ok=True)
    sigma = 0.6  # the Gaussians must be well inside a single (flat) FCC cell
    M = miller_sphere(kpts[0], gamma_only=True, sigma=sigma)
    phi = np.array([gauss_coef(kpts[0], M, t, sigma) for t in taus])
    A = random_unitary(2, rng, real=True)
    write_wfc(f"{save}/wfc1.dat", 1, kpts[0], M, A.T @ phi, gamma_only=True)
    write_umat(f"{d}/si_u.mat", kpts, A.conj().T[None, :, :])
    reference(d, kpts, taus, sigma=sigma)
    json.dump({"qe_save_dir": "si.save", "seedname": "si", "supercell": [1, 1, 1], "output_dir": "out"},
              open(f"{d}/input.json", "w"), indent=1)


def case_spinor(d):
    rng = np.random.default_rng(4)
    taus = [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]
    th, ph = 0.7, 1.9
    chi = [np.array([np.cos(th), np.sin(th) * np.exp(1j * ph)]),
           np.array([-np.sin(th) * np.exp(-1j * ph), np.cos(th)])]
    kpts = kgrid([2, 2, 2])
    sigma = 0.8  # the supercell is only 2x2x2
    save = f"{d}/si.save"
    os.makedirs(save, exist_ok=True)
    U = np.zeros((len(kpts), 2, 2), complex)
    for ik, kc in enumerate(kpts):
        M = miller_sphere(kc, sigma=sigma)
        g = [gauss_coef(kc, M, t, sigma) for t in taus]
        phi = np.array([np.concatenate([chi[n][0] * g[n], chi[n][1] * g[n]])
                        for n in range(2)])
        A = random_unitary(2, rng)
        U[ik] = A.conj().T
        write_wfc(f"{save}/wfc{ik + 1}.dat", ik + 1, kc, M, A.T @ phi, npol=2)
    write_umat(f"{d}/si_u.mat", kpts, U)
    write_xml(save, taus)
    reference(d, kpts, taus, spinors=chi, sigma=sigma)
    json.dump({"qe_save_dir": "si.save", "seedname": "si", "supercell": [2, 2, 2], "output_dir": "out"},
              open(f"{d}/input.json", "w"), indent=1)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for name, fn in [("plain", case_plain), ("dis", case_dis), ("gamma", case_gamma), ("spinor", case_spinor)]:
        os.makedirs(f"{root}/{name}", exist_ok=True)
        fn(f"{root}/{name}")
        print("generated", name)
