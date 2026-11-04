"""
Check of the local potential at finite q in IPA (no laser), in linear response.

The input section local_potential gives V(r,t) = V0 f(t) cos(q0.r), f(t) = exp(-(t-t0)^2/(2 sigma^2)), approximated in
the Wannier basis as
    V = V0 f(t) sum_{nR} cos(q0.(R + tau_n)) c^dag_{nR} c_{nR}
      = V0 f(t) sum_{k,q,n} u_n(q) c^dag_{n,k+q} c_{n,k},   u_n(+-q0) = 1/2 exp(+-i s q0.tau_n)
where s = +1 for c_k = N^-1/2 sum_R e^{-ikR} c_R (wannier90 convention: H(k) = sum_R e^{ikR} H(R), the one of EDUS)
and s = -1 for the opposite one. tau_n are the Wannier centers (diagonal of r(R=0)).
To first order in V0, with rho_nm(k,q) = <c^dag_{m,k} c_{n,k+q}> and X(k,q) = -i V0 [ u(q) rho0(k) - rho0(k+q) u(q) ]
(the two terms are added if +q0 and -q0 are the same point of the q grid):
    rho(k,q,t) = int_0^t f(t') exp(-(i H(k+q) + 1/tau_decay)(t-t')) X(k,q) exp(i H(k)(t-t')) dt'.
The script compares n_m(q, t) = 1/N sum_k rho_mm(k, q, t) with DensityQ.txt for both conventions s = +1, -1.
The relative error is of order V0 (nonlinear terms): it must scale linearly with the amplitude.

Usage: python3 check_local_potential.py input.json Output [--tolerance 1e-3] [--spectrum chi.txt]
"""
import argparse
import json
import os
import sys
import numpy as np

AU_TIME_FS = 2.4188843265857e-2
TIME_UNITS = {"femtoseconds": 1. / AU_TIME_FS, "fs": 1. / AU_TIME_FS, "autime": 1.}
ANGSTROM_TO_BOHR = 1. / 0.529177210903
EV_TO_HARTREE = 1. / 27.211386245988


def read_tb(filename):
    """wannier90 _tb.dat: lattice (A), H(R) (eV), r(R) (A). Returns A (bohr, rows = a_i), Rs, H(R), r(R) in a.u."""
    lines = open(filename).read().split("\n")
    A = np.array([[float(x) for x in lines[i].split()] for i in (1, 2, 3)]) * ANGSTROM_TO_BOHR
    nw = int(lines[4].split()[0])
    nR = int(lines[5].split()[0])
    tokens = " ".join(lines[6:]).split()
    deg = np.array([int(x) for x in tokens[:nR]], dtype=float)
    pos = nR
    Rs = np.zeros((nR, 3), dtype=int)
    H = np.zeros((nR, nw, nw), dtype=complex)
    for iR in range(nR):
        Rs[iR] = [int(x) for x in tokens[pos:pos + 3]]
        pos += 3
        for _ in range(nw * nw):
            m, n = int(tokens[pos]) - 1, int(tokens[pos + 1]) - 1
            H[iR, m, n] = float(tokens[pos + 2]) + 1j * float(tokens[pos + 3])
            pos += 4
    r = np.zeros((nR, 3, nw, nw), dtype=complex)
    for iR in range(nR):
        assert [int(x) for x in tokens[pos:pos + 3]] == list(Rs[iR])
        pos += 3
        for _ in range(nw * nw):
            m, n = int(tokens[pos]) - 1, int(tokens[pos + 1]) - 1
            vals = [float(x) for x in tokens[pos + 2:pos + 8]]
            r[iR, :, m, n] = [vals[0] + 1j * vals[1], vals[2] + 1j * vals[3], vals[4] + 1j * vals[5]]
            pos += 8
    H *= EV_TO_HARTREE / deg[:, None, None]
    r *= ANGSTROM_TO_BOHR / deg[:, None, None, None]
    return A, Rs, H, r


def read_density_q(filename):
    qs = []
    for line in open(filename):
        if not line.startswith("#"):
            break
        if line.startswith("#   iq ="):
            qs.append([float(x) for x in line.split(":")[1].split()])
    data = np.loadtxt(filename, ndmin=2)
    t = data[:, 0]
    values = data[:, 1::2] + 1j * data[:, 2::2]
    nq = len(qs)
    nb = values.shape[1] // nq
    return np.array(qs), t, values.reshape(len(t), nq, nb)


def kgrid_index(kc, size):
    """index of the crystal point kc (folded in [0,1)) in the grid ordered as i0*n1*n2 + i1*n2 + i2"""
    i = np.rint(np.mod(kc, 1.) * size).astype(int) % size
    return (i[..., 0] * size[1] + i[..., 1]) * size[2] + i[..., 2]


parser = argparse.ArgumentParser(description="Check of the local potential at finite q (IPA, no laser, linear response)")
parser.add_argument("input")
parser.add_argument("output_dir")
parser.add_argument("--tolerance", type=float, default=1.e-3, help="maximum of |n - exact| / max|exact| over t, q, m")
parser.add_argument("--spectrum", default=None, help="writes omega (eV), chi_m(q0, omega) = FFT[n_m(q0,t)]/kappa")
args = parser.parse_args()

d = json.load(open(args.input))
if d.get("coulomb", False):
    print("[SKIP] local potential check: valid only in IPA")
    sys.exit(0)
ENERGY_UNITS = {"electronvolt": EV_TO_HARTREE, "auenergy": 1., "Rydberg": 0.5}
lp = d.get("local_potential", {})
if not lp.get("enabled", False) or float(lp.get("amplitude", 0.)) == 0.:
    print("[SKIP] local potential check: local_potential not enabled or amplitude = 0")
    sys.exit(0)
kappa = float(lp["amplitude"]) * ENERGY_UNITS[lp.get("amplitude_units", "electronvolt")]
q0 = np.array(lp.get("q", [0., 0., 0.]), dtype=float)
lp_time = TIME_UNITS[lp.get("time_units", "femtoseconds")]
tau_decay = float(d.get("decay", 0.)) * TIME_UNITS[d.get("decay_units", "femtoseconds").lower()]
size = np.array(d["grid"], dtype=int)
nfilled = int(d["filledbands"])
# tb_file is relative to the directory where EDUS runs (the current one, as in run.sh), or to the input
tb_file = d["tb_file"] + "_tb.dat"
if not os.path.isabs(tb_file) and not os.path.exists(tb_file):
    tb_file = os.path.join(os.path.dirname(os.path.abspath(args.input)), tb_file)

A, Rs, HR, rR = read_tb(tb_file)
B = 2. * np.pi * np.linalg.inv(A).T                       # rows = b_i
nb = HR.shape[1]
iR0 = np.where((Rs == 0).all(axis=1))[0][0]
tau = np.real(np.array([np.diag(rR[iR0, ix]) for ix in range(3)])).T   # tau[n] cartesian, bohr
q0_cart = q0 @ B

qs, t, nq_edus = read_density_q(f"{args.output_dir}/DensityQ.txt")
nq = len(qs)

# k grid, H(k) = sum_R e^{ikR} H(R), eigenvectors, rho0
grid = np.array(np.meshgrid(*[np.arange(n) / n for n in size], indexing="ij")).reshape(3, -1).T
nk = len(grid)
assert (kgrid_index(grid, size) == np.arange(nk)).all()
Hk = np.einsum("kr,rnm->knm", np.exp(2j * np.pi * grid @ Rs.T), HR)
e, U = np.linalg.eigh(Hk)
rho0 = np.einsum("kni,kmi->knm", U[:, :, :nfilled], U[:, :, :nfilled].conj())

from scipy.integrate import cumulative_simpson
sigma_au = float(lp.get("sigma", 1.)) * lp_time
t0_au = float(lp.get("t0", 0.)) * lp_time
if not np.allclose(np.diff(t), t[1] - t[0], rtol=1.e-6, atol=1.e-8):
    print("[FAIL] local potential check: the pulse integral needs equally spaced times (printresolution constant)")
    sys.exit(1)


def pulse_integral(Omega, times, refine=8):
    """G(t) = int_{t_first}^t f(t') e^{i Omega t'} dt' on a grid refine times finer (cumulative Simpson)"""
    tf = np.linspace(times[0], times[-1], refine * (len(times) - 1) + 1)
    f = np.exp(-(tf - t0_au) ** 2 / (2. * sigma_au ** 2))
    integrand = f[:, None, None, None] * np.exp(1j * Omega[None] * tf[:, None, None, None])
    G = cumulative_simpson(integrand, x=tf, axis=0, initial=0.)
    return G[::refine]


iq_of = lambda qc: int(np.argmin(np.abs(np.mod(qs - qc + 0.5, 1.) - 0.5).sum(axis=1)))

worst = {}
for s in (+1, -1):
    exact = np.zeros((len(t), nq, nb), dtype=complex)
    rho_q = np.zeros((nq, nk, nb, nb), dtype=complex)
    for sign in (+1, -1):
        qc = sign * q0
        iq = iq_of(qc)
        if np.abs(np.mod(qs[iq] - qc + 0.5, 1.) - 0.5).max() > 1.e-8:
            print(f"[FAIL] local potential check: q = {qc} is not a point of the q grid")
            sys.exit(1)
        u = 0.5 * kappa * np.exp(1j * s * sign * (tau @ q0_cart))
        ikq = kgrid_index(grid + qc, size)
        rho_q[iq] += -1j * (u[None, :, None] * rho0 - rho0[ikq] * u[None, None, :])
    for iq in range(nq):
        if not rho_q[iq].any():
            continue
        ikq = kgrid_index(grid + qs[iq], size)
        # band basis: X_ij(k) = <i,k+q| rho(k,q) |j,k>, Omega_ij = e_i(k+q) - e_j(k) - i/tau_decay
        X = np.einsum("kin,kij,kjm->knm", U[ikq].conj(), rho_q[iq], U)
        Omega = e[ikq][:, :, None] - e[:, None, :] - (1j / tau_decay if tau_decay > 0. else 0.)
        # X(t) = X e^{-i Omega t} G(t), G(t) = int_0^t f(t') e^{i Omega t'} dt'
        G = pulse_integral(Omega, t)
        for it, time in enumerate(t):
            Xt = X * np.exp(-1j * Omega * time) * G[it]
            rt = np.einsum("kin,knm,kjm->kij", U[ikq], Xt, U.conj())
            exact[it, iq] = np.einsum("kmm->m", rt) / nk
    # q = 0 is the equilibrium (not checked here: Population_wannier.txt)
    mask = np.array([np.abs(q).max() > 1.e-12 for q in qs])
    scale = np.abs(exact[:, mask]).max()
    err = np.abs(nq_edus[:, mask] - exact[:, mask]).max() / scale
    worst[s] = (err, exact, scale)
    print(f"convention s = {s:+d}: max |n - exact| / max|exact| = {err:.3e}   (max|exact| = {scale:.3e})")

s_best = min(worst, key=lambda s: worst[s][0])
err, exact, scale = worst[s_best]

if args.spectrum:
    iq = iq_of(q0)
    dt = t[1] - t[0]
    n_t = nq_edus[:, iq]
    window = np.exp(-(t / t[-1]) ** 2 * 4.)[:, None]
    omega = 2. * np.pi * np.fft.fftfreq(len(t), dt)
    chi = np.fft.fft(n_t * window, axis=0) * dt / kappa
    order = np.argsort(omega)
    np.savetxt(args.spectrum, np.column_stack([omega[order] / EV_TO_HARTREE,
                                                np.column_stack([chi[order].real, chi[order].imag])]),
               header="omega (eV), Re/Im chi_m(q0, omega) for m = 0..nb-1 (Re block, then Im block)")

if err < args.tolerance:
    print(f"[PASS] local potential check (convention s = {s_best:+d}): max relative error {err:.3e} < {args.tolerance:.1e}")
    sys.exit(0)
print(f"[FAIL] local potential check: best convention s = {s_best:+d}, max relative error {err:.3e} >= {args.tolerance:.1e}")
sys.exit(1)
