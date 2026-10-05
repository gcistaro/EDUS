"""Validation of the Ehrenfest phonons on the real hBN (QE -> EPW -> EDUS): the same tests done on the synthetic model
in phonons_implementation.tex, section Validation.

    python3 epw_validation.py run      <simulation dir> [filter]  # writes the inputs in <dir>/validation/<run> and runs
                                                                # EDUS (only the runs whose name contains filter)
    python3 epw_validation.py analyze  <simulation dir>   # reads the outputs and writes data_epw.json (for the figures)

<simulation dir> contains epw/ (EPW run with the Wannier model hbn_tb.dat) and phonon/hbn.dyn1 (ph.x at Gamma).
The executable is $EDUS (default: build/EDUS of the repository), MPI with $MPIEXEC (default mpirun).
"""
import concurrent.futures
import copy
import json
import os
import re
import subprocess
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EDUS = os.environ.get("EDUS", os.path.join(HERE, "..", "..", "build", "EDUS"))
MPIEXEC = os.environ.get("MPIEXEC", "mpirun")
HA_CM = 219474.6313705
AU_FS = 0.02418884326585747
# relative displacement of B and N along x with the center of mass at rest (masses of hbn.dyn1): E' mode
DISPLACEMENT = [0.01, 0., 0., -0.007718449027965188, 0., 0.]


def base_input(sim):
    return {
        "tb_file": os.path.join(sim, "epw", "hbn"),
        "grid": [12, 12, 1],
        "filledbands": 1,
        "dt": 0.01, "dt_units": "femtoseconds",
        "printresolution": 10, "printresolution_pulse": 10,
        "initialtime": 0, "initialtime_units": "femtoseconds",
        "finaltime": 100, "finaltime_units": "femtoseconds",
        "solver": "RK", "order": 4,
        "lasers": [{"intensity": 0.0, "intensity_units": "wcm2", "frequency": 5.5, "frequency_units": "electronvolt",
                    "polarization": [1, 0, 0], "cycles": 4, "t0": 0.0, "t0_units": "fs"}],
        "phonons": {
            "enabled": True,
            "epw_directory": os.path.join(sim, "epw"),
            "dyn_file": os.path.join(sim, "phonon", "hbn.dyn1"),
            "coupling": "screened",
            "adiabatic_reference": "static",
            "initial_displacement": DISPLACEMENT,
            "initial_displacement_units": "angstrom",
        },
    }


def runs(sim):
    """name -> (input, number of MPI ranks)"""
    result = {}

    def add(name, np_=1, laser=False, hsex=False, dt=None, solver=None, **phonons):
        inp = copy.deepcopy(base_input(sim))
        if laser:
            # lattice at rest: the laser alone drives it (displacive excitation)
            inp["lasers"][0]["intensity"] = 1.e11
            inp["finaltime"] = 60
            inp["phonons"]["initial_displacement"] = []
        if hsex:
            inp.update({"coulomb": True, "method": "hsex", "epsilon": 2.0, "r0": [10]})
        if dt is not None:
            inp["dt"] = dt
            # print every 0.1 fs whatever the time step
            inp["printresolution"] = inp["printresolution_pulse"] = int(round(0.1 / dt))
        if solver is not None:
            inp["solver"] = solver
        for key, value in phonons.items():
            if key == "peierls":
                inp["peierls"] = value
            elif key == "enabled" and not value:
                inp["phonons"] = {"enabled": False}
            else:
                inp["phonons"][key] = value
        result[name] = (inp, np_)

    # IPA, no laser: small oscillations of the E' mode
    add("ipa_off", coupling_scale=0., adiabatic_reference="none")
    add("ipa_rest", adiabatic_reference="none", initial_displacement=[])
    for reference in ("none", "static", "dynamic"):
        add("ipa_" + reference, adiabatic_reference=reference)
    # amplitude dependence of the static reference: non-adiabatic correction (harmonic limit) + non-linear response
    for scale in (0.5, 0.25):
        add(f"ipa_static_u{scale}", initial_displacement=[scale * x for x in DISPLACEMENT])
    # IPA, laser above the gap (4.67 eV): energy balance, MPI, gauge, time step
    add("ipa_laser", laser=True)
    add("ipa_laser_np2", np_=2, laser=True)
    add("ipa_laser_np4", np_=4, laser=True)
    add("ipa_laser_nophonons", laser=True, enabled=False)
    add("ipa_laser_gradient", laser=True, peierls=False)
    add("ipa_laser_nophonons_gradient", laser=True, enabled=False, peierls=False)
    for dt in (0.05, 0.025, 0.0125):
        add(f"ipa_laser_rk_{dt}", laser=True, dt=dt)
    add("ipa_laser_ab_0.01", laser=True, dt=0.01, solver="AB")
    add("ipa_laser_ab_0.005", laser=True, dt=0.005, solver="AB")
    # HSEX (epsilon = 2: with epsilon = 1 the TDHF of the DFT bands is excitonically unstable)
    for reference in ("none", "static", "dynamic"):
        add("hsex_" + reference, hsex=True, adiabatic_reference=reference)
    for scale in (0.5, 0.25):
        add(f"hsex_static_u{scale}", hsex=True, initial_displacement=[scale * x for x in DISPLACEMENT])
    add("hsex_static_dt2", hsex=True, dt=0.005)
    add("hsex_none_dt2", hsex=True, dt=0.005, adiabatic_reference="none")
    for reference in ("none", "static", "dynamic"):
        add("hsex_laser_" + reference, hsex=True, laser=True, adiabatic_reference=reference)
    for reference in ("static", "dynamic"):
        for np_ in (2, 4):
            add(f"hsex_laser_{reference}_np{np_}", np_=np_, hsex=True, laser=True, adiabatic_reference=reference)
    add("hsex_laser_nophonons", hsex=True, laser=True, enabled=False)
    return result


def run_one(directory, inp, np_):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "input.json"), "w") as f:
        json.dump(inp, f, indent=2)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(os.path.join(directory, "edus.out"), "w") as out:
        code = subprocess.call([MPIEXEC, "-np", str(np_), EDUS, "input.json"], cwd=directory, stdout=out,
                               stderr=subprocess.STDOUT, env=env)
    return directory, code


def run(sim, only=""):
    todo = {name: value for name, value in runs(sim).items() if only in name}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(run_one, os.path.join(sim, "validation", name), inp, np_)
                   for name, (inp, np_) in todo.items()]
        for future in concurrent.futures.as_completed(futures):
            directory, code = future.result()
            print(("[OK]  " if code == 0 else "[FAIL]") + " " + os.path.basename(directory), flush=True)


# ---------------------------------------------------------------------------------------------------------- analysis
def fitted_frequency(t, x, guess):
    """Frequency (a.u.) of x(t) = a cos(w t) + b sin(w t) + c, by least squares around guess (as check_phonons.py)"""
    def residual(w):
        A = np.vstack([np.cos(w * t), np.sin(w * t), np.ones_like(t)]).T
        coef = np.linalg.lstsq(A, x, rcond=None)[0]
        return np.sum((A @ coef - x) ** 2)
    ws = np.linspace(0.3 * guess, 1.5 * guess, 3001)
    w = ws[np.argmin([residual(w) for w in ws])]
    for width in (ws[1] - ws[0], 1.e-3 * w, 1.e-5 * w):
        ws = np.linspace(w - width, w + width, 201)
        w = ws[np.argmin([residual(w) for w in ws])]
    return w


def cumulative_work(t, P):
    """int P dt with the 3-point rule of ci-test/check_energy.py"""
    W = np.zeros_like(P)
    dt = np.diff(t)
    W[1] = dt[0] / 12. * (5. * P[0] + 8. * P[1] - P[2])
    W[2:] = W[1] + np.cumsum(dt[1:] / 12. * (-P[:-2] + 8. * P[1:-1] + 5. * P[2:]))
    return W


def recap(directory):
    """numbers of the PHONONS recap in edus.out"""
    text = open(os.path.join(directory, "edus.out")).read()
    values = {}
    for key, label in [("unscreening", r"Unscreening: correction"), ("asr", r"ASR correction"),
                       ("translation", r"Translation in g removed"), ("pi_asr", r"ASR of Pi: max\|sum\|"),
                       ("gauge", r"max\|H_EPW - H_tb\|")]:
        m = re.search(label + r"\s*\*\s*([-+0-9.eE]+)", text)
        if m:
            values[key] = float(m.group(1))
    m = re.search(r"Frequencies at Gamma.*?\n.*?\n((?:\*\s+\d+\s+\S+\s+\S+\s+\*\n)+)", text)
    if m:
        rows = [line.split()[1:4] for line in m.group(1).strip().split("\n")]
        values["frequencies_dynamics"] = [float(r[1]) for r in rows]
        values["frequencies_phx"] = [float(r[2]) for r in rows]
    return values


def load(directory):
    lat_file = os.path.join(directory, "Output", "Lattice.txt")
    r = {"energy": np.loadtxt(os.path.join(directory, "Output", "Energy.txt"), ndmin=2),
         "population": np.loadtxt(os.path.join(directory, "Output", "Population.txt"), ndmin=2)}
    if os.path.exists(lat_file):
        r["lattice"] = np.loadtxt(lat_file, ndmin=2)
    r.update(recap(directory))
    return r


def analyze(sim):
    base = os.path.join(sim, "validation")
    R = {name: load(os.path.join(base, name)) for name in runs(sim) if os.path.exists(os.path.join(base, name, "Output"))}
    data = {"scalars": {}, "series": {}}
    S = data["scalars"]
    w_phx = R["ipa_off"]["frequencies_phx"][4]
    S["w_phx"] = w_phx

    def rel(r):
        u = r["lattice"][:, 3:9]
        return u[:, 0] - u[:, 3]

    def time_fs(r):
        return r["lattice"][:, 0] * AU_FS

    def frequency(r):
        t = r["lattice"][:, 0]
        return fitted_frequency(t, rel(r), w_phx / HA_CM) * HA_CM

    def drift(r):
        """max|E - E0| / max(E_lattice per spin)"""
        return np.abs(r["energy"][:, 3]).max() / (np.abs(r["lattice"][:, 1]).max() / 2.)

    def balance(r):
        E = r["energy"]
        W = cumulative_work(E[:, 0], E[:, 4])
        return np.abs(E[:, 3] - W).max(), np.abs(W).max()

    # small oscillations
    for name in ("ipa_off", "ipa_none", "ipa_static", "ipa_dynamic", "hsex_none", "hsex_static", "hsex_dynamic"):
        S[name + "_w"] = frequency(R[name])
        S[name + "_drift"] = drift(R[name])
        data["series"][name] = {"t_fs": time_fs(R[name])[::2].tolist(), "rel_bohr": rel(R[name])[::2].tolist()}
    for name in ("hsex_static_dt2", "hsex_none_dt2"):
        S[name + "_drift"] = drift(R[name])
    for name in ("hsex_none", "hsex_none_dt2", "hsex_static", "hsex_static_dt2"):
        r = R[name]
        data["series"][name + "_energy"] = {"t_fs": (r["energy"][:, 0] * AU_FS).tolist(),
                                            "rel": (r["energy"][:, 3] / (np.abs(r["lattice"][:, 1]).max() / 2.)).tolist()}
    # bare frequencies (K_BO - Pi) and the adiabatic prediction of the coupled frequency without subtraction
    for kind in ("ipa", "hsex"):
        w_bare = R[kind + "_static"]["frequencies_dynamics"][4]
        S[kind + "_w_bare"] = w_bare
        S[kind + "_w_none_predicted"] = np.sqrt(max(2. * w_phx ** 2 - w_bare ** 2, 0.))
    # static reference vs amplitude: omega = omega_K + a + b s^2 (s = displacement / 0.01 A), omega_K the frequency of
    # K_BO (coupling off): a = non-adiabatic correction Pi(omega) - Pi(0) in the harmonic limit, b s^2 = non-linear
    # response of the ground state (kept by the static reference, removed by the dynamic one)
    w_K = S["ipa_off_w"]
    for kind in ("ipa", "hsex"):
        scales, ws = [], []
        for scale, name in [(1., kind + "_static"), (0.5, kind + "_static_u0.5"), (0.25, kind + "_static_u0.25")]:
            if name in R:
                scales.append(scale)
                ws.append(frequency(R[name]))
                S[name + "_w"] = ws[-1]
        if len(scales) == 3:
            A = np.vstack([np.ones(3), np.array(scales) ** 2]).T
            a, b = np.linalg.lstsq(A, np.array(ws) - w_K, rcond=None)[0]
            S[kind + "_nonadiabatic"] = a
            S[kind + "_nonlinear"] = b
    S["ipa_off_E_max"] =np.abs(R["ipa_off"]["energy"][:, 3]).max()
    S["ipa_rest_u"] = np.abs(R["ipa_rest"]["lattice"][:, 3:9]).max()
    S["ipa_rest_F"] = np.abs(R["ipa_rest"]["lattice"][:, 15:21]).max()
    for key in ("unscreening", "asr", "translation", "pi_asr", "gauge"):
        S["ipa_" + key] = R["ipa_static"].get(key)
        S["hsex_" + key] = R["hsex_static"].get(key)

    # laser
    for name in ("ipa_laser", "ipa_laser_nophonons", "hsex_laser_none", "hsex_laser_static", "hsex_laser_dynamic",
                 "hsex_laser_nophonons"):
        S[name + "_balance"], S[name + "_W"] = balance(R[name])
    E = R["ipa_laser"]["energy"]
    W = cumulative_work(E[:, 0], E[:, 4])
    data["series"]["ipa_laser"] = {"t_fs": time_fs(R["ipa_laser"]).tolist(), "rel_bohr": rel(R["ipa_laser"]).tolist(),
                                   "dE": E[:, 3].tolist(), "W": W.tolist(), "residual": (E[:, 3] - W).tolist()}
    tail = time_fs(R["ipa_laser"]) > 10.
    S["ipa_laser_w"] = fitted_frequency(R["ipa_laser"]["lattice"][tail, 0], rel(R["ipa_laser"])[tail], w_phx / HA_CM) * HA_CM
    for name in ("hsex_laser_none", "hsex_laser_static", "hsex_laser_dynamic"):
        data["series"][name] = {"t_fs": time_fs(R[name]).tolist(), "rel_bohr": rel(R[name]).tolist()}
    a, b, c = (rel(R["hsex_laser_" + k]) for k in ("none", "static", "dynamic"))
    S["hsex_laser_static_vs_dynamic"] = np.abs(b - c).max() / np.abs(b).max()
    S["hsex_laser_none_vs_static"] = np.abs(a - b).max() / np.abs(b).max()

    # MPI
    def lattice_difference(a, b):
        return np.abs(R[a]["lattice"][:, 3:] - R[b]["lattice"][:, 3:]).max() / np.abs(R[a]["lattice"][:, 3:]).max()
    S["ipa_mpi"] = max(lattice_difference("ipa_laser", "ipa_laser_np2"), lattice_difference("ipa_laser", "ipa_laser_np4"))
    S["hsex_mpi"] = max(lattice_difference(f"hsex_laser_{k}", f"hsex_laser_{k}_np{n}")
                        for k in ("static", "dynamic") for n in (2, 4))

    # gauge: populations, Peierls vs gradient, with and without phonons
    def population_difference(a, b):
        pa, pb = R[a]["population"], R[b]["population"]
        return np.abs(pa - pb).max() / np.abs(pa).max()
    S["gauge_with"] = population_difference("ipa_laser", "ipa_laser_gradient")
    S["gauge_without"] = population_difference("ipa_laser_nophonons", "ipa_laser_nophonons_gradient")
    S["phonon_population_effect"] = population_difference("ipa_laser", "ipa_laser_nophonons")

    # time step: displacement against AB4 at dt = 0.005 fs, on the common print times (every 0.1 fs)
    ref = rel(R["ipa_laser_ab_0.005"])
    conv = []
    for name, solver, dt in [("ipa_laser_rk_0.05", "RK4", 0.05), ("ipa_laser_rk_0.025", "RK4", 0.025),
                             ("ipa_laser_rk_0.0125", "RK4", 0.0125), ("ipa_laser", "RK4", 0.01),
                             ("ipa_laser_ab_0.01", "AB4", 0.01)]:
        x = rel(R[name])
        n = min(len(x), len(ref))
        conv.append([solver, dt, float(np.abs(x[:n] - ref[:n]).max() / np.abs(ref[:n]).max())])
    data["convergence"] = conv

    with open(os.path.join(HERE, "data_epw.json"), "w") as f:
        json.dump(data, f, indent=1, default=float)
    for key, value in S.items():
        print(f"{key:32s} {value}")
    for row in conv:
        print(row)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in ("run", "analyze"):
        print(__doc__)
        sys.exit(1)
    sim = os.path.abspath(sys.argv[2])
    run(sim, *sys.argv[3:]) if sys.argv[1] == "run" else analyze(sim)
