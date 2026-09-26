#!/usr/bin/env python3
"""
EDUS laser designer.

Designs the set of laser pulses of an EDUS simulation with the same definition used by EDUS
(src/Laser/Laser.cpp), in atomic units:

    E(t) = sqrt(I) * sin^2(pi (t - t0) / (N T)) * sin(w (t - t0) + phi) * p       for 0 < t - t0 <= N T
    A(t) = - int E dt

- linked fields (photon energy, frequency, wavelength, period, intensity, peak field, ...)
- several lasers (e.g. pump-probe, circular polarization), summed as in EDUS
- plots of E(t), A(t), spectrum |E(w)|^2 and polarization plane
- derived quantities: peak field, ponderomotive energy, Keldysh parameter, fluence, ...
- open/save the "lasers" block of an EDUS input, overlay Output/Laser.txt of a run

Usage: python3 laser.py [edus_input.json]
"""
import copy
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

# =========================================================
# Physical constants: the same of EDUS (src/Constants.hpp)
# =========================================================

BOHR = 5.2917721090380e-11            # m
EV = 1.602176634e-19                  # J
RYDBERG = 2.1798723611035e-18         # J
HARTREE = 2. * RYDBERG                # J
AU_TIME = 2.4188843265857e-17         # s
EPS0 = 8.8541878188e-12               # F/m
AU_FIELD = 5.14220674763e11           # V/m
FINE_STRUCTURE = 7.2973525693e-3
C_SI = 299792458.                     # m/s
C_AU = 1. / FINE_STRUCTURE            # speed of light in a.u.
AU_INTENSITY = 0.5 * C_SI * EPS0 * AU_FIELD**2   # W/m^2
FS = 1.e-15                           # s

# units accepted by EDUS (src/ConvertUnits.cpp), as factor to atomic units
ENERGY_UNITS = {"electronvolt": EV / HARTREE, "ev": EV / HARTREE, "joule": 1. / HARTREE,
                "auenergy": 1., "rydberg": RYDBERG / HARTREE}
LENGTH_UNITS = {"nanometers": 1.e-9 / BOHR, "nm": 1.e-9 / BOHR, "angstrom": 1.e-10 / BOHR, "aulength": 1.}
TIME_UNITS = {"femtoseconds": FS / AU_TIME, "fs": FS / AU_TIME, "autime": 1.}
INTENSITY_UNITS = {"wcm2": 1.e4 / AU_INTENSITY, "auintensity": 1.}


# =========================================================
# Physics (no GUI): everything in atomic units
# =========================================================

@dataclass
class Laser:
    """One sin^2 pulse, as in EDUS. All quantities in atomic units, phase in radians."""
    omega: float = 3. * EV / HARTREE
    intensity: float = 1.e10 * 1.e4 / AU_INTENSITY
    cycles: float = 5.
    phase: float = 0.
    t0: float = 0.
    polarization: list = field(default_factory=lambda: [1., 0., 0.])

    @property
    def period(self):
        return 2. * math.pi / self.omega

    @property
    def duration(self):
        return self.cycles * self.period

    @property
    def amplitude(self):
        return math.sqrt(self.intensity)

    def unit_polarization(self):
        p = np.array(self.polarization, dtype=float)
        norm = np.linalg.norm(p)
        return p / norm if norm > 0 else np.array([1., 0., 0.])

    def envelope(self, t):
        s = np.asarray(t, dtype=float) - self.t0
        inside = (s > 1.e-7) & (s <= self.duration)
        return np.where(inside, np.sin(math.pi * s / self.duration)**2, 0.)

    def field(self, t):
        """Electric field, shape (len(t), 3)."""
        s = np.asarray(t, dtype=float) - self.t0
        scalar = self.amplitude * self.envelope(t) * np.sin(self.omega * s + self.phase)
        return scalar[:, None] * self.unit_polarization()[None, :]


def total_field(lasers, t):
    E = np.zeros((len(t), 3))
    for laser in lasers:
        E += laser.field(t)
    return E


def vector_potential(t, E):
    """A(t) = -int_{t[0]}^t E dt' (trapezoidal rule), with A(t[0]) = 0 as in EDUS."""
    A = np.zeros_like(E)
    A[1:] = -np.cumsum(0.5 * (E[1:] + E[:-1]) * np.diff(t)[:, None], axis=0)
    return A


def time_window(lasers, margin=0.08):
    starts = [l.t0 for l in lasers]
    ends = [l.t0 + l.duration for l in lasers]
    t_min, t_max = min(starts), max(ends)
    span = max(t_max - t_min, 1.e-6)
    return t_min - margin * span, t_max + margin * span


def time_grid(lasers, points_per_period=60, max_points=400000, window=None):
    t_min, t_max = window if window else time_window(lasers)
    shortest = min(l.period for l in lasers)
    n = int((t_max - t_min) / shortest * points_per_period) + 2
    return np.linspace(t_min, t_max, min(max(n, 2000), max_points))


def spectrum(t, E, padding=8):
    """Photon energies (Hartree) and power spectrum sum_i |E_i(w)|^2 (normalized to 1)."""
    dt = t[1] - t[0]
    n = 1 << int(math.ceil(math.log2(len(t) * padding)))
    Ew = np.fft.rfft(E, n=n, axis=0)
    power = np.sum(np.abs(Ew)**2, axis=1)
    omega = 2. * math.pi * np.fft.rfftfreq(n, d=dt)
    return omega, power / max(power.max(), 1.e-300)


def derived_quantities(laser, gap_eV=None, reduced_mass=1.):
    """Quantities useful to choose the laser parameters. Energies in eV."""
    E0 = laser.amplitude
    omega = laser.omega
    up = E0**2 / (4. * omega**2) * HARTREE / EV
    q = {
        "E0_V_per_A": E0 * AU_FIELD * 1.e-10,
        "Up_eV": up,
        "photon_eV": omega * HARTREE / EV,
        # FWHM of the intensity envelope sin^4
        "fwhm_fs": laser.duration * (1. - 2. / math.pi * math.asin(2.**-0.25)) * AU_TIME / FS,
    }
    # fluence: c eps0 int E(t)^2 dt, in mJ/cm^2
    t = np.linspace(laser.t0, laser.t0 + laser.duration, 20000)
    E = laser.field(t)
    integral = np.trapezoid(np.sum(E**2, axis=1), t) if hasattr(np, "trapezoid") else np.trapz(np.sum(E**2, axis=1), t)
    q["fluence_mJ_cm2"] = C_SI * EPS0 * integral * AU_FIELD**2 * AU_TIME * 1.e3 / 1.e4
    if gap_eV and gap_eV > 0 and E0 > 0:
        gap = gap_eV * EV / HARTREE
        gamma = omega * math.sqrt(reduced_mass * gap) / E0
        q["keldysh"] = gamma
        q["regime"] = "multiphoton" if gamma > 1.2 else ("tunneling" if gamma < 0.8 else "intermediate")
        q["photons_to_gap"] = math.ceil(gap_eV / q["photon_eV"])
        q["cutoff_eV"] = gap_eV + 3.17 * up
        q["cutoff_order"] = q["cutoff_eV"] / q["photon_eV"]
    return q


# =========================================================
# EDUS input/output
# =========================================================

def _convert(value, units, table, what):
    key = str(units).strip().lower()
    if key not in table:
        raise ValueError(f"unknown {what} units '{units}' (EDUS accepts: {', '.join(table)})")
    return float(value) * table[key]


def laser_from_edus(d):
    """Laser from one element of the "lasers" block of an EDUS input."""
    laser = Laser()
    laser.intensity = _convert(d.get("intensity", 0.), d.get("intensity_units", "wcm2"), INTENSITY_UNITS, "intensity")
    if abs(float(d.get("frequency", 0.))) > 1.e-7:
        laser.omega = _convert(d["frequency"], d.get("frequency_units", "electronvolt"), ENERGY_UNITS, "frequency")
    elif float(d.get("wavelength", 0.)) > 0.:
        wavelength = _convert(d["wavelength"], d.get("wavelength_units", "nanometers"), LENGTH_UNITS, "wavelength")
        laser.omega = 2. * math.pi * C_AU / wavelength
    else:
        raise ValueError("each laser needs a nonzero frequency or wavelength")
    laser.cycles = float(d.get("cycles", 0.))
    laser.phase = float(d.get("phase", 0.))
    laser.t0 = _convert(d.get("t0", 0.), d.get("t0_units", "fs"), TIME_UNITS, "t0")
    laser.polarization = [float(x) for x in d.get("polarization", [1., 0., 0.])]
    return laser


def laser_to_edus(laser):
    """Element of the "lasers" block of an EDUS input, with explicit units."""
    return {
        "intensity": float(f"{laser.intensity * AU_INTENSITY / 1.e4:.10g}"),
        "intensity_units": "wcm2",
        "frequency": float(f"{laser.omega * HARTREE / EV:.10g}"),
        "frequency_units": "electronvolt",
        "polarization": [float(f"{x:.10g}") for x in laser.polarization],
        "cycles": float(f"{laser.cycles:.10g}"),
        "phase": float(f"{laser.phase:.12g}"),
        "t0": float(f"{laser.t0 * AU_TIME / FS:.10g}"),
        "t0_units": "fs",
    }


def read_edus_output(path):
    """Time (a.u.), E and A (a.u.) written by EDUS in path/Output (or path itself)."""
    directory = os.path.join(path, "Output") if os.path.isdir(os.path.join(path, "Output")) else path
    t = np.loadtxt(os.path.join(directory, "Time.txt")).reshape(-1)
    E = np.loadtxt(os.path.join(directory, "Laser.txt")).reshape(len(t), -1)
    A_file = os.path.join(directory, "Laser_A.txt")
    A = np.loadtxt(A_file).reshape(len(t), -1) if os.path.exists(A_file) else None
    return t, E, A


# =========================================================
# Presets
# =========================================================

def _laser(energy_eV=None, wavelength_nm=None, intensity_wcm2=1.e10, cycles=5., phase_pi=0., t0_fs=0., pol=(1, 0, 0)):
    omega = energy_eV * EV / HARTREE if energy_eV else 2. * math.pi * C_AU / (wavelength_nm * 1.e-9 / BOHR)
    return Laser(omega=omega, intensity=intensity_wcm2 * 1.e4 / AU_INTENSITY, cycles=cycles,
                 phase=phase_pi * math.pi, t0=t0_fs * FS / AU_TIME, polarization=list(map(float, pol)))


PRESETS = {
    "Absorption: 3 eV, 1 cycle, weak": (None, [_laser(energy_eV=3., intensity_wcm2=1.e5, cycles=1.)]),
    "HHG: 1.6 µm, 5·10¹¹ W/cm², 10 cycles (hBN, gap 7.25 eV)": (7.25, [_laser(wavelength_nm=1600., intensity_wcm2=5.e11, cycles=10.)]),
    "HHG: 3.2 µm, 10¹¹ W/cm², 8 cycles": (None, [_laser(wavelength_nm=3200., intensity_wcm2=1.e11, cycles=8.)]),
    "Pump-probe: 800 nm pump + 6 eV probe after 30 fs": (None, [
        _laser(wavelength_nm=800., intensity_wcm2=1.e11, cycles=10.),
        _laser(energy_eV=6., intensity_wcm2=1.e8, cycles=5., t0_fs=30., pol=(0, 1, 0))]),
    "Circular polarization: 800 nm, x + y with phase π/2": (None, [
        _laser(wavelength_nm=800., intensity_wcm2=5.e10, cycles=8., pol=(1, 0, 0)),
        _laser(wavelength_nm=800., intensity_wcm2=5.e10, cycles=8., phase_pi=0.5, pol=(0, 1, 0))]),
}


# =========================================================
# GUI
# =========================================================

def fmt(value):
    """5 significant digits."""
    try:
        return f"{float(value):.5g}"
    except (TypeError, ValueError):
        return ""


def parse(text):
    try:
        value = float(text)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


# Linked fields of one laser: (label, [(unit, getter, setter), ...]).
# Getters and setters convert between the canonical atomic units of Laser and the shown units.
def _set(attr, fn):
    def setter(laser, v):
        setattr(laser, attr, fn(laser, v))
    return setter


PHOTON_FIELDS = [
    ("Photon energy", [("eV", lambda l: l.omega * HARTREE / EV, _set("omega", lambda l, v: v * EV / HARTREE)),
                       ("Ha", lambda l: l.omega, _set("omega", lambda l, v: v))]),
    ("Frequency", [("THz", lambda l: l.omega / (2 * math.pi * AU_TIME) * 1.e-12,
                    _set("omega", lambda l, v: 2 * math.pi * v * 1.e12 * AU_TIME)),
                   ("a.u.", lambda l: l.omega, _set("omega", lambda l, v: v))]),
    ("Wavelength", [("nm", lambda l: 2 * math.pi * C_AU / l.omega * BOHR * 1.e9,
                     _set("omega", lambda l, v: 2 * math.pi * C_AU / (v * 1.e-9 / BOHR))),
                    ("a.u.", lambda l: 2 * math.pi * C_AU / l.omega, _set("omega", lambda l, v: 2 * math.pi * C_AU / v))]),
    ("Period", [("fs", lambda l: l.period * AU_TIME / FS, _set("omega", lambda l, v: 2 * math.pi / (v * FS / AU_TIME))),
                ("a.u.", lambda l: l.period, _set("omega", lambda l, v: 2 * math.pi / v))]),
]

PULSE_FIELDS = [
    ("Intensity", [("W/cm²", lambda l: l.intensity * AU_INTENSITY / 1.e4, _set("intensity", lambda l, v: v * 1.e4 / AU_INTENSITY)),
                   ("a.u.", lambda l: l.intensity, _set("intensity", lambda l, v: v))]),
    ("Peak field", [("V/Å", lambda l: l.amplitude * AU_FIELD * 1.e-10, _set("intensity", lambda l, v: (v / (AU_FIELD * 1.e-10))**2)),
                    ("a.u.", lambda l: l.amplitude, _set("intensity", lambda l, v: v**2))]),
    ("Cycles", [("", lambda l: l.cycles, _set("cycles", lambda l, v: v))]),
    ("Duration", [("fs", lambda l: l.duration * AU_TIME / FS, _set("cycles", lambda l, v: v * FS / AU_TIME / l.period)),
                  ("a.u.", lambda l: l.duration, _set("cycles", lambda l, v: v / l.period))]),
    ("Phase", [("π", lambda l: l.phase / math.pi, _set("phase", lambda l, v: v * math.pi)),
               ("rad", lambda l: l.phase, _set("phase", lambda l, v: v))]),
    ("Start t₀", [("fs", lambda l: l.t0 * AU_TIME / FS, _set("t0", lambda l, v: v * FS / AU_TIME)),
                  ("a.u.", lambda l: l.t0, _set("t0", lambda l, v: v))]),
]

# quantities that must be positive (phase and t0 can be anything)
POSITIVE = {"Photon energy", "Frequency", "Wavelength", "Period", "Intensity", "Peak field", "Cycles", "Duration"}

# rows of the table of derived quantities: (name, text from the dictionary of derived_quantities)
DERIVED_ROWS = [
    ("Peak field E₀ (V/Å)", lambda q: fmt(q["E0_V_per_A"])),
    ("Ponderomotive Uₚ (eV)", lambda q: fmt(q["Up_eV"])),
    ("Keldysh γ", lambda q: fmt(q["keldysh"]) if "keldysh" in q else "— (set gap)"),
    ("Regime", lambda q: q.get("regime", "—")),
    ("Photons to cross the gap", lambda q: str(q["photons_to_gap"]) if "photons_to_gap" in q else "—"),
    ("E_g + 3.17 Uₚ (eV / order)", lambda q: f"{q['cutoff_eV']:.4g} / {q['cutoff_order']:.3g}" if "cutoff_eV" in q else "—"),
    ("FWHM of I(t) (fs)", lambda q: fmt(q["fwhm_fs"])),
    ("Fluence (mJ/cm²)", lambda q: fmt(q["fluence_mJ_cm2"])),
]

THEMES = {
    "light": dict(bg="#f4f6fa", panel="#ffffff", fg="#1d2433", muted="#6b7385", accent="#2f6fdf",
                  entry="#ffffff", border="#d5dbe6", grid="#e3e7ef", plot_bg="#ffffff",
                  lines=["#2f6fdf", "#e0543e", "#20a67a"], envelope="#9aa6bd", gap="#c73d8f", edus="#111111"),
    "dark": dict(bg="#161a22", panel="#1f2430", fg="#e6e9f0", muted="#8b93a7", accent="#5b9cff",
                 entry="#2a3040", border="#343b4d", grid="#2c3242", plot_bg="#1b2029",
                 lines=["#5b9cff", "#ff7a66", "#3ed6a0"], envelope="#5d667c", gap="#ff6ec7", edus="#f5f5f5"),
}


class LaserDesigner:

    def __init__(self, root, input_file=None):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        self.tk, self.ttk = tk, ttk

        import tkinter.font as tkfont
        self.root = root
        self.root.title("EDUS laser designer")
        # sizes follow the font, so that the layout works also on HiDPI screens
        self.ui_scale = max(1., tkfont.nametofont("TkDefaultFont").metrics("linespace") / 18.)
        width, height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{int(0.85 * width)}x{int(0.85 * height)}")
        self.root.minsize(int(0.5 * width), int(0.5 * height))

        self.lasers = [copy.deepcopy(l) for l in PRESETS["HHG: 1.6 µm, 5·10¹¹ W/cm², 10 cycles (hBN, gap 7.25 eV)"][1]]
        self.selected = 0
        self.input_file = None          # EDUS input loaded (its other parameters are kept when saving)
        self.input_dict = None
        self.edus_output = None         # (t, E, A) of a run, shown on top of the model
        self.theme = "light"
        self.updating = False           # True while the fields are filled by the program
        self.redraw_job = None

        self.gap = tk.StringVar(value="7.25")
        self.reduced_mass = tk.StringVar(value="1")
        self.log_spectrum = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Ready")
        self.field_vars = {}            # (label, unit index) -> StringVar
        self.pol_vars = [tk.StringVar() for _ in range(3)]
        self.slider_vars = {"Intensity": tk.DoubleVar(), "Cycles": tk.DoubleVar(), "Phase": tk.DoubleVar()}

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self._build_header()
        ttk.Label(self.root, textvariable=self.status, style="Status.TLabel", anchor="w").pack(fill="x", side="bottom")
        body = ttk.Frame(self.root, style="Bg.TFrame")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        left = self._scrollable(body)
        right = ttk.Frame(body, style="Bg.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self._build_laser_list(left)
        self._build_form(left)
        self._build_material(left)
        self._build_derived(right)
        self._build_plots(right, Figure, FigureCanvasTkAgg, NavigationToolbar2Tk)

        self.apply_theme()
        self.refresh_list()
        self.refresh_fields()
        self.redraw()
        if input_file:
            self.open_input(input_file)

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    def _card(self, parent, title):
        frame = self.ttk.LabelFrame(parent, text=f"  {title}  ", style="Card.TLabelframe", padding=(10, 6))
        frame.pack(fill="x", pady=(0, 8))
        return frame

    def _scrollable(self, parent):
        """Left column: a frame inside a canvas, with a vertical scrollbar when it does not fit."""
        tk, ttk = self.tk, self.ttk
        container = ttk.Frame(parent, style="Bg.TFrame")
        container.pack(side="left", fill="y")
        self.left_canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.left_canvas.yview)
        inner = ttk.Frame(self.left_canvas, style="Bg.TFrame")
        self.left_canvas.create_window((0, 0), window=inner, anchor="nw")
        self.left_canvas.configure(yscrollcommand=scrollbar.set)
        inner.bind("<Configure>", lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"),
                                                                       width=inner.winfo_reqwidth()))
        self.left_canvas.pack(side="left", fill="y")
        scrollbar.pack(side="left", fill="y", padx=(2, 0))

        def wheel(event):
            if str(event.widget).startswith(str(container)):
                self.left_canvas.yview_scroll(-1 if (event.num == 4 or event.delta > 0) else 1, "units")
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(sequence, wheel, add="+")
        return inner

    def _build_header(self):
        ttk = self.ttk
        header = ttk.Frame(self.root, style="Bg.TFrame", padding=(14, 8))
        header.pack(fill="x")
        title = ttk.Frame(header, style="Bg.TFrame")
        title.pack(fill="x")
        ttk.Label(title, text="⚡ EDUS laser designer", style="Title.TLabel").pack(side="left")
        ttk.Label(title, text="   sin² pulses exactly as in EDUS", style="Muted.TLabel").pack(side="left", pady=(8, 0))

        tools = ttk.Frame(header, style="Bg.TFrame")
        tools.pack(fill="x", pady=(8, 0))
        self.preset = self.tk.StringVar(value="Presets…")
        box = ttk.Combobox(tools, textvariable=self.preset, values=list(PRESETS), state="readonly", width=48)
        box.bind("<<ComboboxSelected>>", lambda e: self.load_preset(self.preset.get()))
        # the wheel would change the preset while scrolling over it
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            box.bind(sequence, lambda e: "break")
        box.pack(side="left", padx=(0, 8))
        for text, command in [("📂 Open input", self.open_input_dialog), ("💾 Save input", self.save_input),
                              ("📈 Load EDUS output", self.load_output_dialog), ("⤓ Export field", self.export_field)]:
            ttk.Button(tools, text=text, command=command, style="Tool.TButton").pack(side="left", padx=3)
        ttk.Button(tools, text="◐ Theme", command=self.toggle_theme, style="Small.TButton").pack(side="right")

    def _build_laser_list(self, parent):
        ttk = self.ttk
        card = self._card(parent, "Lasers")
        self.listbox = self.tk.Listbox(card, height=4, width=46, activestyle="none", exportselection=False,
                                       borderwidth=0, highlightthickness=1, font=("TkDefaultFont", 10))
        self.listbox.pack(fill="x")
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        buttons = ttk.Frame(card, style="Card.TFrame")
        buttons.pack(fill="x", pady=(6, 0))
        for text, command in [("＋ Add", self.add_laser), ("⧉ Duplicate", self.duplicate_laser), ("✕ Remove", self.remove_laser)]:
            ttk.Button(buttons, text=text, command=command, style="Small.TButton").pack(side="left", padx=(0, 4))

    def _build_form(self, parent):
        ttk = self.ttk
        sliders = {"Intensity": (6., 15., "log₁₀ W/cm²"), "Cycles": (0.5, 40., ""), "Phase": (0., 2., "π")}
        for title, fields in [("Photon", PHOTON_FIELDS), ("Pulse", PULSE_FIELDS)]:
            card = self._card(parent, title)
            row = 0
            for label, units in fields:
                ttk.Label(card, text=label, style="Card.TLabel", width=13).grid(row=row, column=0, sticky="w", pady=2)
                for iu, (unit, _, _) in enumerate(units):
                    var = self.tk.StringVar()
                    self.field_vars[(label, iu)] = var
                    entry = ttk.Entry(card, textvariable=var, width=11, style="Field.TEntry")
                    entry.grid(row=row, column=1 + 2 * iu, sticky="w", pady=2)
                    entry.bind("<KeyRelease>", lambda e, l=label, i=iu: self.on_edit(l, i))
                    entry.bind("<FocusOut>", lambda e: self.refresh_fields())
                    ttk.Label(card, text=unit, style="CardMuted.TLabel", width=6).grid(row=row, column=2 + 2 * iu, sticky="w", padx=(3, 8))
                row += 1
                if label in sliders:
                    lo, hi, _ = sliders[label]
                    scale = ttk.Scale(card, from_=lo, to=hi, variable=self.slider_vars[label],
                                      command=lambda v, l=label: self.on_slider(l))
                    scale.grid(row=row, column=1, columnspan=4, sticky="we", pady=(0, 4))
                    row += 1
            if title == "Pulse":
                ttk.Label(card, text="Polarization", style="Card.TLabel", width=13).grid(row=row, column=0, sticky="w", pady=2)
                pol = ttk.Frame(card, style="Card.TFrame")
                pol.grid(row=row, column=1, columnspan=4, sticky="w")
                for i, name in enumerate("xyz"):
                    ttk.Label(pol, text=name, style="CardMuted.TLabel").pack(side="left", padx=(0 if i == 0 else 8, 2))
                    entry = ttk.Entry(pol, textvariable=self.pol_vars[i], width=6, style="Field.TEntry")
                    entry.pack(side="left")
                    entry.bind("<KeyRelease>", lambda e: self.on_polarization())

    def _build_material(self, parent):
        ttk = self.ttk
        card = self._card(parent, "Material (for Keldysh, cutoff, spectrum)")
        for i, (text, var, unit) in enumerate([("Gap", self.gap, "eV"), ("Reduced mass", self.reduced_mass, "mₑ")]):
            ttk.Label(card, text=text, style="Card.TLabel", width=13).grid(row=0, column=3 * i, sticky="w")
            entry = ttk.Entry(card, textvariable=var, width=8, style="Field.TEntry")
            entry.grid(row=0, column=3 * i + 1, sticky="w")
            entry.bind("<KeyRelease>", lambda e: self.schedule_redraw())
            ttk.Label(card, text=unit, style="CardMuted.TLabel").grid(row=0, column=3 * i + 2, sticky="w", padx=(3, 14))

    def _build_plots(self, parent, Figure, FigureCanvasTkAgg, NavigationToolbar2Tk):
        ttk = self.ttk
        self.notebook = ttk.Notebook(parent, style="Plots.TNotebook")
        self.notebook.pack(fill="both", expand=True)
        self.figures = {}
        for name in ["Electric field", "Vector potential", "Spectrum", "Polarization"]:
            tab = ttk.Frame(self.notebook, style="Card.TFrame")
            self.notebook.add(tab, text=f"  {name}  ")
            fig = Figure(figsize=(8, 4.6), dpi=100 * self.ui_scale)
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figures[name] = (fig, ax, canvas, toolbar)
            if name == "Spectrum":
                ttk.Checkbutton(tab, text="log scale", variable=self.log_spectrum, command=self.redraw,
                                style="Card.TCheckbutton").place(relx=0.99, rely=0.01, anchor="ne")

    def _build_derived(self, parent):
        ttk = self.ttk
        card = self._card(parent, "Derived quantities")
        card.pack_configure(side="bottom", pady=(8, 0))
        # one row per quantity, one column per laser
        self.table = ttk.Treeview(card, show="headings", height=len(DERIVED_ROWS), style="Derived.Treeview")
        self.table.pack(fill="x")

    # -----------------------------------------------------
    # Theme
    # -----------------------------------------------------

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.apply_theme()
        self.redraw()

    def apply_theme(self):
        c = THEMES[self.theme]
        s = self.style
        self.root.configure(bg=c["bg"])
        s.configure(".", background=c["bg"], foreground=c["fg"], fieldbackground=c["entry"], bordercolor=c["border"])
        s.configure("Bg.TFrame", background=c["bg"])
        s.configure("Card.TFrame", background=c["panel"])
        s.configure("Card.TLabelframe", background=c["panel"], bordercolor=c["border"], relief="solid")
        s.configure("Card.TLabelframe.Label", background=c["panel"], foreground=c["accent"], font=("TkDefaultFont", 10, "bold"))
        s.configure("Card.TLabel", background=c["panel"], foreground=c["fg"])
        s.configure("CardMuted.TLabel", background=c["panel"], foreground=c["muted"])
        s.configure("Card.TCheckbutton", background=c["panel"], foreground=c["fg"])
        s.configure("Title.TLabel", background=c["bg"], foreground=c["accent"], font=("TkDefaultFont", 20, "bold"))
        s.configure("Muted.TLabel", background=c["bg"], foreground=c["muted"])
        s.configure("Status.TLabel", background=c["panel"], foreground=c["muted"], padding=(12, 4))
        s.configure("Field.TEntry", fieldbackground=c["entry"], foreground=c["fg"], insertcolor=c["fg"])
        s.configure("Tool.TButton", background=c["accent"], foreground="white", borderwidth=0, padding=(10, 5))
        s.map("Tool.TButton", background=[("active", c["fg"])], foreground=[("active", c["bg"])])
        s.configure("Small.TButton", background=c["panel"], foreground=c["fg"], bordercolor=c["border"], padding=(8, 3))
        s.configure("Plots.TNotebook", background=c["bg"], bordercolor=c["border"])
        s.configure("Plots.TNotebook.Tab", background=c["bg"], foreground=c["muted"], padding=(8, 4))
        s.map("Plots.TNotebook.Tab", background=[("selected", c["panel"])], foreground=[("selected", c["accent"])])
        s.configure("Derived.Treeview", background=c["panel"], fieldbackground=c["panel"], foreground=c["fg"], rowheight=int(26 * self.ui_scale))
        s.configure("Derived.Treeview.Heading", background=c["bg"], foreground=c["muted"])
        s.configure("Horizontal.TScale", background=c["panel"], troughcolor=c["grid"])
        s.configure("TCombobox", fieldbackground=c["entry"], foreground=c["fg"], background=c["panel"], arrowcolor=c["fg"])
        s.map("TCombobox", fieldbackground=[("readonly", c["entry"])], foreground=[("readonly", c["fg"])],
              selectbackground=[("readonly", c["entry"])], selectforeground=[("readonly", c["fg"])])
        self.root.option_add("*TCombobox*Listbox.background", c["entry"])
        self.root.option_add("*TCombobox*Listbox.foreground", c["fg"])
        self.left_canvas.configure(bg=c["bg"])
        self.listbox.configure(bg=c["entry"], fg=c["fg"], selectbackground=c["accent"], selectforeground="white",
                               highlightbackground=c["border"], highlightcolor=c["accent"])
        for fig, ax, canvas, toolbar in self.figures.values():
            fig.set_facecolor(c["panel"])
            toolbar.configure(background=c["panel"])
            for child in toolbar.winfo_children():
                try:
                    child.configure(background=c["panel"])
                except self.tk.TclError:
                    pass

    def _style_axes(self, ax, xlabel, ylabel):
        c = THEMES[self.theme]
        ax.set_facecolor(c["plot_bg"])
        ax.set_xlabel(xlabel, color=c["fg"])
        ax.set_ylabel(ylabel, color=c["fg"])
        ax.tick_params(colors=c["muted"])
        for spine in ax.spines.values():
            spine.set_color(c["border"])
        ax.grid(True, color=c["grid"], linewidth=0.8)
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(c["panel"])
            legend.get_frame().set_edgecolor(c["border"])
            for text in legend.get_texts():
                text.set_color(c["fg"])

    # -----------------------------------------------------
    # Lasers list and fields
    # -----------------------------------------------------

    def laser(self):
        return self.lasers[self.selected]

    def refresh_list(self):
        self.listbox.delete(0, "end")
        for i, l in enumerate(self.lasers):
            q = derived_quantities(l)
            p = l.unit_polarization()
            axis = next((name for name, v in zip("xyz", np.eye(3)) if np.allclose(p, v)), f"({p[0]:.2g},{p[1]:.2g},{p[2]:.2g})")
            self.listbox.insert("end", f" #{i + 1}  {q['photon_eV']:.3g} eV · {l.intensity * AU_INTENSITY / 1.e4:.2g} W/cm² · "
                                       f"{l.cycles:g} cyc · t₀ {l.t0 * AU_TIME / FS:.3g} fs · {axis}")
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.selected)

    def refresh_fields(self, skip=None):
        """Writes the values of the selected laser in all fields but skip (the one being edited)."""
        self.updating = True
        laser = self.laser()
        for label, units in PHOTON_FIELDS + PULSE_FIELDS:
            for iu, (_, getter, _) in enumerate(units):
                if (label, iu) != skip:
                    self.field_vars[(label, iu)].set(fmt(getter(laser)))
        if skip is None or skip[0] != "polarization":
            for var, x in zip(self.pol_vars, laser.polarization):
                var.set(fmt(x))
        self.slider_vars["Intensity"].set(math.log10(max(laser.intensity * AU_INTENSITY / 1.e4, 1.e-30)))
        self.slider_vars["Cycles"].set(laser.cycles)
        self.slider_vars["Phase"].set((laser.phase / math.pi) % 2.)
        self.updating = False

    def on_edit(self, label, iu):
        if self.updating:
            return
        value = parse(self.field_vars[(label, iu)].get())
        if value is None or (label in POSITIVE and value <= 0):
            self.status.set(f"{label}: invalid value")
            return
        units = dict(PHOTON_FIELDS + PULSE_FIELDS)[label]
        units[iu][2](self.laser(), value)
        self.refresh_fields(skip=(label, iu))
        self.status.set("Ready")
        self.schedule_redraw()

    def on_slider(self, label):
        if self.updating:
            return
        v = self.slider_vars[label].get()
        laser = self.laser()
        if label == "Intensity":
            laser.intensity = 10.**v * 1.e4 / AU_INTENSITY
        elif label == "Cycles":
            laser.cycles = round(v * 2.) / 2.
        elif label == "Phase":
            laser.phase = round(v * 20.) / 20. * math.pi
        self.refresh_fields()
        self.schedule_redraw()

    def on_polarization(self):
        values = [parse(v.get()) for v in self.pol_vars]
        if None in values or all(abs(x) < 1.e-12 for x in values):
            self.status.set("Polarization: give three numbers, not all zero")
            return
        self.laser().polarization = values
        self.status.set("Ready")
        self.schedule_redraw()

    def on_select(self, event=None):
        selection = self.listbox.curselection()
        if selection:
            self.selected = selection[0]
            self.refresh_fields()

    def add_laser(self):
        self.lasers.append(Laser())
        self.selected = len(self.lasers) - 1
        self._lasers_changed()

    def duplicate_laser(self):
        self.lasers.insert(self.selected + 1, copy.deepcopy(self.laser()))
        self.selected += 1
        self._lasers_changed()

    def remove_laser(self):
        if len(self.lasers) == 1:
            self.status.set("At least one laser is needed")
            return
        del self.lasers[self.selected]
        self.selected = min(self.selected, len(self.lasers) - 1)
        self._lasers_changed()

    def _lasers_changed(self):
        self.refresh_list()
        self.refresh_fields()
        self.redraw()

    def load_preset(self, name):
        gap, lasers = PRESETS[name]
        self.lasers = [copy.deepcopy(l) for l in lasers]
        if gap is not None:
            self.gap.set(fmt(gap))
        self.selected = 0
        self.edus_output = None
        self.status.set(f"Preset: {name}")
        self._lasers_changed()

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------

    def schedule_redraw(self):
        if self.redraw_job:
            self.root.after_cancel(self.redraw_job)
        self.redraw_job = self.root.after(120, self.redraw)

    def material(self):
        gap, mass = parse(self.gap.get()), parse(self.reduced_mass.get())
        return (gap if gap and gap > 0 else None), (mass if mass and mass > 0 else 1.)

    def simulation_window(self):
        """initialtime and finaltime (a.u.) of the loaded input, if any."""
        d = self.input_dict or {}
        try:
            t_i = _convert(d["initialtime"], d.get("initialtime_units", "fs"), TIME_UNITS, "time")
            t_f = _convert(d["finaltime"], d.get("finaltime_units", "fs"), TIME_UNITS, "time")
            return t_i, t_f
        except (KeyError, ValueError, TypeError):
            return None

    def redraw(self):
        self.redraw_job = None
        c = THEMES[self.theme]
        self.refresh_list()
        window = time_window(self.lasers)
        sim = self.simulation_window()
        if sim:
            window = (min(window[0], sim[0]), max(window[1], sim[1]))
        t = time_grid(self.lasers, window=window)
        E = total_field(self.lasers, t)
        A = vector_potential(t, E)
        t_fs = t * AU_TIME / FS
        active = [i for i in range(3) if np.abs(E[:, i]).max() > 0] or [0]
        gap, mass = self.material()

        # electric field
        fig, ax, canvas, _ = self.figures["Electric field"]
        ax.clear()
        to_VA = AU_FIELD * 1.e-10
        for laser in self.lasers:
            env = laser.amplitude * laser.envelope(t) * to_VA
            ax.fill_between(t_fs, -env, env, color=c["envelope"], alpha=0.18, linewidth=0)
        for i in active:
            ax.plot(t_fs, E[:, i] * to_VA, color=c["lines"][i], linewidth=1.3, label=f"E{'xyz'[i]}")
        self._draw_edus(ax, "E", to_VA)
        self._draw_window(ax, sim, t_fs)
        ax.legend(loc="upper right")
        self._style_axes(ax, "Time (fs)", "Electric field (V/Å)")
        ax.set_xlim(t_fs[0], t_fs[-1])
        fig.tight_layout()
        canvas.draw_idle()

        # vector potential
        fig, ax, canvas, _ = self.figures["Vector potential"]
        ax.clear()
        for i in active:
            ax.plot(t_fs, A[:, i], color=c["lines"][i], linewidth=1.3, label=f"A{'xyz'[i]}")
        self._draw_edus(ax, "A", 1.)
        self._draw_window(ax, sim, t_fs)
        ax.legend(loc="upper right")
        self._style_axes(ax, "Time (fs)", "Vector potential (a.u.)")
        ax.set_xlim(t_fs[0], t_fs[-1])
        fig.tight_layout()
        canvas.draw_idle()

        # spectrum
        fig, ax, canvas, _ = self.figures["Spectrum"]
        ax.clear()
        omega, power = spectrum(t, E)
        energy = omega * HARTREE / EV
        photon_max = max(l.omega for l in self.lasers) * HARTREE / EV
        e_max = max(3. * photon_max, 1.3 * gap if gap else 0.)
        keep = energy <= e_max
        ax.plot(energy[keep], power[keep], color=c["accent"], linewidth=1.4)
        ax.fill_between(energy[keep], power[keep], color=c["accent"], alpha=0.15, linewidth=0)
        if self.log_spectrum.get():
            ax.set_yscale("log")
            ax.set_ylim(1.e-8, 2.)
        if gap:
            ax.axvline(gap, color=c["gap"], linestyle="--", linewidth=1.3, label=f"gap {gap:g} eV")
            ax.axvspan(gap, e_max, color=c["gap"], alpha=0.06)
            above = power[energy >= gap].sum() / power.sum()
            ax.text(0.98, 0.9, f"spectral weight above gap: {above:.2e}", transform=ax.transAxes, ha="right", color=c["gap"])
            ax.legend(loc="lower left")
        self._style_axes(ax, "Photon energy (eV)", "|E(ω)|² (normalized)")
        ax.set_xlim(0., e_max)
        fig.tight_layout()
        canvas.draw_idle()

        # polarization
        fig, ax, canvas, _ = self.figures["Polarization"]
        ax.clear()
        planes = [(0, 1), (0, 2), (1, 2)]
        i, j = next(((a, b) for a, b in planes if a in active and b in active), (0, 1))
        on = np.any(E != 0., axis=1)
        points = ax.scatter(E[on, i] * to_VA, E[on, j] * to_VA, c=t_fs[on], s=2, cmap="plasma")
        limit = max(np.abs(E[:, [i, j]]).max() * to_VA, 1.e-30) * 1.1
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        if not hasattr(self, "colorbar"):
            self.colorbar = fig.colorbar(points, ax=ax, label="Time (fs)")
        else:
            self.colorbar.update_normal(points)
        self.colorbar.ax.yaxis.label.set_color(c["fg"])
        self.colorbar.ax.tick_params(colors=c["muted"])
        self._style_axes(ax, f"E{'xyz'[i]} (V/Å)", f"E{'xyz'[j]} (V/Å)")
        canvas.draw_idle()

        self.refresh_table(gap, mass)

    def _draw_window(self, ax, sim, t_fs):
        """Shades the times outside [initialtime, finaltime] of the loaded input."""
        if sim:
            c = THEMES[self.theme]
            a, b = (x * AU_TIME / FS for x in sim)
            label = "outside simulation"
            for lo, hi in [(t_fs[0], a), (b, t_fs[-1])]:
                if hi > lo:
                    ax.axvspan(lo, hi, color=c["muted"], alpha=0.12, linewidth=0, label=label)
                    label = None

    def _draw_edus(self, ax, what, factor):
        if self.edus_output is None:
            return
        t, E, A = self.edus_output
        data = E if what == "E" else A
        if data is None:
            return
        c = THEMES[self.theme]
        step = max(1, len(t) // 400)
        for i in range(data.shape[1]):
            if np.abs(data[:, i]).max() > 0:
                ax.plot(t[::step] * AU_TIME / FS, data[::step, i] * factor, "o", markersize=2.5,
                        color=c["edus"], alpha=0.7, label=f"EDUS {what}{'xyz'[i]}")

    def refresh_table(self, gap, mass):
        columns = ["quantity"] + [f"laser{n}" for n in range(len(self.lasers))]
        self.table.configure(columns=columns)
        self.table.heading("quantity", text="Quantity", anchor="w")
        self.table.column("quantity", width=int(230 * self.ui_scale), anchor="w", stretch=False)
        for n in range(len(self.lasers)):
            self.table.heading(f"laser{n}", text=f"Laser #{n + 1}")
            self.table.column(f"laser{n}", width=int(130 * self.ui_scale), anchor="center")
        self.table.delete(*self.table.get_children())
        quantities = [derived_quantities(laser, gap, mass) for laser in self.lasers]
        for name, text in DERIVED_ROWS:
            self.table.insert("", "end", values=[name] + [text(q) for q in quantities])

    # -----------------------------------------------------
    # Files
    # -----------------------------------------------------

    def open_input_dialog(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Open EDUS input", filetypes=[("JSON", "*.json"), ("All files", "*")])
        if path:
            self.open_input(path)

    def open_input(self, path):
        from tkinter import messagebox
        try:
            with open(path) as f:
                d = json.load(f)
            lasers = [laser_from_edus(x) for x in d.get("lasers", [])]
            if not lasers:
                raise ValueError("the input has no lasers")
        except Exception as error:
            messagebox.showerror("Open input", f"{path}:\n{error}")
            return
        self.input_file, self.input_dict = path, d
        self.lasers, self.selected, self.edus_output = lasers, 0, None
        self.status.set(f"Opened {path}: {len(lasers)} laser(s). Saving keeps all the other parameters.")
        self._lasers_changed()

    def save_input(self):
        from tkinter import filedialog, messagebox
        initial = os.path.basename(self.input_file) if self.input_file else "input.json"
        path = filedialog.asksaveasfilename(title="Save EDUS input", defaultextension=".json", initialfile=initial,
                                            initialdir=os.path.dirname(self.input_file) if self.input_file else None,
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        d = copy.deepcopy(self.input_dict) if self.input_dict else {}
        d["lasers"] = [laser_to_edus(l) for l in self.lasers]
        try:
            with open(path, "w") as f:
                json.dump(d, f, indent=4)
        except OSError as error:
            messagebox.showerror("Save input", str(error))
            return
        what = "input with the new lasers" if self.input_dict else "lasers block"
        self.status.set(f"Saved {what} in {path}")

    def load_output_dialog(self):
        from tkinter import filedialog
        path = filedialog.askdirectory(title="EDUS run directory (containing Output/)")
        if path:
            self.load_output(path)

    def load_output(self, path):
        from tkinter import messagebox
        try:
            t, E, A = read_edus_output(path)
        except Exception as error:
            messagebox.showerror("Load EDUS output", f"{path}:\n{error}")
            return
        t = self.exact_times(t)
        self.edus_output = (t, E, A)
        model = total_field(self.lasers, t)
        deviation = np.abs(model - E).max() / max(np.abs(model).max(), 1.e-300)
        self.status.set(f"EDUS output {path}: {len(t)} times, max |E_EDUS - E_model| / max|E| = {deviation:.2e}")
        self.redraw()

    def exact_times(self, t):
        """EDUS writes Time.txt with 6 significant digits: if the input is loaded, the times are rebuilt
        as initialtime + n*dt*step, when they agree with the file within its precision."""
        d = self.input_dict or {}
        try:
            dt = _convert(d["dt"], d.get("dt_units", "fs"), TIME_UNITS, "dt")
            t_i = _convert(d.get("initialtime", 0.), d.get("initialtime_units", "fs"), TIME_UNITS, "time")
        except (KeyError, ValueError, TypeError):
            return t
        if len(t) < 2:
            return t
        step = max(1, round((t[1] - t[0]) / dt))
        rebuilt = t_i + np.arange(len(t)) * dt * step
        tolerance = 1.e-5 * max(np.abs(t).max(), 1.) + 1.e-6
        return rebuilt if np.abs(rebuilt - t).max() < tolerance else t

    def export_field(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(title="Export electric field", defaultextension=".txt",
                                            filetypes=[("Text", "*.txt"), ("All files", "*")])
        if not path:
            return
        t = time_grid(self.lasers)
        E = total_field(self.lasers, t)
        A = vector_potential(t, E)
        header = "EDUS laser designer: total field of the lasers\n" + \
                 "\n".join(json.dumps(laser_to_edus(l)) for l in self.lasers) + \
                 "\ntime (fs)  Ex Ey Ez (a.u.)  Ax Ay Az (a.u.)"
        np.savetxt(path, np.column_stack([t * AU_TIME / FS, E, A]), header=header, fmt="%.12e")
        self.status.set(f"Field exported in {path}")

    def screenshot(self, path):
        """Saves an image of the window (used to check the layout)."""
        self.root.update()
        os.system(f"import -window {self.root.winfo_id()} {path}")


def main():
    import tkinter as tk
    args = [a for a in sys.argv[1:] if not a.startswith("--screenshot=")]
    shot = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--screenshot=")]
    root = tk.Tk()
    app = LaserDesigner(root, args[0] if args else None)
    if shot:
        root.after(1500, lambda: (app.screenshot(shot[0]), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
