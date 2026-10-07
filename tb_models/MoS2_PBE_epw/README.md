# Monolayer MoS2 from QE/EPW (electron-phonon coupling at q = 0)

Real output of Quantum ESPRESSO + EPW (EPW 6.1 build) used by the tests `MoS2_epw_phonons_*` of the Ehrenfest
dynamics of the lattice. The inputs are in `qe_inputs/` (`pseudo_dir` is the one of the machine where they ran; the
pseudopotentials are the PBE ONCV of the EPW distribution).

| File | From |
|---|---|
| `mos2_tb.dat` | Wannier90 inside EPW (`write_tb`), the `tb_file` of EDUS |
| `epwdata.fmt`, `wigner.fmt`, `crystal.fmt`, `mos2.epmatwp` | `epw.x` (`epwwrite`, nq = 1): H(R_e), Wigner-Seitz vectors, crystal, g(R_e, R_g) |
| `mos2.dyn1` | `ph.x` at Gamma |

Calculation: PBE without spin-orbit, a = 3.184 A, S height relaxed (S-S 3.127 A, forces 0), 20 A of vacuum,
`assume_isolated = '2D'`, 80 Ry, 12x12 k points. 11 Wannier functions (Mo d + S p, `exclude_bands = 1:6`: Mo 4s 4p and
S 3s), total spread 18.6 A^2, 7 filled bands. Disentanglement windows relative to the VBM: frozen VBM - 8 .. VBM + 2.5 eV,
outer up to VBM + 7 eV.

Checks: Wannier bands on the DFT bands of G-M-K-G within 6 meV in the frozen window (1 meV for the valence), gap
1.671 eV (indirect Gamma -> K: the PBE valence top at Gamma is 11 meV above K, the direct gap at K is 1.682 eV, as in DFT); `max|H_EPW - H_tb|` = 6e-8 eV in the recap of EDUS. ph.x at Gamma (cm^-1): E'' 276.6,
E' 373.7, A1' 396.7, A2'' 458.7, acoustic below 3.1.

Note on the static adiabatic reference: for A1' the electrons of the model give Pi = -6.6 K_BO (frequency of
K0 = K_BO - Pi: 1092 cm^-1), so the electronic anharmonicity of the model is amplified. Without laser the A1'
frequency changes with the initial displacement as omega/omega_ph.x - 1 = -2.0e-3 + 1.15e3 u0^2 (u0 of each S in A):
the linear limit is the non-adiabatic softening, at u0 = 0.005 A the frequency is 2.7 % above ph.x. The dynamic
reference gives the frequency of ph.x exactly at any amplitude.

## Bare and screened coupling

`mos2.epmatwp` here was written by the local EPW build with `bare_only = .TRUE.` (`~/codes/q-e-EPW-6.1`): it is the
fully bare coupling dV_ion (no dvscf, no nonlinear core correction), g_b for `epw_directory`.
`screened/` has the same files from the standard EPW (`bare_only = .FALSE.`, `~/codes/q-e-EPW-6.1-screened`), run
with `wannierize = .false.` on the same `.ukk`: same Wannier functions (`epwdata.fmt` identical), screened coupling g_s
for `epw_directory_screened` (or for `epw_directory` alone). max|g_s| = 0.098, max|g_b| = 6.15 (Ry/bohr units of EPW),
||g_s|| / ||g_b|| = 0.066.

A1' displaced by 0.005 A, IPA, 6x6, static reference, 180 fs (frequency fit / energy drift relative to E_lattice):
- g_b alone: 407.88 cm^-1 (+2.8 %), drift 7e-4 (the bare g gives Pi = -6.6 K_BO for A1');
- g_s alone: 396.707 cm^-1 (ph.x 396.705), drift 2e-7;
- g_b + g_s: 396.665 cm^-1, energy not conserved (max|g_s - Sigma[chi0 g_s] - g_b| / max|g_b| = 0.97: in IPA the two
  couplings should coincide, the fully bare g_b is not the bare coupling of the model).

## cDFPT coupling (`cdfpt/`)

g^c = coupling screened by everything except the bands of the model, from constrained DFPT (Nomura and Arita,
PRB 92, 245108 (2015)) with the elphmod patch for QE 7.6 (`~/codes/q-e-EPW-6.1-cdfpt`, `cdfpt.f90` added to the CMake
of LR_Modules). ph.x at Gamma with `cdfpt_bnd = 7..17`: projwfc on the full 12x12 grid shows that the 11 states with
the largest Mo d + S p weight are exactly the bands 7-17 at every k (a run with `cdfpt_orb` and nosym gives the same
frequencies, but EPW rejects its representations). Then EPW on the same `.ukk` with the constrained dvscf.
`mos2_cdfpt.dyn1` is the cDFPT dynamical matrix (bare lattice of the model; it violates the acoustic sum rule a lot:
acoustic modes at 218 and 424 cm^-1). max|g^c| = 0.39 (g_s 0.098, g_ion 6.15).

Check (IPA, g^c + g_s, static reference): K_BO - Pi with Pi = Tr[g^c chi0 g_s] gives E'' 431.5 and E' 521.5 cm^-1,
against 431.65 and 521.97 of the cDFPT dynamical matrix (A1' and A2'' are mixed with the z acoustic mode there).
