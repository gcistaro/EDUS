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
