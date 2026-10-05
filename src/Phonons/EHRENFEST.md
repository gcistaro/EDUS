# Ehrenfest dynamics of phonons in EDU


Coherent (classical) lattice dynamics coupled to the density matrix of EDUS, at the mean-field
(Ehrenfest) level. Atomic units (ħ = e = m_e = 1) unless stated otherwise.

## 1. Hamiltonian

Electrons in the Wannier basis of EDUS, phonons in normal modes ν, linear electron-phonon coupling:

$$
H = \sum_{k} c^\dagger_{k} H_e(k,t) c_{k}
  + \sum_{q\nu} \omega_{q\nu}\Big(b^\dagger_{q\nu} b_{q\nu} + \tfrac12\Big)
  + \frac{1}{\sqrt{N}} \sum_{k q \nu} \sum_{mn} g_{mn\nu}(k,q)\, c^\dagger_{m,k+q} c_{n,k}\,\big(b_{q\nu} + b^\dagger_{-q\nu}\big)
$$

- $H_e(k,t) = H_0(k) + \mathbf E(t)\cdot\mathbf r(k) + \Sigma[\rho](k)$ is the Hamiltonian EDUS already propagates;
- $g_{mn\nu}(k,q) = \langle m\,k+q|\,\partial_{q\nu}V_{\rm ion}\,|n\,k\rangle$ with the zero-point length
  $\ell_{q\nu} = \sqrt{1/(2\omega_{q\nu})}$ included (EPW convention): $g$ is an energy.
  **$g$ must not contain the screening of the electrons of the model** (section 3.1): EDUS generates it again
  through $\Sigma[\Delta\rho]$. Below $g^{\rm b}$ ("bare") means this coupling, screened only by what is outside the model;
- $N$ = number of unit cells = number of k points of the grid.

## 2. Mean field: only q = 0 is driven

In Ehrenfest the phonon operators are replaced by their expectation values
$\beta_{q\nu}(t) = \langle b_{q\nu}\rangle$. From $i\,\dot b = [b, H]$:

$$
i\,\dot\beta_{q\nu} = \omega_{q\nu}\,\beta_{q\nu}
  + \frac{1}{\sqrt N}\sum_{k,mn} g_{mn\nu}(k,-q)\,\langle c^\dagger_{m,k-q}\, c_{n,k}\rangle
$$

The force on the mode $q$ is given by the electronic coherence **between k and k − q**.
EDUS propagates only $\rho_{nm}(k) = \langle c^\dagger_{m,k} c_{n,k}\rangle$ (diagonal in k), which is exact as long
as the system is translationally invariant. A laser in the dipole approximation (q = 0) and the
displacement of q = 0 modes preserve translational invariance, so if $\beta_{q\neq0}(0) = 0$:

- $\langle c^\dagger_{k-q} c_k\rangle = 0$ at all times for $q \neq 0$,
- $\beta_{q\neq0}(t) = 0$ at all times.

**Only the Γ modes are coherently excited** (displacive excitation / impulsive stimulated Raman
scattering: coherent optical phonons). $q\neq0$ phonons need either a supercell (ρ non diagonal in k)
or a theory beyond mean field (phonon populations, collision integrals): see sections 7 and 8.

Consequence for the code: the coupling needed is $g_{mn\nu}(k) \equiv g_{mn\nu}(k, q=0)$,
an array of size $N_k \times N_{bands}^2 \times N_{modes}$, not the full $(k,q)$ one.

## 3. Equations of motion

Use for each Γ mode the real dimensionless coordinate (per unit cell)

$$
x_\nu = \frac{\beta_\nu + \beta_\nu^*}{\sqrt N},
\qquad \text{physical displacement: } u_{\kappa\alpha} = \sum_\nu \ell_\nu\, \frac{e_{\kappa\alpha,\nu}}{\sqrt{M_\kappa}}\, x_\nu
$$

($e_{\kappa\alpha,\nu}$ eigenvector of the dynamical matrix, $M_\kappa$ mass of atom κ).
The electrons feel the displaced lattice through

$$
H(k,t) = H_e(k,t) + \sum_\nu x_\nu(t)\, g_\nu(k)
$$

and the lattice feels the electrons through

$$
\ddot x_\nu = -\sum_{\nu'}\bar K^0_{\nu\nu'}\, x_{\nu'} \;-\; 2\omega_\nu\, F_\nu(t) \;-\; 2\gamma_\nu\,\dot x_\nu ,
\qquad
F_\nu(t) = \frac{s}{N}\sum_k \mathrm{Tr}\big[g_\nu(k)\,\big(\rho(k,t)-\rho_0(k)\big)\big]
$$

with $\bar K^0_{\nu\nu'} = (\ell_{\nu'}/\ell_\nu)\,K^0_{\nu\nu'}$ and $K^0$ the **bare** force constants
(clamped electrons) in the basis of the modes, and $g_\nu$ the **bare** coupling: see section 3.1.
The equation is second order in time; only the excited part $\Delta\rho=\rho-\rho_0$ exerts a force.

- $s$ = spin degeneracy of the bands (2 if the tight-binding is spinless, 1 with spin-orbit),
  the same factor used for the Hartree term;
- $\rho_0$ = `System::DM0()`: at equilibrium the forces vanish because the DFT geometry is relaxed,
  so only $\Delta\rho = \rho-\rho_0$ exerts a force (same logic of $\Sigma[\rho-\rho_0]$ in MeanField);
- $\gamma_\nu$ = optional phenomenological damping (anharmonic decay, not contained in the model);
- initial conditions: $x_\nu(0) = 0$, $\dot x_\nu(0) = 0$ (classical lattice at rest, no zero-point motion).

Derivation of the factor $2\omega$: with $Q_\nu = \ell_\nu x_\nu$ the mass-weighted coordinate,
the per-cell energy is $\tfrac12\dot Q^2 + \tfrac12 Q K^0 Q + \sum_\nu x_\nu\,\tfrac{s}{N}\sum_k\mathrm{Tr}[g_\nu\,\Delta\rho]$,
so $\ddot Q_\nu = -\sum_{\nu'}K^0_{\nu\nu'} Q_{\nu'} - \tfrac{1}{\ell_\nu}F_\nu$ and
$\ddot x_\nu = \ddot Q_\nu/\ell_\nu = -\sum_{\nu'}\bar K^0_{\nu\nu'} x_{\nu'} - F_\nu/\ell_\nu^2$, with $1/\ell_\nu^2 = 2\omega_\nu$.
Here $\ell_\nu$ (built with the BO frequency) is only a unit of length: the dynamics is fixed by $K^0$.

### 3.1 Bare vs screened quantities (important)

Reference: Perfetto, Stefanucci et al., *The first principles equation for coherent phonons*,
arXiv:2502.06368 (Eq. 27, first-principles Ehrenfest); PRX 13, 031026 (2023).

EDUS propagates the electrons with the mean-field $\Sigma[\Delta\rho]$ (Hartree + SEX), so the electronic
screening is generated **dynamically**. In the static limit the model response is $\Delta\rho = \chi\, g^{\rm b} x$
($\chi$ the interacting static response of the model), and

- the electrons feel $g^{\rm b}x + \Sigma^{\rm H}[\Delta\rho] = g^{\rm s}x$, with $g^{\rm s}=\varepsilon^{-1}g^{\rm b}$ the screened (DFPT/EPW) coupling;
- the lattice feels $K^0 + \Pi$, with $\Pi_{\nu\nu'} = \frac{s}{N}\sum_{kk'}\mathrm{Tr}[g^{\rm b\,\dagger}_\nu\,\chi\,g^{\rm b}_{\nu'}]\le 0$,
  i.e. the Born-Oppenheimer force constants $K^{\rm BO} = K^0 + \Pi$.

Therefore **bare $K^0$, bare $g^{\rm b}$ and $\Delta\rho=\rho-\rho_0$** is the consistent choice. Using the
BO frequencies $\omega_\nu^2$ and/or the screened EPW coupling together with the dynamical $\Sigma[\Delta\rho]$
counts twice the screening of $g$ and/or the softening of the frequencies.

Practical choice of $K^0$: $K^0 \equiv K^{\rm BO} - \Pi^{\rm model}$, with $\Pi^{\rm model}$ computed with the same
electronic model propagated by EDUS ($H_0$, Hartree, $W$), so that the static limit reproduces the DFPT phonons exactly
(section 3.3 gives the formulas as implemented).

The consistent alternative with renormalized quantities is the quasi-phonon equation (Eq. 50 of arXiv:2502.06368):
frequencies $\Omega$, coupling $\tilde g^{\rm s}$ and only the nonlinear part $\Delta\rho^{(r)}$ of the density matrix.
Born effective charges (section 8): with dynamical screening use the bare (ionic) charges.

### 3.2 Which "bare" coupling: unscreening within the model

"Bare" must be understood relative to the model. The fully bare coupling ($\partial V_{\rm ion}$ only, all the
Hxc response removed) is **wrong** for EDUS: it removes also the screening of the bands outside the Wannier model
(core-like and high-energy bands, which in hBN are most of the dielectric response), and the model cannot generate it
again, so the electrons would feel a coupling much larger than the physical one. The right $g^{\rm b}$ is screened by
everything except the electrons of the model, the same idea of the constrained RPA/DFPT (Nomura and Arita,
PRB 92, 245108 (2015)).

This $g^{\rm b}$ is obtained from the screened coupling of a standard EPW run, $g^{\rm s}$. In the static limit the
electrons of the model respond with $\delta\rho = \chi_0 V$ to the total perturbation $V = g^{\rm b} + \Sigma[\delta\rho]$.
Asking that $V$ is the DFPT coupling, $V = g^{\rm s}$, gives $\delta\rho = \chi_0 g^{\rm s}$ and, exactly and without
iterations ($\Sigma$ is linear in $\delta\rho$),

$$ g^{\rm b} = g^{\rm s} - \Sigma\big[\chi_0\, g^{\rm s}\big]. $$

With this $g^{\rm b}$ the static limit gives exactly the coupling of DFPT to the electrons and, with the static adiabatic
reference, exactly the force constants of ph.x to the lattice. The approximations: the screening of the bands outside
the model is static and the one of DFT (it is at high energy), and the response inside the model is the one of its
mean field (Hartree + SEX with the model $W$), not the Hxc of DFT. Without the mean field (IPA) $g^{\rm b} = g^{\rm s}$.
This is the option `coupling = "screened"` (default); `"bare"` uses the files as they are (synthetic tests).

Note on the route "EPW with `lnoloc`": stock ph.x refuses `lnoloc` for phonons (`phq_readin.f90`:
"only dielectric constant with lrpa or lnoloc"). One could zero the induced potential read by EPW (`dvscf_read`),
since EPW recomputes $\partial V_{\rm ion}$ on the fly (`dvqpsi_us3`), but this drops also the nonlinear core
correction (EPW puts it in `dvscf`) and, above all, gives the fully bare coupling discussed above.

The electronic equation is the one of EDUS with the extra term in the Hamiltonian:

$$
i\,\dot\rho(k) = \big[H_e(k,t) + \textstyle\sum_\nu x_\nu(t)\,g_\nu(k),\ \rho(k)\big] + \text{(terms already in EDUS)}
$$

### Force in R space

With the EDUS convention $O(k)=\sum_R e^{ik\cdot R}O(R)$ and $g_\nu(-R) = g_\nu(R)^\dagger$ (hermiticity):

$$
F_\nu = s\sum_R \sum_{mn} g_{\nu,mn}(R)\;\Delta\rho_{mn}(R)^* \qquad (\text{real})
$$

Each rank sums its local R (or k) points, then `MPI_Allreduce`. The extra term in the Hamiltonian is
$H(R) \mathrel{+}= \sum_\nu x_\nu\, g_\nu(R)$, local in R like $\Sigma$.

### Gauge

EDUS propagates in the Wannier gauge: $g_\nu(k)$ must be in the same Wannier basis of $H_0$.
From Bloch-gauge couplings: $g^W_\nu(k) = U(k)\, g^B_\nu(k)\, U^\dagger(k)$, with the $U(k)$ of
`System::bandstructure()` (and the same phase convention of the code that produced $g^B$).

### 3.3 What the code computes before the propagation

This section writes sections 3.1–3.2 in the variables of the code (`Lattice::initialize`), at q = 0. Equation numbers
"BO (n)" refer to the notes *Born–Oppenheimer density response* (Born_Oppenheimer.pdf), whose symbols are used here.

**Variables.** Cartesian displacements $u_\mu$, $\mu = 3\kappa+\alpha$ (bohr); coupling $g_\mu(k) = \partial H(k)/\partial u_\mu$
(Ha/bohr) in the Wannier gauge; $s$ = spin degeneracy, $N$ = number of k points. $U(k)$ has the eigenvectors of $H_0(k)$ as
columns, so that an operator goes to the Bloch gauge as $O^B = U^\dagger O\, U$ and back as $O = U O^B U^\dagger$ (BO (16), (20)).
$\mathrm{Tr}[AB] = \sum_{mn}A_{mn}B_{nm}$.

**Static response function** (`ModelResponse::chi0`). For an operator $V$ at q = 0,

$$ [\chi_0 V]^B_{nm} = \frac{f_n - f_m}{\varepsilon_n - \varepsilon_m}\,V^B_{nm}, \qquad
   \chi_0 V = U\big(D \circ U^\dagger V U\big)U^\dagger $$

with $f$ the occupations and $\varepsilon$ the bands of $H_0$, $D_{nm}$ the factor above ($\circ$ = element by element), set to 0 for
pairs with the same occupation and for degenerate pairs (they would need the intraband term of a metal). At q = 0 the factor
is symmetric in $n \leftrightarrow m$, so the index order of BO (13) does not matter.

**Self energy** (`MeanField::self_energy`, linear in $\delta\rho$): Hartree on the diagonal at R = 0 and SEX,

$$ \Sigma[\delta\rho]_{ij}(R) = \delta_{ij}\,\delta_{R0}\sum_{j'} V^{\rm H}_{ij'}\,\delta\rho_{j'j'}(0) \;-\; W_{ij}(R)\,\delta\rho_{ij}(R). $$

It plays the role of the kernel $K^W$ of BO (27)–(29), restricted to the Wannier functions of the model.

The steps, in the order of `Lattice::initialize`:

1. **Screened coupling** $g^{\rm s}_\mu$ (`read_coupling`): EPW, q = 0, moved to the k grid of the simulation. It is the matrix
   element of $\partial V_{\rm SCF}/\partial u_\mu$ (BO (6), (14)) in the Wannier gauge, i.e. $g_{\rm wan}$ of BO (20).

2. **Force constants** $K^{\rm BO}$ and masses from ph.x, with the simple acoustic sum rule on $K^{\rm BO}$ (below).

3. **Rigid translation removed from g** (`remove_translation_from_coupling`, if `acoustic_sum_rule`):

   $$ g_{\kappa\alpha} \;\to\; g_{\kappa\alpha} - \frac{M_\kappa}{M_{\rm tot}}\sum_{\kappa'} g_{\kappa'\alpha}. $$

   *Acoustic sum rule.* A rigid translation $u_{\kappa\alpha} = t_\alpha$ for every κ does not change the energy, so
   $\sum_{\kappa'} K_{\kappa\alpha,\kappa'\beta} = 0$ (three acoustic modes at zero frequency at Γ), the total force on the crystal
   vanishes, $\sum_\kappa F_{\kappa\alpha} = 0$, and a translation of atoms and electrons together leaves $H$ unchanged.
   In the fixed basis of the Wannier functions the last one fails: translating all the atoms translates the potential, so
   $\sum_\kappa g_{\kappa\alpha} = \langle w_m|-\partial_\alpha V|w_n\rangle \neq 0$. In DFPT the term
   $\langle\partial^2 V/\partial u\,\partial u\rangle$ compensates it and $K^{\rm BO}$ satisfies the rule; that term is not in the model, so
   the $\Pi$ below, built from first-order couplings only, would violate it (hBN from EPW: $\max|\sum_\kappa g| = 2\max|g|$,
   acoustic modes of $K^{\rm BO}-\Pi$ at 370 cm⁻¹, center of mass moving, energy not conserved).
   With the weights $M_\kappa/M_{\rm tot}$, for every displacement with the center of mass at rest ($\sum_\kappa M_\kappa u_\kappa = 0$,
   e.g. the optical modes)

   $$ u\cdot g' = u\cdot g - \frac{\sum_\kappa M_\kappa u_\kappa}{M_{\rm tot}}\cdot\sum_{\kappa'} g_{\kappa'} = u\cdot g, $$

   so the optical physics is unchanged and only the translation is decoupled. Afterwards $\sum_\kappa g_{\kappa\alpha} = 0$: the
   force on the center of mass vanishes and $\Pi$ satisfies the sum rule by itself (checked in the recap), with $H$, the
   force and $\Pi$ built from the same coupling. The correction must be on $g$: imposing the sum rule on $\Pi$ alone
   breaks the static limit, because the propagated electrons generate the uncorrected $\Pi$. The step is linear and done
   before the unscreening, so every quantity below is built from the corrected coupling.

4. **Born–Oppenheimer response and unscreening** (`unscreen_coupling`, if `coupling = "screened"` and the mean field is on).
   The static response per unit displacement is (BO (19) with (20), q = 0)

   $$ \tilde g_\mu = \chi_0\, g^{\rm s}_\mu, \qquad \Delta\rho^{\rm BO}(u) = \sum_\mu u_\mu\,\tilde g_\mu \quad \text{(BO (18))}, $$

   $\chi_0$ acting on the self-consistent perturbation $g^{\rm s}$, as in BO (2)–(9). The electrons of the model generate
   $\Sigma[\Delta\rho]$ during the propagation, so the coupling in $H$ must not contain it (BO (28)):

   $$ g^{\rm b}_\mu = g^{\rm s}_\mu - \Sigma[\tilde g_\mu]. $$

   Then $g^{\rm b}_\mu + \Sigma[\tilde g_\mu] = g^{\rm s}_\mu$, hence $\chi_0\big(g^{\rm b}_\mu + \Sigma[\tilde g_\mu]\big) = \tilde g_\mu$: $\tilde g_\mu$ is exactly the
   self-consistent static response of the model driven by $g^{\rm b}$, and in the static limit the electrons feel $g^{\rm s}$.
   $\tilde g$ is kept (`gtilde_`) for the next step. Without the mean field $g^{\rm b} = g^{\rm s}$; with `coupling = "bare"` the
   files are taken as $g^{\rm b}$ and there is no $\tilde g$ yet.

5. **Electronic force constants** (`static_response`, only with `adiabatic_reference = "static"`):

   $$ \Pi_{\mu\nu} = \frac{s}{N}\sum_k \mathrm{Tr}\big[g^{\rm b}_\mu(k)\,\delta\rho_\nu(k)\big], \qquad \delta\rho_\nu = \chi_0\big(g^{\rm b}_\nu + \Sigma[\delta\rho_\nu]\big), $$

   symmetrized as $(\Pi+\Pi^\dagger)/2$. $\delta\rho_\nu = \tilde g_\nu$ from step 4 when it exists (no iterations); otherwise the
   linear fixed point is solved by Anderson mixing (one step without the mean field). $\Pi = \frac{s}{N}\sum_k\mathrm{Tr}[g^{\rm b}\chi_0 g^{\rm s}]$
   is the $\Pi$ of section 3.1, $\mathrm{Tr}[g^{\rm b}\chi\, g^{\rm b}]$, since $\chi g^{\rm b} = \chi_0 g^{\rm s}$. The force constants of the dynamics are

   $$ K^0 = K^{\rm BO} - \Pi . $$

During the propagation (sections 3 and 5) the Hamiltonian is $H = H_e + \sum_\mu u_\mu g^{\rm b}_\mu$ and the force
$F_\mu[X] = \frac{s}{N}\sum_k \mathrm{Tr}[g^{\rm b}_\mu X]$ is the Hellmann–Feynman derivative of the energy of the model, so

$$ M_\mu \ddot u_\mu = -\sum_\nu K^0_{\mu\nu}u_\nu - F_\mu[\rho-\rho_0]
   \;=\; -\sum_\nu K^{\rm BO}_{\mu\nu}u_\nu - F_\mu\big[\rho-\rho_0-\Delta\rho^{\rm BO}(u)\big], $$

using $F_\mu[\Delta\rho^{\rm BO}(u)] = \sum_\nu \Pi_{\mu\nu}u_\nu$: only the part of $\rho$ beyond the adiabatic response pushes the atoms,
with the force constants of ph.x. In the static limit $\rho-\rho_0 \to \Delta\rho^{\rm BO}(u)$ and the frequencies are those of ph.x.

Check on hBN from QE/EPW (2 $p_z$ Wannier functions, no mean field, `static`): the static limit gives E′ at
1349.59 cm⁻¹ against 1349.53 of ph.x, the energy is conserved to 3·10⁻⁹ and the center of mass stays at rest (5·10⁻¹⁴ bohr).
With the mean field (Hartree + SEX, ε = 1) the DFT $H_0$ of that model is excitonically unstable (the SEX binding exceeds
the DFT gap of ~4.7 eV): ρ₀ is not a minimum of the mean-field energy and noise grows exponentially even without phonons.
A quasiparticle gap in $H_0$ (scissor or KCW) or ε > 1 is needed there.

## 4. Where the coupling comes from

Both routes below give the screened coupling $g^{\rm s}$ of DFT, from which EDUS removes the screening of the
model at the beginning (section 3.2).

1. **EPW**: `epmatwp` gives $g$ in the Wannier representation $g(R_e, R_p)$; for q = 0, sum over $R_p$
   and rotate to the mode basis. Needs the same Wannier functions of the tb model.
   EPW uses `dvscf`, i.e. the **screened** coupling $g^{\rm s}$: use `coupling = "screened"`.
   Workflow tested on hBN (EPW 6.1):
   1. `pw.x` scf, `ph.x` at Γ only with `fildvscf`, then `pp.py` to collect `save/`;
   2. `pw.x` scf and nscf on the full k grid as an explicit list, **without `nosym`/`noinv`** (EPW 6.1 allocates
      `gmapsym` with the nsym of the nscf and then `find_sym` raises it: segfault in `gmap_sym`);
   3. `epw.x` with `nq1 = nq2 = nq3 = 1` (only Γ is needed), `wannierize = .true.`, `epwwrite = .true.`,
      `lpolar = .false.` and `wdata(…) = 'write_tb = .true.'`: the `prefix_tb.dat` of the same run is the `tb_file` of
      EDUS, so that EPW and EDUS use the same Wannier functions (`max|H_EPW - H_tb|` ~ 1e-7 eV in the recap).
      An error in `efermig` at the end (fine grid of one point) comes after the Wannier files are written;
   4. EDUS with `epw_directory` = the EPW directory and `dyn_file` = the dynamical matrix of ph.x at Γ.
2. **Frozen phonon** (simplest with the KCW/wannier90 workflow): displace the atoms along
   $\pm\delta$ of mode ν, recompute the tight-binding $H^W(R)$ with the same projections, and
   $$ g_\nu(R) = \ell_\nu\,\frac{H^W(R;+\delta Q_\nu) - H^W(R;-\delta Q_\nu)}{2\,\delta Q_\nu} $$
   The Wannier gauge must be the same in the three calculations (same projections, no
   disentanglement changes), otherwise the finite difference mixes gauge and physics.
   A self-consistent frozen phonon gives the screened $g^{\rm s}$, as EPW (`coupling = "screened"`).

## 5. Implementation in EDUS

What is implemented (q = Γ only, RK and AB solvers, CPU):

- **Coordinates**: the cartesian displacements $u_\mu$, $\mu = 3\kappa+\alpha$ (bohr), instead of the normal modes.
  With $g_\mu = \partial H/\partial u_\mu$ (Ha/bohr, the coupling of EPW) and the force constants $K_{\mu\nu}$ (Ha/bohr²)
  $$ M_\mu\ddot u_\mu = -\sum_\nu K_{\mu\nu}u_\nu - F_\mu - \frac{2M_\mu}{\tau}\dot u_\mu,\qquad
     F_\mu = \frac{s}{N}\sum_k \mathrm{Tr}[g_\mu(k)\Delta\rho(k)] = s\sum_R\sum_{mn}g_{\mu,mn}(R)\,\Delta\rho_{mn}(R)^* $$
  equivalent to section 3 (no $\ell_\nu$ and no diagonalization of the dynamical matrix are needed; the modes are
  only printed in the recap). $\tau$ = `damping_time`.
- **Acoustic sum rule** (`acoustic_sum_rule`, default true): the simple sum rule on $K$, and the rigid translation
  removed from the coupling (section 3.3, step 3).
- **Data**: `phonon::read_epw` / `read_epmatwp` (`EPW.hpp`) read `epwdata.fmt`, `wigner.fmt`, `crystal.fmt` and
  `prefix.epmatwp` (checked on the source of EPW 6.1 and on the Pb tutorial); $g_\mu(R_e; q=0) = \sum_{R_g} g_\mu(R_e,R_g)$
  is moved to the grid of the simulation like $H_0$ (dft on the k grid, fft to R). The Hamiltonian of EPW is compared
  with the one of `tb_file` (`max|H_EPW - H_tb|` in the recap): they must be the same Wannier functions.
  Masses and $K$ from the dynamical matrix of ph.x at Γ (`DynamicalMatrix.hpp`, text format).
- **Bare vs BO force constants** (sections 3.1 and 3.3): ph.x gives $K^{\rm BO} = K^0 + \Pi$, and with $K^{\rm BO}$ only
  $\rho-\rho_0-\Delta\rho^{\rm BO}(u)$ must push the atoms. Option `adiabatic_reference`:
  - `none`: $\Delta\rho^{\rm BO}=0$ with $K^{\rm BO}$ (the response is counted twice);
  - `static` (default): $\Delta\rho^{\rm BO}(u) = \sum_\mu u_\mu\tilde g_\mu$ with the static response of the model (section 3.3, steps 4–5),
    implemented as $K^0 = K^{\rm BO}-\Pi$ with the whole $\rho-\rho_0$ in the force (identical). The lattice keeps the
    non-adiabatic part of the response: $M\omega^2 = K^{\rm BO}+\Pi(\omega)-\Pi(0)$, the non-adiabatic phonons of
    Lazzeri and Mauri, PRL 97, 266407 (2006), and Calandra, Profeta and Mauri, PRB 82, 165111 (2010);
  - `dynamic`: $\rho_{\rm BO}$ is propagated with the same equation of $\rho$ (mean field, $u(t)$, decay), without the laser, and
    the force is that of $\rho-\rho_{\rm BO}$. Without laser $\rho=\rho_{\rm BO}$ and the lattice oscillates exactly with $K^{\rm BO}$;
    the non-adiabatic part of the response is removed too (it would matter in metals: damping by electron-hole pairs).
    This correction is physical, so `dynamic` is not the Born–Oppenheimer reference of the theory: with excited
    carriers (occupations f) it gives $M\omega^2 = K^{\rm BO}+\Pi_f(\omega)-\Pi_0(\omega)$ instead of $K^{\rm BO}+\Pi_f(\omega)-\Pi_0(0)$.
    The conserved energy is $E[\rho]-E[\rho_{\rm BO}]+E_{\rm lattice}(K^{\rm BO})$, with $E[\rho]$ the electronic energy including
    the coupling $u\cdot F[\rho]$; it changes only because of the work of the field.

  Check of the three options on a two-level model independent of EDUS ($H_0 + u\,g$, gap 1, $\omega_{\rm BO}=0.2$,
  RK4), frequencies against the roots of $M\omega^2 = K_{\rm eff}(\omega)$:

  | reference | $K_{\rm eff}(\omega)$ | f = 0: simulated / predicted | f = 0.1: simulated / predicted |
  |---|---|---|---|
  | `none` | $K^{\rm BO}+\Pi_f(\omega)$ | 0.16385 / 0.16385 | 0.1713 / 0.1716 |
  | `static` | $K^{\rm BO}-\Pi_0(0)+\Pi_f(\omega)$ | 0.19868 / 0.19868 | 0.2050 / 0.2052 |
  | `dynamic` | $K^{\rm BO}+\Pi_f(\omega)-\Pi_0(\omega)$ | 0.20000 / 0.20000 | 0.2066 / 0.2066 |

  (f = 0.1: mean displacement −0.07, small non-linear corrections.) In an insulator $\Pi(\omega)-\Pi(0)<0$: `static`
  softens the phonon by $\sim(\omega/{\rm gap})^2$, as seen on the synthetic hBN (1369.957 against 1370). On the real hBN
  it gives 1349.59 against 1349.53 of ph.x, a hardening of $4\cdot10^{-5}$. On MoS2 from EPW the same hardening is
  large and depends on the amplitude: A1' at $u_0$ = 0.005 Å per S is 2.7 % above ph.x, and
  $\omega/\omega_{\rm ph.x}-1 = -2.0\cdot10^{-3} + 1.15\cdot10^{3}\,u_0^2$ ($u_0$ in Å, from 0.0002 to 0.005 Å; the same with
  dt = 0.02 and 0.04 fs). The linear limit is the non-adiabatic softening; the $u_0^2$ term is the anharmonicity of the
  electronic energy of the model, amplified because for A1' $\Pi = -6.6\,K^{\rm BO}$ (the bare $K^0$ alone gives
  1092 cm⁻¹). With `dynamic` the frequency is exact at any amplitude. Use small displacements to check the static limit.

  Not included: the second-order coupling $\frac12 u_\mu u_\nu\,\mathrm{Tr}[\partial^2H/\partial u_\mu\partial u_\nu\,(\rho-\rho_0)]$
  (Debye–Waller-like), which also changes the frequency with excited carriers; the ions are classical at q = Γ
  (coherent phonons: DECP and ISRS, no incoherent emission of phonons).
- **State**: `EhrenfestState` = (ρ, `phonon::Coordinates`), with `axpby` and `make_workspace`: RK/AB advance ρ and the
  lattice together, so the Hamiltonian at each stage sees $u$ at the same time. `electron::State::DensityMatrix()` is
  the ρ inside it. Without phonons the propagation is the one of before (`DESolver<Operator>`, also Magnus).
- **Propagator**: `build_hamiltonian` adds $\sum_\mu u_\mu g_\mu(R)$ to H(R) after the self energy (before the Peierls phase)
  and computes $F_\mu$ in R from the physical ρ (Peierls phase removed), reduced over the ranks.
- **Output** (`Lattice.txt`): $E_{\rm lattice}$, $u\cdot F$, $u_\mu$, $\dot u_\mu$, $F_\mu$. The velocity includes
  $\nabla_k(u\cdot g) + i[u\cdot g, r]$ and `Energy.txt` includes $E_{\rm ph} = (E_{\rm lattice} + u\cdot F)/s$, so that
  $E - E(0) = W$ holds also with phonons.
- **Tests** (`tb_models/hBN_synthetic_epw`, written by `utility/make_synthetic_epw.py`: hopping $t(d)=t_0e^{-\beta(d/d_0-1)}$,
  spring constants for 1370 and 800 cm⁻¹): `hBN_phonons_static` and `hBN_HSEX_phonons_static` (no laser, static
  reference: the static limit gives back the frequency of the dynamical matrix in IPA and with the mean field, energy
  conserved) and `hBN_HSEX_phonons` (laser + mean field, dynamic reference, energy balance). The synthetic coupling has
  $\sum_\kappa g_\kappa = 0$ by construction (it depends only on bond lengths), so it does not test step 3 of section 3.3;
  the QE → EPW → EDUS run on hBN of section 4 does.
  Real EPW data: `tb_models/MoS2_PBE_epw` (monolayer MoS2, PBE, 11 Wannier functions, see its README) for
  `MoS2_epw_phonons_dynamic` (A1' displaced, no laser, dynamic reference: frequency of ph.x within 1e-8) and
  `MoS2_epw_phonons_linear`/`_circular` (pump at 3 eV, static reference). `ci-test/check_symmetry.py` (run from
  `ci-test/checks/<test>.sh`) classifies the modes by their parity under $\sigma_h$ and checks D3h: the odd modes
  E'' and A2'' stay at the noise (~1e-10 of A1'), A1' (DECP) is the most excited mode, and with the circular pump
  $E(E')/E(A1') \approx 2\cdot10^{-5}$ against 0.14 with the linear one.

Still to do: q ≠ 0 (section 7), Born effective charges
(direct IR driving, section 8), gpu, the Magnus time stepper with the lattice, the dynamical matrix in xml format.

## 6. Tests

1. $g = 0$: results identical to the present EDUS.
2. No laser: $x$ stays 0 (forces vanish because of $\rho-\rho_0$).
3. Energy conservation without laser and damping, starting from $x(0)\neq0$:
   $E = \frac{s}{N}\sum_k \mathrm{Tr}[(H_0 + \Sigma' + \sum_\nu x_\nu g_\nu)\rho] - \sum_\nu x_\nu\frac{s}{N}\sum_k\mathrm{Tr}[g_\nu\rho_0] + E_{ph}$
   is constant ($\Sigma'$: the mean-field energy with its double-counting correction).
4. Displacive limit: a constant $\Delta\rho$ gives $x(t) = \bar x\,(1-\cos\omega t)$ with
   $\bar x = -2\omega F/\bar K^0$ (single mode) (cosine-like oscillation around a shifted equilibrium, as in DECP).
5. Linear response: a weak pulse resonant with a Raman-active mode excites a sine-like oscillation (ISRS).

## 7. General formulation in k and q

Sections 2–6 keep only q = 0. Here the same mean-field equations for all q, coupled to the
electronic coherences between different k points.

### Definitions

$$
\rho_{nm}(k,q) = \langle c^\dagger_{m,k}\, c_{n,k+q}\rangle ,
\qquad \rho(k,0) = \rho(k)\ \text{(the density matrix of EDUS)},
\qquad \rho(k,-q) = \rho(k-q,q)^\dagger
$$

$$
\beta_{q\nu} = \langle b_{q\nu}\rangle, \qquad u_{q\nu} = \beta_{q\nu} + \beta^*_{-q\nu}, \qquad u_{-q\nu} = u^*_{q\nu}
$$

Hermiticity of $H_{ep}$ gives $g_\nu(k+q,-q) = g_\nu(k,q)^\dagger$ (matrices in the band indices).
The q grid must be the k grid (differences of k points), so that $k+q$ is always on the grid.

### Phonons

From $i\dot b_{q\nu} = [b_{q\nu}, H]$, with $\Delta\rho(k,q) = \rho(k,q) - \delta_{q0}\,\rho_0(k)$:

$$
i\,\dot\beta_{q\nu} = (\omega_{q\nu} - i\gamma_{q\nu})\,\beta_{q\nu} + f_{q\nu}(t),
\qquad
f_{q\nu}(t) = \frac{s}{\sqrt N}\sum_k \mathrm{Tr}\big[g_\nu(k,q)^\dagger\,\Delta\rho(k,q)\big]
$$

($\mathrm{Tr}[AB] = \sum_{mn}A_{mn}B_{nm}$). Using $f^*_{-q\nu} = f_{q\nu}$ (from the hermiticity of ρ and g),
the equivalent second-order equation for the displacement is

$$
\ddot u_{q\nu} = -\omega_{q\nu}^2\, u_{q\nu} - 2\omega_{q\nu}\, f_{q\nu}(t) \qquad (\gamma = 0)
$$

With the bare quantities of section 3.1 ($g^{\rm b}$ in $f$, bare force constants) the harmonic term becomes
$-\sum_{\nu'}\bar K^0_{\nu\nu'}(q)\,u_{q\nu'}$ with $\bar K^0_{\nu\nu'}(q) = (\ell_{q\nu'}/\ell_{q\nu})K^0_{\nu\nu'}(q)$.
Since $K^0$ is not diagonal in the BO modes, the first-order form in $\beta$ holds only if the $\omega_{q\nu}$ in it
are the eigenvalues of $K^0$; the second-order form in $u$ (with $u$, $\dot u$ propagated) is the general one.

For q = 0 it reduces to section 3 with $x_\nu = u_{0\nu}/\sqrt N$.
The first-order complex form (β) is the natural one for DESolver: complex arrays like ρ, one
equation per $(q,\nu)$, and $u_{q\nu}$ is rebuilt from $\beta_{q\nu}$ and $\beta_{-q\nu}$.

### Electrons

The phonon field makes the mean-field Hamiltonian non diagonal in k:

$$
h(k+q', k) = \delta_{q'0}\,H_e(k,t) + \frac{1}{\sqrt N}\sum_\nu u_{q'\nu}(t)\, g_\nu(k,q')
$$

and $i\dot{\mathcal P} = [h,\mathcal P]$ in the $(k,\text{band})$ space, with $\mathcal P(k+q,k) = \rho(k,q)$, gives

$$
i\,\dot\rho(k,q) = H_e(k+q)\,\rho(k,q) - \rho(k,q)\,H_e(k)
+ \frac{1}{\sqrt N}\sum_{q'\nu}\Big[u_{q'\nu}\,g_\nu(k+q-q',q')\,\rho(k,q-q') - u_{q'\nu}\,\rho(k+q',q-q')\,g_\nu(k,q')\Big]
$$

plus the terms of the laser and of the mean field already written for ρ(k,q).
The phonon term is a convolution in q: cost $\sim N_k N_q^2 N_{modes} N_b^3$ per evaluation.

### Energy (for tests)

$$
E = s\sum_{k}\mathrm{Tr}[H_0(k)\rho(k,0)] + \frac{s}{\sqrt N}\sum_{kq\nu}u^*_{q\nu}\,\mathrm{Tr}[g_\nu(k,q)^\dagger\Delta\rho(k,q)]
  + \sum_{q\nu}\omega_{q\nu}|\beta_{q\nu}|^2 + E_{MF}
$$

is conserved without laser and damping.

### Important: what the k,q extension changes

The equations are closed and exact at mean-field level, but the symmetry argument of section 2
still holds: if $\rho(k,q\neq0)=0$ and $\beta_{q\neq0}=0$ at t = 0 and the laser has q = 0, then
$\dot\rho(k,q\neq0) = 0$ and $\dot\beta_{q\neq0}=0$: the q ≠ 0 part stays zero. It moves only with a seed:

- **thermal/zero-point initial conditions**: $\beta_{q\nu}(0)$ sampled from the Wigner distribution
  of the harmonic oscillator at temperature T (Gaussian, $\langle|\beta|^2\rangle = n_B + \tfrac12$),
  averaging observables over many trajectories (multi-trajectory Ehrenfest);
- **inhomogeneous excitation**: a field with $q\neq0$ (transient grating, near fields);
- **beyond mean field**: phonon-assisted density matrices $\langle c^\dagger c\, b\rangle$ (cluster
  expansion), whose adiabatic elimination gives the scattering terms of the Boltzmann equation.

Memory: $\rho(k,q)$ and $g_\nu(k,q)$ have $N_k N_q N_b^2$ elements (per mode for g):
e.g. $25^2 \times 25^2 \times 34^2 \approx 4.5\cdot10^8$ complex numbers = 7 GB, to distribute over MPI.

## 8. What Ehrenfest does not describe

- **q ≠ 0 phonons and incoherent phonon populations**: no energy transfer to the full phonon bath,
  no thermalization, no hot-carrier relaxation by phonon emission. These need collision integrals
  (e.g. time-dependent Boltzmann for electrons and phonons) or a supercell.
- **Quantum lattice fluctuations**: no zero-point motion, no spontaneous emission of phonons
  (a classical lattice at rest cannot emit into a mode with $\beta=0$); multi-trajectory sampling of
  the initial $x,\dot x$ partially recovers them.
- **Nonlinear coupling** (Debye-Waller, $\propto x^2$) and **anharmonicity** of the lattice
  ($\gamma_\nu$ is only phenomenological).
- **Direct infrared driving** of polar modes (hBN, LiF, GaAs): the laser also pushes the ions through
  the Born effective charges $Z^*$ (energy $-\sum_\kappa Z^*_\kappa \mathbf E\cdot \mathbf u_\kappa$). It adds to the equation of motion
  $$ \ddot x_\nu \mathrel{+}= \frac{1}{\ell_\nu}\sum_{\kappa\alpha\beta} Z^*_{\kappa,\alpha\beta}\,E_\beta(t)\,\frac{e_{\kappa\alpha,\nu}}{\sqrt{M_\kappa}} $$
  (the $1/\ell_\nu = 2\omega_\nu\ell_\nu$ comes from the change of variable $Q_\nu = \ell_\nu x_\nu$, as for $F_\nu$).
  Only IR-active modes get it; Raman-active modes are driven by $F_\nu$ only.
