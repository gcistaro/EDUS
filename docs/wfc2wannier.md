# Wavefunctions from Quantum ESPRESSO in the Wannier gauge (`EDUS_wfc2wannier`)

`EDUS_wfc2wannier` reads the Kohn–Sham wavefunctions written by `pw.x` and rotates them with the
matrices of wannier90 (`seedname_u.mat`, `seedname_u_dis.mat`), giving

1. the Bloch functions in the Wannier gauge, in plane waves,
   $\tilde\psi_{n\mathbf k}(\mathbf G)=\sum_m c_{m\mathbf k}(\mathbf G)\,V_{mn}(\mathbf k)$, with
   $V(\mathbf k)=U^{\rm dis}(\mathbf k)\,U(\mathbf k)$;
2. the Wannier functions on a real-space supercell,
   $w_n(\mathbf r)=\frac1{N_k}\sum_{\mathbf k}\tilde\psi_{n\mathbf k}(\mathbf r)$.

These are the same functions whose matrix elements ($H$, $\mathbf r$) EDUS reads from `seedname_tb.dat`.

## Workflow

```bash
pw.x < scf.in > scf.out
pw.x < nscf.in > nscf.out          # full k grid of wannier90 (kmesh.pl), nosym/noinv
wannier90.x -pp seedname
pw2wannier90.x < pw2wan.in > pw2wan.out
wannier90.x seedname                # with  write_u_matrices = .true.  (and write_tb = .true. for EDUS)
EDUS_wfc2wannier input.json         # mpirun -n N ... distributes the k points
```

Requirements on the QE side: wavefunctions in the default binary format (`wfc<ik>.dat`; QE compiled with HDF5
writes `wfc<ik>.hdf5`, not supported yet). The `nscf` k points can be in any order and in any periodic image of
the wannier90 ones: each k point of `u.mat` is matched with the QE file with the same k (modulo a reciprocal
lattice vector, whose Miller indices are shifted accordingly).

## Input (`input.json`)

| Key | Default | Description |
|---|---|---|
| `qe_save_dir` | required | `<outdir>/<prefix>.save` |
| `seedname` | required | wannier90 seedname (with path): `_u.mat`, optional `_u_dis.mat` and `.eig` |
| `exclude_bands` | `[]` | as in the `.win` file (1-based QE band indices) |
| `dis_win_min`, `dis_win_max` | all bands | outer window of the `.win` file (eV). Needed, together with `seedname.eig`, only if the window does not contain all the bands |
| `spin` | `"none"` | `"up"`/`"down"` for LSDA (`wfcup<ik>.dat`/`wfcdw<ik>.dat`) |
| `fft_grid` | `[0,0,0]` | unit-cell grid for $w_n(\mathbf r)$; `0` = smallest grid containing all the G vectors. A smaller grid samples the functions at fewer points (like `reduce_unk`) |
| `supercell` | `[2,2,2]` | cells where $w_n(\mathbf r)$ is computed, as `wannier_plot_supercell`. At most the k grid (Born–von Kármán cell) |
| `wannier_list` | all | Wannier functions computed in real space (1-based) |
| `write_bloch` | `true` | write the rotated Bloch functions |
| `real_space` | `true` | compute $w_n(\mathbf r)$ |
| `write_xsf` | `true` | write the XSF files |
| `fix_phase` | `true` | global phase making $w_n(\mathbf r)$ as real as possible (as wannier90) |
| `output_dir` | `"wannier_wfc"` | output folder |

Why `.eig` and the outer window: the rows of `u_dis.mat` refer only to the bands inside the outer window
(packed, the others are zero). If the window contains all the bands nothing else is needed; otherwise the code
locates it from the energies in `seedname.eig`, as wannier90 does, and checks that it is consistent with the
non-zero rows of `u_dis.mat`.

## Output (`output_dir`)

| File | Content |
|---|---|
| `wfc<ik>.dat` | Bloch functions in the Wannier gauge, same format as QE (`nbnd` = `num_wann`), `ik` = k index of `u.mat`. Readable with the same reader (`QEWavefunction::read`) or any QE-aware tool |
| `wannier_<n>.xsf` | $w_n(\mathbf r)$ on the supercell (real part, bohr$^{-3/2}$; $\sqrt{\lvert w_\uparrow\rvert^2+\lvert w_\downarrow\rvert^2}$ for spinors), atoms from `data-file-schema.xml` |
| `wannier_centres.txt` | norm, centre and spread of each $w_n$ computed on the real-space grid |

The standard output also reports $\max\lvert V^\dagger V-1\rvert$ and the orthonormality of the rotated Bloch functions.

## Conventions and caveats

* QE coefficients are normalized in the unit cell, $\psi_{n\mathbf k}(\mathbf r)=\Omega^{-1/2}\sum_{\mathbf G}c_n(\mathbf G)e^{i(\mathbf k+\mathbf G)\cdot\mathbf r}$,
  so $w_n$ is normalized to 1 on the Born–von Kármán supercell.
* Ultrasoft/PAW pseudopotentials: only the smooth part of the wavefunctions is stored, orthonormal with the $S$ matrix.
  $\langle\tilde\psi_n|\tilde\psi_m\rangle\neq\delta_{nm}$ and the norm of $w_n$ is not 1 (the same happens in `wannier_plot`).
* Centres and spreads from the grid are approximate (grid spacing, finite supercell); the reference values are those in `seedname.wout`.
* `gamma_only` files are expanded to the full G sphere; noncollinear (`npol = 2`) is supported.

## Tests

`ci-test/wfc2wannier`: synthetic QE/wannier90 files built from analytic Gaussian Wannier functions
(plain, disentanglement with excluded bands and a k-dependent outer window, gamma-only, spinors). The code must
recover the analytic Bloch sums to 1e-10 and the norm/centre/spread of the Gaussians.
The implementation was also checked against wannier90 on the `Si2_valence` and `Si2` datasets of
[WannierDatasets](https://github.com/qiaojunfeng/WannierDatasets): the Wannier functions coincide with the
`wannier_plot` XSF files, and $\frac1{N_k}\sum_{\mathbf k}e^{-i\mathbf k\cdot\mathbf R}V^\dagger\varepsilon V$ reproduces $H(\mathbf R)$ of `Si2_tb.dat` (with disentanglement).

## Using it from C++

```cpp
#include "Wannier/WannierWavefunctions.hpp"

WannierWavefunctionsParameters p;
p.qe_save_dir = "out/si.save";
p.seedname    = "si";
WannierWavefunctions wwf(p);
QEWavefunction psi = wwf.bloch_wannier_gauge(ik);   // psi.evc(n, ig), psi.miller(ig, x), 0 <= ik < num_kpts
```

The building blocks are `QEWavefunction` (read/write of `wfc*.dat`), `WannierGauge` (reads `u.mat`/`u_dis.mat`/`.eig`,
gives $V(\mathbf k)$ and $H_W(\mathbf k)=V^\dagger\varepsilon V$) and `WannierWavefunctions`.
