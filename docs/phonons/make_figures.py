"""Figures of phonons_implementation.tex from data.json (validation runs on the synthetic hBN model)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, GREY = "#1f5fa8", "#d9730d", "#2e8b57", "#777777"
BOHR_PM = 52.917721
d = json.load(open("data.json"))

# 1. small oscillations: free, coupled, coupled with the response subtracted
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for key, label, color in [("free", "coupling off: 1370.0 cm$^{-1}$", GREY),
                          ("coupled", "coupled: 1140.5 cm$^{-1}$", ORANGE),
                          ("coupled_sub", r"coupled, $K^0=K^{\rm BO}-\Pi$: 1370.0 cm$^{-1}$", BLUE)]:
    t = np.array(d[key]["t_fs"])
    ax[0].plot(t, np.array(d[key]["rel_bohr"]) * BOHR_PM, color=color, lw=1.6 if key == "free" else 1.1,
               ls="--" if key == "free" else "-", label=label, zorder=3 if key == "free" else 2)
ax[0].set_xlim(0, 60)
ax[0].set_ylim(-1.15, 2.3)
ax[0].set_xlabel("time (fs)")
ax[0].set_ylabel(r"$u_{\rm B,x}-u_{\rm N,x}$ (pm)")
ax[0].legend(frameon=False, fontsize=7, loc="upper center")
ax[0].set_title("small oscillations, no laser", fontsize=9)
for key, color in [("coupled", ORANGE), ("coupled_sub", BLUE)]:
    t = np.array(d[key]["t_fs"])
    El = np.array(d[key]["E_lattice"])
    ax[1].plot(t, np.array(d[key]["E_minus_E0"]) / (El.max() / 2.), color=color, lw=1.)
ax[1].set_xlabel("time (fs)")
ax[1].set_ylabel(r"$(E-E_0)\,/\,\max E_{\rm lattice}$")
ax[1].set_title("total energy (per spin channel)", fontsize=9)
ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
fig.tight_layout()
fig.savefig("small_oscillations.pdf")

# 2. coherent phonon driven by the laser, energy balance
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
L = d["laser"]
t = np.array(L["t_fs"])
ax[0].plot(t, np.array(L["rel_bohr"]) * BOHR_PM, color=BLUE, lw=1.1)
ax[0].set_xlabel("time (fs)")
ax[0].set_ylabel(r"$u_{\rm B,x}-u_{\rm N,x}$ (pm)")
ax[0].axvspan(0, 4 * 0.517, color=GREY, alpha=0.15, lw=0)
ax[0].text(2.6, 0.2, "pulse", fontsize=7, color=GREY, transform=ax[0].get_xaxis_transform())
ax[0].set_title(r"laser 8 eV, $10^{11}$ W/cm$^2$, $x$ polarized", fontsize=9)
ax[1].plot(t, 1e5 * np.array(L["dE"]), color=BLUE, lw=1.1, label=r"$E-E_0$ (with phonons)")
ax[1].plot(t, 1e5 * np.array(L["W"]), color=ORANGE, lw=1.1, ls="--", label=r"$W$ (work of the field)")
ax[1].plot(t, 1e5 * np.array(L["residual"]), color=GREEN, lw=1.1, label=r"$E-E_0-W$")
ax[1].set_xlim(0, 6)
ax[1].set_xlabel("time (fs)")
ax[1].set_ylabel(r"energy ($10^{-5}$ Ha, per spin)")
ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("energy balance", fontsize=9)
fig.tight_layout()
fig.savefig("laser.pdf")

# 3. adiabatic reference with the mean field (HSEX)
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
styles = [("none", r"none: 1006 cm$^{-1}$", ORANGE, "-"), ("static", r"static: 1369.7 cm$^{-1}$", BLUE, "-"),
          ("dynamic", r"dynamic: 1370.0 cm$^{-1}$", GREEN, "--")]
for key, label, color, ls in styles:
    r = d["hsex_" + key]
    ax[0].plot(r["t_fs"], np.array(r["rel_bohr"]) * BOHR_PM, color=color, ls=ls, lw=1.1, label=label)
ax[0].set_xlim(0, 60)
ax[0].set_ylim(-1.15, 2.4)
ax[0].set_xlabel("time (fs)")
ax[0].set_ylabel(r"$u_{\rm B,x}-u_{\rm N,x}$ (pm)")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper center")
ax[0].set_title("HSEX, no laser", fontsize=9)
for key, label, color, ls in styles:
    r = d["hsex_laser_" + key]
    ax[1].plot(r["t_fs"], np.array(r["rel_bohr"]) * BOHR_PM, color=color, ls=ls, lw=1.1, label=key)
ax[1].set_xlabel("time (fs)")
ax[1].set_ylabel(r"$u_{\rm B,x}-u_{\rm N,x}$ (pm)")
ax[1].set_ylim(-0.0055, 0.0025)
ax[1].legend(frameon=False, fontsize=6.5, ncol=3, loc="lower center")
ax[1].set_title("HSEX, laser 8 eV", fontsize=9)
for key, label, color in [("hsex_none_energy", r"$\Delta t$ = 0.05 fs", ORANGE), ("hsex_none_dt2_energy", r"$\Delta t$ = 0.025 fs", BLUE)]:
    r = d[key]
    ax[2].plot(r["t_fs"], np.array(r["dE"]) * 1e7, color=color, lw=1.1, label=label)
ax[2].set_xlabel("time (fs)")
ax[2].set_ylabel(r"$E-E_0$ ($10^{-7}$ Ha)")
ax[2].legend(frameon=False, fontsize=6.5)
ax[2].set_title("HSEX, RK4 energy drift", fontsize=9)
fig.tight_layout()
fig.savefig("adiabatic_reference.pdf")
print("figures written")
