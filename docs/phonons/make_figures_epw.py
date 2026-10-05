"""Figures of phonons_implementation.tex for the real hBN (QE -> EPW -> EDUS), from data_epw.json (epw_validation.py)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, GREY = "#1f5fa8", "#d9730d", "#2e8b57", "#777777"
BOHR_PM = 52.917721
d = json.load(open("data_epw.json"))
S, D = d["scalars"], d["series"]


def rel(key):
    return np.array(D[key]["t_fs"]), np.array(D[key]["rel_bohr"]) * BOHR_PM


# 1. IPA small oscillations: coupling off, none, static, dynamic
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for key, label, color, ls, lw in [
        ("ipa_off", f"coupling off: {S['ipa_off_w']:.1f}", GREY, "--", 1.6),
        ("ipa_none", f"none: {S['ipa_none_w']:.1f}", ORANGE, "-", 1.1),
        ("ipa_static", f"static: {S['ipa_static_w']:.1f}", BLUE, "-", 1.1),
        ("ipa_dynamic", f"dynamic: {S['ipa_dynamic_w']:.1f}", GREEN, ":", 1.4)]:
    t, x = rel(key)
    ax[0].plot(t, x, color=color, ls=ls, lw=lw, label=label + r" cm$^{-1}$")
ax[0].set_xlim(0, 100)
ax[0].set_ylim(-1.15, 2.6)
ax[0].set_xlabel("time (fs)")
ax[0].set_ylabel(r"$u_{\rm B,x}-u_{\rm N,x}$ (pm)")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2)
ax[0].set_title("real hBN, IPA, no laser", fontsize=9)
t, x = rel("ipa_laser")
ax[1].plot(t, x, color=BLUE, lw=1.1)
ax[1].axvspan(0, 4 * 4.135667 / 5.5, color=GREY, alpha=0.15, lw=0)
ax[1].text(3.6, 0.85, "pulse", fontsize=7, color=GREY, transform=ax[1].get_xaxis_transform())
ax[1].set_xlabel("time (fs)")
ax[1].set_ylabel(r"$u_{\rm B,x}-u_{\rm N,x}$ (pm)")
ax[1].set_title(r"IPA, laser 5.5 eV, $10^{11}$ W/cm$^2$, static", fontsize=9)
fig.tight_layout()
fig.savefig("epw_ipa.pdf")

# 2. HSEX: no laser, laser, energy drift
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
styles = [("none", ORANGE, "-"), ("static", BLUE, "-"), ("dynamic", GREEN, "--")]
for key, color, ls in styles:
    t, x = rel("hsex_" + key)
    ax[0].plot(t, x, color=color, ls=ls, lw=1.1, label=f"{key}: {S['hsex_' + key + '_w']:.1f}" + r" cm$^{-1}$")
ax[0].set_xlim(0, 100)
ax[0].set_ylim(-1.15, 2.6)
ax[0].set_xlabel("time (fs)")
ax[0].set_ylabel(r"$u_{\rm B,x}-u_{\rm N,x}$ (pm)")
ax[0].legend(frameon=False, fontsize=6, loc="upper center")
ax[0].set_title(r"HSEX ($\epsilon=2$), no laser", fontsize=9)
for key, color, ls in styles:
    t, x = rel("hsex_laser_" + key)
    ax[1].plot(t, x, color=color, ls=ls, lw=1.1, label=key)
ax[1].set_xlabel("time (fs)")
ax[1].set_ylabel(r"$u_{\rm B,x}-u_{\rm N,x}$ (pm)")
ax[1].legend(frameon=False, fontsize=6.5, ncol=3, loc="lower center")
ax[1].set_title("HSEX, laser 5.5 eV", fontsize=9)
for key, label, color in [("hsex_static_energy", r"static, $\Delta t$ = 0.01 fs", BLUE),
                          ("hsex_static_dt2_energy", r"static, $\Delta t$ = 0.005 fs", GREEN)]:
    ax[2].plot(D[key]["t_fs"], D[key]["rel"], color=color, lw=1.1, label=label)
ax[2].set_xlabel("time (fs)")
ax[2].set_ylabel(r"$(E-E_0)\,/\,\max E_{\rm lattice}$")
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax[2].legend(frameon=False, fontsize=6.5)
ax[2].set_title("HSEX, energy, no laser", fontsize=9)
fig.tight_layout()
fig.savefig("epw_hsex.pdf")
