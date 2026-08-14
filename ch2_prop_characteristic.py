"""Measured propeller characteristic for Chapter 2 (Fig. 2.2).

Produces prop-characteristic-uiuc.pdf, the literature-review illustration of
the classical performance metrics: C_T, C_P, and eta against advance ratio J
for one two-blade fixed-pitch propeller at one shaft speed, with the ideal
zero-drag efficiency overlaid as a dashed curve.

Data: APC Thin Electric 12x6 at 3,040 rpm, UIUC database volume 4
(apce_12x6_0630od_3040.txt). This machine and speed are chosen because the
sweep runs the full classical story in a single curve: both coefficients fall
roughly linearly as the advance ratio unloads the blades, the efficiency rises
to its peak and then collapses as the thrust approaches zero, and the final
point crosses into windmill (C_T < 0). The efficiency is drawn only while the
propeller thrusts; past C_T = 0 it loses its meaning as a propulsive
efficiency. The ANCE 8.5x6 is deliberately not used here, since it is the
worked example of Chapters 3 and 4 (Figs. 3.5-3.7 and 4.3).

The dashed curve is Glauert's zero-drag construction evaluated on this
machine's own data: at each measured point the Froude (actuator-disk, induced
losses only) efficiency for the same thrust loading,

    eta_ideal = 2 / (1 + sqrt(1 + T_c)),   T_c = (8/pi) C_T / J^2,

where T_c is the thrust coefficient on disk area and freestream dynamic head.
As the blades unload, T_c -> 0 and the ideal tends to one while the measured
efficiency collapses; the widening gap is the cost of profile drag and the
other real blade losses. This one frame therefore carries what were
previously two separate literature figures (a McCormick-style measured
characteristic and a redrawing of Glauert's zero-drag sketch).
"""

import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 13})

_HERE = os.path.dirname(os.path.abspath(__file__))
_thesis_figs = os.path.join(_HERE, "..", "Reaves-Thesis", "2. Lit Review", "Figs")


def _db_root():
    """Locate the extracted database, tolerating either directory layout."""
    nested = os.path.join(_HERE, "UIUC-propDB", "UIUC-propDB")
    flat = os.path.join(_HERE, "UIUC-propDB")
    for root in (nested, flat):
        if os.path.isdir(os.path.join(root, "volume-4", "data")):
            return root
    raise FileNotFoundError(
        "UIUC database not found; expected volume-4/data under "
        f"{nested} or {flat}. Extract UIUC-propDB.zip alongside this script."
    )


DATA = os.path.join(_db_root(), "volume-4", "data", "apce_12x6_0630od_3040.txt")

J, CT, CP, eta = np.loadtxt(DATA, skiprows=1, unpack=True)
# the archived file carries duplicated trailing rows; keep first occurrences
_, keep = np.unique(np.column_stack([J, CT, CP, eta]), axis=0, return_index=True)
keep = np.sort(keep)
J, CT, CP, eta = J[keep], CT[keep], CP[keep], eta[keep]
order = np.argsort(J)
J, CT, CP, eta = J[order], CT[order], CP[order], eta[order]
print(f"{os.path.basename(DATA)}: {J.size} points, J = {J.min():.2f}..{J.max():.2f}")

thrusting = CT > 0

# Glauert's zero-drag curve on this machine's own loading: the Froude ideal
# for the measured C_T at each J (induced losses only, no profile drag)
Tc = (8.0 / np.pi) * CT[thrusting] / J[thrusting] ** 2
eta_ideal = 2.0 / (1.0 + np.sqrt(1.0 + Tc))

fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor="w")
axr = ax.twinx()

h_ct, = ax.plot(J, CT, marker="o", ms=5, lw=1.8, color="tab:blue",
                label=r"$C_T$")
h_cp, = ax.plot(J, CP, marker="s", ms=5, lw=1.8, color="tab:orange",
                label=r"$C_P$")
ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
h_id, = axr.plot(J[thrusting], eta_ideal, ls="--", lw=1.8, color="k",
                 label=r"$\eta_{prop}$, zero drag (ideal)")
h_eta, = axr.plot(J[thrusting], eta[thrusting], marker="^", ms=6, lw=1.8,
                  color="tab:green", label=r"$\eta_{prop}$, measured")

ax.set_xlabel(r"Advance ratio  $J$")
ax.set_ylabel(r"$C_T$,  $C_P$")
axr.set_ylabel(r"$\eta_{prop}$")
ax.set_xlim(0.1, 0.65)
ax.set_ylim(-0.012, 0.09)
axr.set_ylim(0, 1.0)
ax.legend(handles=[h_ct, h_cp, h_eta, h_id], loc="lower center",
          bbox_to_anchor=(0.42, 0.02), frameon=False, fontsize=11)

fig.tight_layout()
fig.savefig(os.path.join(_HERE, "prop-characteristic-uiuc.pdf"), bbox_inches="tight")
if os.path.isdir(_thesis_figs):
    fig.savefig(os.path.join(_thesis_figs, "prop-characteristic-uiuc.pdf"),
                bbox_inches="tight")
    print("exported prop-characteristic-uiuc.pdf to thesis Figs")
