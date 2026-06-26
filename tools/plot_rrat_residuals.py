"""
Generate fig_rrat_residuals.pdf for SP timing paper.
Compares three TOA strategies for RRAT J1913+1330:
  (a) best       — residuals vs MJD, N=688, WRMS=1.33 ms
  (b) PSRCHIVE   — residuals vs MJD, N=638, WRMS=1.74 ms
  (c) Histogram comparison: best vs PSRCHIVE
  (d) leading_edge histogram, N=688, WRMS=1.41 ms

Input files (in tempo2_res/):
  best_toa.res, psrchive_SP.res, leading_edge_toa.res
Format: MJD  residual(s)  error(us)
"""

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

# ── paths ──────────────────────────────────────────────────────────────
RES_DIR = r"C:\Users\sbzha\sb2\sp_timing\tempo2_res"
OUT_DIR = r"C:\Users\sbzha\sb2\sp_timing\overleaf_sp_timing"
FILES = {
    "best":          os.path.join(RES_DIR, "best_toa.res"),
    "psrchive":      os.path.join(RES_DIR, "psrchive_SP.res"),
    "leading_edge":  os.path.join(RES_DIR, "leading_edge_toa.res"),
}

# ── load data ──────────────────────────────────────────────────────────
MJD_OFFSET = 55091  # TEMPO2 strips this from .res MJD; add back for full MJD

def load_res(fpath):
    """Load a .res file; return MJD (full), residual (ms), error (us)."""
    d = np.loadtxt(fpath)
    mjd   = d[:, 0] + MJD_OFFSET
    res   = d[:, 1] * 1000.0        # s → ms
    err   = d[:, 2]                 # us
    return mjd, res, err

data = {}
for key, path in FILES.items():
    data[key] = load_res(path)

# ── compute weighted RMS ───────────────────────────────────────────────
def wrms(res, err):
    """Weighted RMS: sqrt( sum(w*r^2) / sum(w) ) with w = 1/err^2."""
    # err in us, res in ms; convert err to ms for weight consistency
    w = 1.0 / (err / 1000.0) ** 2
    return np.sqrt(np.sum(w * res**2) / np.sum(w))

stats = {}
for key, (mjd, res, err) in data.items():
    stats[key] = {
        "n":    len(res),
        "wrms": wrms(res, err),
        "rms":  np.sqrt(np.mean(res**2)),
    }

# ── style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":       False,
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "lines.markersize":  2.5,
    "figure.dpi":        150,
    "mathtext.default":  "regular",
})

# Colors
C_BEST   = "#2166AC"   # blue
C_PSR    = "#B2182B"   # red
C_LE     = "#4DAF4A"   # green
ALPHA_PT = 0.45

# ── figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8.5, 6.2))  # two-column width; extra height for spacing

# Layout: wider gap between left (a,c) and right (b,d) columns
LEFT_X  = 0.05
RIGHT_X = 0.53
PANEL_W = 0.42
TOP_Y   = 0.56
BOT_Y   = 0.06
PANEL_H = 0.40

# Shared data
mjd_b, res_b, err_b = data["best"]
mjd_p, res_p, err_p = data["psrchive"]
res_le = data["leading_edge"][1]

# ---- (a) best residuals vs MJD ----
ax_a = fig.add_axes([LEFT_X, TOP_Y, PANEL_W, PANEL_H])
s = stats["best"]
ax_a.errorbar(mjd_b, res_b, yerr=err_b/1000.0, fmt="o", ms=2.2,
              color=C_BEST, alpha=ALPHA_PT, mec="none", rasterized=True)
ax_a.axhline(0, color="grey", ls="--", lw=0.6)
ax_a.set_xlabel("MJD")
ax_a.set_ylabel("Residual (ms)")
ax_a.set_title(r"(a) $\mathtt{best}$  ($N=%d$, WRMS = %.2f ms)" % (s["n"], s["wrms"]),
               fontsize=10)
ax_a.tick_params(direction="in", top=True, right=True)
ax_a.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax_a.ticklabel_format(axis="x", style="plain", useOffset=False)

# ---- (b) PSRCHIVE residuals vs MJD (same axes as a) ----
ax_b = fig.add_axes([RIGHT_X, TOP_Y, PANEL_W, PANEL_H])
s = stats["psrchive"]
ax_b.errorbar(mjd_p, res_p, yerr=err_p/1000.0, fmt="o", ms=2.2,
              color=C_PSR, alpha=ALPHA_PT, mec="none", rasterized=True)
ax_b.axhline(0, color="grey", ls="--", lw=0.6)
ax_b.set_xlim(ax_a.get_xlim())
ax_b.set_ylim(ax_a.get_ylim())
ax_b.set_xlabel("MJD")
ax_b.set_title(r"(b) PSRCHIVE  ($N=%d$, WRMS = %.2f ms)" % (s["n"], s["wrms"]),
               fontsize=10)
ax_b.tick_params(direction="in", top=True, right=True, labelleft=False)
ax_b.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax_b.ticklabel_format(axis="x", style="plain", useOffset=False)

# ---- (c) histogram: best vs PSRCHIVE ----
ax_c = fig.add_axes([LEFT_X, BOT_Y, PANEL_W, PANEL_H])
bins = np.linspace(-max(abs(res_b).max(), abs(res_p).max()) * 1.1,
                    max(abs(res_b).max(), abs(res_p).max()) * 1.1, 50)
ax_c.hist(res_b, bins=bins, histtype="step", lw=1.2,
          color=C_BEST, label=r"$\mathtt{best}$  ($N=%d$)" % stats["best"]["n"])
ax_c.hist(res_p, bins=bins, histtype="step", lw=1.2,
          color=C_PSR, label="PSRCHIVE  ($N=%d$)" % stats["psrchive"]["n"])
ax_c.set_xlabel("Residual (ms)")
ax_c.set_ylabel("Count")
ax_c.set_title("(c) Residual histograms", fontsize=10)
ax_c.legend(loc="upper left", frameon=False, fontsize=8)
ax_c.tick_params(direction="in", top=True, right=True)

# ---- (d) histogram: leading_edge vs PSRCHIVE (same axes as c) ----
ax_d = fig.add_axes([RIGHT_X, BOT_Y, PANEL_W, PANEL_H])
s_le = stats["leading_edge"]
s_psr = stats["psrchive"]
ax_d.hist(res_le, bins=bins, histtype="step", lw=1.2,
          color=C_LE, label=r"$\mathtt{leading\_edge}$  ($N=%d$)" % s_le["n"])
ax_d.hist(res_p, bins=bins, histtype="step", lw=1.2,
          color=C_PSR, label="PSRCHIVE  ($N=%d$)" % s_psr["n"])
ax_d.set_xlim(ax_c.get_xlim())
ax_d.set_ylim(ax_c.get_ylim())
ax_d.set_xlabel("Residual (ms)")
ax_d.set_title("(d) Residual histograms", fontsize=10)
ax_d.legend(loc="upper left", frameon=False, fontsize=8)
ax_d.tick_params(direction="in", top=True, right=True, labelleft=False)

# ── save ───────────────────────────────────────────────────────────────
outpath = os.path.join(OUT_DIR, "fig_rrat_residuals.pdf")
fig.savefig(outpath, dpi=200, bbox_inches="tight")
print(f"Saved → {outpath}")
for k, s in stats.items():
    print(f"  {k:15s}: N={s['n']:4d}  WRMS={s['wrms']:.2f} ms  RMS={s['rms']:.2f} ms")
plt.close()
