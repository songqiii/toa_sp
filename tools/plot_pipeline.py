"""
Generate fig_pipeline.pdf — comparison of traditional PSRCHIVE folding
pipeline (left) and toa_sp direct single-pulse pipeline (right).

Traditional pipeline (left):
  N sub-integration files → summation → dspsr fold → paz RFI →
  paas scrunch → pat template-match → one TOA/obs → tempo2 →
  >3σ outlier deletion → 638 TOAs, WRMS=1.74 ms

toa_sp pipeline (right):
  N single sub-integration files → per-file: dedisperse + RFI →
  pulse detection (heimdall/presto) → toa_sp 9-strategy TOA →
  AICc+Δ_conv selection → collect .tim → tempo2 →
  no outlier deletion → 688 TOAs, WRMS=1.33 ms
"""

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      8,
    "figure.dpi":     150,
})

# Colors
C_TRAD_BG   = "#FDE0DD"  # light red
C_TRAD_EDGE = "#B2182B"  # dark red
C_NEW_BG    = "#D9E6F2"  # light blue
C_NEW_EDGE  = "#2166AC"  # dark blue
C_ARROW     = "#555555"
C_RESULT    = "#333333"
C_BAD       = "#B2182B"
C_GOOD      = "#2166AC"

# ── helper ─────────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, text, bg, edge, fontsize=7.2, bold=False):
    """Draw a rounded rectangle with text."""
    weight = "bold" if bold else "normal"
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08", facecolor=bg, edgecolor=edge,
        linewidth=1.0, alpha=0.92)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, color=edge, wrap=False)

def draw_arrow(ax, x, y1, y2, color=C_ARROW):
    """Downward arrow from (x, y1) to (x, y2)."""
    ax.annotate("", xy=(x, y2 + 0.04), xytext=(x, y1 - 0.04),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0))

def draw_label(ax, x, y, text, color=C_RESULT, fontsize=6.5, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, style="italic")

def draw_section_label(ax, x, y, text, color, fontsize=9):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            weight="bold", color=color)

# ── figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8.5, 7.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Column positions
XL = 0.25   # left column center
XR = 0.75   # right column center
BOX_W = 0.38

# ── Section titles ─────────────────────────────────────────────────────
draw_section_label(ax, XL, 0.97, "Traditional PSRCHIVE Pipeline", C_TRAD_EDGE, 9.5)
draw_section_label(ax, XR, 0.97, "toa\\_sp Direct Framework", C_NEW_EDGE, 9.5)

# Subtitles
ax.text(XL, 0.94, "(folding + template matching)", ha="center", fontsize=7,
        color=C_TRAD_EDGE, style="italic")
ax.text(XR, 0.94, "(single-pulse, no folding)", ha="center", fontsize=7,
        color=C_NEW_EDGE, style="italic")

# Divider line
ax.axvline(0.5, ymin=0.02, ymax=0.92, color="#CCCCCC", lw=1.0, ls="--")

# ── LEFT: Traditional Pipeline ─────────────────────────────────────────
y_positions_left = [0.88, 0.81, 0.74, 0.67, 0.60, 0.53, 0.44, 0.34]

# Box 1: N sub-integration files
draw_box(ax, XL, y_positions_left[0], BOX_W, 0.055,
         "N sub-integration files\n(each $\\sim$6.4 s, search-mode PSRFITS)",
         C_TRAD_BG, C_TRAD_EDGE, 6.8)

# Box 2: Summation
draw_box(ax, XL, y_positions_left[1], BOX_W, 0.045,
         "Sub-integration summation\n$\\rightarrow$ single large file",
         C_TRAD_BG, C_TRAD_EDGE, 6.8)

# Box 3: dspsr fold
draw_box(ax, XL, y_positions_left[2], BOX_W, 0.045,
         "\\textsc{dspsr}: fold at pulsar period\n$\\rightarrow$ fold-mode archive",
         C_TRAD_BG, C_TRAD_EDGE, 6.8)

# Box 4: paz RFI
draw_box(ax, XL, y_positions_left[3], BOX_W, 0.045,
         "\\texttt{paz}: RFI excision\n(broadband + narrow-band flagging)",
         C_TRAD_BG, C_TRAD_EDGE, 6.8)

# Box 5: paas scrunch
draw_box(ax, XL, y_positions_left[4], BOX_W, 0.045,
         "\\texttt{paas}: frequency + time scrunching\n$\\rightarrow$ standard profile",
         C_TRAD_BG, C_TRAD_EDGE, 6.8)

# Box 6: pat template
draw_box(ax, XL, y_positions_left[5], BOX_W, 0.045,
         "\\texttt{pat}: matched-filter template fitting\n$\\rightarrow$ one TOA per observation",
         C_TRAD_BG, C_TRAD_EDGE, 6.8)

# Box 7: tempo2
draw_box(ax, XL, y_positions_left[6], BOX_W, 0.045,
         "\\textsc{tempo2}: timing model fit\n$\\rightarrow$ post-fit residuals",
         C_TRAD_BG, C_TRAD_EDGE, 6.8)

# Box 8: outlier deletion
draw_box(ax, XL, y_positions_left[7], BOX_W, 0.055,
         "$>3\\sigma$ iterative outlier deletion\n(50 of 688 pulses discarded)",
         "#FFCCCC", C_BAD, 6.8)

# Result box
draw_box(ax, XL, 0.24, BOX_W, 0.055,
         "638 TOAs retained  $\\cdot$  WRMS = 1.74 ms",
         "#FFFFFF", C_BAD, 7.5)
# Thicker border for result
rect_r = mpatches.FancyBboxPatch(
    (XL - BOX_W/2, 0.24 - 0.0275), BOX_W, 0.055,
    boxstyle="round,pad=0.08", facecolor="none", edgecolor=C_BAD,
    linewidth=2.0, alpha=1.0)
ax.add_patch(rect_r)

# Draw arrows between left boxes
for i in range(len(y_positions_left) - 1):
    draw_arrow(ax, XL, y_positions_left[i] - 0.028, y_positions_left[i+1] + 0.022)
draw_arrow(ax, XL, y_positions_left[-1] - 0.028, 0.267)

# ── RIGHT: toa_sp Pipeline ─────────────────────────────────────────────
y_positions_right = [0.88, 0.81, 0.74, 0.67, 0.60, 0.44, 0.34]

# Box 1: N sub-integration files
draw_box(ax, XR, y_positions_right[0], BOX_W, 0.055,
         "N sub-integration files\n(each $\\sim$6.4 s, search-mode PSRFITS)",
         C_NEW_BG, C_NEW_EDGE, 6.8)

# Box 2: Per-file processing
draw_box(ax, XR, y_positions_right[1], BOX_W, 0.045,
         "Per-file processing (no summation)\nDedispersion + per-channel RFI excision",
         C_NEW_BG, C_NEW_EDGE, 6.8)

# Box 3: Pulse detection
draw_box(ax, XR, y_positions_right[2], BOX_W, 0.045,
         "Pulse detection (\\texttt{heimdall}/\\texttt{presto})\n$\\rightarrow$ DM, approximate arrival time",
         C_NEW_BG, C_NEW_EDGE, 6.8)

# Box 4: toa_sp TOA extraction
draw_box(ax, XR, y_positions_right[3], BOX_W, 0.055,
         "\\textsc{toa\\_sp}: 9 TOA strategies\nGaussian / EMG / Voigt / shapelet /\nleading\\_edge / peak / center\\_of\\_mass",
         C_NEW_BG, C_NEW_EDGE, 6.8)

# Box 5: AICc + Δ_conv selection
draw_box(ax, XR, y_positions_right[4], BOX_W, 0.045,
         "AICc + $\\Delta_{\\rm conv}$: strategy selection\n$\\rightarrow$ optimal TOA per pulse",
         C_NEW_BG, C_NEW_EDGE, 6.8)

# Box 6: tempo2
draw_box(ax, XR, y_positions_right[5], BOX_W, 0.045,
         "Collect all TOAs $\\rightarrow$ \\texttt{.tim} file\n\\textsc{tempo2}: timing model fit",
         C_NEW_BG, C_NEW_EDGE, 6.8)

# Box 7: No outlier deletion
draw_box(ax, XR, y_positions_right[6], BOX_W, 0.055,
         "No outlier deletion\n(all 688 pulses retained)",
         "#CCE8CC", C_GOOD, 6.8)

# Result box
draw_box(ax, XR, 0.24, BOX_W, 0.055,
         "688 TOAs retained  $\\cdot$  WRMS = 1.33 ms",
         "#FFFFFF", C_GOOD, 7.5)
rect_r2 = mpatches.FancyBboxPatch(
    (XR - BOX_W/2, 0.24 - 0.0275), BOX_W, 0.055,
    boxstyle="round,pad=0.08", facecolor="none", edgecolor=C_GOOD,
    linewidth=2.0, alpha=1.0)
ax.add_patch(rect_r2)

# Draw arrows between right boxes
for i in range(len(y_positions_right) - 1):
    draw_arrow(ax, XR, y_positions_right[i] - 0.028, y_positions_right[i+1] + 0.022)
draw_arrow(ax, XR, y_positions_right[-1] - 0.028, 0.267)

# ── Bottom comparison bar ──────────────────────────────────────────────
ax.text(0.5, 0.16, "Result: 24\\% lower WRMS (1.33 vs. 1.74 ms) with 50 more pulses retained",
        ha="center", va="center", fontsize=9, weight="bold", color="#333333")

# Separator line above result
ax.axhline(0.19, xmin=0.08, xmax=0.92, color="#AAAAAA", lw=0.8)

# ── Key differences callouts ───────────────────────────────────────────
# Time annotation
ax.text(0.5, 0.10, "Traditional: $\\sim$2--4 days on 32 cores     $\\vert$     toa\\_sp: $\\sim$87 min on 1 CPU (10 threads)",
        ha="center", va="center", fontsize=7.2, color="#666666", style="italic")

# ── save ───────────────────────────────────────────────────────────────
outpath = r"C:\Users\sbzha\sb2\sp_timing\overleaf_sp_timing\fig_pipeline.pdf"
fig.savefig(outpath, dpi=200, bbox_inches="tight")
print(f"Saved → {outpath}")
plt.close()
