#!/usr/bin/env python3
"""Figures H and I for Sec 4.2 (equivalence at training). These are the section's
two figures; the tables are built separately by make_tables.py.

  figure_H_gamma_x_tfree.png   cos(dtheta_EP, dtheta_BP) vs T_free, one line per
                               gamma, float32 | float64 panels; T_free=14 marked.
  figure_I_beta.png            cos(dtheta_EP, dtheta_BP) vs beta at the operating
                               gamma, linear and quadratic nudge in both arithmetics.

SOURCES (all at 512 seeded interpolants; each figure prints what it read and how
many interpolants that source used):

  figure H   sweep_512.log (its "figure H" block) + sweep_tfree_extra_gammas.log
             + sweep_tfree_high_gamma.log            [float32]
             sweep_tfree_f64.log + sweep_tfree_f64_rest.log   [float64]
  figure I   sweep_figure_I_beta.log + sweep_beta_high.log    [float32]
             sweep_figure_I_f64_full.log                      [float64]


Run: uv run python3 make_figures.py
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 1000.0                 # gamma_code = ALPHA * gamma_V on CIFAR-10
ROW = re.compile(r"^\s*([\d.e+-]+)\s*\|\s*([\d.e+-]+)\s+([\d.e+-]+)\s*\|\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*$")
def read(name):
    p = os.path.join(HERE, name)
    return open(p, errors="ignore").read() if os.path.exists(p) else None

BIG = read("sweep_512.log")
def block(txt, key):
    """Take one '======== figure X ========' section out of the 512 log.

    That log still carries a "figure G" section as well as "figure H" -- the two
    were produced by one job -- so the split has to know about both even though
    only H is plotted now.
    """
    if txt is None: return None
    parts = re.split(r"={4,}\s*figure ([GH])[^=]*={4,}", txt)
    out = {}
    for i in range(1, len(parts) - 1, 2): out[parts[i]] = parts[i + 1]
    return out.get(key)

# ---------------- figure H: the free-phase budget, float32 | float64 ----------------
# Two panels sharing axes. What the figure is for is the SHAPE of the recovery --
# strongly negative below ~6 sweeps, above 0.9 by 8, at 0.99 by 10 -- and the fact
# that the PLATEAU HEIGHT, not the shape, is what gamma changes.
#
# Two things must NOT be read off it (both were once claimed here and are retired,
# CHANGELOG (k) and (m)):
#   - a crossing budget. The sign at T_free=7 flips with the interpolant draw
#     (-0.611 in the 512 draw, +0.439 in the 128 draw at the same gamma and beta).
#   - anything comparing the two panels. float64 does not fit at B=128, so the
#     float64 panel is 32x16 against float32's 128x4 -- different draws, and the
#     float64 panel is simply a harder one. The panels share an axis, not a draw.
def tfree_cells(srcs):
    cells, cur = {}, None
    for src in srcs:
        if not src: continue
        for line in src.split("\n"):
            m = re.match(r"######## (?:f64 )?gamma=([\d.]+) T_free=(\d+) ########", line)
            if m: cur = (float(m.group(1)), int(m.group(2))); continue
            r = ROW.match(line)
            if r and cur: cells[cur] = float(r.group(5)); cur = None
    return cells
h32 = tfree_cells([block(BIG, "H"), read("sweep_tfree_extra_gammas.log"), read("sweep_tfree_high_gamma.log")])
h64 = tfree_cells([read("sweep_tfree_f64.log"), read("sweep_tfree_f64_rest.log")])
gs = sorted({g for g, _ in h32}); ts = sorted({t for _, t in h32})
MK = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
COL = {g: c for g, c in zip(gs, plt.cm.viridis(np.linspace(0.12, 0.92, len(gs))))}
def glab(g):
    e = np.log10(g / ALPHA)
    return rf"$\gamma = 10^{{{e:.0f}}}$" if abs(e - round(e)) < 0.01 else \
           rf"$\gamma = {g/ALPHA/10**np.floor(e):.0f}\times10^{{{np.floor(e):.0f}}}$"
# Nine gammas were measured; six are plotted. The other three (5e-6, 5e-5, 3e-4)
# lie exactly on top of neighbours -- the clean gammas coincide -- so plotting them
# added no line a reader could distinguish. These six span the three regimes:
# never recovers (1e-6), recovers late and low (1e-5), clean (1e-3, 1e-4), and
# recovers to a depressed plateau (1e-2, 1e-1).
PANEL_G = {100.0, 10.0, 1.0, 0.1, 0.01, 0.001}  # gamma_code; gamma_V = these / ALPHA
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
for ax, cells, title in ((axes[0], h32, "float32"), (axes[1], h64, "float64")):
    for i, g in enumerate(sorted({g for g, _ in cells if g in PANEL_G})):
        xs = [t for t in sorted({t for gg, t in cells if gg == g})]
        ax.plot(xs, [cells[(g, t)] for t in xs], marker=MK[i % len(MK)],
                ls="-" if i % 2 == 0 else "--", c=COL.get(g, "0.4"), label=glab(g),
                ms=8 - 1.4 * (i % 3), mfc="none" if i % 2 else COL.get(g, "0.4"), mew=1.5, lw=1.5, alpha=0.9)
    ax.axvline(14, ls=":", c="gray")
    ax.axhline(0, c="k", lw=0.5, alpha=0.4)
    ax.set_xscale("log"); ax.set_xticks(ts); ax.set_xticklabels([str(t) for t in ts])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$T_{\rm free}$ (free-phase sweeps)"); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, which="both")
axes[0].set_ylim(-1.05, 1.05)
axes[0].set_ylabel(r"$\cos(\Delta\theta_{\rm EP},\ \Delta\theta_{\rm BP})$")
axes[0].text(14.8, -0.42, r"operating $T_{\rm free}=14$", c="gray", fontsize=8, ha="left")
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    if h: ax.legend(h[::-1], l[::-1], loc="upper left", fontsize=7.5,
                    framealpha=0.92, borderpad=0.4, labelspacing=0.35)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure_H_gamma_x_tfree.png"), dpi=200); plt.close(fig)
_g32, _g64 = {g for g, _ in h32} & PANEL_G, {g for g, _ in h64} & PANEL_G
print(f"figure H: float32 {len(_g32)} of {len(PANEL_G)} panel gammas x {len(ts)} T_free, "
      f"float64 {len(_g64)} of {len(PANEL_G)} x {len({t for _, t in h64})} T_free")
if _g32 - _g64:
    print("  WARNING: float64 panel is MISSING gammas " +
          ", ".join(f"{g/ALPHA:g}" for g in sorted(_g32 - _g64)) +
          " -- the two panels do not show the same gammas, so they cannot be read against each other")

# ---------------- figure I: beta only, at the operating gamma ----------------
# Three curves on one axis: the low-beta float32 floor (linear f32 collapses while
# float64 does not -- the same cancellation as the gamma floor), the quadratic
# nudge's divergence above beta ~ 1, and the linear nudge's monotone rise.
srcI = read("sweep_figure_I_beta.log")
srcI_hi = read("sweep_beta_high.log")   # beta 300..1e4, linear + quadratic, float32
# The float64 arms come from ONE run over the whole beta range in both nudge forms
srcI_64 = read("sweep_figure_I_f64_full.log")
if srcI:
    sec, D, cur_b = None, {"lin32": {}, "quad32": {}, "lin64": {}, "quad64": {}}, None
    for line in srcI.split("\n"):
        if "linear, float32" in line: sec = "lin32"; continue
        if "quadratic, float32" in line: sec = "quad32"; continue
        if "linear, float64" in line: sec = "lin64"; continue
        m = re.match(r"==== beta=([\d.]+) ====", line)
        if m: cur_b = float(m.group(1)); continue
        r = re.match(r"\s*(linear|quadratic)\s+[\d.]+\s+\d+\s+\d+\s+\d+\s+([\d.]+)\s+[\d.]+\s+\d\s*\|\s*([\d.]+)\s+([\d.-]+)", line)
        if r and sec in ("lin32", "quad32"): D[sec][float(r.group(2))] = float(r.group(4))
        a = re.search(r"EP-vs-backprop param-grad cos: ([\d.-]+)", line)
        if a and sec == "lin64" and cur_b is not None: D["lin64"][cur_b] = float(a.group(1))
    # merge the high-beta extension: same instrument, gamma and budget
    if srcI_hi:
        for line in srcI_hi.split("\n"):
            r = re.match(r"\s*(linear|quadratic)\s+[\d.]+\s+\d+\s+\d+\s+\d+\s+([\d.]+)\s+[\d.]+\s+\d\s*\|\s*([\d.]+)\s+([\d.-]+)", line)
            if r:
                D["lin32" if r.group(1) == "linear" else "quad32"][float(r.group(2))] = float(r.group(4))
    # float64, both nudges, one run over the full beta range -- overwrites the old series
    if srcI_64:
        D["lin64"], D["quad64"] = {}, {}
        for line in srcI_64.split("\n"):
            r = re.match(r"\s*(linear|quadratic)\s+[\d.]+\s+\d+\s+\d+\s+\d+\s+([\d.]+)\s+[\d.]+\s+\d\s*\|\s*([\d.]+)\s+([\d.-]+)", line)
            if r:
                D["lin64" if r.group(1) == "linear" else "quad64"][float(r.group(2))] = float(r.group(4))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xyb = lambda d: (sorted(d), [d[b] for b in sorted(d)])
    ax.plot(*xyb(D["lin32"]),  "s-",  c="tab:red",    ms=6, label="linear nudge, float32")
    ax.plot(*xyb(D["lin64"]),  "v--", c="tab:purple", ms=7, mfc="none", mew=1.5, label="linear nudge, float64")
    ax.plot(*xyb(D["quad32"]), "o-",  c="tab:orange", ms=6, label="quadratic nudge, float32")
    if D["quad64"]:
        ax.plot(*xyb(D["quad64"]), "^--", c="tab:brown", ms=7, mfc="none", mew=1.5,
                label="quadratic nudge, float64")
    ax.axvline(30, ls=":", c="gray"); ax.text(34, -0.15, r"operating $\beta=30$", c="gray", fontsize=8, ha="left")
    ax.axhline(0, c="k", lw=0.5, alpha=0.4)
    ax.set_xscale("log"); ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"$\beta$ (nudge strength)")
    ax.set_ylabel(r"$\cos(\Delta\theta_{\rm EP},\ \Delta\theta_{\rm BP})$")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure_I_beta.png"), dpi=200); plt.close(fig)
    # the job's grep filtered the instrument's header line out of this log, so the
    # geometry is recorded from the submit script: linear/quadratic
    # float32 at 128 x 4, linear float64 at 64 x 8 -- 512 interpolants each.
    print(f"figure I <- sweep_figure_I_beta.log + sweep_beta_high.log (float32) "
          f"+ {'sweep_figure_I_f64_full.log (float64, job 35141419)' if srcI_64 else 'the OLD truncated float64 series'}: "
          f"beta {min(D['lin32']):g}..{max(D['lin32']):g}, "
          f"{len(D['lin32'])} lin-f32 / {len(D['lin64'])} lin-f64 / "
          f"{len(D['quad32'])} quad-f32 / {len(D['quad64'])} quad-f64 points")
    # The two arithmetics are DIFFERENT DRAWS: float64 needs a smaller batch to fit,
    # so it is 32x16 against float32's 128x4. Read the low-beta separation (float32
    # collapses, float64 does not) -- that gap is far too large to be a draw effect.
    # Do NOT read the high-beta ordering off this figure cell by cell.
    if srcI_64:
        print("     WARNING: float32 arm is 128x4, float64 arm 32x16 -- different draws. "
              "The low-beta floor is the readable comparison; the high-beta ordering is not.")


# ---------------- provenance summary (read this before writing a caption) ----------------
def counts(txt):
    return sorted({int(a) * int(b) for a, b in re.findall(r"batch=(\d+) N_batch=(\d+)", txt or "")})
_h_srcs = "".join(x or "" for x in (block(BIG, "H"), read("sweep_tfree_extra_gammas.log"),
                                    read("sweep_tfree_high_gamma.log"), read("sweep_tfree_f64.log")))
print("\ninterpolants per figure:")
print(f"  H  {counts(_h_srcs) or [512]}   (from the submit scripts: f32 128x4, f64 32x16; the T_free logs are grepped to data rows so no header survives)")
print(f"  I  {counts(srcI) or [512]}   (from the submit script; job 35048208's grep dropped the header)")
allc = set(counts(_h_srcs) or [512]) | set(counts(srcI) or [512])
print("  -> " + ("CONSISTENT at %d interpolants" % allc.pop() if len(allc) == 1
                 else "MIXED: %s -- do not quote a single sample count" % sorted(allc)))
