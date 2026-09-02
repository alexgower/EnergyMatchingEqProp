# make_figures.py — regenerates figure_A_gamma_floors.png,
# figure_B_kh_flatness.png and figure_C_direction_vs_magnitude.png
# from the sweep logs in this directory.
#   cd <this dir> && python make_figures.py
# (Any matplotlib-equipped env; no project imports needed.)
import os, re
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
txt = open(os.path.join(HERE, "sweep_output.log")).read()
sec = {"main": txt.split("FLOAT64 PASS")[0],
       "f64":  txt.split("FLOAT64 PASS")[1].split("TF32 PASS")[0],
       "tf32": txt.split("TF32 PASS")[1]}

def rows(s, dtype):
    """-> {K_h: {gamma: rel_L2_err}} from the diag's table lines."""
    out = {}
    for m in re.finditer(
            rf"{dtype}\s+(\d+)\s+([\d.e-]+) \|\s+([\d.]+)\s+([\d.]+)\s+([\d.e+-]+)", s):
        out.setdefault(int(m.group(1)), {})[float(m.group(2))] = float(m.group(5))
    return out

fp32 = rows(sec["main"], "float32")   # K_h in {1,2,4,8,14}, true fp32 (TF32 off)

# upper-gamma extension (separate job, same diag/checkpoint): splice in
up_path = os.path.join(HERE, "sweep_upper_gamma.log")
if os.path.exists(up_path):
    up = open(up_path).read()
    up32 = rows(up.split("FLOAT64")[0], "float32")
    up64 = rows(up.split("FLOAT64")[1], "float64") if "FLOAT64" in up else {}
    for k, v in up32.get(1, {}).items():
        fp32.setdefault(1, {}).setdefault(k, v)
f64  = rows(sec["f64"],  "float64")   # K_h=1
if os.path.exists(up_path) and up64:
    for k, v in up64.get(1, {}).items():
        f64.setdefault(1, {}).setdefault(k, v)
tf32 = rows(sec["tf32"], "float32")   # K_h=1, TF32 on (production regime)
# TF32 upper-gamma extension (separate job) so all curves span the same axis
tf_up = os.path.join(HERE, "sweep_upper_gamma_tf32.log")
if os.path.exists(tf_up):
    for k, v in rows(open(tf_up).read(), "float32").get(1, {}).items():
        tf32.setdefault(1, {}).setdefault(k, v)
G = sorted(fp32[1].keys())
# V-UNITS (2026-09-01): the paper states gamma in V-units (clamp on V);
# the sweeps ran in code/o-units. ALPHA = output_scale = 1000 on CIFAR-10;
# gamma_V = gamma_code / ALPHA. Axes are gamma_V; the logs stay in code units.
ALPHA = 1000.0
def xs(d): return sorted(d[1].keys())
def xV(vals): return [g / ALPHA for g in vals]


# ---- Figure A: O(gamma) law vs precision floors ----
fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.loglog(xV(xs(tf32)), [tf32[1][g] for g in xs(tf32)], "s-", c="tab:red",   label="float32 + TF32 kernels (our runs)")
ax.loglog(xV(xs(fp32)), [fp32[1][g] for g in xs(fp32)], "o-", c="tab:blue",  label="float32")
ax.loglog(xV(xs(f64)), [f64[1][g] for g in xs(f64)], "^-", c="tab:green", label="float64")
gg = np.array([1e-7, 0.1])  # gamma_V range
ax.loglog(gg, 0.155 * gg, "k--", lw=1, label=r"$\propto\gamma$")
ax.axvline(1e-4, ls=":", c="gray")
ax.set_ylim(6e-9, 4e-2)
ax.text(1.25e-4, 7.5e-4, r"recipe $\gamma{=}10^{-4}$", fontsize=8, c="gray", ha="left")
for y, lab in [(2.0e-4, "TF32 floor"), (4.2e-7, "fp32 floor")]:
    ax.axhline(y, ls=":", lw=0.7, c="gray")
    ax.text(1.15e-7, y * 1.45, lab, fontsize=7, c="gray")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"relative velocity error $\varepsilon$")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "figure_A_gamma_floors.png"), dpi=140)
plt.close()

# ---- Figure B: K_h budget independence (deadbeat) ----
fig, ax = plt.subplots(figsize=(5.0, 4.6))
# gamma_code 1.0 / 0.1 / 0.01 = gamma_V 1e-3 / 1e-4 / 1e-5. NB the third curve
# was gamma_code=0.001 (gamma_V=1e-6) until 2026-09-02: that point sits AT the
# float32 floor (4.6e-7 vs floor 4.2e-7), so its flatness demonstrated floor-
# limited arithmetic rather than K_h-independence of the O(gamma) bias. At
# gamma_V=1e-5 (1.65e-6) the bias is ~4x the floor, so all three curves now
# make the same claim, and the recipe point 1e-4 is the middle one.
for g, c in [(1.0, "tab:purple"), (0.1, "tab:orange"), (0.01, "tab:cyan")]:
    khs = sorted(fp32.keys())
    ax.semilogy(khs, [fp32[k][g] for k in khs], "o-", c=c, label=fr"$\gamma=10^{{{round(math.log10(g/1000.0))}}}$")
ax.set_xlabel(r"$K_h$ (relaxation sweeps)")
ax.set_ylabel(r"relative velocity error $\varepsilon$")
ax.set_ylim(6e-7, 1e-3)          # headroom so the legend clears the top curve
ax.legend(fontsize=8, loc="upper right")
ax.set_xticks([1, 2, 4, 8, 14])
plt.tight_layout()
plt.savefig(os.path.join(HERE, "figure_B_kh_flatness.png"), dpi=140)
print("regenerated figure_A_gamma_floors.png, figure_B_kh_flatness.png")

# ---- Figure C: how the O(gamma) bias splits between direction and magnitude ----
# eps^2 = (r-1)^2 + 2r(1-c)  =>  angular component theta ~ sqrt(eps^2 - (r-1)^2).
# Uses the float64 pass (no precision floor to contaminate the decomposition).
def ratios(s, dtype):
    out = {}
    for m in re.finditer(
            rf"{dtype}\s+(\d+)\s+([\d.e-]+) \|\s+([\d.]+)\s+([\d.]+)\s+([\d.e+-]+)", s):
        out.setdefault(int(m.group(1)), {})[float(m.group(2))] = (
            float(m.group(4)), float(m.group(5)))       # (norm_ratio, rel_err)
    return out

r64 = ratios(sec["f64"], "float64")[1]
# splice the upper-gamma float64 points (same job as figure A's extension)
if os.path.exists(up_path) and "FLOAT64" in open(up_path).read():
    _up = open(up_path).read().split("FLOAT64")[1]
    for k, v in ratios(_up, "float64").get(1, {}).items():
        r64.setdefault(k, v)
gs = sorted(r64)
eps = np.array([r64[g][1] for g in gs])
mag = np.array([abs(r64[g][0] - 1.0) for g in gs])
ang = np.sqrt(np.clip(eps**2 - mag**2, 0, None))
fig, ax = plt.subplots(figsize=(5.6, 4.6))
gsV = [g / ALPHA for g in gs]
ax.loglog(gsV, eps, "o-", c="k", label=r"$\epsilon$ (total)")
ax.loglog(gsV, ang, "^-", c="tab:blue", label=r"$\epsilon_\perp$ (direction)")
ax.loglog(gsV, mag, "s-", c="tab:orange", label=r"$\epsilon_\parallel$ (magnitude)")
ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"relative velocity error")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "figure_C_direction_vs_magnitude.png"), dpi=140)
print("regenerated figure_C_direction_vs_magnitude.png")
