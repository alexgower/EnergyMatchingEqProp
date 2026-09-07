# make_figures.py — regenerates figure_D_imagenet_transfer.png from the sweep
# logs in this directory (plus the CIFAR-10 reference log in the sibling
# in_paper_velocity_correspondence_gamma_grid/ folder, if present).
#   cd <this dir> && python make_figures.py
# (Any matplotlib-equipped env; no project imports needed.)
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 1000.0          # output_scale on ImageNet-32 (and CIFAR-10): gamma_V = gamma_code / ALPHA
ROW = re.compile(r"(float32|float64)\s+(\d+)\s+([\d.e-]+) \|\s+([\d.]+)\s+([\d.]+)\s+([\d.e+-]+)")

def rows(text, dtype, k_h=1):
    """-> {gamma_code: (cos, norm_ratio, rel_L2_err)} for one dtype / K_h."""
    out = {}
    for m in ROW.finditer(text):
        if m.group(1) == dtype and int(m.group(2)) == k_h:
            out[float(m.group(3))] = (float(m.group(4)), float(m.group(5)), float(m.group(6)))
    return out

# ---- main 3-pass log: unclamped fp32 | unclamped TF32 | clamped fp32 ----
txt = open(os.path.join(HERE, "sweep_output.log")).read()
# headers look like "######## ImageNet-32 gamma grid, <pass> ########": splitting on
# the hash run puts each header TEXT and its table in consecutive chunks.
chunks = txt.split("########")
sec, cur = {}, None
for c in chunks:
    s = c.strip()
    if s.startswith("ImageNet-32 gamma grid"):
        if "CLAMP" in s and "FLOAT64" in s: cur = "clamped64"   # 4th pass (coherent rerun)
        elif "CLAMP" in s:                  cur = "clamped"
        elif "TF32" in s:                   cur = "tf32"
        else:                               cur = "fp32"
    elif cur:
        sec[cur] = sec.get(cur, "") + c
fp32 = rows(sec["fp32"], "float32")
tf32 = rows(sec["tf32"], "float32")
clamp = rows(sec["clamped"], "float32")
# clamped float64: 4th pass of the coherent rerun (same 512 interpolants as the
# other three, chunked evaluation); falls back to a separate log if present.
clamp64 = rows(sec.get("clamped64", ""), "float64")
f64_path = os.path.join(HERE, "sweep_clamped_f64.log")
if not clamp64 and os.path.exists(f64_path):
    clamp64 = rows(open(f64_path).read(), "float64")

# ---- TF32 curve = the SAMPLING-RUN arithmetic, where it was measured ----
# The FID samplers run PyTorch's default math mode (matmul fp32, cuDNN TF32) WITH
# the clamp, not the both-flags-on unclamped pass that sweep_output.log's TF32
# section holds. sweep_fid_mathmode.log measures the samplers' own mode over
# gamma_code 1e-4..1, so use it for that range; above gamma_code 1 only the
# both-flags sweep exists (sweep_upper_gamma.log), and it is spliced on to carry
# the merge onto the fp32 line. The splice is justified by measurement, not
# assumption: the both-flags CLAMPED sweep (sweep_tf32_clamped.log) matches the
# sampler-mode clamped sweep in rel_L2_err at every one of its 9 gammas, i.e.
# the matmul flag changes nothing -- the cuDNN convolutions set the floor alone.
# The overlap check at gamma_code=1 is printed at the bottom of this script.
mm_path = os.path.join(HERE, "sweep_fid_mathmode.log")
tf32_sampler = {}
if os.path.exists(mm_path):
    msec, mcur = {}, None
    for c in open(mm_path).read().split("########"):
        t = c.strip()
        if t.startswith("ImageNet-32,"):
            mcur = "clamped" if "clamp" in t else "unclamped"
        elif mcur:
            msec[mcur] = msec.get(mcur, "") + c
    tf32_sampler = rows(msec.get("clamped", ""), "float32")
    if tf32_sampler:
        tf32_bothflags = dict(tf32)          # keep for the overlap report
        tf32 = dict(tf32_sampler)

# ---- optional upper-gamma extension (sweep_upper_gamma.log, gamma_code 1..100):
#      same four passes; spliced so figure D spans gamma_V 1e-7..1e-1 like figure A
#      and shows TF32 merging onto the fp32/f64 line above the 2e-4 floor. ----
up_path = os.path.join(HERE, "sweep_upper_gamma.log")
if os.path.exists(up_path):
    usec, ucur = {}, None
    for c in open(up_path).read().split("########"):
        s = c.strip()
        if s.startswith("ImageNet-32 UPPER GAMMA"):
            if "CLAMP" in s and "FLOAT64" in s: ucur = "clamped64"
            elif "CLAMP" in s:                  ucur = "clamped"
            elif "TF32" in s:                   ucur = "tf32"
            else:                               ucur = "fp32"
        elif ucur:
            usec[ucur] = usec.get(ucur, "") + c
    for d, key, dt in [(fp32, "fp32", "float32"), (tf32, "tf32", "float32"),
                       (clamp, "clamped", "float32"), (clamp64, "clamped64", "float64")]:
        for g, v in rows(usec.get(key, ""), dt).items():
            d.setdefault(g, v)

# (2026-09-03: the CIFAR-10 reference curve was REMOVED from this figure — it is
#  about the ImageNet-32 transfer on its own terms, and the CIFAR comparison is a
#  number in the text rather than a curve here. NB this means the "prefactor about
#  half of CIFAR-10's" clause is no longer VISIBLE in figure D; quote it from the
#  two epsilon values instead.)

def xy(d):
    gs = sorted(d)
    return [g / ALPHA for g in gs], [d[g][2] for g in gs]

# floors (annotated from the data, not typed in)
fp32_floor  = min(v[2] for g, v in fp32.items() if g <= 1e-3)
tf32_floor  = float(np.median([v[2] for g, v in tf32.items() if g <= 1e-2]))   # floor = small-gamma plateau only (the spliced upper points have merged onto the O(gamma) line)
clamp_floor = float(np.mean([v[2] for g, v in clamp.items() if g <= 1e-3]))
slope_un = float(np.mean([v[2] / g for g, v in fp32.items() if 1e-2 <= g <= 1.0]))   # eps per gamma_code (guide anchored on the main grid, not the drifting top decade)
slope_cl = float(np.mean([v[2] / g for g, v in clamp.items() if 1e-2 <= g <= 1.0]))

fig, ax = plt.subplots(figsize=(6.8, 4.8))
# labels mirror figure A: name the arithmetic, and mark the two curves that use the
# recipe's output clamp (its form, c tanh(V/c) with c=1e4 folded into f_L, is caption material).
ax.loglog(*xy(tf32),  "s-", c="tab:red",    label="float32 + TF32 kernels, clamped (sampling-run arithmetic)")
ax.loglog(*xy(fp32),  "o-", c="tab:blue",   label="float32, unclamped")
ax.loglog(*xy(clamp), "^-", c="tab:green",  label="float32, clamped")
if clamp64:
    ax.loglog(*xy(clamp64), "v-", c="tab:purple", label="float64, clamped")
gg = np.array([1e-7, max(fp32) / ALPHA])
ax.loglog(gg, slope_un * ALPHA * gg, "k--", lw=1, label=r"$\propto\gamma$")
# y-limits from the plotted data (the float64 clamped curve reaches ~1e-8 and was
# previously below the axis).
_allv = [v[2] for d in (fp32, tf32, clamp, clamp64) if d for v in d.values()]
ylo, yhi = min(_allv) / 3.0, max(_allv) * 3.0
ax.axvline(1e-6, ls=":", c="gray")
ax.text(1.15e-6, ylo * 1.5, r"operating $\gamma{=}10^{-6}$", fontsize=8, c="gray", ha="left", va="bottom")
# (the pre-fold "clamped floor" annotation is gone: with the clamp folded into
#  f_L the clamped curve shares the fp32 floor — clamp_floor is still computed
#  and printed below as a check that it equals fp32_floor.)
for y, lab in [(tf32_floor, f"TF32 floor ({tf32_floor:.1e})"),
               (fp32_floor, f"fp32 floor ({fp32_floor:.1e})")]:
    ax.axhline(y, ls=":", lw=0.7, c="gray")
    ax.text(1.05e-7, y * 1.45, lab, fontsize=7, c="gray")
ax.set_xlim(8e-8, 1.5 * max(fp32) / ALPHA)
ax.set_ylim(ylo, yhi)
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"relative velocity error $\epsilon$")
ax.legend(fontsize=7.5, loc="upper left")   # upper-left is empty; lower-right sat on the fp32 curve
plt.tight_layout()
plt.savefig(os.path.join(HERE, "figure_D_imagenet_transfer.png"), dpi=140)
print("regenerated figure_D_imagenet_transfer.png"
      + ("" if clamp64 else "  (no sweep_clamped_f64.log yet: float64 curve omitted)"))
print(f"slopes eps/gamma_code: unclamped {slope_un:.3e}, clamped {slope_cl:.3e}, ratio {slope_cl/slope_un:.1f}x")
print(f"floors: fp32 {fp32_floor:.3e}, TF32 {tf32_floor:.3e}, clamped(fp32) {clamp_floor:.3e}")

# ---- provenance / splice self-check (printed, and quoted in EXPLANATION) ----
if tf32_sampler:
    ov = sorted(set(tf32_sampler) & set(tf32_bothflags))
    print("TF32 curve: sweep_fid_mathmode.log (clamped, matmul fp32 + cuDNN TF32) "
          f"for gamma_code {min(tf32_sampler):g}..{max(tf32_sampler):g}; "
          "sweep_upper_gamma.log (both flags, unclamped) above that.")
    for g in ov:
        print(f"  overlap gamma_code {g:g}: sampler-mode {tf32_sampler[g][2]:.3e} "
              f"vs both-flags {tf32_bothflags[g][2]:.3e} "
              f"({100 * abs(tf32_sampler[g][2] / tf32_bothflags[g][2] - 1):.1f}% apart)")
