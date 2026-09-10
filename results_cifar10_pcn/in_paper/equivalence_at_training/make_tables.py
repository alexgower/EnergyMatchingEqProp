#!/usr/bin/env python3
"""Section 4.2 tables, emitted as caption-free `tabular` blocks.

Two tables survive in the paper; both are written to
tables_section42.tex so the paper's hand-written versions can be diffed against
a regeneration:

  table 4        gamma x beta window: 6x6 EP-float32 grid + two control rows
                 (IFT float32, EP float64 at beta=30).            MAIN TEXT
  budget axes    n_CG (IFT), T_nudge (EP), K_h (EP), all at gamma_V = 1e-4.
                                                                   APPENDIX

Captions are written by hand in the paper and deliberately NOT emitted here:
the shading rule (cos < 0.99) and the provenance are the only things this file
asserts, and it prints the provenance to stdout when run.

Two datasets are measured, filed and deliberately NOT in the paper; they are
NOT emitted here either, but their logs stay in this folder (see EXPLANATION
section 2, rows marked "held in reserve"):
  - the beta sweep with update norms (superseded by figure I + two sentences)
  - the per-group breakdown (nothing in the main text uses it)

SOURCES
  table 4 grid       sweep_gamma_beta_table.log     
  table 4 IFT row    sweep_512.log, "figure G" block, cos(ift,ffn) 
  table 4 f64 row    sweep_512_f64.log, cos(ep,ffn)                 
  budget axes        sweep_appendix_512.log                         

All at 512 seeded flow-matching interpolants on the CIFAR-10 replication
warm-up checkpoint (145k, live net_model weights); float32 128x4, float64
32x16. gamma_code = 1000 * gamma_V on CIFAR-10.

Run: uv run python3 make_tables.py
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 1000.0                 # gamma_code = ALPHA * gamma_V on CIFAR-10
SHADE = 0.99                   # cells below this are shaded

# gamma_code -> LaTeX column header, in table-4 column order
GCOLS = [("0.01", r"$10^{-5}$"), ("0.03", r"$3\times10^{-5}$"), ("0.1", r"$10^{-4}$"),
         ("1.0", r"$10^{-3}$"), ("10.0", r"$10^{-2}$"), ("100.0", r"$10^{-1}$")]
BETAS = ["0.3", "1.0", "3.0", "10.0", "30.0", "100.0"]
BLAB = {"0.3": "0.3", "1.0": "1", "3.0": "3", "10.0": "10", "30.0": "30", "100.0": "100"}


def read(name):
    p = os.path.join(HERE, name)
    return open(p, errors="ignore").read() if os.path.exists(p) else None


def cell(v):
    """Format a cosine, shading it if it is below the usable threshold."""
    s = f"{v:.4f}" if abs(v) < SHADE else f"{v:.4f}"
    return rf"\cellcolor{{gray!15}}{s}" if v < SHADE else s


# ---------------- table 4: the gamma x beta window ----------------
# sweep rows:  nudge lam K_h T_free T_nudge beta dt 3ph | |g_ep| cos cost
SWEEP = re.compile(r"^\s*(?:linear|quadratic)\s+[\d.]+\s+\d+\s+\d+\s+\d+\s+([\d.]+)\s+[\d.]+\s+\d+\s*\|"
                   r"\s*([\d.e+-]+)\s+([\d.e+-]+)\s+\d+\s*$", re.M)
# diag rows:   gamma | |g_ift| |g_ep| | cos(ift,ffn) cos(ep,ffn) cos(ep,ift)
DIAG = re.compile(r"^\s*([\d.e+-]+)\s*\|\s*([\d.e+-]+)\s+([\d.e+-]+)\s*\|"
                  r"\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*$", re.M)

grid = {}                                          # (gamma_code, beta) -> cos
txt = read("sweep_gamma_beta_table.log")
for chunk in re.split(r"#{4,}\s*gamma_code=([\d.]+)\s*#{4,}", txt)[1:]:
    if re.fullmatch(r"[\d.]+", chunk.strip()):
        g = chunk.strip()
        continue
    for beta, _norm, cos in SWEEP.findall(chunk):
        grid[(g, beta)] = float(cos)

def diag_col(txt, block_key, col):
    """gamma_code -> one cosine column of a in_paper_diag_ep_vs_ift_vs_ffn table."""
    if block_key:
        parts = re.split(r"={4,}\s*figure ([GH])[^=]*={4,}", txt)
        txt = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}[block_key]
    return {r[0]: float(r[col]) for r in DIAG.findall(txt)}

ift = diag_col(read("sweep_512.log"), "G", 3)       # cos(ift, ffn), float32
f64 = diag_col(read("sweep_512_f64.log"), None, 4)  # cos(ep, ffn), float64, beta=30

lines = [r"% ===== TABLE 4 (main text): the gamma x beta window =====",
         r"% caption written by hand in the paper; shading rule is cos < 0.99",
         r"\begin{tabular}{r cccccc}", r"\toprule",
         r"$\beta$ & " + " & ".join(h for _, h in GCOLS) + r" \\ \midrule"]
for b in BETAS:
    lines.append(BLAB[b] + " & " + " & ".join(cell(grid[(g, b)]) for g, _ in GCOLS) + r" \\")
lines += [r"\midrule", r"\multicolumn{7}{l}{\itshape controls} \\",
          r"$\Delta\theta_{\rm IFT}$, float32 & "
          + " & ".join(cell(ift[g]) for g, _ in GCOLS) + r" \\",
          r"$\Delta\theta_{\rm EP}$, float64 ($\beta{=}30$) & "
          + " & ".join(cell(f64[g]) for g, _ in GCOLS) + r" \\",
          r"\bottomrule", r"\end{tabular}", ""]

# ---------------- budget axes: flat from one sweep ----------------
# Rows 4, 6 and 15 are dropped (T_nudge-only, they left mid-column gaps);
# a dash means the value is not in that axis's swept grid.
ROWS = ["1", "2", "3", "5", "10", "20", "50"]
app = read("sweep_appendix_512.log")


def axis(header, value_re, col):
    """value -> cosine, for one '======== name ========' section of the appendix log."""
    sec = re.split(r"={4,}\s*" + header + r"[^=]*={4,}", app)[1]
    sec = re.split(r"={4,}", sec)[0]
    out = {}
    for chunk in re.split(value_re, sec)[1:]:
        if re.fullmatch(r"\d+", chunk.strip()):
            v = chunk.strip()
            continue
        rows = {r[0]: float(r[col]) for r in DIAG.findall(chunk)}
        if "0.1" in rows:                       # gamma_code 0.1 = gamma_V 1e-4
            out[v] = rows["0.1"]
    return out

ncg = axis(r"n_CG \(IFT\)", r"#{4,}\s*n_cg=(\d+)\s*#{4,}", 3)      # cos(ift, ffn)
tnd = axis(r"T_nudge \(EP\)", r"#{4,}\s*T_nudge=(\d+)\s*#{4,}", 4)  # cos(ep, ffn)
kh = axis(r"K_h \(EP\)", r"#{4,}\s*K_h=(\d+)\s*#{4,}", 4)

lines += [r"% ===== budget axes (appendix): flat from one sweep, all at gamma_V = 1e-4 =====",
          r"% caption written by hand in the paper; dash = value not in that axis's swept grid",
          r"\begin{tabular}{r ccc}", r"\toprule",
          r"value & $n_{\rm CG}$ (IFT) & $T_{\rm nudge}$ (EP) & $K_h$ (EP) \\ \midrule"]
for v in ROWS:
    cells = [f"{d[v]:.4f}" if v in d else "---" for d in (ncg, tnd, kh)]
    lines.append(v + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", ""]

out = os.path.join(HERE, "tables_section42.tex")
open(out, "w").write("\n".join(lines))

# ---------------- provenance (read this before writing a caption) ----------------
print(f"wrote {out}")
print(f"  table 4 grid : {len(grid)} cells, beta {BLAB[BETAS[0]]}..{BLAB[BETAS[-1]]}, "
      f"gamma_code {GCOLS[0][0]}..{GCOLS[-1][0]}")
print(f"  IFT control  : {[f'{ift[g]:.4f}' for g, _ in GCOLS]}")
print(f"  f64 control  : {[f'{f64[g]:.4f}' for g, _ in GCOLS]}")
print(f"  budget axes  : n_CG {sorted(ncg, key=int)}, T_nudge {sorted(tnd, key=int)}, "
      f"K_h {sorted(kh, key=int)}")
ffn = re.search(r"FFN ref: \|g_ff\| mean = ([\d.]+)", read("sweep_512.log"))
print(f"  backprop reference norm |g_BP| (float32, 512): {ffn.group(1)}  "
      f"<- quote THIS against the quadratic nudge's diverged norms, not |g_EP|")
