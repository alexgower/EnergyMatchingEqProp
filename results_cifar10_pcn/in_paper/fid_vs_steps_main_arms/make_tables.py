#!/usr/bin/env python3
"""Build the two CIFAR-10 FID tables from the evaluation logs filed in this folder.

  tables_section5.tex             our rows of Table 5 (final models after the contrastive phase).
                                  A tabular fragment from \\midrule to \\bottomrule; the header and the
                                  published reference rows above it are written by hand in the paper.
  tables_phase1_fid_vs_steps.tex  the phase-1 table (FID at 50k / 100k / 145k training steps).
                                  A tabular fragment from \\toprule to \\bottomrule.

Every cell names the record folder its number comes from, and the FID is read out of that folder's
evaluation log, so no number in either table is typed by hand. Each log must be a 50,000-sample
evaluation; anything else stops the build. Parameter counts come from params_cache.json (counted from
the checkpoints, which are not in the repository).

Run: uv run python3 make_tables.py
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- the two tables
# Table 5: (method, gradient, parameter-count key in params_cache.json, record folder, tau_s)
# Parameter counts, keyed by checkpoint path relative to in_paper/ (counted once from the checkpoints
# into params_cache.json, since the checkpoints themselves are not in the repository). The implicit-solve
# and EP models share the backpropagation model's architecture, so all three use its count.
BACKPROP_50M = "equivalence_at_inference/in_paper_fid_calculation_train_backprop_infer_pcn/checkpoint_postcd_147000.pt"
WS192_78M = "checkpoints_main_arms/ws192_postcd_177000.pt"

TABLE5 = [
    ("Energy Matching (our replication)", "backprop", BACKPROP_50M,
     "fid_ffn_postcd147k_ema_tau3.25_seed1_3.5370", 3.25),
    ("PotentialEP", "implicit solve", BACKPROP_50M,
     "fid_ift_postcd147k_ema_tau3.25_3.6351", 3.25),
    ("PotentialEP", "EP", BACKPROP_50M,
     "fid_ep_postcd102k_ema_tau3.25_3.6401", 3.25),
    None,   # \midrule: the hardware-plausible architecture is set off from the three main arms
    (r"Energy Matching, norm- and attention-free (\S\ref{subsec:hardware})", "backprop", WS192_78M,
     "fid_ws192_postcd177k_ema_tau5.0_3.5089", 5.0),
]

# Phase-1 table: (gradient, record folders at 50k, 100k, 145k steps), all at tau_s = 1.0
PHASE1_STEPS = ["50k", "100k", "145k"]
PHASE1 = [
    ("Backpropagation", ["fid_ffn_step50000_ema_11.2152", "fid_ffn_step100000_ema_6.13",
                         "fid_ffn_step145000_ema_seed1_6.4242"]),
    ("Implicit solve", ["fid_ift_step50000_ema_12.3463", "fid_ift_step100000_ema_6.4463",
                        "fid_ift_step145000_ema_6.7040"]),
    ("EP", ["fid_ep_step50000_ema_12.5959", "fid_ep_step100000_ema_6.2336",
            "fid_ep_step145000_ema_6.8715"]),
]


# ---------------------------------------------------------------- reading the logs
def read_fid(folder, tau):
    """FID at sampling time `tau` from the evaluation log in `folder` (one per folder)."""
    logs = [p for p in glob.glob(os.path.join(HERE, folder, "fid_cifar10.*"))
            if not p.endswith("fid_cifar10.INFO")]          # the .INFO link is a local convenience
    assert len(logs) == 1, f"{folder}: expected one evaluation log, found {len(logs)}"
    text = open(logs[0], errors="ignore").read()
    fids = {float(t): float(v) for t, v in re.findall(r"FID at t=([0-9.]+)\s*=>\s*([0-9.]+)", text)}
    n = re.search(r"total fakes:\s*(\d+)|of (\d+) fakes|fake:\s*(\d+)\)", text)
    assert n and int(next(g for g in n.groups() if g)) == 50000, f"{folder}: not a 50k-sample evaluation"
    assert tau in fids, f"{folder}: no FID at tau {tau} (has {sorted(fids)})"
    ckpt = re.search(r"Loading checkpoint:\s*(\S+)", text).group(1)
    print(f"  {fids[tau]:7.4f}  tau {tau:<4g} {folder}\n           checkpoint {ckpt}")
    return fids[tau]


def millions(key):
    return f"{round(json.load(open(os.path.join(HERE, 'params_cache.json')))[key] / 1e6)}M"


# ---------------------------------------------------------------- Table 5
print("Table 5:")
rows = [r"\midrule"]
for row in TABLE5:
    if row is None:
        rows.append(r"\midrule")
        continue
    method, grad, params_key, folder, tau = row
    tau_text = f"{tau:g}" if f"{tau:g}" != "5" else "5.0"
    rows.append(f"{method} & {grad} & {millions(params_key)} & "
                f"{read_fid(folder, tau):.2f} ($\\tau_s = {tau_text}$) \\\\")
rows.append(r"\bottomrule")
open(os.path.join(HERE, "tables_section5.tex"), "w").write(
    "% ===== Sec 5: OUR rows only, no caption (written by hand) =====\n"
    "% This fragment is the TAIL of the comparison table: it opens with the\n"
    "% \\midrule that separates us from the published block and closes with\n"
    "% \\bottomrule. The tabular preamble, the header row and the five\n"
    "% reference rows above it are maintained BY HAND -- those numbers come\n"
    "% from other papers and are not verifiable from this repository.\n"
    "% The second \\midrule sets off the hardware-plausible architecture.\n"
    "% Column order assumed: Method & Gradient & Params & FID, i.e. {l l r l}.\n"
    "%\n"
    "% Every row is post-CD at that model's optimal sampling time, 50,000\n"
    "% samples, EMA weights, Stratonovich Euler-Heun at dt 0.01. Repeat draws of\n"
    "% one checkpoint differ by 0.05-0.15 FID, so 2 d.p. is the honest precision\n"
    "% and nothing below ~0.2 FID separates two rows.\n"
    "% Params are summed from the checkpoints themselves (net_model state_dict),\n"
    "% rounded to whole megaparams; exact counts are in params_cache.json.\n"
    + "\n".join(rows) + "\n")

# ---------------------------------------------------------------- phase-1 table
print("\nPhase-1 table:")
rows = [r"\toprule",
        r" & \multicolumn{%d}{c}{Phase-1 FID $\downarrow$} \\" % len(PHASE1_STEPS),
        r"\cmidrule(lr){2-%d}" % (len(PHASE1_STEPS) + 1),
        "Gradient & " + " & ".join(f"${s[:-1]}$k steps" for s in PHASE1_STEPS) + r" \\",
        r"\midrule"]
for grad, folders in PHASE1:
    rows.append(f"{grad} & " + " & ".join(f"{read_fid(f, 1.0):.2f}" for f in folders) + r" \\")
rows.append(r"\bottomrule")
open(os.path.join(HERE, "tables_phase1_fid_vs_steps.tex"), "w").write(
    "% ===== Phase-1 FID vs training steps, no caption/label (written by hand) =====\n"
    "% Column order assumed: Gradient & 50k & 100k & 145k, i.e. {l r r r}.\n"
    "% Generated by make_tables.py from the filed logs -- do not edit by hand.\n"
    "% Protocol: 50,000 samples, EMA weights, tau_s = 1.0, Stratonovich Euler-Heun at\n"
    "% dt 0.01, each arm under its own native inference. Repeat draws of one checkpoint\n"
    "% differ by ~0.03-0.15 FID, so 2 d.p. is the honest precision.\n"
    + "\n".join(rows) + "\n")
print("\nwrote tables_section5.tex and tables_phase1_fid_vs_steps.tex")
