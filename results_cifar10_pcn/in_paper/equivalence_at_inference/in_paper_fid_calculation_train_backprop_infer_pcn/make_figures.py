#!/usr/bin/env python3
"""Paper sample grids for the CIFAR-10 train-backprop / infer-PCN section.

Inputs (from submit_scripts/in_paper/in_paper_submit_cifar10_sample_grid.sh,
which replays the seeded first batch of the phase-2 FID row (jobs 34750919/34750920)):
    samples_postcd147k_tau3.25_seed1_{backprop,pcn}.npy   uint8, 32 x 3 x 32 x 32

Outputs:
    figure_E_cifar10_pcn_samples.png        PCN relaxation only
    figure_F_cifar10_ffn_vs_pcn_samples.png two stacked blocks, matched seeds

No selection by eye: the images are the FIRST rows*cols of the seeded batch, in
order. 32 samples are stored so the layout can be retried without regenerating.

Run:  cd <this folder> && python3 make_figures.py            # default 2 x 8
      python3 make_figures.py --rows 3                       # 3 x 8
      python3 make_figures.py --rows 4 --cols 8 --dpi 500
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=4, help="rows in figure E (all 32 = 4 x 8: no selection)")
ap.add_argument("--rows-f", type=int, default=2, dest="rows_f", help="rows per block in figure F")
ap.add_argument("--cols", type=int, default=8, help="grid columns")
ap.add_argument("--dpi", type=int, default=400, help="32 px cells stay crisp at ~0.75 col")
ap.add_argument("--gutter", type=float, default=0.06, help="uniform gap, in cell units")
ap.add_argument("--block-gap", type=float, default=0.25, help="gap between blocks (fig F)")
ap.add_argument("--labels", action="store_true", help="label the two blocks in figure F")
A = ap.parse_args()
N = A.rows * A.cols          # figure E
NF = A.rows_f * A.cols       # figure F, per block


def load(tag, n):
    p = os.path.join(HERE, f"samples_postcd147k_tau3.25_seed1_{tag}.npy")
    a = np.load(p)                                       # (32, 3, 32, 32) uint8
    assert a.shape[0] >= n, f"{p} holds {a.shape[0]} samples, need {n}"
    return np.transpose(a[:n], (0, 2, 3, 1))             # -> (n, 32, 32, 3)


def grid(blocks, out, rows, labels=None):
    """blocks: list of (rows*cols, 32, 32, 3) arrays, drawn top to bottom."""
    nb = len(blocks)
    h = nb * rows + (nb - 1) * A.block_gap
    fig = plt.figure(figsize=(A.cols, h))
    for b, imgs in enumerate(blocks):
        top = 1.0 - (b * (rows + A.block_gap)) / h
        bottom = 1.0 - (b * (rows + A.block_gap) + rows) / h
        gs = fig.add_gridspec(rows, A.cols, left=0.0, right=1.0, top=top,
                              bottom=bottom, wspace=A.gutter, hspace=A.gutter)
        for i in range(rows * A.cols):
            ax = fig.add_subplot(gs[i // A.cols, i % A.cols])
            ax.imshow(imgs[i], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        if labels and A.labels:
            fig.text(-0.012, (top + bottom) / 2, labels[b], rotation=90,
                     va="center", ha="center", fontsize=7)
    fig.savefig(os.path.join(HERE, out), dpi=A.dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {out}  ({rows} x {A.cols}, dpi {A.dpi})")


# Figure E shows ALL saved samples (4 x 8 = 32) -- no subset is chosen at all.
grid([load("pcn", N)], "figure_E_cifar10_pcn_samples.png", A.rows)

# Figure F pairs the first NF of each sampler; two 2 x 8 blocks keep the pairing
# readable at appendix width.
ffn, pcn = load("backprop", NF), load("pcn", NF)
grid([ffn, pcn], "figure_F_cifar10_ffn_vs_pcn_samples.png", A.rows_f,
     labels=["feedforward", "PCN"])

# Matched seeds => same noise realisation into both samplers. The pairs are the
# SAME SCENE but not pixel-identical: a ~1e-6-relative per-step velocity
# difference is amplified by 325 SDE steps of a chaotic sampler. Quantify both
# facts so the caption states them instead of asserting "identical".
d = np.abs(ffn.astype(np.int16) - pcn.astype(np.int16))
cors = [np.corrcoef(ffn[i].ravel(), pcn[i].ravel())[0, 1] for i in range(NF)]
un = [np.corrcoef(ffn[i].ravel(), ffn[(i + 1) % NF].ravel())[0, 1] for i in range(NF)]
print(f"paired (same seed, different sampler), {NF} shown: "
      f"mean|d| {d.mean():.2f}/255, max {d.max()}/255; "
      f"per-image corr min {min(cors):.3f} median {float(np.median(cors)):.3f}")
print(f"unrelated baseline (sample i vs i+1 of the same run): "
      f"mean|d| {np.mean([np.abs(ffn[i] - ffn[(i + 1) % NF]).mean() for i in range(NF)]):.2f}/255, "
      f"mean corr {float(np.mean(un)):.3f}")
