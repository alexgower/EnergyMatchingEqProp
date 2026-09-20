#!/usr/bin/env python3
"""Paper sample-grid figures for the CIFAR-10 arms (2026-09-20).

Inputs are the four .npy arrays in this folder, each (128, 3, 32, 32) uint8: the FULL seeded batch
that the FID runner would have drawn at fid_seed=1, integrated to that arm's optimal sampling time
under that arm's own native inference. Panel i of every arm starts from the same x0, so the arms are
comparable image by image.

Output: one figure per arm, the WHOLE batch of 128 at 16 x 8, nearest-neighbour, uniform gutters,
no borders or labels. Nothing is selected: the figure is the batch the sampler produced, in order.

  samples_backprop.png   backpropagation arm, post-CD 147k, tau_s = 3.25
  samples_ift.png        implicit-solve arm, post-CD 147k, tau_s = 3.25
  samples_ep.png         EP arm, post-CD 102k, tau_s = 3.25
  samples_ws192.png      norm- and attention-free arm, post-CD 177k, tau_s = 5.0

Run: uv run python3 make_figures.py [ncol]        (default 16)
"""
import os, sys, numpy as np
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # figures live beside their .npy inputs

ARMS = [("backprop", "Backpropagation"), ("ift", "Implicit solve"),
        ("ep", "EP"), ("ws192", "Norm- and attention-free")]
NCOL = int(sys.argv[1]) if len(sys.argv) > 1 else 16
GUT = 2          # gutter, pixels; uniform, as in the Sec 4.1 figures
BG = 255


def tile(imgs, ncol):
    """(N,3,32,32) uint8 -> one RGB array, nearest-neighbour, uniform gutters, no borders."""
    n, _, h, w = imgs.shape
    nrow = (n + ncol - 1) // ncol
    out = np.full((nrow * h + (nrow - 1) * GUT, ncol * w + (ncol - 1) * GUT, 3), BG, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, ncol)
        out[r * (h + GUT):r * (h + GUT) + h, c * (w + GUT):c * (w + GUT) + w] = im.transpose(1, 2, 0)
    return out


for arm, label in ARMS:
    a = np.load(f"samples_{arm}.npy")
    assert a.shape == (128, 3, 32, 32) and a.dtype == np.uint8, (arm, a.shape, a.dtype)
    img = tile(a, NCOL)
    Image.fromarray(img).save(f"samples_{arm}.png")
    print(f"  wrote samples_{arm}.png  {img.shape[1]}x{img.shape[0]} px  "
          f"({len(a)} samples, {NCOL} per row) -- {label}")
