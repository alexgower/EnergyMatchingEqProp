#!/usr/bin/env python3
"""Collate every T_free sweep in this folder into one table. Reads only; no runs."""
import os, re, sys

HERE = "results_cifar10_pcn/in_paper/equivalence_at_training"
ALPHA = 1000.0
DIAG = re.compile(r"^\s*([\d.e+-]+)\s*\|\s*([\d.e+-]+)\s+([\d.e+-]+)\s*\|"
                  r"\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*$", re.M)
# every header shape used across the logs
HDR = re.compile(r"#{4,}\s*(f64\s+)?(?:beta=([\d.]+)\s+)?gamma=([\d.]+)\s*"
                 r"(?:beta=([\d.]+)\s+)?T_free=(\d+)\s*#{4,}")

# (file, section-restrict, arithmetic, default beta)
SRC = [("sweep_512.log",              "figure H", "f32", "30"),
       ("sweep_tfree_extra_gammas.log", None,     "f32", "30"),
       ("sweep_tfree_high_gamma.log",   None,     "f32", "30"),
       ("sweep_tfree_beta_dep.log",     None,     "f32", None),
       ("sweep_tfree_gb_corners.log",   None,     "f32", None),
       ("sweep_tfree_f64.log",          None,     "f64", "30"),
       ("sweep_tfree_f64_rest.log",     None,     "f64", "30")]

# superseded: pre-512 draws, kept only so the collation is complete
OLD = [("sweep_gamma_tfree_grid.log",          None, "f32/32",  "30"),
       ("sweep_tfree_fill_and_batch_ceiling.log", None, "f32/128", "30")]

rows = {}                      # (arith, gamma_code, beta) -> {T: cos}, and source files
for fn, sect, arith, dbeta in SRC:
    txt = open(os.path.join(HERE, fn), errors="ignore").read()
    if sect:
        parts = re.split(r"={4,}\s*figure ([GH])[^=]*={4,}", txt)
        txt = {parts[i]: parts[i+1] for i in range(1, len(parts)-1, 2)}["H"]
    hits = list(HDR.finditer(txt))
    for i, m in enumerate(hits):
        chunk = txt[m.end(): hits[i+1].start() if i+1 < len(hits) else len(txt)]
        d = DIAG.findall(chunk)
        if not d:
            continue
        g, beta, T = m.group(3), (m.group(2) or m.group(4) or dbeta), int(m.group(5))
        key = (arith, float(g), float(beta))
        rows.setdefault(key, ({}, set()))
        rows[key][0].setdefault(T, float(d[0][4]))     # cos(ep, ffn); keep first if dup
        rows[key][1].add(fn)

TS = sorted({T for v, _ in rows.values() for T in v})

def gv(gc):
    """gamma_code -> gamma_V as a clean mantissa-exponent string."""
    from decimal import Decimal
    d = Decimal(str(gc)) / Decimal(str(ALPHA))
    e = d.adjusted()
    m = d.scaleb(-e).normalize()
    return f"{m}e{e}"

def cross(v):
    """First T at which the cosine is positive, given the T grid actually measured."""
    ts = sorted(v)
    for i, T in enumerate(ts):
        if v[T] > 0:
            return T if i == 0 or v[ts[i-1]] <= 0 else T
    return None

def reach99(v):
    ts = sorted(v)
    for T in ts:
        if v[T] >= 0.99:
            return T
    return None

def emit(rows, TS, title):
    hdr = f"{'arith':<7} {'gamma_V':>7} {'beta':>5} | " + " ".join(f"{T:>7}" for T in TS) + " | cross  0.99  source"
    print(title); print(hdr); print("-" * len(hdr))
    prev = None
    for (arith, g, b) in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
        v, srcs = rows[(arith, g, b)]
        if prev and prev != arith: print()
        prev = arith
        cells = " ".join((f"{v[T]:>7.4f}" if T in v else f"{'.':>7}") for T in TS)
        c, r = cross(v), reach99(v)
        src = ",".join(sorted(s.replace("sweep_tfree_", "").replace("sweep_", "").replace(".log", "")
                              for s in srcs))
        print(f"{arith:<7} {gv(g):>7} {b:>5.0f} | {cells} | {str(c or '--'):>5} {str(r or '--'):>5}  {src}")

emit(rows, TS, "CURRENT: 512 seeded interpolants")

old = {}
for fn, sect, arith, dbeta in OLD:
    txt = open(os.path.join(HERE, fn), errors="ignore").read()
    hits = list(HDR.finditer(txt))
    for i, m in enumerate(hits):
        chunk = txt[m.end(): hits[i+1].start() if i+1 < len(hits) else len(txt)]
        d = DIAG.findall(chunk)
        if not d: continue
        key = (arith, float(m.group(3)), float(m.group(2) or m.group(4) or dbeta))
        old.setdefault(key, ({}, set()))
        old[key][0].setdefault(int(m.group(5)), float(d[0][4]))
        old[key][1].add(fn)
print()
emit(old, sorted({T for v, _ in old.values() for T in v}),
     "SUPERSEDED by the 512 re-measurement (arith column carries the interpolant count)")


