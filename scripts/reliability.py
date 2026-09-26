#!/usr/bin/env python3
"""Split-half reliability of the emotion directions, per run, with the single estimator
(pev/estimator.cast_vectors). Premises are split at random into two halves; each emotion's
direction is estimated from each half (each half refits its own neutral projector); the cosine
between halves is averaged over emotions, over the strangers + household + invented casts, and
over --splits random splits.

Writes <data_root>/compare/<name>/reliability.json.

Usage:
  python3 scripts/reliability.py --config config.yaml --name main
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pev import config as cfg  # noqa: E402
from pev.estimator import cast_vectors, load_blocks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", default=None, help="comma list (default: every run in the config)")
    ap.add_argument("--name", default="main")
    ap.add_argument("--splits", type=int, default=50)
    a = ap.parse_args()
    study = cfg.load(cfg.config_path(a.config))
    rng = np.random.default_rng(0)
    res = {}
    for run in (a.runs.split(",") if a.runs else list(study.runs)):
        A, M, H, E, T, layer = load_blocks(study, run)
        half = T // 2
        per_cast = {c: [] for c in study.contrast_casts}
        for _ in range(a.splits):
            perm = rng.permutation(T); w1 = np.zeros(T, np.float32); w1[perm[:half]] = 1; w2 = 1 - w1
            for c in study.contrast_casts:
                V1, ok1 = cast_vectors(A[c], M[c], w1, E); V2, ok2 = cast_vectors(A[c], M[c], w2, E)
                ok = ok1 & ok2
                n1 = V1[ok] / np.linalg.norm(V1[ok], axis=1, keepdims=True)
                n2 = V2[ok] / np.linalg.norm(V2[ok], axis=1, keepdims=True)
                per_cast[c].append(float((n1 * n2).sum(1).mean()))
        res[run] = {"label": study.label(run), "mean": float(np.mean([np.mean(v) for v in per_cast.values()])),
                    "per_cast": {c: float(np.mean(v)) for c, v in per_cast.items()}, "splits": a.splits}
        print(f"{study.label(run):20s} split-half cosine {res[run]['mean']:.3f}", flush=True)
    od = os.path.join(study.data_root, "compare", a.name); os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, "reliability.json"), "w"), indent=1)
    print("wrote", os.path.join(od, "reliability.json"))


if __name__ == "__main__":
    main()
