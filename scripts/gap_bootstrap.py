#!/usr/bin/env python3
"""Premise-block bootstrap of the cast profiles and the household-minus-invented contrast.

Uses the single estimator in pev/estimator.py. Premises are resampled with replacement as blocks
(one story per cast x emotion x premise), and the SAME premise draws are used for every cast,
every condition and every run, so within-run contrasts and between-run changes are paired.

Conditions per run:
  all          every retained story
  filtered     invented-cast stories whose text names a household member removed
  random_drop  the same NUMBER of stories removed from each (cast, emotion) cell at random:
               holds sample size fixed and varies only WHICH stories go.
               filtered - random_drop isolates the content of the removed stories from the loss of n.

Also: for each architecture with both a persona run and a base+prompt run, the paired change in
the contrast (persona minus base+prompt).

Writes <data_root>/compare/<name>/premise_bootstrap.json.

Usage:
  python3 scripts/gap_bootstrap.py --config config.yaml --name main --reps 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pev import config as cfg  # noqa: E402
from pev.estimator import load_blocks, profiles, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", default=None, help="comma list (default: every run in the config)")
    ap.add_argument("--name", default="main")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--point-masks", type=int, default=200, help="random-drop masks averaged for its point estimate")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    study = cfg.load(cfg.config_path(a.config))
    runs = a.runs.split(",") if a.runs else list(study.runs)
    casts = study.contrast_casts

    out = {"reps": a.reps, "real_casts": list(study.real_casts), "invented_casts": list(study.invented_casts), "runs": {}}
    per_rep, Wdraws, T0 = {}, None, None
    for run in runs:
        A, M, H, E, T, layer = load_blocks(study, run)
        if Wdraws is None:
            T0 = T
            Wdraws = np.random.default_rng(a.seed).multinomial(T, np.full(T, 1 / T), size=a.reps).astype(np.float32)
        elif T != T0:
            raise SystemExit(f"{run} has {T} premises, earlier runs {T0}: paired resampling needs the same premises")
        ones = np.ones(T, np.float32)
        Mf = {c: M[c] & ~H[c] for c in casts}
        drop = {c: (M[c] & H[c]).sum(1) for c in casts}     # per-emotion drop counts
        mrng = np.random.default_rng(a.seed + 1)

        def random_mask():
            Mr = {}
            for c in casts:
                m = M[c].copy()
                for e in np.nonzero(drop[c])[0]:
                    avail = np.nonzero(m[e])[0]
                    m[e, mrng.choice(avail, size=drop[c][e], replace=False)] = False
                Mr[c] = m
            return Mr

        pt_all = profiles(study, A, M, ones, E)
        pt_f = profiles(study, A, Mf, ones, E)
        pt_r = float(np.mean([profiles(study, A, random_mask(), ones, E)["gap"] for _ in range(a.point_masks)]))
        g_all, g_f, g_r = [], [], []
        for w in Wdraws:
            g_all.append(profiles(study, A, M, w, E)["gap"])
            g_f.append(profiles(study, A, Mf, w, E)["gap"])
            g_r.append(profiles(study, A, random_mask(), w, E)["gap"])
        g_all, g_f, g_r = map(np.asarray, (g_all, g_f, g_r))
        per_rep[run] = g_all
        out["runs"][run] = {
            "label": study.label(run), "layer": layer,
            "profile_point": {c: pt_all[c] for c in study.real_casts + study.invented_casts},
            "gap_all": summarize(pt_all["gap"], g_all),
            "gap_filtered": summarize(pt_f["gap"], g_f),
            "gap_random_drop": summarize(pt_r, g_r),
            "filtered_minus_random_drop": summarize(pt_f["gap"] - pt_r, g_f - g_r),
            "filtered_minus_all": summarize(pt_f["gap"] - pt_all["gap"], g_f - g_all),
            "stories_removed_by_filter": int(sum(drop[c].sum() for c in casts)),
            "gap_all_reps": g_all.tolist(),
        }
        s = out["runs"][run]
        print(f"{study.label(run):20s} gap {s['gap_all']['point']:+.4f} CI [{s['gap_all']['ci95'][0]:+.4f},{s['gap_all']['ci95'][1]:+.4f}] "
              f"| filtered {s['gap_filtered']['point']:+.4f} | random-drop {s['gap_random_drop']['point']:+.4f} | "
              f"filtered-random {s['filtered_minus_random_drop']['point']:+.4f} "
              f"CI [{s['filtered_minus_random_drop']['ci95'][0]:+.4f},{s['filtered_minus_random_drop']['ci95'][1]:+.4f}]", flush=True)

    out["changes"] = {}
    for arch in sorted({study.runs[r].architecture for r in runs if r in study.runs}):
        h = [r for r in runs if study.runs[r].architecture == arch and study.runs[r].condition == "persona"]
        b = [r for r in runs if study.runs[r].architecture == arch and study.runs[r].condition == "base+prompt"]
        if h and b:
            h, b = h[0], b[0]
            pt = out["runs"][h]["gap_all"]["point"] - out["runs"][b]["gap_all"]["point"]
            key = f"{arch}: persona minus base+prompt"
            out["changes"][key] = summarize(pt, per_rep[h] - per_rep[b])
            c = out["changes"][key]
            print(f"{key}: {c['point']:+.4f} CI [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]")
    od = os.path.join(study.data_root, "compare", a.name); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "premise_bootstrap.json"), "w"))
    print("wrote", os.path.join(od, "premise_bootstrap.json"))


if __name__ == "__main__":
    main()
