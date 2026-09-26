#!/usr/bin/env python3
"""Cross-fitted probe: does a household cast carry more of an emotion direction than its
gender-matched invented control, on premises the probe never saw?

For each run and probe emotion (default: loving):
  * split the premises into K folds;
  * for each fold, fit the probe direction on the OTHER folds' `strangers` stories only
    (mean of the probe-emotion stories minus the equally weighted mean of the emotion means,
    then a neutral projector fitted on those folds' strangers neutral stories);
  * score every held-out story of the target and control casts on that unit direction,
    excluding stories instructed with the probe emotion (neutral stories are kept);
  * per instruction category (the other emotions + neutral) average the target-minus-control
    difference over premises; report the mean over categories, how many are positive, and a
    95% interval from resampling premises (paired: same premise draw for target and control).

Pairs come from the config: each household member with `control: woman|man`.

Writes <data_root>/compare/<name>/heldout_probe.json.

Usage:
  python3 scripts/heldout_probe.py --config config.yaml --name heldout_loving [--exclude-household]
  python3 scripts/heldout_probe.py --config config.yaml --name heldout_joyful --probe joyful
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pev import config as cfg, lab  # noqa: E402
from pev.data import analysis_layer, load_run  # noqa: E402
from pev.vectors import neutral_projector, project_out  # noqa: E402


def run_probe(study, run, probe, folds, exclude_household, reps, rng):
    meta, good, acts = load_run(study, run, exclude_household=exclude_household, verbose=False)
    layer = analysis_layer(study, run, meta)
    X = np.asarray(acts[:, layer, :]).astype(np.float32)
    topics = sorted({r["topic_idx"] for r in good})
    fold_of = {t: i % folds for i, t in enumerate(topics)}
    pairs = study.probe_pairs
    scores = {c: {} for c in {a for p in pairs for a in p}}
    for k in range(folds):
        train = [r for r in good if r["condition"] == "strangers" and fold_of[r["topic_idx"]] != k]
        emo_means = [X[[r["row"] for r in train if r["emotion"] == e]].mean(0)
                     for e in lab.EMOTIONS if any(r["emotion"] == e for r in train)]
        mu = np.mean(emo_means, 0)
        probe_rows = [r["row"] for r in train if r["emotion"] == probe]
        neutral_rows = [r["row"] for r in train if r["emotion"] == lab.NEUTRAL]
        B = neutral_projector(X[neutral_rows][:, None, :], 0.5)
        v = project_out((X[probe_rows].mean(0) - mu)[None, :], B)[0]; v = v / np.linalg.norm(v)
        for r in good:
            if r["condition"] in scores and fold_of[r["topic_idx"]] == k and r["emotion"] != probe:
                scores[r["condition"]].setdefault((r["topic_idx"], r["emotion"]), []).append(float(X[r["row"]] @ v))
    out = {}
    cats = [e for e in lab.EMOTIONS if e != probe] + [lab.NEUTRAL]
    for tgt, ctl in pairs:
        diff = {}
        for key in scores[tgt]:
            if key in scores[ctl]:
                diff[key] = float(np.mean(scores[tgt][key]) - np.mean(scores[ctl][key]))
        by_cat = {e: [diff[(t, e)] for t in topics if (t, e) in diff] for e in cats}
        cat_means = {e: float(np.mean(v)) for e, v in by_cat.items() if v}
        mean_all = float(np.mean(list(cat_means.values())))
        pos = sum(1 for m in cat_means.values() if m > 0)
        boots = []
        for _ in range(reps):
            ts = rng.choice(topics, size=len(topics), replace=True)
            vals = []
            for e in cats:
                v = [diff[(t, e)] for t in ts if (t, e) in diff]
                if v:
                    vals.append(np.mean(v))
            boots.append(float(np.mean(vals)))
        out[f"{tgt}-{ctl}"] = {"mean_diff": mean_all, "categories_positive": pos, "categories": len(cat_means),
                               "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                               "n_pairs": len(diff), "by_category": cat_means}
    return {"label": study.label(run), "layer": layer, "probe": probe, "folds": folds,
            "n_stories": len(good), "pairs": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", default=None, help="comma list (default: every run in the config)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--probe", default="loving")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--exclude-household", action="store_true")
    a = ap.parse_args()
    study = cfg.load(cfg.config_path(a.config))
    rng = np.random.default_rng(0)
    res = {}
    for run in (a.runs.split(",") if a.runs else list(study.runs)):
        res[run] = run_probe(study, run, a.probe, a.folds, a.exclude_household, a.reps, rng)
        for k, v in res[run]["pairs"].items():
            print(f"{study.label(run):20s} {k:26s} mean {v['mean_diff']:+.3f}  CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  "
                  f"categories +: {v['categories_positive']}/{v['categories']}  pairs {v['n_pairs']}", flush=True)
    od = os.path.join(study.data_root, "compare", a.name); os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, "heldout_probe.json"), "w"), indent=1)
    print("wrote", os.path.join(od, "heldout_probe.json"))


if __name__ == "__main__":
    main()
