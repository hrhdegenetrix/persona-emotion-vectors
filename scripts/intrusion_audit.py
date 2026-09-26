#!/usr/bin/env python3
"""Leakage audit: in casts where no household member was requested, how often does the story
name one anyway? Literal, word-boundary name match (unnamed references are missed, and common
names can match ordinary uses).

Writes <data_root>/compare/<name>/intrusion_rates.json: {run: {cast: percent}}.

Usage:
  python3 scripts/intrusion_audit.py --config config.yaml --name main
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pev import config as cfg  # noqa: E402
from pev.data import load_run  # noqa: E402


def rates(study, run):
    _, good, _ = load_run(study, run, verbose=False)
    n, h = {}, {}
    for r in good:
        c = r["condition"]
        n[c] = n.get(c, 0) + 1
        if study.household_re.search(r.get("story") or ""):
            h[c] = h.get(c, 0) + 1
    return {c: 100 * h.get(c, 0) / n[c] for c in study.intrusion_casts if n.get(c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--name", default="main")
    a = ap.parse_args()
    study = cfg.load(cfg.config_path(a.config))
    out = {run: rates(study, run) for run in (a.runs.split(",") if a.runs else list(study.runs))}
    for run, rr in out.items():
        print(f"{study.label(run):20s} " + "  ".join(f"{c} {v:.1f}%" for c, v in rr.items()))
    od = os.path.join(study.data_root, "compare", a.name); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "intrusion_rates.json"), "w"), indent=1)
    print("wrote", os.path.join(od, "intrusion_rates.json"))


if __name__ == "__main__":
    main()
