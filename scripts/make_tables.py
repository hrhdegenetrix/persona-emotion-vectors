#!/usr/bin/env python3
"""Markdown result tables from SAVED results: <data_root>/compare/<name>/tables.md.

Reads (whichever exist) in compare/<name>/:
  premise_bootstrap.json   contrast, leakage filter vs random removal, persona-minus-prompt change
  intrusion_rates.json     household names in invented-companion stories
  reliability.json         split-half reliability
and, for the held-out probes, compare/<probe_name>/heldout_probe.json for each --probe-names entry
(optionally with a "<probe_name>_clean" variant produced with --exclude-household).

Usage:
  python3 scripts/make_tables.py --config config.yaml --name main --probe-names heldout_loving,heldout_joyful
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pev import config as cfg  # noqa: E402


def f(x, d=3):
    return f"{x:+.{d}f}".replace("-", "−")


def ci(c, d=3):
    return f"{f(c[0], d)} to {f(c[1], d)}"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="main")
    ap.add_argument("--probe-names", default="heldout_loving")
    a = ap.parse_args()
    study = cfg.load(cfg.config_path(a.config))
    C = os.path.join(study.data_root, "compare")
    L = study.label
    md = [f"# Results tables — {a.name}\n"]

    B = load(os.path.join(C, a.name, "premise_bootstrap.json"))
    if B:
        R = B["runs"]
        md += ["\n## Household-minus-invented contrast (negative = household further from the baseline)\n",
               "| run | contrast | 95% CI | z |", "|---|---|---|---|"]
        for r in R:
            g = R[r]["gap_all"]
            md.append(f"| {L(r)} | {f(g['point'])} | {ci(g['ci95'])} | {abs(g['z'] or 0):.1f} |")
        md += ["\n## Leakage filter against random removal of the same number of stories\n",
               "| run | stories removed | contrast after the filter | after random removal | filter minus random removal | 95% CI |",
               "|---|---|---|---|---|---|"]
        for r in R:
            s = R[r]
            md.append(f"| {L(r)} | {s['stories_removed_by_filter']:,} | {f(s['gap_filtered']['point'])} | "
                      f"{f(s['gap_random_drop']['point'])} | {f(s['filtered_minus_random_drop']['point'])} | "
                      f"{ci(s['filtered_minus_random_drop']['ci95'])} |")
        if B.get("changes"):
            md += ["\n## Change in the contrast from the prompted base to the persona model (premise-paired)\n",
                   "| comparison | change | 95% CI |", "|---|---|---|"]
            for k, c in B["changes"].items():
                md.append(f"| {k} | {f(c['point'])} | {ci(c['ci95'])} |")
        md += ["\n## Cast profiles (mean same-emotion cosine to the strangers baseline)\n",
               "| cast | " + " | ".join(L(r) for r in R) + " |", "|---|" + "---|" * len(R)]
        for c in list(study.invented_casts) + list(study.real_casts):
            md.append(f"| {c} | " + " | ".join(f"{R[r]['profile_point'][c]:.3f}" for r in R) + " |")

    for pn in [p for p in a.probe_names.split(",") if p]:
        H = load(os.path.join(C, pn, "heldout_probe.json"))
        if not H:
            continue
        Hc = load(os.path.join(C, pn + "_clean", "heldout_probe.json"))
        probe = next(iter(H.values()))["probe"]
        head = "| run and contrast | mean over categories | 95% CI (premise-paired) | categories with a positive average |"
        sep = "|---|---|---|---|"
        if Hc:
            head += " after the filter: mean (categories positive) |"; sep += "---|"
        md += [f"\n## Held-out `{probe}` probe ({pn})\n", head, sep]
        for r, v0 in H.items():
            for k, v in v0["pairs"].items():
                row = (f"| {L(r)}: {k} | {f(v['mean_diff'], 2)} | {ci(v['ci95'], 2)} | "
                       f"{v['categories_positive']} / {v['categories']} |")
                if Hc and r in Hc and k in Hc[r]["pairs"]:
                    vc = Hc[r]["pairs"][k]
                    row += f" {f(vc['mean_diff'], 2)} ({vc['categories_positive']} / {vc['categories']}) |"
                md.append(row)

    I = load(os.path.join(C, a.name, "intrusion_rates.json"))
    if I:
        runs = list(I)
        md += ["\n## Stories naming a household member where none was requested\n",
               "| requested cast | " + " | ".join(L(r) for r in runs) + " |", "|---|" + "---|" * len(runs)]
        for c in study.intrusion_casts:
            md.append(f"| {c} | " + " | ".join(f"{I[r].get(c, float('nan')):.1f}%" for r in runs) + " |")

    Rl = load(os.path.join(C, a.name, "reliability.json"))
    if Rl:
        md += ["\n## Split-half reliability\n", "| run | split-half cosine |", "|---|---|"]
        for r, v in Rl.items():
            md.append(f"| {L(r)} | {v['mean']:.3f} |")

    path = os.path.join(C, a.name, "tables.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
