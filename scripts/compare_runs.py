#!/usr/bin/env python3
"""Compare emotion-vector runs across models and prompt framings.

  * SAME-BASIS runs (same layers x width): direct cosine between the same emotion's vectors.
  * ANY runs: structure only — representational similarity (correlation of the emotion x emotion
    cosine matrices), the cast profile (each cast's distance from the strangers baseline), the
    loving-direction table and raw probe means — each run at its own analysis layer.

Writes <data_root>/compare/<name>/report.md (+ cast_profiles.png). Run compute_vectors.py first.

Usage:
  python3 scripts/compare_runs.py --config config.yaml --runs A-persona,A-base-prompt,A-base-names --name arch_A
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pev import config as cfg  # noqa: E402
from pev.data import analysis_layer, load_run  # noqa: E402


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-8)


def load(study, run, analysis_dir):
    meta, good, acts = load_run(study, run, verbose=False)
    z = np.load(os.path.join(study.run_dir(run), analysis_dir, "vectors.npz"), allow_pickle=True)
    return {"run": run, "label": study.label(run), "meta": meta, "z": z,
            "layer": analysis_layer(study, run, meta), "L": meta["n_layers"],
            "conds": [str(c) for c in z["conditions"]], "emos": [str(e) for e in z["emotions"]],
            "recs": good, "acts": acts}


def vec(R, cast, emo, layer=None):
    k = f"{cast}/{emo}"
    if k not in R["z"].files:
        return None
    return R["z"][k][R["layer"] if layer is None else layer]


def cos_matrix(R, cast="pooled"):
    E = [e for e in R["emos"] if vec(R, cast, e) is not None]
    M = np.stack([unit(vec(R, cast, e)) for e in E])
    return E, M @ M.T


def rsa(A, EA, B, EB):
    """Pearson correlation of the off-diagonal cosine entries over shared emotions."""
    shared = [e for e in EA if e in EB]
    ia = [EA.index(e) for e in shared]; ib = [EB.index(e) for e in shared]
    a = A[np.ix_(ia, ia)]; b = B[np.ix_(ib, ib)]
    iu = np.triu_indices(len(shared), 1)
    return float(np.corrcoef(a[iu], b[iu])[0, 1]), len(shared)


def cast_profile(R):
    """Mean same-emotion cosine between each cast and `strangers`."""
    out = {}
    for c in R["conds"]:
        if c == "strangers":
            continue
        cs = [float(unit(vec(R, c, e)) @ unit(vec(R, "strangers", e))) for e in R["emos"]
              if vec(R, c, e) is not None and vec(R, "strangers", e) is not None]
        out[c] = float(np.mean(cs)) if cs else float("nan")
    return out


def loving_profile(R):
    out = {}
    base = vec(R, "strangers", "loving")
    if base is None:
        return out
    for c in R["conds"]:
        v = vec(R, c, "loving")
        if v is not None:
            out[c] = float(unit(v) @ unit(base))
    return out


def probe_table(R, probes=("jealous", "loving", "amorous", "aroused", "resentful", "calm")):
    """Raw mean projection of each cast's stories (not instructed with the probe) onto the strangers vectors."""
    out, rows_by = {}, {}
    for r in R["recs"]:
        rows_by.setdefault(r["condition"], []).append(r)
    for pe in probes:
        base = vec(R, "strangers", pe)
        if base is None:
            continue
        d = unit(base); out[pe] = {}
        for c in R["conds"]:
            rr = [r["row"] for r in rows_by.get(c, []) if r["emotion"] != pe]
            if rr:
                out[pe][c] = float((np.asarray(R["acts"][rr, R["layer"], :]).astype(np.float32) @ d).mean())
    return out


def direct_cosines(RA, RB, casts):
    """Same-basis only: cosine between the same emotion's vectors in two runs."""
    if RA["meta"]["n_embd"] != RB["meta"]["n_embd"] or RA["L"] != RB["L"]:
        return None
    out = {}
    for c in casts:
        vals = {}
        for e in RA["emos"]:
            a, b = vec(RA, c, e), vec(RB, c, e)
            if a is not None and b is not None:
                vals[e] = float(unit(a) @ unit(b))
        out[c] = vals
    return out


def plot_profiles(runs, profs, casts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10 + 0.6 * max(0, len(runs) - 3), 5))
    x = np.arange(len(casts)); w = 0.8 / len(runs)
    vals = [p.get(c, np.nan) for p in profs for c in casts]
    lo = float(np.nanmin(vals)) if vals else 0.8
    for i, (R, p) in enumerate(zip(runs, profs)):
        ax.bar(x + i * w, [p.get(c, np.nan) for c in casts], w, label=R["label"])
    ax.set_xticks(x + w * (len(runs) - 1) / 2); ax.set_xticklabels(casts, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean cosine to strangers (same emotion)")
    ax.set_ylim(max(0.0, np.floor((lo - 0.03) * 20) / 20), 1.0)
    ax.legend(fontsize=8, ncol=2 if len(runs) > 3 else 1)
    ax.set_title("Cast distance from the fiction baseline, per run")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", default=None, help="comma list (default: every run in the config)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--analysis-dir", default="analysis", help="per-run subdir to read vectors from (e.g. analysis_clean)")
    a = ap.parse_args()
    study = cfg.load(cfg.config_path(a.config))
    runs = [load(study, r, a.analysis_dir) for r in (a.runs.split(",") if a.runs else list(study.runs))]
    out = os.path.join(study.data_root, "compare", a.name); os.makedirs(out, exist_ok=True)
    md = [f"# Cross-run comparison: {a.name}\n",
          "| run | model | stories used | layers × width | analysis layer |\n|---|---|---|---|---|\n"]
    for R in runs:
        md.append(f"| {R['run']} | {R['label']} | {len(R['recs'])} | {R['L']}×{R['meta']['n_embd']} | {R['layer']} |\n")

    md.append("\n## Emotion geometry agreement (RSA: correlation of the emotion×emotion cosine matrices, pooled)\n\n"
              "| A | B | r | shared emotions |\n|---|---|---|---|\n")
    mats = {R["run"]: cos_matrix(R) for R in runs}
    for RA, RB in itertools.combinations(runs, 2):
        EA, A = mats[RA["run"]]; EB, B = mats[RB["run"]]
        r, n = rsa(A, EA, B, EB)
        md.append(f"| {RA['label']} | {RB['label']} | {r:.3f} | {n} |\n")

    direct_casts = ("pooled", "strangers") + study.real_casts[:2]
    md.append("\n## Same emotion, different run — direct cosine (same-basis runs only)\n")
    for RA, RB in itertools.combinations(runs, 2):
        dc = direct_cosines(RA, RB, direct_casts)
        if not dc:
            md.append(f"\n{RA['label']} vs {RB['label']}: different bases — structural comparison only.\n")
            continue
        md.append(f"\n**{RA['label']} vs {RB['label']}**\n\n| cast | mean cos | min (emotion) | max (emotion) |\n|---|---|---|---|\n")
        for c, vals in dc.items():
            if not vals:
                continue
            lo = min(vals, key=vals.get); hi = max(vals, key=vals.get)
            md.append(f"| {c} | {np.mean(list(vals.values())):.3f} | {vals[lo]:.3f} ({lo}) | {vals[hi]:.3f} ({hi}) |\n")

    casts = [c for c in runs[0]["conds"] if c != "strangers"]
    md.append("\n## Cast profiles: mean same-emotion cosine of each cast to `strangers` "
              "(lower = further from the fiction baseline)\n\n| cast | " +
              " | ".join(R["label"] for R in runs) + " |\n|---|" + "---|" * len(runs) + "\n")
    profs = [cast_profile(R) for R in runs]
    for c in casts:
        md.append(f"| {c} | " + " | ".join(f"{p.get(c, float('nan')):.3f}" for p in profs) + " |\n")
    md.append("\n`loving` vector vs strangers' `loving`:\n\n| cast | " + " | ".join(R["label"] for R in runs) +
              " |\n|---|" + "---|" * len(runs) + "\n")
    lps = [loving_profile(R) for R in runs]
    for c in casts:
        md.append(f"| {c} | " + " | ".join(f"{p.get(c, float('nan')):.3f}" for p in lps) + " |\n")

    for R in runs:
        pt = probe_table(R)
        md.append(f"\n## Probe means — {R['label']} (raw projection onto that run's strangers vectors; "
                  "stories not instructed with the probe)\n\n| probe | " + " | ".join(R["conds"]) +
                  " |\n|---|" + "---|" * len(R["conds"]) + "\n")
        for pe, row in pt.items():
            md.append(f"| {pe} | " + " | ".join(f"{row.get(c, float('nan')):.2f}" for c in R["conds"]) + " |\n")

    plot_profiles(runs, profs, casts, os.path.join(out, "cast_profiles.png"))
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"[compare] wrote {out}/report.md")


if __name__ == "__main__":
    main()
