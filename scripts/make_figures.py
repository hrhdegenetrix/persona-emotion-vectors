#!/usr/bin/env python3
"""Figures from SAVED results only, so figures and tables cannot drift apart.

  pca_<run>.png        top two PCs of the pooled emotion vectors, one per persona run
  cast_profiles.png    each cast's similarity to the strangers baseline, every run (dot plot)
  contrast.png         household-minus-invented distance excess with premise-bootstrap intervals
  intrusion.png        household names in invented-companion stories, per cast and run

Encoding: colour = condition (persona / base+prompt / base+names), marker = architecture.
Reads compare/<name>/{premise_bootstrap,intrusion_rates}.json and each run's analysis/vectors.npz.
Writes <data_root>/figures/<name>/.

Usage:
  python3 scripts/make_figures.py --config config.yaml --name main
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pev import config as cfg  # noqa: E402
from pev.data import analysis_layer, load_meta  # noqa: E402

INK, INK2, GRID = "#1f2529", "#5a646c", "#e3e6e8"
COND = {"persona": "#2a78d6", "base+prompt": "#eb6834", "base+names": "#1baf7a"}
MARKS = ["o", "s", "^", "D", "v", "P"]
ORDER = {"base+names": 0, "base+prompt": 1, "persona": 2}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titleweight": "bold",
                     "axes.titlesize": 10.5, "savefig.facecolor": "white"})


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


def repel(xy, iters=300, step=0.02, min_d=0.32):
    p = xy.copy()
    for _ in range(iters):
        moved = False
        for i in range(len(p)):
            d = p[i] - p; n = np.linalg.norm(d, axis=1); n[i] = 9
            close = n < min_d
            if close.any():
                p[i] += step * (d[close] / np.maximum(n[close, None], 1e-3)).sum(0); moved = True
        if not moved:
            break
    return p


def cast_label(study, c):
    if c == "self_stranger":
        return f"{study.persona.name} + unnamed invented person"
    if c == "self_woman":
        return f"{study.persona.name} + {study.invented_woman} (invented)"
    if c == "self_man":
        return f"{study.persona.name} + {study.invented_man} (invented)"
    if c == "strangers":
        return "two invented strangers"
    p = {f"self_{h.key}": h.name for h in study.household}.get(c)
    return f"{study.persona.name} + {p}" if p else c


def fig_pca(study, run, out):
    meta = load_meta(study, run); L = analysis_layer(study, run, meta)
    z = np.load(os.path.join(study.run_dir(run), "analysis", "vectors.npz"), allow_pickle=True)
    E = [str(e) for e in z["emotions"]]
    X = np.stack([z[f"pooled/{e}"][L] for e in E]); X = X - X.mean(0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False); ev = S ** 2 / (S ** 2).sum()
    P = X @ Vt[:2].T; P = P / P.std(0)
    if np.corrcoef(P[:, 0], [1 if e in ("joyful", "content", "grateful", "loving") else 0 for e in E])[0, 1] < 0:
        P[:, 0] *= -1                                        # orient PC1 so pleasant is on the right
    fig, ax = plt.subplots(figsize=(7.2, 5.6)); style(ax, "both")
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    ax.scatter(P[:, 0], P[:, 1], s=26, color=COND["persona"], edgecolor="white", lw=0.8, zorder=3)
    lab_pos = repel(P + np.array([0.06, 0.06]))
    for (x, y), (lx, ly), e in zip(P, lab_pos, E):
        ax.annotate(e, (x, y), (lx, ly), fontsize=7.6, color=INK,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6) if np.hypot(lx - x, ly - y) > 0.2 else None)
    ax.set_xlabel(f"PC1, {ev[0]*100:.0f}% of variance  (unpleasant ← → pleasant)")
    ax.set_ylabel(f"PC2, {ev[1]*100:.0f}% of variance")
    ax.set_title(f"{study.label(run)}: emotion vectors, top two principal components (layer {L})", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"pca_{run}.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="main")
    a = ap.parse_args()
    study = cfg.load(cfg.config_path(a.config))
    cmp_dir = os.path.join(study.data_root, "compare", a.name)
    out = os.path.join(study.data_root, "figures", a.name); os.makedirs(out, exist_ok=True)
    boot = json.load(open(os.path.join(cmp_dir, "premise_bootstrap.json")))
    R = boot["runs"]
    archs = sorted({study.runs[r].architecture for r in R})
    mark = {arch: MARKS[i % len(MARKS)] for i, arch in enumerate(archs)}
    runs = sorted(R, key=lambda r: (study.runs[r].architecture, ORDER[study.runs[r].condition]))
    handles = [Line2D([], [], ls="", marker=mark[study.runs[r].architecture], ms=7,
                      mfc=COND[study.runs[r].condition], mec="white", mew=1.2, label=study.label(r)) for r in runs]

    for r in runs:
        if study.runs[r].condition == "persona" and os.path.exists(os.path.join(study.run_dir(r), "analysis", "vectors.npz")):
            fig_pca(study, r, out)

    # cast profiles, casts as rows
    casts = list(study.invented_casts) + list(study.real_casts)
    fig, ax = plt.subplots(figsize=(7.8, 1.2 + 0.65 * len(casts))); style(ax, "x")
    offs = np.linspace(-0.27, 0.27, len(runs))
    for j, r in enumerate(runs):
        ys = np.arange(len(casts))[::-1] + offs[j]
        ax.scatter([R[r]["profile_point"][c] for c in casts], ys, marker=mark[study.runs[r].architecture], s=34,
                   color=COND[study.runs[r].condition], edgecolor="white", lw=1, zorder=3)
    ax.axhspan(-0.5, len(study.real_casts) - 0.5, color="#f3f5f6", zorder=0)
    ax.set_yticks(np.arange(len(casts))[::-1]); ax.set_yticklabels([cast_label(study, c) for c in casts])
    ax.set_ylim(-0.6, len(casts) - 0.4)
    ax.set_xlabel("mean cosine similarity to the strangers baseline, same emotion\n(lower = further from generic fiction)")
    ax.set_title("How close each cast sits to the strangers baseline", loc="left", pad=16)
    ax.text(0.0, 1.01, "shaded rows: household members", transform=ax.transAxes, fontsize=8, color=INK2, ha="left", va="bottom")
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(out, "cast_profiles.png"), dpi=150); plt.close(fig)

    # contrast with intervals, runs as rows
    fig, ax = plt.subplots(figsize=(7.6, 1.5 + 0.4 * len(runs))); style(ax, "x")
    for i, r in enumerate(runs[::-1]):
        g = R[r]["gap_all"]; p = -g["point"]; lo, hi = -g["ci95"][1], -g["ci95"][0]
        col = COND[study.runs[r].condition]
        ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([p], [i], marker=mark[study.runs[r].architecture], s=52, color=col, edgecolor="white", lw=1.2, zorder=3)
        ax.text(max(hi, 0) + 0.0015, i, f"{p:+.3f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(range(len(runs))); ax.set_yticklabels([study.label(r) for r in runs[::-1]])
    ax.set_xlabel("excess cosine distance of household casts over invented-companion casts\n"
                  "(positive = the household members sit further from the strangers baseline)")
    ax.set_title("The household-versus-invented contrast, per run", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "contrast.png"), dpi=150); plt.close(fig)

    # intrusion rates
    ip = os.path.join(cmp_dir, "intrusion_rates.json")
    if os.path.exists(ip):
        rates = json.load(open(ip))
        icasts = [c for c in study.intrusion_casts if any(c in rates.get(r, {}) for r in runs)]
        fig, ax = plt.subplots(figsize=(8.4, 4.3)); style(ax, "y")
        x = np.arange(len(icasts)); w = 0.8 / len(runs)
        top = 0
        for i, r in enumerate(runs):
            vals = [rates.get(r, {}).get(c, 0.0) for c in icasts]; top = max([top] + vals)
            xs = x + (i - (len(runs) - 1) / 2) * w
            ax.bar(xs, vals, w * 0.9, color=COND[study.runs[r].condition],
                   hatch="////" if archs.index(study.runs[r].architecture) % 2 else None,
                   edgecolor="white", lw=0.6, label=study.label(r))
        ax.set_xticks(x); ax.set_xticklabels([cast_label(study, c) for c in icasts], fontsize=8)
        ax.set_ylabel("stories naming a household member (%)"); ax.set_ylim(0, max(10, top * 1.15))
        ax.set_title("Household members in stories where none was requested", loc="left")
        ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper right")
        fig.tight_layout(); fig.savefig(os.path.join(out, "intrusion.png"), dpi=150); plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main()
