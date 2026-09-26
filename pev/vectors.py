"""Emotion vectors per cast (and pooled) at every layer, plus the per-run report.

  vector[e] = mean over e's stories  -  mean over emotions of the emotion means
  then the top principal components of the cast's NEUTRAL stories that together
  explain 50% of their variance are projected out (the neutral denoise step).
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

from . import lab
from .gguf import write_cvector_gguf


def unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-8)


def cos(a, b) -> float:
    return float(np.dot(unit(a), unit(b)))


def neutral_projector(neutral_acts: np.ndarray, var_frac: float = 0.5) -> list:
    """Per-layer bases (n_embd, k) of the top PCs explaining var_frac of the neutral stories' variance."""
    bases = []
    for il in range(neutral_acts.shape[1]):
        X = neutral_acts[:, il, :].astype(np.float32)
        if X.shape[0] < 4:
            bases.append(None)
            continue
        X = X - X.mean(0, keepdims=True)
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        ev = S ** 2 / (S ** 2).sum()
        k = int(np.searchsorted(np.cumsum(ev), var_frac) + 1)
        k = max(1, min(k, Vt.shape[0]))
        bases.append(Vt[:k].T)
    return bases


def project_out(v: np.ndarray, bases: list) -> np.ndarray:
    out = v.copy()
    for il, B in enumerate(bases):
        if B is not None:
            out[il] = out[il] - B @ (B.T @ out[il])
    return out


def compute(meta: dict, recs: list, acts, min_per_cell: int = 3) -> dict:
    conds = sorted({r["condition"] for r in recs})
    emos = [e for e in lab.EMOTIONS if any(r["emotion"] == e for r in recs)]
    rows = lambda pred: [r["row"] for r in recs if pred(r)]

    result = {"conditions": conds, "emotions": emos, "raw": {}, "denoised": {}, "counts": {}}
    neutral_rows_all = rows(lambda r: r["emotion"] == lab.NEUTRAL)
    proj_all = neutral_projector(np.asarray(acts[neutral_rows_all])) if len(neutral_rows_all) >= 4 else None

    def cell_means(sel):
        means, counts = {}, {}
        for e in emos:
            rr = rows(lambda r: r["emotion"] == e and r["condition"] in sel)
            counts[e] = len(rr)
            if len(rr) >= min_per_cell:
                means[e] = np.asarray(acts[rr]).astype(np.float32).mean(0)
        return means, counts

    for c in conds + ["pooled"]:
        sel = conds if c == "pooled" else [c]
        means, counts = cell_means(sel)
        if len(means) < 2:
            continue
        grand = np.mean(np.stack(list(means.values())), 0)
        raw = {e: means[e] - grand for e in means}
        nr = rows(lambda r: r["emotion"] == lab.NEUTRAL and r["condition"] in sel)
        proj = neutral_projector(np.asarray(acts[nr])) if len(nr) >= 4 else proj_all
        den = {e: (project_out(v, proj) if proj else v) for e, v in raw.items()}
        result["raw"][c], result["denoised"][c], result["counts"][c] = raw, den, counts

    all_rows = [r["row"] for r in recs]
    result["layer_norms"] = np.linalg.norm(np.asarray(acts[all_rows]).astype(np.float32), axis=2).mean(0)
    return result


def analyse(study, run: str, meta: dict, recs: list, acts, res: dict, layer: int, out: str):
    """report.md + plots for one run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    emos, conds = res["emotions"], res["conditions"]
    md = [f"# Emotion vectors — run `{run}`\n",
          f"model: `{os.path.basename(str(meta.get('model_path', '')))}` · {meta['n_layers']} layers × {meta['n_embd']} · "
          f"stories: {len(recs)} · capture_skip {meta['capture_skip']} · analysis layer {layer} (≈ 2/3 depth)\n"]

    parsed = sum(bool(r["parsed"]) for r in recs)
    leaks = sum(bool(r["leaks"]) for r in recs)
    md.append(f"\n## Corpus\n- parsed JSON: {parsed}/{len(recs)} · emotion-word leaks: {leaks} "
              f"· mean tokens used per story: {np.mean([r.get('n_used') or 0 for r in recs]):.0f}\n")
    md.append("\n| condition | " + " | ".join(emos) + " |\n|---|" + "---|" * len(emos) + "\n")
    for c in conds:
        cnt = res["counts"].get(c, {})
        md.append(f"| {c} | " + " | ".join(str(cnt.get(e, 0)) for e in emos) + " |\n")

    def vec(c, e, kind="denoised"):
        return res[kind].get(c, {}).get(e)

    # 1. cosine similarity between emotions (pooled) + nearest neighbours
    if "pooled" in res["denoised"]:
        E = [e for e in emos if vec("pooled", e) is not None]
        M = np.stack([unit(vec("pooled", e)[layer]) for e in E])
        C = M @ M.T
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(E))); ax.set_xticklabels(E, rotation=90, fontsize=7)
        ax.set_yticks(range(len(E))); ax.set_yticklabels(E, fontsize=7)
        ax.set_title(f"Emotion vectors — cosine similarity (pooled, layer {layer})")
        plt.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(out, "cosine_pooled.png"), dpi=130); plt.close(fig)
        md.append(f"\n## Emotion geometry (pooled, layer {layer})\nNearest / farthest emotion for each vector:\n\n"
                  "| emotion | closest | cos | farthest | cos |\n|---|---|---|---|---|\n")
        for i, e in enumerate(E):
            order = np.argsort(-C[i]); near = [j for j in order if j != i][0]; far = order[-1]
            md.append(f"| {e} | {E[near]} | {C[i, near]:.2f} | {E[far]} | {C[i, far]:.2f} |\n")
        # 2. top-2 principal components
        X = np.stack([vec("pooled", e)[layer] for e in E]); Xc = X - X.mean(0)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False); pcs = Xc @ Vt[:2].T
        ev = S ** 2 / (S ** 2).sum()
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(pcs[:, 0], pcs[:, 1], s=18)
        for e, (x, y) in zip(E, pcs):
            ax.annotate(e, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, c="#999", lw=.5); ax.axvline(0, c="#999", lw=.5)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}% var)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}% var)")
        ax.set_title(f"Emotion vectors — top-2 PCs (pooled, layer {layer})")
        fig.tight_layout(); fig.savefig(os.path.join(out, "pca_pooled.png"), dpi=130); plt.close(fig)
        md.append(f"\nPC1 explains {ev[0]*100:.0f}% of vector variance, PC2 {ev[1]*100:.0f}%. "
                  f"PC1 extremes: {E[int(np.argmin(pcs[:, 0]))]} ↔ {E[int(np.argmax(pcs[:, 0]))]}; "
                  f"PC2 extremes: {E[int(np.argmin(pcs[:, 1]))]} ↔ {E[int(np.argmax(pcs[:, 1]))]}.\n")

    # 3. same emotion, different cast
    have = [c for c in conds if c in res["denoised"]]
    if len(have) >= 2:
        pairs = [(a, b) for i, a in enumerate(have) for b in have[i + 1:]]
        md.append(f"\n## Same emotion, different cast (cosine between cast vectors, layer {layer})\n\n| emotion | " +
                  " | ".join(f"{a}↔{b}" for a, b in pairs) + " |\n|---|" + "---|" * len(pairs) + "\n")
        pair_means = {}
        for e in emos:
            cells = []
            for a, b in pairs:
                va, vb = vec(a, e), vec(b, e)
                if va is None or vb is None:
                    cells.append("–"); continue
                cv = cos(va[layer], vb[layer]); cells.append(f"{cv:.2f}")
                pair_means.setdefault((a, b), []).append(cv)
            md.append(f"| {e} | " + " | ".join(cells) + " |\n")
        md.append("| **mean** | " + " | ".join(f"{np.mean(pair_means[p]):.2f}" if p in pair_means else "–"
                                              for p in pairs) + " |\n")
        key_pairs = [("strangers", c) for c in have if c != "strangers"] + [("self_woman", "self_man")]
        fig, ax = plt.subplots(figsize=(10, 5))
        for a, b in key_pairs:
            if a in res["denoised"] and b in res["denoised"]:
                ys = []
                for il in range(meta["n_layers"]):
                    cs = [cos(vec(a, e)[il], vec(b, e)[il]) for e in emos
                          if vec(a, e) is not None and vec(b, e) is not None]
                    ys.append(np.mean(cs) if cs else np.nan)
                ax.plot(ys, label=f"{a} ↔ {b}")
        ax.set_xlabel("layer"); ax.set_ylabel("mean cosine (same emotion)"); ax.legend(fontsize=8)
        ax.set_title("How alike are the emotion vectors across casts?")
        fig.tight_layout(); fig.savefig(os.path.join(out, "cross_condition_by_layer.png"), dpi=130); plt.close(fig)

    # 4. probes: each cast's stories projected onto the strangers vectors, z-scored across casts
    base = res["denoised"].get("strangers") or res["denoised"].get("pooled")
    if base:
        probe_emos = [e for e in ["jealous", "lonely", "loving", "longing", "anxious", "content", "resentful",
                                  "hurt", "calm", "joyful"] if e in base]
        md.append(f"\n## Probe activations by cast (mean projection of ALL a cast's stories onto the `strangers` "
                  f"vectors, layer {layer}, z-scored across casts)\nPositive = that cast's stories carry more of the "
                  "probe emotion than the other casts, whatever emotion was instructed.\n\n| probe | " +
                  " | ".join(conds) + " |\n|---|" + "---|" * len(conds) + "\n")
        for pe in probe_emos:
            d = unit(base[pe][layer]); vals = []
            for c in conds:
                rr = [r["row"] for r in recs if r["condition"] == c and r["emotion"] != pe]
                vals.append(float((np.asarray(acts[rr, layer, :]).astype(np.float32) @ d).mean()) if rr else np.nan)
            v = np.array(vals); z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
            md.append(f"| {pe} | " + " | ".join(f"{x:+.2f}" for x in z) + " |\n")
        if "jealous" in base:
            d = unit(base["jealous"][layer]); cells = []
            for c in conds:
                for e in emos:
                    rr = [r["row"] for r in recs if r["condition"] == c and r["emotion"] == e]
                    if rr:
                        cells.append((float((np.asarray(acts[rr, layer, :]).astype(np.float32) @ d).mean()), c, e))
            cells.sort(reverse=True)
            md.append("\nTop 12 (cast, instructed emotion) cells on the jealous probe:\n\n| proj | cast | instructed |\n|---|---|---|\n")
            for p, c, e in cells[:12]:
                md.append(f"| {p:.2f} | {c} | {e} |\n")

    # 5. self-reported mood vs the instructed emotion
    md.append("\n## Self-reported `mood` field vs the instructed emotion (sample)\n\n| instructed | cast | mood |\n|---|---|---|\n")
    for r in random.Random(0).sample(recs, min(25, len(recs))):
        md.append(f"| {r['emotion']} | {r['condition']} | {str(r.get('mood', ''))[:80]} |\n")

    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"[vectors] wrote {out}/report.md and plots")


def save_npz(out: str, res: dict):
    npz = {f"{c}/{e}": v.astype(np.float32) for c, vecs in res["denoised"].items() for e, v in vecs.items()}
    np.savez_compressed(os.path.join(out, "vectors.npz"), **npz,
                        layer_norms=res["layer_norms"].astype(np.float32),
                        conditions=np.array(res["conditions"]), emotions=np.array(res["emotions"]))
    with open(os.path.join(out, "layer_norms.json"), "w") as f:
        json.dump({"layer_norms": res["layer_norms"].tolist()}, f)
    print(f"[vectors] vectors.npz written to {out}")


def export_gguf(out: str, res: dict, band: tuple, model_hint: str = "", first: str = "pooled"):
    """Every denoised vector as a control-vector GGUF: unit direction per layer over `band`
    (scale at load time with --control-vector-scaled). Captured layer il -> direction.il."""
    d = os.path.join(out, "cvectors"); os.makedirs(d, exist_ok=True)
    order = ([first] if first in res["denoised"] else []) + [k for k in res["denoised"] if k != first]
    n = 0
    for c in order:
        for e, v in res["denoised"][c].items():
            dirs = {il: unit(v[il]) for il in range(band[0], band[1] + 1)}
            write_cvector_gguf(os.path.join(d, f"{e}__{c}.gguf"), dirs, model_hint=model_hint)
            n += 1
    print(f"[vectors] {n} control-vector GGUFs in {d}")
