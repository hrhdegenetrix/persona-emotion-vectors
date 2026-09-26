"""The single estimator behind every contrast, interval and reliability figure.

Per run and cast, at the analysis layer:
  * each emotion's mean activation over its stories (weighted by premise weights);
  * the cast's centre = the EQUALLY weighted mean of the emotion means (neutral excluded);
  * each emotion's vector = its mean minus that centre;
  * the principal components of the cast's own neutral stories that together explain 50% of
    their variance are projected out (refitted under every weighting, so resampling intervals
    include projector uncertainty).

Cast profile = mean over emotions of cos(cast vector, strangers vector).
Contrast ("gap") = mean profile of the household casts minus that of the invented-companion casts;
negative = the household members sit further from the fiction baseline.

The design has one story per (cast, emotion, premise), so premise weights select whole blocks.
"""
from __future__ import annotations

import numpy as np

from . import lab
from .config import Study
from .data import analysis_layer, load_run, names_household


def load_blocks(study: Study, run: str):
    """Arrays indexed [emotion, premise] for every contrast cast at the analysis layer.
    Returns A (E,T,W) activations, M (E,T) present, H (E,T) names-a-household-member, E, T, layer.
    The design is one story per (cast, emotion, premise); with --reps > 1 only rep 0 is used here."""
    meta, good, acts = load_run(study, run, verbose=False)
    layer = analysis_layer(study, run, meta)
    W = meta["n_embd"]
    T = int(meta.get("topics") or study.n_premises)
    emos = list(lab.EMOTIONS) + [lab.NEUTRAL]; E = len(emos)
    casts = study.contrast_casts
    A = {c: np.zeros((E, T, W), np.float32) for c in casts}
    M = {c: np.zeros((E, T), bool) for c in casts}
    H = {c: np.zeros((E, T), bool) for c in casts}
    for r in good:
        c = r["condition"]
        if c not in A or r.get("rep", 0) != 0 or r["topic_idx"] >= T:
            continue
        e = emos.index(r["emotion"]); t = r["topic_idx"]
        A[c][e, t] = np.asarray(acts[r["row"], layer], dtype=np.float32)
        M[c][e, t] = True
        H[c][e, t] = names_household(study, r)
    return A, M, H, E, T, layer


def cast_vectors(Ac: np.ndarray, Mc: np.ndarray, w: np.ndarray, E: int):
    """Canonical estimator for one cast. w: (T,) premise weights (bootstrap counts, or 0/1 halves)."""
    ne = E - 1                                            # neutral is the last index
    Wt = Mc * w[None, :]
    den = Wt.sum(1)
    ok = den[:ne] >= 1
    means = np.einsum("et,etd->ed", Wt[:ne], Ac[:ne]) / np.maximum(den[:ne], 1e-9)[:, None]
    grand = means[ok].mean(0)
    raw = means - grand
    # weighted PCA of this cast's neutral stories, via the small Gram matrix
    wn = Wt[ne]; idx = wn > 0
    if idx.sum() < 2:                                     # degenerate draw: no neutral stories to fit a projector
        return raw, ok
    Xn = Ac[ne][idx]; wv = wn[idx]
    mu = (wv[:, None] * Xn).sum(0) / wv.sum()
    Xc = np.sqrt(wv)[:, None] * (Xn - mu)
    G = Xc @ Xc.T
    ev, U = np.linalg.eigh(G); ev = ev[::-1]; U = U[:, ::-1]
    pos = ev > 1e-6 * ev[0]
    ev, U = ev[pos], U[:, pos]
    frac = ev / ev.sum()
    k = int(np.searchsorted(np.cumsum(frac), 0.5) + 1); k = max(1, min(k, len(ev)))
    V = (U[:, :k].T @ Xc) / np.sqrt(ev[:k])[:, None]     # (k, W) orthonormal rows
    return raw - (raw @ V.T) @ V, ok


def profiles(study: Study, A, M, w, E) -> dict:
    V, OK = {}, {}
    for c in study.contrast_casts:
        V[c], OK[c] = cast_vectors(A[c], M[c], w, E)
    u = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    Us = u(V["strangers"])
    prof = {}
    for c in study.real_casts + study.invented_casts:
        both = OK[c] & OK["strangers"]
        prof[c] = float((u(V[c])[both] * Us[both]).sum(1).mean())
    prof["gap"] = float(np.mean([prof[c] for c in study.real_casts]) - np.mean([prof[c] for c in study.invented_casts]))
    return prof


def summarize(point: float, boots) -> dict:
    b = np.asarray(boots)
    sd = float(b.std(ddof=1)) if len(b) > 1 else 0.0
    return {"point": point, "se": sd, "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "z": float(point / sd) if sd > 0 else None, "reps": int(len(b))}
