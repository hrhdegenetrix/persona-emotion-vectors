"""On-disk run format and loading.

<data_root>/<run>/
  meta.json      n_layers, n_embd, run spec, capture_skip, n_predict, ...
  stories.jsonl  one record per story (emotion, condition, topic_idx, story, mood, row, flags, ...)
  acts.f16       float16 rows of shape (n_layers, n_embd): the mean residual activation per layer
                 over the story's generated tokens (after capture_skip); row index = record["row"]
"""
from __future__ import annotations

import json
import os

import numpy as np

from .config import Study


def load_meta(study: Study, run: str) -> dict:
    with open(os.path.join(study.run_dir(run), "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def load_records(study: Study, run: str) -> list:
    with open(os.path.join(study.run_dir(run), "stories.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def open_acts(study: Study, run: str, meta: dict) -> np.memmap:
    return np.memmap(os.path.join(study.run_dir(run), "acts.f16"), dtype=np.float16, mode="r").reshape(
        -1, meta["n_layers"], meta["n_embd"])


def is_good(r: dict, n_predict: int) -> bool:
    """Retained for analysis: parsed JSON, not flagged bad, under the token cap."""
    return (not r.get("bad")) and bool(r.get("parsed")) and (r.get("n_tokens") or 0) < n_predict


def names_household(study: Study, r: dict) -> bool:
    """A story in a cast where no household member was requested that names one anyway."""
    return r["condition"] in study.intrusion_casts and bool(study.household_re.search(r.get("story") or ""))


def load_run(study: Study, run: str, exclude_household: bool = False, verbose: bool = True):
    meta = load_meta(study, run)
    acts = open_acts(study, run, meta)
    recs = [r for r in load_records(study, run) if r["row"] < acts.shape[0]]
    n_pred = int(meta.get("n_predict") or study.n_predict)
    good = [r for r in recs if is_good(r, n_pred)]
    if verbose and len(good) < len(recs):
        print(f"[data] {run}: excluding {len(recs) - len(good)} degenerate/truncated stories")
    if exclude_household:
        before = len(good)
        good = [r for r in good if not names_household(study, r)]
        if verbose:
            print(f"[data] {run}: household filter excluded {before - len(good)} invented-cast stories naming a household member")
    return meta, good, acts


def analysis_layer(study: Study, run: str, meta: dict) -> int:
    spec = study.runs.get(run)
    if spec is not None and spec.analysis_layer is not None:
        return int(spec.analysis_layer)
    return round(meta["n_layers"] * 2 / 3)
