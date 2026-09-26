#!/usr/bin/env python3
"""Compute one run's emotion vectors (per cast and pooled, every layer) and its report.

Outputs in <run>/analysis/ (or analysis_clean/ with --exclude-household):
  vectors.npz        denoised vectors "<cast>/<emotion>" -> (n_layers, n_embd), plus layer_norms
  layer_norms.json   mean residual norm per layer (the unit for steering strength)
  report.md, *.png   geometry, cross-cast cosines, probe tables, self-report sample
  cvectors/*.gguf    control-vector export (skip with --no-export)

Usage:
  python3 scripts/compute_vectors.py --config config.yaml --run A-persona [--layer 20] [--band 12-24]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pev import config as cfg, vectors  # noqa: E402
from pev.data import analysis_layer, load_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer", type=int, default=None, help="analysis layer (default: run's analysis_layer or round(2/3 depth))")
    ap.add_argument("--band", default=None, help="layer band for the GGUF export, e.g. 12-24 (default: middle third-ish)")
    ap.add_argument("--min-per-cell", type=int, default=3)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--exclude-household", action="store_true",
                    help="drop invented-cast stories that name a household member (leakage filter)")
    ap.add_argument("--analysis-dir", default=None)
    args = ap.parse_args()

    study = cfg.load(cfg.config_path(args.config))
    meta, recs, acts = load_run(study, args.run, exclude_household=args.exclude_household)
    layer = args.layer if args.layer is not None else analysis_layer(study, args.run, meta)
    L = meta["n_layers"]
    band = tuple(int(x) for x in args.band.split("-")) if args.band else (round(L * 0.4), round(L * 0.8))
    out = os.path.join(study.run_dir(args.run),
                       args.analysis_dir or ("analysis_clean" if args.exclude_household else "analysis"))
    print(f"[vectors] {args.run}: {len(recs)} stories, acts {acts.shape}, layer {layer} -> {out}")
    res = vectors.compute(meta, recs, acts, args.min_per_cell)
    vectors.analyse(study, args.run, meta, recs, acts, res, layer, out)
    vectors.save_npz(out, res)
    if not args.no_export:
        vectors.export_gguf(out, res, band, model_hint=study.runs[args.run].architecture if args.run in study.runs else "")


if __name__ == "__main__":
    main()
