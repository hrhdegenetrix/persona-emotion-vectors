#!/usr/bin/env python3
"""Steer a persona run with one of its emotion vectors and compare against a baseline.

Adds  strength x mean_residual_norm(layer) x unit_vector(layer)  to the residual stream at each
layer of the band on every decode step (strengths are relative to the mean residual norm, as in
the original study). Uses the patched server's POST /activations/steer "add" mode, which is live
(no restart) and is always cleared in a finally block.

Only point this at a server you are free to experiment on.

Usage:
  python3 scripts/steer_demo.py --config config.yaml --run A-persona --emotion calm --strength 0.5 --band 12-20
  python3 scripts/steer_demo.py --config config.yaml --run A-persona --clear
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pev import config as cfg, lab  # noqa: E402


def steer(url, vec, norms, band, strength):
    add = []
    for il in range(band[0], band[1] + 1):
        u = vec[il] / max(float(np.linalg.norm(vec[il])), 1e-8)
        add.append({"layer": il, "vector": (u * float(norms[il]) * strength).astype(np.float32).tolist(), "scale": 1.0})
    r = requests.post(f"{url}/activations/steer", json={"add": add}, timeout=60).json()
    if not r.get("add_active"):
        raise SystemExit(f"steer failed: {r}")
    return r


def clear(url):
    return requests.post(f"{url}/activations/steer", json={"clear_add": True}, timeout=30).json()


def ask(study, run, url, prompt_text, n_predict=160, seed=7, temperature=0.8):
    """The run's own prompt form with a free question in the TASK slot; JSON answer."""
    rules = "Answer the question in the task in one or two sentences, as a JSON object with one key, \"answer\"."
    if run.scaffold:
        body = lab.fill_scaffold(study, rules, prompt_text)
        if run.fmt == "flat":
            prompt = body + run.marker
        else:
            prompt = lab.apply_template(url, [{"role": "system", "content": study.scaffold_system},
                                              {"role": "user", "content": body}], run.apply_template_extra)
    else:
        prompt = lab.apply_template(url, [{"role": "user", "content": f"{prompt_text}\n{rules}"}], run.apply_template_extra)
    payload = {"prompt": prompt, "n_predict": n_predict, "stream": False, "cache_prompt": True,
               "stop": list(run.stop), "json_schema": {"type": "object"}, "seed": seed,
               **{**study.sampling, "temperature": temperature}}
    return requests.post(f"{url}/completion", json=payload, timeout=300).json().get("content", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--emotion")
    ap.add_argument("--condition", default="pooled", help="which cast's vector to use (default pooled)")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--band", default=None, help="e.g. 12-20 (default: 40-67%% of depth)")
    ap.add_argument("--prompt", default="How are you feeling right now?")
    ap.add_argument("--url", default=None)
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    study = cfg.load(cfg.config_path(args.config))
    run = study.run(args.run)
    url = args.url or study.server_url
    if args.clear:
        print(clear(url)); return
    if not args.emotion:
        ap.error("--emotion required")
    z = np.load(os.path.join(study.run_dir(run.name), "analysis", "vectors.npz"), allow_pickle=True)
    key = f"{args.condition}/{args.emotion}"
    if key not in z.files:
        raise SystemExit(f"{key} not in vectors.npz")
    vec, norms = z[key], z["layer_norms"]
    L = vec.shape[0]
    band = tuple(int(x) for x in args.band.split("-")) if args.band else (round(L * 0.4), round(L * 0.67))
    print("== baseline ==")
    for s in range(args.seeds):
        print(f"[seed {s}] {ask(study, run, url, args.prompt, seed=s)}\n")
    try:
        r = steer(url, vec, norms, band, args.strength)
        print(f"== steered: {key} x {args.strength} at layers {band[0]}-{band[1]} ({json.dumps(r)}) ==")
        for s in range(args.seeds):
            print(f"[seed {s}] {ask(study, run, url, args.prompt, seed=s)}\n")
    finally:
        print("cleared:", clear(url))


if __name__ == "__main__":
    main()
