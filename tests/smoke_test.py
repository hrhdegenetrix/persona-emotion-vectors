#!/usr/bin/env python3
"""End-to-end smoke test of the analysis chain on SYNTHETIC data (no server, no model).

Fabricates six runs (two architectures x persona / base+prompt / base+names) in the on-disk
format generate_stories.py writes, with planted structure: every emotion has a direction,
household casts are shifted in informed runs, and some invented-cast stories name a household
member. Then runs every analysis script and checks its outputs exist and parse.

Usage: python3 tests/smoke_test.py [--keep]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pev import config as cfg, lab  # noqa: E402

N_LAYERS, N_EMBD, T = 6, 48, 10

CONFIG = """
study: {{name: smoke, data_root: {root}}}
server: {{url: http://127.0.0.1:1}}
persona: {{name: Ava, noun: woman, pronoun: she, scaffold: {scaffold}}}
household:
  - {{key: sam, name: Sam, noun: man, pronoun: he, control: man}}
  - {{key: robin, name: Robin, noun: woman, pronoun: she, control: woman}}
invented_controls: {{woman: Clara, man: Daniel}}
absent_casts:
  - {{bearer: sam, with: stranger}}
generation: {{premises: {T}}}
runs:
  - {{name: A-persona, label: A-Persona, architecture: A, condition: persona, fmt: flat, marker: "\\n\\nAnswer:\\n"}}
  - {{name: A-base-prompt, label: A-Base+Prompt, architecture: A, condition: base+prompt, fmt: chat}}
  - {{name: A-base-names, label: A-Base+Names, architecture: A, condition: base+names, fmt: chat}}
  - {{name: B-persona, label: B-Persona, architecture: B, condition: persona, fmt: flat, marker: "\\n\\nAnswer:\\n"}}
  - {{name: B-base-prompt, label: B-Base+Prompt, architecture: B, condition: base+prompt, fmt: chat}}
  - {{name: B-base-names, label: B-Base+Names, architecture: B, condition: base+names, fmt: chat}}
"""


def fabricate(study, run, rng):
    spec = study.runs[run]
    d = study.run_dir(run); os.makedirs(d, exist_ok=True)
    emos = lab.EMOTIONS + [lab.NEUTRAL]
    emo_dir = {e: rng.normal(size=(N_LAYERS, N_EMBD)) for e in emos}
    # cast x emotion interaction (a constant per-cast offset would cancel in the centring)
    cast_emo = {(c, e): rng.normal(size=(N_LAYERS, N_EMBD)) * 0.6 for c in study.casts for e in emos}
    informed = spec.condition != "base+names"
    recs, rows = [], []
    for e in emos:
        for c, cast in study.casts.items():
            for t in range(T):
                shift = cast_emo[(c, e)] * (2.0 if informed and cast.kind == "household" else 1.0)
                x = 3.0 * emo_dir[e] + shift + rng.normal(size=(N_LAYERS, N_EMBD))
                rows.append(x.astype(np.float16))
                intr = c in study.intrusion_casts and spec.condition == "persona" and rng.random() < 0.3
                story = ("Later, Sam called. " if intr else "") + f"A {e} story about {c}, premise {t}."
                recs.append({"key": f"{e}|{c}|{t}|0", "row": len(rows) - 1, "emotion": e, "condition": c,
                             "topic_idx": t, "topic": lab.PREMISES[t], "rep": 0, "mood": "settled",
                             "story": story, "parsed": True, "bad": False, "leaks": [],
                             "n_tokens": 200, "n_used": 176, "skip": 24})
    # one degenerate story, which every loader must exclude
    recs[5] = {**recs[5], "bad": True, "parsed": False}
    np.stack(rows).tofile(os.path.join(d, "acts.f16"))
    with open(os.path.join(d, "stories.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump({"n_layers": N_LAYERS, "n_embd": N_EMBD, "model_path": "synthetic.gguf", "run": run,
               "condition": spec.condition, "capture_skip": 24, "n_predict": 700, "topics": T, "reps": 1},
              open(os.path.join(d, "meta.json"), "w"))


def sh(*args):
    cmd = [sys.executable, *args]
    print("$", " ".join(os.path.relpath(a, ROOT) if a.startswith(ROOT) else a for a in args), flush=True)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(args)}")
    tail = [line for line in p.stdout.strip().splitlines() if line][-2:]
    for line in tail:
        print("   ", line)


def check_gguf(path):
    b = open(path, "rb").read()
    assert b[:4] == b"GGUF", path
    version, n_tensors, n_kv = np.frombuffer(b[4:24], dtype="<u4", count=1)[0], *np.frombuffer(b[8:24], dtype="<u8")
    assert version == 3 and n_kv == 3 and n_tensors > 0, (version, n_tensors, n_kv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="pev_smoke_")
    try:
        conf = os.path.join(tmp, "config.yaml")
        with open(conf, "w") as f:
            f.write(CONFIG.format(root=os.path.join(tmp, "data"), T=T,
                                  scaffold=os.path.join(ROOT, "examples", "scaffold.example.json")))
        study = cfg.load(conf)
        rng = np.random.default_rng(0)
        for run in study.runs:
            fabricate(study, run, rng)
        # prompt rendering for every cast, both prompt forms
        for c in study.casts.values():
            for e in ("loving", lab.NEUTRAL):
                assert json.loads(lab.render_scaffold(study, e, c, lab.PREMISES[0]))
                assert lab.render_plain_user(e, c, lab.PREMISES[0])

        S = lambda name: os.path.join(ROOT, "scripts", name)
        for i, run in enumerate(study.runs):
            extra = ["--band", "1-4"] if i == 0 else ["--no-export"]
            sh(S("compute_vectors.py"), "--config", conf, "--run", run, *extra)
        sh(S("compute_vectors.py"), "--config", conf, "--run", "A-persona", "--exclude-household", "--no-export")
        sh(S("compare_runs.py"), "--config", conf, "--name", "main")
        sh(S("gap_bootstrap.py"), "--config", conf, "--name", "main", "--reps", "40", "--point-masks", "5")
        sh(S("heldout_probe.py"), "--config", conf, "--name", "heldout_loving", "--reps", "40")
        sh(S("heldout_probe.py"), "--config", conf, "--name", "heldout_loving_clean", "--reps", "40", "--exclude-household")
        sh(S("heldout_probe.py"), "--config", conf, "--name", "heldout_joyful", "--probe", "joyful", "--reps", "40",
           "--runs", "A-persona,B-persona")
        sh(S("reliability.py"), "--config", conf, "--name", "main", "--splits", "3")
        sh(S("intrusion_audit.py"), "--config", conf, "--name", "main")
        sh(S("make_figures.py"), "--config", conf, "--name", "main")
        sh(S("make_tables.py"), "--config", conf, "--name", "main", "--probe-names", "heldout_loving,heldout_joyful")
        sh(S("filter_run.py"), os.path.join(study.data_root, "A-persona"), "--dry-run")

        D = study.data_root
        for run in study.runs:
            z = np.load(os.path.join(D, run, "analysis", "vectors.npz"))
            assert z["pooled/loving"].shape == (N_LAYERS, N_EMBD)
        ggufs = os.listdir(os.path.join(D, "A-persona", "analysis", "cvectors"))
        assert ggufs, "no GGUF export"
        check_gguf(os.path.join(D, "A-persona", "analysis", "cvectors", ggufs[0]))
        boot = json.load(open(os.path.join(D, "compare", "main", "premise_bootstrap.json")))
        assert set(boot["runs"]) == set(study.runs) and len(boot["changes"]) == 2
        for run in ("A-persona", "B-persona", "A-base-prompt"):
            assert boot["runs"][run]["gap_all"]["point"] < 0, (run, "planted household shift not recovered")
        for f in ("cast_profiles.png", "contrast.png", "intrusion.png", "pca_A-persona.png"):
            assert os.path.getsize(os.path.join(D, "figures", "main", f)) > 1000, f
        tables = open(os.path.join(D, "compare", "main", "tables.md")).read()
        assert "Held-out `loving` probe" in tables and "Split-half reliability" in tables
        rates = json.load(open(os.path.join(D, "compare", "main", "intrusion_rates.json")))
        assert rates["A-persona"]["self_stranger"] > 0 and rates["A-base-names"]["self_stranger"] == 0
        print(f"\nSMOKE TEST PASSED ({len(study.runs)} runs, {len(study.casts)} casts, "
              f"{len(lab.EMOTIONS) + 1} instructions, {T} premises)")
    finally:
        if a.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
