#!/usr/bin/env python3
"""Generate one run's story corpus and per-story activation means.

Talks to a patched llama-server (see patches/) directly over HTTP. Resumable:
every finished story is appended to <run>/stories.jsonl and its per-layer means
to <run>/acts.f16 (row index in the record); re-running skips finished jobs.

Usage:
  python3 scripts/generate_stories.py --config config.yaml --run A-persona
  python3 scripts/generate_stories.py --config config.yaml --run A-persona --premises 1 \
      --emotions loving,sad,neutral --conditions strangers,self_sam          # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pev import config as cfg, lab  # noqa: E402


def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", required=True, help="a run name defined in the config")
    ap.add_argument("--url", default=None, help="server URL (default: config server.url)")
    ap.add_argument("--premises", type=int, default=None, help="premises per (emotion, cast) cell (default: config)")
    ap.add_argument("--reps", type=int, default=1, help="stories per (emotion, cast, premise)")
    ap.add_argument("--emotions", default=None, help="comma list (default: all 41 + neutral)")
    ap.add_argument("--conditions", default=None, help="comma list of cast keys (default: all casts)")
    ap.add_argument("--skip", type=int, default=None, help="capture_skip: leading decoded tokens excluded from the mean")
    ap.add_argument("--n-predict", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="stop after N new stories (0 = all)")
    ap.add_argument("--seed", type=int, default=None, help="plan shuffle seed (default: config)")
    ap.add_argument("--sample-seed", action="store_true", help="pass a per-job sampling seed (reproducible text)")
    ap.add_argument("--deadline", default=None, help="HH:MM local — start no new story after this time")
    ap.add_argument("--stop-file", default=None, help="stop cleanly (resumable) as soon as this file exists")
    args = ap.parse_args()

    study = cfg.load(cfg.config_path(args.config))
    run = study.run(args.run)
    url = args.url or study.server_url
    n_topics = args.premises or study.n_premises
    skip = study.capture_skip if args.skip is None else args.skip
    n_predict = args.n_predict or study.n_predict
    seed = study.plan_seed if args.seed is None else args.seed

    emotions = args.emotions.split(",") if args.emotions else lab.EMOTIONS + [lab.NEUTRAL]
    conditions = args.conditions.split(",") if args.conditions else list(study.casts)
    for e in emotions:
        assert e in lab.EMOTIONS or e == lab.NEUTRAL, f"unknown emotion {e}"
    for c in conditions:
        assert c in study.casts, f"unknown cast {c} (have: {', '.join(study.casts)})"
    if run.scaffold and not study.scaffold_path:
        raise SystemExit(f"run {run.name} uses the identity scaffold but persona.scaffold is not set")

    deadline = None
    if args.deadline:
        hh, mm = (int(x) for x in args.deadline.split(":"))
        now = time.localtime()
        deadline = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hh, mm, 0, 0, 0, -1))
        if deadline < time.time():
            deadline += 86400

    info = lab.assert_server(run, url)
    n_layers, n_embd = int(info["n_layers"]), int(info["n_embd"])
    print(f"[gen] run {run.name} ({run.label}, {run.condition}, {run.fmt})")
    print(f"[gen] server {url}: {os.path.basename(str(info['model_path']))} ({n_layers} layers x {n_embd})")

    run_dir = study.run_dir(run.name)
    os.makedirs(run_dir, exist_ok=True)
    stories_path = os.path.join(run_dir, "stories.jsonl")
    acts_path = os.path.join(run_dir, "acts.f16")
    meta_path = os.path.join(run_dir, "meta.json")
    meta = {
        "n_layers": n_layers, "n_embd": n_embd, "model_path": os.path.basename(str(info["model_path"])),
        "run": run.name, "label": run.label, "architecture": run.architecture, "condition": run.condition,
        "fmt": run.fmt, "scaffold": run.scaffold, "capture_skip": skip, "n_predict": n_predict,
        "emotions": emotions, "conditions": conditions, "topics": n_topics, "reps": args.reps,
        "plan_seed": seed, "sampling": study.sampling, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dtype": "float16", "row_bytes": n_layers * n_embd * 2,
    }
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        assert old["n_layers"] == n_layers and old["n_embd"] == n_embd, "run dir belongs to a different model"
        assert old.get("condition", run.condition) == run.condition, "run dir belongs to a different condition"
        meta["started"] = old.get("started", meta["started"])
    json.dump(meta, open(meta_path, "w"), indent=2)

    done = load_done(stories_path)
    row_bytes = n_layers * n_embd * 2
    if os.path.exists(acts_path) and os.path.getsize(acts_path) % row_bytes:
        raise SystemExit(f"[gen] {acts_path} has a partial row — truncate it to a multiple of {row_bytes} bytes first")
    n_rows = os.path.getsize(acts_path) // row_bytes if os.path.exists(acts_path) else 0

    jobs = [j for j in lab.plan_jobs(emotions, conditions, n_topics, args.reps, seed) if j.key not in done]
    print(f"[gen] {len(done)} done, {len(jobs)} to go, {n_rows} activation rows on disk")

    t_start = time.time()
    n_new = 0
    with open(stories_path, "a", encoding="utf-8") as sf, open(acts_path, "ab") as af:
        for i, job in enumerate(jobs):
            if args.limit and n_new >= args.limit:
                break
            if args.stop_file and os.path.exists(args.stop_file):
                print(f"[gen] stop file present — stopping with {len(jobs) - i} jobs left (resumable)")
                break
            if deadline and time.time() >= deadline:
                print(f"[gen] deadline {args.deadline} reached — stopping with {len(jobs) - i} jobs left (resumable)")
                break
            cast = study.casts[job.condition]
            topic = lab.PREMISES[job.topic_idx]
            sseed = (zlib.crc32(job.key.encode()) & 0x7FFFFFFF) if args.sample_seed else None   # stable across processes
            try:
                res = lab.generate_story(study, run, job.emotion, cast, topic, url=url,
                                         capture_skip=skip, n_predict=n_predict, seed=sseed)
                if res.get("capped") or not res.get("parsed"):
                    # degenerate/truncated: one retry with a fresh seed; a second failure is
                    # stored with bad=True (kept for the job ledger, excluded from analysis)
                    res2 = lab.generate_story(study, run, job.emotion, cast, topic, url=url,
                                              capture_skip=skip, n_predict=n_predict, seed=(sseed or 0) + 7919)
                    if res2.get("parsed") and not res2.get("capped"):
                        res = res2
            except Exception as e:
                print(f"[gen] {job.key}: FAILED {e!r} — sleeping 15s")
                time.sleep(15)
                try:
                    lab.assert_server(run, url)
                except SystemExit as se:
                    print(se)
                    return
                continue
            layers = res["layers"]
            if len(layers) != n_layers:
                print(f"[gen] {job.key}: only {len(layers)}/{n_layers} layers returned (n_used={res['n_used']}) — skipping")
                continue
            mat = np.stack([np.asarray(layers[il], dtype=np.float32) for il in range(n_layers)]).astype(np.float16)
            af.write(mat.tobytes()); af.flush()
            rec = {
                "key": job.key, "row": n_rows, "emotion": job.emotion, "condition": job.condition,
                "topic_idx": job.topic_idx, "topic": topic, "rep": job.rep,
                "mood": res["mood"], "story": res["story"], "parsed": res["parsed"],
                "bad": bool(res.get("capped") or not res.get("parsed")),
                "leaks": lab.leaked_words(job.emotion, res["story"]) if job.emotion != lab.NEUTRAL else [],
                "n_tokens": res["n_tokens"], "n_used": res["n_used"], "skip": res["skip"],
                "prompt_tokens": res["prompt_tokens"], "cached": res["cached"],
                "gen_ms": res["timings"].get("predicted_ms"), "prompt_ms": res["timings"].get("prompt_ms"),
                "prompt_n": res["timings"].get("prompt_n"),
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            sf.write(json.dumps(rec, ensure_ascii=False) + "\n"); sf.flush()
            n_rows += 1; n_new += 1
            if n_new % 10 == 1 or n_new < 5:
                el = time.time() - t_start
                rate = n_new / el if el else 0
                eta = (len(jobs) - i - 1) / rate / 3600 if rate else float("inf")
                print(f"[gen] {n_new:5d}/{len(jobs)} {job.key:40s} tok={res['n_tokens']} used={res['n_used']} "
                      f"cached={res['cached']} | {rate*3600:.0f}/h, ETA {eta:.1f}h"
                      + (f" LEAK{rec['leaks']}" if rec["leaks"] else "") + (" BAD" if rec["bad"] else ""))
    print(f"[gen] finished: {n_new} new stories in {(time.time()-t_start)/60:.1f} min; total rows {n_rows}")


if __name__ == "__main__":
    main()
