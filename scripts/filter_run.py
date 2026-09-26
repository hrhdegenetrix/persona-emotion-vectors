#!/usr/bin/env python3
"""Drop degenerate/truncated stories from a run IN PLACE (stories.jsonl + acts.f16 rewritten,
rows renumbered). Their job keys disappear, so generate_stories.py regenerates those cells on
its next pass. Keeps a backup of both files as *.bak.<timestamp>.

Usage: python3 scripts/filter_run.py <run_dir> [--n-predict 700] [--dry-run]
"""
import argparse
import json
import os
import shutil
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n-predict", type=int, default=None, help="default: meta.json n_predict, else 700")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.run_dir, "meta.json")))
    n_predict = a.n_predict or int(meta.get("n_predict") or 700)
    L, D = meta["n_layers"], meta["n_embd"]
    sp, apath = os.path.join(a.run_dir, "stories.jsonl"), os.path.join(a.run_dir, "acts.f16")
    recs = [json.loads(line) for line in open(sp, encoding="utf-8") if line.strip()]
    acts = np.memmap(apath, dtype=np.float16, mode="r").reshape(-1, L, D)
    assert len(recs) <= acts.shape[0], (len(recs), acts.shape)

    def good(r):
        return (not r.get("bad")) and r.get("parsed") and (r.get("n_tokens") or 0) < n_predict

    keep = [r for r in recs if good(r)]
    print(f"{a.run_dir}: {len(recs)} stories, keeping {len(keep)}, dropping {len(recs) - len(keep)}")
    if a.dry_run or len(keep) == len(recs):
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(sp, f"{sp}.bak.{ts}")
    shutil.copy2(apath, f"{apath}.bak.{ts}")
    new_acts = np.asarray(acts[[r["row"] for r in keep]])
    with open(apath + ".tmp", "wb") as f:
        f.write(new_acts.tobytes())
    with open(sp + ".tmp", "w", encoding="utf-8") as f:
        for i, r in enumerate(keep):
            r["row"] = i
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(apath + ".tmp", apath)
    os.replace(sp + ".tmp", sp)
    print(f"rewritten: {len(keep)} rows; backups *.bak.{ts}")


if __name__ == "__main__":
    main()
