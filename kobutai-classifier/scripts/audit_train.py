#!/usr/bin/env python3
"""Ask the teacher to label every existing training row and report where it disagrees with the stored label.
Writes data/deepseek/audit.tsv (one row per confident disagreement). Needs the same env vars as distill_deepseek.py."""
import collections, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts")
from train.data import load_train
from distill_deepseek import label, valid
rows = [r for r in load_train() if r["file"] not in ("t20_apple_charts.tsv", "t40_deepseek.tsv")]
ans = label(rows)
dis = collections.Counter(); tot = collections.Counter(); out = []
for r, a in zip(rows, ans):
    tot[r["file"]] += 1
    if not valid(a) or float(a.get("confidence") or 0) < 0.9:
        continue
    t = a["type"]; g = a["genre"] if t == "music" else "-"; v = a["vclass"] if t == "video" else "-"
    if (t, g, v) != (r["type"], r["genre"], r["vclass"]):
        dis[r["file"]] += 1
        out.append("\t".join([r["file"], r["type"], r["genre"], r["vclass"], t, g, v, r["source"], r["artist"], r["title"]]) + "\n")
open(ROOT + "/data/deepseek/audit.tsv", "w", encoding="utf-8").writelines(out)
print(f"confident disagreements: {sum(dis.values())} of {sum(tot.values())} rows")
for f in sorted(tot):
    print(f"  {f:40s} {dis[f]:4d} / {tot[f]:5d}  ({dis[f]/tot[f]:.1%})")
