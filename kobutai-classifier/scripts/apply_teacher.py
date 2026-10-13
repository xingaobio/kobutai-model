#!/usr/bin/env python3
"""Turn cached teacher (DeepSeek) answers into training changes. No API calls: run distill_deepseek.py label /
audit_train.py first.
  - data/raw/train/t40_deepseek.tsv : pool titles the teacher labelled with confidence >= 0.8
  - data/deepseek/overrides.tsv     : hand-written rows the teacher confidently (>= 0.9) disagrees with -> DROP;
                                       rule-labelled Apple chart rows it disagrees with -> its label
  - data/raw/artists/c02_deepseek.tsv: artists with >= 2 teacher-labelled songs agreeing >= 80 %"""
import collections, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts")
from train.data import load
from distill_deepseek import load_cache, key_of, valid, cmd_build
cmd_build(0.8)
cache = load_cache()
ov = []
for r in load("data/raw/train/*.tsv"):
    if r["file"] == "t40_deepseek.tsv":
        continue
    a = cache.get(key_of(r))
    if not valid(a) or float(a.get("confidence") or 0) < 0.9:
        continue
    t = a["type"]; g = a["genre"] if t == "music" else "-"; v = a["vclass"] if t == "video" else "-"
    if (t, g, v) == (r["type"], r["genre"], r["vclass"]):
        continue
    action = "\t".join([t, g, v]) if r["file"] == "t20_apple_charts.tsv" else "DROP"
    ov.append("\t".join([r["file"], r["artist"], r["title"], action]) + "\n")
open(ROOT + "/data/deepseek/overrides.tsv", "w", encoding="utf-8").writelines(ov)
print(len(ov), "overrides")
votes = collections.defaultdict(collections.Counter)
for r in load("data/raw/train/t40_deepseek.tsv"):
    if r["type"] == "music" and r["artist"]:
        votes[r["artist"]][r["genre"]] += 1
art = []
for a, v in sorted(votes.items()):
    (g, c), n = v.most_common(1)[0], sum(v.values())
    if n >= 2 and c / n >= 0.8:
        art.append(f"{a}\t\t{g}\n")
open(ROOT + "/data/raw/artists/c02_deepseek.tsv", "w", encoding="utf-8").writelines(art)
print(len(art), "teacher-confirmed artists")
