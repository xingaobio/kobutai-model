#!/usr/bin/env python3
"""Score the meme / Japanese idol / anime focus set (data/raw/focus) and, for regression, the real-world test split."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kobutai_py.core import Classifier
from train.data import load, load_real
from train.evaluate import run, metrics
clf = Classifier(ROOT + "/model")
focus = load("data/raw/focus/*.tsv")
def show(name, rows):
    res = run(clf, rows)
    m = metrics(res)
    right = 0
    for r, p in res:
        ok = p["type"] == r["type"] and (r["type"] != "music" or p["genre"] == r["genre"]) and (r["type"] != "video" or p["vclass"] == r["vclass"])
        right += ok
    print(f"{name:11s} n={m['n']}: type {m['type_acc']:.3f} | genre acc@ans {m['genre_acc_answered']:.3f} cov {m['genre_coverage']:.2f} | "
          f"vclass {m['vclass_acc_answered']:.3f} | fully right {right/len(res):.3f} | wrong {m['wrong_rate']:.3f}")
    return res
res = show("focus", focus)
# the validator keeps file order: memes, idols, anime, 120 rows each
for i, t in enumerate(("  memes", "  idols", "  anime")):
    show(t, focus[i * 120:(i + 1) * 120])
show("real_test", load_real()[1])
if len(sys.argv) > 1:
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        for r, p in res:
            if p["type"] != r["type"] or (r["type"] == "music" and p["genre"] != r["genre"]) or (r["type"] == "video" and p["vclass"] != r["vclass"]):
                f.write("\t".join([r["type"], r["genre"], r["vclass"], str(p["type"]), str(p["genre"]), str(p["vclass"]), r["source"], r["artist"], r["title"]]) + "\n")
