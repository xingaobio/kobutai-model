#!/usr/bin/env python3
"""Un-thresholded accuracy on all real rows and all synthetic val rows, for comparing data variants."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kobutai_py.core import Classifier
from train.data import load_val, load_real
from train.evaluate import run
clf = Classifier(ROOT + "/model")
clf.cfg["thresholds"] = {"type": 0, "genre": 0, "vclass": 0}
for ba in (1, 2, 3):
    clf.cfg["artistBonus"] = {"artist": ba, "title": 0}
    o = []
    for name, (a, b) in (("val", load_val()), ("real", load_real())):
        res = run(clf, a + b)
        mus = [(r, p) for r, p in res if r["type"] == "music"]
        vid = [(r, p) for r, p in res if r["type"] == "video"]
        o.append(f"{name}: type {sum(p['type']==r['type'] for r,p in res)/len(res):.3f} genre {sum(p['genre']==r['genre'] for r,p in mus)/len(mus):.3f} "
                 f"vclass {sum(p['vclass']==r['vclass'] for r,p in vid)/len(vid):.3f}")
    print(f"bonus {ba} | " + " | ".join(o))
