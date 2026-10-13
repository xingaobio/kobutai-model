#!/usr/bin/env python3
"""Train the three-head hashed n-gram linear model and export model/kobutai.bin."""
import argparse, json, os, random, struct, sys, time
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kobutai_py.core import (norm, norm_source, dur_bucket, feature_strings, fnv1a,
                             TYPES, GENRES, VCLASSES)
from train.data import load_train

_cache = {}
def idx_of(t, a, s, b, mask):
    out = set()
    for f in feature_strings(t, a, s, b):
        h = _cache.get(f)
        if h is None:
            h = _cache[f] = fnv1a(f.encode("utf-8"))
        out.add(h & mask)
    return sorted(out)


def build(rows, buckets):
    indptr, indices, data = [0], [], []
    for r in rows:
        idx = idx_of(r["t"], r["a"], r["s"], r["b"], buckets - 1)
        indices.extend(idx)
        data.extend([1.0 / np.sqrt(len(idx))] * len(idx))
        indptr.append(len(indices))
    return sp.csr_matrix((data, indices, indptr), shape=(len(rows), buckets), dtype=np.float32)


def prep(r):
    return dict(r, t=norm(r["title"]), a=norm(r["artist"]), s=norm_source(r["source"]),
                b=dur_bucket(r["dur"]))


def augment(rows, rng, copies):
    out = []
    for r in rows:
        out.append(r)
        for _ in range(copies):
            q = dict(r)
            changed = False
            if q["a"] and rng.random() < 0.35:
                q["a"] = ""; changed = True
            if q["b"] not in ("U", "L") and rng.random() < 0.5:
                q["b"] = "U"; changed = True
            if rng.random() < 0.15:
                q["s"] = ""; changed = True
            if changed:
                out.append(q)
    return out


def fit(X, y, C, n_classes):
    clf = LogisticRegression(C=C, max_iter=300, tol=1e-4, class_weight="balanced")
    clf.fit(X, y)
    W = np.zeros((n_classes, X.shape[1]), dtype=np.float64)
    b = np.zeros(n_classes)
    if len(clf.classes_) == 2 and clf.coef_.shape[0] == 1:
        W[1], W[0] = clf.coef_[0] / 2, -clf.coef_[0] / 2
        b[1], b[0] = clf.intercept_[0] / 2, -clf.intercept_[0] / 2
    else:
        W[clf.classes_] = clf.coef_
        b[clf.classes_] = clf.intercept_
    return W, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets-log2", type=int, default=17)
    ap.add_argument("--C", type=float, default=30.0)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--artist-rows", type=int, default=1)
    ap.add_argument("--out", default=ROOT + "/model/kobutai.bin")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    t0 = time.time()
    rng = random.Random(7)
    B = 1 << a.buckets_log2
    rows = [prep(r) for r in load_train()]
    rows = augment(rows, rng, a.aug)
    music = [r for r in rows if r["type"] == "music"]
    if a.artist_rows:
        # teach the genre head artist names (artist-only and title-only views)
        for line in open(ROOT + "/model/artists.tsv", encoding="utf-8"):
            k, g = line.rstrip("\n").split("\t")
            base = {"genre": GENRES[int(g)], "s": "", "b": "U"}
            music.append(dict(base, t="", a=k))
            music.append(dict(base, t=k, a=""))
    video = [r for r in rows if r["type"] == "video"]
    print(f"rows: type={len(rows)} genre={len(music)} vclass={len(video)}")
    heads = []
    for name, labels, rs, key in (("type", TYPES, rows, "type"), ("genre", GENRES, music, "genre"),
                                  ("vclass", VCLASSES, video, "vclass")):
        X = build(rs, B)
        y = np.array([labels.index(r[key]) for r in rs])
        W, b = fit(X, y, a.C, len(labels))
        heads.append((W, b))
        print(f"  {name}: train acc {(np.argmax(X @ W.T + b, axis=1) == y).mean():.4f}  ({time.time()-t0:.0f}s)")
    # ---- quantise + export
    with open(a.out, "wb") as f:
        f.write(struct.pack("<4sIII", b"KBT1", 1, B, len(heads)))
        qs = []
        for W, b in heads:
            scale = float(np.abs(W).max() / 127.0) or 1.0
            f.write(struct.pack("<If", W.shape[0], scale))
            f.write(np.asarray(b, dtype="<f4").tobytes())
            qs.append(np.clip(np.rint(W / scale), -127, 127).astype(np.int8))
        f.write(np.ascontiguousarray(np.concatenate(qs, axis=0).T).tobytes())
    nz = sum(int((q != 0).any(axis=0).sum()) for q in qs)
    print(f"wrote {a.out}: {os.path.getsize(a.out)/1e6:.2f} MB, buckets={B}")


if __name__ == "__main__":
    main()
