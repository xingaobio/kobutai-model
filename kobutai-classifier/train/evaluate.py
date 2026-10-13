#!/usr/bin/env python3
"""Evaluate the full pipeline on the dev and test splits.
--tune: choose artistBonus and thresholds on dev and write them to config.json."""
import argparse, collections, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kobutai_py.core import Classifier, GENRES
from train.data import load_val, load_real


def run(clf, rows):
    out = []
    for r in rows:
        p = clf.classify(r["title"], r["artist"], r["source"], r["dur"])
        out.append((r, p))
    return out


def metrics(res):
    m = {}
    n = len(res)
    answered = [(r, p) for r, p in res if p["type"]]
    m["n"] = n
    m["type_acc"] = sum(p["type"] == r["type"] for r, p in res) / n
    m["type_coverage"] = len(answered) / n
    m["type_acc_answered"] = sum(p["type"] == r["type"] for r, p in answered) / max(1, len(answered))
    mus = [(r, p) for r, p in res if r["type"] == "music"]
    ans = [(r, p) for r, p in mus if p["genre"]]
    m["music_n"] = len(mus)
    m["genre_coverage"] = len(ans) / max(1, len(mus))
    m["genre_acc_answered"] = sum(p["genre"] == r["genre"] for r, p in ans) / max(1, len(ans))
    m["genre_acc_all"] = sum(p["genre"] == r["genre"] for r, p in mus) / max(1, len(mus))
    vid = [(r, p) for r, p in res if r["type"] == "video"]
    va = [(r, p) for r, p in vid if p["vclass"]]
    m["video_n"] = len(vid)
    m["vclass_coverage"] = len(va) / max(1, len(vid))
    m["vclass_acc_answered"] = sum(p["vclass"] == r["vclass"] for r, p in va) / max(1, len(va))
    # wrong character shown: a confident but wrong answer anywhere
    wrong = 0
    for r, p in res:
        if p["type"] and p["type"] != r["type"]:
            wrong += 1
        elif p["type"] == "music" and p["genre"] and p["genre"] != r["genre"]:
            wrong += 1
        elif p["type"] == "video" and p["vclass"] and p["vclass"] != r["vclass"]:
            wrong += 1
    m["wrong_rate"] = wrong / n
    generic = sum(1 for r, p in res if not p["type"] or (p["type"] == "music" and not p["genre"])
                  or (p["type"] == "video" and not p["vclass"]))
    m["generic_rate"] = generic / n
    via = collections.Counter()
    for r, p in ans:
        via[p["genreVia"] + ("_ok" if p["genre"] == r["genre"] else "_bad")] += 1
    m["genre_via"] = dict(via)
    pg = {}
    for g in GENRES:
        rs = [(r, p) for r, p in mus if r["genre"] == g]
        a2 = [(r, p) for r, p in rs if p["genre"]]
        pg[g] = [len(rs), round(len(a2) / max(1, len(rs)), 2),
                 round(sum(p["genre"] == r["genre"] for r, p in a2) / max(1, len(a2)), 2)]
    m["per_genre(n,cov,acc)"] = pg
    bl = {}
    for lang in ("zh", "ja", "en", "xx"):
        rs = [(r, p) for r, p in res if r["lang"] == lang]
        if rs:
            bl[lang] = round(sum(p["type"] == r["type"] for r, p in rs) / len(rs), 3)
    m["type_acc_by_lang"] = bl
    return m


def tune(clf, dev, real_dev):
    cfg = clf.cfg
    best = None
    cfg["thresholds"].update(type=0.0, genre=0.0, vclass=0.0)
    for ba in (0, 1, 2, 3, 4, 6, 8):
        for bt in (0, 1, 2, 3, 4):
            if bt > ba:
                continue
            cfg["artistBonus"] = {"artist": ba, "title": bt}
            res = run(clf, dev)
            acc = sum(p["genre"] == r["genre"] for r, p in res if r["type"] == "music")
            if best is None or acc > best[0]:
                best = (acc, ba, bt)
    cfg["artistBonus"] = {"artist": best[1], "title": best[2]}
    res = run(clf, real_dev)       # thresholds come from the real-world rows only: the synthetic rows are too easy
    # thresholds: the lowest that reaches the target accuracy among answered rows
    def pick(pairs, target, lo):
        # pairs: (confidence, correct)
        for th in [x / 100 for x in range(int(lo * 100), 96, 5)]:
            a = [c for conf, c in pairs if conf >= th]
            if a and sum(a) / len(a) >= target:
                return th
        return 0.95
    cfg["thresholds"]["type"] = pick([(p["typeConfidence"], p["type"] == r["type"]) for r, p in res], 0.97, 0.4)
    cfg["thresholds"]["genre"] = pick([(p["genreConfidence"], p["genre"] == r["genre"]) for r, p in res
                                       if r["type"] == "music" and p["type"] == "music"], 0.88, 0.3)
    cfg["thresholds"]["genre"] = min(cfg["thresholds"]["genre"], 0.7)
    cfg["thresholds"]["vclass"] = pick([(p["vclassConfidence"], p["vclass"] == r["vclass"]) for r, p in res
                                        if r["type"] == "video" and p["type"] == "video"], 0.90, 0.5)
    json.dump(cfg, open(ROOT + "/model/config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("tuned:", cfg["artistBonus"], cfg["thresholds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    clf = Classifier(ROOT + "/model")
    dev, test = load_val()
    rdev, rtest = load_real()
    if a.tune:
        tune(clf, dev + rdev, rdev)
    rep = {}
    splits = [("dev", dev), ("test", test)] + ([("real_dev", rdev), ("real_test", rtest)] if rtest else [])
    for name, rows in splits:
        res = run(clf, rows)
        rep[name] = metrics(res)
        if name.endswith("dev") and a.tag:
            with open(f"{ROOT}/reports/{name}_errors_{a.tag}.tsv", "w", encoding="utf-8") as f:
                for r, p in res:
                    bad = p["type"] != r["type"] or (r["type"] == "music" and p["genre"] != r["genre"]) or \
                          (r["type"] == "video" and p["vclass"] != r["vclass"])
                    if bad:
                        f.write("\t".join([r["type"], r["genre"], r["vclass"], str(p["type"]), str(p["genre"]),
                                           str(p["vclass"]), f'{p["typeConfidence"]:.2f}', f'{p["genreConfidence"]:.2f}',
                                           r["source"], str(r["dur"]), r["artist"], r["title"]]) + "\n")
    rep["config"] = {"thresholds": clf.cfg["thresholds"], "artistBonus": clf.cfg["artistBonus"]}
    rep["model_bytes"] = os.path.getsize(ROOT + "/model/kobutai.bin")
    if a.tag:
        json.dump(rep, open(f"{ROOT}/reports/metrics_{a.tag}.json", "w"), ensure_ascii=False, indent=1)
    for name, _ in splits:
        m = rep[name]
        print(f"{name:9s} n={m['n']}: type {m['type_acc']:.3f} | genre cov {m['genre_coverage']:.2f} acc@ans {m['genre_acc_answered']:.3f} "
              f"acc_all {m['genre_acc_all']:.3f} | vclass cov {m['vclass_coverage']:.2f} acc {m['vclass_acc_answered']:.3f} "
              f"| wrong {m['wrong_rate']:.3f} generic {m['generic_rate']:.3f}")
        if not a.quiet:
            print("   via", m["genre_via"], "lang", m["type_acc_by_lang"])
            print("   ", m["per_genre(n,cov,acc)"])


if __name__ == "__main__":
    main()
