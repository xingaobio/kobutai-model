#!/usr/bin/env python3
"""Recompute every comparable number for the research archive from saved models and cached teacher answers.

Run from anywhere:  DEEPSEEK_MODEL=deepseek-ai/DeepSeek-V4.1-Flash python3 collect_results.py
No API calls are made: DeepSeek answers come from kobutai-classifier/data/deepseek/cache.jsonl.
Outputs go to ../results/*.csv|json."""
import collections, csv, glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PROJ = os.environ.get("KBT_PROJECT", os.path.join(HERE, "..", "kobutai-classifier"))
if not os.path.isdir(PROJ):
    PROJ = "/home/claude/kobutai-classifier"
MODELS = os.path.join(HERE, "..", "models")
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
sys.path.insert(0, PROJ); sys.path.insert(0, PROJ + "/scripts"); sys.path.insert(0, PROJ + "/experiments/transformer")
from kobutai_py.core import Classifier, GENRES, TYPES
from train.data import load_val, load_real, load, load_train
from distill_deepseek import load_cache, key_of

os.makedirs(RES, exist_ok=True)
SETS = {
    "real_test": load_real()[1],
    "real_dev": load_real()[0],
    "focus": load("data/raw/focus/*.tsv"),
    "synthetic_val": sum(load_val(), []),
}


def ok(r, t, g, v):
    return t == r["type"] and (r["type"] != "music" or g == r["genre"]) and (r["type"] != "video" or v == r["vclass"])


def score(rows, preds):
    n = len(rows)
    mus = [(r, p) for r, p in zip(rows, preds) if r["type"] == "music"]
    vid = [(r, p) for r, p in zip(rows, preds) if r["type"] == "video"]
    return {"n": n,
            "fully_right": sum(ok(r, *p) for r, p in zip(rows, preds)) / n,
            "type": sum(p[0] == r["type"] for r, p in zip(rows, preds)) / n,
            "genre": sum(p[1] == r["genre"] for r, p in mus) / max(1, len(mus)),
            "vclass": sum(p[2] == r["vclass"] for r, p in vid) / max(1, len(vid))}


class Fake:
    def __init__(self, b): self.buckets, self.z = b, None
    def logits(self, idx): return self.z


def linear_preds(model_dir, rows, logits=None, keep=False):
    clf = Classifier(model_dir)
    clf.cfg["thresholds"] = {"type": 0, "genre": 0, "vclass": 0}     # raw accuracy: never abstain
    clf.cfg["artistBonus"] = {"artist": 3, "title": 0}                 # same bonus for every model compared
    real = clf.model
    out, raw = [], []
    for i, r in enumerate(rows):
        if logits is not None:
            clf.model = Fake(real.buckets); clf.model.z = logits[i]
        p = clf.classify(r["title"], r["artist"], r["source"], r["dur"], detail=True)
        clf.model = real
        out.append((p["type"], p["genre"], p["vclass"]))
        raw.append(p)
    return (out, raw) if keep else out


# ---------------------------------------------------------------- transformer logits
def transformer_logits(rows):
    import torch, train_tf as T
    out_dir = PROJ + "/experiments/transformer/out"
    if not os.path.exists(out_dir + "/net.pt"):
        return None
    tok = T.AutoTokenizer.from_pretrained(out_dir)
    net = T.Net(T.BASE); net.load_state_dict(torch.load(out_dir + "/net.pt")); net.eval()
    texts = [T.text_of(r["title"], r["artist"], r["source"], r["dur"]) for r in rows]
    z = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=T.MAXLEN, padding=True, return_tensors="pt")
            o = net(enc["input_ids"], enc["attention_mask"])
            z += [[o[k][j].tolist() for k in range(3)] for j in range(len(texts[i:i + 64]))]
    return z


cache = load_cache()
comparison, hybrid, conf_rows, pergenre = [], [], [], []
for sname, rows in SETS.items():
    t0 = time.time()
    preds = {}
    for it in ("iter1", "iter4", "iter5"):
        preds[f"linear_{it}"] = linear_preds(os.path.join(MODELS, it), rows)
    p5, raw5 = linear_preds(os.path.join(MODELS, "iter5"), rows, keep=True)
    zt = transformer_logits(rows)
    if zt is not None:
        preds["transformer"] = linear_preds(os.path.join(MODELS, "iter5"), rows, logits=zt)
        ens = [[[(a + b) / 2 for a, b in zip(r5["_z"][k], z[k])] for k in range(3)] for r5, z in zip(raw5, zt)]
        preds["linear+transformer"] = linear_preds(os.path.join(MODELS, "iter5"), rows, logits=ens)
    te = [cache.get(key_of(r)) or {} for r in rows]
    preds["deepseek_v4.1_flash"] = [(a.get("type"), a.get("genre"), a.get("vclass")) for a in te]
    # hybrid: shipped model (with its real thresholds) + hand-off when not confident
    clf = Classifier(os.path.join(MODELS, "iter5"))
    st = [clf.classify(r["title"], r["artist"], r["source"], r["dur"]) for r in rows]

    def conf(p):
        if not p["type"]:
            return 0.0
        c = p["typeConfidence"]
        if p["type"] == "music":
            c = min(c, p["genreConfidence"]) if p["genre"] else 0.0
        if p["type"] == "video":
            c = min(c, p["vclassConfidence"]) if p["vclass"] else 0.0
        return c
    for th in (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01):
        hp, sent = [], 0
        for r, p, a, q in zip(rows, st, te, p5):
            if conf(p) >= th:
                hp.append((p["type"], p["genre"], p["vclass"]))
            else:
                sent += 1; hp.append((a.get("type"), a.get("genre"), a.get("vclass")))
        s = score(rows, hp)
        hybrid.append({"set": sname, "threshold": th, "share_sent_to_deepseek": sent / len(rows), **s})
        if th == 0.9:
            preds["hybrid_iter5+deepseek@0.9"] = hp
    for m, p in preds.items():
        comparison.append({"set": sname, "model": m, **score(rows, p)})
    if sname == "real_test":
        for name, p in (("iter5", preds["linear_iter5"]), ("deepseek", preds["deepseek_v4.1_flash"])):
            cm = collections.Counter((r["genre"], q[1]) for r, q in zip(rows, p) if r["type"] == "music")
            for (gt, pr), c in cm.items():
                conf_rows.append({"model": name, "true": gt, "pred": pr, "count": c})
            for g in GENRES:
                rs = [(r, q) for r, q in zip(rows, p) if r["type"] == "music" and r["genre"] == g]
                pergenre.append({"model": name, "genre": g, "n": len(rs),
                                 "acc": sum(q[1] == g for r, q in rs) / max(1, len(rs))})
    print(f"{sname}: done in {time.time()-t0:.0f}s", flush=True)


def write(name, rows):
    with open(os.path.join(RES, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


write("model_comparison.csv", comparison)
write("hybrid_curve.csv", hybrid)
write("genre_confusion_real_test.csv", conf_rows)
write("per_genre_accuracy_real_test.csv", pergenre)

# focus subsets (memes / idols / anime, 120 rows each, in file order)
focus = SETS["focus"]; fs = []
for it in ("iter4", "iter5"):
    pr = linear_preds(os.path.join(MODELS, it), focus)
    for i, t in enumerate(("memes", "japanese_idols", "anime")):
        fs.append({"model": f"linear_{it}", "subset": t, **score(focus[i*120:(i+1)*120], pr[i*120:(i+1)*120])})
te = [cache.get(key_of(r)) or {} for r in focus]
pr = [(a.get("type"), a.get("genre"), a.get("vclass")) for a in te]
for i, t in enumerate(("memes", "japanese_idols", "anime")):
    fs.append({"model": "deepseek_v4.1_flash", "subset": t, **score(focus[i*120:(i+1)*120], pr[i*120:(i+1)*120])})
write("focus_subsets.csv", fs)

# thresholded metrics recorded by train/evaluate.py at the end of each round
it_rows = []
for f in sorted(glob.glob(PROJ + "/reports/metrics_iter*.json")):
    tag = os.path.basename(f)[8:-5]
    m = json.load(open(f))
    for split in ("dev", "test", "real_dev", "real_test"):
        if split in m:
            d = m[split]
            it_rows.append({"round": tag, "split": split, "n": d["n"], "type_acc": d["type_acc"],
                            "genre_acc_when_answered": d["genre_acc_answered"], "genre_coverage": d["genre_coverage"],
                            "vclass_acc_when_answered": d["vclass_acc_answered"], "vclass_coverage": d["vclass_coverage"],
                            "wrong_rate": d["wrong_rate"], "generic_rate": d["generic_rate"]})
write("round_metrics_thresholded.csv", it_rows)

# training data inventory
KIND = {"t0": "LLM-written (Sonnet sub-agents, from memory)", "t1": "LLM-written (Sonnet sub-agents, targeted hard cases)",
        "t20": "Apple public genre charts (rule-labelled)", "t21": "Bilibili popular list (partition-labelled)",
        "t22": "YouTube trending titles (LLM-labelled)", "t3": "LLM-written (memes / idols / anime)"}
inv = []
for f in sorted(glob.glob(PROJ + "/data/raw/train/*.tsv")):
    rows = load("data/raw/train/" + os.path.basename(f))
    b = os.path.basename(f)
    num = int(b[1:3])
    kind = ("LLM-written (Sonnet sub-agents, from memory)" if num <= 12 else
            "LLM-written (Sonnet sub-agents, targeted hard cases)" if num <= 16 else
            "Apple public genre charts (rule-labelled)" if num == 20 else
            "Bilibili popular list (partition-labelled)" if num == 21 else
            "YouTube trending titles (LLM-labelled)" if num == 22 else
            "LLM-written (memes / idols / anime)" if 30 <= num <= 33 else "other")
    c = collections.Counter(r["type"] for r in rows)
    inv.append({"file": b, "source": kind, "rows": len(rows), **{t: c.get(t, 0) for t in TYPES}})
for f in sorted(glob.glob(PROJ + "/data/raw/artists/*.tsv")):
    n = sum(1 for l in open(f, encoding="utf-8") if l.strip())
    inv.append({"file": "artists/" + os.path.basename(f), "source": "artist list", "rows": n,
                **{t: "" for t in TYPES}})
for d in ("val", "real", "focus"):
    for f in sorted(glob.glob(PROJ + f"/data/raw/{d}/*.tsv")):
        rows = load(f"data/raw/{d}/" + os.path.basename(f))
        c = collections.Counter(r["type"] for r in rows)
        inv.append({"file": f"{d}/" + os.path.basename(f), "source": {"val": "evaluation: LLM-written, independent",
                    "real": "evaluation: scraped live chart / trending pages 2026-10-13", "focus": "evaluation: memes/idols/anime"}[d],
                    "rows": len(rows), **{t: c.get(t, 0) for t in TYPES}})
write("data_inventory.csv", inv)
print("training rows after guards:", len(load_train()))
print("written to", os.path.abspath(RES))
