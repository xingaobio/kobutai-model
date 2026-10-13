#!/usr/bin/env python3
"""Compare the fine-tuned transformer with the linear model inside the SAME pipeline (site rules, artist-table bonus),
on the synthetic val set, the real-world set and the memes/idols/anime set. Also: ONNX int8 size and latency."""
import json, os, sys, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/experiments/transformer"); sys.path.insert(0, ROOT + "/scripts")
from kobutai_py.core import Classifier
from train.data import load_val, load_real, load
import train_tf as T

OUT = ROOT + "/experiments/transformer/out"
tok = T.AutoTokenizer.from_pretrained(OUT)
net = T.Net(T.BASE); net.load_state_dict(torch.load(OUT + "/net.pt")); net.eval()


def tf_logits(rows, bs=64):
    texts = [T.text_of(r["title"], r["artist"], r["source"], r["dur"]) for r in rows]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=T.MAXLEN, padding=True, return_tensors="pt")
            z = net(enc["input_ids"], enc["attention_mask"])
            out += [[z[k][j].tolist() for k in range(3)] for j in range(len(texts[i:i + bs]))]
    return out


class Fake:
    def __init__(self, buckets): self.buckets, self.z = buckets, None
    def logits(self, idx): return self.z


def run(clf, rows, logits=None):
    res = []
    if logits is not None:
        real = clf.model; clf.model = Fake(real.buckets)
    for i, r in enumerate(rows):
        if logits is not None:
            clf.model.z = logits[i]
        res.append((r, clf.classify(r["title"], r["artist"], r["source"], r["dur"])))
    if logits is not None:
        clf.model = real
    return res


def score(res):
    n = len(res); mus = [(r, p) for r, p in res if r["type"] == "music"]; vid = [(r, p) for r, p in res if r["type"] == "video"]
    full = sum(p["type"] == r["type"] and (r["type"] != "music" or p["genre"] == r["genre"]) and (r["type"] != "video" or p["vclass"] == r["vclass"]) for r, p in res) / n
    return (f"fully right {full:.3f} | type {sum(p['type']==r['type'] for r,p in res)/n:.3f} | "
            f"genre {sum(p['genre']==r['genre'] for r,p in mus)/max(1,len(mus)):.3f} | vclass {sum(p['vclass']==r['vclass'] for r,p in vid)/max(1,len(vid)):.3f}"), full


clf = Classifier(ROOT + "/model")
clf.cfg["thresholds"] = {"type": 0, "genre": 0, "vclass": 0}          # compare raw accuracy, no abstaining
sets = [("synthetic val (963)", sum(load_val(), [])), ("REAL test (383)", load_real()[1]), ("REAL dev (254)", load_real()[0]),
        ("memes/idols/anime (360)", load("data/raw/focus/*.tsv"))]
report = {}
for name, rows in sets:
    zt = tf_logits(rows)
    a, fa = score(run(clf, rows))
    b, fb = score(run(clf, rows, zt))
    # ensemble: average the two models' logits
    zl = []
    for r in rows:
        clf.classify(r["title"], r["artist"], r["source"], r["dur"], detail=True)
    ens = []
    for r, z in zip(rows, zt):
        zlin = clf.classify(r["title"], r["artist"], r["source"], r["dur"], detail=True)["_z"]
        ens.append([[(x + y) / 2 for x, y in zip(zlin[k], z[k])] for k in range(3)])
    c, fc = score(run(clf, rows, ens))
    print(f"== {name}\n  linear (current)  {a}\n  transformer       {b}\n  average of both   {c}", flush=True)
    report[name] = {"linear": fa, "transformer": fb, "ensemble": fc}
json.dump(report, open(ROOT + "/experiments/transformer/report.json", "w"), indent=1)

# ---- deployment cost: ONNX export, int8 quantisation, size, latency
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
class Export(torch.nn.Module):
    def __init__(s, n): super().__init__(); s.n = n
    def forward(s, ids, mask):
        a, b, c = s.n(ids, mask); return a, b, c
ex = tok(["test | test | youtube.com | unknown"], return_tensors="pt")
onnx_path = OUT + "/model.onnx"
torch.onnx.export(Export(net), (ex["input_ids"], ex["attention_mask"]), onnx_path, input_names=["ids", "mask"],
                  output_names=["type", "genre", "vclass"], dynamic_axes={"ids": {0: "b", 1: "t"}, "mask": {0: "b", 1: "t"}},
                  opset_version=17, dynamo=False)
quantize_dynamic(onnx_path, OUT + "/model.int8.onnx", weight_type=QuantType.QInt8)
mb = lambda p: os.path.getsize(p) / 1e6
print(f"\nONNX fp32 {mb(onnx_path):.0f} MB, int8 {mb(OUT + '/model.int8.onnx'):.0f} MB "
      f"(word-embedding table alone: {250002*384/1e6:.0f}M params; pruning the vocabulary to the tokens real titles use would shrink this a lot)")
sess = ort.InferenceSession(OUT + "/model.int8.onnx", providers=["CPUExecutionProvider"])
rows = load_real()[1][:200]
texts = [T.text_of(r["title"], r["artist"], r["source"], r["dur"]) for r in rows]
t0 = time.time()
for t in texts:
    e = tok([t], truncation=True, max_length=T.MAXLEN, return_tensors="np")
    sess.run(None, {"ids": e["input_ids"], "mask": e["attention_mask"]})
print(f"int8 ONNX latency, one title at a time on this 2-core CPU: {(time.time()-t0)/len(texts)*1000:.1f} ms")
# int8 accuracy check on the real test set
zq = []
for t in [T.text_of(r["title"], r["artist"], r["source"], r["dur"]) for r in load_real()[1]]:
    e = tok([t], truncation=True, max_length=T.MAXLEN, return_tensors="np")
    o = sess.run(None, {"ids": e["input_ids"], "mask": e["attention_mask"]})
    zq.append([o[k][0].tolist() for k in range(3)])
print("int8 transformer on REAL test:", score(run(clf, load_real()[1], zq))[0])
