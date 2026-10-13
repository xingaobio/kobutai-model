#!/usr/bin/env python3
"""Option 1 experiment: fine-tune a small pretrained multilingual transformer (mMiniLMv2 L6-H384, distilled from
XLM-R Large) with the same three heads and the same training data as the linear model.
Output: experiments/transformer/out/ (weights + tokenizer)."""
import json, math, os, random, sys, time
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from kobutai_py.core import TYPES, GENRES, VCLASSES, norm, norm_source, dur_bucket
from train.data import load_train

BASE = os.environ.get("TF_BASE", "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large")
OUT = ROOT + "/experiments/transformer/out"
MAXLEN, BS, EPOCHS, LR = 64, 32, int(os.environ.get("TF_EPOCHS", "2")), 6e-5
torch.set_num_threads(2); torch.manual_seed(0); random.seed(0)
BUCKET_WORD = {"U": "unknown", "L": "live", "a": "<1m", "b": "1-2m", "c": "3-5m", "d": "5-10m", "e": "10-25m", "f": "25-60m", "g": ">1h"}


def text_of(title, artist, source, dur):
    b = dur_bucket(dur) if not isinstance(dur, str) else dur
    return f"{norm(title)} | {norm(artist)} | {norm_source(source)} | {BUCKET_WORD[b]}"


class Net(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base).float()
        # the 250k x 384 word-embedding table is 90 % of the parameters; with ~30k examples it is better left frozen
        self.enc.embeddings.word_embeddings.weight.requires_grad_(False)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(h, len(TYPES)), nn.Linear(h, len(GENRES)), nn.Linear(h, len(VCLASSES))])

    def forward(self, ids, mask):
        x = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)        # mean pooling
        x = self.drop(x)
        return [h(x) for h in self.heads]


def examples():
    ex = []
    for r in load_train():
        ex.append({"t": r["title"], "a": r["artist"], "s": r["source"], "d": r["dur"],
                   "y": [TYPES.index(r["type"]),
                         GENRES.index(r["genre"]) if r["type"] == "music" else -100,
                         VCLASSES.index(r["vclass"]) if r["type"] == "video" else -100]})
    keys = [l.rstrip("\n").split("\t") for l in open(ROOT + "/model/artists.tsv", encoding="utf-8")]
    random.Random(1).shuffle(keys)
    for k, g in keys[:9000]:      # artist names teach the genre head, as in the linear model (subsampled for CPU time)
        ex.append({"t": "", "a": k, "s": "", "d": 0, "y": [-100, int(g), -100]})
    return ex


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    net = Net(BASE)
    ex = examples()
    print(len(ex), "examples", flush=True)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(ex) / BS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * max(0.0, 1 - s / steps))
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    step, t0 = 0, time.time()
    net.train()
    for ep in range(EPOCHS):
        rng = random.Random(ep)
        rows = []
        for e in ex:      # same augmentation idea as the linear model: drop the artist / duration sometimes
            a = "" if (e["a"] and e["y"][0] != -100 and rng.random() < 0.2) else e["a"]
            d = 0 if rng.random() < 0.25 else e["d"]
            rows.append((text_of(e["t"], a, e["s"], d), e["y"]))
        rows.sort(key=lambda x: len(x[0]))                 # length bucketing: much less padding on CPU
        batches = [rows[i:i + BS] for i in range(0, len(rows), BS)]
        rng.shuffle(batches)
        for b in batches:
            enc = tok([x[0] for x in b], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
            y = torch.tensor([x[1] for x in b])
            outs = net(enc["input_ids"], enc["attention_mask"])
            loss = sum(ce(o, y[:, k]) for k, o in enumerate(outs) if (y[:, k] != -100).any())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"epoch {ep+1} step {step}/{steps} loss {loss.item():.3f} | {el/60:.1f} min, eta {(steps-step)*el/step/60:.0f} min", flush=True)
    os.makedirs(OUT, exist_ok=True)
    torch.save(net.state_dict(), OUT + "/net.pt")
    tok.save_pretrained(OUT)
    json.dump({"base": BASE, "maxlen": MAXLEN}, open(OUT + "/meta.json", "w"))
    print("DONE", f"{(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
