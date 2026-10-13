#!/usr/bin/env python3
"""Use DeepSeek as a teacher for the small classifier.

  export DEEPSEEK_API_KEY=...            # never commit the key
  python3 scripts/distill_deepseek.py check              # 1. is the teacher better than the student? (real-world test rows)
  python3 scripts/distill_deepseek.py label              # 2. label the unlabelled real titles in data/pool/*.tsv
  python3 scripts/distill_deepseek.py build              # 3. keep confident labels -> data/raw/train/t40_deepseek.tsv

Pool files use the 8-column sample format with type/genre/vclass set to "?".
Answers are cached in data/deepseek/cache.jsonl, so reruns cost nothing."""
import glob, hashlib, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train.data import load, load_real
from kobutai_py.core import TYPES, GENRES, VCLASSES

MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")   # e.g. deepseek-ai/DeepSeek-V4.1-Flash on a self-hosted endpoint
URL = os.environ.get("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")
CACHE = ROOT + "/data/deepseek/cache.jsonl"
CAP_GENRE = 600       # max teacher rows per music genre
CAP_PODCAST = 1500
BATCH = 20
GUIDE = open(ROOT + "/spec/LABELING.md", encoding="utf-8").read().split("## Other fields")[0]
SYSTEM = ("You label media items for a classifier. You only see text metadata, exactly like the classifier does.\n\n" + GUIDE +
          "\nExtra rules: meme videos, 鬼畜, 音MAD → video/ent; a meme song played as a song → music with its genre; anime episodes, PVs, "
          "clips → video/ent; anime OP/ED/character songs/OST → music/anime; Japanese idol MVs and live footage → music/jpop; idol variety "
          "→ video/ent; idol live streams → live; idol radio → podcast.\n\n"
          'Answer with JSON only: {"items":[{"i":0,"type":"music|video|live|podcast","genre":"<genre or null>","vclass":"ent|info|null",'
          '"confidence":0.0-1.0,"entity":"<the specific artist, group, anime, show or meme this is, or null>"}]}. '
          "confidence is how sure you are of the whole label from the text alone; use below 0.6 when the text does not reveal it.")


def key_of(r):
    return hashlib.sha1("\t".join([r["source"], str(r["dur"]), r["artist"], r["title"], MODEL]).encode()).hexdigest()


def load_cache():
    c = {}
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            d = json.loads(l); c[d["k"]] = d["a"]
    return c


def ask(rows):
    lines = [json.dumps({"i": i, "title": r["title"], "artist": r["artist"], "site": r["source"],
                         "duration_s": None if r["dur"] == 0 else ("live" if r["dur"] < 0 else r["dur"])}, ensure_ascii=False)
             for i, r in enumerate(rows)]
    payload = {"model": MODEL, "temperature": 0, "max_tokens": 4000,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]}
    if os.environ.get("DEEPSEEK_JSON_MODE", "1") == "1":
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                           "Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"]})
    for t in range(4):
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=120).read())
            txt = d["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, flags=re.S)
            items = json.loads(m.group(0))["items"]
            return {int(x["i"]): x for x in items}, d.get("usage", {})
        except Exception as e:
            print("  retry", t, str(e)[:80], flush=True); time.sleep(3 * (t + 1))
    return {}, {}


def label(rows):
    cache = load_cache()
    todo = [r for r in rows if key_of(r) not in cache]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    out = open(CACHE, "a", encoding="utf-8")
    tok, done = 0, 0
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    with ThreadPoolExecutor(max_workers=int(os.environ.get("DEEPSEEK_WORKERS", "8"))) as ex:
        for batch, (ans, usage) in zip(batches, ex.map(ask, batches)):
            tok += usage.get("total_tokens", 0)
            for j, r in enumerate(batch):
                if j in ans:
                    cache[key_of(r)] = ans[j]
                    out.write(json.dumps({"k": key_of(r), "a": ans[j]}, ensure_ascii=False) + "\n")
            out.flush()
            done += len(batch)
            if done % 400 < BATCH or done == len(todo):
                print(f"  {done}/{len(todo)} labelled, {tok} tokens so far", flush=True)
    return [cache.get(key_of(r)) for r in rows]


def valid(a):
    if not a or a.get("type") not in TYPES:
        return False
    if a["type"] == "music" and a.get("genre") not in GENRES:
        return False
    if a["type"] == "video" and a.get("vclass") not in VCLASSES:
        return False
    return True


def cmd_check():
    for name, rows in (("REAL TEST (383)", load_real()[1]), ("FOCUS memes/idols/anime (360)", load("data/raw/focus/*.tsv"))):
        print("==", name)
        _check(rows)


def _check(rows):
    from kobutai_py.core import Classifier
    ans = label(rows)
    clf = Classifier(ROOT + "/model")
    def score(name, pred):
        n = len(rows); mus = [(r, p) for r, p in zip(rows, pred) if r["type"] == "music"]; vid = [(r, p) for r, p in zip(rows, pred) if r["type"] == "video"]
        full = sum(p[0] == r["type"] and (r["type"] != "music" or p[1] == r["genre"]) and (r["type"] != "video" or p[2] == r["vclass"])
                   for r, p in zip(rows, pred)) / n
        print(f"{name:10s} fully right {full:.3f} | type {sum(p[0]==r['type'] for r,p in zip(rows,pred))/n:.3f} | genre {sum(p[1]==r['genre'] for r,p in mus)/len(mus):.3f} "
              f"| vclass {sum(p[2]==r['vclass'] for r,p in vid)/len(vid):.3f}   (no thresholds, n={n})")
    clf.cfg["thresholds"] = {"type": 0, "genre": 0, "vclass": 0}
    stu = [clf.classify(r["title"], r["artist"], r["source"], r["dur"]) for r in rows]
    score("student", [(p["type"], p["genre"], p["vclass"]) for p in stu])
    score("teacher", [((a or {}).get("type"), (a or {}).get("genre"), (a or {}).get("vclass")) for a in ans])
    conf = [(float((a or {}).get("confidence") or 0), (a or {}).get("type") == r["type"] and (r["type"] != "music" or (a or {}).get("genre") == r["genre"])
             and (r["type"] != "video" or (a or {}).get("vclass") == r["vclass"])) for r, a in zip(rows, ans)]
    for th in (0.6, 0.8, 0.9):
        k = [c for x, c in conf if x >= th]
        print(f"   teacher with confidence >= {th}: keeps {len(k)/len(conf):.0%}, fully right {sum(k)/max(1,len(k)):.3f}")


def pool():
    rows = []
    for f in sorted(glob.glob(ROOT + "/data/pool/*.tsv")):
        for line in open(f, encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            if len(c) == 8:
                rows.append({"lang": c[3], "source": c[4], "dur": int(c[5]), "artist": c[6], "title": c[7]})
    return rows


def cmd_label():
    rows = pool()
    print(len(rows), "pool rows")
    label(rows)


def cmd_build(min_conf=0.8):
    rows = pool()
    cache = load_cache()
    held = {r["title"].strip().lower() for r in load("data/raw/val/*.tsv") + load("data/raw/real/*.tsv") + load("data/raw/focus/*.tsv")}
    out, ents, seen, cap = [], [], set(), {}
    for idx, r in enumerate(rows):
        a = cache.get(key_of(r))
        if not valid(a) or float(a.get("confidence") or 0) < min_conf or r["title"].strip().lower() in held:
            continue
        if (r["artist"], r["title"]) in seen:
            continue
        cls = a["type"] + "/" + (a.get("genre") or "")
        if cap.get(cls, 0) >= (CAP_PODCAST if a["type"] == "podcast" else CAP_GENRE if a["type"] == "music" else 10**9):
            continue      # keep the teacher set balanced: a pile of one class shifts the student's priors
        cap[cls] = cap.get(cls, 0) + 1
        seen.add((r["artist"], r["title"]))
        g = a["genre"] if a["type"] == "music" else "-"
        v = a["vclass"] if a["type"] == "video" else "-"
        src = r["source"]
        if a["type"] == "podcast" and src == "podcasts.apple.com":
            # the same episodes are published on every platform; Spotify is where the student needs them
            src = ("open.spotify.com", "open.spotify.com", "podcasts.apple.com")[idx % 3]
        out.append("\t".join([a["type"], g, v, r["lang"], src, str(r["dur"]), r["artist"], r["title"]]) + "\n")
        if a.get("entity"):
            ents.append(json.dumps({"title": r["title"], "artist": r["artist"], "entity": a["entity"], "type": a["type"]}, ensure_ascii=False) + "\n")
    open(ROOT + "/data/raw/train/t40_deepseek.tsv", "w", encoding="utf-8").writelines(out)
    open(ROOT + "/data/deepseek/entities.jsonl", "w", encoding="utf-8").writelines(ents)
    print(f"kept {len(out)} of {len(rows)} rows (confidence >= {min_conf}); {len(ents)} with a named entity")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("check", "label", "build"):
        print(__doc__); sys.exit(1)
    if cmd != "build" and not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY is not set"); sys.exit(1)
    {"check": cmd_check, "label": cmd_label, "build": cmd_build}[cmd]()
