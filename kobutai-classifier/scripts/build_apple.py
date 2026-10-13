#!/usr/bin/env python3
"""Turn data/apple/raw.jsonl (public genre charts) into an artist file and a training sample file."""
import collections, json, os, random, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kobutai_py.core import norm
from train.data import load_real

tree = json.load(open(ROOT + "/data/apple/genres.json"))["34"]["subgenres"]
top_of = {}
def walk(node_id, node, top):
    top_of[node_id] = top
    for k, v in (node.get("subgenres") or {}).items():
        walk(k, v, top)
for k, v in tree.items():
    walk(k, v, k)

WEST = {"us", "gb", "au", "de", "fr", "br"}
HAN = re.compile(r"[一-鿿]"); KANA = re.compile(r"[぀-ヿ]"); HANGUL = re.compile(r"[가-힯]")
def script(s):
    if KANA.search(s): return "ja"
    if HANGUL.search(s): return "ko"
    if HAN.search(s): return "zh"
    return "latin"

FIXED = {"21": "rock", "20": "rock", "12": "western-pop", "51": "western-pop", "1243": "western-pop", "27": "jpop",
         "28": "other", "29": "anime", "1232": "cpop", "18": "hiphop", "17": "electronic", "7": "electronic",
         "5": "classical", "11": "other", "6": "other", "2": "other", "24": "other", "4": "other", "22": "other",
         "13": "lofi"}
def label(d):
    top = top_of.get(d["catid"], d["catid"])
    sc = script(d["artist"] + " " + d["name"])
    if top in FIXED:
        g = FIXED[top]
        if g == "hiphop" or g == "rock" or g == "electronic":
            return g
        return g
    if top in ("14", "15", "1289"):                 # Pop, R&B/Soul, Folk: language decides
        if sc == "ja": return "jpop" if top != "1289" else "other"
        if sc == "ko": return "western-pop"
        if sc == "zh": return "cpop" if d["cc"] in ("tw", "hk", "cn", "sg") else None
        if d["cc"] in WEST: return "western-pop" if top != "1289" else "other"
        return None
    return None

# hand-curated artist lists follow our labelling guide; where they know the artist, they overrule the chart's genre
curated = {}
import glob
for f in sorted(glob.glob(ROOT + "/data/raw/artists/[ab]*.tsv")):
    for line in open(f, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) == 3:
            for k in [c[0]] + [a for a in c[1].split(";") if a.strip()]:
                curated.setdefault(norm(k), c[2])
_label = None
rows = [json.loads(l) for l in open(ROOT + "/data/apple/raw.jsonl")]
votes = collections.defaultdict(collections.Counter)
songs = {}
for d in rows:
    g = label(d)
    if not g:
        continue
    g = curated.get(norm(d["artist"]), g)
    votes[d["artist"]][g] += 1
    if d["kind"] == "topsongs":
        songs.setdefault((d["artist"], d["name"]), (g, d["cc"]))
with open(ROOT + "/data/raw/artists/c01_apple_charts.tsv", "w", encoding="utf-8") as f:
    n = 0
    for a, v in sorted(votes.items()):
        (g, c), tot = v.most_common(1)[0], sum(v.values())
        if c / tot < 0.7 or "\t" in a or len(a) > 60:
            continue
        f.write(f"{a}\t\t{g}\n"); n += 1
print("artists:", n)
rt = load_real()                                   # never train on a row that is in the real evaluation set
real_keys = {(norm(r["artist"]), norm(r["title"])) for part in rt for r in part}
rng = random.Random(11)
DOM = {"jp": ["music.apple.com", "open.spotify.com", "music.youtube.com", "music.amazon.co.jp", "music.line.me"],
       "cn": ["music.163.com", "y.qq.com", "www.kugou.com", "music.apple.com"],
       "tw": ["music.apple.com", "open.spotify.com", "music.youtube.com", "www.kkbox.com"],
       "hk": ["music.apple.com", "open.spotify.com", "music.youtube.com", "www.kkbox.com"]}
DEF = ["open.spotify.com", "open.spotify.com", "music.apple.com", "music.youtube.com", "soundcloud.com", "www.deezer.com", "tidal.com"]
per = collections.Counter(); out = []
items = list(songs.items()); rng.shuffle(items)
for (a, t), (g, cc) in items:
    if (norm(a), norm(t)) in real_keys or "\t" in t or "\t" in a:
        continue
    if per[g] >= 900:
        continue
    per[g] += 1
    sc = script(t)
    lang = {"ja": "ja", "zh": "zh", "ko": "xx", "latin": "en"}[sc]
    out.append(f"music\t{g}\t-\t{lang}\t{rng.choice(DOM.get(cc, DEF))}\t0\t{a}\t{t}\n")
open(ROOT + "/data/raw/train/t20_apple_charts.tsv", "w", encoding="utf-8").writelines(out)
print("songs:", len(out), dict(per))
