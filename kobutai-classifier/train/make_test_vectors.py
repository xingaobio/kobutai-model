#!/usr/bin/env python3
"""Write model/test-vectors.json: inputs and the reference implementation's outputs.
Every runtime must reproduce these exactly (confidences to 1e-9)."""
import json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kobutai_py.core import Classifier, feature_indices, norm, norm_source, dur_bucket
from train.data import load_val, load_real, load_train

clf = Classifier(ROOT + "/model")
rng = random.Random(3)
dev, test = load_val()
rd, rt = load_real()
rows = rng.sample(dev + test, 260) + rng.sample(rd + rt, 160) + rng.sample(load_train(), 120)
cases = []
def add(title, artist, source, duration, is_live=False, cat=None):
    r = clf.classify(title, artist, source, duration, is_live=is_live, platform_category=cat)
    idx = feature_indices(norm(title), norm(artist), clf.aliases.get(norm_source(source), norm_source(source)) if False else norm_source(source),
                          dur_bucket(-1 if is_live else duration), clf.model.buckets)
    h = 0
    for x in idx:
        h = (h * 31 + x) & 0xFFFFFFFF
    cases.append({"input": {"title": title, "artist": artist, "source": source, "duration": duration,
                            "isLive": is_live, "platformCategory": cat},
                  "featureHash": h, "expected": r})
for r in rows:
    add(r["title"], r["artist"], r["source"], r["dur"])
# hand-written edge cases: normalisation, empty fields, aliases, flags, astral characters
edge = [
    ("", "", "", None), ("   ", "", "www.youtube.com", 0), ("ＹＯＡＳＯＢＩ「アイドル」　Official Music Video", "Ａｙａｓｅ", "WWW.YouTube.com", 213),
    ("Beyoncé – CUFF IT (Official Lyric Video)", "BEYONCÉ", "m.youtube.com", 229.5), ("晴天", "周杰倫", "Spotify.exe", 269),
    ("lofi hip hop radio 📚 beats to relax/study to", "Lofi Girl", "www.youtube.com", float("inf")),
    ("𝓛𝓸𝓿𝓮 𝓢𝓽𝓸𝓻𝔂 🎵🎶", "👩‍🎤", "www.tiktok.com", 15), ("Symphony No. 5 in C Minor, Op. 67: I. Allegro con brio", "Berliner Philharmoniker", "com.apple.Music", 445),
    ("#2100 – Guest Name", "The Joe Rogan Experience", "open.spotify.com", 9800), ("방탄소년단 'Dynamite' Official MV", "HYBE LABELS", "www.youtube.com", 224),
    ("Tab\tand\nnewline  title", "A B", "www.bilibili.com", 59.999), ("x" * 400, "y" * 300, "example.org", 3600),
    ("Ёлка — Прованс", "Ёлка", "music.yandex.ru", 200), ("Für Elise", "Ludwig van Beethoven", "music.apple.com", 175),
    ("ゲーム実況 #12", "", "www.youtube.com", -1), ("Episode 5", "", "www.netflix.com", 2700),
]
for t, a, s, d in edge:
    add(t, a, s, d)
add("Some stream", "someone", "www.youtube.com", 100, is_live=True)
add("Why Is Everyone Suddenly Wearing Crocs?", "Search Engine", "open.spotify.com", 2400, cat="podcast")
add("Opaque title", "", "www.youtube.com", 200, cat="music")
def clean(o):
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o
for c in cases:
    d = c["input"]["duration"]
    if isinstance(d, float) and d == float("inf"):
        c["input"]["duration"] = -1          # JSON has no Infinity; -1 means the same thing
json.dump({"buckets": clf.model.buckets, "cases": cases}, open(ROOT + "/model/test-vectors.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=0)
print(len(cases), "cases")
