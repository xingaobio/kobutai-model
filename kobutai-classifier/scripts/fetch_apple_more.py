#!/usr/bin/env python3
"""Real podcast episode titles (Apple podcast charts + iTunes lookup) and music-video titles (Apple music-video charts)
for the unlabelled pool. Output: data/pool/p04_podcasts.tsv, data/pool/p05_musicvideos.tsv."""
import json, os, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get(u):
    return json.loads(urllib.request.urlopen(u, timeout=30).read())
pods, eps = {}, []
for cc in ("us", "gb", "jp", "tw", "hk", "cn", "au", "ca", "sg"):
    try:
        for e in get(f"https://itunes.apple.com/{cc}/rss/toppodcasts/limit=100/json")["feed"]["entry"]:
            pods[e["id"]["attributes"]["im:id"]] = cc
    except Exception as ex:
        print(cc, "ERR", str(ex)[:60])
print(len(pods), "podcasts")
for pid, cc in list(pods.items()):
    try:
        res = get(f"https://itunes.apple.com/lookup?id={pid}&entity=podcastEpisode&limit=6&country={cc}")["results"]
        show = res[0].get("collectionName", "") if res else ""
        for r in res[1:]:
            t = (r.get("trackName") or "").replace("\t", " ").replace("\n", " ").strip()
            if t:
                eps.append(f"?\t?\t?\txx\tpodcasts.apple.com\t{int((r.get('trackTimeMillis') or 0) / 1000)}\t{show}\t{t}\n")
    except Exception:
        pass
    time.sleep(0.25)
open(ROOT + "/data/pool/p04_podcasts.tsv", "w", encoding="utf-8").writelines(dict.fromkeys(eps))
print(len(eps), "episodes")
mv = []
for cc in ("us", "gb", "jp", "tw", "hk", "kr", "de", "fr", "au"):
    try:
        for e in get(f"https://itunes.apple.com/{cc}/rss/topmusicvideos/limit=100/json")["feed"]["entry"]:
            mv.append(f"?\t?\t?\txx\twww.youtube.com\t0\t{e['im:artist']['label']}\t{e['im:name']['label'].replace(chr(9),' ')}\n")
    except Exception as ex:
        print(cc, "ERR", str(ex)[:60])
open(ROOT + "/data/pool/p05_musicvideos.tsv", "w", encoding="utf-8").writelines(dict.fromkeys(mv))
print(len(set(mv)), "music videos")
