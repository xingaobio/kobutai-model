#!/usr/bin/env python3
"""Download Apple's public genre charts (top songs per storefront and genre) into data/apple/raw.jsonl."""
import json, os, sys, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENRES = [21, 20, 14, 15, 12, 51, 27, 29, 18, 17, 7, 5, 11, 6, 2, 24, 4, 28, 1289, 13, 1232, 1243, 30, 10, 16, 19, 22, 53, 25]
COUNTRIES = ["us", "gb", "jp", "tw", "hk", "kr", "de", "fr", "cn", "sg", "au", "br"]
out = open(ROOT + "/data/apple/raw.jsonl", "a")
done = set()
if os.path.exists(ROOT + "/data/apple/done.txt"):
    done = set(open(ROOT + "/data/apple/done.txt").read().split())
dn = open(ROOT + "/data/apple/done.txt", "a")
for kind in ("topsongs", "topalbums"):
    for cc in COUNTRIES:
        for g in GENRES:
            key = f"{kind}/{cc}/{g}"
            if key in done:
                continue
            url = f"https://itunes.apple.com/{cc}/rss/{kind}/limit=100/genre={g}/json"
            try:
                d = json.loads(urllib.request.urlopen(url, timeout=30).read())
                ents = d.get("feed", {}).get("entry", [])
                if isinstance(ents, dict):
                    ents = [ents]
                for e in ents:
                    out.write(json.dumps({"kind": kind, "cc": cc, "feed": g, "name": e["im:name"]["label"],
                                          "artist": e["im:artist"]["label"],
                                          "cat": e["category"]["attributes"]["label"],
                                          "catid": e["category"]["attributes"]["im:id"],
                                          "date": e.get("im:releaseDate", {}).get("label", "")[:10]}, ensure_ascii=False) + "\n")
                out.flush(); dn.write(key + "\n"); dn.flush()
                print(key, len(ents), flush=True)
            except Exception as ex:
                print(key, "ERR", str(ex)[:60], flush=True)
            time.sleep(0.4)
print("DONE")
