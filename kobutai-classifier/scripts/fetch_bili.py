#!/usr/bin/env python3
"""Download Bilibili per-partition rankings and the popular list (title, uploader, partition, duration)."""
import json, os, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com/"}
out = open(ROOT + "/data/bili/raw.jsonl", "w")
def get(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=30).read())
def dump(lst, src):
    for x in lst:
        out.write(json.dumps({"src": src, "title": x["title"], "owner": x["owner"]["name"], "tname": x.get("tname", ""),
                              "tnamev2": x.get("tnamev2", ""), "dur": x.get("duration", 0), "pubdate": x.get("pubdate", 0)},
                             ensure_ascii=False) + "\n")
    out.flush()
for rid in (0, 1, 3, 4, 5, 36, 188, 160, 211, 217, 119, 155, 181, 234, 223, 129, 202, 1005, 1006, 1008, 1010, 1011, 1012, 1013, 1020, 1021):
    try:
        d = get(f"https://api.bilibili.com/x/web-interface/ranking/v2?rid={rid}&type=all")
        dump(d["data"]["list"], f"rank{rid}")
        print("rank", rid, len(d["data"]["list"]), flush=True)
    except Exception as e:
        print("rank", rid, "ERR", str(e)[:60], flush=True)
    time.sleep(1.5)
for pn in range(1, 26):
    try:
        d = get(f"https://api.bilibili.com/x/web-interface/popular?ps=20&pn={pn}")
        dump(d["data"]["list"], "popular")
    except Exception as e:
        print("popular", pn, "ERR", str(e)[:60], flush=True)
    time.sleep(1.0)
print("DONE", flush=True)
