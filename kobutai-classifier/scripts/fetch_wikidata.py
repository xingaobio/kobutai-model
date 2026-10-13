#!/usr/bin/env python3
"""Download artist -> genre facts from Wikidata (CC0) into data/wikidata/.
Step 1: all music genres. Step 2: (artist, genre, sitelinks) pairs, batched by genre.
Step 3: labels/aliases/country for the artists, via wbgetentities."""
import csv, io, json, os, sys, time, urllib.parse, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = ROOT + "/data/wikidata"
UA = {"User-Agent": "kobutai-classifier-research/0.1 (xingaouk@gmail.com)"}
MIN_LINKS = int(os.environ.get("MIN_LINKS", "6"))


_last = [0.0]
def sparql(q, tries=4):
    wait = 65 - (time.time() - _last[0])      # the endpoint allows this client about 1 request a minute
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": q})
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=dict(UA, Accept="text/csv"))
            txt = urllib.request.urlopen(req, timeout=90).read().decode("utf-8")
            if "java.util.concurrent" in txt or "TimeoutException" in txt:
                raise RuntimeError("timeout")
            return list(csv.DictReader(io.StringIO(txt)))
        except Exception as e:
            print("  retry", t, str(e)[:80], flush=True)
            time.sleep(70)
    return None


def qid(u):
    return u.rsplit("/", 1)[-1]


def step_genres():
    p = OUT + "/genres.json"
    if os.path.exists(p):
        return json.load(open(p))
    rows = sparql('SELECT ?g ?gLabel WHERE { ?g wdt:P31 wd:Q188451 . SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }')
    g = {qid(r["g"]): r["gLabel"] for r in rows}
    json.dump(g, open(p, "w"), ensure_ascii=False)
    return g


def step_pairs(genres):
    p = OUT + "/pairs.tsv"
    done = set()
    if os.path.exists(p):
        done = {l.split("\t")[1] for l in open(p)}
    ids = [g for g in sorted(genres) if g not in done]
    out = open(p, "a")
    def fetch(batch):
        vals = " ".join("wd:" + g for g in batch)
        q = ("SELECT ?a ?g ?n WHERE { VALUES ?g { %s } ?a wdt:P136 ?g ; wikibase:sitelinks ?n . FILTER(?n >= %d) "
             "FILTER NOT EXISTS { ?a wdt:P175 [] } FILTER NOT EXISTS { ?a wdt:P577 [] } }" % (vals, MIN_LINKS))
        rows = sparql(q, tries=2)
        if rows is None and len(batch) > 1:
            h = len(batch) // 2
            fetch(batch[:h]); fetch(batch[h:]); return
        for r in rows or []:
            out.write(f"{qid(r['a'])}\t{qid(r['g'])}\t{r['n']}\n")
        for g in batch:                       # mark genre as done even when empty
            out.write(f"-\t{g}\t0\n")
        out.flush()
    for i in range(0, len(ids), 400):
        fetch(ids[i:i + 400])
        print(f"pairs: {i + 400}/{len(ids)} genres", flush=True)
    out.close()


def step_entities():
    arts = {}
    for l in open(OUT + "/pairs.tsv"):
        a, g, n = l.rstrip("\n").split("\t")
        if a != "-":
            arts[a] = int(n)
    p = OUT + "/entities.jsonl"
    done = set()
    if os.path.exists(p):
        done = {json.loads(l)["id"] for l in open(p)}
    todo = [a for a in sorted(arts, key=lambda a: -arts[a]) if a not in done]
    out = open(p, "a")
    langs = ["en", "ja", "zh", "zh-hans", "zh-hant", "zh-cn", "zh-tw", "ko"]
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
            "action": "wbgetentities", "ids": "|".join(batch), "format": "json",
            "props": "labels|aliases|claims", "languages": "|".join(langs)})
        for t in range(5):
            try:
                d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
                break
            except Exception as e:
                print("  retry", str(e)[:60], flush=True); time.sleep(5 * (t + 1)); d = None
        if not d:
            continue
        for q, e in d.get("entities", {}).items():
            def claim_ids(pid):
                r = []
                for c in e.get("claims", {}).get(pid, []):
                    v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                    if isinstance(v, dict) and "id" in v:
                        r.append(v["id"])
                return r
            out.write(json.dumps({
                "id": q, "n": arts.get(q, 0),
                "labels": {k: v["value"] for k, v in e.get("labels", {}).items()},
                "aliases": {k: [x["value"] for x in v] for k, v in e.get("aliases", {}).items()},
                "p31": claim_ids("P31"), "genre": claim_ids("P136"), "country": claim_ids("P27") + claim_ids("P495"),
                "occ": claim_ids("P106"), "lang": claim_ids("P1412") + claim_ids("P407"),
            }, ensure_ascii=False) + "\n")
        out.flush()
        if i % 2000 == 0:
            print(f"entities: {i}/{len(todo)}", flush=True)
        time.sleep(0.3)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    g = step_genres(); print(len(g), "genres", flush=True)
    step_pairs(g)
    step_entities()
    print("DONE", flush=True)
