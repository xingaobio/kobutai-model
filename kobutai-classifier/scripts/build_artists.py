#!/usr/bin/env python3
"""Merge data/raw/artists/*.tsv into model/artists.tsv (normalised key -> genre index)."""
import collections, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kobutai_py.core import norm, GENRES, words

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# keys that are ordinary words or site furniture: never usable as an artist match
STOP = set("""official music video audio lyrics lyric live cover remix mv pv ost op ed full album mix playlist
radio edit version remastered remaster the a an and feat ft various artists topic vevo records channel tv
hd hq 4k 8k 1080p bgm asmr vlog news podcast episode trailer teaser dance practice karaoke instrumental
love you me my day night one air eve joy key winter mother kids sleep relax study chill music song songs
piano guitar jazz rock pop rap classic classical anime game games gaming best top new old""".split())


def ok_key(k):
    if k in STOP:
        return False
    if all(ord(c) < 128 for c in k):
        return len(k) >= 3
    return len(k) >= 2


def main():
    votes = collections.defaultdict(collections.Counter)   # from training samples
    for f in sorted(glob.glob(ROOT + "/data/raw/train/*.tsv")):
        for line in open(f, encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            if len(c) == 8 and c[0] == "music":
                votes[norm(c[6])][c[1]] += 1
    entries = {}          # key -> (genre, is_name, file)
    conflicts = []
    skip = [x for x in os.environ.get("KBT_EXCLUDE", "").split(",") if x]
    for f in sorted(glob.glob(ROOT + "/data/raw/artists/*.tsv")):
        if any(x in os.path.basename(f) for x in skip):
            continue
        for line in open(f, encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            if len(c) != 3:
                continue
            name, aliases, g = c
            keys = [(norm(name), True)] + [(norm(a), False) for a in aliases.split(";") if a.strip()]
            for k, is_name in keys:
                if not ok_key(k):
                    continue
                old = entries.get(k)
                if old is None or old[0] == g:
                    if old is None or (is_name and not old[1]):
                        entries[k] = (g, is_name, os.path.basename(f))
                    continue
                # conflict: primary name beats alias; else training votes; else first file
                if is_name != old[1]:
                    win = (g, is_name, os.path.basename(f)) if is_name else old
                else:
                    v = votes.get(k)
                    top = v.most_common(1)[0][0] if v else None
                    win = (g, is_name, os.path.basename(f)) if top == g else old
                conflicts.append(f"{k}\t{old[0]} ({old[2]}) vs {g} ({os.path.basename(f)}) -> {win[0]}")
                entries[k] = win
    os.makedirs(ROOT + "/model", exist_ok=True)
    with open(ROOT + "/model/artists.tsv", "w", encoding="utf-8") as out:
        for k in sorted(entries):
            out.write(f"{k}\t{GENRES.index(entries[k][0])}\n")
    with open(ROOT + "/reports/artist_conflicts.txt", "w", encoding="utf-8") as out:
        out.write("\n".join(conflicts) + "\n")
    cnt = collections.Counter(e[0] for e in entries.values())
    print(len(entries), "keys;", len(conflicts), "conflicts;", dict(cnt))


if __name__ == "__main__":
    main()
