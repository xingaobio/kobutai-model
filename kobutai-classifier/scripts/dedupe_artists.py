#!/usr/bin/env python3
"""Remove from <file> every artist whose name already exists in another artist file. Rewrites <file>."""
import glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kobutai_py.core import norm
path = os.path.abspath(sys.argv[1])
seen = set()
for f in glob.glob(ROOT + "/data/raw/artists/*.tsv"):
    if os.path.abspath(f) == path:
        continue
    for line in open(f, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) == 3:
            seen.add(norm(c[0]))
            seen.update(norm(a) for a in c[1].split(";") if a.strip())
keep, dropped = [], 0
for line in open(path, encoding="utf-8"):
    c = line.rstrip("\n").split("\t")
    if len(c) == 3 and norm(c[0]) in seen:
        dropped += 1
        continue
    if len(c) == 3:
        seen.add(norm(c[0]))
    keep.append(line if line.endswith("\n") else line + "\n")
open(path, "w", encoding="utf-8").writelines(keep)
print(f"dropped {dropped} already-known artists, {len(keep)} rows remain")
