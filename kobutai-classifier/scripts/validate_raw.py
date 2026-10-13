#!/usr/bin/env python3
"""Validate a raw sample file (8 columns) or artist file (3 columns)."""
import sys

TYPES = {"music", "video", "live", "podcast"}
GENRES = {"rock", "western-pop", "jpop", "anime", "cpop", "electronic",
          "lofi", "classical", "hiphop", "other"}
VCLASS = {"ent", "info"}
LANGS = {"zh", "ja", "en", "xx"}


def check_sample(c):
    t, g, v, lang, dom, dur, artist, title = c
    if t not in TYPES:
        return f"bad type {t!r}"
    if t == "music":
        if g not in GENRES:
            return f"music needs a genre, got {g!r}"
    elif g != "-":
        return f"genre must be '-' for type {t}"
    if t == "video":
        if v not in VCLASS:
            return f"video needs vclass ent/info, got {v!r}"
    elif v != "-":
        return f"vclass must be '-' for type {t}"
    if lang not in LANGS:
        return f"bad lang {lang!r}"
    if not dom or " " in dom or "." not in dom or dom != dom.lower():
        return f"bad domain {dom!r}"
    try:
        d = int(dur)
        if d < -1:
            return "dur < -1"
    except ValueError:
        return f"dur not an integer: {dur!r}"
    if not title.strip():
        return "empty title"
    return None


def check_artist(c):
    name, aliases, g = c
    if not name.strip():
        return "empty name"
    if g not in GENRES:
        return f"bad genre {g!r}"
    return None


def main(path):
    errs, seen, n = [], set(), 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            n += 1
            c = line.split("\t")
            if len(c) == 8:
                e = check_sample(c)
                key = (c[6], c[7])
            elif len(c) == 3:
                e = check_artist(c)
                key = c[0]
            else:
                e = f"{len(c)} columns (need 8 for samples, 3 for artists)"
                key = line
            if e is None and key in seen:
                e = "duplicate"
            seen.add(key)
            if e:
                errs.append(f"line {i}: {e}")
    for e in errs[:40]:
        print(e)
    if errs:
        print(f"{len(errs)} problem(s) in {n} rows")
        sys.exit(1)
    print(f"OK ({n} rows)")


if __name__ == "__main__":
    main(sys.argv[1])
