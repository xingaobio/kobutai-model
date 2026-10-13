# Kobutai classifier specification (v1)

The Python file `kobutai_py/core.py` is the reference. A port is correct when it reproduces
`model/test-vectors.json` exactly (labels equal, confidences within 1e-9).

## 1. Input

| field | meaning |
|---|---|
| `title` | media title as the site or OS reports it |
| `artist` | artist, or channel / uploader name; may be empty |
| `source` | host name (`www.youtube.com`) or desktop app id (`Spotify.exe`) |
| `duration` | seconds; missing, 0 or NaN = unknown; negative or infinite = live |
| `isLive` | optional flag; same effect as an infinite duration |
| `platformCategory` | optional: `"music"` or `"podcast"` when the platform itself says so |

## 2. Normalisation

`norm(s)`: Unicode NFKD → drop code points U+0300–U+036F → map ASCII `A–Z` to `a–z` (no other case mapping) →
turn TAB, LF, VT, FF, CR, U+00A0, U+3000 into a space → split on spaces, drop empty parts, join with one space.

`normSource(s)`: `norm`, then strip one leading `www.` and then one leading `m.`; then replace through
`config.sourceAliases` if present.

Duration bucket: unknown `U`; live `L`; `<60` a; `<150` b; `<330` c; `<600` d; `<1500` e; `<3600` f; else g.

## 3. Features

All strings are sequences of Unicode code points. For each field (tag `t` = title, `a` = artist), take the
first 160 code points, then:

- with `^` prepended and `$` appended, every n-gram for n = 1..4 → `"{tag}{n}:{gram}"`; for n ≥ 2 also `"x{n}:{gram}"`;
- every maximal run of `[a-z0-9]` (a "word") → `"{tag}w:{word}"` and `"xw:{word}"`;
- every pair of adjacent words → `"{tag}b:{w1} {w2}"`.

Plus `"s:{source}"`, `"d:{bucket}"`, `"sd:{source}|{bucket}"`.

Each feature string is hashed with 32-bit FNV-1a over its UTF-8 bytes (offset 0x811C9DC5, prime 0x01000193)
and reduced with `& (buckets - 1)`. The feature set is the set of distinct indices; `n` is its size.

## 4. Model file `kobutai.bin` (little-endian)

```
"KBT1"  u32 version(=1)  u32 buckets  u32 headCount
per head:  u32 classCount  f32 scale  f32 bias[classCount]
int8 weights[buckets][sum of classCount]      row-major, heads concatenated in order
```

Heads in order: type (4), genre (10), vclass (2). For head h and class j:

```
z[j] = bias[j] + scale * (Σ over feature indices i of weights[i][offset_h + j]) / sqrt(n)
```

The sum is an exact integer; everything after it is IEEE double arithmetic in the order written.
Softmax subtracts the maximum first; "argmax" takes the first maximum.

## 5. Pipeline

1. Type.
   - `platformCategory == "music"` or source ∈ `rules.musicSources` → music, confidence 1, via `rule`.
   - else `platformCategory == "podcast"` or source ∈ `rules.podcastSources` → podcast, confidence 1, via `rule`.
   - else source ∈ `rules.videoSources` → softmax restricted to {video}.
   - else bucket `L` → softmax restricted to {music, live}.
   - else softmax over all four.
   - confidence < `thresholds.type` → `type = null`, stop.
2. Music → genre. Look the artist up in `artists.tsv` (§6). On a hit add `artistBonus.artist` or
   `artistBonus.title` to that genre's logit. Softmax; confidence < `thresholds.genre` → `genre = null`.
   `genreVia` is `table` when the winning genre is the table's, otherwise `model`.
3. Video → vclass. Softmax; confidence < `thresholds.vclass` → `vclass = null`.

`null` anywhere means "use the generic character pool".

## 6. Artist table

`artists.tsv`: one `normalised key<TAB>genre index` per line. Lookup order, first hit wins:

1. the whole normalised artist field;
2. each segment of the artist field;
3. each segment of the title (this hit counts as `title`).

Segments: split on any of ``-–—|｜/／「」『』【】()（）[]《》〈〉<>,、，&×·・:：;；"“”'‘’~〜～!！?？#@*+=。``; collapse spaces;
for each segment, if it contains ` feat`, ` ft `, ` ft.` or ` x ` at a position > 0, the part before it
(trimmed) is tried first, then the segment itself. Keys are compared code point by code point.

## 7. Labels

See `LABELING.md`. Notable decisions: music wins over live; K-pop and R&B are `western-pop`;
game soundtracks, Vocaloid and VTuber songs are `anime`; ambient and sleep sounds are `lofi`;
enka, jazz, country, folk, children's music are `other`.
