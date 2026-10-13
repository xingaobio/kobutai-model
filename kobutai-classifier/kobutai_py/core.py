"""Reference implementation of the Kobutai media classifier (see spec/SPEC.md).

Every runtime (TypeScript, Swift, C#) must reproduce this file's behaviour
exactly; model/test-vectors.json is generated from it.
"""
import json
import math
import struct
import unicodedata

TYPES = ["music", "video", "live", "podcast"]
GENRES = ["rock", "western-pop", "jpop", "anime", "cpop", "electronic",
          "lofi", "classical", "hiphop", "other"]
VCLASSES = ["ent", "info"]
HEADS = [("type", TYPES), ("genre", GENRES), ("vclass", VCLASSES)]


# ---------------------------------------------------------------- text
def norm(s):
    """NFKD, drop combining marks U+0300..U+036F, ASCII-lowercase,
    collapse whitespace."""
    if not s:
        return ""
    out = []
    for ch in unicodedata.normalize("NFKD", s):
        o = ord(ch)
        if 0x300 <= o <= 0x36F:
            continue
        if 0x41 <= o <= 0x5A:
            ch = chr(o + 32)
        elif ch in "\t\n\r\x0b\x0c 　":
            ch = " "
        out.append(ch)
    return _collapse("".join(out))


def _collapse(s):
    parts = [p for p in s.split(" ") if p]
    return " ".join(parts)


def norm_source(s):
    s = norm(s)
    for p in ("www.", "m."):
        if s.startswith(p):
            s = s[len(p):]
    return s


def dur_bucket(d):
    """d: seconds; None/0/NaN = unknown; negative or inf = live."""
    if d is None or d != d or d == 0:
        return "U"
    if d < 0 or d == float("inf"):
        return "L"
    for lim, b in ((60, "a"), (150, "b"), (330, "c"), (600, "d"),
                   (1500, "e"), (3600, "f")):
        if d < lim:
            return b
    return "g"


def fnv1a(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _is_word_char(c):
    return ("a" <= c <= "z") or ("0" <= c <= "9")


def words(text):
    out, cur = [], []
    for c in text:
        if _is_word_char(c):
            cur.append(c)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def feature_strings(title, artist, source, bucket, max_chars=160):
    """Yield the feature strings for one (already normalised) input."""
    feats = []
    for tag, text in (("t", title), ("a", artist)):
        cps = list(text)[:max_chars]
        seq = ["^"] + cps + ["$"]
        n_seq = len(seq)
        for n in (1, 2, 3, 4):
            for i in range(n_seq - n + 1):
                g = "".join(seq[i:i + n])
                feats.append(tag + str(n) + ":" + g)
                if n >= 2:
                    feats.append("x" + str(n) + ":" + g)
        ws = words("".join(cps))
        for w in ws:
            feats.append(tag + "w:" + w)
            feats.append("xw:" + w)
        for i in range(len(ws) - 1):
            feats.append(tag + "b:" + ws[i] + " " + ws[i + 1])
    feats.append("s:" + source)
    feats.append("d:" + bucket)
    feats.append("sd:" + source + "|" + bucket)
    return feats


def feature_indices(title, artist, source, bucket, buckets):
    mask = buckets - 1
    idx = {fnv1a(f.encode("utf-8")) & mask
           for f in feature_strings(title, artist, source, bucket)}
    return sorted(idx)


# ---------------------------------------------------------------- model
class Model:
    def __init__(self, path):
        raw = open(path, "rb").read()
        magic, ver, self.buckets, nheads = struct.unpack_from("<4sIII", raw, 0)
        assert magic == b"KBT1" and ver == 1
        off = 16
        self.heads = []
        for _ in range(nheads):
            (k,) = struct.unpack_from("<I", raw, off)
            (scale,) = struct.unpack_from("<f", raw, off + 4)
            bias = struct.unpack_from("<%df" % k, raw, off + 8)
            off += 8 + 4 * k
            self.heads.append((k, scale, bias))
        self.total = sum(h[0] for h in self.heads)
        self.w = memoryview(raw)[off:off + self.buckets * self.total].cast("b")
        assert len(self.w) == self.buckets * self.total

    def logits(self, idx):
        sums = [0] * self.total
        w, total = self.w, self.total
        for i in idx:
            base = i * total
            for k in range(total):
                sums[k] += w[base + k]
        inv = 1.0 / math.sqrt(len(idx))
        out, off = [], 0
        for k, scale, bias in self.heads:
            out.append([bias[j] + scale * sums[off + j] * inv
                        for j in range(k)])
            off += k
        return out


def softmax(z, allowed=None):
    m = max(v for j, v in enumerate(z) if allowed is None or allowed[j])
    e = [math.exp(v - m) if (allowed is None or allowed[j]) else 0.0
         for j, v in enumerate(z)]
    s = 0.0
    for v in e:
        s += v
    return [v / s for v in e]


def argmax(p):
    best = 0
    for j in range(1, len(p)):
        if p[j] > p[best]:
            best = j
    return best


# ---------------------------------------------------------------- artists
SEPARATORS = "-–—|｜/／「」『』【】()（）[]《》〈〉<>,、，&×·・:：;；\"“”'‘’~〜～!！?？#@*+=。"


def title_segments(title_norm):
    segs, cur = [], []
    for c in title_norm:
        if c in SEPARATORS:
            segs.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    segs.append("".join(cur))
    out = []
    for s in segs:
        s = _collapse(s)
        for cut in (" feat", " ft ", " ft.", " x "):
            p = s.find(cut)
            if p > 0:
                out.append(s[:p].strip())
        if s:
            out.append(s)
    return out


class ArtistTable:
    def __init__(self, path):
        self.map = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    k, g = line.split("\t")
                    self.map[k] = int(g)

    def lookup(self, artist_norm, title_norm):
        """Return (genre_index, where) or (None, None)."""
        if artist_norm:
            g = self.map.get(artist_norm)
            if g is not None:
                return g, "artist"
            for seg in title_segments(artist_norm):
                g = self.map.get(seg)
                if g is not None:
                    return g, "artist"
        for seg in title_segments(title_norm):
            g = self.map.get(seg)
            if g is not None:
                return g, "title"
        return None, None


# ---------------------------------------------------------------- pipeline
class Classifier:
    def __init__(self, model_dir):
        self.cfg = json.load(open(model_dir + "/config.json", encoding="utf-8"))
        self.model = Model(model_dir + "/kobutai.bin")
        self.artists = ArtistTable(model_dir + "/artists.tsv")
        r = self.cfg["rules"]
        self.music_sources = set(r["musicSources"])
        self.podcast_sources = set(r["podcastSources"])
        self.video_sources = set(r["videoSources"])
        self.aliases = self.cfg["sourceAliases"]

    def classify(self, title="", artist="", source="", duration=None,
                 is_live=False, platform_category=None, detail=False):
        cfg = self.cfg
        t, a = norm(title), norm(artist)
        s = norm_source(source)
        s = self.aliases.get(s, s)
        if is_live:
            duration = -1
        b = dur_bucket(duration)
        idx = feature_indices(t, a, s, b, self.model.buckets)
        z_type, z_genre, z_vclass = self.model.logits(idx)

        # --- type
        via = "model"
        allowed = None
        forced = None
        if platform_category == "music" or s in self.music_sources:
            forced = 0
        elif platform_category == "podcast" or s in self.podcast_sources:
            forced = 3
        elif s in self.video_sources:
            allowed = [False, True, False, False]
        elif b == "L":
            allowed = [True, False, True, False]
        if forced is not None:
            ti, tconf, via = forced, 1.0, "rule"
            p_type = [1.0 if j == forced else 0.0 for j in range(4)]
        else:
            p_type = softmax(z_type, allowed)
            ti = argmax(p_type)
            tconf = p_type[ti]
        res = {"type": TYPES[ti], "typeConfidence": tconf, "typeVia": via,
               "genre": None, "genreConfidence": 0.0, "genreVia": None,
               "vclass": None, "vclassConfidence": 0.0}
        th = cfg["thresholds"]
        if tconf < th["type"]:
            res["type"] = None
        elif ti == 0:
            g, where = self.artists.lookup(a, t)
            zg = list(z_genre)
            if g is not None:
                zg[g] += cfg["artistBonus"][where]
            p = softmax(zg)
            gi = argmax(p)
            res["genreConfidence"] = p[gi]
            if p[gi] >= th["genre"]:
                res["genre"] = GENRES[gi]
                res["genreVia"] = "table" if (g is not None and gi == g) else "model"
        elif ti == 1:
            p = softmax(z_vclass)
            vi = argmax(p)
            res["vclassConfidence"] = p[vi]
            if p[vi] >= th["vclass"]:
                res["vclass"] = VCLASSES[vi]
        if detail:
            res["_p_type"] = p_type
            res["_z"] = [z_type, z_genre, z_vclass]
        return res
