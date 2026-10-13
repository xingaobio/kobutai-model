"""Shared data loading for training and evaluation."""
import glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kobutai_py.core import fnv1a


def load(pattern):
    rows = []
    for f in sorted(glob.glob(ROOT + "/" + pattern)):
        for line in open(f, encoding="utf-8"):
            c = line.rstrip("\n").rstrip("\r").split("\t")
            if len(c) != 8:
                continue
            rows.append({"type": c[0], "genre": c[1], "vclass": c[2], "lang": c[3],
                         "source": c[4], "dur": int(c[5]), "artist": c[6], "title": c[7],
                         "file": os.path.basename(f)})
    return rows


def load_train():
    skip = [x for x in os.environ.get("KBT_EXCLUDE", "").split(",") if x]
    # guard: a title that appears in any evaluation set is never trained on
    held = {r["title"].strip().lower() for r in load("data/raw/val/*.tsv") + load("data/raw/real/*.tsv") + load("data/raw/focus/*.tsv")}
    rows = [r for r in load("data/raw/train/*.tsv")
            if not any(x in r["file"] for x in skip) and r["title"].strip().lower() not in held]
    # teacher corrections (scripts/apply_teacher.py): drop disputed hand-written rows, relabel rule-labelled ones
    ovp = ROOT + "/data/deepseek/overrides.tsv"
    if os.path.exists(ovp) and "teacher" not in skip:
        ov = {}
        for line in open(ovp, encoding="utf-8"):
            c = line.rstrip("\n").split("\t")
            ov[(c[0], c[1], c[2])] = c[3:]
        out = []
        for r in rows:
            o = ov.get((r["file"], r["artist"], r["title"]))
            if o is None:
                out.append(r)
            elif o[0] != "DROP":
                out.append(dict(r, type=o[0], genre=o[1], vclass=o[2]))
        rows = out
    return rows


def load_val(pattern="data/raw/val/*.tsv"):
    """Returns (dev, test). dev tunes thresholds and drives error analysis;
    test is only ever used for the reported numbers."""
    rows = load(pattern)
    dev, test = [], []
    for r in rows:
        (dev if fnv1a((r["artist"] + "|" + r["title"]).encode()) % 5 < 2 else test).append(r)
    return dev, test


def load_real():
    """Rows taken from live chart / trending pages (data/raw/real). Same dev/test split rule."""
    return load_val("data/raw/real/*.tsv")
