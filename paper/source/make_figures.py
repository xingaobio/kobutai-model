#!/usr/bin/env python3
"""Figures for the paper. Every number is copied from inputs/experimental_log.md.
Run: python3 scripts/make_figures.py  ->  figures/*.png and figures/*.pdf"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


def wilson(p, n=383, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


# ------------------------------------------------------------------ 1 pipeline
def pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    boxes = [
        (0.5, "Input\ntitle · artist\nsite · duration"),
        (15.0, "Normalise\nNFKD, ASCII\nlower-case"),
        (29.5, "Site rules\nforce or\nrestrict type"),
        (44.0, "Hashed\nfeatures\nchar n-grams\n$2^{17}$ buckets"),
        (58.5, "3 int8 linear\nheads\n+ artist-table\nbonus"),
        (73.0, "Thresholds\nlabel or\nabstain"),
    ]
    w, h, y = 12.0, 15.5, 13.5
    for x, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc="#eef4fc", ec=BLUE, lw=1.0))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.0, color=INK, linespacing=1.25)
    for (x, _), (x2, _) in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((x + w + 0.5, y + h / 2), (x2 - 0.5, y + h / 2), arrowstyle="-|>",
                                     mutation_scale=8, color=INK2, lw=0.9))
    # server branch
    sx = 87.5
    ax.add_patch(FancyBboxPatch((sx, y), 12.0, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc="#fdf0ea", ec=ORANGE, lw=1.0, ls="--"))
    ax.text(sx + 6, y + h / 2, "Large\nlanguage\nmodel\n(server)", ha="center", va="center",
            fontsize=7.0, color=INK, linespacing=1.25)
    ax.add_patch(FancyArrowPatch((73.0 + w + 0.5, y + h / 2), (sx - 0.5, y + h / 2), arrowstyle="-|>",
                                 mutation_scale=8, color=ORANGE, lw=0.9, ls="--"))
    ax.text(sx + 6, y + h + 2.2, "only if confidence < $\\tau$", ha="center", fontsize=6.8, color=ORANGE)
    # on-device bracket
    ax.plot([0.5, 86.0], [8.5, 8.5], color=BLUE, lw=0.8)
    for xx in (0.5, 86.0):
        ax.plot([xx, xx], [8.5, 10.5], color=BLUE, lw=0.8)
    ax.text(43.25, 3.2, "On device: 2.1 MB weights + artist table, < 1 ms per item; Python and TypeScript outputs verified identical",
            ha="center", fontsize=7.2, color=INK2)
    save(fig, "fig_pipeline_overview")


# ------------------------------------------------------------------ 2 models
def models():
    rows = [  # name, size label, fully right, kind
        ("Linear, round 1", "2.1 MB", 77.8, "dev"),
        ("Linear, no chart data", "2.1 MB", 79.9, "dev"),
        ("Transformer (mMiniLMv2-L6)", "428 MB", 84.1, "dev"),
        ("Linear, round 5 (shipped)", "2.1 MB", 84.9, "dev"),
        ("Linear + transformer", "430 MB", 86.7, "dev"),
        ("DeepSeek V4.1 Flash", "server", 95.3, "srv"),
        ("Hybrid: linear, LLM below 0.9", "2.1 MB + server", 95.6, "srv"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for i, (name, size, v, kind) in enumerate(rows):
        lo, hi = wilson(v / 100)
        c = BLUE if kind == "dev" else ORANGE
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round", alpha=0.45)
        ax.plot(v, i, "o" if kind == "dev" else "s", color=c, ms=6.5, mec="white", mew=1.0, zorder=3)
        ax.text(hi + 0.6, i, f"{v:.1f}%", va="center", fontsize=7.8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{n}  ({s})" for n, s, _, _ in rows], fontsize=7.8)
    ax.set_xlim(70, 101); ax.set_xlabel("Fully right on 383 real titles (%), with 95% Wilson interval")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=BLUE, label="on device"); ax.plot([], [], "s", color=ORANGE, label="uses a server LLM")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "fig_size_vs_accuracy_real_test")


# ------------------------------------------------------------------ 3 hybrid
def hybrid():
    share_rt = [0.0, 3.9, 11.2, 16.7, 21.4, 25.3, 31.9, 40.5, 100.0]
    sets = {  # fully right at tau = 0, .5, .6, .7, .8, .85, .9, .95, always
        "Real test (383)": ([84.1, 86.2, 88.5, 90.6, 92.4, 93.5, 95.6, 96.3, 95.3], BLUE, "o"),
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ys, c, m = sets["Real test (383)"]
    ax.plot(share_rt, ys, color=c, lw=2, marker=m, ms=5, mec="white", mew=0.8, zorder=3)
    taus = ["0", ".5", ".6", ".7", ".8", ".85", ".9", ".95", "all"]
    for x, y, t in zip(share_rt, ys, taus):
        if t in ("0", ".7", ".9", "all"):
            ax.annotate(f"τ={t}" if t != "all" else "everything\nto LLM", (x, y), textcoords="offset points",
                        xytext=((4, -12) if t != ".9" else (7, -11)) if t != "all" else (-6, -22), fontsize=7.2, color=INK2,
                        ha="left" if t != "all" else "right")
    ax.axhline(95.3, color=ORANGE, lw=1, ls="--")
    ax.text(99, 95.3 + 0.5, "LLM alone 95.3%", ha="right", fontsize=7.2, color=INK2)
    ax.axhline(84.9, color=MUTED, lw=1, ls=":")
    ax.text(99, 84.9 + 0.5, "linear alone, no thresholds 84.9%", ha="right", fontsize=7.2, color=INK2)
    ax.annotate("95.6% with 31.9% sent", (31.9, 95.6), xytext=(42, 89.5), fontsize=7.6, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_xlabel("Items handed to the large model (%)")
    ax.set_ylabel("Fully right, real test (%)")
    ax.set_xlim(-2, 102); ax.set_ylim(82, 98)
    save(fig, "fig_hybrid_handoff_curve")


# ------------------------------------------------------------------ 4 per genre
def per_genre():
    g = [  # genre, n, shipped, no-chart, deepseek
        ("rock", 23, 91.3, 69.6, 95.7), ("western-pop", 26, 92.3, 88.5, 100.0), ("jpop", 28, 96.4, 82.1, 100.0),
        ("anime", 29, 58.6, 69.0, 79.3), ("cpop", 24, 100.0, 87.5, 100.0), ("electronic", 14, 92.9, 78.6, 92.9),
        ("lofi", 15, 46.7, 46.7, 80.0), ("classical", 15, 100.0, 100.0, 100.0), ("hiphop", 18, 61.1, 61.1, 83.3),
        ("other", 22, 68.2, 50.0, 100.0),
    ]
    g.sort(key=lambda r: r[4] - r[2])  # largest gap at the top
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    bh = 0.26
    series = [("Linear, shipped", 2, BLUE), ("Linear, no chart data", 3, AQUA), ("DeepSeek V4.1 Flash", 4, ORANGE)]
    for k, (lab, col, c) in enumerate(series):
        ys = [i + (1 - k) * (bh + 0.02) for i in range(len(g))]
        ax.barh(ys, [r[col] for r in g], height=bh, color=c, label=lab, edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{r[0]} ({r[1]})" for r in g], fontsize=7.6)
    ax.set_xlim(0, 102); ax.set_xlabel("Genre accuracy, real-test music (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=1, fontsize=7.4)
    save(fig, "fig_per_genre_accuracy_real_test")


# ------------------------------------------------------------------ 5 synthetic vs real
def synth_real():
    rows = [("Round 1", 96.7, 81.4), ("Round 2", 97.7, 84.1), ("Round 5 (shipped)", 96.0, 85.5)]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, (r, s, re) in enumerate(rows[::-1]):
        ax.plot([re, s], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.plot(s, i, "o", color=BLUE, ms=7, mec="white", mew=1, zorder=3)
        ax.plot(re, i, "D", color=ORANGE, ms=6.5, mec="white", mew=1, zorder=3)
        ax.text((s + re) / 2, i + 0.22, f"{s - re:.1f} pts", ha="center", fontsize=7.6, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xlim(79, 100); ax.set_ylim(-0.6, 2.7)
    ax.set_xlabel("Genre accuracy when answered (%)")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=BLUE, label="synthetic test (562, LLM-written)")
    ax.plot([], [], "D", color=ORANGE, label="real test (383, scraped)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, fontsize=7.4)
    save(fig, "fig_synthetic_vs_real_gap")


if __name__ == "__main__":
    pipeline(); models(); hybrid(); per_genre(); synth_real()
    captions = {
        "fig_pipeline_overview": "The classification pipeline. Everything left of the dashed branch runs on the device; items whose combined confidence falls below the threshold tau can optionally be handed to a server-side large language model that sees the same four fields.",
        "fig_size_vs_accuracy_real_test": "Fully-right accuracy on the 383-item real test set with 95% Wilson intervals; model size in parentheses. A 200-times larger multilingual encoder does not beat the 2.1 MB linear model; only access to a large language model closes the gap.",
        "fig_hybrid_handoff_curve": "Real-test accuracy when items whose combined confidence is below tau are handed to the large language model, against the share handed off. The leftmost point is the shipped model alone with its thresholds, where abstentions count as errors (84.1%); the dotted line is the same model always answering (84.9%). Handing off about one third of items matches or exceeds sending everything.",
        "fig_per_genre_accuracy_real_test": "Per-genre accuracy on real-test music items, ordered by the gap between the large language model and the shipped linear model. The largest gaps (other, lofi, hiphop, anime) are genres that require recognising the specific artist or song.",
        "fig_synthetic_vs_real_gap": "Genre accuracy when answered on the LLM-written synthetic test split versus the scraped real test set, by development round. The synthetic set overstated real-world genre accuracy by 10.5 to 15.3 points.",
    }
    json.dump(captions, open(os.path.join(OUT, "captions.json"), "w"), indent=1)
    print("ok", sorted(os.listdir(OUT)))
