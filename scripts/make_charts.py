#!/usr/bin/env python3
"""Draw every chart in ../charts from the CSV files in ../results. Re-run after collect_results.py."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES, OUT = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "charts")
os.makedirs(OUT, exist_ok=True)

# reference palette (validated: adjacent CVD dE >= 9.2, normal-vision >= 27.6 for slots 1-3)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titleweight": "semibold",
                     "axes.titlesize": 12, "axes.titlecolor": INK, "legend.frameon": False})


def rd(name):
    return list(csv.DictReader(open(os.path.join(RES, name), encoding="utf-8")))


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def rounded_barh(ax, y, w, h, color, left=0.0):
    """Horizontal bar, square at the baseline, ~rounded data end."""
    ax.barh(y, w, height=h, left=left, color=color, linewidth=0)


def top_legend(ax, handles, labels, ncol=3):
    ax.legend(handles=handles, labels=labels, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, fontsize=9,
              handlelength=1.2, columnspacing=1.4, borderaxespad=0.3)


def save(fig, name, note):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())          # inches; place the note under everything drawn
    fig.text(0.01, bb.y0 / fig.get_figheight() - 0.02, note, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)
    fig.savefig(os.path.join(OUT, name), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


pct = lambda v: f"{100*float(v):.1f}%"

# ------------------------------------------------------------ 1. progress over adopted rounds (real test, thresholded)
rows = [r for r in rd("round_metrics_thresholded.csv") if r["split"] == "real_test"]
order = [("iter1b", "1"), ("iter2", "2"), ("iter3", "3"), ("iter4", "4"), ("iter5", "5"), ("iter6b", "6*")]
by = {r["round"]: r for r in rows}
labs = [lab for k, lab in order if k in by]
xs = list(range(len(labs)))
fig, ax = plt.subplots(figsize=(8, 4.6))
for key, col, name in (("type_acc", S1, "Content type"), ("genre_acc_when_answered", S2, "Genre (when it answers)"),
                       ("vclass_acc_when_answered", S3, "Video ent / info (when it answers)")):
    ys = [float(by[k][key]) * 100 for k, lab in order if k in by]
    ax.plot(xs[4:], ys[4:], color=col, lw=1.5, ls=(0, (2, 2)))
    ax.plot(xs[5:], ys[5:], color=col, lw=0, marker="o", ms=7, mfc=SURF, mec=col, mew=2)
    ax.plot(xs[:5], ys[:5], color=col, lw=2, solid_capstyle="round", label=name, marker="o", ms=7, mec=SURF, mew=2)
ax.set_xticks(xs); ax.set_xticklabels(labs)
ax.set_ylim(75, 100); ax.set_ylabel("Accuracy on real test set (%)"); ax.set_xlabel("Round (6* = DeepSeek distillation, not adopted)")
ax.set_title("Accuracy on the real-world test set, round by round", loc="left", pad=30)
style(ax)
from matplotlib.lines import Line2D
top_legend(ax, [Line2D([], [], color=c, lw=2, marker="o", ms=6, mec=SURF) for c in (S1, S2, S3)],
           ["Content type", "Genre (when it answers)", "Video ent / info (when it answers)"])
save(fig, "01_rounds_real_test.png", "Real test = 383 titles scraped from live chart / trending pages on 2026-10-13, never used for training. "
     "Shipped thresholds; 'when it answers' excludes items routed to the generic character.")

# ------------------------------------------------------------ 2. model comparison (fully right, no thresholds)
cmp_rows = rd("model_comparison.csv")
models = [("linear_iter1", "Linear, round 1 (2.1 MB)"), ("linear_iter4", "Linear, round 4 (2.1 MB)"),
          ("linear_iter5", "Linear, round 5, shipped (2.1 MB)"), ("transformer", "Transformer mMiniLM-L6 (428 MB fp32)"),
          ("linear+transformer", "Linear + transformer averaged"), ("deepseek_v4.1_flash", "DeepSeek V4.1 Flash (server)"),
          ("hybrid_iter5+deepseek@0.9", "Hybrid: linear, DeepSeek when unsure")]
sets = [("real_test", "Real test (383)", S1), ("focus", "Memes / idols / anime (360)", S2)]
fig, ax = plt.subplots(figsize=(9, 5.6))
h = 0.36
for j, (sk, sl, col) in enumerate(sets):
    vals = {r["model"]: float(r["fully_right"]) for r in cmp_rows if r["set"] == sk}
    for i, (mk, ml) in enumerate(models):
        y = len(models) - 1 - i + (h / 2 + 0.02 if j == 0 else -h / 2 - 0.02)
        rounded_barh(ax, y, vals[mk] * 100, h, col)
        ax.text(vals[mk] * 100 + 0.4, y, f"{vals[mk]*100:.1f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(models))); ax.set_yticklabels([m[1] for m in models][::-1], color=INK2)
ax.set_xlim(60, 100); ax.set_xlabel("Fully right (%) - type, and genre or video class, all correct")
ax.set_title("Every model compared on the same items, same pipeline", loc="left", pad=30)
style(ax, "x")
top_legend(ax, [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in sets], [l for _, l, _ in sets], 2)
save(fig, "02_model_comparison.png", "No confidence thresholds (every item answered); the same site rules and artist-table bonus for every model. "
     "Axis starts at 60%.")

# ------------------------------------------------------------ 3. hybrid curve
hy = rd("hybrid_curve.csv")
fig, ax = plt.subplots(figsize=(8, 4.6))
for sk, sl, col in (("real_test", "Real test (383)", S1), ("real_dev", "Real dev (254)", S3), ("focus", "Memes / idols / anime (360)", S2)):
    pts = sorted([(float(r["share_sent_to_deepseek"]) * 100, float(r["fully_right"]) * 100, float(r["threshold"]))
                  for r in hy if r["set"] == sk])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2, marker="o", ms=6, mec=SURF, mew=2, label=sl)
    p9 = [p for p in pts if abs(p[2] - 0.9) < 1e-9][0]
    if sk == "real_test":
        ax.annotate(f"threshold 0.9:\n{p9[0]:.0f}% sent, {p9[1]:.1f}% right", xy=(p9[0], p9[1]), xytext=(p9[0] + 12, p9[1] - 6),
                    fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.set_xlabel("Share of items handed to DeepSeek (%)"); ax.set_ylabel("Fully right (%)"); ax.set_ylim(80, 100)
ax.set_title("Small model + DeepSeek: how much to hand off", loc="left", pad=30)
style(ax)
h_, l_ = ax.get_legend_handles_labels(); top_legend(ax, h_, l_)
save(fig, "03_hybrid_curve.png", "Left end = small model alone (shipped thresholds, abstentions count as wrong); right end = DeepSeek alone. "
     "Points = confidence thresholds 0, .5, .6, .7, .8, .85, .9, .95, all.")

# ------------------------------------------------------------ 4. focus subsets before / after / teacher
fs = rd("focus_subsets.csv")
subs = [("memes", "Internet memes"), ("japanese_idols", "Japanese idols"), ("anime", "Anime / anime music")]
mods = [("linear_iter4", "Before (round 4)", S1), ("linear_iter5", "After targeted data (round 5)", S2), ("deepseek_v4.1_flash", "DeepSeek V4.1 Flash", S3)]
fig, ax = plt.subplots(figsize=(8, 4.4))
w = 0.26
for j, (mk, ml, col) in enumerate(mods):
    for i, (sk, sl) in enumerate(subs):
        v = [float(r["fully_right"]) for r in fs if r["model"] == mk and r["subset"] == sk][0] * 100
        x = i + (j - 1) * (w + 0.02)
        ax.bar(x, v, width=w, color=col, linewidth=0)
        ax.text(x, v + 0.6, f"{v:.0f}", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(range(3)); ax.set_xticklabels([s[1] for s in subs], color=INK2); ax.set_ylim(60, 100)
ax.set_ylabel("Fully right (%)"); ax.set_title("Targeted training on memes, idols and anime (120 items each)", loc="left", pad=30)
style(ax); top_legend(ax, [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in mods], [l for _, l, _ in mods])
save(fig, "04_focus_subsets.png", "Focus set written by an independent sub-agent; optimistic in absolute terms. No thresholds. Axis starts at 60%.")

# ------------------------------------------------------------ 5. per-genre accuracy, shipped model vs DeepSeek
pg = rd("per_genre_accuracy_real_test.csv")
genres = [r["genre"] for r in pg if r["model"] == "iter5"]
n = {r["genre"]: int(r["n"]) for r in pg if r["model"] == "iter5"}
fig, ax = plt.subplots(figsize=(8, 5.2))
for j, (mk, ml, col) in enumerate((("iter5", "Linear, shipped", S1), ("deepseek", "DeepSeek V4.1 Flash", S2))):
    vals = {r["genre"]: float(r["acc"]) * 100 for r in pg if r["model"] == mk}
    for i, g in enumerate(genres):
        y = len(genres) - 1 - i + (0.2 if j == 0 else -0.2)
        rounded_barh(ax, y, vals[g], 0.36, col)
        ax.text(vals[g] + 0.8, y, f"{vals[g]:.0f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(genres))); ax.set_yticklabels([f"{g} (n={n[g]})" for g in genres][::-1], color=INK2)
ax.set_xlim(0, 108); ax.set_xlabel("Genre accuracy on real-test music items (%)")
ax.set_title("Where the small model loses: genre by genre", loc="left", pad=30)
style(ax, "x"); top_legend(ax, [plt.Rectangle((0, 0), 1, 1, color=c) for c in (S1, S2)], ["Linear, shipped", "DeepSeek V4.1 Flash"], 2)
save(fig, "05_per_genre_real_test.png", "Small n per genre: one item is 4-7 percentage points. No thresholds.")

# ------------------------------------------------------------ 6. confusion matrix (shipped model)
cf = [r for r in rd("genre_confusion_real_test.csv") if r["model"] == "iter5"]
G = genres + ["(not music)"]
M = [[0] * len(G) for _ in genres]
for r in cf:
    M[genres.index(r["true"])][G.index(r["pred"] if r["pred"] in genres else "(not music)")] += int(r["count"])
fig, ax = plt.subplots(figsize=(7.6, 6.4))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seq", SEQ)
rowfrac = [[c / max(1, sum(row)) for c in row] for row in M]
ax.imshow(rowfrac, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i, row in enumerate(M):
    for k, c in enumerate(row):
        if c:
            ax.text(k, i, str(c), ha="center", va="center", fontsize=8.5, color="#ffffff" if rowfrac[i][k] > 0.55 else INK)
ax.set_xticks(range(len(G))); ax.set_xticklabels(G, rotation=45, ha="right", color=INK2)
ax.set_yticks(range(len(genres))); ax.set_yticklabels(genres, color=INK2)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Genre confusion, shipped model, real test (counts)", loc="left")
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
save(fig, "06_genre_confusion.png", "Shade = share of the true genre's items. Biggest off-diagonal cells: anime read as jpop (5) or cpop (3), "
     "lofi read as rock (3), 'other' music not recognised as music (5).")

# ------------------------------------------------------------ 7. round 6 distillation variants
ab = [r for r in rd("ablations_recorded.csv") if r["artist_bonus"] == "3" and r["round"] in ("5", "6")]
labels = {"+ memes / idols / anime data (adopted)": "Round 5 (no teacher)", "+ 8982 teacher rows + 286 teacher corrections": "+9k teacher rows\n+ corrections",
          "+ 4030 balanced teacher rows + corrections": "+4k balanced rows\n+ corrections", "teacher rows only (no corrections)": "Teacher rows only",
          "teacher corrections only (no new rows)": "Corrections only"}
fig, ax = plt.subplots(figsize=(9, 4.4))
w = 0.26
for j, (key, nm, col) in enumerate((("real_type", "Content type", S1), ("real_genre", "Genre", S2), ("real_vclass", "Video ent / info", S3))):
    for i, r in enumerate(ab):
        v = float(r[key]) * 100; x = i + (j - 1) * (w + 0.02)
        ax.bar(x, v, width=w, color=col, linewidth=0)
        ax.text(x, v + 0.5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(len(ab))); ax.set_xticklabels([labels.get(r["experiment"], r["experiment"]) for r in ab], fontsize=8.5, color=INK2)
ax.set_ylim(70, 100); ax.set_ylabel("Accuracy, real dev + test (%)")
ax.set_title("Distilling DeepSeek into the small model did not help", loc="left", pad=30)
style(ax); top_legend(ax, [plt.Rectangle((0, 0), 1, 1, color=c) for c in (S1, S2, S3)], ["Content type", "Genre", "Video ent / info"])
save(fig, "07_distillation_variants.png", "Recorded from run output (models not kept). No thresholds. Video swings of 3-5 points are partly noise (about 100 videos). Axis starts at 70%.")

# ------------------------------------------------------------ 8. data composition
inv = rd("data_inventory.csv")
groups = {}
for r in inv:
    if r["file"].startswith(("val/", "real/", "focus/")) or r["file"].startswith("artists/"):
        continue
    groups[r["source"]] = groups.get(r["source"], 0) + int(r["rows"])
items = sorted(groups.items(), key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(8.5, 3.6))
for i, (k, v) in enumerate(items):
    rounded_barh(ax, i, v, 0.55, S1)
    ax.text(v + 80, i, f"{v:,}", va="center", fontsize=9, color=INK2)
ax.set_yticks(range(len(items))); ax.set_yticklabels([k for k, _ in items], color=INK2, fontsize=9)
ax.set_xlabel("Training rows"); ax.set_title("Where the training data came from", loc="left"); style(ax, "x")
art = sum(int(r["rows"]) for r in inv if r["file"].startswith("artists/"))
save(fig, "08_training_data_sources.png", f"Plus {art:,} artist-list rows (25,826 normalised names in the shipped table). "
     "Evaluation sets kept separate: 963 synthetic, 637 real (scraped), 360 focus.")

# ------------------------------------------------------------ 9. size vs accuracy
sz = [("Linear (shipped)", 2.1, 84.86, S1), ("Transformer int8", 107, 79.9, S2), ("Transformer fp32", 428, 84.07, S2)]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for name, mb, acc, col in sz:
    ax.scatter([mb], [acc], s=90, color=col, edgecolors=SURF, linewidths=2, zorder=3)
    ax.text(mb * 1.15, acc + 0.4, f"{name}\n{mb:g} MB, {acc:.1f}%", fontsize=9, color=INK2)
ax.axhline(95.3, color=S3, lw=2)
ax.text(1.35, 93.9, "DeepSeek V4.1 Flash on a server: 95.3%   (linear + DeepSeek for the ~32% it is unsure of: 95.6%)", fontsize=8.5, color=INK2)
ax.set_xscale("log"); ax.set_xlim(1.2, 2000); ax.set_ylim(76, 98)
ax.set_xlabel("On-device model size (MB, log scale)"); ax.set_ylabel("Real test, fully right (%)")
ax.set_title("Bigger on-device model, no better", loc="left"); style(ax)
save(fig, "09_size_vs_accuracy.png", "No thresholds. int8 transformer = dynamic quantisation without calibration, scored with the shipped artist-bonus settings "
     "(fp32 under those settings: 84.6%). Axis starts at 76%.")
