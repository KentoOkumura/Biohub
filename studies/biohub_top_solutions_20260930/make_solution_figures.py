"""Draw source-grounded teaching figures with synthetic images and coordinates.

Run from the repository root with:
    uv run --with matplotlib python studies/biohub_top_solutions_20260930/make_solution_figures.py

No competition images, trained weights, or measured predictions are used.
Architecture names and processing choices come from the archived Discussions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/biohub-solutions-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/images/biohub_top_solutions_20260930"
INK = "#17304b"
MUTED = "#596b7b"
BLUE = "#3679c9"
TEAL = "#087f79"
ORANGE = "#df8b29"
RED = "#c94a62"
PURPLE = "#8060b0"
GREY = "#a8b4c0"
LIGHT = "#e9eef3"
SOURCE = "https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/"
RECORDS: list[dict] = []


def setup():
    fonts = [
        os.environ.get("BIOHUB_FIGURE_FONT"),
        "/mnt/c/Windows/Fonts/meiryo.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in fonts:
        if candidate and Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            family = font_manager.FontProperties(fname=candidate).get_name()
            break
    else:
        raise RuntimeError("Set BIOHUB_FIGURE_FONT to a Japanese font file.")
    bold = Path("/mnt/c/Windows/Fonts/meiryob.ttc")
    if bold.exists():
        font_manager.fontManager.addfont(bold)
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GREY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "path",
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)


def page(rank, title, subtitle, *, six=False):
    fig = plt.figure(figsize=(18, 11) if six else (16, 10.2))
    fig.text(0.045, 0.963, f"{rank}位  |  {title}", fontsize=24, weight="bold", va="top")
    fig.text(0.045, 0.917, subtitle, fontsize=12.5, color=MUTED, va="top")
    grid = fig.add_gridspec(
        2, 3 if six else 2, left=0.055, right=0.96, bottom=0.1, top=0.835,
        wspace=0.23 if six else 0.2, hspace=0.38,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3 if six else 2)]
    for ax in axes:
        ax.set(xlim=(0, 10), ylim=(0, 7))
        ax.axis("off")
    fig.text(
        0.045, 0.047,
        "原文に基づく説明図。画像・位置・得点・attentionの強さは模式例で、実データ・学習結果ではありません。",
        fontsize=10.5, color=MUTED,
    )
    fig.text(0.045, 0.021, "青: 通常の接続  緑: 分裂・採用  橙: 回収・補正  赤: 競合・棄却  灰: 未選択", fontsize=10, color=MUTED)
    return fig, axes


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=14, weight="bold", pad=15)


def txt(ax, x, y, text, *, color=INK, size=11, ha="left", va="center", **kwargs):
    return ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va, **kwargs)


def arrow(ax, start, end, *, color=BLUE, lw=2, dashed=False, bend=0, alpha=1):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        color=color, alpha=alpha, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={bend}", shrinkA=10, shrinkB=10,
    )
    ax.add_patch(patch)


def dot(ax, xy, label="", *, color=BLUE, hollow=False, size=390):
    ax.scatter(
        *xy, s=size, c="white" if hollow else color, edgecolors=color if hollow else "white",
        linewidths=2, zorder=8,
    )
    if label:
        txt(ax, *xy, label, color=color if hollow else "white", size=10, ha="center", weight="bold", zorder=9)


def heat_image(centres, *, seed=0, size=96):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:1:complex(size), 0:1:complex(size)]
    field = np.maximum(0.01, rng.normal(0.07, 0.018, (size, size)))
    for centre in centres:
        x, y, sx, sy = centre[:4]
        amp = centre[4] if len(centre) == 5 else 0.8
        field += amp * np.exp(-((xx - x) ** 2 / sx**2 + (yy - y) ** 2 / sy**2) / 2)
    field = np.clip(field, 0, 1)
    return np.stack((field * 0.14, field * 0.92, field * 0.79), axis=-1)


def division_images():
    return [
        heat_image([(0.5, 0.5, 0.095, 0.13)], seed=1),
        heat_image([(0.5, 0.5, 0.16, 0.095)], seed=2),
        heat_image([(0.37, 0.5, 0.08, 0.09), (0.63, 0.5, 0.08, 0.09)], seed=3),
        heat_image([(0.25, 0.48, 0.08, 0.09), (0.75, 0.52, 0.08, 0.09)], seed=4),
    ]


def strip(ax, images, labels, *, x=0.2, y=4.6, width=9.4, height=1.55):
    gap = 0.16
    tile = (width - gap * (len(images) - 1)) / len(images)
    for i, (image, label) in enumerate(zip(images, labels, strict=True)):
        left = x + i * (tile + gap)
        ax.imshow(image, extent=(left, left + tile, y, y + height), aspect="auto", zorder=2)
        txt(ax, left + tile / 2, y + height + 0.32, label, ha="center", size=10)


def cube(ax, x, y, w=1.2, h=1.0, *, color=BLUE, label="", layers=3):
    for k in reversed(range(layers)):
        ax.add_patch(Rectangle((x + 0.13 * k, y + 0.13 * k), w, h, facecolor=color, edgecolor="white", alpha=0.55 + 0.13 * (layers - k) / layers))
    for frac in (0.25, 0.5, 0.75):
        ax.plot([x + w * frac, x + w * frac], [y, y + h], color="white", lw=0.4, alpha=0.4)
        ax.plot([x, x + w], [y + h * frac, y + h * frac], color="white", lw=0.4, alpha=0.4)
    if label:
        txt(ax, x + w / 2, y - 0.34, label, ha="center", size=9.5)


def tree(ax, x, y, *, scale=0.6, color=PURPLE):
    points = [(x, y), (x - scale, y - scale), (x + scale, y - scale)]
    for side in (-1, 1):
        child = (x + side * scale, y - scale)
        ax.plot([x, child[0]], [y, child[1]], color=color, lw=1.5)
        for offset in (-0.4, 0.4):
            leaf = (child[0] + offset * scale, child[1] - 0.8 * scale)
            ax.plot([child[0], leaf[0]], [child[1], leaf[1]], color=color, lw=1.2)
            points.append(leaf)
    ax.scatter(*np.asarray(points).T, color=color, s=35)


def matrix(ax, values, xlabels, ylabels, *, extent=(3.9, 8.5, 2.1, 5.8)):
    left, right, bottom, top = extent
    ax.imshow(values, extent=extent, cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="upper")
    row_height = (top - bottom) / len(ylabels)
    col_width = (right - left) / len(xlabels)
    for i, name in enumerate(ylabels):
        txt(ax, left - 0.25, top - (i + 0.5) * row_height, name, ha="right")
    for j, name in enumerate(xlabels):
        txt(ax, left + (j + 0.5) * col_width, top + 0.35, name, ha="center")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            txt(ax, left + (j + 0.5) * col_width, top - (i + 0.5) * row_height,
                f"{value:.2f}", ha="center", color="white" if value > 0.65 else INK, size=12)


def save(fig, filename, rank, topic, claims):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        bounds = artist.get_window_extent(renderer)
        if bounds.x0 < -2 or bounds.y0 < -2 or bounds.x1 > fig.bbox.width + 2 or bounds.y1 > fig.bbox.height + 2:
            outside.append(artist.get_text())
    if outside:
        raise RuntimeError(f"Text outside figure {filename}: {outside}")
    png = OUT / f"{filename}.png"
    svg = OUT / f"{filename}.svg"
    fig.savefig(png, dpi=150)
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip(" \t") for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    RECORDS.append({
        "rank": rank, "source": SOURCE + str(topic),
        "png": str(png.relative_to(ROOT)), "svg": str(svg.relative_to(ROOT)),
        "verified_source_claims": claims,
        "example_data": "synthetic teaching example; not measured predictions",
        "text_bounds": "passed",
    })
    print("saved", filename)


def third():
    fig, axes = page(3, "画像による分裂判定と、接続・分裂の同時選択", "画像から母の分裂得点を作り、母＋2娘の候補を比較する。座標は接続を固定して最後に補正。")
    ax = axes[0]
    panel(ax, "A", "分裂モデル: 5つの画像を共有encoderで読む")
    images = division_images()
    original_before = heat_image([(0.33, 0.51, 0.095, 0.13)], seed=5)
    original_after = heat_image([(0.53, 0.5, 0.08, 0.09), (0.79, 0.5, 0.08, 0.09)], seed=6)
    strip(ax, [original_before, images[0], images[1], images[2], original_after],
          ["元 t−1", "整列 t−1", "t", "整列 t+1", "元 t+1"], y=4.7, height=1.5)
    for i in range(5):
        cube(ax, 0.35 + i * 1.9, 3.12, 1.18, 0.64, color=PURPLE)
    txt(ax, 5, 2.56, "共有EfficientNetV2-Sの各段の特徴を結合", ha="center", size=11)
    cube(ax, 3.0, 0.97, 2.0, 0.91, color=BLUE, label="3D decoder / GroupNorm")
    arrow(ax, (5.1, 1.4), (6.8, 1.4))
    ax.barh([1.4], [0.92 * 2.3], left=6.9, height=0.4, color=TEAL)
    txt(ax, 6.9, 2.0, "母の分裂得点", size=10.5)
    txt(ax, 8.0, 0.85, "0.92（例）", color=TEAL, ha="center")

    ax = axes[1]
    panel(ax, "B", "対応と分裂候補の得点は、別々に作る")
    cube(ax, 0.35, 4.85, 1.4, 1.0, color=BLUE)
    txt(ax, 0.1, 4.1, "対応用2.5D U-Net\n画像特徴64ch＋位置", size=10)
    txt(ax, 0.1, 2.7, "Self / Cross-Attention\n＋対応なしの出力", size=10)
    matrix(ax, np.array([[.82, .79, .10], [.11, .73, .88]]), ["C", "D", "E"], ["母A", "母B"])
    txt(ax, 6.0, 1.54, "通常辺: ロジスティック回帰", ha="center", size=10.5, color=BLUE)
    tree(ax, 1.4, 1.3, scale=.4)
    tree(ax, 2.7, 1.3, scale=.4, color=TEAL)
    txt(ax, 4.0, .75, "母A＋娘C,D → 分裂確率\nLightGBM / XGBoost", size=10.5)
    txt(ax, .2, .1, "評価対象確率 a の算出器は本文未公開", color=MUTED, size=9.5)

    ax = axes[2]
    panel(ax, "C", "通常辺を確定せず、2娘の分裂も選択肢にする")
    points = {"a0": (.65, 4.7), "b0": (.65, 1.55), "A": (3.1, 4.7), "B": (3.1, 1.55),
              "C": (6.3, 5.6), "D": (6.3, 3.4), "E": (6.3, 1.0),
              "c2": (9.1, 5.85), "d2": (9.1, 3.25), "e2": (9.1, .9)}
    arrow(ax, points["B"], points["D"], color=RED, dashed=True)
    for u, v in [("a0", "A"), ("b0", "B"), ("B", "E"), ("C", "c2"), ("D", "d2"), ("E", "e2")]:
        arrow(ax, points[u], points[v], color=BLUE)
    for u, v in [("A", "C"), ("A", "D")]:
        arrow(ax, points[u], points[v], color=TEAL, lw=3)
    for key, position in points.items():
        dot(ax, position, key if len(key) == 1 else "", color=TEAL if key in {"A", "C", "D"} else BLUE)
    txt(ax, 4.7, 2.4, "B→Dは競合", color=RED, size=10)
    for x, label in [(.65, "t−1"), (3.1, "t"), (6.3, "t+1"), (9.1, "t+2")]:
        txt(ax, x, 6.65, label, ha="center")
    txt(ax, .2, .05, "娘の親は1つ / 母は通常継続か分裂 / 両娘をセットで採用", size=10)

    ax = axes[3]
    panel(ax, "D", "通常接続から運動を推定し、座標の揺れを抑える")
    rng = np.random.default_rng(20)
    bases = np.array([[1.4, 1.5], [3.5, 2.5], [5.2, 1.9], [6.7, 4.0], [8, 2.1]])
    displacement = np.array([.8, .35])
    originals = bases + displacement + rng.normal(0, .35, bases.shape)
    corrected = .25 * originals + .75 * (bases + displacement)
    for start, raw, new in zip(bases, originals, corrected, strict=True):
        arrow(ax, start, start + displacement, color=GREY, lw=1.2)
        ax.scatter(*raw, s=130, facecolors="none", edgecolors=ORANGE, linewidths=2)
        ax.scatter(*new, s=90, c=TEAL, zorder=5)
        arrow(ax, raw, new, color=ORANGE, lw=1.6)
    for xy in [(3.7, 5.25), (5.2, 5.6), (5.25, 4.9)]:
        ax.scatter(*xy, marker="*", s=230, c=PURPLE, zorder=5)
    txt(ax, 5.8, 5.4, "分裂周辺は固定", color=PURPLE, size=10.5)
    txt(ax, .3, 6.35, "灰矢印: 推定アフィン運動  橙○: 元位置  緑●: 補正後", size=10)
    txt(ax, .3, .32, "接続と点数は固定 / 元の検出から動かしすぎない", size=11)
    save(fig, "03_joint_division", 3, 744484, ["5-volume shared division encoder", "matching attention", "joint selection", "affine coordinate refinement"])


def tracker_arch(ax, *, six=False):
    left = [(1.4, 4.9), (1.4, 3.35), (1.4, 1.8)]
    right = [(5.3, 5.45), (5.3, 4.1), (5.3, 2.6), (5.3, 1.1)]
    txt(ax, 1.4, 6.4, "t", ha="center")
    txt(ax, 5.3, 6.4, "t+1", ha="center")
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if not six or abs(i - j) <= 1:
                arrow(ax, a, b, color=BLUE, lw=1 + (i == j) * 1.6, alpha=.55)
    if not six:
        for side in (left, right):
            for i in range(len(side) - 1):
                arrow(ax, side[i], side[i + 1], color=PURPLE, bend=.55, lw=1.3)
    for i, position in enumerate(left):
        dot(ax, position, chr(65 + i), color=BLUE, size=300)
    for i, position in enumerate(right):
        dot(ax, position, str(i + 1), color=TEAL, size=300)
    txt(ax, 3.2, .25, "画像特徴＋位置 → 細胞間の照合", ha="center", size=10)
    labels = ["接続", "新規細胞", "母の分裂", "娘pair" if six else "見た目\n16次元"]
    colours = [BLUE, ORANGE, TEAL, PURPLE]
    for k, (label, colour) in enumerate(zip(labels, colours, strict=True)):
        txt(ax, 6.4, 5.3 - k * 1.3, label, size=10)
        for j in range(5):
            ax.add_patch(Rectangle((8.1 + j * .23, 5.03 - k * 1.3), .18, .48, color=colour, alpha=.3 + .12 * j))


def stage_graph(ax, stage):
    nodes = {}
    for t, y in enumerate([5.5, 5.45, 5.30, 5.2, 5.05, 4.95]):
        nodes[f"a{t}"] = (t, y, .72 if t in (2, 3) else .97, "a")
    for t, y in [(0, 2), (1, 2.2)]:
        nodes[f"b{t}"] = (t, y, .97, "b")
    for t in range(2, 6):
        nodes[f"c{t}"] = (t, 2.2 + .26 * (t - 1), .96, "c")
        nodes[f"d{t}"] = (t, 2.2 - .26 * (t - 1), .96, "d")
    nodes["x3"] = (3, 4.13, .67, "x")
    nodes["x4"] = (4, 4.04, .95, "x")
    nodes["x5"] = (5, 3.98, .95, "x")
    nodes["g0"] = (0, 3.75, .96, "g")
    nodes["g1"] = (1, 3.7, .96, "g")
    nodes["g5"] = (5, 3.6, .96, "g")
    if stage == 4:
        for t in range(2, 5):
            nodes[f"g{t}"] = (t, 3.7 - .025 * (t - 1), 1, "fill")
    threshold = .9 if stage == 1 else .6
    active = {key for key, (_, _, p, _) in nodes.items() if p >= threshold}
    positions = {key: (.45 + t * 1.76, y) for key, (t, y, _, _) in nodes.items()}
    edges = [("b0", "b1"), ("b1", "c2"), ("b1", "d2"), ("g0", "g1")]
    for family in ("a", "c", "d", "x", "g"):
        edges += [(f"{family}{t}", f"{family}{t+1}") for t in range(5)
                  if f"{family}{t}" in nodes and f"{family}{t+1}" in nodes]
    for u, v in edges:
        if u in active and v in active:
            colour = TEAL if u == "b1" else ORANGE if nodes[v][3] == "fill" else BLUE
            arrow(ax, positions[u], positions[v], color=colour, lw=1.7)
    if stage == 2:
        arrow(ax, positions["a2"], positions["x3"], color=RED, lw=2.4)
        txt(ax, 4.4, 6.1, "新しい偽分裂", color=RED, size=10)
    elif stage >= 3:
        arrow(ax, positions["a2"], positions["x3"], color=GREY, dashed=True, lw=1.2)
        txt(ax, 4.4, 6.1, "この分裂は取り消す", color=RED, size=9.5)
    for key, position in positions.items():
        _, _, p, family = nodes[key]
        if key not in active:
            ax.scatter(*position, s=90, facecolors="none", edgecolors=GREY, alpha=.45)
        else:
            dot(ax, position, color=ORANGE if p < .9 or family == "fill" else BLUE,
                hollow=family == "fill", size=100)
    for t in range(6):
        txt(ax, .45 + t * 1.76, .38, str(t), size=9, ha="center")
    txt(ax, 5, -.1, "時刻（模式例）", size=9, ha="center")
    if stage == 1:
        txt(ax, .2, 6.65, "検出信頼度 ≥ 0.9", size=10)
    elif stage == 2:
        txt(ax, .2, 6.65, "検出信頼度 ≥ 0.6", size=10)
    elif stage == 3:
        txt(ax, .2, 6.65, "低信頼の点は残し、通常辺を再選択", size=9.5)
    else:
        txt(ax, .2, 6.65, "2〜5フレーム離れた末端を接続", size=9.5)
        txt(ax, 2.35, 4.65, "欠落位置を補う", color=ORANGE, size=9.5)


def fifth():
    fig, axes = page(5, "Transformerの複数出力と、4段階のILP", "点を回収する判断と分裂を増やす判断を分ける。最後に軌跡の位置を平滑化する。", six=True)
    ax = axes[0]
    panel(ax, "A", "Temporal 3D U-Net＋Transformer")
    tracker_arch(ax)
    txt(ax, 6.4, .55, "専用出力で\n開始・分裂費用を作る", size=9.5)
    for ax, stage, letter in zip(axes[1:5], range(1, 5), "BCDE", strict=True):
        panel(ax, letter, f"ILP{stage}" + (": 点・分裂を固定" if stage == 3 else ""))
        stage_graph(ax, stage)
    ax = axes[5]
    panel(ax, "F", "前後4フレームで位置の揺れを抑える")
    t = np.arange(-4, 5)
    raw = 2.9 + .3 * t + np.array([.15, -.3, .2, -.1, .95, -.2, .05, -.15, .1])
    fit = np.polyval(np.polyfit(t, raw, 1), t)
    result = .2 * raw + .8 * fit
    ax.plot(t + 5, raw, "o--", color=ORANGE, lw=1.2, label="元の位置")
    ax.plot(t + 5, fit, "-", color=GREY, lw=1.8, label="直線当てはめ")
    ax.scatter(t + 5, result, c=TEAL, s=38, label="80%当てはめ＋20%元位置", zorder=5)
    arrow(ax, (5, raw[4]), (5, result[4]), color=ORANGE)
    for x, label in [(1, "t−4"), (5, "t"), (9, "t+4")]:
        txt(ax, x, .35, label, ha="center", size=9)
    txt(ax, .15, 6.65, "同じ細胞の最大9時点を使用", size=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0, .84), frameon=False, fontsize=8.5)
    txt(ax, .2, .85, "模式例: 縦軸は1軸の位置", color=MUTED, size=9)
    save(fig, "05_multistage_ilp", 5, 744549, ["self/cross attention and separate heads", "four ILP stages", "no new divisions after ILP2", "nine-timepoint line fit"])


def sixth():
    fig, axes = page(6, "速度・不確かさを使う、疎なgraph attention", "3フレームの画像特徴を学び、候補接続の運動誤差も入力する。検出と追跡を共同学習。")
    ax = axes[0]
    panel(ax, "A", "3フレームの画像特徴を、動きに合わせて取得")
    strip(ax, division_images()[:3], ["t−1", "t", "t+1"], y=4.35, height=1.75, width=8.8)
    for i in range(3):
        cube(ax, .7 + i * 2.9, 2.65, 1.35, .9, color=BLUE)
    txt(ax, 5, 2.0, "3D U-Net型encoder / 各フレームの特徴を再利用", ha="center", size=11)
    ax.scatter([1.1, 2.0, 3.0, 5.8, 6.7, 7.6], [1.0, 1.3, .8, 1.2, .75, 1.4], c=ORANGE, s=65)
    for x, y in [(1.1, 1), (2, 1.3), (3, .8)]:
        arrow(ax, (4.4, 1.05), (x, y), color=TEAL, lw=1.1)
    for x, y in [(5.8, 1.2), (6.7, .75), (7.6, 1.4)]:
        arrow(ax, (4.4, 1.05), (x, y), color=TEAL, lw=1.1)
    txt(ax, 5, .15, "予測変位の場所から近隣フレームの特徴を採る", ha="center", size=10.5)

    ax = axes[1]
    panel(ax, "B", "近さだけでなく、予測した動きとの差を見る")
    dot(ax, (1.8, 3), "A", color=BLUE)
    predicted = (6.4, 3.0)
    ax.add_patch(Ellipse(predicted, 2.5, 1.3, facecolor=TEAL, edgecolor=TEAL, alpha=.12, lw=2))
    ax.add_patch(Ellipse(predicted, 2.5, 1.3, fill=False, edgecolor=TEAL, lw=1.5))
    ax.scatter(*predicted, marker="+", s=200, color=TEAL, linewidths=2)
    arrow(ax, (1.8, 3), predicted, color=TEAL, lw=3)
    dot(ax, (6.7, 3.3), "C", color=TEAL)
    dot(ax, (3.5, 4.55), "D", color=RED)
    arrow(ax, (1.8, 3), (3.5, 4.55), color=RED, dashed=True)
    txt(ax, 5.1, 2.0, "楕円: 各軸の不確かさ", color=TEAL, size=10.5)
    txt(ax, .3, 6.3, "Dは近いが、予測した移動と合わない", size=11)
    txt(ax, 4.5, 5.4, "Cは予測位置に近い", color=TEAL, size=10.5)
    txt(ax, .3, .65, "入力: 変位＋予測速度からの残差\n残差を各軸の不確かさで正規化", size=11)

    ax = axes[2]
    panel(ax, "C", "候補graph上だけで双方向attention")
    tracker_arch(ax, six=True)
    txt(ax, .1, 6.85, "3〜4層 / 密な全組合せではなく近傍候補", size=10)
    txt(ax, 6.35, .4, "親候補＋新規クラス\nをsoftmaxで出力", size=10)

    ax = axes[3]
    panel(ax, "D", "途中の仮の対応で、運動予測も更新する")
    for y, label, centre in [(4.8, "更新前", 4.55), (2.6, "更新後", 6.0)]:
        dot(ax, (1.1, y), "A", color=BLUE)
        dot(ax, (5.9, y + .55), "C", color=TEAL)
        dot(ax, (7.25, y - .45), "D", color=ORANGE)
        ax.add_patch(Ellipse((centre, y), 2.0 if label == "更新前" else 1.3,
                             1.1 if label == "更新前" else .9, color=TEAL, alpha=.12))
        arrow(ax, (1.1, y), (centre, y), color=TEAL, lw=2.3)
        txt(ax, .1, y + 1.05, label, size=10.5)
    txt(ax, 8.05, 4.8, "仮の対応\nCを強く支持", size=10)
    txt(ax, .2, .6, "一方の構造では、attention途中で速度・不確かさを更新\n更新方向・強さはこの図の模式例", size=10.5)
    save(fig, "06_motion_attention", 6, 744582, ["three-frame native input", "velocity and uncertainty", "bidirectional sparse graph attention", "in-graph motion refinement"])


def repair_tracks(ax, *, after=False):
    xs = [.8, 2.7, 4.6, 6.5, 8.4]
    top = [(x, 5.35 - .12 * i) for i, x in enumerate(xs)]
    bottom = [(x, 2.4 + .08 * i) for i, x in enumerate(xs)]
    for family in (top, bottom):
        for i in range(4):
            if family is top and i in (1, 2) and not after:
                continue
            arrow(ax, family[i], family[i + 1], color=ORANGE if after and family is top and i in (1, 2) else BLUE)
        for i, position in enumerate(family):
            if family is top and i == 2 and not after:
                ax.scatter(*position, s=110, facecolors="none", edgecolors=GREY)
            else:
                dot(ax, position, color=ORANGE if after and family is top and i == 2 else BLUE, size=150)
    if not after:
        arrow(ax, bottom[1], top[2], color=RED, lw=2)
    else:
        arrow(ax, bottom[1], top[2], color=GREY, dashed=True, lw=1)
    for i, x in enumerate(xs):
        txt(ax, x, 1.0, f"t+{i}", ha="center", size=9)


def twelfth():
    fig, axes = page(12, "公開モデルの後で、画像判定・修復・中心補正", "1娘＋親なし候補を画像で確かめ、接続と座標を別々に修復する。公開検出・接続モデルは固定。")
    ax = axes[0]
    panel(ax, "A", "幾何だけでは判別できない第2娘候補")
    strip(ax, division_images()[:3], ["t−1", "母の時刻 t", "t+1"], y=4.55, height=1.75)
    dot(ax, (2.0, 2.5), "母", color=BLUE)
    dot(ax, (6.3, 3.3), "娘1", color=TEAL)
    dot(ax, (6.3, 1.4), "娘2", color=ORANGE, hollow=True)
    dot(ax, (8.6, 2.0), "別", color=RED, hollow=True)
    arrow(ax, (2, 2.5), (6.3, 3.3), color=BLUE)
    arrow(ax, (2, 2.5), (6.3, 1.4), color=ORANGE, dashed=True)
    arrow(ax, (2, 2.5), (8.6, 2), color=GREY, dashed=True)
    txt(ax, .3, .35, "軽い表形式モデルで候補を絞り、上位だけ画像で読む", size=11)

    ax = axes[1]
    panel(ax, "B", "2解像度の3D CNNと、別の画像特徴")
    image = division_images()[2]
    ax.imshow(image, extent=(.4, 2.2, 4.3, 6.3), aspect="auto")
    ax.imshow(image[20:76, 20:76], extent=(.8, 2.2, 2.3, 3.8), aspect="auto")
    txt(ax, .3, 1.9, "時空間crop\n2つのスケール", size=10)
    cube(ax, 3.1, 4.85, 1.2, .8, color=BLUE)
    cube(ax, 4.65, 4.95, .85, .62, color=PURPLE)
    txt(ax, 3.1, 4.1, "分裂CNN: 2.8Mパラメータ\nZebraHubからfine-tuning", size=10.5)
    cube(ax, 3.25, 2.6, 1.25, .8, color=TEAL)
    txt(ax, 3.1, 2.05, "DINOv2 ViT-S/14（固定）\nXY / XZ / YZ断面の特徴", size=10)
    tree(ax, 7.8, 4.75, scale=.65)
    txt(ax, 6.7, 2.35, "画像変形・明るさ\n＋候補の文脈\nLightGBMで統合", size=10.5)
    txt(ax, .2, .5, "CNNの各層の詳細は本文にないため省略", size=10, color=MUTED)

    ax = axes[2]
    panel(ax, "C", "切れた軌跡と誤分裂を、画像特徴でも修復")
    sub1 = ax.inset_axes([.01, .52, .98, .43])
    sub2 = ax.inset_axes([.01, .03, .98, .43])
    for sub in (sub1, sub2):
        sub.set(xlim=(0, 10), ylim=(.8, 6)); sub.axis("off")
    repair_tracks(sub1)
    repair_tracks(sub2, after=True)
    txt(ax, .15, 6.9, "修復前: 欠落と誤った分裂腕", size=10, color=RED)
    txt(ax, .15, 3.35, "修復後: 欠落を補い、通常の接続を取り戻す", size=10, color=TEAL)

    ax = axes[3]
    panel(ax, "D", "最終座標を動かす。接続は変えない")
    image = heat_image([(.43, .56, .12, .15), (.79, .39, .085, .12)], seed=31)
    ax.imshow(image, extent=(.5, 8.2, 1.2, 6.35), aspect="auto")
    original = (4.5, 3.1)
    corrected = (3.83, 4.0)
    ax.scatter(*original, marker="o", s=230, facecolors="none", edgecolors=ORANGE, linewidths=2)
    ax.scatter(*corrected, marker="+", s=220, c="white", linewidths=2)
    arrow(ax, original, corrected, color=ORANGE, lw=2.5)
    neighbour = (6.6, 3.2)
    ax.add_patch(Ellipse(neighbour, 1.4, 1.0, fill=False, edgecolor=RED, lw=1.5, linestyle="--"))
    arrow(ax, original, neighbour, color=RED, dashed=True)
    txt(ax, .2, 6.75, "3D CNN＋軸別LightGBMで中心のずれを学習", size=10.5)
    txt(ax, 7.4, 4.75, "近隣点へ\n寄りすぎる\n移動を制限", color=RED, size=9.5)
    txt(ax, .2, .35, "近隣衝突・時間的な不整合を防ぐ / graphの接続は固定", size=10.5)
    save(fig, "12_image_repair", 12, 744501, ["single child plus orphan candidates", "two-scale division CNN", "DINOv2 features and image witnesses", "guarded coordinate refinement"])


def voxel_flow(ax, centres, *, to_parent=None):
    image = heat_image([(x / 10, y / 7, .11, .15) for x, y in centres], seed=41)
    ax.imshow(image, extent=(0, 10, 0, 7), aspect="auto", alpha=.88)
    for cx, cy in centres:
        for dx in (-.7, 0, .7):
            for dy in (-.7, 0, .7):
                if dx == dy == 0:
                    continue
                if to_parent is None:
                    ax.quiver(cx + dx, cy + dy, -.7 * dx, -.7 * dy, angles="xy", scale_units="xy", scale=1,
                              color="white", width=.008, headwidth=3.5)
                else:
                    arrow(ax, (cx + dx, cy + dy), to_parent, color=TEAL, lw=.9, alpha=.45)
        ax.scatter(cx, cy, marker="+", s=150, color="white", linewidths=1.8)


def fourteenth():
    fig, axes = page(14, "ボクセルの変位で細胞を分け、2娘から母を探す", "FlowSegとdivflowは作者の名称。分類だけに頼らず、細胞内の多数のボクセルで変位を学ぶ。")
    ax = axes[0]
    panel(ax, "A", "FlowSeg: 各ボクセルから自分の中心へ")
    voxel_flow(ax, [(3.8, 3.55), (6.0, 3.65)])
    txt(ax, .25, 6.5, "接している核も、矢印の収束先で分かれる", color="white", size=11)
    txt(ax, .25, .6, "出力: foreground確率＋中心への変位\n3 / 5フレームと差分画像を使う2構成", color="white", size=10.5)

    ax = axes[1]
    panel(ax, "B", "divflow: 2娘が同じ母の位置へ投票する")
    ax.add_patch(Ellipse((6.7, 4.9), 1.5, 1.1, color=TEAL, alpha=.2))
    ax.add_patch(Ellipse((6.7, 1.7), 1.5, 1.1, color=TEAL, alpha=.2))
    for cx, cy in [(6.7, 4.9), (6.7, 1.7)]:
        for dx, dy in [(-.4, -.25), (-.4, .25), (.4, -.25), (.4, .25)]:
            arrow(ax, (cx + dx, cy + dy), (2.9 + .12 * dy, 3.3 + .12 * dx), color=TEAL, lw=1.1, alpha=.7)
    ax.scatter([2.72, 2.88, 3.02, 3.15], [3.2, 3.42, 3.28, 3.35], s=55, c=PURPLE)
    ax.add_patch(Ellipse((2.95, 3.3), 1.15, 1.0, facecolor=PURPLE, edgecolor=PURPLE, alpha=.12))
    txt(ax, 6.7, 6.1, "観測された娘（後の時刻）", ha="center", size=10.5)
    txt(ax, 2.9, 4.55, "予測された母の位置", ha="center", color=PURPLE, size=10.5)
    txt(ax, .25, .35, "入力: 時刻の異なる2画像 / 教師: 娘のmask内の母への変位", size=10.5)

    ax = axes[2]
    panel(ax, "C", "投票先の一致と、娘の距離を両方調べる")
    cases = [(5.6, 6.5, "分裂候補", TEAL), (3.25, 1.5, "1核の過分割", RED), (.9, 19, "離れすぎた2核", RED)]
    for y, distance, label, colour in cases:
        start = 1.0
        end = start + distance * .36
        dot(ax, (start, y), color=colour, size=180)
        dot(ax, (end, y), color=colour, size=180)
        ax.plot([start, end], [y, y], color=colour, lw=1.1)
        txt(ax, 8.5, y + (.5 if distance > 16 else 0), label, ha="center", color=colour, size=10)
        txt(ax, (start + end) / 2, y + .45, f"{distance:g} µm", ha="center", size=10)
    txt(ax, .15, 6.75, "母への投票先は4 µm以内 / 娘の観測位置は3〜16 µm", size=10)

    ax = axes[3]
    panel(ax, "D", "2つのgraphから、欠けた点・分裂を補う")
    for xoff, label in [(0, "片方のgraph"), (5.2, "補完後")]:
        txt(ax, xoff + 2.0, 6.4, label, ha="center", size=11)
        p = {"A": (xoff + .5, 4.2), "C": (xoff + 3.6, 5.1), "D": (xoff + 3.6, 3.05),
             "B": (xoff + .5, 1.1), "E": (xoff + 3.6, 1.1)}
        for u, v in [("A", "C"), ("B", "E")]:
            arrow(ax, p[u], p[v], color=BLUE)
        arrow(ax, p["A"], p["D"], color=ORANGE if xoff else GREY, dashed=not bool(xoff))
        for key, position in p.items():
            dot(ax, position, key, color=ORANGE if key == "D" else BLUE, hollow=key == "D" and not xoff, size=260)
    txt(ax, .2, .12, "他方が見つけた点・分裂を移植 / 最後に一定座標ずれを補正", size=10.5)
    save(fig, "14_dense_displacement", 14, 744486, ["flow integration separates cells", "daughter-to-parent dense regression", "parent convergence and daughter separation gates", "graph grafting"])


def eighteenth():
    fig, axes = page(18, "小型モデルで分裂を採点し、修復後のgraphで見直す", "画像の分裂判定と時間方向の分類を使い、通常辺を外す損失も考える。点は残して誤った腕を戻す。")
    ax = axes[0]
    panel(ax, "A", "4時点の画像を小型3D CNNへ入力")
    images = division_images()
    strip(ax, images, ["t−1", "t", "t+1", "t+3"], y=4.6, height=1.45)
    for i in range(3):
        cube(ax, .5 + 1.6 * i, 2.7, 1.0, .7, color=BLUE if i % 2 == 0 else PURPLE)
    txt(ax, .2, 2.0, "3 conv層＋GroupNorm\n60,449パラメータ / 入力は8³のcube", size=10.5)
    strip(ax, [images[3], images[2], images[1], images[0]], ["逆再生", "", "", ""], x=5.7, y=2.5, width=3.8, height=.9)
    txt(ax, 5.7, 1.87, "CatBoost: 1→2か、2→1か\nLinajeaのみで学習", size=10.5)
    txt(ax, .2, .4, "CNNは合成→Linajea→コンペで学習\n確率をCatBoost・TabPFNと統合して母＋2娘を採点", size=10.5)

    ax = axes[1]
    panel(ax, "B", "分裂を採用するため、競合する通常辺を外す")
    points = {"A": (1.3, 4.8), "B": (1.3, 1.8), "C": (6.7, 5.6), "D": (6.7, 3.6), "E": (6.7, 1.0)}
    arrow(ax, points["B"], points["D"], color=RED, dashed=True, lw=2.2)
    for u, v in [("A", "C"), ("A", "D")]:
        arrow(ax, points[u], points[v], color=TEAL, lw=3)
    arrow(ax, points["B"], points["E"], color=BLUE)
    for key, position in points.items():
        dot(ax, position, key, color=TEAL if key in {"A", "C", "D"} else BLUE)
    txt(ax, 3.4, 2.35, "外す辺の得点も比較", color=RED, size=11)
    txt(ax, .2, .25, "普通の辺: 341特徴LightGBM\n分裂: 303特徴CatBoost＋画像・時間方向＋TabPFN", size=10.5)

    ax = axes[2]
    panel(ax, "C", "娘の軌跡をつなぎ、その結果を判定へ渡す")
    mother = (1.1, 3.7)
    upper = [(3.2, 5.1), (5.1, 5.4), (7.0, 5.6), (8.9, 5.7)]
    lower = [(3.2, 2.3), (5.1, 1.85), (7.0, 1.5), (8.9, 1.2)]
    arrow(ax, mother, upper[0], color=TEAL)
    arrow(ax, mother, lower[0], color=ORANGE, lw=3)
    for family in (upper, lower):
        for a, b in zip(family[:-1], family[1:], strict=True):
            arrow(ax, a, b, color=TEAL)
        for position in family:
            dot(ax, position, color=TEAL, size=200)
    dot(ax, mother, "母", color=BLUE)
    txt(ax, 1.75, 1.25, "未使用の候補辺で接続", color=ORANGE, size=10.5)
    txt(ax, .2, 6.65, "長さ・信頼度・娘間距離の増加も確認", size=11)
    txt(ax, .2, .35, "追加された腕も含むgraphから、再判定用の特徴を作る", size=10.5)

    ax = axes[3]
    panel(ax, "D", "完成した軌跡の運動で、誤った分裂腕を戻す")
    horizons = np.array([0, 1, 3, 7, 15])
    x = .5 + horizons / 15 * 8.5
    good = np.array([3.2, 3.55, 4.0, 4.65, 5.1])
    bad = np.array([3.2, 2.95, 2.6, 5.8, 5.1])
    ax.plot(x, good, "o-", color=TEAL, lw=2)
    ax.plot(x, bad, "o--", color=RED, lw=2)
    for xi, horizon in zip(x, horizons, strict=True):
        txt(ax, xi, 1.75, str(horizon), size=9, ha="center")
    txt(ax, 5, 1.35, "母の時刻からのフレーム数", ha="center", size=10)
    txt(ax, .15, 6.6, "周囲の組織運動を引いた相対運動（模式例）", size=10)
    txt(ax, .15, .55, "42 graph特徴＋51運動特徴 → TabPFN＋幾何判定\n最終確率 < 0.3なら弱い腕を切り、通常辺へ戻す。点は残す。", size=10.5)
    save(fig, "18_completed_graph", 18, 744531, ["four-timepoint DivisionCubeNet", "external-only time-direction CatBoost", "ordinary-edge competition", "post-completion graph and motion critic"])


def contact_sheet():
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.96, hspace=.07, wspace=.025)
    fig.suptitle("Biohub 上位6解法の説明図", fontsize=23, weight="bold")
    for ax, record in zip(axes.flat, RECORDS, strict=True):
        ax.imshow(plt.imread(ROOT / record["png"]))
        ax.axis("off")
    fig.savefig(OUT / "00_contact_sheet.png", dpi=130)
    plt.close(fig)


def main():
    drawing_functions = {3: third, 5: fifth, 6: sixth, 12: twelfth, 14: fourteenth, 18: eighteenth}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, nargs="+", choices=tuple(drawing_functions), help="Regenerate only these ranks, preserving the other figures.")
    args = parser.parse_args()
    setup()
    selected = set(args.ranks or drawing_functions)
    manifest_path = OUT / "manifest.json"
    if args.ranks and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        RECORDS.extend(record for record in existing["figures"] if record["rank"] not in selected)
    if args.ranks and not manifest_path.exists() and selected != set(drawing_functions):
        parser.error("Run once without --ranks to create all figures before selectively regenerating.")
    for rank in sorted(selected):
        drawing_functions[rank]()
    RECORDS.sort(key=lambda record: record["rank"])
    contact_sheet()
    manifest = {
        "purpose": "standalone teaching visualizations for the archived top-solution report",
        "no_real_data": True,
        "no_training_or_inference": True,
        "unverified_layers": "never filled in; feature stacks are schematic",
        "figures": RECORDS,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
