"""Draw original teaching figures; all images and numbers are synthetic.

Run from the repository root with uv run --with matplotlib (see the report).
This script does not read competition images or train a model.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/biohub-techniques-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/images/biohub_tracking_techniques"
BLUE, TEAL, ORANGE, RED = "#2563b5", "#087f78", "#be6b15", "#bc4252"
INK, MUTED, LIGHT = "#182d43", "#536579", "#e4eaf0"


def setup():
    configured = os.environ.get("BIOHUB_FIGURE_FONT")
    candidates = [
        configured,
        "/mnt/c/Windows/Fonts/meiryo.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            name = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = name
            break
    else:
        raise RuntimeError("Set BIOHUB_FIGURE_FONT to a Japanese font file.")
    plt.rcParams.update(
        {
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": LIGHT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)


def canvas(title, subtitle, height=6):
    fig = plt.figure(figsize=(14, height))
    fig.text(0.04, 0.95, title, size=23, weight="bold", va="top")
    fig.text(0.04, 0.87, subtitle, size=12, color=MUTED, va="top")
    ax = fig.add_axes((0.025, 0.04, 0.95, 0.75))
    ax.set(xlim=(0, 100), ylim=(0, 50))
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=155)
    plt.close(fig)


def box(ax, x, y, w, h, label, color=BLUE, size=14, fill=True):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.3,rounding_size=1.1",
        facecolor=color if fill else "white",
        edgecolor=color,
        lw=1.7,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        size=size,
        color="white" if fill else color,
        linespacing=1.65,
    )


def arrow(ax, start, end, color=MUTED, dashed=False, lw=2):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=17,
            color=color,
            lw=lw,
            linestyle="--" if dashed else "-",
        )
    )


def node(ax, x, y, label, color=TEAL):
    ax.scatter([x], [y], s=790, c=color, edgecolors="white", linewidths=2, zorder=4)
    ax.text(x, y, label, color="white", ha="center", va="center", weight="bold", zorder=5)


def overview():
    fig, ax = canvas(
        "6つの技術は、担当する処理が違う",
        "矢印は情報の流れ。各欄の手法は選択肢で、すべてを順に使う必要はない。",
    )
    cols = [2, 27, 52, 77]
    labels = ["① 見つける", "② 特徴を数値にする", "③ 対応の費用を作る", "④ 接続を選ぶ"]
    for x, label in zip(cols, labels, strict=True):
        ax.text(x + 10, 47, label, ha="center", size=15, weight="bold")
    box(ax, 2, 25, 20, 16, "画像 → 細胞中心\nDoG / 学習済み検出器", BLUE, 12)
    box(ax, 27, 25, 20, 16, "局所画像 → 特徴\nHOG / 学習した特徴", TEAL, 12)
    box(ax, 52, 25, 20, 16, "位置・動き・特徴\n→ 接続候補の費用", ORANGE, 12)
    box(ax, 77, 25, 20, 16, "1対1：ハンガリアン法\n分裂などの制約：ILP", BLUE, 11)
    for x in [22, 47, 72]:
        arrow(ax, (x + 1, 33), (x + 4, 33))
    box(ax, 27, 5, 20, 12, "対照学習\n似てほしい特徴を近づける", TEAL, 11, False)
    arrow(ax, (37, 18), (37, 24), TEAL)
    box(
        ax,
        52,
        5,
        45,
        12,
        "SORT：検出結果を受け取り、毎フレーム\n動きを予測 → 1対1に割り当て → 軌跡を更新",
        ORANGE,
        12,
        False,
    )
    ax.text(12, 11, "検出点だけで\n対応費用を作る方法もある", ha="center", size=11, color=MUTED)
    save(fig, "00_overview")


def ilp():
    fig, ax = canvas(
        "ILP：接続・出現・消失・分裂を、制約の下で一緒に選ぶ",
        "細胞候補をすべて残す簡略例。破線は接続候補、実線は選んだ接続。位置と費用は模式的。",
    )
    for offset, title in [(0, "入力：候補が競合している"), (53, "出力例：矛盾のない接続")]:
        ax.text(offset + 22, 47, title, ha="center", weight="bold", size=15)
        ax.text(offset + 8, 40, "時刻 t", ha="center", color=MUTED)
        ax.text(offset + 34, 40, "時刻 t+1", ha="center", color=MUTED)
        pos = {
            "A": (offset + 8, 29),
            "B": (offset + 8, 10),
            "C": (offset + 34, 34),
            "D": (offset + 34, 22),
            "E": (offset + 34, 7),
        }
        edges = [("A", "C"), ("A", "D"), ("B", "D"), ("B", "E")]
        for a, b in edges:
            if offset and (a, b) == ("B", "D"):
                continue
            xa, ya = pos[a]
            xb, yb = pos[b]
            arrow(
                ax, (xa + 2, ya), (xb - 2, yb), TEAL if offset else MUTED, dashed=not offset, lw=2.5
            )
        for label, (x, y) in pos.items():
            node(ax, x, y, label, TEAL if offset else BLUE)
    arrow(ax, (44, 25), (53, 25), BLUE, lw=2.5)
    ax.text(18, 0, "Dの親候補はAとB → 同時には選べない", size=12, ha="center")
    ax.text(75, 0, "Aは2娘へ分裂 / BはEへ継続", size=12, ha="center", color=TEAL)
    ax.text(
        71,
        30,
        "分裂",
        color=TEAL,
        size=12,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
    )
    save(fig, "01_ilp")


COST = np.array([[1, 2, 8], [2, 8, 9], [8, 9, 1]])


def matrix(ax, data, x, y, title, selected=(), cell=5.5):
    ax.text(x + 1.5 * cell, y + 4.6 * cell, title, ha="center", weight="bold", size=14)
    for j, label in enumerate("XYZ"):
        ax.text(x + (j + 0.5) * cell, y + 3.45 * cell, label, ha="center")
    for i, label in enumerate("ABC"):
        ax.text(x - 0.6 * cell, y + (2.5 - i) * cell, label, ha="center", va="center")
        for j in range(3):
            chosen = (i, j) in selected
            ax.add_patch(
                Rectangle(
                    (x + j * cell, y + (2 - i) * cell),
                    cell,
                    cell,
                    fc="#dff3ed" if chosen else "#f4f7fa",
                    ec="white",
                    lw=2,
                )
            )
            ax.text(
                x + (j + 0.5) * cell,
                y + (2.5 - i) * cell,
                str(data[i, j]),
                ha="center",
                va="center",
                color=TEAL if chosen else INK,
                weight="bold" if chosen else "normal",
                size=18,
            )


def hungarian():
    rows, cols = linear_sum_assignment(COST)
    assert cols.tolist() == [1, 0, 2]
    assert COST[rows, cols].sum() == 5
    fig, ax = canvas(
        "ハンガリアン法：全員の割り当て費用の合計を最小にする",
        "A・B・Cは前のフレーム、X・Y・Zは次のフレームの細胞。小さい数ほど接続しやすい。",
    )
    matrix(ax, COST, 7, 20, "入力：費用行列")
    for x, title, links, color in [
        (38, "近い順に確定する例", [("A", "X", 1), ("B", "Y", 8), ("C", "Z", 1)], RED),
        (73, "合計が最小の割り当て", [("A", "Y", 2), ("B", "X", 2), ("C", "Z", 1)], TEAL),
    ]:
        ax.text(x + 9, 45, title, ha="center", weight="bold", size=14)
        for i, a in enumerate("ABC"):
            node(ax, x, 34 - i * 11, a, color)
        for i, b in enumerate("XYZ"):
            node(ax, x + 18, 34 - i * 11, b, color)
        for a, b, _ in links:
            arrow(ax, (x + 2, 34 - "ABC".index(a) * 11), (x + 16, 34 - "XYZ".index(b) * 11), color)
        ax.text(
            x + 9,
            1,
            f"合計 = {' + '.join(str(c) for _, _, c in links)} = {sum(c for _, _, c in links)}",
            ha="center",
            color=color,
            size=14,
            weight="bold",
        )
    ax.text(15, 8, "AがXを使うと\nBに高い費用が残る", ha="center", color=MUTED)
    save(fig, "02_hungarian")


def hungarian_steps():
    row_reduced = COST - COST.min(axis=1, keepdims=True)
    reduced = row_reduced - row_reduced.min(axis=0, keepdims=True)
    fig, ax = canvas(
        "なぜ行・列の最小値を引いてよいのか",
        "完全な1対1割り当てなら、どの解も各行・各列から必ず1つずつ選ぶ。",
        5.5,
    )
    selected = [(0, 1), (1, 0), (2, 2)]
    matrix(ax, COST, 8, 20, "元の費用", selected)
    matrix(ax, row_reduced, 42, 20, "各行の最小値を引く", selected)
    matrix(ax, reduced, 76, 20, "各列の最小値を引く", selected)
    arrow(ax, (28, 29), (37, 29))
    arrow(ax, (62, 29), (71, 29))
    ax.text(17, 7, "選んだ値：2 + 2 + 1 = 5", ha="center", color=TEAL)
    ax.text(50, 7, "どの割り当ても合計が4減る", ha="center")
    ax.text(84, 7, "さらに合計が1減る", ha="center")
    ax.text(
        50,
        -1,
        "緑の3か所は行・列が重ならないゼロ → 変換後の下限0を達成 → 元の費用でも最適",
        ha="center",
        color=TEAL,
        size=12,
    )
    save(fig, "03_hungarian_steps")


def sort():
    fig, ax = canvas(
        "SORT：予測 → 割り当て → 修正を、フレームごとに繰り返す",
        "上段は原型SORTの処理。下段はカルマンフィルタの位置更新だけを1次元で説明した例。",
        6.5,
    )
    stages = [
        "① 動きを予測\nカルマンフィルタ",
        "② 費用を計算\n予測boxと検出boxのIoU",
        "③ 1対1に割り当て\nハンガリアン法",
        "④ 状態を更新\n新規・未対応の軌跡を管理",
    ]
    for x, s in zip([1, 26, 51, 76], stages, strict=True):
        box(ax, x, 33, 22, 13, s, BLUE, 11)
    for x in [23, 48, 73]:
        arrow(ax, (x + 0.5, 39), (x + 2.2, 39))
    ax.text(50, 28, "IoU：2つのboxの重なり面積 ÷ 和集合の面積", ha="center", size=12, color=MUTED)
    arrow(ax, (8, 12), (93, 12), MUTED, lw=1)
    for x, label, color, y in [
        (14, "前回 10", MUTED, 6),
        (52, "予測 12", BLUE, 19),
        (63.4, "更新 12.6", TEAL, 6),
        (71, "観測 13", ORANGE, 19),
    ]:
        ax.scatter([x], [12], s=170, color=color, zorder=3)
        ax.text(x, y, label, ha="center", color=color, size=13, weight="bold")
    arrow(ax, (17, 16), (49, 16), BLUE)
    ax.text(32, 21, "速度 +2 で予測", ha="center", color=BLUE, size=12)
    ax.text(
        51,
        -1,
        "更新値 = 12 + 0.6 × (13 − 12) = 12.6　　予測と観測を、不確かさに応じて混ぜる",
        ha="center",
        size=12,
    )
    save(fig, "04_sort")


def dog():
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[:120, :160]
    raw = 0.2 + 0.38 * xx / 160
    for x, y, amplitude, sigma in [(38, 44, 0.75, 5), (110, 45, 0.44, 6), (83, 87, 0.8, 9)]:
        raw += amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    raw += rng.normal(0, 0.035, raw.shape)
    small, large = gaussian_filter(raw, 2), gaussian_filter(raw, 6)
    response = small - large
    peaks = (response == maximum_filter(response, size=19)) & (response > 0.09)
    py, px = np.nonzero(peaks)
    assert len(px) == 3
    fig = plt.figure(figsize=(14, 8))
    fig.text(
        0.04, 0.96, "DoG：ぼかしの差で、局所的に明るい塊を浮かび上がらせる", size=22, weight="bold"
    )
    fig.text(
        0.04,
        0.915,
        "合成画像に実際にフィルタを適用。最初の3枚は共通の明るさ表示範囲、差の画像は専用の色目盛り。",
        color=MUTED,
        size=12,
    )
    for i, (arr, title) in enumerate(
        zip(
            [raw, small, large, response],
            ["① 元画像", "② 小さくぼかす：σ=2", "③ 大きくぼかす：σ=6", "④ 差：② − ③"],
            strict=True,
        )
    ):
        ax = fig.add_axes((0.035 + i * 0.244, 0.50, 0.217, 0.34))
        im = ax.imshow(
            arr,
            cmap="RdBu_r" if i == 3 else "gray",
            vmin=-0.35 if i == 3 else 0,
            vmax=0.35 if i == 3 else 1.4,
        )
        ax.set_title(title, size=13)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 3:
            ax.scatter(px, py, s=170, facecolors="none", edgecolors="#17392b", lw=2)
            fig.colorbar(
                im, ax=ax, orientation="horizontal", fraction=0.08, pad=0.04, ticks=[-0.3, 0, 0.3]
            )
    ax = fig.add_axes((0.07, 0.13, 0.57, 0.27))
    for arr, label, c in [
        (raw, "元画像", MUTED),
        (small, "小さいぼかし", BLUE),
        (large, "大きいぼかし", ORANGE),
        (response, "DoG応答", TEAL),
    ]:
        ax.plot(arr[45], label=label, color=c, lw=2)
    ax.axhline(0, color=LIGHT, lw=1)
    ax.set(xlabel="横位置（pixel）", ylabel="明るさ / 応答")
    ax.set_title("y=45の断面：背景の傾きが差で弱まり、2つの山が残る", size=12)
    ax.legend(ncol=2, fontsize=10, frameon=False)
    fig.text(
        0.70,
        0.36,
        "○ はDoGの局所最大\n\n明るい背景そのものより、\n"
        "周囲との差が大きい場所が残る。\n\nσはぼかしの広がり。\n"
        "実データでは物理的な大きさに合わせる。",
        size=13,
        va="top",
        linespacing=1.6,
    )
    save(fig, "05_dog")


def hog():
    yy, xx = np.mgrid[:32, :32]
    patch = 1 / (1 + np.exp(-(xx - 15) / 1.4))
    gy, gx = np.gradient(patch)
    mag = np.hypot(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180
    weights, _ = np.histogram(
        angle[8:16, 8:16], bins=np.arange(0, 181, 20), weights=mag[8:16, 8:16]
    )
    assert weights[0] > 0 and np.count_nonzero(weights) == 1
    fig = plt.figure(figsize=(14, 7))
    fig.text(
        0.04, 0.95, "HOG：明るさが変わる向きと強さを、小領域ごとに数える", size=22, weight="bold"
    )
    fig.text(
        0.04,
        0.89,
        "縦の輪郭を持つ合成画像。勾配の向きは、輪郭に垂直な横方向（0°）になる。",
        size=12,
        color=MUTED,
    )
    ax = fig.add_axes((0.04, 0.34, 0.25, 0.46))
    ax.imshow(patch, cmap="gray", vmin=0, vmax=1, origin="lower")
    for t in [7.5, 15.5, 23.5]:
        ax.axhline(t, color="#b8cadb", lw=0.8)
        ax.axvline(t, color="#b8cadb", lw=0.8)
    ax.add_patch(Rectangle((7.5, 7.5), 8, 8, fill=False, ec=ORANGE, lw=3))
    ax.set_title("① 8×8 pixelの小領域へ分割", size=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_axes((0.37, 0.34, 0.25, 0.46))
    ax.imshow(patch, cmap="gray", vmin=0, vmax=1, origin="lower", alpha=0.25)
    ax.quiver(
        xx[::4, ::4],
        yy[::4, ::4],
        gx[::4, ::4],
        gy[::4, ::4],
        color=BLUE,
        angles="xy",
        scale_units="xy",
        scale=0.035,
        width=0.009,
    )
    ax.set_title("② 勾配：暗い方から明るい方へ", size=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_axes((0.71, 0.39, 0.255, 0.35))
    ax.bar(np.arange(0, 180, 20) + 10, weights, width=16, color=ORANGE)
    ax.set(
        xlim=(0, 180),
        xticks=[0, 40, 80, 120, 160, 180],
        xlabel="勾配の向き（度）",
        ylabel="勾配の強さの合計",
    )
    ax.set_title("③ オレンジの小領域のヒストグラム", size=12)
    fig.text(
        0.06,
        0.22,
        "④ 隣り合う2×2小領域をまとめて正規化 → 順番につないで特徴ベクトルにする",
        size=15,
        weight="bold",
    )
    fig.text(
        0.06,
        0.11,
        "ここでの「cell」は画像の小領域を指す。生物の細胞ではない。\n図の投票は説明用の単純化。実装により、角度・位置の隣接区間にも重みを配分する。",
        size=12,
        color=MUTED,
    )
    save(fig, "06_hog")


def contrastive():
    fig, ax = canvas(
        "対照学習：「同じ」と教えた組の特徴を近く、「違う」組を遠くする",
        "既知の同一細胞Aを正例、既知の別細胞Bを負例にする模式図。右図は説明用の配置で、学習結果ではない。",
        7,
    )
    box(ax, 1, 32, 19, 12, "細胞A：時刻 t\n画像または固定特徴", BLUE, 11)
    box(ax, 1, 17, 19, 12, "細胞A：時刻 t+1\n正例", BLUE, 11)
    box(ax, 1, 2, 19, 12, "別の細胞B\n負例", ORANGE, 11)
    box(ax, 27, 12, 20, 25, "同じ特徴変換 f\n（重みを共有）\n↓\n特徴ベクトル", TEAL, 12)
    for y in [38, 23, 8]:
        arrow(ax, (21, y), (26, 24), MUTED)
    arrow(ax, (48, 24), (55, 24), TEAL)
    ax.text(77, 45, "特徴空間：距離が類似度を表す", ha="center", size=14, weight="bold")
    ax.add_patch(Rectangle((58, 6), 39, 33, facecolor="#f4f7fa", edgecolor=LIGHT))
    ax.scatter([67, 74, 87], [26, 31, 14], color=[BLUE, BLUE, ORANGE], s=180, zorder=3)
    ax.text(63, 24, "A(t)", color=BLUE, size=12)
    ax.text(73, 34, "A(t+1)", color=BLUE, size=12)
    ax.text(89, 13, "B", color=ORANGE, size=13)
    arrow(ax, (72, 30), (68, 27), TEAL)
    ax.text(63, 35, "近づける", color=TEAL, size=12)
    arrow(ax, (71, 25), (83, 17), ORANGE)
    ax.text(77, 24, "離す", color=ORANGE, size=12)
    ax.text(77, 0, "教師の組み方が「似る」の意味を決める", ha="center", color=TEAL, size=13)
    save(fig, "07_contrastive")


if __name__ == "__main__":
    setup()
    for draw in [overview, ilp, hungarian, hungarian_steps, sort, dog, hog, contrastive]:
        draw()
    print(f"Saved 8 figures to {OUT.relative_to(ROOT)}")
