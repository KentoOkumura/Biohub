"""Illustrate nine newly published solutions using synthetic teaching examples.

Reuse the existing report's colours and drawing primitives. No competition
images, trained predictions, or remote image assets are used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "studies/biohub_top_solutions_20260930"))
import make_solution_figures as b

OUT = ROOT / "docs/images/biohub_top_solutions_20261002"
b.OUT = OUT
np, plt = b.np, b.plt
page, panel, txt, dot, arrow = b.page, b.panel, b.txt, b.dot, b.arrow
cube, strip, heat, tree, save = b.cube, b.strip, b.heat_image, b.tree, b.save
BLUE, GREEN, ORANGE, RED, GREY, PURPLE = b.BLUE, b.TEAL, b.ORANGE, b.RED, b.GREY, b.PURPLE


def frames(ax, *, labels=("t−1", "t", "t+1"), y=4.6):
    images = b.division_images()
    strip(ax, images[: len(labels)], labels, y=y, height=1.5)


def graph(ax, *, x=1.2, y=4.5, fork=False, orphan=True, other=False):
    parent, first, second = (x, y), (x + 2.1, y + .65), (x + 2.1, y - .85)
    dot(ax, parent, "母")
    dot(ax, first, "娘1", color=GREEN if fork else BLUE)
    dot(ax, second, "娘2", color=GREEN if fork else ORANGE, hollow=orphan and not fork)
    arrow(ax, parent, first, color=GREEN if fork else BLUE)
    if fork:
        arrow(ax, parent, second, color=GREEN)
    if other:
        p2 = (x, y - 2.1)
        dot(ax, p2, "B", color=RED)
        arrow(ax, p2, second, color=RED, dashed=fork)
    return parent, first, second


def first():
    fig, axes = page(1, "細胞の状態・前後の位置を予測し、娘の数と相手を選ぶ", "3D U-Net + 作者のSoon Net + 局所Cross-Attention。最終選択はILPではなく貪欲法。")
    ax = axes[0]
    panel(ax, "A", "3D U-Net: 核の領域と中心への向き")
    ax.imshow(heat([(.5, .5, .20, .16)]), extent=(.2, 3, 3.1, 6.3), aspect="auto")
    for x, y in [(0.8, 4), (2.5, 4), (1.1, 5.7), (2.4, 5.8)]:
        arrow(ax, (x, y), (1.65, 4.7), color=GREEN)
    dot(ax, (1.65, 4.7), size=95)
    for i, (ch, h) in enumerate([(24, 1.6), (48, 1.2), (96, .85), (192, .6)]):
        cube(ax, 3.7 + i * 1.5, 4.5 - h / 2, .95, h, label=f"{ch}ch", layers=2)
    txt(ax, 3.7, 2.65, "各block: 3³ Conv / BatchNorm / ReLU ×2", size=10)
    txt(ax, .2, 1.55, "前景確率 + 中心へ向かう3軸flow\n高い確率のvoxelをflowに沿って集める", size=11)
    txt(ax, .2, .25, "事前学習: 小領域を隠して画像を復元するMAE", color=PURPLE, size=10.5)
    ax = axes[1]
    panel(ax, "B", "query細胞の状態と、前後の存在位置")
    frames(ax)
    txt(ax, .1, 3.8, "3 frames × 16×64×64 + query位置channel", size=10)
    cube(ax, .5, 2.4, 1.4, .6, color=PURPLE, label="共有3D CNN")
    cube(ax, 3.0, 2.4, 1.8, .6, color=PURPLE, label="小型Transformer")
    txt(ax, 6.0, 3.0, "通常 / 分裂直前 / 直後", size=10, color=GREEN)
    ax.imshow(heat([(.32, .5, .06, .08), (.70, .5, .06, .08)]), extent=(6.0, 9.1, .25, 2.2), aspect="auto")
    txt(ax, .2, .6, "FPNから各時点の\n通常・分裂の存在位置map", size=10.5)
    txt(ax, 7.5, 2.55, "次時点の2娘の位置（例）", ha="center", size=9.5)
    ax = axes[2]
    panel(ax, "C", "137候補を「娘の数」と「誰か」に分ける")
    txt(ax, .1, 6.3, "42 node特徴 / 24 pair特徴\n4 blockの局所Cross-Attention + 採点MLP", size=10.5)
    for x, colour, label in [(1.8, GREY, "0娘"), (4.9, BLUE, "1娘"), (8.0, GREEN, "2娘")]:
        dot(ax, (x, 4.55), label, color=colour, size=850)
    txt(ax, 5, 3.55, "まず kind = 0 / 1 / 2 を選ぶ", ha="center", size=11)
    for x, n, colour in [(1.8, 1, GREY), (4.9, 16, BLUE), (8.0, 120, GREEN)]:
        for j in range(min(n, 16)):
            ax.scatter(x - .65 + j % 4 * .43, 2.75 - j // 4 * .33, color=colour, s=65)
        txt(ax, x, 1.1, f"{n}候補" + ("（一部表示）" if n > 16 else ""), color=colour, ha="center", size=10.2)
    txt(ax, .1, .15, "次に who を条件付きで選ぶ。16個から2娘は120組。", size=10.3)
    ax = axes[3]
    panel(ax, "D", "利得順に採用し、親・娘の重複を避ける")
    graph(ax, x=1.1, y=4.8, fork=True, other=True)
    dot(ax, (8.4, 2.7), "E", color=BLUE)
    arrow(ax, (1.1, 2.7), (8.4, 2.7))
    txt(ax, .2, 1.8, "母Aの2娘候補を先に採用した模式例\nB→娘2は重複するので採用できない", size=11)
    txt(ax, .2, .3, "親は1候補、各娘は1親。未接続の点は新しいtrack。", size=10.5)
    save(fig, "01_occupancy_hierarchical_fork", 1, 744801,
         ["detector channels 24/48/96/192", "Soon Net query crops and occupancy maps", "4 local cross-attention blocks", "137 options with kind/who factorisation", "greedy conflict-aware decoding"])


def second():
    fig, axes = page(2, "弱い検出を保ち、長い軌跡として回収する", "Residual 3D U-Netの複数出力 + 複数候補bank。最後は未使用の検出列を画像で確認する。")
    ax = axes[0]
    panel(ax, "A", "弱い細胞を学ぶため、実画像をdimにして移植")
    strong = heat([(.5, .5, .12, .15)])
    faint = heat([(.5, .5, .12, .15, .22)])
    strip(ax, [strong, faint, heat([(.3, .4, .1, .12), (.7, .7, .1, .12, .22)])],
          ["実際の単独細胞", "減光・blur", "同じmovieへ合成"], y=4.4)
    for k in range(5):
        dot(ax, (1 + k * 1.65, 2.0 + .12 * k), color=ORANGE, size=120)
        if k:
            arrow(ax, (1 + (k - 1) * 1.65, 2 + .12 * (k - 1)), (1 + k * 1.65, 2 + .12 * k), color=ORANGE)
    txt(ax, .1, 3.5, "5〜15 framesの位置・対応が既知の弱いtrack", size=10.5)
    txt(ax, .1, .65, "同じ学習movie内で移植し、別foldへ持ち込まない", size=10.5)
    ax = axes[1]
    panel(ax, "B", "画像の複数出力と、平均で消えるpeakの保存")
    cube(ax, .4, 4.6, 1.7, 1.1, color=PURPLE, label="3 frames × 64³")
    for i, label in enumerate(["核中心heatmap", "sub-voxel offset", "逆向きmotion + 不確かさ", "見た目のdescriptor"]):
        ax.barh(5.8 - i * .65, 4.4, left=4.1, height=.38, color=BLUE if i < 2 else GREEN, alpha=.6)
        txt(ax, 4.2, 5.8 - i * .65, label, size=10)
    x = np.linspace(0, 10, 160)
    ax.plot(x, 1 + 1.8 * np.exp(-(x - 2.5) ** 2 / .3), color=BLUE)
    ax.plot(x, 1 + .9 * np.exp(-(x - 2.5) ** 2 / .4), color=GREY)
    dot(ax, (2.5, 2.8), color=ORANGE, size=90)
    txt(ax, 4.2, 1.7, "原viewのpeakも保存\n平均heatmapだけで捨てない", size=10.5)
    txt(ax, .1, .2, "層数・channel数は本文未提示", color=b.MUTED, size=10)
    ax = axes[2]
    panel(ax, "C", "単発の弱い点ではなく、9時点以上の列を確認")
    for row, colour, n in [(4.8, BLUE, 9), (2.8, ORANGE, 9), (1.3, RED, 3)]:
        for i in range(n):
            xy = (.6 + i * 1.02, row + .10 * np.sin(i))
            dot(ax, xy, color=colour, size=90)
            if i:
                arrow(ax, (.6 + (i - 1) * 1.02, row + .10 * np.sin(i - 1)), xy, color=colour, lw=1.5)
    txt(ax, .1, 6.2, "既存の強いtrack", color=BLUE)
    txt(ax, .1, 3.7, "未使用候補: 隣接移動≤5 µm、≥9 frames", color=ORANGE, size=10.5)
    txt(ax, .1, .2, "raw画像の中央contrast中央値 > 0.1 も必要", size=10.5)
    ax = axes[3]
    panel(ax, "D", "独立した弱いtrackを、空いている端だけへ接続")
    for i, xy in enumerate([(1.1, 4.8), (3.2, 4.2), (5.3, 3.6), (7.4, 3.0)]):
        dot(ax, xy, color=BLUE if i < 2 else ORANGE, size=220)
        if i:
            arrow(ax, [(1.1, 4.8), (3.2, 4.2), (5.3, 3.6)][i - 1], xy, color=ORANGE if i == 2 else BLUE)
    dot(ax, (3.2, 2.4), color=RED, size=220)
    arrow(ax, (3.2, 2.4), (5.3, 3.6), color=RED, dashed=True)
    txt(ax, .1, 1.5, "近い既存trackがあること、画像の支持、競合を確認\n親・子が既に埋まる場合は上書きしない", size=10.5)
    txt(ax, .1, .2, "保存済み候補を使用。追加の検出器推論は行わない。", size=10.3)
    save(fig, "02_faint_track_recovery", 2, 744723,
         ["3-frame residual 3D U-Net with detection/offset/backward-motion/descriptors", "within-fold dim-cell copy paste", "preserve original-view peaks", "recover unused >=9-frame paths with image contrast", "attach only free endpoints"])


def fourth():
    fig, axes = page(4, "母ごとの分裂確率をILPの費用に反映する", "Dual-encoder 3D U-Net + 学習した辺・分裂の得点。画像を読む検証器で偽分裂を落とす。")
    ax = axes[0]
    panel(ax, "A", "現在画像と時間差分を、別encoderへ")
    frames(ax)
    cube(ax, .8, 2.7, 2.0, .8, color=BLUE, label="現在画像encoder")
    cube(ax, 6.0, 2.7, 2.0, .8, color=PURPLE, label="t+1 − t−1 encoder")
    arrow(ax, (2.9, 3.1), (4.6, 1.6))
    arrow(ax, (6.0, 3.1), (4.6, 1.6))
    txt(ax, 4.6, 1.4, "temporal attention", ha="center", size=11)
    txt(ax, .1, .3, "3D decoder → 前景確率 + 中心flow（18M parameters）", size=10.5)
    ax = axes[1]
    panel(ax, "B", "pseudo-labelで増えたtrackに、別モデルの支持")
    for i in range(6):
        xy = (.7 + i * 1.55, 4.7 + .12 * np.sin(i))
        dot(ax, xy, color=ORANGE, size=130)
        if i in (0, 1, 3, 5):
            dot(ax, (xy[0], 3.1), color=BLUE, hollow=True, size=130)
            ax.plot([xy[0], xy[0]], [3.4, 4.4], color=GREY, ls="--")
    txt(ax, .1, 6.1, "GT + pseudo-labelモデルのtrack", color=ORANGE)
    txt(ax, .1, 2.0, "GTだけのモデルが6 µm内に検出\n50%以上の点で支持されたtrackを採用", size=11)
    txt(ax, .1, .2, "感度を上げるだけでは、別胚の偽trackも増える。", size=10.5)
    ax = axes[2]
    panel(ax, "C", "qに応じて分裂費用を変え、通常辺と比較")
    graph(ax, x=1.1, y=4.8, fork=True, other=True)
    txt(ax, 5.3, 5.4, "母の分裂確率 q", size=11)
    txt(ax, 5.3, 4.4, "CNN / CatBoost\n娘pairのtree models", size=10.5)
    txt(ax, 5.3, 2.7, "費用 = 6 − 16q", color=GREEN, size=12)
    txt(ax, 5.3, 1.8, "例 q=0.1 → 4.4\n例 q=0.8 → −6.8", size=10.5)
    txt(ax, .1, .2, "ILPは母ごとの費用と辺得点を使い、競合する接続を選ぶ。", size=10.1)
    ax = axes[3]
    panel(ax, "D", "分裂を選んだ後も、画像内の分離を確認")
    xx = np.arange(9)
    ax.plot(.4 + xx * .75, 4.1 + .18 * xx, color=GREEN, lw=3)
    ax.plot(.4 + xx * .75, 4.1 - .18 * xx, color=GREEN, lw=3)
    ax.plot(.4 + xx * .75, 2.0 + .03 * xx, color=RED, lw=2, ls="--")
    ax.plot(.4 + xx * .75, 2.0 - .03 * xx, color=RED, lw=2, ls="--")
    txt(ax, 7.0, 4.7, "離れる2娘", color=GREEN, size=10)
    txt(ax, 7.0, 2.0, "同じ核の\n重複候補", color=RED, size=10)
    txt(ax, .1, 6.2, "raw画像のDoG peak: t−3 … t+5", size=11)
    txt(ax, .1, .25, "157特徴のCatBoostで検証。得点<0.25なら弱い腕を切る。", size=10)
    save(fig, "04_learned_division_cost", 4, 744673,
         ["18M dual-encoder temporal-attention U-Net", "pseudo-label tracks supported by GT-only model", "edge LightGBM and division CNN/tree models", "node-dependent ILP division cost 6-16q", "post-fork image verifier"])


def seventh():
    fig, axes = page(7, "逆向きmotion・分裂CNNと、2段階のILP", "nnU-Net ResEncMが検出とmotionを同時予測。暫定graphから代替の親も調べ、候補全体を再最適化。")
    ax = axes[0]
    panel(ax, "A", "motionが示した親の位置から候補を探す")
    dot(ax, (7.7, 4.7), "娘", color=BLUE)
    dot(ax, (3.4, 3.5), "親", color=GREEN)
    dot(ax, (6.1, 4.0), "B", color=RED)
    arrow(ax, (7.7, 4.7), (3.4, 3.5), color=PURPLE, lw=3)
    ax.add_patch(b.Ellipse((3.4, 3.5), 4.0, 3.6, fill=False, edgecolor=GREEN, ls="--", lw=2))
    txt(ax, .1, 6.3, "t−d / t / t+d + 間隔channel、d=1→2→3", size=10.5)
    txt(ax, .1, 1.0, "逆向き変位で予測した位置の7 µm内・上位5候補\n移動自体を7 µm以下へ制限する意味ではない", size=10.5)
    ax = axes[1]
    panel(ax, "B", "分裂CNN: 3つの窓 + 双方向GRU")
    imgs = b.division_images()
    strip(ax, [imgs[0], imgs[0], imgs[1], imgs[2], imgs[3]],
          ["t−2", "t−1", "t", "t+1", "t+2"], y=4.8)
    for i in range(3):
        ax.plot([.25 + i * 1.91, 5.83 + i * 1.91], [4.1 - i * .26] * 2, color=PURPLE, lw=4)
    txt(ax, .1, 3.15, "各窓14ch: raw / 時間差分 / 候補heatmap\n共有3D CNN 24→48→96→128 → BiGRU", size=10.5)
    txt(ax, .1, 1.65, "主出力: 分裂確率\n補助出力: 娘の数0/1/2、相対位置2×3軸", size=11)
    txt(ax, .1, .2, "補助出力は学習の手がかり。提出する娘を直接決めない。", size=10)
    ax = axes[2]
    panel(ax, "C", "第1ILPの辺を固定せず、第2ILPで入れ替える")
    graph(ax, x=.5, y=4.7, fork=False, other=True, orphan=False)
    graph(ax, x=5.6, y=4.7, fork=True, other=True)
    arrow(ax, (3.3, 3.6), (5.4, 3.6), color=ORANGE)
    txt(ax, 2.0, 6.3, "暫定graph", ha="center", size=11)
    txt(ax, 7.3, 6.3, "第2ILPの選択例", ha="center", size=11)
    txt(ax, .1, 1.45, "娘pair・通常接続の代替親を暫定graphから特徴化\n83→32→1 MLPで費用更新し、元の候補graphを再び解く", size=10.3)
    txt(ax, .1, .15, "本文の2段階版。他の提出構成には1段階版もある。", color=b.MUTED, size=10)
    ax = axes[3]
    panel(ax, "D", "画像が同じ時刻には、座標も一致させる")
    image = heat([(.5, .5, .13, .15)])
    strip(ax, [image] * 3, ["同じraw frame", "同じraw frame", "同じraw frame"], y=4.5)
    for x, y in [(1.7, 2.5), (4.8, 3.0), (8.0, 2.7)]:
        dot(ax, (x, y), hollow=True, color=GREY, size=200)
        dot(ax, (x, 2.1), color=ORANGE, size=130)
        arrow(ax, (x, y), (x, 2.1), color=ORANGE)
    txt(ax, .1, .85, "1対1の対応内で実際の検出点をanchorにする\n分裂・競合・補間した点をanchorから除く", size=10.5)
    save(fig, "07_motion_two_stage_ilp", 7, 744937,
         ["joint nnU-Net ResEncM detection and backward motion", "14-channel overlapping windows CNN 24/48/96/128 with BiGRU and auxiliaries", "two-stage ILP version reoptimises original candidates", "alternative-parent features", "identical-frame coordinate anchors"])


def ninth():
    fig, axes = page(9, "自前検出器を融合し、後処理で消えた分裂を戻す", "公開検出器・tracker + 自前6検出器・1 linker。完成CSVへ、別ILPで得た分裂候補を追加する。")
    ax = axes[0]
    panel(ax, "A", "heatmapと対応得点を融合し、自前offsetを使う")
    x = np.linspace(0, 10, 160)
    ax.plot(x, 2.4 + 2.7 * np.exp(-(x - 4.6) ** 2 / .6), color=BLUE, label="public")
    ax.plot(x, 2.4 + 2.4 * np.exp(-(x - 5.1) ** 2 / .6), color=PURPLE, label="own")
    ax.axvline(5, ymin=.28, ymax=.75, color=GREY, ls="--")
    arrow(ax, (5, 1.3), (5.8, 1.3), color=ORANGE)
    txt(ax, .1, 6.3, "public 50% + 自前6モデル平均 50%", size=11)
    txt(ax, .1, .4, "TemporalUNet3D + 1×1 Convの3軸offset head\n自前linker S6も、publicとの辺得点融合に使用", size=10.5)
    ax = axes[1]
    panel(ax, "B", "ILPで作った分裂が、motion relinkで消える")
    graph(ax, x=.5, y=4.3, fork=True)
    graph(ax, x=5.5, y=4.3, fork=False)
    arrow(ax, (3.3, 3.4), (5.2, 3.4), color=ORANGE)
    txt(ax, 1.8, 6.1, "元候補からのILP", ha="center", size=11)
    txt(ax, 7.1, 6.1, "後処理のgraph", ha="center", size=11)
    txt(ax, .1, 1.4, "ILP費用を変えても、再接続が選択を上書きすると\n最終graphにはその変更が伝わらない。", size=11)
    ax = axes[2]
    panel(ax, "C", "最後に、親なし第2娘だけを追加する")
    graph(ax, x=1.0, y=4.4, fork=True)
    txt(ax, 5.0, 5.6, "作者のDIVCARRY", color=ORANGE, size=12)
    txt(ax, 5.0, 4.15, "元候補を別費用のILPで解く\n母は最終graphで子1つ\n第2娘は親なし", size=10)
    txt(ax, .1, 1.5, "両娘の長さ・分離・近傍との間隔を確認\n完成CSVに追加するので、後段relinkに消されない", size=10.5)
    txt(ax, .1, .2, "追加上限: 元の辺数の1%。点の位置は変更しない。", size=10.2)
    ax = axes[3]
    panel(ax, "D", "最終座標の整数化も、空間的なずれを作る")
    for i in range(8):
        ax.plot([i + 1, i + 1], [2.0, 5.6], color=b.LIGHT)
        txt(ax, i + 1, 1.5, str(i), ha="center", size=10)
    dot(ax, (5.8, 4.8), color=ORANGE, size=170)
    arrow(ax, (5.8, 4.8), (5, 3.0), color=RED)
    arrow(ax, (5.8, 4.8), (6, 3.0), color=GREEN)
    txt(ax, .1, 6.3, "説明例: 出力4.8 voxel → 切り捨て4 / 丸め5", size=10.5)
    txt(ax, .1, .2, "9位はnp.rintを採用。こちらexp043もroundを使用済み。", size=10)
    save(fig, "09_division_after_relink", 9, 744835,
         ["six own TemporalUNet3D detectors with offset heads", "50/50 public and own fusion", "motion relink can overwrite ILP forks", "DIVCARRY restores orphan-only forks at final CSV", "np.rint output rounding"])


def tenth():
    fig, axes = page(10, "見た目と辺のattention、画像による分裂検証", "独立したDINO特徴とHOCT型tracker。別検出器の新trackは、完成graphへ後から加える。")
    ax = axes[0]
    panel(ax, "A", "pixel-unshuffle: XYの位置をchannelへ移す")
    colours = [BLUE, GREEN, ORANGE, PURPLE]
    for row in range(4):
        for col in range(4):
            ax.add_patch(b.Rectangle((.5 + col * .65, 3.1 + row * .65), .61, .61, facecolor=colours[(row % 2) * 2 + col % 2]))
    arrow(ax, (3.5, 4.4), (5, 4.4))
    for k, colour in enumerate(colours):
        cube(ax, 5.2 + k * .9, 3.3 + .25 * k, .8, 1.2, color=colour, layers=1)
    txt(ax, .1, 6.3, "高解像度temporal U-Netのstem", size=11)
    txt(ax, .1, 1.6, "2×2位置を4 channelへ並べ直す模式例\nXYを小さくしても、単純平均より細部を保つ", size=11)
    txt(ax, .1, .2, "encoder初段はXYだけ縮小。高解像度skipをdecoderへ。", size=10)
    ax = axes[1]
    panel(ax, "B", "細胞の見た目と、候補辺同士の関係を学ぶ")
    pts = [(1.2, 5.0), (1.2, 2.8), (7.9, 5.5), (7.9, 3.7), (7.9, 2.0)]
    for xy in pts:
        dot(ax, xy, size=190)
    for p, q in [(0, 2), (0, 3), (1, 3), (1, 4)]:
        arrow(ax, pts[p], pts[q], color=BLUE, lw=1.5)
    arrow(ax, (4, 4.4), (4, 3.25), color=PURPLE, bend=.45)
    arrow(ax, (4.4, 3.25), (4.4, 4.4), color=PURPLE, bend=.45)
    txt(ax, .1, 6.5, "DINO ViT-B/16 → 4層frame Self-Attention → 256d", size=10)
    txt(ax, .1, .7, "HOCT型: node attentionに加え、辺の間にもattention\n距離・見た目・候補順位・周囲の運動を入力", size=10.3)
    ax = axes[2]
    panel(ax, "C", "純画像の分裂判定は、誤ったforkも取り消す")
    frames(ax)
    cube(ax, .6, 2.5, 2.1, .7, color=PURPLE, label="ConvNeXt-Tiny")
    cube(ax, 4.0, 2.5, 2.8, .7, color=PURPLE, label="時空間Transformer")
    txt(ax, .1, 1.15, "3 frames × 32²、各画像は隣接3 z断面\n4 seedの平均で分裂を検証し、弱い腕を切る", size=11)
    txt(ax, .1, .2, "高い分裂確率なら、既存の親なし娘の回収にも使用。", size=10.2)
    ax = axes[3]
    panel(ax, "D", "別検出器の新trackを加え、既存の軌跡を保つ")
    for row, colour in [(4.7, BLUE), (2.7, ORANGE)]:
        for i in range(5):
            xy = (1 + i * 1.7, row + .08 * i)
            dot(ax, xy, color=colour, size=160)
            if i:
                arrow(ax, (1 + (i - 1) * 1.7, row + .08 * (i - 1)), xy, color=colour)
    txt(ax, .1, 6.3, "main: ILPで点選択 → Hungarianで辺を選択", size=10.5)
    txt(ax, .1, 1.6, "追加: DSB2018事前学習の2.5D ResNet34 U-Net\n64chのslice特徴 → 4つの3D residual block", size=10.5)
    txt(ax, .1, .2, "独立に追跡した新しい3時点以上のtrackを、離れた所へ追加。", size=10)
    save(fig, "10_appearance_edge_attention", 10, 744980,
         ["high-resolution temporal U-Net with pixel-unshuffle", "DINO ViT-B/16 and 4-layer frame attention", "HOCT-style node and edge attention", "ConvNeXt-Tiny temporal image division verifier", "late 2.5D ResNet34 feature + 3D residual detector track addition"])


def eleventh():
    fig, axes = page(11, "検出・逆向きmotion・母の分裂を同じU-Netで学ぶ", "Neural headsは密な画像特徴を共有。検出点の選別やILPは学習したネットワークの後で行う。")
    ax = axes[0]
    panel(ax, "A", "3つの教師が、共有encoderへ戻る")
    strip(ax, b.division_images()[1:3], ["前時点", "後時点"], y=4.8)
    cube(ax, 3.0, 2.8, 3.0, .8, color=PURPLE)
    txt(ax, 4.5, 3.2, "共有Temporal 3D U-Net", color="white", ha="center", size=10)
    for x, label, colour in [(1.3, "核中心", BLUE), (5.0, "逆向きmotion", ORANGE), (8.5, "母の分裂", GREEN)]:
        dot(ax, (x, 1.2), color=colour, size=280)
        arrow(ax, (4.5, 2.8), (x, 1.2), color=colour)
        arrow(ax, (x + .45, 1.3), (5.0, 2.8), color=RED, dashed=True, lw=1.2)
        txt(ax, x, .5, label, ha="center", size=10.5)
    txt(ax, .1, 4.25, "赤い破線: 各headの損失からbackboneも更新", size=10)
    ax = axes[1]
    panel(ax, "B", "母の教師: 既知の子2つ / 子1つ / 不明")
    for i, (n, colour, label) in enumerate([(2, GREEN, "正例"), (1, BLUE, "負例"), (0, GREY, "教師から除外")]):
        x = 1.6 + i * 3.1
        dot(ax, (x, 4.8), "母", color=colour)
        for j in range(n):
            child = (x + (j - (n - 1) / 2) * 1.3, 3.0)
            dot(ax, child, color=colour, size=160)
            arrow(ax, (x, 4.8), child, color=colour)
        txt(ax, x, 1.8, label, color=colour, ha="center", size=11)
    txt(ax, .1, 6.2, "分裂head: 前特徴 + 後特徴 + その差分", size=11)
    txt(ax, .1, .2, "母の分裂確率を予測する。2娘のIDを出すheadではない。", size=10)
    ax = axes[2]
    panel(ax, "C", "合成系列では、すべての中心・移動・分裂が既知")
    strip(ax, b.division_images(), ["合成 t−1", "合成 t", "合成 t+1", "合成 t+2"], y=4.5)
    for k in range(3):
        p = (.9 + k * 2.5, 2.8)
        dot(ax, p, color=ORANGE, size=170)
        arrow(ax, p, (p[0] + 1.9, 2.8), color=ORANGE)
    txt(ax, .1, 1.3, "FOCUS: 中心の数・間隔の統計\nコンペGT: 動き・分裂頻度の統計", size=11)
    txt(ax, .1, .2, "FOCUSの実画像を時間方向の正解として使う意味ではない。", size=10)
    ax = axes[3]
    panel(ax, "D", "学習済みのqとmotionを、graph選択の費用へ")
    graph(ax, x=1.1, y=4.8, fork=True, other=True)
    txt(ax, 5.3, 5.3, "例: 分裂費用 0.5 − q", color=GREEN, size=11)
    txt(ax, 5.3, 4.0, "q=.1 → .4\nq=.8 → −.3", size=11)
    txt(ax, 5.3, 2.7, "辺費用は\n逆向きmotion残差から", size=10.5)
    txt(ax, .1, 1.0, "モデル学習後に、候補graph・費用を公式指標で調整", size=10.5)
    txt(ax, .1, .1, "ILP・公式指標からニューラルネットへ勾配は流さない。", color=b.MUTED, size=10)
    save(fig, "11_joint_dense_heads", 11, 745092,
         ["joint temporal U-Net heat/motion/division dense heads", "division earlier/later/difference features", "known two-child/one-child division labels", "FOCUS statistics and GT motion synthetic generation", "ILP after network training, not differentiable solver"])


def sixteenth():
    fig, axes = page(16, "検出器を学習し、完成graphをpseudo-labelへ戻す", "nnU-Net ResEnc-Lと事前学習ResNet3D18 U-Net。新しい検出候補に合わせ、Transformer部分も再学習。")
    ax = axes[0]
    panel(ax, "A", "Goldのほかに、teacherが検出したSilverを使う")
    for i, xy in enumerate([(1.3, 4.7), (4.6, 5.1), (7.8, 4.6), (2.7, 2.6), (6.1, 2.5)]):
        dot(ax, xy, color=BLUE if i < 2 else ORANGE, size=230)
    txt(ax, .1, 6.3, "teacher: Kinetics事前学習R(2+1)D18 + 時間Conv1D", size=10)
    txt(ax, .1, 1.25, "Gold: 人が注釈した中心 / Silver: teacherの予測中心\nGoldとSilverの損失を分け、Silverは0.25倍", size=11)
    txt(ax, .1, .15, "Silverは追加の予測教師。人の注釈と同じ信頼度にしない。", size=10)
    ax = axes[1]
    panel(ax, "B", "単なるheatmap peakでなく、追跡で残った点を教師へ")
    for i in range(5):
        xy = (.8 + i * 1.7, 4.1 + .1 * i)
        dot(ax, xy, color=ORANGE, size=160)
        if i:
            arrow(ax, (.8 + (i - 1) * 1.7, 4.1 + .1 * (i - 1)), xy, color=ORANGE)
    dot(ax, (4.2, 2.7), color=GREY, hollow=True, size=160)
    txt(ax, .1, 6.3, "検出 → ILP + 後処理 → 完成graph", size=11)
    txt(ax, .1, 1.6, "次roundのSilverには、完成graphの点を使う\n候補peakの短い・孤立した誤検出を減らす", size=11)
    txt(ax, .1, .2, "各roundは同じ初期状態から学習。round間で重みを継続しない。", size=9.8)
    ax = axes[2]
    panel(ax, "C", "異なる検出器の融合 + 候補に合わせたedge head")
    cube(ax, .6, 4.4, 2.4, 1.0, color=BLUE, label="nnU-Net ResEnc-L")
    cube(ax, 5.6, 4.4, 2.8, 1.0, color=PURPLE, label="ResNet3D18 U-Net")
    txt(ax, .1, 3.0, "random初期化", color=BLUE, size=10)
    txt(ax, 5.3, 3.0, "Kinetics400事前学習", color=PURPLE, size=10)
    txt(ax, .1, 1.5, "最終: logits 50% / 50%、4 XY反転view\n新検出器の候補に合わせTransformer部分を5 epochs FT", size=10.8)
    txt(ax, .1, .2, "専用の分裂CNNは最終構成に採用していない。", color=b.MUTED, size=10.5)
    ax = axes[3]
    panel(ax, "D", "FOCUSの中心は、背景と断定しないために使う")
    ax.add_patch(b.Rectangle((.3, 2), 9.1, 3.7, facecolor=b.LIGHT))
    for xy, colour in [((2.0, 4.2), BLUE), ((5.0, 4.0), ORANGE), ((8.0, 3.8), PURPLE)]:
        ax.add_patch(b.Ellipse(xy, 2.15, 2.7, facecolor="white", edgecolor=colour, ls="--"))
        dot(ax, xy, color=colour, size=200)
    txt(ax, 2.0, 1.4, "Gold", ha="center", color=BLUE)
    txt(ax, 5.0, 1.4, "Silver", ha="center", color=ORANGE)
    txt(ax, 8.0, 1.4, "FOCUS", ha="center", color=PURPLE)
    txt(ax, .1, 6.3, "追加の背景は、これらの中心から5 µm以上離れた領域", size=10)
    txt(ax, .1, .2, "FOCUSを検出の正例や時間方向の対応ラベルにはしない。", size=10.2)
    save(fig, "16_graph_pseudo_labels", 16, 744671,
         ["R(2+1)D18 + temporal Conv1D teacher", "Gold and Silver separate losses", "completed graph nodes used for next pseudo round", "ResEnc-L and Kinetics ResNet3D18 U-Net logit ensemble", "Transformer fine-tuning and FOCUS background exclusion"])


def seventeenth():
    fig, axes = page(17, "Tree modelの対応と、画像・公式指標を使うgraph修正", "3D U-Net + 2回の線形割当。分裂の隙間を画像で確認し、修正の教師には実際の評価差を使う。")
    ax = axes[0]
    panel(ax, "A", "大きな核には入力の倍率を合わせ、二重peakを減らす")
    large = heat([(.5, .5, .24, .16)])
    ax.imshow(large, extent=(.4, 4.3, 3.0, 6.0), aspect="auto")
    dot(ax, (1.8, 4.5), color=RED, hollow=True, size=140)
    dot(ax, (2.9, 4.5), color=RED, hollow=True, size=140)
    ax.imshow(large, extent=(6.3, 8.7, 3.6, 5.45), aspect="auto")
    dot(ax, (7.5, 4.5), size=140)
    arrow(ax, (4.5, 4.5), (6.1, 4.5), color=ORANGE)
    txt(ax, .1, 1.6, "4 level 3D U-Net: 32〜256ch、InstanceNorm / SiLU\n核間隔を見て入力を縮小し、再検出する", size=10.5)
    txt(ax, .1, .2, "大きな核の二重検出を、無条件に平均する方法ではない。", size=10)
    ax = axes[1]
    panel(ax, "B", "1回目の対応からvelocityを求め、2回目の割当へ")
    b.matrix(ax, np.array([[.91, .22, .11], [.28, .82, .53]]), ["C", "D", "E"], ["A", "B"], extent=(3.5, 9.1, 2.6, 5.6))
    tree(ax, 1.1, 5.6, scale=.55)
    txt(ax, .1, 3.6, "47特徴\n400 trees\nHistGB", size=10.5)
    txt(ax, .1, 1.4, "対応probability → birth/death付きLAP\n1回目: motion推定 → 2回目: velocity・local flowを利用", size=10.8)
    txt(ax, .1, .2, "通常辺の線形割当。分裂は38特徴の別モデルで追加。", size=10.2)
    ax = axes[2]
    panel(ax, "C", "2娘の間に暗い隙間が、2時点とも存在するか")
    x = np.linspace(0, 10, 160)
    two = 3.2 + 1.9 * (np.exp(-(x - 2.5) ** 2 / .9) + np.exp(-(x - 7.5) ** 2 / .9))
    one = .7 + 1.9 * np.exp(-(x - 5) ** 2 / 12)
    ax.plot(x, two, color=GREEN, lw=2.8)
    ax.plot(x, one, color=RED, lw=2.8)
    for y, colour in [(3.2, GREEN), (2.6, RED)]:
        dot(ax, (5.0, y), color=colour, size=100)
    txt(ax, .1, 6.3, "t+1 と t+2: 両端peakに対する中央の明るさ", size=10.5)
    txt(ax, .1, .15, "深い谷: 分裂を支持 / 谷が浅い: 同じ核の二重検出を疑う\nCNNの分裂確率ではなく、raw画像から直接計算する。", size=10.2)
    ax = axes[3]
    panel(ax, "D", "修正の正誤は、graph変更による評価差で学ぶ")
    for y in [4.9, 3.8]:
        for i in range(3):
            dot(ax, (.8 + i * 1.3, y), color=GREY, size=130)
    for i in range(3):
        dot(ax, (6.0 + i * 1.3, 4.4), color=GREEN, size=130)
    arrow(ax, (3.8, 4.4), (5.7, 4.4), color=ORANGE)
    txt(ax, .1, 6.3, "例: 平行している重複trackを1本へまとめる", size=11)
    txt(ax, .1, 2.3, "学習時だけ: 変更前後をGTで採点\nTP増加とFP増加、点数の変化から有益かを決める", size=10.5)
    txt(ax, .1, .7, "重複修正: 22特徴 / gap補完: 49特徴 / edge cut・relink\n提出時は特徴をモデルへ入力し、GTは参照しない。", size=10.5)
    save(fig, "17_assignment_metric_edits", 17, 744930,
         ["4-level 32-256ch U-Nets with adaptive input scaling", "47-feature HistGradientBoosting and two augmented LAP passes", "38-feature division scorer plus two-frame intensity valley", "graph-edit labels from official metric effect", "22-feature duplicate and 49-feature gap models"])


DRAW = {1: first, 2: second, 4: fourth, 7: seventh, 9: ninth, 10: tenth,
        11: eleventh, 16: sixteenth, 17: seventeenth}


def contact_sheet():
    old = json.loads((ROOT / "docs/images/biohub_top_solutions_20260930/manifest.json").read_text())
    old_records = old["figures"] if isinstance(old, dict) else old
    all_records = sorted(old_records + b.RECORDS, key=lambda item: item["rank"])
    fig, axes = plt.subplots(5, 3, figsize=(18, 19.5))
    for ax, record in zip(axes.flat, all_records, strict=True):
        ax.imshow(plt.imread(ROOT / record["png"]))
        ax.axis("off")
    fig.suptitle("Biohub: 公開上位15解法の説明図（詳細は個別画像）", fontsize=20, y=.989)
    fig.subplots_adjust(left=.015, right=.985, top=.96, bottom=.008, wspace=.025, hspace=.04)
    fig.savefig(OUT / "00_contact_sheet.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", nargs="*", type=int, default=list(DRAW))
    args = parser.parse_args()
    b.setup()
    manifest_path = OUT / "manifest.json"
    previous = json.loads(manifest_path.read_text())["figures"] if manifest_path.exists() else []
    for rank in args.ranks:
        DRAW[rank]()
    merged = {item["rank"]: item for item in previous + b.RECORDS}
    b.RECORDS[:] = sorted(merged.values(), key=lambda item: item["rank"])
    manifest_path.write_text(json.dumps({"created_for": "2026-10-02 report update", "figures": b.RECORDS}, ensure_ascii=False, indent=2) + "\n")
    if len(b.RECORDS) == 9:
        contact_sheet()


if __name__ == "__main__":
    main()
