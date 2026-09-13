"""Package the shared viewer modules into a standalone, private Kaggle notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "app/kaggle"
MODULES = ["tracking_data.py", "tracking_plots.py", "tracking_notebook.py", "tracking_viewer.yaml"]


def build():
    files = {name: (ROOT / "app" / name).read_text() for name in MODULES}
    intro = """# Cell tracking — 検出位置と正解の時間変化

exp014 の検出候補と、同名の train GEFF / Zarr を比較する EDA ビューアーです。
**Edit を開いて Run All** すると、時刻・Z断面・正解の系譜を操作できます。
保存された静的な Notebook ページでは Python の操作部は動きません。

- CPU / Internet OFF で実行。画像は選択時刻だけ読み、ローカルへダウンロードしません。
- Input: `biohub-cell-tracking-during-development` と
  `kentookumura/exp014-exact-window-cache-inference` の出力。
- 正解は疎です。未対応の検出は誤検出とは限りません。
- 候補 ID は各検出点の識別子です。同一細胞を時間方向につなぐ ID ではありません。
- 公開 test と同名 train の比較は EDA 用です。汎化性能の評価ではありません。
- 対応表の保存ボタンは作業ディレクトリに CSV と設定 JSON を作ります。
"""
    setup = (
        """from pathlib import Path
import sys
import json

BASE = Path('/kaggle/working') if Path('/kaggle/input').exists() else Path.cwd()
PACKAGE = BASE / 'tracking_viewer_app'
(PACKAGE / 'app').mkdir(parents=True, exist_ok=True)
(PACKAGE / 'app' / '__init__.py').write_text('')
"""
        + f"FILES = {files!r}\n"
        + """for filename, source in FILES.items():
    (PACKAGE / 'app' / filename).write_text(source, encoding='utf-8')
sys.path.insert(0, str(PACKAGE))
import subprocess
# Existing support input supplies Zarr wheels; never access the package index.
INPUT_ROOT = Path("/kaggle/input")
wheel_dirs = []
for depth in range(6):
    wheel_dirs.extend(p for p in INPUT_ROOT.glob("*/" * depth + "wheels") if p.is_dir())
if not wheel_dirs:
    raise FileNotFoundError("Attach pilkwang/biohub-tracking-support-pack-50ep-v1")
command = [sys.executable, "-m", "pip", "install", "--no-index", "zarr>=3.0.10,<4"]
for directory in sorted(wheel_dirs):
    command.extend(["--find-links", str(directory)])
subprocess.run(command, check=True)
import numpy, pandas, scipy, plotly, ipywidgets, zarr
print({m.__name__: m.__version__ for m in [numpy, pandas, scipy, plotly, ipywidgets, zarr]})
"""
    )
    paths = """import yaml
from app.tracking_notebook import TrackingNotebookViewer, find_input_directory

# 自動検出が曖昧な場合だけ、下の2行を明示パスへ変更してください。
INPUT = Path('/kaggle/input')
CACHE_ROOT = find_input_directory(INPUT, 'window_cache')
TRAIN_ROOT = find_input_directory(INPUT, 'train')
CONFIG = yaml.safe_load((PACKAGE / 'app' / 'tracking_viewer.yaml').read_text())
print('cache:', CACHE_ROOT)
print('train:', TRAIN_ROOT)
print('settings:', CONFIG)
"""
    run = """import plotly.io as pio
# Kaggle 推奨の表示方式。描画ライブラリはブラウザー側で読み込みます。
pio.renderers.default = 'kaggle'
import faulthandler
faulthandler.dump_traceback_later(90)
try:
    print("Loading first image and viewer...", flush=True)
    viewer = TrackingNotebookViewer(CACHE_ROOT, TRAIN_ROOT, CONFIG)
    viewer.show()
    print("Initial viewer ready", flush=True)
finally:
    faulthandler.cancel_dump_traceback_later()
"""
    checks = """# batch 保存時も入力・対応計算・初期表示の到達点を確認できます。
import json
from app.tracking_plots import detail_table
receipt = {**viewer.receipt, 'gt_nodes': len(viewer.gt), 'gt_edges': len(viewer.edges),
           'matched_gt': int(viewer.matches.detection_id.notna().sum()),
           'image_shape': list(viewer.shape) if viewer.shape else None,
           'matching_radius_um': viewer.radius.value, 'diagnostic_only': True}
Path('viewer_input_check.json').write_text(json.dumps(receipt, indent=2))
initial_matches = detail_table(viewer.gt, viewer.filtered, viewer.matches)
initial_matches.to_csv('viewer_initial_matches.csv', index=False)
print(receipt)
# 静的ページにも初期フレームを残します。操作時は Edit → Run All を使ってください。
display(viewer.figure)
"""
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(intro),
            nbformat.v4.new_code_cell(setup),
            nbformat.v4.new_code_cell(paths),
            nbformat.v4.new_code_cell(run),
            nbformat.v4.new_code_cell(checks),
        ]
    )
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    # Stable cell IDs make rebuilding the package reviewable.
    for cell, cell_id in zip(
        notebook.cells, ["intro", "modules", "inputs", "viewer", "check"], strict=True
    ):
        cell.id = cell_id
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT / "tracking_viewer.ipynb")
    metadata = {
        "id": "kentookumura/biohub-tracking-eda-viewer",
        "title": "Biohub Tracking EDA Viewer",
        "code_file": "tracking_viewer.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "run_on_push": True,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": ["kentookumura/exp014-exact-window-cache-inference"],
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
        "model_sources": [],
    }
    (OUTPUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(OUTPUT / "tracking_viewer.ipynb")


if __name__ == "__main__":
    build()
