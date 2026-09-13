"""Interactive, CPU-only Kaggle/Jupyter entry point."""

from __future__ import annotations

import html
import json
from functools import lru_cache
from pathlib import Path

import ipywidgets as widgets
from IPython.display import HTML, clear_output, display

from app.tracking_data import (
    analyze_sequence,
    image_array,
    load_candidates,
    load_geff,
    read_projection,
)
from app.tracking_plots import detail_table, spatial_figure, timeline_figure


def find_input_directory(root: Path, name: str) -> Path:
    matches = []
    for depth in range(5):
        matches.extend(p for p in root.glob("*/" * depth + name) if p.is_dir())
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(f"Expected one {name} directory; found {unique}. Set the path explicitly.")
    return unique[0]


class TrackingNotebookViewer:
    def __init__(self, cache_root: Path, train_root: Path, config: dict):
        self.cache_root, self.train_root, self.config = cache_root, train_root, config
        datasets = sorted(
            p.name
            for p in cache_root.iterdir()
            if p.is_dir() and any(p.glob("*.npz")) and (train_root / f"{p.name}.geff").is_dir()
        )
        if not datasets:
            raise ValueError("No matching cache/GEFF pairs. Attach competition and exp014 outputs.")
        style = {"description_width": "initial"}
        train_count = sum(path.is_dir() for path in train_root.glob("*.geff"))
        self.dataset = widgets.Dropdown(
            options=datasets, description=f"サンプル ({len(datasets)}件)", style=style
        )
        self.dataset_note = widgets.HTML(
            value=(
                f"<small>検出キャッシュあり: {len(datasets)}件 / "
                f"正解 train: {train_count}件。"
                "このビューアーでは検出キャッシュがあるサンプルだけを選べます。</small>"
            )
        )
        self.frame = widgets.SelectionSlider(
            options=[0], description="時刻 t", continuous_update=False
        )
        self.play = widgets.Play(
            min=0,
            max=0,
            interval=int(config["playback_interval_seconds"] * 1000),
            layout=widgets.Layout(display="none"),
        )
        self.previous_frame = widgets.Button(description="前の時刻")
        self.play_pause = widgets.Button(description="再生")
        self.next_frame = widgets.Button(description="次の時刻")
        self.plane = widgets.Dropdown(options=["XY", "XZ", "YZ", "3D"], description="表示")
        self.lineage = widgets.Dropdown(
            options=[("すべて", -1)], description="正解の系譜", style=style
        )
        self.node = widgets.Dropdown(options=[("選択なし", -1)], description="正解 ID", style=style)
        self.zrange = widgets.IntRangeSlider(
            value=[0, 63], min=0, max=63, description="Z断面", continuous_update=False
        )
        self.radius = widgets.BoundedFloatText(
            value=config["matching_radius_um"],
            min=0.01,
            max=1000.0,
            description="対応距離 µm",
            style=style,
        )
        self.score = widgets.FloatSlider(
            value=config["minimum_detection_score"],
            min=0,
            max=1,
            step=0.01,
            description="検出score ≥",
            style=style,
            continuous_update=False,
        )
        self.trail = widgets.IntSlider(
            value=config["trail_frames"],
            min=0,
            max=30,
            description="軌跡の過去frame数",
            style=style,
            continuous_update=False,
        )
        self.background = widgets.Checkbox(value=True, description="画像を重ねる")
        self.unmatched = widgets.Checkbox(value=True, description="未対応の検出を表示")
        self.links = widgets.Checkbox(value=True, description="対応線を表示")
        self.prev_issue = widgets.Button(description="前の未対応時刻")
        self.next_issue = widgets.Button(description="次の未対応時刻")
        self.export = widgets.Button(description="対応表を CSV 保存")
        self.message = widgets.HTML()
        self.output = widgets.Output()
        self.busy = False
        self._projection = lru_cache(maxsize=6)(read_projection)
        self.dataset.observe(self._load, "value")
        for w in [self.radius, self.score]:
            w.observe(self._analyze, "value")
        for w in [
            self.frame,
            self.plane,
            self.lineage,
            self.zrange,
            self.trail,
            self.background,
            self.unmatched,
            self.links,
        ]:
            w.observe(self._render, "value")
        self.node.observe(self._select_node, "value")
        self.play.observe(self._play_frame, "value")
        self.play.observe(self._sync_play_button, "playing")
        self.previous_frame.on_click(lambda _: self._step_frame(-1))
        self.play_pause.on_click(self._toggle_play)
        self.next_frame.on_click(lambda _: self._step_frame(1))
        self.prev_issue.on_click(lambda _: self._jump(-1))
        self.next_issue.on_click(lambda _: self._jump(1))
        self.export.on_click(self._export)
        self._load()

    def _sync_play_button(self, change):
        self.play_pause.description = "一時停止" if change["new"] else "再生"

    def _toggle_play(self, _=None):
        self.play.playing = not self.play.playing

    def _step_frame(self, direction):
        self.play.playing = False
        index = self.frames.index(self.frame.value)
        target = min(max(index + direction, 0), len(self.frames) - 1)
        self.frame.value = self.frames[target]

    def _load(self, _=None):
        self.busy = True
        try:
            self.play.playing = False
            name = self.dataset.value
            self.detections, self.receipt = load_candidates(self.cache_root / name)
            self.gt, self.edges = load_geff(
                self.train_root / f"{name}.geff", tuple(self.config["voxel_scale_um"])
            )
            self.frames = self.receipt["frames"]
            self.frame.options = self.frames
            self.frame.value = self.frames[0]
            self.play.value, self.play.max = 0, len(self.frames) - 1
            self.image_path = self.train_root / f"{name}.zarr"
            self.shape = image_array(self.image_path).shape if self.image_path.exists() else None
            self.background.disabled = self.shape is None
            self.background.value = self.shape is not None
            depth = (
                self.shape[1]
                if self.shape
                else max(1, int(self.gt.z.max() / self.config["voxel_scale_um"][0]) + 1)
            )
            self.zrange.value = (0, 0)
            self.zrange.max = depth - 1
            self.zrange.value = (0, depth - 1)
            self.lineage.options = [("すべて", -1)] + [
                (str(n), int(n)) for n in sorted(self.gt.lineage.unique())
            ]
            self.lineage.value = -1
            self.node.options = [("選択なし", -1)] + [
                (f"{r.node_id} · t={r.t}" + (" · 分裂" if r.division else ""), int(r.node_id))
                for r in self.gt.itertuples()
            ]
            self.node.value = -1
            self._projection.cache_clear()
        finally:
            self.busy = False
        self._analyze()

    def _analyze(self, _=None):
        if self.busy:
            return
        self.filtered = self.detections[self.detections.score >= self.score.value]
        self.matches, self.summary = analyze_sequence(
            self.filtered, self.gt, self.frames, self.radius.value
        )
        self._render()

    def _play_frame(self, change):
        if not self.busy:
            self.frame.value = self.frames[change["new"]]

    def _select_node(self, _=None):
        if self.busy or self.node.value == -1:
            return
        row = self.gt[self.gt.node_id == self.node.value].iloc[0]
        self.busy = True
        self.lineage.value = int(row.lineage)
        if int(row.t) in self.frames:
            self.frame.value = int(row.t)
        self.busy = False
        self._render()

    def _jump(self, direction):
        missing = self.matches[self.matches.detection_id.isna()]
        if self.lineage.value != -1:
            missing = missing[
                missing.gt_id.isin(self.gt[self.gt.lineage == self.lineage.value].node_id)
            ]
        times = sorted(set(missing.t))
        eligible = [t for t in times if (t - self.frame.value) * direction > 0]
        if eligible:
            self.frame.value = eligible[0] if direction > 0 else eligible[-1]

    def _export(self, _=None):
        path = Path.cwd() / f"tracking_matches_{self.dataset.value}.csv"
        detail_table(self.gt, self.filtered, self.matches).to_csv(path, index=False)
        receipt = {
            **self.receipt,
            "radius_um": self.radius.value,
            "minimum_score": self.score.value,
            "voxel_scale_um": self.config["voxel_scale_um"],
            "diagnostic_only": True,
        }
        path.with_suffix(".json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        self.message.value = f"保存: {html.escape(str(path))}（同名 JSON に設定を保存）"

    def _render(self, _=None):
        if self.busy:
            return
        # Keep resume playback at the time selected by the slider or a GT node.
        self.busy = True
        self.play.value = self.frames.index(self.frame.value)
        self.busy = False
        with self.output:
            clear_output(wait=True)
            try:
                t, plane = self.frame.value, self.plane.value
                scale = tuple(self.config["voxel_scale_um"])
                lo, hi = self.zrange.value
                bounds = ((lo - 0.5) * scale[0], (hi + 0.5) * scale[0])
                background = None
                if self.background.value and plane != "3D" and self.shape and t < self.shape[0]:
                    background = self._projection(self.image_path, t, plane, (lo, hi))
                row = self.summary[self.summary.t == t].iloc[0]
                display(
                    HTML(
                        f"<b>t={t}</b>　検出 {int(row['検出候補'])}　正解 {int(row['正解'])}　"
                        f"対応 {int(row['対応あり'])}　未対応の正解 {int(row['未対応の正解'])}"
                        "<br><small>件数は全視野・全Z。表示範囲や系譜の選択は対応計算を変えません。</small>"
                    )
                )
                lineage = None if self.lineage.value == -1 else self.lineage.value
                self.figure = spatial_figure(
                    self.filtered,
                    self.gt,
                    self.edges,
                    self.matches,
                    t=t,
                    plane=plane,
                    lineage=lineage,
                    trail=self.trail.value,
                    z_bounds=bounds,
                    background=background,
                    scale=scale,
                    image_z_start=lo,
                    show_unmatched=self.unmatched.value,
                    show_links=self.links.value,
                    dataset=self.dataset.value,
                )
                display(self.figure)
                display(timeline_figure(self.summary, t))
                details = detail_table(self.gt, self.filtered, self.matches, lineage)
                columns = [
                    "t",
                    "gt_id",
                    "detection_id",
                    "distance_um",
                    "nearest_distance_um",
                    "candidates_in_radius",
                    "detection_score",
                    "division",
                    "state",
                ]
                display(
                    details[columns] if lineage is not None else details[details.t == t][columns]
                )
            except (ValueError, KeyError, OSError, IndexError) as exc:
                display(HTML(f"<b>読み込み・表示エラー:</b> {html.escape(str(exc))}"))
                raise

    def show(self):
        display(
            HTML(
                "<h2>Cell tracking · 検出と正解の時間変化</h2>"
                "<p>正解は疎です。未対応の検出は誤検出を意味しません。"
                "対応は同時刻・3次元の距離による1対1割当で、公式スコアではありません。"
                "候補IDは各検出点のIDで、細胞の軌跡IDではありません。</p>"
            )
        )
        display(
            widgets.VBox(
                [
                    widgets.HBox([self.dataset, self.plane]),
                    self.dataset_note,
                    self.play,
                    widgets.HBox(
                        [self.previous_frame, self.play_pause, self.next_frame, self.frame]
                    ),
                    widgets.HBox([self.prev_issue, self.next_issue]),
                    widgets.HBox([self.lineage, self.node]),
                    widgets.HBox([self.zrange, self.trail]),
                    widgets.HBox([self.radius, self.score]),
                    widgets.HBox([self.background, self.unmatched, self.links, self.export]),
                    self.message,
                    self.output,
                ]
            )
        )
        return self
