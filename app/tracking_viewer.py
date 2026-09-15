"""Run with make tracking-app. The Kaggle notebook is the primary large-data UI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from app.tracking_data import (
    analyze_sequence,
    image_array,
    load_candidates,
    load_geff,
    load_prediction_graph,
    read_projection,
)
from app.tracking_plots import detail_table, spatial_figure, timeline_figure
from scripts.config_utils import load_project_config, project_path

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(max_entries=4)
def cached_candidates(path, signature):
    return load_candidates(Path(path))


@st.cache_data(max_entries=4)
def cached_gt(path, scale, signature):
    return load_geff(Path(path), scale)


@st.cache_data(max_entries=4)
def cached_prediction(path, scale, signature):
    return load_prediction_graph(Path(path), scale)


@st.cache_data(max_entries=6)
def cached_analysis(detections, gt, frames, radius):
    return analyze_sequence(detections, gt, frames, radius)


@st.cache_data(max_entries=6)
def cached_projection(path, t, plane, bounds):
    return read_projection(Path(path), t, plane, bounds)


def main():
    st.set_page_config(page_title="Cell tracking viewer", layout="wide")
    st.title("Cell tracking · 検出と正解の時間変化")
    st.caption("正解の軌跡・分裂と検出候補を、同じ時刻・同じ物理座標で確認します。")
    config = yaml.safe_load((ROOT / "app/tracking_viewer.yaml").read_text())
    project = load_project_config()
    experiments = project_path(project, "paths.experiments_dir")
    roots = sorted(experiments.glob("*/artifacts/*/window_cache"), reverse=True)
    cache_root = Path(
        st.sidebar.text_input("検出キャッシュのルート", str(roots[0]) if roots else "")
    )
    train_root = Path(
        st.sidebar.text_input("正解・画像の train", str(project_path(project, "data.train_dir")))
    )
    prediction_roots = sorted(experiments.glob("*/artifacts/*/oracle_final_graphs"), reverse=True)
    prediction_root_text = st.sidebar.text_input(
        "最終予測 graph のルート",
        str(prediction_roots[0]) if prediction_roots else "",
    )
    folders = sorted(p for p in cache_root.glob("*") if p.is_dir() and any(p.glob("*.npz")))
    if not folders:
        st.info("exp015 の window_cache を指定してください。Kaggle 版は app/README.md を参照。")
        return
    folder = st.sidebar.selectbox("サンプル", folders, format_func=lambda p: p.name)
    signature = tuple(
        (p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(folder.glob("*.npz"))
    )
    try:
        detections, receipt = cached_candidates(str(folder), signature)
        gt_path = train_root / f"{folder.name}.geff"
        scale = tuple(config["voxel_scale_um"])
        if gt_path.is_dir():
            gt_signature = tuple(
                (str(p.relative_to(gt_path)), p.stat().st_mtime_ns, p.stat().st_size)
                for p in sorted(gt_path.rglob("*"))
                if p.is_file()
            )
            gt, edges = cached_gt(str(gt_path), scale, gt_signature)
        else:
            st.info(
                "正解 GEFF はローカルにありません。検出位置のみ表示中です。"
                "正解・元画像との比較には Kaggle Notebook 版を使ってください。"
            )
            gt = pd.DataFrame(columns=["node_id", "t", "z", "y", "x", "lineage", "division"])
            edges = pd.DataFrame(columns=["source_id", "target_id"])
        prediction_path = (
            Path(prediction_root_text) / f"{folder.name}.npz" if prediction_root_text else None
        )
        if prediction_path is not None and prediction_path.is_file():
            prediction_nodes, prediction_edges = cached_prediction(
                str(prediction_path),
                scale,
                (prediction_path.stat().st_mtime_ns, prediction_path.stat().st_size),
            )
        else:
            if prediction_root_text:
                st.info(f"最終予測 graph がありません: {prediction_path}")
            prediction_nodes = pd.DataFrame(
                columns=["node_id", "t", "z", "y", "x", "lineage", "division"]
            )
            prediction_edges = pd.DataFrame(columns=["source_id", "target_id"])
        radius = st.sidebar.number_input(
            "対応距離の上限 (µm)", min_value=0.01, value=config["matching_radius_um"]
        )
        minimum_score = st.sidebar.slider(
            "検出 score の下限", 0.0, 1.0, config["minimum_detection_score"], 0.01
        )
        filtered = detections[detections.score >= minimum_score]
        matches, summary = cached_analysis(filtered, gt, receipt["frames"], radius)
        st.sidebar.caption(
            "同時刻の3次元距離から1対1で割当。表示範囲・系譜の選択は計算を変えません。"
        )
        plane = st.sidebar.radio("表示", ["XY", "XZ", "YZ", "3D"], horizontal=True)
        lineage = st.sidebar.selectbox(
            "正解の系譜",
            [None, *sorted(gt.lineage.unique())],
            format_func=lambda x: "すべて" if x is None else str(x),
        )
        trail = st.sidebar.slider("軌跡の過去 frame 数", 0, 30, config["trail_frames"])
        show_unmatched = st.sidebar.checkbox("未対応の検出を表示", True)
        show_links = st.sidebar.checkbox("対応線を表示", True)
        show_predictions = st.sidebar.checkbox(
            "予測軌跡を表示", bool(len(prediction_nodes)), disabled=prediction_nodes.empty
        )
        image_path = train_root / f"{folder.name}.zarr"
        shape = image_array(image_path).shape if image_path.exists() else None
        depth = shape[1] if shape else max(1, int(detections.z.max() / scale[0]) + 1)
        bounds = (
            st.sidebar.slider("Z 断面 (voxel、両端を含む)", 0, depth - 1, (0, depth - 1))
            if depth > 1
            else (0, 0)
        )
        show_image = st.sidebar.checkbox("画像を重ねる", shape is not None, disabled=shape is None)
        playing = st.sidebar.toggle("自動再生", False)
        frames = receipt["frames"]
        if st.session_state.get("sample") != str(folder):
            st.session_state.sample, st.session_state.frame = str(folder), frames[0]
        st.warning(
            "正解は疎です。未対応の検出は誤検出を意味しません。候補 ID は軌跡の ID ではありません。"
        )

        @st.fragment(run_every=config["playback_interval_seconds"] if playing else None)
        def render():
            if playing:
                index = frames.index(st.session_state.frame)
                st.session_state.frame = frames[(index + 1) % len(frames)]
            a, b, c = st.columns([1, 1, 6])
            if a.button("← 前"):
                index = frames.index(st.session_state.frame)
                st.session_state.frame = frames[max(0, index - 1)]
            if b.button("次 →"):
                index = frames.index(st.session_state.frame)
                st.session_state.frame = frames[min(len(frames) - 1, index + 1)]
            t = c.select_slider("時刻 t", options=frames, key="frame")
            row = summary[summary.t == t].iloc[0]
            metric_columns = st.columns(5)
            for column, label in zip(metric_columns[:2], ["検出候補", "正解"], strict=True):
                column.metric(label, int(row[label]))
            metric_columns[2].metric("最終予測", int((prediction_nodes.t == t).sum()))
            for column, label in zip(metric_columns[3:], ["対応あり", "未対応の正解"], strict=True):
                column.metric(label, int(row[label]))
            background = (
                cached_projection(str(image_path), t, plane, bounds)
                if show_image and plane != "3D"
                else None
            )
            figure = spatial_figure(
                filtered,
                gt,
                edges,
                matches,
                t=t,
                plane=plane,
                lineage=lineage,
                trail=trail,
                z_bounds=((bounds[0] - 0.5) * scale[0], (bounds[1] + 0.5) * scale[0]),
                background=background,
                scale=scale,
                image_z_start=bounds[0],
                show_unmatched=show_unmatched,
                show_links=show_links,
                dataset=folder.name,
                predicted_nodes=prediction_nodes,
                predicted_edges=prediction_edges,
                show_predictions=show_predictions,
            )
            st.plotly_chart(figure, width="stretch", key="spatial")
            st.caption(
                "件数は全視野・全Z。紫は正解、橙は予測、青緑は正解edgeとの対応を"
                "確認できた予測、黄色の破線は同時刻の対応です。"
            )
            if not gt.empty:
                st.plotly_chart(
                    timeline_figure(summary, t, prediction_nodes),
                    width="stretch",
                    key="timeline",
                )
                table = detail_table(gt, filtered, matches, lineage)
                st.dataframe(table if lineage is not None else table[table.t == t], hide_index=True)

        render()
        if not gt.empty:
            st.download_button(
                "全時刻の対応表を CSV 保存",
                detail_table(gt, filtered, matches).to_csv(index=False),
                file_name=f"tracking_matches_{folder.name}.csv",
                mime="text/csv",
            )
        with st.expander("読み込み情報・対応付け設定"):
            st.code(
                json.dumps(
                    {
                        **receipt,
                        "radius_um": radius,
                        "minimum_score": minimum_score,
                        "voxel_scale_um": scale,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                language="json",
            )
    except (ValueError, KeyError, OSError, IndexError) as exc:
        st.error(f"入力データを確認してください: {exc}")


if __name__ == "__main__":
    main()
