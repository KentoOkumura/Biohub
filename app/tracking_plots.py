"""Plotly views shared by the local app and the Kaggle notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.tracking_data import XYZ

COLORS = {
    "gt": "#4ce3b0",
    "miss": "#ff7889",
    "det": "#76b7ff",
    "other": "#8c97ab",
    "link": "#f5cd73",
    "trail": "#bc98ff",
    "prediction": "#ff9f43",
    "prediction_supported": "#2dd4bf",
    "prediction_division": "#ffd166",
}


def spatial_figure(
    detections,
    gt,
    edges,
    matches,
    *,
    t,
    plane="XY",
    lineage=None,
    trail=5,
    z_bounds=None,
    background=None,
    scale=(1.625, 0.40625, 0.40625),
    image_z_start=0,
    show_unmatched=True,
    show_links=True,
    dataset="",
    predicted_nodes=None,
    predicted_edges=None,
    prediction_matches=None,
    show_predictions=True,
):
    if predicted_nodes is None:
        predicted_nodes = pd.DataFrame(columns=["node_id", "t", *XYZ, "lineage", "division"])
    if predicted_edges is None:
        predicted_edges = pd.DataFrame(columns=["source_id", "target_id"])
    if prediction_matches is None:
        prediction_matches = pd.DataFrame(columns=["gt_id", "detection_id", "t"])
    det = detections[detections.t == t].copy()
    now = gt[gt.t == t].copy()
    prediction_now = predicted_nodes[predicted_nodes.t == t].copy()
    pairs = matches[matches.t == t].dropna(subset=["detection_id"])
    matched_gt, matched_det = set(pairs.gt_id), set(pairs.detection_id)
    if z_bounds is not None:
        det = det[det.z.between(*z_bounds)]
        now = now[now.z.between(*z_bounds)]
        prediction_now = prediction_now[prediction_now.z.between(*z_bounds)]
    prediction_lineages = None
    if lineage is not None:
        now = now[now.lineage == lineage]
        selected_gt_ids = set(gt.loc[gt.lineage == lineage, "node_id"])
        selected_prediction_ids = set(
            prediction_matches.loc[
                prediction_matches.gt_id.isin(selected_gt_ids)
                & prediction_matches.detection_id.notna(),
                "detection_id",
            ].astype(np.int64)
        )
        prediction_lineages = set(
            predicted_nodes.loc[predicted_nodes.node_id.isin(selected_prediction_ids), "lineage"]
        )
        prediction_now = prediction_now[prediction_now.lineage.isin(prediction_lineages)]
    is3d = plane == "3D"
    horizontal, vertical = {"XY": ("x", "y"), "XZ": ("x", "z"), "YZ": ("y", "z"), "3D": ("x", "y")}[
        plane
    ]
    figure = go.Figure()
    if background is not None and not is3d:
        high = float(np.percentile(background, 99.7))
        axis_scale = dict(zip(XYZ, scale, strict=True))
        figure.add_trace(
            go.Heatmap(
                z=background,
                x=np.arange(background.shape[1]) * axis_scale[horizontal],
                y=(np.arange(background.shape[0]) + (image_z_start if vertical == "z" else 0))
                * axis_scale[vertical],
                colorscale="Gray",
                zmin=0,
                zmax=max(high, 1),
                showscale=False,
                hoverinfo="skip",
                name="画像",
            )
        )

    def points(frame, name, color, symbol, size, kind):
        if frame.empty:
            return
        custom = [
            [kind, int(row.node_id), int(row.t), float(row.z), float(row.y), float(row.x)]
            for row in frame.itertuples()
        ]
        kwargs = dict(
            x=frame[horizontal],
            y=frame[vertical],
            mode="markers",
            name=name,
            marker=dict(size=size, color=color, symbol=symbol, line=dict(width=1)),
            customdata=custom,
            hovertemplate=(
                "ID %{customdata[1]} · t=%{customdata[2]}<br>"
                "z=%{customdata[3]:.2f} y=%{customdata[4]:.2f} x=%{customdata[5]:.2f} µm"
                "<extra>%{fullData.name}</extra>"
            ),
        )
        if is3d:
            kwargs["z"] = frame.z
            figure.add_trace(go.Scatter3d(**kwargs))
        else:
            figure.add_trace(go.Scatter(**kwargs))

    def lines(segments, name, color, width, dash="solid"):
        if not segments:
            return
        points_array = np.array([p for a, b in segments for p in [a, b, [np.nan] * 3]])
        positions = dict(zip(XYZ, points_array.T, strict=True))
        kwargs = dict(
            x=positions[horizontal],
            y=positions[vertical],
            mode="lines",
            name=name,
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="skip",
        )
        if is3d:
            figure.add_trace(go.Scatter3d(z=positions["z"], **kwargs))
        else:
            figure.add_trace(go.Scatter(**kwargs))

    history = gt[gt.t.between(t - trail, t)]
    if lineage is not None:
        history = history[history.lineage == lineage]
    if z_bounds is not None:
        history = history[history.z.between(*z_bounds)]
    by_id = history.set_index("node_id")
    segments = [
        (by_id.loc[s, XYZ].to_numpy(), by_id.loc[d, XYZ].to_numpy())
        for s, d in edges.itertuples(index=False, name=None)
        if s in by_id.index and d in by_id.index
    ]
    lines(segments, "正解の軌跡・分裂", COLORS["trail"], 3)
    prediction_history = predicted_nodes[predicted_nodes.t.between(t - trail, t)]
    if prediction_lineages is not None:
        prediction_history = prediction_history[
            prediction_history.lineage.isin(prediction_lineages)
        ]
    if z_bounds is not None:
        prediction_history = prediction_history[prediction_history.z.between(*z_bounds)]
    prediction_by_id = prediction_history.set_index("node_id")
    gt_edge_set = set(edges.itertuples(index=False, name=None))
    gt_by_detection = {
        int(row.detection_id): int(row.gt_id)
        for row in prediction_matches.dropna(subset=["detection_id"]).itertuples()
    }
    supported_prediction_segments = []
    other_prediction_segments = []
    for source, target in predicted_edges.itertuples(index=False, name=None):
        if source not in prediction_by_id.index or target not in prediction_by_id.index:
            continue
        segment = (
            prediction_by_id.loc[source, XYZ].to_numpy(),
            prediction_by_id.loc[target, XYZ].to_numpy(),
        )
        mapped_edge = (gt_by_detection.get(int(source)), gt_by_detection.get(int(target)))
        destination = (
            supported_prediction_segments
            if mapped_edge in gt_edge_set
            else other_prediction_segments
        )
        destination.append(segment)
    if show_predictions:
        lines(
            other_prediction_segments,
            "その他の予測軌跡",
            COLORS["prediction"],
            2,
        )
        lines(
            supported_prediction_segments,
            "正解edgeとの対応を確認できた予測軌跡",
            COLORS["prediction_supported"],
            3,
        )
    if show_links:
        g_index, d_index = now.set_index("node_id"), det.set_index("node_id")
        segments = [
            (g_index.loc[r.gt_id, XYZ].to_numpy(), d_index.loc[int(r.detection_id), XYZ].to_numpy())
            for r in pairs.itertuples()
            if r.gt_id in g_index.index and int(r.detection_id) in d_index.index
        ]
        lines(segments, "同時刻の対応", COLORS["link"], 2, "dash")
    if show_unmatched:
        points(
            det[~det.node_id.isin(matched_det)],
            "未対応の検出",
            COLORS["other"],
            "circle-open",
            5 if is3d else 8,
            "detection",
        )
    points(
        det[det.node_id.isin(matched_det)],
        "対応ありの検出",
        COLORS["det"],
        "circle-open",
        6 if is3d else 12,
        "detection",
    )
    points(
        now[now.node_id.isin(matched_gt)],
        "対応ありの正解",
        COLORS["gt"],
        "circle",
        4 if is3d else 7,
        "gt",
    )
    points(
        now[~now.node_id.isin(matched_gt)],
        "未対応の正解",
        COLORS["miss"],
        "diamond",
        5 if is3d else 10,
        "gt",
    )
    if "division" in now:
        points(now[now.division], "正解の分裂親", COLORS["trail"], "diamond-open", 12, "gt")
    if show_predictions:
        points(
            prediction_now,
            "最終予測で選択された細胞",
            COLORS["prediction"],
            "square-open",
            7 if is3d else 13,
            "prediction",
        )
        points(
            prediction_now[prediction_now.division],
            "予測の分裂親",
            COLORS["prediction_division"],
            "diamond-open",
            9 if is3d else 16,
            "prediction",
        )
    all_coords = pd.concat([detections[XYZ], gt[XYZ], predicted_nodes[XYZ]])
    ranges = {
        axis: [
            min(0.0, float(all_coords[axis].min()) - 2),
            max(1.0, float(all_coords[axis].max()) + 2),
        ]
        for axis in XYZ
    }
    if background is not None and not is3d:
        axis_scale = dict(zip(XYZ, scale, strict=True))
        ranges[horizontal] = [
            -axis_scale[horizontal] / 2,
            (background.shape[1] - 0.5) * axis_scale[horizontal],
        ]
        offset = image_z_start if vertical == "z" else 0
        ranges[vertical] = [
            (offset - 0.5) * axis_scale[vertical],
            (offset + background.shape[0] - 0.5) * axis_scale[vertical],
        ]
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        height=620,
        title=f"{dataset} · t = {t} · {plane}",
        margin=dict(l=40, r=20, t=60, b=65),
        legend=dict(orientation="h", y=-0.12),
        uirevision=f"{dataset}/{plane}",
        xaxis=dict(title=f"{horizontal} (µm)", range=ranges[horizontal]),
        yaxis=dict(
            title=f"{vertical} (µm)", range=ranges[vertical][::-1], scaleanchor="x", scaleratio=1
        ),
        scene=dict(
            xaxis=dict(title="x (µm)", range=ranges["x"]),
            yaxis=dict(title="y (µm)", range=ranges["y"][::-1]),
            zaxis=dict(title="z (µm)", range=ranges["z"]),
            aspectmode="data",
        ),
    )
    return figure


def timeline_figure(summary, t, predicted_nodes=None):
    figure = go.Figure()
    for col, color in [
        ("正解", COLORS["gt"]),
        ("対応あり", COLORS["det"]),
        ("未対応の正解", COLORS["miss"]),
    ]:
        figure.add_trace(
            go.Scatter(x=summary.t, y=summary[col], mode="lines", name=col, line=dict(color=color))
        )
    if predicted_nodes is not None and len(predicted_nodes):
        counts = predicted_nodes.groupby("t").size().reindex(summary.t, fill_value=0)
        figure.add_trace(
            go.Scatter(
                x=summary.t,
                y=counts,
                mode="lines",
                name="最終予測",
                line=dict(color=COLORS["prediction"]),
            )
        )
    if "予測対応" in summary:
        figure.add_trace(
            go.Scatter(
                x=summary.t,
                y=summary["予測対応"],
                mode="lines",
                name="予測と正解の対応",
                line=dict(color=COLORS["prediction_supported"]),
            )
        )
    figure.add_vline(x=t, line_color=COLORS["link"])
    figure.update_layout(
        template="plotly_dark",
        height=230,
        margin=dict(l=40, r=15, t=15, b=30),
        xaxis_title="時刻 t",
        yaxis_title="細胞数",
        legend=dict(orientation="h"),
    )
    return figure


def detail_table(gt, detections, matches, lineage=None):
    table = gt.rename(columns={"node_id": "gt_id"}).merge(matches, on=["gt_id", "t"], how="left")
    if lineage is not None:
        table = table[table.lineage == lineage]
    scores = (
        detections.set_index("node_id").score if "score" in detections else pd.Series(dtype=float)
    )
    table["detection_score"] = table.detection_id.map(scores)
    table["state"] = np.where(
        ~table.t.isin(matches.t),
        "キャッシュなし",
        np.where(table.detection_id.notna(), "対応あり", "未対応"),
    )
    return table.sort_values(["t", "gt_id"])
