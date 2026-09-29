# ruff: noqa: E501
"""Patch the materialized x138 predictor to retain the frozen window features."""

from __future__ import annotations


def add_feature_capture(source: str) -> str:
    """Capture each corrected node's primary/secondary features and detector score."""
    replacements = [
        (
            "_CACHE_EDGES: list = []\n",
            "_CACHE_EDGES: list = []\n_CACHE_FEATURE_SUM: dict = {}\n_CACHE_FEATURE_COUNT: dict = {}\n_CACHE_FEATURE_WIDTH = [0]\n",
        ),
        (
            "            raw = edge_logits_pair[0]\n",
            """            if _CACHE_DIR:
                if secondary_model is None:
                    raise RuntimeError('exp050 requires the frozen secondary tracker')
                _fs = torch.cat((
                    unet_feat_src[0].float(), secondary_feat_src[0].float(),
                    torch.sigmoid(model._index_features(
                        det_logits[f_idx], p_coords_src, p_mask_src,
                    )[0].float()),
                ), dim=1).detach().cpu().numpy()
                _ft = torch.cat((
                    unet_feat_tgt[0].float(), secondary_feat_tgt[0].float(),
                    torch.sigmoid(model._index_features(
                        det_logits[f_idx + 1], p_coords_tgt, p_mask_tgt,
                    )[0].float()),
                ), dim=1).detach().cpu().numpy()
                if _fs.shape[1] != _ft.shape[1] or not np.isfinite(_fs).all() or not np.isfinite(_ft).all():
                    raise RuntimeError('invalid exp050 frozen node features')
                _CACHE_FEATURE_WIDTH[0] = int(_fs.shape[1])
                for _ids, _values in ((idx_src, _fs), (idx_tgt, _ft)):
                    for _node_id, _vector in zip(_ids, _values):
                        _key = int(_node_id)
                        if _key in _CACHE_FEATURE_SUM:
                            _CACHE_FEATURE_SUM[_key] += _vector
                            _CACHE_FEATURE_COUNT[_key] += 1
                        else:
                            _CACHE_FEATURE_SUM[_key] = _vector.copy()
                            _CACHE_FEATURE_COUNT[_key] = 1
            raw = edge_logits_pair[0]
""",
        ),
        (
            "            np.savez_compressed(\n                _cd / f'{name}.npz', coords=coords, low_coords=_lc, low_score=_lsc,\n",
            """            _node_features = np.zeros((len(coords), _CACHE_FEATURE_WIDTH[0]), np.float32)
            _node_feature_count = np.zeros(len(coords), np.int16)
            for _key, _sum in _CACHE_FEATURE_SUM.items():
                _node_features[_key] = _sum / _CACHE_FEATURE_COUNT[_key]
                _node_feature_count[_key] = _CACHE_FEATURE_COUNT[_key]
            if not np.isfinite(_node_features).all():
                raise RuntimeError('nonfinite exp050 feature cache')
            np.savez_compressed(
                _cd / f'{name}.npz', coords=coords, low_coords=_lc, low_score=_lsc,
                node_features=_node_features, node_feature_count=_node_feature_count,
""",
        ),
        (
            "            _CACHE_EDGES.clear()\n",
            "            _CACHE_EDGES.clear()\n            _CACHE_FEATURE_SUM.clear()\n            _CACHE_FEATURE_COUNT.clear()\n",
        ),
        (
            "        if cfg.use_ilp and graph.num_edges() > 0:\n",
            "        if cfg.use_ilp and graph.num_edges() > 0 and os.environ.get('BIOHUB_SKIP_CACHE_ILP') != '1':\n",
        ),
        (
            '        save_graph(graph, output_dir / f"{name}.geff")\n',
            '        if os.environ.get("BIOHUB_SKIP_CACHE_ILP") != "1":\n'
            '            save_graph(graph, output_dir / f"{name}.geff")\n',
        ),
    ]
    for old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(f"x138 feature patch anchor occurs {count} times: {old[:48]!r}")
        source = source.replace(old, new, 1)
    compile(source, "predict_unet_transformer.py", "exec")
    return source
