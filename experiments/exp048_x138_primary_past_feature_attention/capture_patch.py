"""Insert label-free pair-window capture into the fully patched exp043 predictor."""

from __future__ import annotations


def patch_x138_pair_capture(source: str) -> str:
    replacements = [
        (
            "            _bidirectional_weight = float(\n",
            "            _exp048_primary_forward = edge_logits_pair.detach().clone()\n"
            "            _bidirectional_weight = float(\n",
        ),
        (
            "                reverse_logits_pair = reverse_logits_native.transpose(1, 2)\n",
            "                _exp048_primary_reverse = reverse_logits_native.detach().clone()\n"
            "                reverse_logits_pair = reverse_logits_native.transpose(1, 2)\n",
        ),
        (
            '                if secondary_link_mode == "raw":\n',
            "                _exp048_secondary_raw = secondary_logits_pair.detach().clone()\n"
            '                if secondary_link_mode == "raw":\n',
        ),
        (
            "            raw = edge_logits_pair[0]\n",
            """            _exp048_capture_root = os.environ.get('EXP048_TRACKER_CAPTURE_DIR', '')
            if _exp048_capture_root:
                if secondary_model is None or _bidirectional_weight <= 0:
                    raise RuntimeError(
                        'exp048 capture requires x138 secondary and reverse branches'
                    )
                _exp048_path = Path(_exp048_capture_root) / ds_path.stem
                _exp048_path.mkdir(parents=True, exist_ok=True)
                _exp048_prev_start, _exp048_prev_end = coord_offset.get(
                    t_src - 1, (0, 0)
                )
                _exp048_prev = coords_so_far[_exp048_prev_start:_exp048_prev_end]
                np.savez_compressed(
                    _exp048_path / f'{t_src:06d}_{t_tgt:06d}.npz',
                    window_frames=np.asarray([t_src, t_tgt], dtype=np.int32),
                    candidate_ids_prev=np.arange(
                        _exp048_prev_start, _exp048_prev_end, dtype=np.int64
                    ),
                    candidate_ids_src=idx_src,
                    candidate_ids_tgt=idx_tgt,
                    coords_prev_grid=_exp048_prev[:, 1:].astype(np.float32),
                    coords_src_grid=c_src[:, 1:].astype(np.float32),
                    coords_tgt_grid=c_tgt[:, 1:].astype(np.float32),
                    primary_features_src=unet_feat_src[0].float().cpu().numpy(),
                    primary_features_tgt=unet_feat_tgt[0].float().cpu().numpy(),
                    position_features_src=p_pos_src[0].float().cpu().numpy(),
                    position_features_tgt=p_pos_tgt[0].float().cpu().numpy(),
                    primary_forward_logits=_exp048_primary_forward[0].float().cpu().numpy(),
                    primary_reverse_logits=_exp048_primary_reverse[0].float().cpu().numpy(),
                    secondary_logits=_exp048_secondary_raw[0].float().cpu().numpy(),
                    x138_fused_logits=edge_logits_pair[0].float().cpu().numpy(),
                )
            raw = edge_logits_pair[0]
""",
        ),
        (
            "        graph = build_graph(coords, edges)\n",
            "        if os.environ.get('EXP048_TRACKER_CAPTURE_DIR', ''):\n"
            "            continue  # Pair diagnostics precede whole-video graph inference.\n"
            "        graph = build_graph(coords, edges)\n",
        ),
    ]
    for old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise ValueError(f"exp043 predictor capture anchor count={count}: {old[:60]!r}")
        source = source.replace(old, new, 1)
    compile(source, "exp048_patched_predictor.py", "exec")
    return source
