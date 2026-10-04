# 上位解法を伏せた発想評価の保存記録

結論は[調査レポート](../../docs/surveys/biohub-source-hidden-idea-evaluation_20261002.md)を参照する。このディレクトリには入力、生成出力、採点、検証を保存する。

- `packet/`: 両生成回へ渡した5ファイル。仕様・方針・基準構成・数値はGit `7b648a3`、skillとschemaは評価時点のコピー。
- `input_manifest.json`、`audit_sources/`、`prepare_evaluation.py`: 元記録と抽出方法。生成担当には`audit_sources/`を渡していない。
- `protocol.json`、`withheld_rubric.json`、`effective_rubric.json`、`rubric_errata.json`: 生成出力を読む前に固定した手順・採点基準と、原文照合による訂正。
- `run_a/`、`run_b/`: 最初の6案、独立検討メモ、最終10案と優先5案、アクセス記録。改善効果は未検証。
- `initial_output_freeze.json`、`final_output_freeze.json`: 固定時点のSHA-256。保存済み出力を採点に合わせて修正しない。
- `finalization_intervention.json`: 親による中断と同じコンテキストでの再開・短い最終案への確定指示。新しい証拠や期待解は伝えていない。
- `judge/`: 原文照合、各項目の採点と根拠。
- `verification.json`、`score_summary.json`: 来歴・出力形式の検査と、採点表から再計算した件数。

リポジトリルートから、保存記録を再検査・再集計する。

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run python studies/biohub_source_hidden_eval_20261002/verify_evaluation.py
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run python studies/biohub_source_hidden_eval_20261002/summarize_scores.py
```

`prepare_evaluation.py`と`amend_rubric.py`は今回使った準備処理の記録で、保存済み入力・採点基準への再実行は不要。生成と採点は上記scriptで再現される処理ではない。新しい生成回は別の出力先とアクセス制限を用意する。
