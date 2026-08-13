# 評価指標

このファイルは、公式メトリックをローカル実装と実験判断に落とし込むための運用メモです。実験テンプレートが読む metric 名は `project.yml` の `defaults.metric` を正とし、公式資料からの抜粋と出典は `docs/official/evaluation.md` に置きます。

## 公式メトリック要約

- 名前:
- 最適化方向: maximize / minimize
- 式:
- 実装の出典:

## ローカル実装

- 実装先: `src/metrics.py`、各実験の補助モジュール、または train notebook
- 入力:
- 出力:
- 公式例との照合:

## エッジケース

- 欠損予測:
- 重複 ID:
- 不正な値域:
- 同点:

## 解釈

- 意味のあるスコア変動:
- 想定される public/private のノイズ:
- 既知の不一致リスク:
