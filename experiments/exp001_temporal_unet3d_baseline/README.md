# exp001_temporal_unet3d_baseline

## 概要

- 仮説要約: 主催者公開の TemporalUNet3D と SimpleNodeTransformer を random initialization から公式READMEの3 epochsで学習し、Kaggle制限内で有効なtracking graph提出を生成できるか検証する。
- 変更点要約: seed 42、source・split・checkpoint・submissionのSHA記録、smokeによる11時間runtime gateを追加する。model、loss、split 0、3 epochs、推論条件は固定する。
- リスク: 公式sample splitは embryo_id を跨いだ汎化評価ではなく、3D convolutionとnode pair scoringはT4 x2でも12時間またはmemory制限を超える可能性がある。
- 次: Kaggleのsmokeはメモリ不足で停止した。[結果と停止理由](result.md)を保持し、実験の完了・採否はユーザー判断を待つ。固定条件のまま再実行する予定はない。

## 正の記録

- 数値、実験status、構造化された実行証拠: [metrics.json](metrics.json)
- route、設定、系譜: [config.yaml](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [requirements.md](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [result.md](result.md)
- 実行中の作業ログ: [SESSION_NOTES.md](SESSION_NOTES.md)
- 固定した主催者source: [official_source/SOURCE.json](official_source/SOURCE.json)

## 実行入口

- 学習 Notebook: exp001_temporal_unet3d_baseline_train.ipynb
- 推論 Notebook: exp001_temporal_unet3d_baseline_inference.ipynb
- Jupytext source: 同名の .py
- Notebookフル実行はKaggleを正とし、ローカルデータがないためlocal smokeは行わない。
