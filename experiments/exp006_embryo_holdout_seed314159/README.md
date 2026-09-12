# exp006_embryo_holdout_seed314159

## 概要

- 仮説要約: exp005と同じ胚分離・batch size 8構成をseed 314159で再学習すると、1seedだけでは分からない予測変動と誤りの違いを測れる。
- 変更点要約: 学習seedだけを42から314159へ変更する。ensembleは別実験へ分ける。
- リスク: 親sourceのaugmentationとGPU kernelはbitwise deterministicではないため、差をseedだけの因果効果とは断定できない。
- 次: exp005との比較結果を確認し、実験の完了・採否をユーザーが判断する。

## 現在の状態

- train version 1はKaggle上で2fold smokeを完走し、合計推定28,963.242秒が7時間gateを超えたためfull training開始前に停止した。
- train version 2は9時間gateを通過し、2fold、各3 epochsを完走した。model manifestと2 checkpointのSHA一致を確認済み。
- `metrics.json.status`: `debug_completed`
- full model 2個と199動画のinferenceがKaggle上で正常終了した。欠落は0件で、保存済みgraphから再計算した公式評価が一致した。
- 公式CVは0.553379。exp005の0.124906に対する差は+0.428473で、44b6と6bbaの両方で悪化しなかった。
- 199動画すべてでexp005と異なるgraphが得られ、動画別誤差相関はPearson 0.612546、Spearman 0.724068だった。
- competition submissionとPublic LB取得は行っていない。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- train Notebookはseed 314159で2foldをscratch学習し、inference Notebookは199動画を公式評価してexp005との検出・edge差と動画別誤差相関を保存する。
- version 1のsmoke証拠は`artifacts/train_v1/`、version 2のモデルと学習証拠は`artifacts/train_v2/`、inference version 1の評価・比較・metadata・logは`artifacts/inference_v1/`へ保存した。
- 構造化された数値とSHAは[`metrics.json`](metrics.json)、解釈とユーザー判断は[`result.md`](result.md)を参照する。

## 表記

用語は`AGENTS.md`と`docs/glossary.md`を正とする。
