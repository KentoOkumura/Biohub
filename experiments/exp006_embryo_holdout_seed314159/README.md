# exp006_embryo_holdout_seed314159

## 概要

- 仮説要約: exp005と同じ胚分離・batch size 8構成をseed 314159で再学習すると、1seedだけでは分からない予測変動と誤りの違いを測れる。
- 変更点要約: 学習seedだけを42から314159へ変更する。ensembleは別実験へ分ける。
- リスク: 親sourceのaugmentationとGPU kernelはbitwise deterministicではないため、差をseedだけの因果効果とは断定できない。
- 次: ユーザーが実装を指示した後、exp005との差分testとtrain/inference Notebookを実装する。

## 現在の状態

- 実験化済み、未実装、未実行。
- `metrics.json.status`: `planned`
- Notebookは雛形のままであり、Kaggleへpushしない。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- train/inference Notebookは未実装の雛形であり、実装承認後に必要な処理とテストを追加する。
- Kaggle prepare、push、実行は今回の実験化に含めない。予定は[`SESSION_NOTES.md`](SESSION_NOTES.md)を参照する。

## 表記

用語は`AGENTS.md`と`docs/glossary.md`を正とする。
