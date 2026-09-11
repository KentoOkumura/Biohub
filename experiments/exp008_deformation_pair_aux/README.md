# exp008_deformation_pair_aux

## 概要

- 仮説要約: 学習画像へ既知の滑らかな3D変形を加え、対応する中間featureを近づけると、疎い実注釈だけの学習より外側胚の中心・接続予測を改善できる。
- 変更点要約: training windowの50%へfold-safeな3D変形pairと重み0.1のfeature cosine lossを追加する。推論とdecodeはexp005のまま。
- リスク: paired forwardによるOOM・時間超過、物理座標とfeature座標の変換誤り、変形痕跡への過適合。
- 次: ユーザーが実装を指示した後、変形・Jacobian・valid maskのtestから実装する。

## 現在の状態

- backlog `deformation_pairs`から実験化済み、未実装、未実行。
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
