# exp011_public_detector_selection 結果

## 仮説

公開コード・checkpoint・接続用特徴の対応を確認して版を固定すれば、画像モデルを更新せず既存trackerを初期値にした下流学習の基準を構成できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 実験statusと構造化された実行証拠: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- 機械可読な選定内容: [`assets/public_detector_selection.json`](assets/public_detector_selection.json)
- 比較根拠と限界: [`../../docs/surveys/biohub-public-detector-selection_20260912.md`](../../docs/surveys/biohub-public-detector-selection_20260912.md)

## 実行証拠

- 比較対象: Kaggleから取得した公開Notebook 4件、Pilkwang公開dataset 3件のmetadata・artifact manifest、実際にdownloadしたcheckpoint 3件。
- `metrics.json` の参照キー: `evidence.artifacts.input_file_sha`、`model_manifest_sha`、`model_count`、`model_shas`、`selected_mode`、`selected_model`。
- `SESSION_NOTES.md` の実行記録: 2026-09-12の公開Notebook一覧取得、pull、source比較、dataset metadata確認、checkpoint download、SHA照合。
- 参照する生成物パス: `assets/public_detector_selection.json`。
- 未実行: detector/tracker学習、full inference、Public LB再現、submission。

## 解釈

推奨は`reyhanksatria/biohub-cell-tracking-0-946-lb`の取得版を処理構成の参照とし、checkpointは同じ期待SHAを持つPilkwangのoriginal dataset 3件から取得する組である。primaryとsecondaryの`TemporalUNet3D`は2-frame window、z/y/x stride 1/4/4、32-channel candidate featureを使う。初回の下流学習ではprimary checkpointの`SimpleNodeTransformer`を初期値とし、primary image encoderとdetect head、secondary branch全体、DeepCenter、候補点生成、ILP、repairを固定する。

これにより、画像側を再学習せずtrackerだけを変更する後続比較を具体化できた。一方、0.946は複数model、association、ILP、repairを含む公開pipeline全体の報告であり、detector単体精度ではない。現行Public Scoreの独立取得、full inference、hidden test runtime測定、original dataset mountでの動作確認は行っていない。

## ユーザー判断

- 判断: 採用
- 確認日時 / 依頼メッセージ: 2026-09-12 / 「採用でいいです」
- 採用内容: 上記の組を初回controlとして固定する。
- 判断時に残る注意: primary checkpointの詳細な学習来歴、Notebook独自patchのcode license、小規模Kaggle動作確認は未確認。

## 次

採用済みmanifestを戦略と後続候補の具体的な入力とする。次にoriginal dataset mountの小規模確認と`exact_window_cache`を行い、公開基準の診断後に`frozen_image_encoder`を設計する。tracker学習とsubmissionは自動で開始しない。
