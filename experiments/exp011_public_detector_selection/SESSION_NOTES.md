# exp011_public_detector_selection セッションノート

## 目的

公開検出器・画像特徴抽出器を固定してtrackerだけを学習する方針に対し、参照Notebook、checkpoint、feature contract、既存tracker初期値を一意に指定する。

## 現在の作業

- 作業内容: 採用判断を実験記録と後続候補へ反映し、対象変更をcommit・pushする。
- ブロック要因: なし。
- 次: 採用済みmanifestを後続の小規模動作確認と特徴cache検証へ渡す。

## コマンドログ

### 2026-09-12

- `task new-exp EXP=exp011_public_detector_selection`を試したが`task`コマンドがなかったため、AGENTS.mdのfallbackに従い`make new-exp EXP=exp011_public_detector_selection`で雛形を作成した。
- `make new-survey-report ...`で`docs/surveys/biohub-public-detector-selection_20260912.md`を作成した。
- lockfileで固定したKaggle CLIからcompetitionの公開Notebook一覧をJSONで取得した。CLI認証は利用可能で、追加tokenの記録は行っていない。
- 次の4件を`/tmp/public_detector_selection_20260912/`へpullし、metadata、Notebook source、参照dataset、model構成、特徴抽出、association、decode、repairの差分を比較した。
  - `reyhanksatria/biohub-cell-tracking-0-946-lb`
  - `sjlee101/biohub-lf-dctta-v020`
  - `flexonafft/biohub-lineage-forge-precision-tracking`
  - `redoctopusk/biohub-942tta`
- Pilkwang公開dataset 3件の現行metadataとartifact manifestを確認し、exact checkpointを`/tmp`へdownloadした。3件すべてで実ファイルSHAが0.946 Notebook内の期待SHAと一致した。
- variant数は4 Notebook、checkpoint数は3。fold、booster、control再学習は0。
- detector/tracker学習、full inference、Kaggle Notebook push/run、submissionは実施していない。GPU使用時間は0。
- 2026-09-12: ユーザー「採用でいいです」により、推奨した公開Notebook・3 checkpoint・primary tracker初期値の組を採用した。

### 検証結果

`task`が利用できないため、同名の`make` targetを使った。

```bash
make validate-exp EXP=exp011_public_detector_selection
make check-exp EXP=exp011_public_detector_selection
make test-exp EXP=exp011_public_detector_selection
make update-survey-index
make validate-surveys
make check-strategy-docs
```

- `validate-exp`: strict validation passed。
- `check-exp`: ruff checkとformat check passed。
- `test-exp`: 3 passed。
- `validate-surveys`: survey index is up to date。
- `check-strategy-docs`: passed。
- `review_exp_docs.py exp011 --strict`: target experimentの主要な証拠区分はすべて存在する。

## 変更点

- 0.946公開Notebookの取得版をkernel id、取得時刻、content SHAで固定した。
- Notebook内のmirrorではなく、同じcheckpoint SHAを持ちlicenseを確認できるPilkwang original dataset 3件をcanonical artifactにした。
- primary trackerだけを最初の学習対象にするfreeze方針と、未確認の学習来歴・license・runtime・動作確認を記録した。
- 学習・推論Notebookはvalidator必須の未実行雛形だけを保持し、本実験では実行対象にしない。

## 次のアクション

1. original Pilkwang dataset mountの小規模動作確認と`exact_window_cache`を実験化する。
2. 公開基準の診断後、primary `SimpleNodeTransformer`を初期値にする`frozen_image_encoder`を設計する。
