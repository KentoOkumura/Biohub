# exp011_public_detector_selection

## 概要

- 仮説要約: 取得可能な公開checkpointと互換コードを版固定すれば、画像モデルを更新せずprimary trackerだけを学習する後続実験を定義できる。
- 変更点要約: 公開0.946 Notebookを参照構成とし、Pilkwang公開dataset 3件の版・checkpoint SHA、feature contract、primary `SimpleNodeTransformer`初期値を固定した。
- リスク: 公開0.946は独立再現しておらず、primaryの学習来歴、hidden test runtime、Notebook独自patchのcode licenseが未確認。
- 次: 採用済みmanifestを入力にoriginal dataset mountの小規模確認と`exact_window_cache`を進め、公開基準診断後に`frozen_image_encoder`を実験化する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- 選定manifest: [`assets/public_detector_selection.json`](assets/public_detector_selection.json)
- 選定根拠: [`../../docs/surveys/biohub-public-detector-selection_20260912.md`](../../docs/surveys/biohub-public-detector-selection_20260912.md)
- 選定Notebookの処理解説: [`../../docs/surveys/biohub-cell-tracking-0946-notebook-explanation_20260913.md`](../../docs/surveys/biohub-cell-tracking-0946-notebook-explanation_20260913.md)

## 実行入口

validator用の学習・推論Notebook雛形を保持するが、本実験では実行しない。Kaggle上の小規模動作確認、特徴cache、tracker学習は選定承認後の別実験で実施する。

## 表記

用語は`AGENTS.md`の規則を正とし、公式資料、参加者の説明、論文、既存コードで実際に使われている専門用語を優先する。コンペ固有の略語とリポジトリ内の管理用語は`docs/glossary.md`で定義する。
