# exp005_embryo_holdout_batch8

## 概要

- 仮説要約: exp004のOOM原因が学習batch size 16のGPU memory要求なら、8へ下げることで2方向の胚holdout学習を完走し、全199動画の独立予測を作れる。
- 変更点要約: exp004を親に、学習batch sizeだけを16から8へ変更する。2fold、3 epochs、model、loss、checkpoint selection、推論、候補保存、公式評価は固定する。
- リスク: 2胚だけのholdoutで、胚方向のscore差が大きい。後続実験で同じ結果を繰り返し選択へ使うと開発集合になる。
- 追加診断: `6bba`でも0.99候補は既知GT nodeの94.58%を回収したが、接続確率0.5通過edgeは候補node当たり0.00514本で、`44b6`の約120分の1だった。
- 次: exp006とexp007の結果が揃った後に、動画単位のgroup validationと胚holdoutの役割分担を含む実験方針を更新する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習 notebook: `exp005_embryo_holdout_batch8_train.ipynb`
- 推論 notebook: `exp005_embryo_holdout_batch8_inference.ipynb`
- 追加診断 notebook: `exp005_embryo_holdout_batch8_diagnostic.ipynb`
- Kaggle 準備と実行: [`SESSION_NOTES.md`](SESSION_NOTES.md)の予定を埋め、`kaggle-review-exp`と`kaggle-platform`の手順に従う
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 表記

用語は`AGENTS.md`の規則を正とし、公式資料、参加者の説明、論文、既存コードで実際に使われている専門用語を優先する。コンペ固有の略語とリポジトリ内の管理用語は`docs/glossary.md`で定義する。
