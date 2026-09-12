# exp010_exp006_deterministic_replay

## 概要

- 仮説要約: augmentation乱数とPyTorch/CUDA algorithmを固定すれば、同じKaggle T4 x2環境でmodel、held-out予測、CVを再現できる。
- 変更点要約: epoch/item別augmentation RNG、決定論的samplerとruntime設定、canonical model/prediction/metric SHA、2 run比較scriptを追加する。
- リスク: 決定論的algorithmによる停止または速度低下、固定環境外では同一性を保証できないこと。
- 次: byte-identicalなtrain/inference Notebookを2回ずつKaggleで実行し、全canonical SHAとCVを比較する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習 notebook: `exp010_exp006_deterministic_replay_train.ipynb`
- 推論 notebook: `exp010_exp006_deterministic_replay_inference.ipynb`
- Kaggle 準備と実行: [`SESSION_NOTES.md`](SESSION_NOTES.md)の予定を埋め、`kaggle-review-exp`と`kaggle-platform`の手順に従う
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 表記

用語は`AGENTS.md`の規則を正とし、公式資料、参加者の説明、論文、既存コードで実際に使われている専門用語を優先する。コンペ固有の略語とリポジトリ内の管理用語は`docs/glossary.md`で定義する。
