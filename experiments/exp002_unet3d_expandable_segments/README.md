# exp002_unet3d_expandable_segments

## 概要

- 仮説要約: CUDA利用前に`PYTORCH_ALLOC_CONF=expandable_segments:True`を設定すれば、exp001でOOMになったbatch size 16の同一smokeを完了し、3 epochs学習へ進める。
- 変更点要約: exp001のmodel、source、split、batch size、loss、3 epochsを固定し、PyTorch allocator設定とmemory証拠記録だけを追加する。
- リスク: OOMがmemory fragmentationではなく容量不足なら解消しない。validationはsample単位holdoutの診断値であり、汎化性能のprimary CVには使わない。
- 次: ユーザー判断で初期baselineとして`usable`にした。胚を分けた主評価用CVは未取得であり、後続実験でPublic LBとの整合を確認する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習 notebook: `exp002_unet3d_expandable_segments_train.ipynb`
- 推論 notebook: `exp002_unet3d_expandable_segments_inference.ipynb`
- Kaggle 準備と実行: [`SESSION_NOTES.md`](SESSION_NOTES.md)の予定を埋め、`kaggle-review-exp`と`kaggle-platform`の手順に従う
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 表記

用語は`AGENTS.md`の規則を正とし、公式資料、参加者の説明、論文、既存コードで実際に使われている専門用語を優先する。コンペ固有の略語とリポジトリ内の管理用語は`docs/glossary.md`で定義する。
