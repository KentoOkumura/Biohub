# exp038_past_candidate_knn_attention

## 概要

- 仮説要約: 近傍の過去候補だけで有用な文脈を保持し、予算内で両胚の接続を改善できるか。
- 変更点要約: 過去候補を物理距離の近傍へ制限してから特徴とattentionを計算する。
- リスク: 前身候補の除外、固定公開重みによる条件付き評価、部分注釈の限界。
- 次: 保持率と費用を確認し、条件成立時だけ学習・単体比較へ進む。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習 notebook: `exp038_past_candidate_knn_attention_train.ipynb`
- 全graph推論と提出: 初段の進行条件成立・追加承認まで実装・実行しない。
- Kaggle 準備と実行: [`SESSION_NOTES.md`](SESSION_NOTES.md)の予定を埋め、`kaggle-review-exp`と`kaggle-platform`の手順に従う
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 表記

用語は`AGENTS.md`の規則を正とし、公式資料、参加者の説明、論文、既存コードで実際に使われている専門用語を優先する。コンペ固有の略語とリポジトリ内の管理用語は`docs/glossary.md`で定義する。
