# exp044_x138_past_candidate_knn_attention

## 概要

- 仮説要約: exp043の候補と画像特徴上で、直前frameの近傍8候補をattentionで参照すると既知edgeの接続が改善するか。
- 変更点要約: exp043の検出・座標補正・画像encoder・secondary tracker・融合を固定し、公開primary trackerとK=8 attentionを同時学習する別実験。
- リスク: GEFFの部分注釈、公開画像モデルに条件付けられたCV、capture・学習のGPU費用。
- 次: Kaggle学習・隣接2フレームの評価と追加診断は終了した。[結果](result.md)のgraph進行条件を満たさなかったため全graph推論を保留し、実験の完了・採否についてユーザー判断を待つ。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習 notebook: `exp044_x138_past_candidate_knn_attention_train.ipynb`
- 全graph推論Notebook: 進行条件未達のため作成していない。
- 実行済みの手順と診断経緯: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 表記

用語は`AGENTS.md`の規則を正とし、公式資料、参加者の説明、論文、既存コードで実際に使われている専門用語を優先する。コンペ固有の略語とリポジトリ内の管理用語は`docs/glossary.md`で定義する。
