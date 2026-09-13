# exp014_exact_window_cache

## 概要

- 仮説要約: 固定公開モデルの候補点特徴を元の2-frame window単位で可逆保存すれば、下流予測を変えずに画像からの再抽出費用を省ける。
- 変更点要約: exp013の予測parameterを固定し、primary/secondaryの32次元候補特徴、候補ID、座標、検出scoreをNPZへ保存・再読込して完全一致を検査する。
- リスク: window identityまたは座標対応の誤り、I/O費用・保存量、GPU処理の再計算差により等価性または費用削減が成立しない可能性がある。
- 次: Kaggle T4 2基でのpublic test全件検証に成功し、後続tracker比較用cacheとして利用可能と判断された。同じ固定公開モデルを使う次の診断へ引き継ぐ。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- cache検証 notebook: `exp014_exact_window_cache_inference.ipynb`
- Kaggle 準備と実行: [`SESSION_NOTES.md`](SESSION_NOTES.md)の予定を埋め、`kaggle-review-exp`と`kaggle-platform`の手順に従う
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 表記

用語は`AGENTS.md`の規則を正とし、公式資料、参加者の説明、論文、既存コードで実際に使われている専門用語を優先する。コンペ固有の略語とリポジトリ内の管理用語は`docs/glossary.md`で定義する。
