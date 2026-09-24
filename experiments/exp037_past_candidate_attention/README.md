# exp037_past_candidate_attention

## 概要

- 仮説要約: 中央の各接続候補に対し、直前時点の全候補との3点座標関係を学習可能なattentionで集約すると、予測済み履歴へ固定せずにexp016より両胚の接続判断を改善できる。
- 変更点要約: exp016の固定画像特徴・候補・教師・loss・学習条件を維持し、13次元座標特徴を32次元MLPと全候補softmaxで集約する1,570 parameterのdelta branchだけを追加する。
- リスク: 全候補の3点組み合わせが最大11億を超え、memory-safe実装でも2-fold・3 epochの保守的予測が198.17時間となる。座標だけの曖昧さ、条件付きvalidation、単体診断と公式graph scoreの差も残る。
- 次: runtime gateで本学習前に停止した。計算方法または実験契約を見直すまで再実行せず、全graphへ進まない。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習・単体診断Notebook: `exp037_past_candidate_attention_train.ipynb`
- 編集元: `exp037_past_candidate_attention_train.py`
- 実行本体: `attention_train_pipeline.py`
- Kaggle Notebookのフル実行を正とし、ローカルでは静的検査と合成testだけを行う。
- 全graph推論Notebookは単体進行条件とユーザー承認後まで作成しない。

