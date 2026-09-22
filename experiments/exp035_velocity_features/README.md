# exp035_velocity_features

## 概要

- 仮説要約: 対象sampleを学習に使わないfirst-pass trackerの予測履歴から直前の移動ベクトルを作り、次候補とのずれをpair logitへ追加すると、固定画像特徴と現在位置だけを使うexp016より接続診断が改善する。
- 変更点要約: exp016の候補、画像特徴、教師、loss、外側胚分割を固定し、15次元の速度pair特徴をzero-initialized branchで加える。履歴生成model 4個とvelocity model 2個を学習する。
- リスク: 誤接続した履歴の連鎖、分裂後の履歴、疎い注釈、6個の新規trackerの実行費用。pair診断は公式graph metricではない。
- 結果: 保存済みexp016とのpair診断で既知edge recallと分裂親回収数が両胚とも低下し、gateは0/2胚で未達だった。全graph推論と公式評価は開始していない。
- 次: 今回のvelocity表現を不採用として実験を終了するか、ユーザー判断を確認する。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習Notebook: `exp035_velocity_features_train.ipynb`
- 実行本体: `train_pipeline.py`
- 履歴生成: `velocity_history.py`
- 速度特徴とmodel: `velocity_tracker.py`
- Kaggle Notebook実行を正とし、submissionは作成しない。全graph推論はpair診断gateを通過した場合だけ同じ実験へ追加する。

## 表記

用語は`AGENTS.md`の規則を正とし、公式資料、参加者の説明、論文、既存コードで実際に使われている専門用語を優先する。コンペ固有の略語とリポジトリ内の管理用語は`docs/glossary.md`で定義する。
