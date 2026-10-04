# exp026_division_candidate_budget

## 概要

固定検出中心から作る母・2娘の組を、各窓で同じ件数に制限して比較する。距離順位と、固定trackerの画像得点＋exp025の自己予測運動費用を使う順位を評価する。

- 差分: 候補組の順位と選別。検出器、画像特徴、tracker重み、元の9/14 µm幾何候補は固定。
- リスク: [exp025のカルマン実験](../exp025_kalman_hungarian_links/result.md)は完走し、実行receiptと校正ファイルは取得済みだが、状態のNPZファイルは未回収。初段の組回収だけでは公式graph精度を判断できない。
- 次: [残る入力確認](result.md)に従い、状態ファイルの利用可否、取得済みreceipt・校正と本実験の入力の対応、設定への接続を確認する。Kaggle診断は未実行。後段の組モデルによる選別は `division_triplets` の完成後に同じ実験へ追加する。

## 正の記録

[要件](requirements.md) · [設定](config.yaml) · [作業ログ](SESSION_NOTES.md) · [結果](result.md) · [数値](metrics.json)

## 実行入口

実行対象は `exp026_division_candidate_budget_diagnostic.ipynb`。入力の照合と[次の実行の条件](SESSION_NOTES.md#次の実行)を確認してからKaggleで実行する。提出ファイルは作らない。
