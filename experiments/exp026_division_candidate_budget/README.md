# exp026_division_candidate_budget

## 概要

固定検出中心から作る母・2娘の組を、各窓で同じ件数に制限して比較する。距離順位と、固定trackerの画像得点＋exp025の自己予測運動費用を使う順位を評価する。

- 差分: 候補組の順位と選別。検出器、画像特徴、tracker重み、元の9/14 µm幾何候補は固定。
- リスク: exp025の完走出力が必要。初段の組回収だけでは公式graph精度を判断できない。
- 次: exp025の完走receiptを確認し、Kaggleで199動画の初段診断を実行する。後段の組モデルによる選別は `division_triplets` の完成後に同じ実験へ追加する。

## 正の記録

[要件](requirements.md) · [設定](config.yaml) · [作業ログ](SESSION_NOTES.md) · [結果](result.md) · [数値](metrics.json)

## 実行入口

`exp026_division_candidate_budget_diagnostic.ipynb` をKaggleで実行する。提出ファイルは作らない。
