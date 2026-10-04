# exp026_division_candidate_budget 結果

## 仮説

固定9/14 µm候補から各窓で同数の母・2娘組を選ぶとき、固定画像得点と自己予測運動費用を用いる順位が距離順位より既知分裂を両胚で多く残す。

## 実行証拠

初段の実装とローカルの静的・単体検証は完了。先行する[exp025のカルマン実験](../exp025_kalman_hungarian_links/result.md)は全199動画の評価を終え、実行receiptと両foldの校正ファイルを取得済みである。取得済みファイルのSHAと状態ファイルの取得状況は[上流のmetrics.json](../exp025_kalman_hungarian_links/metrics.json)を参照する。

本実験で必要な状態のNPZファイルは未回収であり、Kaggle入力としての利用可否と内容の照合も未確認。`config.yaml`の`data.motion.execution_receipt_sha256`と`calibration_sha256`は未設定のままで、本実験の入力との対応を確認してから接続する。上流の完走をもって本実験が実行可能とは判断しない。Kaggleの全199動画診断、候補回収値、公式combined score、division Jaccardは本実験では未実行・未計測。

## 解釈

初段で測るのは既知151分裂に対する候補回収と費用である。疎い注釈から全候補のprecisionや通常継続の確定負例は得られない。固定公開画像モデルがtrain胚を見ているため、胚別結果も独立CVとは扱わない。組モデルによる最終graph選別は `division_triplets` 完成後に同じ実験へ追加する。

## ユーザー判断

実験完了・採否は未判断。Kaggle実測と後段の公式評価が揃った後に判断を求める。
