# exp004_embryo_holdout_baseline 結果

## 仮説

exp002と同じモデル、損失、3 epochs、checkpoint選択、decodeを維持し、学習胚と評価胚を2方向で分ければ、全199動画について学習から除外された胚の基準予測、検出・接続候補、固定公式評価を作成できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

設定とSHAをこのファイルへ独立した記録として転記しない。CV・Public LB・Private LBは`metrics.json`を数値の正本とする。

## 実行証拠

- 比較対象: exp002の両胚混在sample holdoutは学習時診断であり、本実験の胚を分けたCVと得点を直接比較しない。
- `metrics.json`の参照キー: `status=failed`、`cv=null`、`evidence.reruns`。Kaggle train versions 1-3はいずれもsmokeでCUDA OOMとなり、full trainingには進んでいない。
- version 1はfold 0の最初のbackwardで失敗した。version 2はfold 0を47.64秒で完走後、fold 1の最初のbackwardで失敗した。version 3はfold間で返却model、garbage collection、CUDA cacheを明示解放し、fold 0を44.44秒で完走したが、fold 1の最初のbackwardで再び失敗した。
- 生成済み証拠: `artifacts/train_v1_failure/`、`artifacts/train_v2_failure/`、`artifacts/train_v3_failure/`のfold別smoke log、kernel log、dataset index、split。model manifest、全199動画の予測、候補cache、公式評価結果は未生成。

## 解釈

Kaggle T4 x2で固定batch size 16を維持したfold 1 smokeは、fold間のCUDA resource解放後も最初のbackwardで再現してOOMとなった。これは低い評価得点ではなく未解決の実行失敗であり、契約の停止条件に該当する。全199動画の胚を分けた基準予測は成立しておらず、batch size、precision、または実行単位を変更した結果を本実験の固定条件と混ぜない。

## ユーザー判断

- 判断: 未判断。
- 確認日時 / 依頼メッセージ: 2026-09-11の「kaggleで実行してください」でKaggle Notebook実行が承認された。実験完了、採用、不採用、submissionの判断ではない。
- 理由: 3回のtrain smokeで固定batch size 16のOOMが解消せず、学習、推論、公式評価を完走していない。

## 次

batch size 8への変更は[exp005](../exp005_embryo_holdout_batch8/result.md)で実施済み。本実験の固定条件での失敗を保持し、採否・完了は上記のとおり未判断とする。固定条件の追加実行は予定しない。

## 2026-09-11 失敗後の提案（履歴）

推奨は、この固定条件の失敗をexp004に残し、batch size 8へ変更する別実験を作ることである。microbatchとgradient accumulationはBatchNorm3dの統計が変わるため親学習と等価ではなく、AMPも固定sourceでは未実装で学習数値を変える。いずれの再開案も結果に影響するため、ユーザー判断後に実験化する。
