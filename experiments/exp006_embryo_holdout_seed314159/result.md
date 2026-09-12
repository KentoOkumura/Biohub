# exp006_embryo_holdout_seed314159 結果

## 仮説

exp005と同一の胚分離・batch size 8構成をseed 314159で再学習すると、1seedだけでは分からない予測変動と誤りの違いを測れる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- 比較対象: [exp005](../exp005_embryo_holdout_batch8/)
- Kaggle kernel: [train version 2](https://www.kaggle.com/code/kentookumura/exp006-embryo-holdout-seed314159-train)、[inference version 1](https://www.kaggle.com/code/kentookumura/exp006-embryo-holdout-seed314159-inference)
- `metrics.json` の参照キー: `status=debug_completed`、`cv`、`comparison`、`diagnostic_validation`、`evidence.reruns[0:3]`、`evidence.artifacts`。
- train version 2はTesla T4 2基で22,940.600秒、full training 22,866.104秒。2foldとも3 epochsを完走した。fold 0とfold 1のbest epochはともに2、内部選択scoreはそれぞれ0.9357と0.9310だった。
- inference version 1はTesla T4 2基、internet無効で19,032.269秒。12時間gateを通過し、外側評価胚の199動画を欠落なく推論した。
- 保存済みgraphから公式評価を再計算し、実行中の集計との一致を確認した。全体scoreは0.5533789422、44b6は0.7045717889、6bbaは0.5267756514だった。
- exp005の全体score 0.1249055162との差は+0.4284734261。胚別では44b6が+0.0797788939、6bbaが+0.5007270983で、両胚とも悪化しなかった。
- exp005とのexact比較では199/199動画でgraph、検出集合、選択edge集合が異なった。動画別score誤差の相関はPearson 0.6125461826、Spearman 0.7240679729だった。
- detectionのmicro Jaccardは0.4095544888、選択edgeのmicro Jaccardは0.1696636607、動画別score差の平均絶対値は0.3525097899だった。
- model manifest SHAは`15175596a731e172a88699aae18423986b5f8ef0270486b28609348fb18943d6`、prediction manifest SHAは`234619c58e1ea4001b953eefec72cd226119650d4668c51cd523687b3d42f588`、seed comparison SHAは`34a7597b958ad16d5f537777888cb5d60668b8edda5e0909f22f356dae87239c`。
- 回収した評価、比較、metadata、kernel logは`artifacts/inference_v1/`へ保存した。competition submission、hidden test推論、Public LB取得は行っていない。

## 解釈

実行成功条件である2fold学習、199動画の欠落なし推論、保存graphからの公式評価再現を満たした。追加seedの有用性について事前に定めた条件も満たした。両胚でexp005よりscoreが悪化せず、すべての動画で異なるgraphが得られたため、seed 314159のモデルは後続の2-seed融合を検証する候補として支持される。

ただし、誤差相関は0ではなく、特に44b6ではPearson 0.8316204415、Spearman 0.7856136821と高い。予測差が大きいことだけで融合改善を保証しないため、融合の有効性は本実験では判断しない。また、親sourceのaugmentationとGPU kernelはbitwise deterministicではなく、観測差をseedだけの厳密な因果効果とは断定しない。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: 2026-09-12の「完了しました」はinference完了の通知として扱い、実験の完了・採用・不採用の判断とはまだ扱わない。
- 判断材料: 実行成功条件と追加seedの有用性に関する事前条件は満たした。推奨は、本実験を`completed`として記録し、seed 314159モデルを後続の2-seed融合比較に使用すること。

## 次

ユーザー判断後に実験statusを更新し、この実験に関係する変更だけをcommit・pushする。2-seed融合、competition submission、Public LB取得は別途の明示承認なしに行わない。
