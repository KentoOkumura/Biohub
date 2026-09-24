# exp037_past_candidate_attention 結果

## 仮説

中央の各接続候補に対して直前時点の全検出候補との3点座標関係を学習可能な重みで集約すると、単一の予測済み過去対応へ固定せず、保存済みexp016より両胚の既知接続recallを改善できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV、実験status、Kaggle情報、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- Kaggle kernel: [`kentookumura/exp037-past-candidate-attention-train`](https://www.kaggle.com/code/kentookumura/exp037-past-candidate-attention-train)、version 1と2、T4、internet無効。
- version 1: runtime benchmarkの最初のtrain stepでCUDA OOM。14.56 GiB中14.54 GiB使用時に62 MiBを追加確保できなかった。
- version 2: target軸chunk 32とgradient checkpointingを追加し、候補集合・fold・epoch・batch size・lossを変えずにOOMを解消した。
- 入力監査: 19,701 windows中18,707 windowsが対象、994 windowsはGT nodeが空のframeを含む既存方針により除外。
- runtime benchmark: 66 windows、最大候補積1,102,004,400、train 510.33秒（7.732秒/window）、評価129.89秒（1.968秒/window）、peak GPU memory 1,628,975,616 bytes（1.52 GiB）。
- 全2-fold・各3 epochの予測: 係数適用前475,602.32秒（132.11時間）、保守係数1.5適用後713,403.49秒（198.17時間）。12時間gateの16.51倍だったため、本学習前に停止した。
- 構造化された数値とSHA: [`metrics.json`](metrics.json) の `evidence.benchmark` と `evidence.artifacts`。取得した小規模JSONは [`artifacts/kaggle_v2/`](artifacts/kaggle_v2/) に保存した。
- model、外側胚pair prediction、CV、LB、公式scoreは未生成・未計測。単体benchmarkのmetricをexp016との比較や公式scoreの代用にしない。

## 解釈

全過去候補を中央pairごとに明示的に列挙する今回の13次元特徴とMLPは、memory-safeには実行できるが、最大候補積が11億を超え、合意した全候補・2-fold・3 epoch条件では12時間内に学習できない。したがって精度仮説は未検証であり、negative resultが示すのは今回の全組み合わせ実装の計算可能性だけである。過去情報全般やattention自体の有効性は棄却しない。全graph推論とsubmissionは実行していない。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: 2026-09-22にKaggle単体診断の実行承認を受けた。実験の採用・不採用・完了判断は未受領。
- 理由: runtime gateで本学習前に停止し、精度比較は得られていない。`failed`は実行状態であり、不採用判断ではない。

## 次

同じ全組み合わせ実装をそのまま再実行しない。候補集合を変えず12時間以内へ落とせる表現・計算方法、または候補選択を含む別の実験契約を検討し、変更が精度仮説の意味を変える場合はユーザー確認後に別実験として扱う。
