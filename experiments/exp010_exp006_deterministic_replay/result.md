# exp010_exp006_deterministic_replay 結果

## 仮説

augmentationの乱数をepochとitemへ安定に対応付け、PyTorch/CUDAの決定論的設定を強制すれば、固定したKaggle T4 x2環境で同じ2fold学習とheld-out評価を再実行したときにmodel tensor、予測内容、CVを一致させられる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 構造化された実行証拠: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

未実行。2回のフルtrain/inferenceが揃うまで再現性を担保済みとは扱わない。

## 解釈

未判断。

## ユーザー判断

- 判断: 未判断
- 理由: authoritative rerunと比較が未完了。

## 次

決定論的処理を実装・検証し、Kaggleで同一構成を2回実行する。
