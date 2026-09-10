# exp001_temporal_unet3d_baseline 結果

## 仮説

主催者公開の TemporalUNet3D と SimpleNodeTransformer をT4 x2でrandom initializationから3 epochs学習し、生成checkpointだけを使ってhidden test全datasetの有効なsubmission.csvを12時間制限内に生成できる。

## 記録の参照先

- 設定、系譜、再現性方針: [config.yaml](config.yaml)
- CV/LB、実験status、kernel情報、Notebook実行時間、生成物SHA: [metrics.json](metrics.json)
- 実行コマンドと途中経過: [SESSION_NOTES.md](SESSION_NOTES.md)
- 固定source: [official_source/SOURCE.json](official_source/SOURCE.json)

## 実行証拠

- 比較対象: 採点済み親実験なし。公開checkpointは学習入力にせず、参照実装の出力としてのみ扱う。
- metrics.json の参照キー: `status=failed`、`evidence.kaggle.kernel_version=3`、`evidence.kaggle.notebook_runtime_seconds=367.043`、`evidence.artifacts.model_count=0`、`evidence.reruns`。
- SESSION_NOTES.md の実行記録: version 1・2の依存解決失敗、version 3のoffline依存修正、GPU quota、T4 x2、split、smoke OOM、回収した証拠SHAを記録した。
- Kaggle Notebook: [exp001_temporal_unet3d_baseline train](https://www.kaggle.com/code/kentookumura/exp001-temporal-unet3d-baseline-train) version 3、private、internet無効、`NvidiaTeslaT4`。
- 実行条件と観測: 199 sampleをsplit 0で180 train / 19 validationへ分割し、2,076,706 parameters、batch size 16、T4 x2 DataParallelでsmokeを開始した。最初のbackwardでGPU 0の追加1.84 GiBを確保できずOOMになった。
- 回収した生成物: dataset index、split JSON、smoke log、失敗時metrics。checkpointとmodel manifestは生成されなかった。

## 解釈

固定した主催者公開設定は、現在のKaggle T4 x2環境ではsmokeのbackwardを完了できず、3 epochsのscratch trainingへ進めなかった。この実験が直接検証した「T4 x2で公式batch size 16の3 epochsを完走できる」という仮説はOOMの観測と整合しない。実験契約はsmoke OOMを停止条件とし、同じ実験内のbatch size変更を禁止しているため、救済再実行は行っていない。checkpointがないためinferenceと提出形式検証も未実行で、CVとLBは得られていない。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: なし
- 理由: OOMの実行証拠は得られたが、実験を完了・採用・不採用のどれにするかはユーザー判断が必要。

## 次

batch size 8などの変更を新しい実験として行うかユーザーに確認する。実際のsubmission、実験の完了・採用・不採用はユーザー判断まで確定しない。
