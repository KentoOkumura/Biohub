# exp008_deformation_pair_aux 結果

## 仮説

既知の滑らかな3D変形に対応する中間featureを補助lossで近づけると、疎い実注釈だけのexp005より外側2胚の中心・接続予測を改善できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- 比較対象: [exp005](../exp005_embryo_holdout_batch8/)
- `metrics.json` の参照キー: 未実行のためなし。
- `SESSION_NOTES.md` の実行記録: 実験化だけを記録。
- 参照する生成物パス: 未実装・未実行のためなし。

## 解釈

未実装・未実行。3D変形、feature対応、外側胚の精度について結論を出さない。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: 2026-09-11の依頼は実験化だけの承認であり、完了・採用・不採用の判断ではない。
- 理由: 実行証拠がないため。

## 次

実装の明示指示を待つ。実行後は両胚の公式指標に加え、変形valid率、Jacobian最小値、base/aux loss、Notebook実行時間を提示して判断を求める。
