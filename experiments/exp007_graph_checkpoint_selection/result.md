# exp007_graph_checkpoint_selection 結果

## 仮説

固定公式graph指標でcheckpointを選ぶと、現行のedge accuracyとnode recallの積による選択より、外側の胚holdoutで良いepochを選べる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- 比較対象: 同じ学習runの現行proxy selector。exp005は既存基準の副参照。
- `metrics.json` の参照キー: 未実行のためなし。
- `SESSION_NOTES.md` の実行記録: 実験化だけを記録。
- 参照する生成物パス: 未実装・未実行のためなし。

## 解釈

未実装・未実行。selectorの順位差や公式指標への影響について結論を出さない。

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: 2026-09-11の依頼は実験化だけの承認であり、完了・採用・不採用の判断ではない。
- 理由: 実行証拠がないため。

## 次

実装の明示指示を待つ。実行後は両胚の公式指標、選択epoch、失敗件数、Notebook実行時間を提示して判断を求める。
