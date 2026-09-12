# exp007_graph_checkpoint_selection 結果

## 仮説

固定公式graph指標でcheckpointを選ぶと、現行のedge accuracyとnode recallの積による選択より、外側の胚holdoutで良いepochを選べる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- 比較対象: 同じ学習runの現行proxy selector。exp005は既存基準の副参照。
- `metrics.json` の参照キー: `status=discarded`、`evidence.kaggle.kernel_version=1`、`evidence.kaggle.notebook_runtime_seconds=23633.582541708998`、`evidence.artifacts.model_count=6`、`evidence.artifacts.model_manifest_sha`、`evidence.reruns`のevaluation version 1失敗・選択結果。
- `SESSION_NOTES.md` の実行記録: 実装、静的検査、push、正常終了、実測T4 2基、runtime gate、生成物確認を記録。
- 参照する生成物: Kaggle train version 1の6 checkpoint、bundle/fold manifest、split、smoke/training summary。小規模な証拠だけを`/tmp/kaggle-output/exp007_graph_checkpoint_selection/train-evidence/`へ取得し、モデル本体はKaggle kernel outputを正とする。

## 解釈

trainは正常終了し、2 fold × 3 epochのcheckpointを保存できた。evaluation version 1は全6 checkpointの内部選択評価とruntime gateまで完了し、edge accuracyとnode recallの積による選択と固定公式graph指標による選択はいずれも両foldでepoch 2を選んだ。その後、full outer評価を始める前にKaggle working diskが枯渇し、PapermillのNotebook保存で失敗した。内部選択の中間生成物だけを評価・SHA計算後に削除する修正は実装・検証済みだが、再実行しても2つの選択方法へ同じcheckpointの予測を共有するため、選択方法間の差は得られない。したがって、この1回の学習runについて「固定公式graph指標が異なるcheckpointを選ぶ」という主張は支持されなかった。ただし外側2胚の公式指標と生成物SHAは未取得であり、異なる学習runや固定公開検出器下のトラッカーへ一般化しない。

## ユーザー判断

- 判断: `discarded`
- 確認日時 / 依頼メッセージ: 2026-09-12の「それでは閉じてください。最後にgit commitとpushしてください。」
- 理由: 両選択方法が両foldで同じepoch 2を選び、外側評価を再実行しても選択方法の差を検証できない。今後は公開検出器を固定してトラッカーを学習する方針であり、自前検出器を含むこのrunの外側score取得へGPU時間を追加投入しない。

## 次

再実行しない。次の作業は現行方針の`public_detector_selection`を優先する。
