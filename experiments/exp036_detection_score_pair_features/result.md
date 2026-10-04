# exp036_detection_score_pair_features 結果

## 仮説

固定候補に付随する検出得点をcandidate pairごとの追加情報として使うと、固定画像特徴と位置だけを使う同構造対照より、隣接接続と分裂親の回収を改善できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 数値、実験status、Kaggle実行証拠、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- 実装前の比較契約と進行条件: [`requirements.md`](requirements.md)

## 実行証拠

- 比較対象: `neutral_score_control`対`detection_score_pair_residual`、外側2fold。
- Kaggle train run: version 2が`COMPLETE`。Notebook実行時間は8,882.15秒、学習・評価区間は7,472.25秒。
- 生成物: 4 model、fold別normalizer、各variant・foldの6得点帯診断、model manifestを取得した。
- SHA検証: model manifest、normalizer、GT window filter audit、4 checkpointの実ファイルSHAが記録値と一致した。
- 公式graph評価、Public LB、submission: 未実行。Notebookも`full_graph_inference_run: false`、`submission_created: false`を記録した。

| fold（外側評価胚） | positive edge recall 対照 | 得点あり | 差 | edge accuracy 差 | division parent recall 対照 / 得点あり | 支持件数（edge / division） |
|---|---:|---:|---:|---:|---:|---:|
| 0（6bba） | 0.9694079870 | 0.9693015968 | -0.0001063902 | -0.0000000472 | 0.4444444444 / 0.4444444444 | 103,393 / 108 |
| 1（44b6） | 0.9477544989 | 0.9473323131 | -0.0004221859 | 0.0000000000 | 0.2727272727 / 0.2727272727 | 18,949 / 22 |

差分は`detection_score_pair_residual - neutral_score_control`。positive edgeはfold 0で11件、fold 1で8件少なく、division parentの回収件数はそれぞれ48件と6件で同値だった。

## 解釈

両foldで支持件数は非0だったが、追加情報ありのpositive edge recallが対照を下回ったため、事前に定めた進行条件は未達となった。検出得点4特徴を4→16→1の残差headでprimary接続logitへ加える今回の実装は、同構造対照より隣接接続と分裂親回収を改善しなかった。

このnegative resultが否定するのは、固定候補、今回のfold内標準化、pair特徴、残差head、教師、3 epoch、外側2foldという条件での追加効果である。検出得点の別表現、別の融合位置、別objectiveまで否定しない。部分注釈により誤接続の絶対評価には限界があり、初段指標を公式scoreとして扱わない。

## ユーザー判断

- 判断: 不採用
- 確認日時 / 依頼メッセージ: 2026-09-23「不採用で終了し、commitとpushしてください」。
- 理由: 両foldでpositive edge recallが同構造対照を下回り、分裂親の回収も改善しなかったため、事前に定めた進行条件を満たさなかった。

## 次

全graph推論は実行せず、この実験は不採用で終了する。DoGの後続比較は[exp039の結果](../exp039_dog_features/result.md)に記録され、そこで検証した構成は不採用・完了となった。DoG全般の有効性を否定する判断ではない。HOG特徴の候補は未検証であり、この実験の完了だけで上位仮説全体を終了しない。
