# exp024_tracker_six_epochs 結果

## 仮説

固定画像特徴からprimary trackerを最大6エポック学習し、[exp016の保存済み3エポック基準](../exp016_frozen_image_encoder/result.md)と同じ候補・復号で比較する。両foldで4～6エポック目の重みが内部検証から選ばれた場合だけ、199動画の公式graph評価へ進む。

## 実行証拠

- [Kaggle学習Notebook version 1](https://www.kaggle.com/code/kentookumura/exp024-tracker-six-epochs-train)はT4で正常完了した。2foldとも6エポックを実行し、全12件のエポックcheckpointと最良モデル2件を取得した。保存ファイルのSHAはKaggle出力のmanifestと一致した。
- 実測学習時間は6,794.58秒、Notebook全体は7,512.31秒（約2時間5分）。構造化された実行証拠、各エポックの指標、model SHAは[metrics.json](metrics.json)の`train_stage`と`evidence`、時系列は[SESSION_NOTES.md](SESSION_NOTES.md)を正とする。取得した生成物は`artifacts/kaggle_train_v1/`に保存した。
- 内部検証で選ばれた最良checkpointはfold 0が2エポック目、fold 1が4エポック目。fold 0の選択重みはexp016の3エポック基準とstate SHAまで一致した。fold 1の4エポック目は2エポック目と選択値が同点で、後のcheckpointを選ぶ既存ルールにより選択された。
- 続行条件「両foldで4～6エポック目が選ばれる」は不成立だった。契約どおり公式graph推論は実行しておらず、exp024の公式combined score、Public LB、submissionはない。exp016の公式combined score 0.9120545013は保存済み対照であり、exp024との公式指標比較結果ではない。

## 解釈

この6エポック一変更では、少なくともfold 0で追加学習した重みが選ばれず、事前に決めた次段の条件を満たさなかった。fold 1で後半重みが選ばれたのも内部検証値の同点によるもので、精度向上の証拠とは扱わない。外側胚の接続診断は[metrics.json](metrics.json)に残したが、固定候補・復号後の公式graph scoreの代わりにはならない。公開初期モデルの学習来歴により、この胚別評価も独立CVとは呼ばない。

## 2時点の接続診断（追加実行）

[Kaggle診断Notebook version 1](https://www.kaggle.com/code/kentookumura/exp024-tracker-six-epochs-diagnostic)で、3エポック実験の選択重みと同一state SHAの2エポック目、および6エポック目を同じ外側胚の2時点windowで比較した。18,707 windowを処理し、Notebook全体は1,048.48秒だった。件数、動画別内訳、モデルと入力のSHAは[metrics.json](metrics.json)の`tracker_pair_diagnostic`と、Kaggleから取得した`artifacts/kaggle_diagnostic_v1/tracker_pair_diagnostic.json`を正とする。

| 外側評価胚 | 既知親を持つ子 | 接続Jaccard 2→6エポック | 親1位正答率 2→6エポック | 誤接続 2→6エポック | 動画別Jaccard 改善/悪化/同点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6bba | 103,393 | 0.95686→0.95301 | 0.97923→0.97776 | 1,635→1,742 | 24/88/16 |
| 44b6 | 18,949 | 0.93116→0.93049 | 0.96844→0.96828 | 444→459 | 23/32/16 |

この比較では、6エポック目のprimary trackerは両胚で改善していない。6bbaでは正しい接続の回収も100,497→100,195件に減り、44b6では回収が18,058→18,059件とほぼ同じ一方で誤接続が増えた。既知分裂親の両娘接続も6bbaで50/108→48/108、44b6で6/22→5/22だった。分裂の母数は小さいため、この項目だけで優劣を確定しない。

この接続Jaccardは、注釈上の親が候補内にある子だけを対象とする、graph構築前の限定指標である。注釈のない子、候補から漏れた既知接続、双方向・secondary trackerの融合、ILP、graph repairは採点していない。公式のadjusted edge Jaccardやdivision Jaccardの値ではなく、公式scoreとPublic LBは引き続き未計測である。公開初期重みの学習来歴上、外側胚の比較は独立したCVではない。ユーザーが選んだとおり、この診断でcheckpoint選択規則や当初のgraph続行条件は変更しない。

## ユーザー判断

2026-09-20にユーザーはこの6エポック条件を**不採用として実験完了**と判断した。`metrics.json`の実験statusを`discarded`へ変更した。内部検証で両foldの後半重みが選ばれる条件は成立せず、追加の2時点接続診断でも6エポック目の改善は確認できなかったため、exp024の全graph推論と公式評価は実行していない。公式score・Public LBが未計測であることを維持する。

この判断の範囲は、exp016と同じ固定特徴・教師・損失・分割のまま学習上限だけを6エポックへ延ばした条件である。trackerの別の教師、構造、学習条件まで不採用とするものではない。
