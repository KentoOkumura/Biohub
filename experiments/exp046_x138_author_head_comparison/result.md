# exp046_x138_author_head_comparison 結果

## 仮説

採用済みexp043の座標補正checkpoint一式をx138作者版へ交換すると、同じ20動画で両胚の公式combined scoreが改善するかを調べた。公開画像モデル、tracker、後処理、評価器、元検出候補を固定し、headの`state_dict`と対応する`mean`・`scale`だけを交換した。自前headは[exp045の保存済み20動画結果](../exp045_x138_coordinate_effect_audit/result.md)を対照に使った。

## 実行証拠

[予備確認version 1](https://www.kaggle.com/code/kentookumura/exp046-x138-author-head-comparison-pilot)で両胚各1動画・両headを実行し、自前headの元候補SHA、最終graph 42ファイルのSHA、公式評価全行がexp045と一致した。[作者head全件version 1](https://www.kaggle.com/code/kentookumura/exp046-x138-author-head-comparison-author-only)はT4で正常完了し、両胚各10動画の最終graph、固定ID診断、公式評価を生成した。Notebook内所要時間は6794.812秒。作者headの予備2動画も全件再実行と最終graph 42ファイルおよび公式評価全行が一致した。

[比較スクリプト](compare_saved_results.py)は20動画の選択、両headと評価器のSHA、各動画の元候補SHAと候補数、固定した教師対応、公式評価行、取得した評価・診断・gateの4ファイルの出力SHAを照合して[比較JSON](artifacts/comparison.json)を作成した。3305ファイルのKaggle出力manifestを確認したが、全ファイルはローカル取得していない。数値とSHAの正本は[metrics.json](metrics.json)、時系列とコマンドは[SESSION_NOTES.md](SESSION_NOTES.md)。

## 公式評価

| 対象 | 自前head | 作者head | 作者−自前 |
| --- | ---: | ---: | ---: |
| 全体（20動画） | 0.890348 | 0.886189 | −0.004160 |
| 44b6（10動画） | 0.935902 | 0.933817 | −0.002084 |
| 6bba（10動画） | 0.873577 | 0.868707 | −0.004870 |

両胚ともcombined scoreは低下した。作者headが上回った動画は44b6で4/10、6bbaで5/10。動画別では`44b6_706092f0`が+0.055207、`44b6_87bba6c4`が−0.046591、`6bba_57b7cc1e`が−0.034952と、符号も大きさも一様ではない。全20動画の対応値はmetricsと比較JSONに記録した。予備2動画では両headとも再実行差が0だったが、残り18動画を反復した結果ではない。

44b6のadjusted edge Jaccardは0.921616→0.919532、6bbaは0.873577→0.868707に低下した。division TP/FP/FNは全体で両headとも1/8/16。公式edgeのTP/FP/FNは44b6で2899/174/142→2908/175/133、6bbaで7762/523/593→7770/553/585。44b6はmicro edge Jaccardが上がった一方、動画サイズで加重されたadjusted edge Jaccardとcombined scoreは下がった。

## 同じ候補IDでの診断

補正前候補への7 µm以内の対応を固定すると、最終段階で選択された既知edgeは44b6で2759→2783、6bbaで7333→7385と作者head側が多い。既知の分裂は44b6で1、6bbaで0のまま。既知中心との平均距離を対応済みGT node数で重み付けすると、44b6は1.556→1.464 µm、6bbaは1.740→2.028 µm。44b6では中心距離と既知edge回収が改善しても公式combined scoreは低下した。6bbaでは既知edgeの増加に対して公式edgeのFPが30増えた。固定IDで未注釈の辺を誤接続と断定できず、これらの集計だけで個々の接続差の原因を特定しない。

## 解釈

事前に定めた「両胚のcombined score改善」は満たされず、この条件付き20動画評価は作者headへの置換を支持しない。作者headの学習動画は不明で、公開画像モデルはcompetition train画像を学習済みのため、これは独立した交差検証ではない。exp043のPublic LB 0.950と作者の公開Notebook 0.953は同条件のhead交換結果ではなく、本実験ではPublic LB・hidden testとも未測定。位置変更と補間特徴の寄与を分離する追加診断も実施していない。

## ユーザー判断

作者headをexp043の既定checkpointへ置換しないことを推奨する。採否とexp046の完了はユーザー判断待ち。competition submissionは行っていない。

## 次

ユーザーが作者headの採否とexp046の完了を判断する。Public LBでの比較を行う場合は、別途competition submissionの明示承認を受けてから実施する。
