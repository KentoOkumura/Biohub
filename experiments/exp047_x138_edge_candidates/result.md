# exp047_x138_edge_candidates 結果

## 仮説

exp043の融合後接続確率から、対照の `p > 0.48` の全辺を残し、`p > 0.10` かつ娘ごと上位3親の辺を追加した。検出点、得点、ILP費用、後処理、評価器は固定した。head学習側20動画の前段診断と両胚各1動画のpilotを通過し、head学習外の両胚各10動画をKaggleで評価した。

## 実行証拠

- [Kaggle pilot version 3](https://www.kaggle.com/code/kentookumura/exp047-x138-edge-candidates-pilot)は完了。固定2動画の公式combined scoreは対照0.953039、拡張0.950998。進行条件の確定的な誤親増分は0で、同じ候補・ILP出力をversion 2から再現した。20動画の候補回収診断はpilot version 1で取得した。
- [Kaggle inference version 1](https://www.kaggle.com/code/kentookumura/exp047-x138-edge-candidates-inference)は`KernelWorkerStatus.COMPLETE`。head学習外20動画の選択は各胚10動画で、head学習動画との重複は0。対照の公式評価20行はexp045のrefined構成と数値まで一致した。
- [metrics.json](metrics.json)に胚別・動画別の公式指標、段階別の既知接続、ILP費用、Notebook所要時間、最終graphのSHAを記録した。回収した小さなJSONはローカルの[評価証拠](artifacts/evidence/inference_v1/)と[pilot証拠](artifacts/evidence/pilot_v3/)に保存した。公式評価JSONのSHA256は`bce246fa9783db769ece67d9959c1a5939edb74c37f0df9157070219566ed59e`、receiptは`624f2894de140e817425a360f169e62d1cb5c1e68040a6411edc6befa16267ae`。

## 公式評価

| 対象 | 対照combined score | 拡張combined score | 差 | division Jaccard（対照→拡張） |
| --- | ---: | ---: | ---: | ---: |
| 44b6、10動画 | 0.935902 | 0.929943 | -0.005959 | 0.142857→0.166667 |
| 6bba、10動画 | 0.873577 | 0.859244 | -0.014333 | 0→0 |
| 全20動画 | 0.890348 | 0.878088 | -0.012261 | 0.040000→0.045455 |

node recallは全体で0.974275→0.983619へ上がったが、調整済みedge Jaccardは0.886348→0.873542へ下がった。動画別の調整済みedge Jaccardは44b6で7/10、6bbaで8/10動画が低下した。公式評価のedge true positiveは44b6で2,899→2,915、6bbaで7,762→7,801となった一方、false positiveは174→194、523→635へ増えた。

## 候補から最終graphまで

| 胚 | 候補で選ばれた既知接続 | ILP後の既知接続 | 最終graphの既知接続 | 候補の既知2娘 | ILP後の既知2娘 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 44b6 | 2,841→2,992 | 2,775→2,822 | 2,764→2,805 | 1→2 | 0→0 |
| 6bba | 7,621→8,017 | 7,384→7,561 | 7,338→7,501 | 3→11 | 0→0 |

同じ元候補IDに固定した対応での既知接続は増えた。しかし候補に残した既知2娘はILPで同時選択されず、最終graphの既知2娘は44b6で1→1、6bbaで0→0だった。部分注釈上で親が確定した娘への最終graphの矛盾は44b6で0→0、6bbaで4→3だが、これを未知接続の正しさとはみなさない。公式評価では上記のfalse positive増加が見られた。

## 解釈

候補辺の合計は547,895→904,102本。拡張ILPは全20動画で最適終了し、最大814.1秒/動画（上限1200秒）。Notebookは9,847秒（約2時間44分）、メモリ最大22.39 GB / 32.21 GB（69.5%、上限80%未満）。GPU quotaの終了後残量は35.84時間。費用条件は満たしたが、「両胚でcombined score改善」という成功条件は満たさなかった。

この20動画はhead学習外だが、公開画像モデルが学習に使った画像なので独立CVではない。Public LBとcompetition submissionは未実施。固定条件の比較結果から、この候補拡張を現行構成へ採用しないことを推奨する。

## ユーザー判断

2026-09-26、ユーザーの「先ほど実行した実験は閉じでください」により、この実験を完了とした。両胚でcombined score改善という事前の成功条件を満たさなかったため、候補拡張は現行のexp043推論へ組み込まない。`discarded`としての不採用判断やcompetition submissionの指示は受けていない。

この判断が閉じるのは、exp043の固定得点から`p > 0.10`かつ娘ごと上位3親を追加し、ILP費用・後処理を固定した1構成の比較である。接続得点の再学習、ILP費用・制約の変更、別候補源や局所graph選択の有効性は未判断。次の切り分けは[接続選択の段階別診断](../../backlog/x138_edge_selection_diagnostic.md)に記録した。
