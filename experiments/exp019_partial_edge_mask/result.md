# exp019_partial_edge_mask 結果

## 仮説

exp016と同じ固定候補・正例行列・primary tracker初期値・復号で、正例のない子列への接続損失項だけを除くと、未記録の第2娘を抑える学習が減り、両胚の公式graph指標が改善するかを検証する。未知親はsoftmax分母に残る。

## 実行証拠

- [設定と系譜](config.yaml)、[数値と実行状態](metrics.json)、[実行の時系列](SESSION_NOTES.md)。主対照は[exp016の公式graph評価](../exp016_frozen_image_encoder/result.md)。
- [Kaggle train Notebook](https://www.kaggle.com/code/kentookumura/exp019-partial-edge-mask-train) version 1は`COMPLETE`。変更したprimary trackerだけを2foldで学習した。fold 0は3 epoch目、fold 1は1 epoch目を選択。Notebook全体4264.85秒。2つのmodel file SHAとmodel manifest SHAを取得物で照合した。
- [Kaggle inference Notebook](https://www.kaggle.com/code/kentookumura/exp019-partial-edge-mask-inference) version 1は`COMPLETE`。T4を2枚使用し、追加学習0、読み込みtracker 2、main画像encoder forward 0。Notebook所要17645.01秒、うちcache replay予測4352.74秒。trainとinferenceのNotebook所要合計は約6時間05分。
- Kaggle出力の`official_graph_evaluation.json`、`official_graph_per_sample.json`、`graph_inference_receipt.json`、`official_evaluator_preflight.json`を`artifacts/kaggle_inference_v1/`へ取得した。公式評価JSONのSHA256は`ce16bbb7afec7b4e765a8c89dc58e04c96fa3829d17ea189dc27fc08801a943a`。receipt内のSHAと一致し、exp016保存済みの対照指標も完全一致した。公開trackerの候補graphと固定候補座標の一致、公式評価器の事前検証を確認した。
- 対象199動画をすべて評価し、失敗・skipは0件。個別評価JSONの199件の一意性と両胚の71件・128件を照合した。competition submissionは作成していない。GEFF graph本体はローカルに全件取得しておらず、その内容はKaggle側のcompact graph SHAと評価receiptを根拠にする。公開画像encoderと初期trackerの学習来歴にtrain動画が含まれるため、胚別比較は独立CVと呼ばない。

## 公式graph評価

主指標はadjusted edge Jaccardとdivision Jaccardの加重和。差分はexp019からexp016を引いた値。

| 対象 | 動画数 | exp016 score | exp019 score | score差分 | adjusted edge Jaccard（exp016 → exp019） | division Jaccard（exp016 → exp019） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全体 | 199 | 0.912055 | 0.908982 | -0.003073 | 0.901809 → 0.899458 | 0.102459 → 0.095238 |
| 44b6 | 71 | 0.909766 | 0.910124 | +0.000358 | 0.900151 → 0.900690 | 0.096154 → 0.094340 |
| 6bba | 128 | 0.912531 | 0.908778 | -0.003753 | 0.902114 → 0.899231 | 0.104167 → 0.095477 |

全体のdivision件数はexp016のTP 25・FP 93・FN 126に対し、exp019はTP 24・FP 101・FN 127。node recallは0.982227から0.985826、通常のedge Jaccardは0.899732から0.900131へ上がったが、主指標のadjusted edge Jaccardとdivision Jaccardは下がった。

## 解釈

正例子列へのmask変更は、従来maskのpairの48.87%を損失から除いた。一方、残ったpartial maskのpair 28,669,782件のうち、未知親を含むものは27,432,958件だった。未知親がsoftmax分母と負例項に残るため、この実験だけで疎な注釈への対処全体を判断できない。

事前に定めた「両胚の公式scoreが改善する」という条件は満たさなかった。44b6の小幅な改善より6bbaの低下が大きく、全体scoreは0.003073低下した。この結果が否定するのは、固定候補・固定復号・3 epochs・今回の子列maskという条件での改善であり、softmax分母や教師、検出側を変える案の結果は含まない。現時点ではexp016の主経路への置き換えを推奨しない。

## ユーザー判断

- 2026-09-20: ユーザーがexp019の不採用と実験完了を判断した。`metrics.json`のstatusを`discarded`に確定した。固定候補・固定復号での主経路はexp016の保存済みtrackerを維持する。

## 次

正例子列へのmask変更だけでは両胚の公式scoreを改善できなかった証拠を、次候補の教師設計に引き継ぐ。追加の方式変更やKaggle submissionは今回の実行に含めない。
