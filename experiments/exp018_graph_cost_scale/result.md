# exp018_graph_cost_scale 結果

## 仮説

exp015の固定ILP前候補graphでedge costだけを`-alpha * edge_prob`へ変えると、固定した出現・消失・分裂費用との相対スケールが整い、現行`alpha=1`より正しい接続と分裂をILP解に残せる可能性がある。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 実験status、kernel情報、Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- 比較契約と後段gate: [`requirements.md`](requirements.md)

## 現在の状態

初段のKaggle CPU実行は完了した。5つのalpha shardは各199件をfailure 0で処理し、aggregate version 2が全995行、2胚、5 alphaのSHAとcoverageを検証した。promotion gateは`false`で、契約どおり固定repair段階は実行していない。Kaggle submissionも実行していない。

44b6で選択した`alpha=0.5`は、6bbaでcombined scoreを`0.9165304582867454`から`0.916798008840978`へ`+0.00026755055423255403`改善し、division Jaccardは`0.0`のままだった。一方、6bbaで選択した`alpha=2`は、44b6でdivision Jaccardを`0.0`から`0.03968253968253968`へ改善したが、combined scoreを`0.9061411877889494`から`0.9036626645370643`へ`-0.002478523251885023`低下させた。両方向でcombined scoreが厳密改善する条件を満たさない。

## 実行証拠

最終証拠は`artifacts/kaggle_aggregate_v2/artifacts/ilp_only_v1/`へ回収した。Kaggle aggregateは`kentookumura/exp018-graph-cost-scale-aggregate` version 2、CPU、internet無効で実行し、aggregate処理は`731.5177280590001`秒、5 shardの処理時間合計は`22380.816667248`秒だった。

- sample数: 199、per-sample行数: 995、failure数: 0。
- candidate graph bundle SHA: `3df8d9fe071e10cfca159fffbe1636ae9b1131c538021b1c3621cf1ab4d155e9`。
- solution bundle SHA: `c07b6f85326b5ffafe68d94d706e31757bf1ced4c3990aa69fe28722448fd89c`。
- per-sample SHA: `459726bc34e3bf9d819fb1e178afc73e0309e97b22c1f51f63e348e0a4a5b74b`。
- embryo-alpha SHA: `a07345a4c821f808f4f107dc2b59dc2359d6151cd6a146b30ea578452fc84525`。
- cross-embryo selection SHA: `1cbb3b12d2673949bbbcdc93fdb4f6c84c14303a0b0c3f92694b87699959d4c8`。
- summary SHA: `6f8de7dc6837f32fbb9c20f5ff8e7a4ad5f8d3fdcaccd0d31ac6b2a84fbfb898`。
- manifest payload SHA: `4d553972be05e05c1014cb2cdd8f26270971c6e8f4eaf64b14da0399eb70aa29`。
- manifest file SHA: `777cf81fe3265792766173314a4146a51535135f973bfa4dace4fc0176038344`。

構造化された全score、方向別判断、kernel source version、runtimeは`metrics.json`を正とする。途中の失敗証拠は`artifacts/kaggle_v3_failed/`、各成功shardは`artifacts/kaggle_alpha_*`へ分離して保存している。

## 解釈

`alpha=1`以外の費用スケールが一方向では小さく改善したが、逆方向で再現せず、事前に固定した両方向gateを通らなかった。したがって、`graph_cost_scale`単独を固定repair比較へ進める根拠は得られていない。これはrepair前の固定候補比較であり、公開モデルの学習来歴により独立CVではないため、手法一般やrepair後の効果を否定する結果ではない。

## 後段gate

次のすべてを満たす場合だけ、同じ実験内で固定repairを含む比較へ進む。

- 2方向とも、選択alphaの外側胚combined scoreが`alpha=1`を厳密に上回る。
- 2方向ともdivision Jaccardが悪化しない。
- 5 alphaの全199件が有効で、失敗と非有限metricがない。
- 入力manifest、candidate graph、公式評価source、solver、solutionの実行証拠が保存される。

gateを通っても、この時点ではrepair後の改善、Public LB改善、実験の採用・完了を確定しない。Kaggle submissionは行わない。

実測では全件coverage、division非悪化、実行証拠の条件は満たしたが、6bbaで選んだ`alpha=2`がheld-outの44b6 combined scoreを低下させたためgateは失敗した。次段は`stop_without_full_repair`であり、固定repair段階は実装・実行しない。

## ユーザー判断

- 判断: 実験は完了、`graph_cost_scale`は不採用（`discarded`）。
- 理由: 初段は全件coverageと実行証拠を満たしたが、事前に固定した両方向のcombined score改善gateが不成立だったため、固定repair段階へ進める根拠がない。
- 判断が閉じる範囲: exp015の固定候補graph、固定event cost、`alpha=[0.25, 0.5, 1, 2, 4]`、胚を入れ替える2方向評価におけるedge cost scalar単独変更。異なる候補生成、学習済みtracker、別のevent cost、repair後の手法一般は棄却しない。
- 確認日時 / 依頼メッセージ: 2026-09-16「実験は完了、手法は不採用（discarded）として確定でいいです。」
