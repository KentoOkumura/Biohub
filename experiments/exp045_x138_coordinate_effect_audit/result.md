# exp045_x138_coordinate_effect_audit 結果

## 仮説

同じ検出候補と公開trackerで座標headだけを切り替えれば、中心距離の変化と、既知接続の得点・候補残存・graph処理後の回収の変化を段階別に区別できる。

## 記録の参照先

- 実験条件と系譜: [`config.yaml`](config.yaml)
- 実験status、数値、Kaggle実行証拠: [`metrics.json`](metrics.json)
- 実行経過と停止理由: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- Kaggle予備実行の生成物一覧とSHA: [`artifacts/pilot/exp045_pilot_receipt.json`](artifacts/pilot/exp045_pilot_receipt.json)（ローカル生成物、Git管理外）

## 実行証拠

[Kaggle Notebook version 4](https://www.kaggle.com/code/kentookumura/exp045-x138-coordinate-effect-audit-pilot) がT4・internet無効で `COMPLETE`。両胚から事前固定した各1本、計2本を補正なし・ありの両条件で最終graphまで処理し、同じ公開評価器で採点した。20本の選択名、headとmanifestのSHA、入力候補IDの一致、各段階の生成物SHAはreceiptに記録した。予測処理はGEFF注釈を参照していない。competition submissionは行っていない。

| 予備実行2本の公式評価器集計 | 補正なし | 補正あり |
| --- | ---: | ---: |
| adjusted edge Jaccard | 0.889538 | 0.916229 |
| edge Jaccard | 0.885428 | 0.899103 |
| node recall | 0.962396 | 0.971600 |
| division true positive / false negative | 0 / 3 | 0 / 3 |

両条件の元検出候補は、44b6動画で54,942件、6bba動画で36,330件と一致し、補正後の最大移動量はそれぞれ1.83 µm、1.91 µmだった。予備実行22.7分から、25%の余裕を含む20本の推定所要時間は4.63時間。生成物は663ファイル、約352 MBだった。パイロット2本の指標は主評価の20本の結果として扱わない。

### 20動画の主評価

[Kaggle GPU推論version 1](https://www.kaggle.com/code/kentookumura/exp045-x138-coordinate-effect-audit-inference)は両胚各10本・2条件の最終graphを40本生成した後、GT GEFFの返り値をtupleとして扱わず固定ID診断で停止した。予測結果は6,604ファイルすべて回収し、選択manifest・実行条件のSHAと、各条件20本の候補cache・ILP graph・最終graph・9段階の保存物を確認した。GT読込を修正した元NotebookからGPU推論を繰り返していない。

保存済み最終graphを非公開Datasetへ移し、[Kaggle CPU公式評価version 3](https://www.kaggle.com/code/kentookumura/exp045-x138-coordinate-effect-audit-official) が `COMPLETE`。公開評価器のsource SHAを照合し、20動画×2条件の全行を評価した。評価器が使わない画像読込だけを `load_image=False` とし、同じGT graph・voxel scale・評価関数を使った。入力842ファイルのmanifest SHAは `868282ef3065bfb7d30249a38d80234d3681856d9284559d82f2af5ab3b788a2`。数値の正本は [`metrics.json`](metrics.json)、実行receiptは [`artifacts/official_v3/exp045_official_resume/exp045_official_receipt.json`](artifacts/official_v3/exp045_official_resume/exp045_official_receipt.json)（Git管理外）。

| 公開評価器によるtrain内20動画集計 | 補正なし | 補正あり | 差 |
| --- | ---: | ---: | ---: |
| score | 0.879239 | 0.890348 | +0.011109 |
| adjusted edge Jaccard | 0.875906 | 0.886348 | +0.010443 |
| edge Jaccard | 0.876180 | 0.881584 | +0.005404 |
| node recall | 0.976985 | 0.974275 | -0.002710 |
| 分裂 TP / FP / FN | 1 / 13 / 16 | 1 / 8 / 16 | FP -5 |

胚別scoreは44b6が0.917297→0.935902、6bbaが0.864930→0.873577。動画別adjusted edge Jaccardは44b6で10/10本改善したが、6bbaでは4/10本改善、6/10本悪化した。後者の集計改善は一部の大きな改善による。

同じ元検出IDを固定して読む段階別診断は、回収した20動画のGT GEFF 420ファイルを使ってローカルで完了した。[診断receipt](artifacts/local_fixed_id_v1/local_fixed_id_receipt.json)と[胚別集計](artifacts/local_fixed_id_v1/fixed_id_by_embryo.json)はGit管理外で、SHAは`metrics.json`へ記録した。両条件の元検出候補680,033件と2,000 frameの生座標が一致し、最大補正量は1.912 µmだった。

| 固定IDの既知edge回収 | 母数 | 候補: なし→あり | ILP: なし→あり | 最終: なし→あり |
| --- | ---: | ---: | ---: | ---: |
| 44b6 | 3,041 | 2,880→2,836 | 2,807→2,768 | 2,787→2,759 |
| 6bba | 8,355 | 7,680→7,612 | 7,471→7,385 | 7,413→7,333 |

両胚で固定IDの既知edge回収は候補段階から減った。注釈された母との確定的な矛盾は最終graphで44b6が1→0、6bbaが4→4。既知分裂の両娘edgeが最終graphに残った数は44b6が1/4→1/4、6bbaが0/13→0/13。未知のedgeを負例として数えていない。

## 解釈

主結果は、headの学習に使わなかった20動画で、公開評価器のscoreが0.879239から0.890348へ改善したこと（+0.011109）である。両胚の集計でも改善した。一方、元座標で一度対応を固定した同じ候補ID間の既知edge回収は両胚で減った。これは主結果を否定しない。公開評価器は補正後の最終座標からGTへ対応し直してgraph全体を採点するため、固定ID診断とは測る対象が違う。このscore上昇を同じ検出ID間の接続改善だけで説明することはできないが、どの作用が改善を生んだかは現時点では確定していない。

融合後の既知edge順位または候補残存が変わり、同じedgeのILP以降の回収も変わったものは183件、18動画にあった。要件に記した追加pair診断の実施条件は成立した。この診断では、同じ候補ID対に対して、元座標・補正座標と、画像特徴を元位置・補正位置のどちらで取得するかを組み合わせた4条件の融合後得点・順位を比べる。座標入力と特徴取得位置の寄与を調べるための診断であり、score改善の成立を確認するための追加評価ではない。2026-09-25の確認時点でKaggle GPU残枠は0.00時間、更新は2026-09-26 00:00 UTCだったため未実行で、両経路の寄与は分離できていない。

対象20動画は座標headにとって動画単位の学習外だが、固定公開画像モデルはこれらのtrain画像を学習済みである。ここで示したのは固定公開モデル下の条件付きtrain subset評価であり、独立CV、Public LB、hidden testのscoreではない。分裂のGT母数は17件と少ない。competition submissionは行っていない。

## ユーザー判断

- 2026-09-26: ユーザーが「実験は閉じてcommitとpushしてください」と明示。実験statusを`completed`にする。
- 固定した公開画像モデル・trackerの構成では、head使用によるtrain内20動画のscore改善を確認したため、同構成でheadを使う判断を支持する。独立CV・Public LB・hidden testでの改善は未確認であり、この実験をその証拠とは扱わない。
- 追加pair診断は実施せずに閉じる。座標入力と画像特徴の取得位置の寄与は未分離のまま残す。

## 次

未知動画またはPublic LBでの効果を確かめる場合は、評価条件を定めた別実験で比較する。exp045での追加実行は予定しない。
