# exp022_synthetic_division_teacher_audit 結果

## 仮説

公開合成系譜の完全な生成edgeなら、分裂母の正しい2娘組・同じ母の誤組・通常継続母の候補組を区別でき、元のdivision_triplets学習で不足した教師量を診断できる。

## 実行証拠

[公開合成データの生成Notebook](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset)のoutputを、[Kaggle private CPU診断Notebook](https://www.kaggle.com/code/kentookumura/exp022-synthetic-division-teacher-audit)で読んだ。version 1・2はともに先頭32/32時系列を完走した。最終記録はversion 2。GPUとinternetは無効。画像配列の値は読まず、node・edge・divisionと画像のschemaを検証した。診断処理は9.72秒で、Notebook全体の確定実行時間ではない。学習、公式CV、LB、submissionは行っていない。

source Notebook SHAは`04ed49b48c896dea94b2d47d5f59c00479d53a1cba109925642be0dccb9a97e5`、source manifest SHAは`e8b5376b2ac6fdd55bd6e45d1b07b401339d375b211b6f67be93fb0de4d8ce14`、metadata SHAは`328b9bb2545309e545cf68663ef985034ec68c36c0b709408a0f91801fedf89e`。選択した32ファイルは個別SHAとsizeを照合し、全時系列の系譜完全性を検証した。version 2の32行CSV SHAは`2f7971596f4a1b36cc99d681baa610bb26fecdae722beab0119da9370eb8a24f`、summary SHAは`f34970334f4eaba90f057fbaa4ea5fd6079d91c39b14396580da554b266710db`。取得したファイルとKaggle output manifestが一致した。取得物はignored `artifacts/kaggle_v2/`に保存し、構造化数値の正は[metrics.json](metrics.json)と同じKaggle outputのsummary.jsonとする。

| 9/14 µmの幾何候補と合成系譜 | 件数 |
| --- | ---: |
| 対象の分裂母 | 2,266 |
| 正しい2娘組を回収した分裂母 | 2,193（96.78%） |
| 分裂母から作られた誤組 | 6,637 |
| 正例を回収し、同じ母に誤組もある母 | 1,385 / 2,193（63.16%） |
| 通常継続母 | 42,220 |
| 候補組を持つ通常継続母 | 25,808 |
| 通常継続母から作られた候補組 | 76,092 |
| 候補組の合計 | 84,922 |

真の母娘距離は中央値3.97 µm、p95は7.49 µm、p99は10.83 µm。真の娘間距離は中央値7.06 µm、最大12.41 µm。したがって、この32時系列での未回収73分裂は母娘9 µm条件に起因する。全ての合成nodeを候補中心に使った値であり、固定公開検出器の回収率ではない。

## 解釈

生成系譜内では出力edgeが各時刻で完全なため、通常継続母の76,092組を「分裂ではない組」と判定できる。正例を回収した母のうち1,385母には同じ母の誤組もあり、母内の組を順位付けする学習用の比較が合成データでは成立する。exp021の実データでは正例100件に対し、同母の確定誤組は最大2件だった。両者は中心の生成方法、分裂頻度、注釈密度が違うため、件数や回収率の直接比較で学習効果を判断しない。

公開sourceのmetadataは分裂を意図的に多く生成したと記録している。さらにsource作者は画像のtexture・contrast差を説明している。[公開説明](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103)にも沿って、合成教師の豊富さはBiohub実画像での真の非分裂ラベルやモデル改善を示さない。固定公開検出器と特徴抽出器を通した候補で同じ教師が残るか、合成から実画像へ予測が転移するか、lossと復号をどう定めるかは未解決である。

推奨する次の判断は、この診断を完了とし、合成教師を使う小規模な学習実験の入力・検証を別途設計すること。実画像の未注釈組を確定負例へ変更せず、元のdivision_triplets候補は学習方式が確定するまで未着手とする。

## ユーザー判断

2026-09-20にユーザーはこの診断実験の完了を判断した。次は固定公開検出器から得る候補・特徴に合成系譜の教師を対応付けられるか、小規模に診断する。元のdivision_tripletsの学習方式、loss、復号は引き続き未確定。

## 後続の診断

当時の次の作業は[exp023](../exp023_synthetic_detector_teacher_audit/result.md)で実施済み。固定公開検出器を通した教師の回収と残る未検証事項は、その結果を参照する。当時の完了判断は変更しない。
