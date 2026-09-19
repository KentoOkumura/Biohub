# exp020_division_triplet_candidates 結果

## 仮説

固定公開検出器の中心候補から母と2娘の組を作ると、現行candidate edgeに限る61/151件より多くの既知分裂を回収できるかを診断した。組スコアによる精度改善はこの実験の検証対象ではない。

## 実行証拠

Kaggleのprivate CPU Notebook [version 1](https://www.kaggle.com/code/kentookumura/exp020-division-triplet-candidates-diagnostic)が199動画・19,701 windowで完走した。GPU学習、モデル生成、公式CV、LB、submissionは行っていない。実行部分の経過時間は1,209.55秒。Kaggle live SSEで観測した最終ログ時刻は起動後1,257.49秒で、Notebook全体の確定実行時間とは扱わない。

exp015 cache identity SHAは 440891c4550adf8540e3c47f9784e68b6ca1b64bbe56dbb93a25eb87c46bc1ee と一致した。GT内容bundle SHAは 05459d7c5b398690849590d1032c8883061bd3e26c9df617e10c6110c9dbf2d4。Kaggle生成物をローカルのignored artifacts/diagnostic_v1にも取得し、597行の表SHA 2f299da75b333ce659503716b815827b8d14b68696351089c231ac018a179247 とsummary SHA b28beb7c24a36df5b7677bfae40704aa6351a32c5a645bac0f4dae7bdbcd7899 がmanifestと一致した。詳細な数値の正は metrics.json とKaggleのsummary.jsonとする。

| 事前固定した幾何条件（母娘 / 娘間） | 組候補数 | 既知分裂の回収 | 44b6 | 6bba | 既知分裂母で確定した誤組 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 最終分裂geometry filter（10.5 / 8 µm） | 4,078,872 | 23 / 151（15.23%） | 4 / 26 | 19 / 125 | 1 |
| safe division repair（9 / 14 µm） | 8,807,617 | 100 / 151（66.23%） | 20 / 26 | 80 / 125 | 4 |
| 広い診断範囲（14 / 14 µm） | 61,810,786 | 129 / 151（85.43%） | 24 / 26 | 105 / 125 | 11 |

母と2娘の3中心すべてがGTへ対応した分裂は145/151件だった。既存のILP前candidate edge graphで組が揃った61/151件、最終graphの15/151件はexp015の別段階の数値であり、この実験の幾何候補回収と同じ処理ではない。

## 解釈

GTを使わず固定中心から組を列挙すると、9/14 µmでは現行candidate edge graphの61件より多い100件が候補に入る。広い14/14 µmでは129件だが、候補数は6,181万件となる。候補の幅だけでは、推論時間や分裂precisionを改善できるとは言えない。10.5/8 µmの娘間条件では23件しか残らず、娘間距離の条件が回収を大きく左右する。

GEFF node propertyは node_id、t、x、y、z だけで、完全注釈領域を示す既知の列はなかった。出力edgeが1本の注釈は128,581件あるが、第2娘が記録されていないだけの可能性を除外できないため、確定した通常継続の負例は0件として扱った。この0は通常継続が実際に存在しないという意味ではない。GTへ対応した確定誤組も9/14 µmでは4件で、現状の教師だけを使った組の分類・順位学習は根拠が足りない。

この結果は公開検出器を固定したtrain胚での条件付き診断であり、公開重みの学習来歴から独立CVと呼べない。公式combined scoreへの影響は未測定である。

## ユーザー判断

2026-09-19、ユーザーはこの診断を完了と判断し、元のdivision_tripletsの学習は教師の根拠を調査してから設計する方針を選択した。exp020は診断実験として完了。学習候補は設計不可のまま残す。
