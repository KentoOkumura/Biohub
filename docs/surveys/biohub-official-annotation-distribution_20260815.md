---
title: Biohub 公式アノテーション分布調査
date: '2026-08-15'
types:
- survey
hypotheses: []
experiments: []
topics:
- data
- annotation
status: final
summary: 公式GEFFは全体で推定細胞nodeの2.82%を収録し、胚別annotation率とsample間分布に大きな差がある。
---

# Biohub 公式アノテーション分布調査

作成日: 2026-08-15

## 結論

- train 199 sampleの公式GEFFには133,318 nodeと128,883 edgeがあり、GEFF metadataの`estimated_number_of_nodes`合計4,725,117に対するnode数の比率は2.82%だった。これはsample別比率の単純平均6.13%とは異なり、推定全node数で重み付けした全体比率である。
- sample別の比率は0.13%～20.21%で、中央値3.56%だった。胚別の重み付き比率は`44b6`が0.77%、`6bba`が5.37%で、公式アノテーションの密度は胚間でも均一ではない。
- 140 / 199 sampleでは100 frameすべてに1個以上のannotationがある。しかし、1 active frame当たりのannotation node数のsample中央値は6.61にすぎず、全frameにannotationがあることは、そのsampleまたはframeが完全にannotationされていることを意味しない。
- GEFFにはannotation済みnode・edgeとsample単位の推定全node数はあるが、完全にannotationされた空間・時間領域を示すmaskはない。したがって、公式データだけから「この領域内の全細胞がannotation済み」と判定したり、未アノテーション細胞の正確な位置分布を求めたりすることはできない。
- 手動で完全にannotationした領域を学習へ使う場合は、疎な公式ラベルと同じ教師データとして単純に混ぜない。公式データの未アノテーション位置は未知として扱い、完全annotation領域だけで負例を定義できるよう、領域maskとデータの由来を分ける必要がある。

## 調査目的

公式アノテーションが各sampleの全細胞をどの程度覆っているか、胚・sample・時間・空間・tracking graphの連結成分ごとに分布を確認する。あわせて、公式データ内に完全にannotationされた領域があると判断できるか、手動annotationを追加する際に公式ラベルと同じ扱いができるかを明確にする。

## 対象と証拠範囲

- 対応する上位仮説: なし
- 対象実験: なし。公式trainデータの独立した分布調査である。
- 入力データ・生成物: Kaggle上のcompetition inputにあるtrain 199 sampleの`.geff`。画像`.zarr`の輝度chunkは読み込んでいない。
- 実行環境: Kaggle Notebook [`biohub-annotation-distribution-audit`](https://www.kaggle.com/code/kentookumura/biohub-annotation-distribution-audit)。集計日時は2026-08-15 00:26:55 UTC。
- 集計方法: GEFFのnode ID、時刻`t`、座標`z, y, x`、edge、`estimated_number_of_nodes`を読み、sample別・胚別のannotation比率、時間・空間histogram、連結成分を集計した。空間分布は各軸を正規化して8等分した。
- このレポートからは判断できないこと: 真の未アノテーション細胞の座標、画像上の細胞密度を基準にした位置別annotation率、公式annotation対象の選択規則、局所的に完全annotationされた領域の有無。これらには画像から独立に作った細胞候補または完全な手動annotationと、評価対象領域を明示するmaskが必要である。

## 集計方法

各sampleについて、次を計算した。

- `annotated_nodes / estimated_number_of_nodes`をsample別annotation比率とした。`estimated_number_of_nodes`はGEFF metadataにある概算値であり、完全な正解node一覧ではない。
- nodeが1個以上あるframe数をactive frame数とした。最初と最後のannotation時刻の間にnodeが0個のframeがあるsampleも数えた。
- edgeを無向に接続して連結成分を求め、その個数とnode数を集計した。
- edge両端の時刻差を確認し、隣接frame以外を直接結ぶedgeの有無を調べた。
- 空間座標はvolume shape `(Z,Y,X)=(64,256,256)`で正規化し、各軸を8 binに分けた。これはannotation node自体の位置分布であり、真の細胞分布で割ったannotation率ではない。

## 分析結果

### 全体とsample別annotation比率

| 指標 | 値 |
| --- | ---: |
| sample数 | 199 |
| annotation node | 133,318 |
| annotation edge | 128,883 |
| 推定全node数 | 4,725,117 |
| 推定全node数で重み付けしたannotation比率 | 2.82% |
| sample別annotation比率の単純平均 | 6.13% |

| sample別annotation比率 | min | 5%点 | 25%点 | median | 75%点 | 95%点 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 比率 | 0.13% | 0.25% | 1.04% | 3.56% | 11.80% | 16.55% | 20.21% |

全sampleで、annotation node数はsample単位の推定全node数を大幅に下回った。ただし、このsample単位の比率から局所領域の完全性までは判断できない。

### 胚別分布

| 胚 | sample数 | annotation node | 推定全node数 | 重み付きannotation比率 | sample別比率のmedian | sample別比率の範囲 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `44b6` | 71 | 20,197 | 2,618,970 | 0.77% | 0.77% | 0.13%～5.02% |
| `6bba` | 128 | 113,121 | 2,106,147 | 5.37% | 9.71% | 0.93%～20.21% |

`6bba`の重み付きannotation比率は`44b6`の約7倍だった。手動annotationの件数を公式データのsample数だけに比例させても、この密度差は再現されない。また、この差が細胞密度、取得条件、annotation対象の選択規則のどれに由来するかはGEFFだけでは分離できない。

### 時間方向の分布

| 指標 | 値 |
| --- | ---: |
| active frame数の範囲 | 40～100 |
| active frame数のmedian | 100 |
| active frame数の平均 | 95.14 |
| 100 frameすべてにannotationがあるsample | 140 / 199 |
| annotation期間内に空frameがあるsample | 25 / 199 |
| 隣接frame以外を直接結ぶedgeがあるsample | 0 / 199 |

1 active frame当たりのannotation node数は、sample別に1.00～20.05、中央値6.61だった。全sample合計の時刻別node数は`t=0`で983、`t=46`で最大1,419、`t=99`で1,063だった。時間の中央付近がやや多いが、全時刻にannotationが存在する。

### 空間方向の分布

各軸の8 binすべてにannotation nodeがあった。中央2 bin、すなわち正規化座標0.375～0.625に含まれるnodeの比率は、z軸30.5%、y軸33.0%、x軸30.7%だった。annotation nodeは中央寄りだが端にも存在する。

この値はannotation済みnodeの位置構成である。位置ごとの真の全細胞数がないため、「中央の細胞ほどannotationされやすい」とは結論できない。

### tracking graphの連結成分

| 指標 | min | 25%点 | median | 75%点 | 95%点 | max | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sample当たり連結成分数 | 1 | 6 | 19 | 33 | 59.1 | 79 | 22.29 |
| 連結成分当たりnode数 | 2 | 12 | 21 | 39 | 93 | 187 | 30.06 |

分裂元として2本の出edgeを持つnodeは全体で151個だった。annotationは、frame内の全細胞を列挙したものというより、複数の疎なtrajectoryまたはlineageの断片として構成されている。ただし、公式の選択手順自体は公開GEFFから確定できない。

## 解釈

- 支持されたこと: 公式annotationはsample全体で疎であり、密度はsample間・胚間で大きく異なる。未アノテーション位置を通常のnegativeとして扱う学習は公式ラベルの意味と整合しない。
- 否定されたこと: 「annotationが全100 frameに存在するsampleなら、その時間範囲は完全annotationされている」という解釈。多くのsampleで全frameにannotationがある一方、frame当たりnode数は少なく、完全性を示すmaskもない。
- 手動annotationへの含意: 公式annotationに近い学習データを追加する目的なら、既存trajectoryの修正・延長や少数trajectoryの追加を優先する。完全annotation領域は、見逃し・false positiveの診断や負例学習に有用だが、公式の疎なラベルとは別のannotation種別として管理する。
- 未解決: どの画像領域・細胞・発生段階が選択的にannotationされたか、局所的な完全annotation領域が実際に存在するか、推定全node数の誤差幅、画像上の真の細胞密度で補正したannotation確率。

## 関連ファイル

- 実験の`result.md` / `metrics.json`: なし。
- 調査コード: [`annotation_distribution_audit.py`](../../studies/biohub_annotation_distribution/annotation_distribution_audit.py)。Kaggle competition inputを直接読み、ローカルにcompetition raw dataを置かずに実行する。
- 機械可読な集計要約: [`annotation_distribution_summary.json`](../../studies/biohub_annotation_distribution/annotation_distribution_summary.json)。
- Kaggle Notebook: [`biohub-annotation-distribution-audit`](https://www.kaggle.com/code/kentookumura/biohub-annotation-distribution-audit)。

## 次のアクション

1. 手動annotationを行う場合は、公式の疎なtrajectory追加用と、完全annotation領域による診断・負例学習用を分け、annotationの由来と有効領域maskを保存する。
2. 完全annotation領域を学習へ投入する前に、公式ラベルだけのbaseline、疎な手動追加、完全annotation領域を別batch・別maskで加えた条件をKaggle上で比較する。
3. 位置別の未アノテーション分布が必要なら、完全annotationした少数領域または画像から独立に生成した細胞候補を分母にし、今回のannotation node位置histogramとは別の調査として記録する。
