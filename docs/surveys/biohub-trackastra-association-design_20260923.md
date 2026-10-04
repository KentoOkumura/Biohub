---
title: Trackastraの部分注釈への適用と既存追跡処理の互換性
date: '2026-09-23'
types:
- survey
hypotheses:
- HYP-20260910-10
experiments:
- exp016
- exp040
topics:
- tracking
- architecture
- validation
status: final
summary: 6時点の対応学習には部分教師と確率入力adapterが必要。既存softmaxへの接続は対応なし確率を消す。ユーザーが公式実装標準規模・primary単体の設計を確定し、実験契約へ移行。
---

# Trackastraの部分注釈への適用と既存追跡処理の互換性

## 結論

固定検出点と固定画像特徴をTrackastraの対応モデルへ接続する設計は具体化できる。ただし、疎い注釈の未知を負例にせず、親候補を正規化した確率をそのまま復号へ渡す変更が必要である。既存のexp016推論へlogitの形だけ合わせると、Trackastraが残す「親を割り当てない確率」が失われる。

`HYP-20260910-10`のうち短い時間窓での対応学習を調べた。モデルの有効性や上位仮説の支持・棄却は本調査では判断していない。設計の正本は[exp040の実験契約](../../experiments/exp040_trackastra_association/requirements.md)。このレポートの`final`は実装照合の完了を表す。2026-09-23にユーザーが推奨する設計を確定し、その後に実験化・実装を承認した。調査時点では学習結果は未測定だった。後日の学習・診断と採否判断は[exp040の結果](../../experiments/exp040_trackastra_association/result.md)を参照する。

## 対象と証拠範囲
- 対応する上位仮説: `HYP-20260910-10`

2026-09-23に論文、公式GitHub source、ローカルのexp016/027実装を確認した。公式sourceの参照revisionは`aa57a95160002e0fc70b915ab74178b39c99fd6a`。外部コードは実行していない。exp015 cacheとexp016保存重みは今回取得せず、有効教師件数・時間・メモリ・精度は未測定。

## 一次資料と確認事項

| 資料 | 確認した内容と設計への意味 |
| --- | --- |
| [Trackastra論文、2024、v2](https://arxiv.org/html/2405.15700v2) | 短窓の点集合にencoder-decoderを適用し、祖先・子孫の対応を学習する。parental softmaxは親側だけを競合させ、固定の対応なし項を含む。正規化後BCEと0.01倍の補助BCE、1・2時点先の教師、重複窓の確率平均を用いる。記載された実験構成は6時点・幅256・各6層。 |
| [model.py、固定revision](https://github.com/weigertlab/trackastra/blob/aa57a95160002e0fc70b915ab74178b39c99fd6a/trackastra/model/model.py) | `TrackingTransformer`は点座標と任意次元の特徴を受け取れる。API標準値は幅128・4 heads・各4層・window 6・dropout 0.1。論文の実験規模とAPI標準値を区別する。新しい32次元特徴を使うため、配布checkpointの直接適用とはしない。 |
| [utils.py、固定revision](https://github.com/weigertlab/trackastra/blob/aa57a95160002e0fc70b915ab74178b39c99fd6a/trackastra/utils/utils.py) | `blockwise_causal_norm`は時点block単位で親方向に正規化する。`quiet_softmax`の固定項は単なる数値安定化定数ではなく、対応を割り当てない余地を持たせる。 |
| [scripts/train.py、固定revision](https://github.com/weigertlab/trackastra/blob/aa57a95160002e0fc70b915ab74178b39c99fd6a/scripts/train.py) | BCEWithLogitsと正規化後BCEを組み合わせ、時間差1・2へmaskを掛ける。標準実装は密な教師を前提とし、対応行列から正例を重み付けし、大きい窓にも追加の重みを付ける。本適用では部分注釈mask、GEFFで確認できる分裂、窓の等重み平均へ変更する。 |
| [LICENSE](https://github.com/weigertlab/trackastra/blob/aa57a95160002e0fc70b915ab74178b39c99fd6a/LICENSE) | BSD-3-Clause。実験へsourceを同梱するときに著作権表示・ライセンスを保持する。 |

TrackastraのCTC/TRA等の論文実験での性能は、Biohubの公式combined scoreの証拠ではない。公開の領域画像入力APIと、内部の点・特徴入力moduleも区別する。

## 既存コードから判明した制約

- [exp027の複数時点入力](../../experiments/exp027_multi_frame_tracker/frozen_tracker.py)は、重複するframeの候補IDと座標の完全一致を検査している。一方、画像特徴は元の2時点窓に依存するので一致を要求しない。従って6時点入力には、各tokenの特徴元pairとsideを明記する必要がある。
- [exp016の教師と損失](../../experiments/exp016_frozen_image_encoder/frozen_tracker.py)は5 µmのgreedy one-to-one対応と、正例の行または列を含むlegacy maskを使う。このmaskの負例には未注釈候補が含まれ得る。Trackastraの新教師にはそのまま引き継がず、過去比較の診断指標として保存する。
- [exp016のgraph推論](../../experiments/exp016_frozen_image_encoder/graph_inference.py)は、正逆方向とsecondary trackerのlogitを融合した後、`select_cached_candidate_edges`で親方向にsoftmaxする。parental softmax確率のlogをここへ渡すと、接続先の確率和が再び1になり、対応なしの余地を失う。
- 従って推奨する最初の比較はTrackastra primary単体の確率を直接扱う。補助trackerの寄与が消える差は、同じ保存済みexp016のprimary単体を共通診断・後段graph対照に置くことで観測する。保存済み通常構成との比較も残し、単一部品だけの変更とは説明しない。

## 部分注釈への適用判断

正例はGEFFの既知edge、または2本の連続edgeで保証できる祖先対応から作る。負例は、その子の真の祖先とその検出が存在し、別の注釈IDに対応した親候補に限定する。別の注釈経路が見つからないだけでは負例にしない。

未注釈や真の親の検出欠落を対応なし教師にしない。未知候補もmodelの文脈と正規化の分母には残るので、直接のBCEをmaskしても競合を通じた勾配は残る。また、対応なしの明示教師がないため、本候補だけで新規出現・消失の確率が正しく校正されるとは主張できない。

これは原論文の教師のそのままの再現ではなく、Biohubの部分注釈へ適用する設計である。方法、重み、停止条件の具体値はexp040の`requirements.md`と`config.yaml`を正とし、ここへ重複させない。

## 残る実測と次のアクション

- ユーザーは「推奨案で設計を確定」と回答。公式実装標準の幅128・各4層、6時点の部分教師、10 epochs×2fold、secondary融合なしを確定し、その後の実装依頼を受け、候補詳細をexp040の実験契約へ移した。
- 実験化後に教師の有効件数、窓のidentity、通常・密な窓の費用、学習成立を確認する。設計だけで実行可能性を実証したとは扱わない。
- 学習後は両胚の共通隣接ペアで接続・分裂・教師負例予測を比較し、進行条件を満たした場合に全graph評価を検討する。
