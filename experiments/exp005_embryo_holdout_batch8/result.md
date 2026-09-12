# exp005_embryo_holdout_batch8 結果

## 仮説

exp004のOOMが学習batch size 16のGPU memory要求によるなら、batch size 8へ下げる最小差分で2foldの3 epochs学習と全199動画の胚holdout推論を完走できる。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

設定とSHAをこのファイルへ独立した記録として転記しない。CV・Public LB・Private LBは`metrics.json`を数値の正本とし、比較や乖離の説明に必要な場合は、参照キーを添えて同じ値を本文で使用してよい。

## 実行証拠

- 比較対象: `exp004_embryo_holdout_baseline`のbatch size 16でのKaggle OOM。保存済みcontrolを再学習せず、失敗記録を参照する。
- train Notebook version 1はT4 2基、internet無効で2foldのsmokeと各3 epochsを完走し、2 checkpointを保存した。Notebook計測時間は23,046.864秒である。
- inference Notebook version 1は同じ実行環境で全199動画を処理し、Notebook計測時間は16,625.959秒である。
- `metrics.json`の参照キー: `status=completed`、`cv`、`diagnostic_validation`、`evidence.kaggle`、`evidence.artifacts`、`evidence.reruns`。CV overall scoreは0.1249055162、胚別scoreは`44b6=0.6247928950`、`6bba=0.0260485532`である。Public LBとPrivate LBは取得していない。
- prediction manifest、動画別公式評価は各199件で、欠落、重複、skippedは0。候補cache 199件とprediction graph 199件のKaggle上のpath集合はmanifestと一致した。
- 保存graphから公式評価を再計算し、動画別・胚別・全体の集計が初回評価と一致した。model、split、source、candidate、predictionのSHAは`metrics.json.evidence.artifacts`を正とする。
- 検証用のJSON、kernel log、metadataはignore済みの`artifacts/inference_v1/`へ保存した。大容量の候補とgraph本体はKaggleのinference version 1を正とする。
- competition submission、hidden test submission、Public LB取得は行っていない。
- strict experiment validation、実験固有13テスト、strategy文書検査、strict experiment reviewer、差分検査は成功した。共通テストは304件中302件が成功し、既存repository状態と矛盾するtemplate用前提の2件だけが失敗した。

## 解釈

batch sizeを16から8へ下げたことで、exp004のOOMを回避し、2方向の学習と全199動画の胚holdout予測を契約どおり完走できた。低い得点を実行失敗条件にしない契約であり、欠落0と再採点一致を満たすため、後続の誤差分析と段階別上限の基準予測として使用できる。

一方で、一般化性能は胚方向によって大きく異なる。`6bba`で学習して`44b6`を評価したscoreは0.6248、node recallは0.8565だったが、`44b6`で学習して`6bba`を評価したscoreは0.0260、node recallは0.0348だった。overall score 0.1249は199動画をまとめた公式集計であり、2胚のscoreの単純平均ではない。division true positiveは0で、division Jaccardも0である。2胚だけの評価なので未知胚全体への性能を断定せず、次は動画別・特徴別の誤差と、保存候補から計算する段階別上限で、`6bba`方向の主因が検出候補、接続score、最終graph選択のどこにあるかを切り分ける。

## 追加診断

Kaggleのprivate CPU Notebook `kentookumura/exp005-embryo-holdout-batch8-diagnostic` version 1で、inference version 1の固定候補と同じ199件のGTを対応付けた。Notebook計測時間は804.711秒で、prediction manifestと動画別metricsの入力SHAは既存記録と一致した。動画別CSV、summary、plot、manifest、kernel log、metadataはignore済みの`artifacts/diagnostic_v1/`へ保存し、SHAは`metrics.json.evidence.artifacts`を正とする。

| 評価胚 | 0.99候補のnode recall平均 | 最終graphのnode recall平均 | 候補後の低下 | 候補nodeの最終graph残存率平均 | 0.5通過edge / 候補node |
| --- | ---: | ---: | ---: | ---: | ---: |
| `44b6` | 0.9665 | 0.8565 | 0.1100 | 0.7613 | 0.6155 |
| `6bba` | 0.9458 | 0.0348 | 0.9110 | 0.0118 | 0.00514 |

`6bba`でも検出確率0.99を超えた候補は既知GT nodeの94.58%を回収していた。したがって、最終graph node recall 3.48%を「検出候補がほぼ作れていない」と解釈するのは誤りである。主な崩壊は候補生成後にある。

固定prediction manifestでは、`6bba`の接続確率0.5通過edgeは12,689本で、候補node当たり0.00514本だった。`44b6`の1,873,469本、候補node当たり0.6155本に対して約120分の1である。両胚とも閾値通過edge数とgraph入力edge数は同数なので、graph入力前の追加枝刈りは原因ではない。整数線形計画後には`6bba`で10,196本、`44b6`で1,813,797本が残るため、整数線形計画にも追加低下はあるが、支配的な失敗は接続scoreが0.5を超えない段階である。

画像の固定samplingでは、`6bba`のraw 0.001–0.999 quantile幅の中央値は1,415、`44b6`は2,579で、`6bba`が狭かった。動画ごとのquantile正規化後も、sample平均の中央値は0.0986対0.1903、標準偏差は0.1269対0.1602で差が残った。画像の見え方の胚差は存在するが、候補node recall自体は両胚で高いため、今回の証拠は初期局在より接続特徴または接続scoreの胚間一般化を優先して調べる方向を支持する。2胚とsparse samplingの記述的分析であり、画像統計との因果関係や最適な補正方法は断定しない。

## ユーザー判断

- 判断: `completed`
- 確認日時 / 依頼メッセージ: 2026-09-12 / ユーザー「ひとまずexp005は閉じてgit commitとpushしてください。exp006と007の結果が出た後に方針を更新することとします。」
- 理由: 2方向の学習、全199動画の胚holdout推論、公式評価、保存graphの再採点、追加診断まで完了し、実験契約の受け入れ条件を満たしたため、exp005を完了として閉じる。これは低いCVを採用する判断や、今後のvalidation方針を確定する判断ではない。

## 次

exp006とexp007の結果が揃った後に、動画単位のgroup validationと胚holdoutの役割分担を含む実験方針を更新する。それまではexp005の結果を、2胚間の非対称な一般化と接続score崩壊を示す固定証拠として保持し、この結果だけでvalidation方針を変更しない。
