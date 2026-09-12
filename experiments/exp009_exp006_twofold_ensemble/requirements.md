# exp009_exp006_twofold_ensemble 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録する。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: 直接承認。2026-09-12にユーザーが、exp006の2foldモデルを使うアンサンブルを先に実施し、その後に再現性を担保する実験へ進むよう依頼した。
- 移行元backlog / 対応する上位仮説: `N/A`。形式的なbacklogは作らない。
- 親実験: [`exp006_embryo_holdout_seed314159`](../exp006_embryo_holdout_seed314159/)。同実験がKaggleで保存した2foldのcheckpointだけをモデル入力にする。
- 根拠: exp006の2モデルは異なる胚だけで学習され、単一foldだけでは一方の胚由来の学習情報を捨てる。最終graph同士はnode座標が異なり単純平均できないため、共通node候補を作る前後の確率を融合する。
- 固定するもの: exp006のsource commit、モデル構造、2 checkpoint、画像正規化、2時刻窓、TTA、検出・edge閾値、最大子・親数、ILP重み、hidden testの動的列挙、CSV schema。
- 変更するもの: fold routingを廃止し、各test動画を両モデルで推論する。検出確率を等重み平均して1つのnode集合を抽出し、その共通node集合に対する両モデルのedge確率を等重み平均して、閾値とILPを各1回だけ適用する。
- 最小の反証可能な検証: 公開testをKaggle Notebook上で全件推論し、両checkpointのSHA、全test graph、`submission.csv`、行数・重複・欠損・有限値・edge参照整合性を確認する。その後、承認済みのcode submissionを1件作成する。
- 成功条件: 2 checkpointを検証して使用し、実行時testとsample submissionから動的に有効な`submission.csv`を生成し、repository submit-checkがPASSし、Kaggleがsubmission refを返すこと。
- 停止条件: checkpoint/SHA/source不一致、2 GPU不足、test欠落、sample schema不一致、edge参照不整合、12時間超過見込み、submit-check FAIL、またはKaggle側の実行失敗。
- 実行しないこと: 再学習、checkpoint選択、重み探索、閾値調整、完成graphのunion/intersection、公開test固有ID・件数・SHAによる分岐、train GEFF参照。
- 未決事項: なし。

## 判断履歴

- 2026-09-12: 単一fold提出案に対し、ユーザーが通常の複数fold ensembleを使わない理由を確認した。
- 2026-09-12: 最終graphではなく検出・edge確率を融合する2fold推論を提示した。
- 2026-09-12: ユーザーがこの2fold ensembleを先に実施し、その後に再現性実験を行うよう承認した。以前の「exp006は提出します」と合わせて、実行・提出・監視までを承認済みと扱う。
- 2026-09-12: 安全審査でexp009としての明示承認を再確認し、ユーザーの「提出していいです」を受けて検証済みversion 1を提出した。

## 手法契約

- 依頼原文: 「まずはこれを実施してください。次に再現性を担保する実験を行ってください。」
- input: hidden testのZarr画像、exp006 train kernelのmodel manifest、fold 0/1 checkpointとmodel config。
- target / objective: 学習は行わない。保存済み2モデルの細胞中心検出確率と隣接時刻間接続確率を融合する。
- output: 各test動画のGEFF graph、variable-row graph形式の`submission.csv`、inference summary、更新済みmetrics。
- loss: `N/A`。保存済みモデルを再利用する。
- decode: TTA後の各モデルのsigmoid検出確率を算術平均し、その平均volumeでlocal maximumと0.99閾値を適用する。共通node候補上の各モデルのsoftmax edge確率を算術平均し、0.5閾値、既存の最大子・親数制約、ILPを1回適用する。
- context unit: モデル入力は2時刻の3D画像窓、融合は同一窓・同一node候補・同一隣接時刻pair、出力は1動画単位。
- 実装区分: `faithful`。承認された確率平均と1回のdecodeをそのまま実装し、省略する機構はない。
- この実験が支持 / 棄却できる主張: 2fold確率ensembleがhidden testで実行可能か、単一fold提出より妥当な提出候補になるかをLBで確認できる。
- この実験では判断できない主張: ensembleがCVを改善するか、exp006再学習のbitwise再現性、seed分布全体。

## 実装方法

- Jupytext percent形式のself-contained inference sourceから通常のNotebookを生成する。train Notebookは実装・実行しない。
- model manifestによりscratch学習、source commit、seed 314159、2fold、各checkpoint/model-config/fold-manifestのSHAを検証する。
- test Zarrとsample submissionを実行環境から動的に解決し、名前順で動画・windowを処理する。
- fold 0を`cuda:0`、fold 1を`cuda:1`へ固定し、各windowを両GPUへ送り、検出確率と共通候補のedge確率だけをCPUで平均する。
- graphは平均edge確率から1回だけ構築・ILP solveし、GEFFへ保存する。全graphをCSVへ変換し、schemaと参照整合性をNotebook内とrepository validatorで検査する。
- 実験固有testで親モデル設定、2fold平均、decode回数、dynamic test discovery、sample schema検査、Kaggle metadataを固定する。

## 探索幅とpivot判定

- 変更class: `postprocess`。完成graphではなくモデル確率を融合する推論方針の変更。
- active variant 1、inference config 1、再学習fold 0、booster 0、再利用model 2。親controlの再学習はしない。
- `kaggle-idea-forge`は追加実行しない。これは新しい微調整探索ではなく、ユーザーが明示した2fold提出方針を1構成で実装する作業である。
- 追加GPUコスト: T4 x2で推論1回。2モデルを別GPUへ固定し、12時間以内を停止条件とする。

## 再現性・リスク

- 推論は乱数を使わず、保存済みモデルSHA、sorted test/window順、fold-to-device対応を固定する。ただし今回1回の実行だけではdeterministic anchorと呼ばない。
- model manifest SHA、各checkpoint SHA、test dataset数、submission SHA、kernel id/version、resource、Notebook実行時間を`metrics.json`へ記録する。
- hidden testの入力分布は未知であり、2モデル同時保持のGPUメモリと12時間制約が主要リスクである。動画単位で生成物を保存してメモリを解放する。
- 公開test固有値に依存せず、実行時sample submissionは列schemaの正として使う。variable-row形式のため公開sampleの行数やID集合へ予測行を合わせない。

## 受け入れ基準

- [x] 直接承認、親実験、融合位置、固定事項、成功条件、停止条件、実行しないことを実装前に記録した。
- [x] inference source/Notebookと実験固有testを実装する。
- [x] `validate-exp`、`check-exp`、`test-exp`を通す。
- [x] Kaggle T4 x2で公開test推論を完了し、outputを取得する。
- [x] repository submit-checkをPASSする。
- [x] code submissionを作成し、submission ref `56182729`を固定して監視を開始する。
- [ ] 実験の完了、採用、不採用は結果提示後にユーザーが判断する。
