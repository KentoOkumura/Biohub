# exp046_x138_author_head_comparison セッションノート

## 現在の作業

全20動画の比較と結果回収は終了した。[結果とユーザー判断](result.md)に従い、作者headの採否と実験完了の判断を待つ。

## 2026-09-25 実装時点の目的と計画

2026-09-25の実装依頼により、設計済み候補をexp046へ移行した。exp043の自前headと取得済み作者headを、exp045と同じ20動画・固定推論設定で比較する。competition submissionの承認はない。

予定は2条件、固定head各1個、設定1組、新規学習0、fold 0、booster 0。control再学習なし。exp045の自前head対照は全件Kaggle実行中で、条件と生成物が一致すれば再利用可能性を確認する。未取得なら本比較内で再実行する。

## 実装と検証

- make new-expでexp045の診断Notebookを構成参照としてコピーし、正本のlineage.parentはexp043へ設定した。
- 両headのSHAとstate_dict・mean・scaleを同じNotebookで検証する。各条件は別プロセスで実行し、head cacheを共有しない。
- 元検出候補で教師対応を固定し、各headの最終graphは公式評価器で独立採点する。
- make validate-exp、make check-exp、make test-exp（3件）、Jupytext変換検証とmake check-strategy-docsは成功。両NotebookのKaggle packageもprepare済み。
- 2026-09-25 10:54 UTCのKaggle quotaはGPU使用46.49/45.00時間、残り0.00時間、refresh 2026-09-26 00:00 UTC。exp045全件Notebookはlive logs上で推論中。予備確認の見込みはexp045 pilotの2動画・2条件で約0.38時間だが作者headの実測はない。週45時間の範囲で新たなGPU起動はできないためexp046はpushしていない。
- 生成metadataはpilot/inferenceともprivate、T4、GPU true、TPU false、internet false、固定Dataset 5件。作者headのローカルSHAは期待値と一致した。
- Kaggleの予備確認、全件推論、公式指標、runtimeと生成物SHAは未取得。

## 次のアクション

[全件比較の結果](result.md)を基にユーザーの採否・完了判断を待つ。competition submissionは行っていない。

## 2026-09-25 初回比較前の予定（履歴）

1. quota refresh後に残時間とexp045全件出力を確認し、対照再利用の条件一致と必要GPU時間を判断する。
2. Kaggleで両胚各1動画の予備確認を行い、候補ID、head補正量、公式評価、費用を確認する。
3. 条件を満たせば全20動画の比較を行う。competition submissionは行わない。

## 2026-09-26 実行開始

- ユーザーが「実行してください」と指示。exp045の自前head対照は20動画のGPU推論・Kaggle CPU公式評価とローカル固定ID診断を完了済み。保存済み対照の条件と再実行差をexp046 pilotで確認し、全件では作者headだけの新規推論への切替可否を判断する。
- exp045全件version 1で判明したGT GEFFのtuple返り値を、exp046の全件診断で既存graph_from_geff helper経由に修正した。validate-exp、check-exp、test-exp（3件）、Notebook sourceのF821に成功。
- 2026-09-26 00:41 UTCのGPU quotaは残り45.00/45.00時間、refreshは2026-10-03 00:00 UTC。exp045 pilotの2動画・2条件は約0.38時間。exp046 pilotは作者headの実測がないためこの値を上限保証とはせず、予備確認として実行する。
- pilot package metadataはprivate、NvidiaTeslaT4、GPU true、TPU false、internet false、competition trainと固定Dataset 5件、run_on_push true。新規学習0、head 2条件、設定1、fold 0、booster 0、control再学習なし。competition submissionは行わない。
- `make push-kaggle-notebook EXP=exp046_x138_author_head_comparison NOTEBOOK=pilot` でprivate Kaggle Notebook version 1（kernel ID 135901589、[Kaggleの予備実行](https://www.kaggle.com/code/kentookumura/exp046-x138-author-head-comparison-pilot)）を起動。metadataをpullしてT4、GPU true、internet false、5 Datasetを再確認した。00:54 UTC時点で実行中。
- Kaggle pilot version 1成功。receipt SHA256 `48b3d367a6b1d0caeb456cdd4e29a812d758aef93470271d5d429ba81e3f78db`、Notebook内所要時間1263.9秒。自前headはexp045 pilotと選択20動画・評価器SHA・head SHAが一致し、両胚各1動画の候補数・元候補SHA・初期graph node数・最大移動量、最終graph 42ファイルのSHA、公式評価の全行と集計が完全一致した。対照の再実行差は両動画で0。作者headの予備2動画score 0.908381、自前0.916229。
- 2026-09-26 01:14 UTCのauthor_only push前gate: metadataはprivate、NvidiaTeslaT4、GPU true、TPU false、internet false。GPU残44.48/45.00時間、refresh 2026-10-03 00:00 UTC。2条件のpilot実測1263.9秒から作者1条件20動画は約2.2時間を25%余裕込みで見込み、12時間上限と残枠内。保存済みexp045対照を使い作者headだけ全件実行する。
- `make push-kaggle-notebook EXP=exp046_x138_author_head_comparison NOTEBOOK=author_only` で20動画の作者head Notebook version 1（kernel ID 135904131、[Kaggle全件実行](https://www.kaggle.com/code/kentookumura/exp046-x138-author-head-comparison-author-only)）を起動。pullしたKaggle側metadataでprivate、NvidiaTeslaT4、GPU true、TPU false、internet false、固定Dataset 5件を確認した。

## 2026-09-26 全件実行と比較結果

- Kaggle author_only version 1（kernel ID 135904131）は`KernelWorkerStatus.COMPLETE`。Notebook内所要時間は6794.812秒（約1時間53分）、先行2動画は675.805秒、残り18動画の25%余裕付き見積りは7602.811秒で、43200秒のNotebook上限を満たした。追加18動画の予測部分は4507.5秒。予備確認の1263.889秒と合わせたNotebook内実行時間は8058.701秒。Kaggle GPU quotaはpush前44.48/45.00時間、2026-09-26 03:16 UTCの実行後40.71/45.00時間（同一週の他利用も含むスナップショット）。
- 20動画すべての元候補cache、ILP graph、最終graph、固定ID診断、公式評価を生成。Notebookの出力manifestは3305ファイル・1,365,342,745 bytes。必要なreceipt、manifest、公式評価、固定ID診断、runtime gateだけを`kaggle kernels output --file-pattern`で取得し、取得した評価・診断・gateの4ファイルのSHAとbyte数をmanifestに照合した。全3305ファイルはローカルには取得していない。
- 取得した[全件receipt](artifacts/author_only/exp046_author_only/exp046_author_only_receipt.json)、[公式評価](artifacts/author_only/exp046_author_only/official_metric.json)、[固定ID診断](artifacts/author_only/exp046_author_only/fixed_id_by_embryo.json)を保存済みexp045対照と結合。`PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/exp046_x138_author_head_comparison/compare_saved_results.py`で20動画の選択・元候補SHA・候補数・固定教師対応、両head・評価器SHA、公式行と集計、出力manifestを照合し、[比較JSON](artifacts/comparison.json)を生成した。全検証に成功。
- 自前headのexp045 pilotとexp046 pilotは2動画の最終graph 42ファイルと公式評価全行が完全一致。作者headのexp046 pilotとauthor_only全件実行でも同じ2動画の最終graph 42ファイルと公式評価全行が完全一致。観測した対照・作者headの再実行差は0。
- 公式combined scoreは全体で自前0.890348、作者0.886189、差は作者−自前で−0.004160。44b6は0.935902→0.933817（−0.002084）、6bbaは0.873577→0.868707（−0.004870）。作者headが高い動画は44b6で4/10、6bbaで5/10。division TP/FP/FNは全体で両条件とも1/8/16。Public LBは未測定。
- 固定IDの最終段階で選択された既知edgeは44b6で2759→2783、6bbaで7333→7385。一方、公式評価のadjusted edge Jaccardは両胚で低下。6bbaの公式edge FPは523→553、TPは7762→7770。44b6はedge TPが2899→2908、FPが174→175、FNが142→133で、micro edge Jaccardは改善したがcombined scoreを決める加重adjusted edge Jaccardは低下した。中心距離（固定ID、matched GT node数で重み付け）は44b6で1.556→1.464 µm、6bbaで1.740→2.028 µm。固定IDの既知edge回収だけを公式scoreの代理とはしない。
- 本実験は新規学習・追加探索・competition submissionなし。作者headの学習動画は不明で、公開画像モデルはcompetition train画像を学習済み。数値は条件付きtrain subsetの20動画比較であり、独立CV・hidden testの優劣ではない。採否・実験完了はユーザー判断待ち。
- 結果記録後に`make validate-exp EXP=exp046_x138_author_head_comparison`、`make check-exp EXP=exp046_x138_author_head_comparison`、`make test-exp EXP=exp046_x138_author_head_comparison`（3件）、`make check-strategy-docs`、実験ドキュメントreviewerの`--strict`が成功。`git diff --check`も成功。戦略索引のHYP-20260910-02はexp046の条件付き結果と採否未判断へ更新したが、上位仮説全体の支持・棄却は決めていない。
