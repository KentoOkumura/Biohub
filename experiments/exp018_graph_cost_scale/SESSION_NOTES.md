# exp018_graph_cost_scale セッションノート

## 目的

exp015の固定ILP前候補graphでedge cost scalarだけを有限比較し、胚を入れ替える2方向の選択と評価で、固定repairを含む後段へ進む価値があるか判定する。

## 現在の作業

- 作業内容: Kaggle CPU実行、公式metricによる両方向gate判定、実行証拠の回収、ユーザー判断まで完了した。
- ブロック要因: なし。
- 次: この実験内の追加実行は行わない。固定repair段階とKaggle submissionへは進まない。

## コマンドログ

### 2026-09-15

- `task new-exp EXP=exp018_graph_cost_scale`: `task` commandが環境にないため実行不可。
- `make new-exp EXP=exp018_graph_cost_scale`: templateから実験ディレクトリを作成。
- `make validate-exp EXP=exp018_graph_cost_scale`: strict validation通過。
- `make check-exp EXP=exp018_graph_cost_scale`: Ruff lintとformat check通過。
- `make test-exp EXP=exp018_graph_cost_scale`: 5 tests通過。
- `uv run --extra notebook jupytext --test --to ipynb experiments/exp018_graph_cost_scale/exp018_graph_cost_scale_diagnostic.py`: round-trip検査通過。
- `.venv/bin/ruff check --select F821 experiments/exp018_graph_cost_scale/exp018_graph_cost_scale_diagnostic.py`: 未定義名検査通過。
- `make check-strategy-docs`: backlog移行後の戦略文書検査通過。
- `make validate-surveys`: survey索引検査通過。
- `.venv/bin/python scripts/check_markdown_links.py`: 今回削除した`graph_cost_scale.md`への参照は0件。既存の`oracle_stage_limits.md`移行漏れなど11件で全体検査は失敗した。
- 2026-09-15: ユーザー依頼「移行漏れを対処してください」により、`oracle_stage_limits`の参照を`exp015_oracle_stage_limits`へ移し、exp001の4件の誤った相対パスも修正した。再実行した全205 Markdownファイルのリンク検査は通過した。
- `make prepare-kaggle-notebooks EXP=exp018_graph_cost_scale EXTRA_ARGS="--notebook diagnostic"`: strict package準備通過。生成metadataでCPU、internet無効、competition source、support dataset、2 kernel sourceを確認した。pushと実行はしていない。
- 2026-09-15T20:26:30+09:00: ユーザーからKaggle実行の明示依頼を受領した。submissionは依頼範囲外のため実行しない。
- `check_all_credentials.py --require cli`: Kaggle CLIで使用できるOAuth credentialsとlegacy API keyを確認した。
- push直前に`make validate-exp`、`make check-exp`、`make test-exp`を再実行し、strict validation、Ruff、5 testsが通過した。
- `jupytext --to ipynb --test`とRuff `F821`検査を再実行し、Notebook同期と未定義名検査が通過した。
- `make prepare-kaggle-notebooks EXP=exp018_graph_cost_scale EXTRA_ARGS="--notebook diagnostic --run-on-push"`: strict package準備通過。
- 2026-09-15T20:26:30+09:00 resource guard: `enable_gpu=false`、`enable_tpu=false`、`enable_internet=false`、`run_on_push=true`、competition source 1件、support dataset 1件、kernel source 2件を確認した。CPU実行なのでGPU quota確認は対象外。canonical slugは`kentookumura/exp018-graph-cost-scale-diagnostic`。push可と判断した。
- `make push-kaggle-notebook EXP=exp018_graph_cost_scale NOTEBOOK=diagnostic`: version 1を同一slugへpushし、Kaggle上の実行を開始した。pullしたmetadataの`id_no=134483142`、CPU、internet無効、入力sourceを再確認した。
- version 1は公式metric source探索中に`FileNotFoundError`で停止した。`evaluator_path=repo/scripts/evaluate.py`に対して、fallbackで見つけた`.../tracking_repo/repo/scripts/evaluate.py`のrootを`.../tracking_repo/repo`と誤って導出し、`repo/repo/scripts/evaluate.py`を開こうとしたことが原因。
- 2026-09-15T20:39:52+09:00: evaluator相対パス全体を除いてrootを導出するよう修正し、support datasetの明示rootから余分な`repo`を除いた。SHA照合前の存在確認と再発防止testを追加した。実験条件、alpha grid、公式metric、選択規則、gateは変更していない。
- 修正後にNotebookを再生成し、strict validation、Ruff、6 tests、Jupytext round-trip、Ruff `F821`が通過した。version 2を同一slugへ再pushする。
- version 2はevaluator fileのSHA照合まで通過したが、公式metric module import時に`ModuleNotFoundError: biohub_tracking`で停止した。SHA固定されたmetrics fileが`<dataset-root>/repo/src/biohub_tracking/metrics.py`にある一方、`<dataset-root>/src`を`sys.path`へ追加していたことが原因。
- 2026-09-15T20:42:20+09:00: SHA照合済み`metrics_path`の親からPython source rootを導出するよう修正し、再発防止testを追加した。Notebookを再生成し、strict validation、Ruff、7 tests、Jupytext round-trip、Ruff `F821`が通過した。version 3を同一slugへ再pushする。
- version 3は全199 sample、995 alpha評価のILP solveまで実行し、`elapsed_seconds=22521.33920039`だった。ただし`ILPSolver.solve()`の返り値`GraphView`を公式metricへ直接渡したため、metric内部の`copy()`で全995行が失敗した。promotion gateは有効なscoreがないため判定不能であり、`false`を実験結果として採用しない。
- version 3の失敗証拠を`artifacts/kaggle_v3_failed/`へ回収した。candidate graph bundle SHAは`3df8d9fe071e10cfca159fffbe1636ae9b1131c538021b1c3621cf1ab4d155e9`、失敗manifest SHAは`85684584b034e20b3917f8e3b972fcdb3e7220178821ca5c68be231d0a4c19c6`。
- 2026-09-16T06:44:30+09:00: solver出力へ`detach()`を適用してreference-less graphにしてから公式metricへ渡すよう修正した。`GraphView.detach()`の適用を検証する回帰testを追加し、Notebookを再生成した。strict validation、Ruff、8 tests、Jupytext round-trip、Ruff `F821`が通過した。比較条件、alpha grid、ILP cost、公式metric、gateは変更していない。
- version 4は最初のsampleで公式metric smokeに成功し、その後のsampleでも有効なmetricを生成した。ただし20/199時点で約3時間を要し、単一NotebookではKaggleの12時間上限内に完走できないと判断した。この実行は性能診断として扱い、途中scoreやgate判定を実験結果に使わない。
- 2026-09-16T10:00:45+09:00: alphaごとに199 sampleを固定順序で処理する5つのCPU shard Notebookと、全shardのmanifest・SHA・coverageを検査して公式`summary`と両方向gateを計算するaggregate Notebookへ分割した。alpha grid、sample集合、solver、公式metric、選択規則、promotion gateは変更していない。alpha間のprocess実行順序だけが変わるため、solution topology SHAを各shardとaggregateに保存する。
- 分割後の`make validate-exp`、`make check-exp`、`make test-exp`は通過し、9 testsが成功した。
- 7つのJupytext sourceからNotebookを再生成し、全Notebookのround-trip検査とRuff `F821`検査が通過した。
- 5つのalpha shard packageを`--run-on-push`で準備し、すべてCPU、GPU/TPU無効、internet無効、competition source 1件、support dataset 1件、exp015・exp012 kernel source 2件であることを確認した。
- `alpha_025`、`alpha_05`、`alpha_1`、`alpha_2`を各canonical slugへversion 1としてpushした。4件とも`RUNNING`で、offline dependency installと最初の公式metric smokeが成功した。
- `alpha_4`の初回pushは、旧diagnostic version 4を含むKaggleの同時CPU session数が5に達していたため開始されなかった。`alpha_4`のslugは未作成である。公式CLIのstatusとGetKernelはcurrent session IDを返さず、slugを指定した安全なcancel手段もないため、非公開APIや推測したIDで旧sessionを停止しない。4 shardのいずれかが終了して枠が空き次第、`alpha_4`を再pushする。
- 2026-09-16T11:04:20+09:00 resource guard: 旧diagnosticは`CANCEL_ACKNOWLEDGED`、`alpha_1`は`COMPLETE`となりCPU枠が空いた。`alpha_4` packageの`enable_gpu=false`、`enable_tpu=false`、`enable_internet=false`、`run_on_push=true`、competition source 1件、support dataset 1件、kernel source 2件を再確認した。CPU実行なのでGPU quota確認は対象外。canonical slugは`kentookumura/exp018-graph-cost-scale-alpha-4`で、push可と判断した。
- `alpha_4` version 1をpushしたが、exp015 kernel input探索が0件となり直ちに`ERROR`で停止した。Kaggle側metadataにはexp015とexp012のkernel sourceが正しく設定されていた一方、このrunではsupport datasetが`/kaggle/input/<slug>/`へmountされ、従来runの`/kaggle/input/datasets/...`と異なるlayoutだった。kernel source resolverも`/kaggle/input/notebooks`だけを探索していたため、Kaggle input全体をslugと固定ファイル名で探索するよう修正する。入力SHA検査は維持する。
- 2026-09-16T11:11:06+09:00: kernel source resolverとaggregateのshard resolverを`/kaggle/input`全体の探索へ修正し、direct mountと`notebooks/` mountの両方へ対応した。strict validation、Ruff、10 tests、diff checkが通過し、diagnostic Notebookを再生成した。version 2 packageはCPU、GPU/TPU無効、internet無効、run-on-push、入力source不変を再確認し、同じ`alpha_4` slugへ再push可能と判断した。
- `alpha_4` version 2を同じslugへpushした。入力sourceの解決とSHA検査を通過し、公式metric smoke evaluationが成功して199 sampleの処理を開始した。
- 完了した`alpha_1` version 1のoutputを`artifacts/kaggle_alpha_1_v1/`へ回収した。199 sample、failure 0、`elapsed_seconds=3331.682260155`、manifest SHAは`b588e48fe68b1716420e356ba6ca66f653bca9af5d4394c07cb2e90b8f6e9ca7`だった。
- CPU枠の解放検知が遅れたため、監視heartbeatを30分間隔から10分間隔へ変更した。各巡回で5 shardの状態を確認し、完了時は待たずにoutput回収と次段の準備を進める。
- 監視間隔変更後の状態確認で`alpha_05` version 1の完了を検知し、outputを`artifacts/kaggle_alpha_05_v1/`へ回収した。199 sample、failure 0、`elapsed_seconds=3863.5732610719997`、manifest SHAは`d4206c0bf134d08ac1f1d5e70f17dc53f79ad2f347abd101894b0c8407e357ab`だった。この時点で`alpha_025`、`alpha_2`、`alpha_4`は`RUNNING`、`alpha_05`と`alpha_1`は`COMPLETE`。
- 2026-09-16T11:37頃の監視で`alpha_025`と`alpha_2` version 1の完了を検知した。live logsで両方とも公式metric smoke成功、199/199到達、failure 0を確認し、outputをそれぞれ`artifacts/kaggle_alpha_025_v1/`と`artifacts/kaggle_alpha_2_v1/`へ回収した。
- `alpha_025`: 199 sample、failure 0、`elapsed_seconds=4906.455502731`、manifest SHAは`cf946dd38d09df5cb2602069ffe5e709508308cb9801c78e1b3a0cd915d6be98`、solution bundle SHAは`07acfb63f702e9b55b5616ad99dd16d6a2711c1da81f0734367d1ff6dd51c575`。
- `alpha_2`: 199 sample、failure 0、`elapsed_seconds=4967.256729818`、manifest SHAは`12e5119e05fc5bbde5b35abbb7840c37b22eb3ac48db674df17dfd44d783d663`、solution bundle SHAは`45d88a86bb2684c902cd5c1598f107fb3345a78b90b316e95982f26a83c5f04a`。
- 同時点で`alpha_025`、`alpha_05`、`alpha_1`、`alpha_2`は`COMPLETE`、`alpha_4` version 2は`RUNNING`。`alpha_4`のlive logsは公式metric smoke成功後38/199まで進み、エラーは観測されていない。全5 shardが揃うまでaggregateはpushしない。
- 2026-09-16T12:41頃の監視で`alpha_4` version 2の`COMPLETE`を検知した。live logsで公式metric smoke成功、199/199到達、failure 0を確認し、outputを`artifacts/kaggle_alpha_4_v2/`へ回収した。Notebook内のartifact名は設定どおり`alpha_4_v1`である。
- `alpha_4`: 199 sample、failure 0、`elapsed_seconds=5311.848913471999`、manifest SHAは`65fc46ae5c33f6aaebb9e6ae4d90d13c459100f0c26db08746a9ad6bfb7a28a3`、solution bundle SHAは`ce407b7b5a737e06d83f290494837199b56b0100298c55d3d2bd863e255b54b0`。これで5 shardすべてが成功・回収済みとなったため、契約どおりaggregateへ進む。
- aggregate push前にstrict validation、Ruff、10 testsが通過した。`make prepare-kaggle-notebooks EXP=exp018_graph_cost_scale EXTRA_ARGS="--notebook aggregate --run-on-push"`でpackageを生成した。
- 2026-09-16T12:44:34+09:00 resource guard: aggregate metadataは`enable_gpu=false`、`enable_tpu=false`、`enable_internet=false`、`run_on_push=true`、competition source 1件、support dataset 1件、5つのalpha shard kernel sourceを確認した。CPU実行なのでGPU quota確認は対象外。canonical slugは`kentookumura/exp018-graph-cost-scale-aggregate`で、push可と判断した。
- aggregate version 1をcanonical slugへpushした。Kaggle側metadataをpullし、`id_no=134554758`、CPU、GPU/TPU無効、internet無効、competition source、support dataset、5 shard kernel sourceを再確認した。
- aggregate version 1はbootstrap復元、offline wheel解決、graph package installを通過して`RUNNING`。全5 shardの検証・集約結果が出るまで監視を継続する。
- aggregate version 1は`ValueError: official metric source differs across alpha shards`で`ERROR`になった。4つの旧mount shardは`official_metric_source.support_repo=/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1`、`alpha_4`はdirect mountの`/kaggle/input/biohub-tracking-support-pack-50ep-v1`を記録しており、同一の公式metric file SHA 3件にKaggle上のmount pathを含めて比較したためだった。
- aggregateのmetric source整合判定を`evaluator_sha256`、`metrics_sha256`、`division_metrics_sha256`のcontent SHA 3件だけで行うよう修正した。`support_repo`は監査証拠として各manifestに残すが、Kaggleのmount layout差はmetric実体の差と扱わない。公式metric、alpha別score、選択規則、promotion gateは変更していない。mount pathだけが異なる場合は一致し、content SHAが変われば不一致になる回帰testを追加した。
- 修正後にdiagnostic Notebookを再生成し、strict validation、Ruff、11 tests、Jupytext round-trip、Ruff `F821`、diff checkが通過した。
- 2026-09-16T13:06:09+09:00 resource guard: aggregate version 2 packageは`enable_gpu=false`、`enable_tpu=false`、`enable_internet=false`、`run_on_push=true`、入力source 5件を再確認した。CPU実行なのでGPU quota確認は対象外。同じcanonical slugへ再push可能と判断した。
- aggregate version 2を同じcanonical slugへpushした。bootstrap復元、offline wheel解決、graph package installを通過して`RUNNING`。最終集約結果を待つ。
- aggregate version 2は`COMPLETE`。outputを`artifacts/kaggle_aggregate_v2/`へ回収し、199 sample、995 per-sample rows、failure 0、5 alphaの完全coverageを確認した。aggregate処理は`731.5177280590001`秒、5 shard合計は`22380.816667248`秒。
- 44b6で選択した`alpha=0.5`はheld-out 6bba combined scoreを`+0.00026755055423255403`改善し、division Jaccard差は`0.0`だった。この方向はpass。
- 6bbaで選択した`alpha=2`はheld-out 44b6 division Jaccardを`+0.03968253968253968`改善したが、combined scoreを`-0.002478523251885023`低下させた。この方向はfail。
- 両方向のcombined score厳密改善を満たさず、promotion gateは`false`。`next_stage=stop_without_full_repair`として固定repair段階とKaggle submissionは実行しない。
- 最終artifact SHA: per-sample=`459726bc34e3bf9d819fb1e178afc73e0309e97b22c1f51f63e348e0a4a5b74b`、embryo-alpha=`a07345a4c821f808f4f107dc2b59dc2359d6151cd6a146b30ea578452fc84525`、selection=`1cbb3b12d2673949bbbcdc93fdb4f6c84c14303a0b0c3f92694b87699959d4c8`、summary=`6f8de7dc6837f32fbb9c20f5ff8e7a4ad5f8d3fdcaccd0d31ac6b2a84fbfb898`、manifest payload=`4d553972be05e05c1014cb2cdd8f26270971c6e8f4eaf64b14da0399eb70aa29`、manifest file=`777cf81fe3265792766173314a4146a51535135f973bfa4dace4fc0176038344`、solution bundle=`c07b6f85326b5ffafe68d94d706e31757bf1ced4c3990aa69fe28722448fd89c`。
- `metrics.json`を`debug_completed`へ更新し、Kaggle kernel/version、runtime、coverage、score、direction判定、SHAを記録した。`result.md`へ証拠と解釈を記録したが、採用・不採用・完了はユーザー判断待ちのため確定していない。
- 2026-09-16: ユーザーが「実験は完了、手法は不採用（discarded）として確定でいいです」と判断した。`metrics.json`のstatusを`discarded`へ変更し、固定repair段階へ進まないことを確定した。

## 変更点

- exp015のKaggle inference outputにある199個の`oracle_candidate_graphs/*.geff`を入力に固定した。
- `edge_weight=-alpha * edge_prob`のalphaだけを0.25、0.5、1、2、4で変更する。出現0、消失2、分裂1.2とILP制約は固定した。
- SHA固定したexp012 readoutから胚対応を読み、44b6から選択して6bbaで評価する方向と、その逆方向を実装した。
- score tieはalpha=1、次にalpha=1へのlog2距離、次に小さいalphaの順で決める。
- 両方向のcombined score厳密改善、division非悪化、全件有効を後段gateにした。
- 初段はmotion、gap、safe-division、DeepCenter repairを実行しない。gate成立後の固定repair比較も同じ実験内で扱う。
- model config 0、fold学習0、booster 0、control再学習なし。Kaggle CPU、GPU無効、internet無効。
- Kaggle submissionは作成・実行しない。

## Kaggle実行経路

5つのalpha shardを独立したslugへpushし、全件成功後にaggregateをpushした。

```bash
make prepare-kaggle-notebooks EXP=exp018_graph_cost_scale EXTRA_ARGS="--notebook alpha_025 --run-on-push"
make push-kaggle-notebook EXP=exp018_graph_cost_scale NOTEBOOK=alpha_025
# alpha_05、alpha_1、alpha_2、alpha_4も同様
make prepare-kaggle-notebooks EXP=exp018_graph_cost_scale EXTRA_ARGS="--notebook aggregate --run-on-push"
make push-kaggle-notebook EXP=exp018_graph_cost_scale NOTEBOOK=aggregate
```

各shardの`kernel-metadata.json`でCPU、internet無効、exp015・exp012 kernel source、support dataset、id/titleを検査する。aggregateでは5 shard kernel sourceを検査する。実行後は各slugのmetadataをpullして存在を確認し、logsとoutputを回収する。

## 次のアクション

この実験内の次アクションはない。事前gate不成立のため固定repair段階は追加せず、Kaggle submissionも行わない。上位仮説`HYP-20260910-11`には別候補の検証が残るため、この結果だけで仮説全体を終了しない。
