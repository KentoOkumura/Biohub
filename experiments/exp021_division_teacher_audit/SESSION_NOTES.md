# exp021_division_teacher_audit 実行記録

## 2026-09-19

- ユーザーはexp020の完了と教師根拠調査を選択した。主催者の疎注釈仕様と評価器を確認後、「推奨：負例候補を追加診断し、教師を確定する」を選択した。
- make new-exp EXP=exp021_division_teacher_audit SOURCE=experiments/exp020_division_triplet_candidates で親診断をコピー。GTに対応した組の教師分類を追加する。元backlogの学習候補は移行しない。
- 実行予定: CPU diagnostic Notebook 1本、active variant 1、model/config 0、fold学習0、booster 0、control再学習なし。母娘9 µm・娘間14 µmだけを用いる。
- push前の2026-09-19 13:15:54 UTC確認: 生成metadataのidはkentookumura/exp021-division-teacher-audit-diagnostic、title由来slugと一致、private・run_on_push=true。enable_gpu=false、enable_tpu=false、enable_internet=false。CPUのみなのでGPU quota照会は不要。
- 静的検証: make validate-exp、make check-exp、make test-exp（5件）、Jupytext一致を確認した。
- Kaggle CPU Notebook version 1（id_no 134983138）が199/199動画・19,701 windowを完走。exp020の8,807,617組・151分裂・100正例・厳格誤組4件とGT bundle SHAを再現した。別母edgeからの誤組3,933件、厳格誤組とのunion3,935件、単一edge母に既知娘を含む未確定組89,703件を得た。Kaggle outputの表199行とsummary/表SHAをmanifestと照合した。
- 3,933件が既知分裂母と単一edge母のどちらに付くか、100正例母に同じ母の負例があるかが学習設計上重要なため、母のGT出力次数と正例母との重複を追加したversion 2を同slugで実行する。候補・教師定義・幾何条件は変えない。
- version 2 push前の2026-09-19 13:29:23 UTC確認: 同じcanonical id/title、private・run_on_push=true、enable_gpu=false、enable_tpu=false、enable_internet=false。CPUのみなのでGPU quota照会は不要。
- Kaggle CPU Notebook version 2が同じid_no 134983138で199/199動画・19,701 windowを完走。診断処理495.43秒。exp020の候補8,807,617件、151分裂、100正例、厳格誤組4件、GT bundle SHAを再現した。
- 別母edgeによる誤組3,933件の内訳はGT出力2本の母3件、1本の母3,811件、0本の母119件。厳格誤組との重複2件、union3,935件。正例100母のうち同じ母に別母edge由来の誤組があるのは1件、厳格誤組があるのは2件。単一edge母で既知娘を含む未確定組89,703件。
- make kaggle-outputでversion 2のmanifest.json、summary.json、per_sample.csvを取得。199行の表SHA 683725138ab575f4442e12eb9fbc13448663959a7ac2294e1715665698ba9834、summary SHA 5efbe85936643be7676a3014bf384b811e4d727f57730f27922631d719fed1baをローカルsha256sumと照合。小さな生成物はignored artifacts/diagnostic_v2に保存した。Notebook全体の確定runtimeは未取得。
- metrics.jsonへdebug_completedとして実行証拠を記録し、ユーザーが本診断の完了を判断した後にcompletedへ更新した。元のdivision_tripletsは追加の教師を探してから学習する方針となり、学習の実験化はしていない。
