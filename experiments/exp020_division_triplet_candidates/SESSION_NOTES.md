# exp020_division_triplet_candidates 実行記録

## 2026-09-19

- ユーザーはdivision_tripletsの実装を依頼した。候補詳細が設計不可で、候補範囲・負例・loss・decodeが未決であることを確認した。
- ユーザーは「推奨：候補と教師の診断から進める」を選択した。直接承認の診断実験exp020を作成した。元のbacklog候補は移行しない。
- 実行予定: CPU diagnostic Notebook 1本、model/config 0、fold学習0、booster 0、control再学習なし。exp015の固定cache 19,701 windowとtrain GEFFを読む。
- コマンド: make new-exp EXP=exp020_division_triplet_candidates。
- Kaggle実行と生成物SHAは後続記録へ追記する。
- 静的検証: make validate-exp、make check-exp、make test-exp（4件）、Jupytext一致を確認した。
- 2026-09-19 12:26:12 UTCのpush前確認: diagnostic metadataはenable_gpu=false、enable_tpu=false、enable_internet=false。CPUのみを使うためGPU quota照会は不要。kernel idはkentookumura/exp020-division-triplet-candidates-diagnostic、title由来slugと一致し、private・run_on_push=true。
- Kaggle CPU Notebook kentookumura/exp020-division-triplet-candidates-diagnostic version 1をmake push-kaggle-notebookで実行。push後に同slugをpullしてid_no 134978284とCPU/internet falseを確認した。
- live SSE logsで20動画ごとの進捗、199/199完走、3条件の集計、出力保存を確認。診断処理の経過1,209.55秒、最終ログイベント1,257.49秒。Notebook全体の確定runtimeは未取得。
- make kaggle-outputでmanifest.json、summary.json、per_sample.csvを取得。manifestの597行・summary SHA b28beb7c24a36df5b7677bfae40704aa6351a32c5a645bac0f4dae7bdbcd7899・表SHA 2f299da75b333ce659503716b815827b8d14b68696351089c231ac018a179247をローカルsha256sumと照合した。
- safe_repairは組8,807,617件、既知分裂100/151、確定誤組4件。broad_diagnosticは組61,810,786件、129/151、確定誤組11件。通常継続の確定負例はなし。metrics.jsonへdebug_completedとして記録し、ユーザーの完了判断を待つ。
- ユーザーは「推奨：診断を完了し教師を調査」を選択した。metrics.jsonの実験statusをcompletedに更新。元のdivision_tripletsの学習実験化は未承認で、教師の根拠を先に調査する。
