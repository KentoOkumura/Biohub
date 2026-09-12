# exp011_public_detector_selection 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog候補`public_detector_selection`の状態は`設計可能・実験化未承認`、未決事項は`なし`。2026-09-12のユーザー依頼「public_detector_selectionを実装してください」を実験化承認として扱う。
- 移行元backlog: `backlog/public_detector_selection.md`
- 対応する上位仮説: `HYP-20260910-12`
- 上位仮説のうちこの実験が検証する範囲: 固定画像特徴を再利用できる公開モデルが存在するかを確認し、取得可能な版、特徴抽出方法、トラッカー初期値、学習来歴、利用条件、計算費用の前提を固定する。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: `exact_window_cache`での保存・読み込み等価性と実費用、`frozen_image_encoder`での下流学習成立と精度、hidden testを含む全件推論費用。
- 親実験: N/A。保存済み公開Notebook調査を比較の入口とする選定調査であり、自前の親実験はない。
- 根拠 / 一次資料 / 参照実装: `docs/surveys/biohub-public-baselines_20260910.md`、`docs/surveys/biohub-backlog-readiness_20260910.md#exact_window_cache`、Kaggleの公開Notebook 4件、Pilkwang公開dataset 3件のmetadata・artifact manifest・checkpoint。
- 固定するもの: 公開検出器と画像特徴抽出器を更新せずトラッカーを学習する方針、Kaggle Notebookのみ、GPU週30時間以内、課金なし、既存の評価方針。初回比較では候補点生成、特徴抽出、secondary branch、ILP、repair設定も固定する。
- 変更するもの: 未指定だった公開Notebook、検出器・画像特徴抽出器checkpoint、既存トラッカー初期値の具体的な版を選定する。
- 最小の反証可能な検証: 公開Notebook、参照checkpoint、metadata、manifestを照合し、取得可能性、SHA、モデル対応、時間窓、前処理、特徴取得、tracker input、学習来歴、利用条件を確認する。静的確認で決まらない動作・費用は必要な小規模Kaggle確認として明示する。
- 成功条件: 推奨Notebookと重みの組を一意に指定し、固定対象、下流学習への渡し方、判断理由、未確認事項をユーザーへ提示できる。
- 停止条件: 必要な重みが取得不能、必要な特徴が取得不能、画像側の再学習が必須、または予算内で使える根拠がない場合は不適合理由を記録し、同じ方針内の別の公開構成を調べる。自前検出器学習や外部GPUへ切り替えない。
- 実行しないこと: タイトルの最高スコアだけで選ぶこと、自前重みで公開重みを代用すること、全公開モデルの再学習、全件特徴cache、トラッカー学習、Kaggle submission、後続候補の完了待ち。
- 未決事項: なし。選定案の採用は実装後のユーザー判断であり、調査開始前の未決事項ではない。
- backlog記録から解釈を変更した箇所とユーザー承認: N/A。

## 判断履歴

- 2026-09-12: 公開モデルの選定を既存候補の前提・未決事項だけに残さず、P1先頭の独立候補として追加した。選定や採用が完了したという記録ではない。
- 2026-09-12: 調査で明らかにする事項と設計前の未決事項を分離し、状態を`設計可能・実験化未承認`、未決事項を`なし`へ訂正した。
- 2026-09-12: ユーザー依頼「public_detector_selectionを実装してください」により実験化を承認された。
- 2026-09-12: ユーザー「採用でいいです」により、推奨した公開Notebook・3 checkpoint・primary tracker初期値の組を採用した。

## 手法契約

実装区分は`docs/glossary.md`に定義したこのリポジトリ内の管理用ラベルを使う。本実験はモデルを学習・実行せず、公開構成の静的な選定結果を固定する。

- 依頼原文: 「公開検出器を固定し、トラッカーを学習する こと前提として、バックログを修正してください」「選定するバックログはありますよね？」「public_detector_selectionを実装してください」
- 期待する成果: 使用する公開Notebookの版、checkpoint、既存トラッカー、利用条件、学習来歴、特徴取得方法を特定し、選定理由と未確認事項を後続候補へ渡す。
- input: 公開Notebook、参照コード、dataset metadata、artifact manifest、checkpoint、公開結果の記録、Kaggleの資源制約。
- target / objective: 固定検出器の下流でトラッカーだけを学習できる具体的な構成を比較し、根拠付きの一意な推奨案を作る。
- output: Notebook URLとcontent SHA、dataset版、checkpoint SHA、検出と接続に使うモデルの対応、時間窓、前処理、候補特徴、tracker input、初期値案、学習来歴、利用条件、費用根拠、未確認事項を含むmachine-readable manifestと調査レポート。
- loss: なし。検出器・トラッカーを学習しない。
- decode: 公開構成のcandidate detection、association、ILP、graph repairを分離して記録し、初回比較で固定する処理を特定する。
- context unit: Notebookの取得版と、その版が参照するcheckpoint、前処理、候補特徴、下流モデルの組。
- 実装区分: `staged-faithful`。公開0.946構成と参照artifactを静的に固定する段階で、Kaggle上のfull inferenceやスコア再現は実施しない。
- 省略する機構と理由: モデル実行、全件特徴cache、再学習、submissionを省略する。選定に必要な静的証拠を先に確定し、実行費用と予測等価性は後続の独立比較に残すため。
- proxyで検証できない主張: N/A。proxyではないが、静的調査だけではPublic LB、hidden test runtime、予測等価性、下流学習の精度を検証できない。
- proxyの場合のユーザー承認: N/A。
- この実験が支持 / 棄却できる主張: 取得可能な公開checkpointと互換する公開コードから、画像モデルを更新せずにcandidate featureを取り出し、既存trackerを初期値にした後続実験を定義できるか。
- この実験では判断できない主張: detector単体精度、0.946 Public LBの独立再現、保存特徴の完全等価性、tracker再学習の改善幅、hidden test完走時間。

## 実装方法

- アプローチ: 現行の公開Notebook 4件を取得して比較し、0.946 Notebookの処理と参照SHAを抽出する。Notebook内のmirrorではなく、同じ期待SHAを持ち利用条件を確認できるPilkwangのoriginal datasetをcanonical artifactとして固定する。
- inputの実装箇所と変換: `assets/public_detector_selection.json`にNotebook、dataset、checkpoint、feature schema、制約を保存する。Notebookは取得時content SHA、datasetはversion、artifact manifest SHA、checkpoint SHAで固定する。
- target / objectiveの構築箇所: `assets/public_detector_selection.json`の`selection_state`と`pipeline_contract.downstream_training_recommendation`に選定案を表現する。
- outputの生成箇所と表現: machine-readableな正は`assets/public_detector_selection.json`、人が読む根拠と限界は`docs/surveys/biohub-public-detector-selection_20260912.md`。
- lossの実装箇所: N/A。
- decode / postprocessの実装箇所: コードは実装せず、`pipeline_contract.tracker_control`と`pipeline_contract.decode`に初回比較で固定する公開処理を記録する。
- context unitを保つ処理箇所: manifestの`reference_notebook`、`canonical_public_artifacts`、`pipeline_contract`を一組として扱う。
- 変更するファイル / component: exp011の契約・設定・記録、選定manifest、manifest contract test、調査レポート、移行後の戦略索引。
- 固定事項を保つ確認方法: testでNotebook識別子、artifact数・version・SHA、window size、downsample、feature channel、freeze方針を検査する。
- 参照sourceとの一致を確認するテスト: 取得した3 checkpointのSHAと公開Notebook内の期待SHAが一致することを確認し、testで値と形式を固定する。
- 承認済み差分を確認するテスト: 選定状態が`adopted`であり、公開検出器・secondary branch・DeepCenterをfreezeし、primary `SimpleNodeTransformer`だけを最初の学習対象にする記述を検査する。

## 探索幅とpivot判定

- 変更class: `selector-only`。
- 同じ親 / familyで連続した小改善実験数: 0。選定調査であり学習実験ではない。
- positiveなoracle headroom / coverage / 誤差非相関性: 未測定。後続の公開基準診断とOOFで測定する。
- 比較したtarget、output、decode、context unitを変える案: 4公開Notebookのうち、secondary association-feature TTA、DeepCenter TTA、別のrepair設定を持つ構成を比較したが、初回controlへ同時導入しない。
- 小改善の継続またはpivotを選ぶ根拠: 固定画像特徴とtracker初期値を一意に指定できたため、まず後続の等価性確認とtracker学習へ進む。静的調査だけで別representationへpivotしない。
- `kaggle-idea-forge` の実行要否と根拠: 不要。承認済みの具体的な選定候補を実験へ移行する作業であり、新規アイデア探索ではない。

## 再現性・リスク

- seed policy: 本実験には乱数を使う処理がない。公開学習来歴のseedとdeterministic情報は判明した範囲だけmanifestへ記録する。
- stochastic 処理の有無: なし。
- stochastic feature generation / augmentation / seed bagging の有無: なし。公開NotebookのD4変換は決定的な8-view列挙として記録する。
- 並列処理と乱数の関係: N/A。
- CPU/GPU runtime と deterministic flags: metadata・SHA比較はCPUのみ。Kaggle Notebookは実行しない。secondary公開checkpointは学習時`deterministic=false`であることを明示する。
- train cache / test feature regeneration の SHA 記録方針: 本実験ではcacheを生成しない。後続のcacheはdecompressed content SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: 選定manifest SHAと3 checkpoint SHAを`metrics.json.evidence.artifacts`へ記録する。predictionとsubmissionは生成しない。
- Kaggle package bootstrap 確認方針: Notebookを実行しないため対象外。後続の小規模Kaggle確認でoffline dependencyとoriginal dataset mountを確認する。
- リークリスク: primaryのbest epoch・split・seed・独立validationは不明。secondaryは独立validationではなく、公開スコアも複数モデル・後処理込みである。独立評価済みdetectorと扱わない。
- CV/LB 不一致リスク: 0.946はNotebook題名・本文に基づく報告で、現行Public Scoreの取得や独立再現をしていない。detector単体の性能へ帰属しない。
- ランタイム/メモリリスク: hidden test全件のruntime、candidate数、feature保存量を未測定。公開ページの30–40分程度という先行調査は比較情報に留める。
- 再現性リスク: Kaggle CLIがkernel version番号を返さないため、kernel id、取得時刻、content SHAで取得版を識別する。Notebook独自patchのcode licenseは未確認。
- 手法忠実性リスク: original Pilkwang datasetの3 checkpoint SHAはNotebook期待値と一致したが、mirrorからoriginal datasetへmountを変えた実行等価性は未確認。
- 過度な縮小 / proxy化リスク: 静的選定をスコア再現や動作確認と同一視しない。必要な小規模Kaggle確認を後続へ明示する。

## 受け入れ基準

- [x] 手法契約の `input / target / output / loss / decode / context unit` が選定manifestと一致する。
- [x] 実装区分と実験名が実装した機構を正確に表す。
- [x] `proxy`ではなく`staged-faithful`であり、省略点と検証不能な主張が記録されている。
- [x] backlogの根拠、差分、成功条件、停止条件、実行しないこと、判断履歴を移行した。
- [x] `config.yaml`の`lineage.hypothesis_id`と`lineage.backlog_candidate`がこの文書と一致する。
- [x] 実験固有テストと静的検証が通る。
- [x] deterministic anchorとして扱わず、取得版の識別子と必要なSHAを`metrics.json`へ記録する方針を定めた。
- [x] gzip生成物を比較しない。
