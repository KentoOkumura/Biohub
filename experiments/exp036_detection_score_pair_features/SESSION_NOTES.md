# exp036_detection_score_pair_features セッションノート

## 目的

exp015の固定候補検出得点からcandidate pair特徴を作り、primary接続logitへの学習可能な残差が、同構造の0入力対照より隣接接続と分裂親回収を改善するか検証する。

## 現在の作業

- 作業内容: 案3のKaggle version 2結果を回収・検証し、ユーザー判断により不採用で終了した。
- ブロック要因: なし。
- 次: この実験に関係する変更だけをcommit・pushする。追加実行は行わない。

## 時系列ログ

### 2026-09-22

- ユーザーが「案3で進めてください」と判断し、candidate pair特徴から接続logitへの残差を学習する方針を承認した。
- `exp016_frozen_image_encoder`を親に`exp036_detection_score_pair_features`を作成した。
- 入力はsource標準化logit、target標準化logit、小さい方、絶対差の4特徴とした。
- 4→16→1の残差headを追加し、最終層を0初期化した。
- 対照は同じhead・学習条件を持ち、標準化後の得点を0へ置換する。
- fold別normalizerはgradient-update windowだけからfitし、internal validationと外側胚は統計計算へ使わない。
- 2 variant×2foldの4 trackerを同じfold seedとDataLoader順で学習するようにした。
- GPU実行予定はactive variant 2件、model/config 1件、fold 2件、合計4 tracker、booster 0件。0入力対照の再学習あり。追加コストは新規variantだけを学習する場合の約2倍だが、同じ追加headと今回の学習乱数をそろえて検出得点の情報だけを比較するため、保存済み親実験の結果では代替しない。ユーザーは案3の実装後、Kaggle実行を明示的に依頼した。
- 各foldでoptimizer更新前のstate SHAとlogitがvariant間で完全一致することを検査する。
- 外側評価へ得点帯別のpositive edge recall、active negative pair false-positive rate、division parent recallと件数を追加した。
- 両foldで`positive_edge_recall`、`edge_accuracy`、`division_parent_recall`がすべて対照以上、かつ正例と分裂親の件数が非0の場合だけ進行可能とした。
- inference、公式graph評価、submissionは初段から除外した。
- PyTorchなしのローカル環境でもconfig、normalizer、進行条件を検査できるテストと、PyTorchがある環境向けのhead・診断テストを分離した。
- Jupytext sourceからtrain Notebookを再生成した。
- 実装途中の局所テスト: 4 passed、PyTorch固有1 file skipped。
- `make validate-exp EXP=exp036_detection_score_pair_features`: strict validation passed。
- `make check-exp EXP=exp036_detection_score_pair_features`: Ruff check・format check passed。
- `make test-exp EXP=exp036_detection_score_pair_features`: 4 passed、PyTorch固有1 file skipped。
- strategy文書と変更Markdown・Notebookのlocal math checkがpassed。数式は0件で、GitHub変換と実表示は対象外。
- train-only Kaggle packageを準備し、metadataと埋込みsourceを検証した。push-ready idは`kentookumura/exp036-detection-score-pair-features-train`。pushとKaggle実行は行っていない。
- 2026-09-22 22:12 JST: run-on-push packageを再生成し、push直前metadataがGPU有効、TPU無効、`NvidiaTeslaT4`、internet無効であることを確認した。Kaggle GPU quotaは45.00時間中16.56時間残、refreshは2026-09-26 00:00で、実験の12時間runtime gateを上回るため実行可能と判断した。CLI 2.2.4ではActive Sessions数を取得できないためpush前gateには使わない。
- Kaggle train version 1をpushし、Kaggle側metadataの`NvidiaTeslaT4`をpullで確認した。入力解決、cache identity、19,701件中18,707件のGT window filter、fold split、公開tracker state SHAまでは成功したが、最初のruntime benchmarkで`TypeError: ~ (operator.invert) is only implemented on integer and Boolean-type tensors`により失敗した。
- 原因は`tracker_logits`が`DetectionScorePairTracker.forward`へ検出得点、座標、maskを異なる位置順で渡し、公開Transformerへ座標tensorがmaskとして入ったことだった。呼出順をwrapper契約へ合わせ、Boolean maskと引数順のPyTorch回帰テスト、ローカルで常時動くAST回帰テストを追加した。修正後はRuff、strict validation、5件のCPUテストが成功し、PyTorch固有1 fileはローカル依存不足によりskipした。
- 2026-09-22 22:43 JST: version 2 push直前のGPU quotaは45.00時間中16.18時間残、refreshは2026-09-26 00:00。12時間runtime gateを上回るため再実行可能と判断した。
- version 2は同じcanonical kernel IDへpushし、Kaggle側metadataの`NvidiaTeslaT4`を再確認した。両foldの64-window benchmarkは15.77秒と15.61秒、peak GPU memoryは494,823,424 bytesと534,053,888 bytes、校正済み総実行時間見積りは19,638.53秒（約5時間27分）で、43,200秒のgateを通過して本学習へ進んだ。version 1の引数順エラーは再現しなかった。

### 2026-09-23

- ユーザーからKaggle run完了の連絡を受け、CLI statusが`COMPLETE`であることと保存ログを確認した。
- version 2 outputを`artifacts/kaggle_train_v2/`へ取得した。Notebook実行時間は8,882.15秒、学習・評価区間は7,472.25秒だった。
- model manifest、fold別normalizer、GT window filter audit、4 checkpoint、各variant・foldのsummary、training summaryを取得した。manifest、normalizer、audit、4 checkpointの実ファイルSHAは記録値と一致した。
- 4 model、fold内のvariant間初期state一致、gradient-update候補だけでfitしたnormalizer、各variant・foldの6得点帯診断、支持件数、全graph停止flagを機械検証した。
- fold 0は得点ありvariantのpositive edge recallが対照比`-0.0001063902`、edge accuracyが`-0.0000000472`、division parent recallは同値。fold 1はpositive edge recallが`-0.0004221859`、他2指標は同値だった。
- 両foldでpositive edge recallが低下したため進行条件は未達。`full_graph_inference_allowed: false`に従い、全graph推論、公式評価、submissionは実行しない。
- 結果反映後のstrict validation、Ruff check・format check、実験固有テスト13件、変更Markdown 3件のlocal math check、strict実験記録reviewerがすべて成功した。
- ユーザーが「不採用で終了し、commitとpushしてください」と判断した。`metrics.json`の実験statusを`discarded`へ更新し、公式scoreとPublic LBは未計測のまま終了する。

## 実行結果と次回

train-only比較は完走したが、追加した検出得点pair特徴は同構造対照を改善しなかった。事前契約どおり全graph推論へ進まず、ユーザー判断により不採用で終了した。公式scoreとPublic LBは未計測である。
