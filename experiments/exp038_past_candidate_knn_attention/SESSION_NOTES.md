# exp038_past_candidate_knn_attention セッションノート

## 2026-09-23: 実験化・実装

- ユーザーが再設計候補に「実行してください」と依頼。Kaggle上の保持率・費用診断と、条件成立時の学習・単体評価まで実行する。全graph・submissionは含めない。
- `kaggle-review-exp`でexp037から`make new-exp EXP=exp038_past_candidate_knn_attention SOURCE=experiments/exp037_past_candidate_attention EXTRA_ARGS=--copy-tests`を実行。実行記録をリセットし、候補契約をrequirementsへ移行した。
- active variant 1、設定1、外側2fold、各3 epoch、新規model 2個。exp016 controlの再学習なし。保存済みfold重みを同じ条件で評価する。
- `uv run kaggle quota --format json`でGPU使用31.95h、残13.05h、無料枠45h、refresh `2026-09-26T00:00:00`を確認。通常の週30hを超えたため確認し、ユーザーから「今回のみ週45時間まで許可する」を受領。一般方針は変更せず、この実験のconfigだけに例外を記録した。push直前にも再確認する。
- 近傍選択はsource単位で行い、三点特徴・MLPの前に最大8点をgatherする。中央の接続候補は全て保持する。
- CPU版PyTorchをローカル依存へ追加し、skipではなく出力・勾配のテストを実行。最初は親由来の契約テスト1件が旧系譜を期待して失敗し、新しい候補名・親・設定へ更新した。残り13件は通過した。
- 親Notebookは133行のhelper呼び出し中心でcompact版なし。新NotebookはASTで到達する定義のみを選び、入力監査、候補保持率、費用、学習、単体比較、保存を個別セルに展開する。動的なsource実行や同実験helperのimportをNotebookへ残さない。

## 検証・実行記録

- ローカル検証: `check-exp`、`validate-exp`が通過。PyTorchを含む実験固有テスト21件が通過し、skipなし。
- `kaggle-strategy`で元候補を未着手索引から削除。詳細はrequirementsへ全て移行してあり、情報は復元可能。対応する仮説へexp038リンクを追加した。必須リンク増加で仮説行が1000文字を超えたため、その行種別だけの上限を1200へ調整し、影響する`tests/test_strategy_docs.py`25件が通過。通常行800文字・文書全体50000bytesの制限は維持した。
- 2026-09-23 00:04:19 UTC: push前quota再確認。GPU使用31.95h・残13.05h。今回承認の45h上限内。T4割当数を掛けた安全側のGPU時間を使い、Notebook 12hとの小さい方を実行上限にする。本学習は保持率・費用診断通過時のみ開始する。
- Jupytext初回往復検証では冒頭のlintコメントにcell markerがなく差分を検出し、生成器へmarkerを追加した。

1. `make check-exp EXP=exp038_past_candidate_knn_attention`、`make test-exp EXP=exp038_past_candidate_knn_attention`、`make validate-exp EXP=exp038_past_candidate_knn_attention`。
2. Jupytext変換・往復検証、バックログ移行確認。
3. quota・metadata確認後に`make prepare-kaggle-notebooks EXP=exp038_past_candidate_knn_attention EXTRA_ARGS="--notebook train --run-on-push"`と`make push-kaggle-train EXP=exp038_past_candidate_knn_attention`。
4. 同じkernelのlive logsで結果を確認。保持率または費用条件未達ならそこで停止し、結果を回収する。

### Kaggle version 1

- 2026-09-23 00:09 UTC: push直前のquotaも使用31.95h・残13.05hで変化なし。`make push-kaggle-train EXP=exp038_past_candidate_knn_attention`がversion 1の転送成功を返した。
- canonical kernel: `kentookumura/exp038-past-candidate-knn-attention-train`。private、T4、GPU有効、TPU無効、internet無効を生成metadataとpush後の`kernels pull ... -m`で確認。Kaggleのkernel IDは135446623。
- `make kaggle-logs KERNEL=kentookumura/exp038-past-candidate-knn-attention-train`でlive SSEへ接続した。初期の無出力はqueue/provisioning中として扱い、別slug・別versionの再pushを行っていない。
- `record-exp`で実行開始と検証証拠をmetricsへ記録。採用・不採用・完了は未判断。元のexp037の判断も変更していない。
- live logと回収したmetricsで、Setup and configurationの`KeyError: past_candidate_knn_attention`を確認。設定の枝名は`past_candidate_attention`のままであるため、読み取りを一致させた。実行variantのassertionとconfigのkernel IDにも親の名前が残っており、新実験へ訂正した。候補数・学習・評価条件に変更なし。
- `kaggle-review`の失敗調査手順でconfigとsourceを照合。pipelineと生成Notebookの実際の設定読み込みコードを実行する回帰テスト2件を追加し、全23件通過。check-exp、validate-exp、Jupytext往復も通過した。
- `kernels output .../1 --file-pattern 'metrics.json$'`で初回停止の証拠を`artifacts/kaggle_v1/`へ回収。実行時間とSHAはmetricsのrerunsへ保存する。候補保持率・費用測定・学習はいずれも未開始だった。

### Kaggle version 2

- 2026-09-23 00:16 UTC: quotaはGPU使用31.97h・残13.03h。今回のみ45hの許可を維持し、通常方針は変更しない。同じcanonical kernelへ修正版をpushする。
- `make push-kaggle-train EXP=exp038_past_candidate_knn_attention`でversion 2の転送に成功。00:18 UTCにlive logでSetup and configurationの通過とInput cache and source integrityへの移行を確認した。割当GPUは2台で、予算チェックの実行時間上限はmetricsへ記録した。
- push後に`kernels pull`でprivate、GPU有効、TPU無効、internet無効、NvidiaTeslaT4を再確認。保持率・費用・本学習の測定結果はまだ出ていない。公式score・submissionは未実行。
- Kaggle version 2は候補保持率と費用の診断を終え、`Stratified runtime and stress benchmark`で事前の時間条件により本学習前に停止した。`kernels output .../2`で4つのJSONとlogを`artifacts/kaggle_v2/`へ回収した。
- 学習側の既知edgeで、両端が固定候補と対応するものを分母にした過去候補8点の保持率はfold 0が16552/16554 = 99.9879%、fold 1が93628/93681 = 99.9434%。両foldとも事前条件99%以上を満たした。両端欠落は別計上であり、この保持率に含まれない。
- 費用診断は残作業25,649.13秒、準備済み876.47秒、残作業への保守係数1.5で計39,350.16秒（10.93時間）。push前の残GPU枠13.03時間を割当T4 2台で按分した上限23,454秒（6.515時間）を超えた。peak allocated memory 408,843,776 bytesは上限12,508,830,105 bytes内。Notebook記録時間876.50秒。時間条件のみ未達で、2-fold本学習とexp016との単体比較は未実行。
- 2026-09-23 13:26 UTCにaccount全体のGPU使用40.52h・残4.48h、週次reset `2026-09-26T00:00:00`を確認。この増分をexp038だけの使用量とは推定しない。新たなKやfold/epochの変更、再pushは行っていない。
