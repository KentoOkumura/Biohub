# exp034_frame_self_attention_distance_bias セッションノート

## 目的

exp025の恒等初期化Model Bへ学習可能な物理距離biasを1設定だけ追加し、2fold×3epochでexp033の保存済みpair対照と比較できるKaggle train notebookを作る。

## 現在の作業

- 作業内容: 実装と局所検証。
- ブロック要因: なし。
- 次: 距離bias実装、test、notebook再生成、静的検証。

## コマンドログ

### 2026-09-22

- make new-expでexp025からexp034を作成。
- exp033の保存済み診断を契約へ反映し、1設定×2fold×3epoch、fold別scale、pair gate、停止条件を固定。
- 実行: validate-exp strict通過、Ruff check/format通過、py_compile通過。
- 実行: pair評価規則test 6件通過。ローカル環境にPyTorchがないためmodel test 1 fileはskip。
- 実行: jupytext round-trip test通過。Kaggle train packageを生成しmetadata push-readyを確認。
- 実行: backlog移行直後のstrategy documents検査は通過。後続の全体再検査は並行追加されたexp035_velocity_featuresがHYP-20260910-10表に未登録のため失敗し、exp034行とは無関係。
- Kaggle kernelのpush、学習実行、graph replay、submissionは未実行。

## 変更点

- 各Self-Attention headへ物理距離の負の二乗biasと非負の学習可能係数を追加する。
- fold別distance scaleはexp033の学習側最近傍距離中央値へ固定する。
- 追加枝は恒等初期化し、公開trackerとの初期logit完全一致を必須にする。
- graph replayはpair gate通過後まで追加しない。

設定と再現性方針はconfig.yaml、実行後の構造化結果はmetrics.jsonへ記録する。

## 予定コマンド

- make validate-exp EXP=exp034_frame_self_attention_distance_bias
- make check-exp EXP=exp034_frame_self_attention_distance_bias
- make test-exp EXP=exp034_frame_self_attention_distance_bias
- make prepare-kaggle-notebooks EXP=exp034_frame_self_attention_distance_bias EXTRA_ARGS=notebook-train

Kaggle実行が承認された場合だけ、prepare後のmetadataを確認してtrain kernelをpushする。

## 次のアクション

1. 実装差分と未実行範囲をレビューする。
2. Kaggle実行が指示された場合だけtrain kernelをpushする。
3. 実行後に両foldのpair gateを判定する。

### 2026-09-22T09:39:48+09:00 Kaggle push前確認

- 対象: kentookumura/exp034-frame-distance-bias-train。
- resource: GPU、machine_shape: NvidiaTeslaT4、TPU無効、internet無効。
- Kaggle quota: GPU 18.03h remaining / 45.00h、refresh 2026-09-26T00:00:00。
- 判断: runtime gate最大12hを上回る残時間があるためpush可能。
- validate-exp、Ruff、metadata検証は通過。testは6 passed、PyTorch未導入のローカルmodel test 1 file skipped。

### 2026-09-22T09:59:06+09:00 Kaggle train version 1

- push: kentookumura/exp034-frame-distance-bias-train version 1。run-on-push。
- Kaggle metadata: Tesla T4、GPU有効、TPU無効、internet無効、id_no 135306553。
- live logs: 接続直後にAPI 500が2回発生。補助statusでRUNNINGとpullで同一kernelを確認し、再接続後に保存済みlogを取得。別slugへの再pushは行っていない。
- 入力: 19,701 window中18,707 eligible、994 skipped。fold window countsは5622/693/12392と11236/1156/6315。
- benchmark fold 0: 64 window、16.383375667秒、projected 23708.408527秒、peak 483793920 bytes。
- benchmark fold 1: 64 window、17.053016207秒、projected 26921.250297秒、peak 514729472 bytes。
- 合計保守見積り50629.658823秒が12h gate 43200秒を超え、full training前にRuntimeErrorで停止。Kaggle statusはERROR。
- checkpoint、OOF pair prediction、pair gate、graph score、submissionは未生成。
- output取得先: artifacts/kaggle_train_v1。benchmark SHA 3b08b4543ab1fa9998db38e9056461cfdbd501aa3a3a78f98c0e72705ee9701a、log SHA 4ebe0ee7bcdbe7ffa364c3f85f334089d5b33d06726af3b8a336e663442984d2。
- 現契約の停止条件を満たしたため、自動再pushは行わない。

### 停止後quota確認

- GPU 17.84h remaining / 45.00h、used 27.16h、refresh 2026-09-26T00:00:00。
- 2fold合計の保守見積りは約14.06hで残quota内だが、単一Notebookの12h上限を超える。fold別kernel分割は再設計承認後の候補。

### 2026-09-22 方針変更: 最終graphを直接評価

- ユーザー判断: 「普通に最終的な予測結果を評価すればいい」。
- pair ranking、bucket境界計算、pair gate、学習前後の重複outer pair評価をtrain Notebookから削除。
- train runtime見積りは3 epochのtrain/internal validationと2窓identity確認だけに変更。
- 学習後は独立Inference Notebookで全199動画を固定cacheからreplayし、exp015と同じthreshold、双方向・secondary融合、ILP、graph repairを適用する。
- fold 0を6bba、fold 1を44b6へ適用したholdout最終graphを、保存済みexp015 controlと同じ公式評価器でoverall・胚別・sample別に比較する。
- exp025のgraph replayを移植し、空間biasとfold別distance scaleをcheckpointからstrict復元する経路を追加。
- Kaggle submissionは作成しない。

### 2026-09-22 改訂train version 2 push前確認

- 対象: kentookumura/exp034-frame-distance-bias-train。同じslugへversion 2としてpushする。
- resource: GPU、machine_shape: NvidiaTeslaT4、TPU無効、internet無効。
- Kaggle quota: GPU 17.84h remaining / 45.00h、refresh 2026-09-26T00:00:00。
- version 1実測を改訂後のwork window数へ当てた保守見積りは約6.15h。pair診断と重複outer評価は含まない。
- validate-exp、Ruff、py_compile、testを通過。testは3 passed、PyTorch未導入のmodel test 1 file skipped。
- 残quotaがtrain見積りを上回るためpush可能と判断。


### 2026-09-22 改訂train version 2完了

- Kaggle status COMPLETE。2fold×3epochを完走し、2 checkpointとmodel manifestを保存した。
- 実測学習時間3439.393329秒。benchmarkの合計保守見積り21493.307195秒、runtime gate 43200秒。
- fold 0はepoch 0を選択し、internal selection score 0.985950574622095。checkpoint SHA 83dd2421921c80137dbe806f48584cad8d883269c494c868b7c5620ee864c845。
- fold 1はepoch 2を選択し、internal selection score 0.9782032195598147。checkpoint SHA 0a39707a63b30afcacf59b6e5b3b2cb4c2c6125df2a80271237717438c3f54c1。
- model manifest SHA df7a94d5a04651ac8922f3dd8fe5e672a99b91f99ab956a677b95616939c12e2。取得後のcheckpoint SHA再計算はmanifestと一致。
- 出力保存先: artifacts/kaggle_train_v2。submissionは作成していない。
- 次: manifest SHAをconfigへ固定し、Inference Notebookで全199動画のholdout最終graphを公式評価する。


### 2026-09-22 Inference push前確認

- manifest SHAをconfigへ固定し、Inference Notebookを再生成した。validate-exp、Ruff、test 3 passed / 1 skipped、diff check、Kaggle package validatorを通過。
- canonical kernel: kentookumura/exp034-frame-distance-bias-inference。private T4、TPU無効、internet無効、run-on-push、submissionなし。
- 実行対象: 全199動画・19,701 window・tracker forward 98,505回。追加学習0、control再学習0。
- Kaggle quotaはGPU 28.44/45.00時間使用、16.56時間残、2026-09-26T00:00:00更新。
- 現行方針の週30時間上限に対する残りは1.56時間。同じ98,505 forward構成のexp016実測は23,217.36秒（6.45時間）で、残予算を約4.89時間超える。
- packageはpush-readyだが、週30時間上限を守るためpushしていない。上限超過の明示承認またはquota更新を待つ。


### 2026-09-22 ユーザー完了判断

- tracker学習結果を前段評価として再確認した。fold 0はepoch 0が最良で、その後の改善なし。fold 1のepoch 0からepoch 2へのinternal selection score差は約+0.00000345。
- 上記は保存済みcontrolとの公式graph比較ではないが、週30時間のGPU上限を約4.89時間超えて全graph推論へ進む改善根拠としては弱い。
- Inference packageはpush-readyまで検証したが、Kaggleへpushしていない。最終graph、公式score、submissionはいずれも未生成。
- ユーザーが本実験を完了と判断した。metrics statusをcompletedへ更新し、公式graph改善の仮説は未確認のまま終了する。
