# exp024_tracker_six_epochs セッションノート

## 2026-09-20 実験化と実装

- ユーザーの「実装に進んでください」を、設計可能・未決事項なしの`tracker_six_epochs`の実験化承認として受けた。`exp016_frozen_image_encoder`を親に`exp024_tracker_six_epochs`を作成し、候補詳細の契約を`requirements.md`と`config.yaml`へ移した。
- 学習予定: active variant 1、model/config 1、胚を入れ替える2fold、booster 0、出力tracker 2個。保存済みexp016の3エポック対照は再学習しない。公開画像encoder、検出器、secondary trackerも学習しない。
- 変更する学習条件はエポック上限3→6だけ。初期checkpoint、固定feature cache、split、教師、loss、optimizer、学習率、復号はexp016と揃える。両foldで4～6エポック目が選ばれた場合だけgraph推論へ進む。
- exp016の短いlargest-window benchmarkは3エポックのフル実行を大幅に過大予測したため、6エポックの12時間gateにはexp016のフル学習実測時間をエポック数で換算し、当日のbenchmark速度と1.5倍の余裕を掛ける。largest-window外挿値も診断として残す。この変更は学習更新には影響しない。
- Kaggle上で完走したexp019の固定cache replayと公式評価を、新実験のinferenceへ移した。exp016保存済みの公開control指標と全199件の候補graph一致をruntimeで検証し、loss maskはexp016のものへ戻す。Colab経路は現行のKaggle専用方針から除いた。
- 本段階ではcompetition submissionは承認されていない。

## 次

1. 完了: exp024固有の契約テスト4件、Ruff、Notebook round-trip、実験validator、戦略文書検査、Markdownリンク検査。
2. Kaggle GPU quotaを確認し、train packageをpushする。Active Sessions数はKaggle CLIで取得できない。
3. foldごとの選択エポック・接続診断・model SHA・GPU費用を記録し、続行条件に従う。

## 2026-09-20 10:59 JST 学習Notebook push前確認

- `make prepare-kaggle-notebooks EXP=exp024_tracker_six_epochs EXTRA_ARGS='--notebook train --run-on-push'` 成功。生成metadataは`kentookumura/exp024-tracker-six-epochs-train`、private、T4 GPU、TPU/internet無効、exp015固定cacheのNotebook出力を入力とする。slugは50文字以内でtitleと一致する。
- Kaggle CLI認証はOAuthとlegacy keyが利用可能。`kaggle kernels list --mine --search exp024-tracker-six-epochs --format json` は`Not found`で、自分の既存Notebookとの衝突は見つからなかった。未作成slugの`kernels pull`は403を返したため一覧で確認した。
- 2026-09-20 10:59 JSTの`kaggle quota --format json`: GPU使用6.19時間、残38.81時間、Kaggle上限45時間、更新2026-09-26 00:00。リポジトリ方針の週30時間を基準にすると残23.81時間。exp016の3エポック学習Notebook実測4,620.86秒に基づく6エポック単純換算は約2.57時間。今回の学習と後続の推論を別計上しても残量内と判断した。Notebook内の短いbenchmarkで12時間gateを再確認する。
- `make validate-exp`、`make check-exp`、`make test-exp`、JupytextのNotebook同期、`make check-strategy-docs`、Markdownリンク検査に成功。実験の完了・採否、competition submissionは未判断・未承認。

## 2026-09-20 学習実行開始

- `make push-kaggle-train EXP=exp024_tracker_six_epochs`成功。Kaggle Notebook `kentookumura/exp024-tracker-six-epochs-train` version 1: https://www.kaggle.com/code/kentookumura/exp024-tracker-six-epochs-train
- push後に`kaggle kernels pull ... -m`でKaggle側metadataを取得し、GPU有効、TPU無効、`machine_shape: NvidiaTeslaT4`、exp015 outputとcompetition inputを確認した。実行ログとfoldごとの選択checkpointを監視する。

## 2026-09-20 学習中の中間確認

- Kaggle学習version 1で固定cache SHAとT4を確認。64 windowずつのbenchmarkは14.21秒・14.50秒、exp016フル学習実測からの6エポック保守的見積もりは12,000.52秒で12時間gateを通過した。
- fold 0は6エポックを実行し、内部検証の最良checkpointは2エポック目（epoch index 1）。4～6エポック目はいずれも最良値を更新しなかった。両foldで後半checkpointが選ばれるという契約の続行条件は、この時点で不成立。公式graph推論は起動しない。学習Notebook自体はfold 1を含めて完走させ、最終出力と費用を回収する。

## 2026-09-20 13:10 JST 学習完了・出力検証

- Kaggle学習Notebook version 1は`KernelWorkerStatus.COMPLETE`で終了。`make kaggle-output`で`artifacts/kaggle_train_v1/`へ生成物を回収した。
- `training_summary.json`は学習6,794.58秒、Notebook全体7,512.31秒を記録。各fold 6エポックのcheckpoint計12件と最良モデル2件、およびmodel manifestのSHAをローカルで照合し、すべて一致した。model manifest SHA256は`18010a8864f6ab38c3e48658399089825b0fdf22008a391bf64ca29dbd35b276`。
- 選択はfold 0がepoch index 1、fold 1がindex 3。fold 0の選択state SHAはexp016の保存済み3エポック基準と一致。fold 1のindex 3はindex 1と内部検証の選択値が同点で、既存の`>=`ルールにより後者を選んだ。両fold後半選択のgateは`false`で、公式graph inferenceはpushしなかった。
- 学習開始前のKaggle週次GPU使用6.19時間・残38.81時間、終了後の確認では使用9.58時間・残35.42時間。アカウントの使用差3.39時間には同時進行の他実行が含まれ得るため、本Notebookの費用は保存済み実測7,512.31秒を正とする。週30時間のリポジトリ方針で見た現時点の残目安は20.42時間。
- `metrics.json`へKaggle出力の`train_stage`と構造化された生成物証拠を統合。公式scoreとLBは未取得。採否・完了のユーザー判断は未実施。
- 実行結果の反映後、`make validate-exp EXP=exp024_tracker_six_epochs`、`make check-strategy-docs`、Markdownローカルリンク検査、`git diff --check`を通した。コードとNotebookは実行前にRuff、実験テスト4件、Jupytext同期を通過済み。ユーザー判断前のためcommit・pushは行っていない。


## 2026-09-20: 2時点のtracker接続診断

- ユーザーはgraph全体の指標が高コストであるため、tracker精度を安価に比較できる指標を依頼し、checkpoint選択には使わず比較用の診断として先に検証する方針を選んだ。
- 追加診断の契約を`requirements.md`へ記録。exp016選択重みと同一state SHAの2エポック目、および6エポック目を、同じ外側胚の全eligible windowで比べる。1 variant、2 checkpoint/fold、2 fold、0 booster、再学習なし。primary tracker以外、graph、公式評価器は実行しない。
- `src/tracker_pair_metrics.py`と`exp024_tracker_six_epochs_diagnostic.ipynb`を追加。既知親を持つ子だけについて親1位、閾値0.48の接続Jaccardと誤接続・見逃し、既知分裂親の娘2件回収を数える。未知の子を負例としない。checkpoint選択規則と当初のgraph続行条件は変更しない。
- 準備前検証: `make check-exp`、`make test-exp` (4件)、metric単体テスト5件、Jupytext round-trip、`make validate-exp`が成功。Kaggle packageを`make prepare-kaggle-notebooks EXP=exp024_tracker_six_epochs EXTRA_ARGS='--notebook diagnostic --run-on-push'`で生成し、T4 GPU、TPU/internet無効、exp015 cacheとexp024学習出力を入力とするmetadataを確認した。
- 2026-09-20 05:11 UTCの`uv run kaggle quota --format json`: GPU使用10.17h、残34.83h、次回refresh 2026-09-26 00:00 UTC。週30hの内部上限に対して残19.83h。2 checkpointのforward評価は学習・graph処理より軽く、この範囲で開始可能と判断した。実測runtimeを後で記録する。
- Kaggle diagnostic Notebook `kentookumura/exp024-tracker-six-epochs-diagnostic` version 1をpushし、KaggleからpullしたmetadataでT4 GPU・TPU/internet無効・指定した2つのNotebook入力を確認した。https://www.kaggle.com/code/kentookumura/exp024-tracker-six-epochs-diagnostic。実行中で、結果は未回収。
- Kaggle診断version 1は正常終了した。18,707 windowの入力確認と2 checkpoint/foldのforward評価を含むNotebook時間は1,048.48秒。胚別forward評価は6bba 259.18秒、44b6 181.29秒。`make kaggle-output KERNEL=kentookumura/exp024-tracker-six-epochs-diagnostic OUT=experiments/exp024_tracker_six_epochs/artifacts/kaggle_diagnostic_v1`で生成物を取得した。
- `tracker_pair_diagnostic.json`のSHA256は`831332102fba99b3484c337d34b3dcaa4b239753ff67ea40eb434999d284e096`。各動画の件数から胚別件数・比率を再計算し一致を確認。2エポック目と6エポック目の採点母数・動画集合も各foldで一致した。
- 6bbaは既知親対象103,393件、接続Jaccard 0.956859→0.953013、親1位正答率0.979235→0.977764、誤接続1,635→1,742。動画別は改善24、悪化88、同点16。44b6は18,949件、0.931161→0.930493、0.968442→0.968283、誤接続444→459。動画別は改善23、悪化32、同点16。両胚とも6エポック目のprimary tracker改善は確認できなかった。
- 数値を`metrics.json`の`tracker_pair_diagnostic`へ記録し、解釈を`result.md`へ追記した。公式graph scoreは引き続き未計測で、当初のgraph続行条件は不成立のまま。実験の完了・採否はユーザー判断待ちとする。
- 診断後のKaggle quota表示はGPU使用10.84h、残34.16h。開始前との差0.67hはアカウント全体の変化であり、他sessionの稼働も含み得るため、このNotebook単体のGPU割当時間とは断定しない。Notebook経過時間1,048.48秒とは分けて記録する。

## 2026-09-20: ユーザーの不採用・完了判断

- ユーザーは「不採用として完了としてください。commitとpushしてください」と明示した。6エポック条件に限って不採用として実験を閉じる。`metrics.json`の実験statusを`discarded`にし、`result.md`に判断理由と公式score未計測の範囲を記録した。
- この実験に関係する変更だけをcommitし、現在の`main`ブランチをpushする。共有作業ツリーの他実験・backlog・調査の変更は含めない。
