# exp032_three_frame_ten_epoch_training

## 概要

固定画像特徴と候補を使う3時点trackerをColabで10 epoch学習し、保存済みexp016の2時点モデルと同じ128窓で比較する。新規学習は3時点の2foldだけ。

- 差分: exp027の構造・教師・splitを固定し、1 epochから10 epochへ延長する。
- 実行入口: Colab CLI。ローカルの colab_cli_run.py がGPU sessionを作り、ZIPを分割転送し、Kaggleのexp015 cacheを期限付きURLでColabへ直接取得する。公開supportと保存済みexp016・exp027のmodelはZIPに入る。GEFFとexp027の128件の保存済み予測はKaggleからColabへ直接取得する。
- 保存: 各epochの指標と最新の再開状態をCLIで回収する。古い再開状態は次の状態のSHA確認後に削除し、epoch 1と内部選択されたweightを残す。診断はこの2点を固定128窓で比較し、予測配列をローカル回収対象から除く。trainとdiagnosticの小さな完了archiveをSHA照合する。
- 実行状態: 2 fold × 10 epochの学習と固定128窓の診断を完了。採用・不採用はユーザー判断待ち。
- 限界: 部分窓で分裂事象が少なく、CV・公式graph評価・LBは実施していない。

## 実行入口

- colab_cli_run.py: ローカルからColab CLIを操作する。stage=train、続いてstage=diagnosticを指定する。bundle、期限付きURL、回収物、再開seedは異なる転送として明示承認を必要とする。
- colab_cli_manifest.py: ローカルKaggle認証からexp015 cacheの短命なURLを生成する。資格情報はColabへ送らない。
- geff_export/: exp015 cacheに含まれない正解データを渡すため、Kaggleのcompetition mountから199件のtrain GEFFだけを小さなZIPとして出力する非公開CPU kernel。URLだけをColabへ渡し、ZIP本体はローカルへ取得しない。ZIP作成にGPUは不要。
- colab_cli_worker.py: Colab内でZIPの検証、入力取得、学習・診断、完了archiveを処理する。
- build_colab_bundle.py: コード、公開support、保存済みmodelとmanifestを1個のZIPにまとめる。生成物は artifacts/colab_bundle/exp032_three_frame_ten_epoch_training_colab_bundle.zip。
- exp032_three_frame_ten_epoch_training_colab_train.py と exp032_three_frame_ten_epoch_training_colab_diagnostic.py: CLI workerが実行する学習・診断コード。対応するipynbは手動実行経路として残す。

## 正の記録

- 実装契約: [requirements.md](requirements.md)
- 設定と系譜: [config.yaml](config.yaml)
- 数値と証拠: [metrics.json](metrics.json)
- 実行時系列: [SESSION_NOTES.md](SESSION_NOTES.md)
- 結果の解釈: [result.md](result.md)

CLIで回収した生成物は artifacts/colab_runs/ten_epoch_v1_cli/ に置く。実験の採否と公式graph評価はユーザーが判断する。

## CLI再実行手順

初回の学習・診断は実行済みで、追加の手動操作は不要。再実行する場合、転送承認を確認してWSLのリポジトリ直下で以下を順に実行する。CLI実行ではGoogle Driveへの配置やノートブックの手動実行は不要。約4.15 GBのcacheはexp015 Kaggle kernel outputから、GEFF ZIPと約136 MBの保存済み予測もKaggleからColabへ直接ダウンロードする。ローカルへダウンロードする大容量入力はない。

```bash
.venv/bin/kaggle kernels push -p experiments/exp032_three_frame_ten_epoch_training/geff_export
.venv/bin/kaggle kernels status kentookumura/exp032-train-geff-export
.venv/bin/python experiments/exp032_three_frame_ten_epoch_training/build_colab_bundle.py
.venv/bin/python experiments/exp032_three_frame_ten_epoch_training/colab_cli_run.py --stage train --approve-bundle-transfer --approve-signed-manifest-transfer --approve-geff-manifest-transfer --approve-result-transfer
.venv/bin/python experiments/exp032_three_frame_ten_epoch_training/colab_cli_run.py --stage diagnostic --approve-bundle-transfer --approve-signed-manifest-transfer --approve-geff-manifest-transfer --approve-prediction-manifest-transfer --approve-result-transfer --approve-resume-transfer
```

GEFF export kernelが完了したことをstatusで確認してからtrainを開始する。GPU session切断後にtrainを再実行する場合は `--approve-resume-transfer` も付ける。各epochのreceipt、最新の再開state、epoch 1と内部選択weightは `artifacts/colab_runs/ten_epoch_v1_cli/` に保存する。今回の結果は `metrics.json`、`SESSION_NOTES.md`、`result.md` に記録済み。config.yamlのDrive項目は予備の手動経路用で、実行済みrunのconfig SHAを保持するため変更していない。
