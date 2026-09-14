# exp017_cross_crop_registration_audit セッションノート

## 目的

同一胚内のcropに共通視野があるかをCPU画像登録で調べ、領域をまたぐ軌跡を復元する前提が実データにあるか確認する。

## 現在の作業

- 作業内容: Kaggle CPU version 2の実行・生成物回収・結果記録・ユーザー判断まで完了した。
- ブロック要因: なし。
- 次: exp017に関係する変更だけをcommitし、現在のbranchをpushする。

## コマンドログ

### 2026-09-14 実装とローカル検証

- `make new-exp EXP=exp017_cross_crop_registration_audit`: 実験雛形を作成した。`task`コマンドが環境にないため、AGENTS.mdに従ってMakefile targetを使用した。
- `make validate-exp EXP=exp017_cross_crop_registration_audit`: strict validation通過。
- `make check-exp EXP=exp017_cross_crop_registration_audit`: Ruff checkとformat check通過。
- `make test-exp EXP=exp017_cross_crop_registration_audit`: 合成画像の部分重複shift、位相相関候補、D4座標、3D z shift、負例、停止schedule選択の8 testが通過。
- JupytextのNotebook生成とround-trip testを通過した。ローカルにはcompetition画像がないため、実データのsmoke実行は行わなかった。

### 2026-09-14 Kaggle CPU version 1

- `make prepare-kaggle-notebooks EXP=exp017_cross_crop_registration_audit EXTRA_ARGS="--notebook audit --run-on-push"`。
- metadataでprivate、competition source、GPU false、TPU false、internet false、auditのみを確認した。
- `make push-kaggle-notebook EXP=exp017_cross_crop_registration_audit NOTEBOOK=audit`: kernel version 1を実行した。
- live logで完走を確認し、`make kaggle-output ... OUT=experiments/exp017_cross_crop_registration_audit/artifacts/kaggle_v1`で生成物を回収した。
- version 1は全10,613対をscreeningしたが、停止frameの完全一致scheduleを持つ24対すべてをD4・3Dへ強制投入していなかった。ユーザーが指定した確認範囲を直接満たすため修正した。

### 2026-09-14 Kaggle CPU version 2

- 停止frameの完全一致scheduleを持つ24対をD4と3Dへ必ず投入し、score上位対と合わせてD4 150対、3D・GEFF 48対を確認するよう変更した。
- local 8 test、strict validation、Ruff、Jupytext round-tripを再度通過した。
- 同じmetadataでkernel version 2をpushし、live logで正常完走を確認した。監査処理時間は1,303.05秒。
- `make kaggle-output ... OUT=experiments/exp017_cross_crop_registration_audit/artifacts/kaggle_v2`で生成物を回収した。
- repo-local `analyze_audit_outputs.py` で反復schedule、2D・3D相関、GEFF一致を再集計し、`artifacts/kaggle_v2/artifacts/audit_v1/postrun_readout.json`へ保存した。
- version 2を正とし、version 1は予備実行として採用しない。competition submissionは実行していない。

## 変更点

- 2D最大値投影のFFT位相相関で同一胚内の全crop対を低解像度screeningする。
- 上位候補と停止frameの完全一致scheduleを持つ全対を、重なり領域の正規化相互相関と8種類の回転・反転で再評価する。
- 別時刻と異なる胚を負例対照にし、複数時刻で移動量が安定しているかを測る。
- 上位候補を低解像度3D画像とGEFF注釈の座標・node ID・edgeで確認する。

設定と再現性方針は `config.yaml`、kernel情報、Kaggle Notebook実行時間、生成物SHA、実験statusは `metrics.json` を正とする。

## 次のアクション

1. 2026-09-14、ユーザーが実験の完了と、crop間の軌跡を直接接続せずsample単位のtracker学習を維持する判断を承認した。
2. `metrics.json` のstatusと `result.md` の判断欄へ承認内容を反映した。
3. exp017に関係する変更だけをcommitして現在branchをpushする。
