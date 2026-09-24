# exp042_public_x138_replay セッションノート

## 目的

公開 biohub x138 V1 を追加座標補正checkpointを含めて固定し、公開test全件の2回再実行一致と費用を検証する。

## 2026-09-23 実装

- `public_x138_replay` は「設計可能・未決事項なし」だった。ユーザーの「実装してください」を実験化承認として `exp042` を作成し、候補契約・根拠・判断履歴を `requirements.md` へ移した。
- 公開Notebook SHA256 `6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d` を `assets/reference_notebook/` に保存。12個のcode cellを元の順序でJupytext sourceに生成し、V1284補正前後を記録するcellだけ推論直前へ挿入した。
- 追加重みは `biohub-v1284-head-s075/v1284_head.pt`。公開metadataのdataset_sourcesに空欄があり、Kaggle CLIで `v1284`、`biohub-v1284-head-s075` を検索してもdatasetなし。作者の公開dataset一覧にも該当なし。`pilkwang/biohub-v1284-head-s075` と `anvithpothula/biohub-v1284-head-s075` のfiles APIは403。正確な取得先・version・SHAは未確認。
- 追加確認: Kaggle公開実行の `GetKernelVersion` 生metadataに4番目のdataset attachmentとして `mountSlug=datasets/anvithpothula/biohub-v1284-head-s075`、`datasetId=12103746`、`sourceId=19822532` がある。後者はDataset Version IDで、version番号と混同しない。[取得したmetadata](../../studies/biohub_public_notebooks_20260923/x138_v1284_attachment.json)を保存した。正確なrefとversion IDを `config.yaml`へ反映した。
- `kaggle datasets files` と `kaggle datasets download` は同refに403を返し、作者の公開Notebook output一覧にも `v1284_head.pt` はない。公開Notebookの入力参照から重みの存在は分かるが、こちらのアカウントではファイル本体とSHAを取得できない。以前のSHA一致は既存3重みだけで、V1284は対象外。
- 3既存artifact、追加重み、T4 2基、動的test一覧、offline wheelsのguardを追加。追加重みの設定が空なら元sourceを実行する前に停止する。
- 座標補正前後のfloat32座標をframe単位でSHA記録し、元sourceの検出座標manifest、graph topology、決定的なrun統計、raw提出SHAとともに `replay_receipt.json` へ集約。`compare_replays.py` はinputとreceiptのSHAを再計算して2実行を照合する。
- active variant 1、config 1、fold 0、booster 0。control再学習なし。
- 実行済み: `make validate-exp EXP=exp042_public_x138_replay`、`make check-exp EXP=exp042_public_x138_replay`、`make test-exp EXP=exp042_public_x138_replay`。いずれも最終再実行で成功した。
- Kaggleへのpush・公開test実行・submissionは行っていない。追加重み未取得の停止条件が成立している。GPU quotaと12時間条件の確認はpush直前に行う。

## 次のアクション

1. 追加checkpointへのアクセス可能な共有先を確保し、ファイル本体のSHA256を計測して `config.yaml`の `data.v1284_head.sha256` を固定する。
2. `build_notebook.py` からJupytext sourceとNotebookを再生成し、同じ3検証を通す。
3. `kaggle-platform` のquota確認後、cleanな2回の公開test推論を同じT4環境で実行し、出力を比較する。
4. `kaggle-submit-check` で提出前検証を行う。実際のsubmissionはユーザーの別途承認後だけ行う。
