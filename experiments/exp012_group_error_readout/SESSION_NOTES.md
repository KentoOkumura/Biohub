# exp012_group_error_readout セッションノート

## 目的

exp005・exp006 の固定した胚 holdout 予測を、胚・画像輝度・候補密度・候補の画像境界距離・既知分裂で再集計し、改善と悪化の条件を分ける。

## 現在の作業

- 作業内容: backlog 契約を exp012 へ移行し、CPU diagnostic Notebook と実験固有 test を実装する。
- ブロック要因: Kaggle 実行前の実装にはなし。正式な条件別結果には kernel source 上の candidate cache が必要。
- 次: 静的検証後、Kaggle CPU diagnostic の push 前に resource / quota を確認する。

## GPU 学習コスト確認

- active variant 数: 固定予測 2 route。
- model / config 数: 0。
- fold 学習数: 0。
- booster 数: 0。
- control 再学習: なし。
- 実行資源: CPU のみ。GPU は使わない。

## コマンドログ

- 2026-09-12: `task new-exp EXP=exp012_group_error_readout` を試行したが、この環境に `task` がなく失敗した。
- 2026-09-12: 規約の fallback として `make new-exp EXP=exp012_group_error_readout` を実行し、template から exp012 を作成した。
- 2026-09-13: backlog 契約、exp011 の採用 manifest、exp003 の公式集計、exp005・exp006 の保存済み入力 schema と SHA を確認した。
- 2026-09-13: 学習・推論用 placeholder Notebook を除き、diagnostic の Jupytext source、設定、契約、test を実装した。
- 2026-09-13: Jupytext round-trip、`make validate-exp`、`make check-exp`、`make test-exp` を実行し、schema validation、lint、format、8 test が合格した。保存済み exp005 の動画別成分から公式全体集計を再現する test と、exp005・exp006 の固定 SHA・対象集合を照合する test も合格した。
- 2026-09-13: `make prepare-kaggle-notebooks EXP=exp012_group_error_readout EXTRA_ARGS="--notebook diagnostic --run-on-push"` が strict mode で成功した。生成 metadata は CPU、internet 無効、固定 kernel source 3 件、private Notebook である。
- 2026-09-13: `kaggle-strategy` の移行手順に従い、上位仮説 ID と候補名の移行後に未着手行と候補詳細を削除し、`HYP-20260910-14` と依存候補を exp012 参照へ更新した。`make check-strategy-docs` は合格した。
- 2026-09-13T08:06:42+09:00: push 前に再度 `make validate-exp`、`make check-exp`、`make test-exp`、`make prepare-kaggle-notebooks` を実行し、strict validation、lint、format、8 test、package 再生成がすべて成功した。
- 2026-09-13T08:06:42+09:00: 生成済み metadata で `enable_gpu=false`、`enable_tpu=false`、`enable_internet=false`、`run_on_push=true` を確認した。実行資源は CPU、GPU 残時間は対象外、private Notebook の push を許可できると判断した。

### 予定

```bash
make validate-exp EXP=exp012_group_error_readout
make check-exp EXP=exp012_group_error_readout
make test-exp EXP=exp012_group_error_readout
make prepare-kaggle-notebooks EXP=exp012_group_error_readout EXTRA_ARGS="--notebook diagnostic --run-on-push"
```

Kaggle push は `kaggle-platform` の runtime resource / quota 確認後に行う。competition submission は予定に含めない。

## 変更点

- exp005 を条件境界の固定基準、exp006 を対応比較 route とした。
- 輝度は同一動画の保存済み画像 sampling、候補密度と境界距離は各 route の candidate cache から計算する。
- 分位点境界は各評価胚に対する反対側胚の exp005 条件から決める。
- 公式評価は動画別 TP/FP/FN を条件 group 内で micro 集計し、adjusted edge Jaccard は exp003 と同じ edge 数重みを使う。
- 失敗、NaN、有効件数、route 別・対応動画別の改善と悪化、入力・出力 SHA を保存する。

## 次のアクション

1. Kaggle CPU diagnostic の実行前に quota と Active Sessions を確認する。
2. 実行結果を `metrics.json` と `result.md` へ記録し、ユーザーへ採否・完了判断を求める。
