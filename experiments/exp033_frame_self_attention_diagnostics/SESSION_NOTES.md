# exp033_frame_self_attention_diagnostics セッションノート

## 2026-09-21 実験化と実装

- ユーザーが `frame_self_attention_diagnostics` の実装を依頼。候補は設計不可で、PR比較precisionと混雑度の定義が未決だった。
- ユーザーはprecision 0.95、同点score一括処理、未達の記録、子候補最近傍距離と学習側分布で固定するbucketを選択した。4構成×2foldの保存checkpointを再推論し、tracker学習0、booster 0、control再学習なし。
- `exp033_frame_self_attention_diagnostics` を `make new-exp` で作成。既存の `exp030_frame_self_attention_diagnostics` は互換性・GPU費用の別実験なので変更しない。
- 親の `frozen_tracker.py` を同じ内容でコピーし、cacheとGEFF教師の処理を保持した。pair PR・bucket関数、manifest/source/checkpoint SHA照合、sample単位の出力、0.5指標の再現gateを実装した。
- 共有作業ツリーには別作業の未コミット変更が多い。exp033と対象backlogの移行以外に触れない。

## 次

1. Jupytext変換と `make validate-exp`、`make check-exp`、`make test-exp` を実行する。
2. Notebook packageにreference metricsと4構成のsourceが入り、Kaggle mountの5つのkernel outputを参照できることを静的に確認する。
3. Kaggle実行前にGPU残量を確認し、小規模実測の12時間gateを通してから全件へ進む。公式scoreは未計測として扱う。

## 2026-09-21 ローカル検証

- `make validate-exp EXP=exp033_frame_self_attention_diagnostics`、`make check-exp EXP=exp033_frame_self_attention_diagnostics`、`make test-exp EXP=exp033_frame_self_attention_diagnostics`に成功。pair順位とshard集計の7 testがpass。
- `make check-strategy-docs`が成功。元候補を削除し、上位仮説の対応実験をexp033へ更新。空間候補の依存リンクも移行。
- JupytextのNotebook→percent round-trip testが成功。strict testはJupytextが追加するcell metadata `trusted` の差で失敗し、コード・Markdownの差ではない。
- `make prepare-kaggle-notebooks EXP=exp033_frame_self_attention_diagnostics EXTRA_ARGS="--notebook diagnostic"`に成功。private T4、TPU・internet無効、`run_on_push: false`。exp015 cache、exp016 control、exp025 A/B/identityの5 kernel sourceを設定した。
- `validate_kaggle_metadata.py`はpackageをpush-readyと判定。reference metrics JSON、config、教師source、診断sourceのbootstrap同梱を確認した。KaggleへのpushとNotebook実行はまだ行っていない。

## 2026-09-21 Kaggle execution preflight

- User explicitly requested execution of the diagnostic Notebook; no submission was requested.
- Kaggle quota at 2026-09-21 00:05:02 UTC: GPU used 23.30h, remaining 21.70h of 45.00h, refresh 2026-09-26T00:00:00; TPU used 0h. The repository's 30h weekly GPU ceiling leaves 6.70h. The run configuration uses a conservative 6.50h projection gate alongside the 12h Notebook limit.
- Revalidated exp033: strict experiment validation, Ruff, and 7 focused tests passed. Regenerated the package with run_on_push=true. Inspected private T4 metadata; TPU and internet are disabled, with five expected kernel sources and the support dataset.
- Decision: push one diagnostic run; eight saved checkpoints, no training or submission. The Notebook measures large windows before full inference and stops if the projection exceeds 6.50h.

## 2026-09-21 Kaggle version 1 failure and version 2 preflight

- Kernel v1 (id_no 135162569) was pushed and its pulled metadata confirmed a private T4 GPU, no TPU or internet, and the five expected kernel sources. It reached 18,707 filtered windows, 199 GEFF groups, and the fixed cache identity SHA before failing in the first benchmark with KeyError: candidate_ids_src. Kaggle status was ERROR.
- Root cause: frozen_tracker.validate_window_cache returns only REQUIRED_CACHE_ARRAYS, which excludes candidate IDs. diagnostic_runner now reads the two ID arrays directly from the same NPZ after cache validation and checks their lengths. A focused regression test was added; strict validation, Ruff, and all 8 focused tests pass.
- Fresh quota at 2026-09-21 00:23:15 UTC: GPU used 23.50/45.00h, remaining 21.50h; the repository's 30h weekly ceiling leaves 6.50h. Version 2 sets a 6.30h projection gate, leaving 0.20h of margin. The private T4 run-on-push package was regenerated for the same kernel slug.

## 2026-09-21 Kaggle version 2 完走と結果回収

- 同じprivate kernel `kentookumura/exp033-frame-self-attention-diagnostics-diagnostic` へversion 2をpushし、status `COMPLETE`を確認。pullしたmetadataの`id_no=135162569`、T4 GPU有効、TPU・internet無効、5つのkernel sourceを確認した。
- 固定cacheは空GT除外後18,707 window、199 GEFFサンプル。cache identity SHA `440891c4550adf8540e3c47f9784e68b6ca1b64bbe56dbb93a25eb87c46bc1ee`。4構成×2foldのmanifest/source/checkpoint/state SHAを通過した。
- 大きいwindow 8件/foldの4モデル推論はfold 0が6.32秒・peak GPU 106,755,584 bytes、fold 1が2.22秒・peak GPU 110,694,400 bytes。安全係数込み全体予測20,018.98秒で6.30時間gate内。実Notebookは3,330.12秒で完走した。
- 内部検証7/12サンプル、外側評価128/71サンプルのpair shardを保存。確率0.5の参照指標は2胚×4構成の8通りで件数と値が一致した。precision 0.95固定の主PRも両胚で得た。対照比はModel Aが両胚で低下、Model Bも両胚で低下、恒等初期化Bは6bbaで低下し44b6で0.35 percentage point上昇。数値と条件別結果は`metrics.json`、解釈は`result.md`を正とする。
- `make kaggle-output`でversion 2 outputを`experiments/exp033_frame_self_attention_diagnostics/artifacts/kaggle_diagnostic_v2/`へ回収。218件のpair shard（977,678,779 bytes）、32本のPR曲線をすべてSHA照合。manifest SHA `2e34f8fd0b7408bcd0ee1a14cbdc335ebe56327f0b600a02bf64038f5226073d`、summary SHA `c5c93911407cbbe75a4570a02736b7a7a373e42b16a659d3d60ff1fef4fd9ebd`。
- 2026-09-21 01:37 UTCのquotaはGPU使用24.43/45.00時間、アカウント残20.57時間、リポジトリの週30時間上限まで残5.57時間。TPU使用0。公式graph scoreとsubmissionは未計測。
- 全graph推論は進めず保留。Model A/Bのprecision 0.95時recallは両胚で現行を下回り、恒等初期化Bも一貫しないため。部分GEFF注釈の限界と単体診断でgraph効果を測れない範囲を`result.md`に記録した。実験完了と採否はユーザー判断待ち。

## 2026-09-21 ユーザー完了判断

- ユーザーがexp033を完了と判断したため、`metrics.json`の実験statusを`completed`へ更新した。
- Self-Attention構成の採否、公式graph評価、後続学習は未判断のまま。exp033の結果を受け、`frame_self_attention_spatial`は方式選択の根拠が得られるまでP4の再開条件付き候補とした。
- exp033と対応するバックログ移行だけをcommitし、共有作業ツリーにある他作業の変更は含めない。
