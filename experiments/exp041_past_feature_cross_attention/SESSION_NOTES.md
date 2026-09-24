# exp041_past_feature_cross_attention セッションノート

## 目的

承認された過去3時点の固定特徴cross-attentionを実験化し、Kaggle学習と単体診断を実行できる状態へ整える。

## 現在の作業

- 2026-09-23: ユーザーの「実装に進んでください」を受け、exp041を作成。exp016を比較親、exp037をコード構成の参照元として、候補契約をrequirementsへ移した。
- 2026-09-23: 過去3窓のsource特徴・時刻・物理座標を整列し、cache SHAと重複frameの候補ID・座標を検査する処理を実装。共有1層attentionを対象2時点に適用し、過去不使用選択肢、ゼロ初期化残差、分割softmaxを追加。
- 2026-09-23: 学習Notebookを入力確認、モデル・学習、runtime gate、外側比較、補助診断、生成物保存のセルへ展開。全graph推論とsubmissionは含めない。
- 検証: `make validate-exp EXP=exp041_past_feature_cross_attention`、`make check-exp EXP=exp041_past_feature_cross_attention`、`make test-exp EXP=exp041_past_feature_cross_attention`（6件通過）、Jupytext変換テストを通過。`make check-strategy-docs`も通過。
- `make prepare-kaggle-notebooks EXP=exp041_past_feature_cross_attention EXTRA_ARGS='--notebook train --run-on-push'`でローカルpackageを作成し、metadataのpush-ready検証を通過。Kaggleへのpush・実行はまだ行っていない。
- 2026-09-23: `uv run kaggle quota --format json`でGPU使用33.10時間、残11.90時間、Kaggle枠45時間、refreshは2026-09-26 00:00 UTCと確認。生成metadataはT4 GPU、TPUなし、12時間上限。承認済みの週30 GPU時間を既に超えているためpush・Kaggle実行を保留した。既存セッションの停止は行っていない。
- 2026-09-23: ユーザーから「quotaの利用枠は現在45時間なので問題ありません」と訂正を受け、旧30時間上限による停止を撤回。最新quotaはGPU使用33.10時間、残11.90時間、45時間枠。モデル・教師・学習量を保ったまま、残量に対する安全余裕を取って本回のNotebook内runtime gateを11時間に設定した。
- 2026-09-23 07:32 UTC: push直前のKaggle quotaはGPU使用33.10時間・残11.90時間・総45.00時間、refreshは2026-09-26 00:00 UTC。生成metadataはT4 GPU、TPUなし、internetなし。Notebook内の保守的な学習・評価見積もり上限を11.0時間とし、準備・benchmarkの余裕を含めて残量内で開始すると判断した。承認済み構成1件・各fold3 epoch・対照再学習なしを確認。
- 2026-09-23: `make push-kaggle-train EXP=exp041_past_feature_cross_attention`でkernel version 1をpush。Kaggle statusは`RUNNING`。提出は行っていない。
- 2026-09-23 07:52 UTC: Kaggleからpullしたmetadataで`machine_shape=NvidiaTeslaT4`を確認。Kernel statusは引き続き`RUNNING`。bootstrap・依存読み込み以降のbenchmark／精度ログは未出力。ローカルのlogs followerだけを終了し、Kaggle kernelは継続中。
- 2026-09-23 08:02 UTC: version 1が`ERROR`で終了。約1614秒時点、benchmarkの最初のDataLoader窓で`history cache content mismatch`。過去readerは12個の必須配列だけを読みSHAを再計算していたが、exp015の記録SHAはsecondary特徴を含む全配列が対象だった。入力cache破損の証拠ではない。`artifacts/kaggle_v1/`へ失敗log、GT窓監査、split manifestを回収。GT条件を満たす対象窓は19,701中18,707。モデル学習とpair評価には到達していない。
- 2026-09-23: `_load_history_cache`を全配列のschema・SHA照合へ修正し、secondary配列を含む合成cacheのテストを通した。producerの`array_content_sha256`とreaderの実装も比較した。残GPU枠11.17時間に合わせ、version 2のruntime gateを10.0時間に設定。入力候補・教師・fold・epochは変更しない。
- 2026-09-23 08:05 UTC: version 2 push直前のquotaはGPU使用33.87時間・残11.13時間・総45時間、refreshは2026-09-26 00:00 UTC。metadataはT4 GPU、TPUなし、internetなし。前処理・benchmarkの費用を除く保守的な学習評価見積もりの停止閾値10.0時間を残量内と判断した。
- 2026-09-23: `make push-kaggle-train EXP=exp041_past_feature_cross_attention`でversion 2をpush。submissionは行っていない。
- 2026-09-23 08:45 UTC: version 2は`RUNNING`で、version 1の停止時刻を超過。実行中ログと部分成果物はまだ公開されておらず、ベンチマーク通過は未確認。単一window入力とバッチ学習の回帰テストを追加し、ローカル`make test-exp` 8件、`make check-exp`、`make validate-exp`を通過。点検時に重複して表示した同じソース行をコード重複と誤認して修正済みと報告したが、実際にモデルの変更はなく、version 1・2・ローカルの`past_feature_attention.py`のSHAは同一だった。
- 2026-09-23 09:00 UTC: Kaggle quotaはGPU使用35.68時間・残9.32時間。08:05から約55分で使用量が1.81時間増え、`NvidiaTeslaT4`はKaggle CLI公式文書上T4×2の構成。version 2の`runtime_gate_hours=10.0`はGPU枠消費を壁時計時間から換算していないため、安全な残予算判定には使えない。実行中版のbenchmark結果が未公開で、停止手段を確認中。
- 2026-09-23 09:09 UTC: `kaggle kernels logs --follow`のSSEでversion 2のfold 0 epoch 0完了を確認。epoch所要1240.885秒、学習窓5622、内部検証窓693。最初のcache SHA失敗箇所とruntime gateを通過して2fold学習中。quotaはGPU使用36.00時間・残9.00時間。実行中の部分成果物はAPIから未取得で、最終CV・graph条件は未確定。
- 2026-09-23 12:41 UTC: version 2のstatus `COMPLETE`、Kaggle quotaはGPU使用40.52時間・残4.48時間。`make kaggle-output EXP=exp041_past_feature_cross_attention NOTEBOOK=train KERNEL=kentookumura/exp041-past-feature-cross-attention-train/2 OUT=experiments/exp041_past_feature_cross_attention/artifacts/kaggle_v2`でモデル2件・単体診断・実行ログを回収。モデル・manifest・GT窓監査SHA、教師一致、有限値を確認した。Notebook実行16,245.96秒、進行条件0/2で全graphと公式scoreは未実施。
- 2026-09-23 12:49 UTC: 保存済みfoldモデル2件を`torch.load(weights_only=True)`で読み、実験configから再構築したmodelへ`strict=True`で再ロードできることを確認。外側評価の対照差は6bbaで既知接続+5本・教師負例+2.39%・分裂親-2件、44b6で既知接続-96本・教師負例-6.64%・分裂親-3件。両胚で事前の進行条件未達のため、全graphを保留。`result.md`、`metrics.json`と横断summaryを更新し、`make check-exp`、実験test8件、strict validate-expを通過した。実験の完了・採否はユーザー判断待ち。

## 変更点

config・モデル・cache整列・単体診断の詳細は[`requirements.md`](requirements.md)と[`config.yaml`](config.yaml)を正とする。親にcompact self-contained学習Notebookはなく、exp037の正規Notebookはhelperのmainを呼ぶ構成だった。exp041は学習・比較・保存の上位処理をNotebookの見出し付きセルへ展開した。

## 次のアクション

1. 完了した両胚別の単体結果をユーザーへ示し、実験を今回で完了とするか判断を仰ぐ。
2. 今回の進行条件未達に従い、全graph評価とKaggle submissionは開始しない。

### 2026-09-23 精度が上がらなかった理由の考察

- ユーザーの考察依頼を受け、`kaggle-review` と実験記録規約に沿って、exp041/exp016の保存済み指標とモデル・評価実装を確認した。追加学習・Kaggle実行・submissionは行っていない。
- `studies/exp041_past_feature_analysis_20260923/analyze_metrics.py` で同一教師・分母と親実験との窓数を照合し、順位と確率0.5での採否、密集度別の差、内部検証の整数件数を再集計した。派生値は同ディレクトリの `readout.json` に保存した。
- 44b6の親順位正解は18,351→18,360、順位正解でも確率0.5以下は392→497。回収減96本は順位正解+9本と閾値未達+105本に分解できる。接続別の相殺・閾値校正後の改善は未測定。
- fold 1の内部検証で3 epoch目は1 epoch目より接続回収+23、分裂回収+2、教師負例予測+30だった。誤予測総数は402→409となり、現行selection scoreは1 epoch目を選んだ。単純な学習不足や、3 epoch目を選べば改善するという主張はできない。
- [原因考察レポート](../../docs/surveys/biohub-exp041-past-feature-analysis_20260923.md)に観測と仮説、今後の診断を保存し、resultからリンクした。実験の完了・採否判断は変更せず、statusを維持した。

### 2026-09-23 手法の機構に焦点を置いた考察への修正

- ユーザーから、求めているのは結果の評価ではなく適用手法がその結果を生む理由だとの指摘を受けた。分析レポートの冒頭を、過去候補の独立採点、時点をまとめる加重和、接続pairに依存しない事前集約、全候補と単一null候補の競争、間接的な教師、ゼロ初期化時の勾配、分裂の損失の仕組みから説明する内容へ修正した。
- これらは実装から確認できる性質と、そこから導く原因仮説に分けた。特徴混合・得点差の縮小・過去対応の未獲得を実測済みの原因として扱わず、追加学習や設定変更は行っていない。

### 2026-09-23 past_candidate_attentionとの比較の限界を追記

- ユーザーから、説明がpast_candidate_attentionなら解決できるように聞こえるとの指摘を受けた。exp037の契約・実装・実行記録を確認した。
- 接続候補ごとの集約と3点の変位特徴はexp037が明示的に扱う一方、全候補softmax、過去対応への直接教師なし、既存損失・本体同時更新などは共通する。座標だけでは誤接続にも滑らかな移動を支持する別細胞の過去候補があり得ることを、未実測の構造上の曖昧さとして追記した。
- 設計上の相違を確認したことと、exp041の失敗原因を特定したこと、exp037の改善を証明したことを区別した。exp037は本学習前の費用停止で精度未測定であり、この考察から採用・再実行は判断していない。
