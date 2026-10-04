# exp031_frame_self_attention_spatial セッションノート

## 現在の作業

学習・隣接2フレーム評価・結果回収は終了した。[結果とユーザー判断](result.md)に従い、全graph推論を保留し、実験の採否・完了と追加評価の判断を待つ。実装した範囲と未検証事項も同記録を参照する。

## 2026-09-20 実装時点の目的と計画

- 2026-09-20: ユーザーの `frame_self_attention_spatialを実装してください` を、設計済み `frame_self_attention` の実験化承認として受けた。
- exp016を親に新規実験を作り、実験内model sourceとtrain/inference Notebook sourceを実装した。
- train予定は現行、Selfのみ、Self後Crossの3 variant、各1 config・2fold・3 epoch、合計6モデル、booster 0。現行対照の再学習を含む。
- 3構成で共通moduleの初期stateとcache・教師・optimizer・内部選択を揃える。新Encoderは同一seedから構成ごとに初期化する。
- モデルのPyTorch動的テストはローカルにPyTorchがないためskip。静的契約テストと固定pipelineのpatch構文検証は通過した。
- Kaggle GPUのtrain pushは未実施。公開対照の再学習は `kaggle-review-exp` のGPU費用ガードにより明示承認が必要。push直前に `kaggle-platform` のquota確認も必要。
- 実験status、採否、完了はユーザー判断前に確定しない。submissionは行わない。

## 実行ログ

- `make new-exp EXP=exp031_frame_self_attention_spatial SOURCE=experiments/exp016_frozen_image_encoder`: 作成。
- `make check-exp EXP=exp031_frame_self_attention_spatial`: 通過。
- `make validate-exp EXP=exp031_frame_self_attention_spatial`: 通過。
- `make test-exp EXP=exp031_frame_self_attention_spatial`: 2 passed、PyTorchがないため1 skipped。
- `jupytext --to ipynb` と `--test` をtrain/inferenceに実行。再変換後に最終確認する。
- `graph_inference.patch_exp015_source` を実sourceへ適用してsetup/replayの構文を確認。構成選択はreplay側に置いた。

## 2026-09-20 学習開始前のgate

1. Kaggle実行quotaを確認し、control再学習の明示承認を得る。Active Sessions数はCLIで取得できず、push前gateには用いない。
2. 学習Notebook内で代表的なcell数と最大のcell数を含む64 windowをvariant/foldごとに測定する。保守的な総学習時間が12時間を超えたらフル学習を停止する。
3. PyTorch動的テストで旧stateのstrict読み込みとlogits一致、A/Bのmask・順序・空集合・勾配・復元を通す。
4. 予算gateを通れば6モデルを学習し、同一decodeで3構成の199動画を両胚別に公式評価する。推論も12時間gateを設ける。
5. 実行のkernel version、Notebook実行時間、model/graph SHA、両胚の指標、失敗件数を `metrics.json` と `result.md` に記録してユーザー判断を求める。

## 2026-09-20 実装検証の追記

- 親exp016の保存記録では2fold学習のNotebook実行時間が4,620.86秒だった。ただし本実験は6モデルでSelf-Attentionの費用が未測定のため、この値から所要時間を保証しない。
- 親の最大級windowだけを使った学習時間外挿は実測より大きかった。本実験のbenchmarkはcell数に応じたAttention負荷の分布から64 windowを選び、最大と中央値を含める。trainとevalの時間・peak GPU memoryを各構成とfoldで記録する。
- 固定入力の照合用に、exp015 cache identityと公開候補graph SHAが構成間で一致することを推論Notebookへ追加した。
- train/inferenceのKaggle packageをローカルで生成し、GPU T4・internet無効・train inputからinferenceへの参照をmetadataで確認した。pushは未実施。

## 2026-09-20 Kaggle train push 前の確認

- ユーザーの「実行してください。」を、同一runの現行対照再学習を含む6モデル学習と、条件を満たした場合の公式graph評価の実行承認として受けた。competition submissionの承認は含まない。
- 2026-09-20 11:46:53 UTC に `uv run kaggle quota --format json` を確認。GPUは使用16.32時間、残28.68時間、refreshは2026-09-26 00:00:00 UTC。週30 GPU時間の作業上限まで13.68時間。trainとinferenceを合わせてこの範囲で実行する。
- train package metadataは `enable_gpu=true`、`enable_tpu=false`、`machine_shape=NvidiaTeslaT4`、internet無効。parent exp016の2fold学習4,620.86秒を単純に3倍すると約3.85時間だが、Self-Attentionの増分は未測定。Notebook内の64 window benchmarkと12時間gateに加え、残予算と推論費用を考慮して続行を判断する。
- `make validate-exp`、`make check-exp`、`make test-exp`、生成package metadata検証が通過。PyTorch動的テストはローカルにtorchがなくskipし、Notebook内で実行する。

## 2026-09-20 Kaggle train v1 実行

- `make push-kaggle-train EXP=exp031_frame_self_attention_spatial` で `kentookumura/exp031-frame-self-attention-spatial-train` version 1を起動。KaggleからpullしたmetadataでもGPU T4、TPU無効、internet無効を確認。
- Kaggle preflightは旧checkpointのstrict読み込み・logits完全一致、3構成のpadding・順序・空集合・勾配・checkpoint復元を通過。GTが空のwindow 994件を公開処理と同じ条件で除外し、18,707 windowを使用。
- 6構成×64 windowのbenchmarkで保守的な学習時間外挿は24,794.75秒 = 約6.89時間（1.5倍係数込み）。構成別のpeak allocated GPU memory最大は281,273,344 byte。12時間のNotebook gateと週30 GPU時間の残予算13.68時間を確認し、フル学習へ進んだ。graph評価前に残quotaを再確認する。

## 2026-09-20 公式graph評価前の条件を追記

- exp030の診断は旧checkpoint互換性、モデルの機能確認、最大級windowでの実行可能性までを示し、接続精度は測っていない。exp031への引き継ぎを `requirements.md` に明記した。
- train v1の開始後、ユーザーからの指摘を受け、同一runの現行対照と各Self-Attention構成を両胚の隣接2フレーム指標で比較する条件を `requirements.md` に追記した。上記の当初手順4と異なり、graph推論前にこの比較と進行条件の判定を挟む。これは学習開始前に定めていた条件として扱わない。
- train成果物を取得後、指標と件数を `metrics.json` に記録し、条件の成立・不成立とgraphへ進む理由をここに記録する。既存のユーザー指示「実行してください。」は公式graph評価までの実行承認として記録済みだが、前段の結果と残quotaを確認せずにinferenceを開始しない。
- 同じKaggle train v1で現行対照 fold 0 のepoch 0が完了。epoch所要400.46秒、内部検証edge_accuracy 0.9998668、positive_edge_recall 0.9586115。これは内部選択用であり、outer評価または公式scoreではない。Kaggle statusはRUNNING。

## 2026-09-21 Kaggle train v1 完了とgraph進行gate

- `kentookumura/exp031-frame-self-attention-spatial-train` version 1 はKaggle status `COMPLETE`。outputを `/tmp/kaggle-output/exp031_frame_self_attention_spatial/train` に取得。Notebook実行時間 11662.80秒（約 3.24時間）。6つのmodel file SHAとmanifest SHAを保存値と照合して一致。
- Kaggle quota確認時のGPU使用は22.07時間、アカウント残22.93時間。週30 GPU時間の作業上限まで7.93時間。
- 同一runのlegacyを主対照として、同じouter window・教師・分母で隣接2フレーム指標を比較。教師負例pairへの予測数は 保存済みの評価pair数、正答pair数、既知edge回収数から整数復元し、丸め後の元指標との一致を確認。

| 評価胚 | 構成 | 既知edge回収 | 教師負例pairへの予測 | 既知分裂母回収 |
| --- | --- | ---: | ---: | ---: |
| 44b6 | legacy | 17,959/18,949 (94.775%) | 1,341 | 6/22 |
| 44b6 | self_only | 17,214/18,949 (90.844%) | 1,674 | 4/22 |
| 44b6 | self_cross | 18,074/18,949 (95.382%) | 1,528 | 6/22 |
| 6bba | legacy | 100,230/103,393 (96.941%) | 4,355 | 48/108 |
| 6bba | self_only | 98,232/103,393 (95.008%) | 7,202 | 39/108 |
| 6bba | self_cross | 99,874/103,393 (96.596%) | 5,197 | 53/108 |

- `self_only` は両胚の回収率と教師負例pair数で現行対照より悪化。`self_cross` は44b6で既知edge回収が増え、6bbaで分裂母回収が増えた一方、教師負例pairへの予測が両胚で増加。いずれも `requirements.md` の自動進行条件を満たさない。
- 既存mask内のpairのうち未知endpointを含む割合は6bbaが97.20%、44b6が99.28%。部分注釈由来の教師負例pair数だけで真の誤接続全体は判断できず、隣接2フレーム指標はILP・graph repair後の公式scoreを代替しない。
- 全199動画のgraph推論は開始せず、公式score未計測。契約に従いユーザーの明示判断を待つ。実験statusは `running` のまま。Kaggle submissionは行っていない。

## 2026-09-21 コミット履歴との対応確認

- 指定コミット 7237a6532fc48b3df36f4b0a84c2b25a325c79a8 の acklog/frame_self_attention_spatial.md は、exp025のSelf-Attentionに距離・相対位置biasまたは近傍制限を加える別候補を明示している。現在のexp031は config.yaml の lineage.backlog_candidate: frame_self_attention のとおり旧候補の実装であり、実験名が別候補と衝突した。
- exp031の学習結果は距離bias・近傍制限の効果の証拠ではない。これらの実装・Kaggle実行・公式graph評価は行っていない。採否・完了判断はユーザーへ残す。
