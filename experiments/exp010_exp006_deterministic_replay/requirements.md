# exp010_exp006_deterministic_replay 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録する。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: 直接承認。2026-09-12にユーザーが、exp006の2fold ensembleを先に実施し、その次にexp006の再現性を担保する実験を行うよう依頼した。
- 移行元backlog / 対応する上位仮説: `N/A`。再現性監査の直接依頼であり、形式的なbacklogや仮説IDは作らない。
- 親実験: [`exp006_embryo_holdout_seed314159`](../exp006_embryo_holdout_seed314159/)。外側胚holdout、内部checkpoint選択、モデル、損失、3 epochs、推論、decode、公式評価を継承する。
- 根拠: exp006はPython・NumPy・PyTorch・DataLoaderのseedを設定したが、各itemで`np.random.default_rng()`を引数なしに生成していた。このaugmentation乱数はentropy由来でworker実行順にも依存し、同じseedから同一学習を再構成できない。
- 固定するもの: exp006のデータ、split、seed 314159、モデル構造、教師、損失、optimizer、batch size 8、2fold、3 epochs、checkpoint選択、推論、閾値、ILP、公式評価、T4 x2、internet無効を維持する。
- 変更するもの: 学習windowの順番をepochごとの決定論的samplerで生成し、augmentation RNGをglobal seed・epoch・stable item indexから作る。PyTorch/CUDAの決定論的algorithm、cuDNN、cuBLAS、TF32設定を固定する。決定論的CUDA backwardがない`MaxPool3d(kernel=2, stride=2)`は、同じ非重複2x2x2領域を同じ順でflattenしてfirst maximumを選ぶreshape+`max`へ置き換える。checkpoint tensor、候補予測、最終graph、metric rowsのcanonical content SHAを記録する。
- 最小の反証可能な検証: byte-identicalなtrain NotebookをKaggle T4 x2で2回フル実行し、各回に対応するbyte-identicalなinference Notebookを全199動画で実行する。2回のfold別checkpoint tensor SHA、候補予測SHA、最終graph内容SHA、per-sample metric SHA、公式summary SHA、CVを比較する。
- 成功条件: 2回とも2fold・3 epochsの学習と199動画のheld-out推論が完了し、設定した全canonical SHAとCVが一致すること。全項目一致後だけ`reproducibility.deterministic_anchor`を`true`へ変更する。
- 停止条件: split不一致、外側評価胚の混入、決定論的algorithm違反、2 GPU不足、run間のinput/code SHA差、学習9.25時間gateまたはNotebook 12時間制約超過見込み、199動画欠落、公式再採点不一致、canonical SHAまたはCV不一致で停止する。不一致時は原因を記録し、担保済みと扱わない。
- 実行しないこと: exp006旧checkpointとのbyte比較、seed探索、複数seed ensemble、threshold調整、epoch・fold・動画の縮小、hidden test推論、competition submission、cross-hardware・cross-library-version再現性の主張。
- 未決事項: なし。2回のtrainと2回のheld-out inferenceをこの実験のauthoritative runとする。
- backlog記録から解釈を変更した箇所とユーザー承認: `N/A`。

## 判断履歴

- 2026-09-12: exp006でaugmentationとGPU計算が完全には決定的でない点を説明し、ユーザーが今後の実験評価への影響を指摘して再現性を担保する実験を依頼した。
- 2026-09-12: ユーザーがexp006 2fold ensembleを先に行い、その次に本実験を行う順序を指定した。
- 2026-09-12: exp009を提出後、同じKaggle環境で決定論化した2回のフル学習と2回のheld-out評価を実行する契約としてexp010を開始した。

## 手法契約

- 依頼原文: 「まずはこれを実施してください。次に再現性を担保する実験を行ってください。」
- 期待する成果: 同一条件を2回実行して同じmodel tensor・OOF相当予測・CVへ到達することをSHAで確認できる後継実験。
- input: 公式train Zarr/GEFF 199組、exp006と同じ胚分離split、固定official sourceと記録済みの決定論化patch。
- target / objective: 細胞中心検出と隣接時刻間の接続予測。教師、検出loss、edge loss、内部checkpoint選択指標はexp006から変えない。
- output: 各runのfold別checkpoint/model manifest、全199動画の最終graphと候補cache、per-sample metrics、公式summary、canonical content SHA比較。
- loss: exp006と同じ重み付き検出lossと接続loss。
- decode: exp006と同じTTA、中心抽出、edge softmax、0.5閾値、最大子・親数、ILP。
- context unit: 入力は2時刻の3D画像window、乱数keyはepochとwindow item、予測・評価は1動画、外側分割は1胚。
- 実装区分: `faithful`。exp006のモデル評価機構を維持し、依頼された再現性を成立させるため乱数供給とruntime設定だけを変更する。
- 省略する機構と理由: exp005とのseed比較は本実験の目的ではないため除く。run間の完全一致だけを検証する。
- proxyで検証できない主張 / proxyの場合のユーザー承認: `N/A`。短縮proxyではなく2回とも契約どおりフル実行する。
- この実験が支持 / 棄却できる主張: 固定したKaggle T4 x2・package・source・seed条件でexp006後継の学習およびheld-out評価を同一結果へ再実行できるか。
- この実験では判断できない主張: 元exp006の旧出力の再構成、異なるGPU・driver・PyTorch版間の一致、LB改善、seed分布全体。

## 実装方法

- `FrameWindowDataset`はsamplerから`(epoch, item_index)`を受け、`SeedSequence([seed, epoch, item_index])`からaugmentation RNGを再構成する。
- train DataLoaderはmain processで決定論的なepoch別`randperm`を作るsamplerを使う。worker数8とprefetchは維持するが、worker完了順がaugmentationへ影響しない。
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`をTorch import前に設定し、`torch.use_deterministic_algorithms(True)`、deterministic debug mode、cuDNN deterministic、benchmark無効、TF32無効を学習前にassertする。
- T4上で決定論的backwardを持たない3D max-poolingは、2x2x2非重複blockのreshapeと`torch.max(dim=-1)`で置換する。forwardの領域、flatten順、first-max tie処理を元のpoolと一致させ、決定論的algorithm errorが解消するかKaggle smokeで確認する。
- checkpointのfile SHAに加え、state dictをkey順に並べ、key・dtype・shape・contiguous tensor bytesからcontent SHAを算出する。
- inferenceは候補arrayのcanonical SHA、最終graphを座標と選択edgeのcanonical arrayへしたSHA、per-sample metric payload SHA、公式summary payload SHAを保存する。raw NPZ/GEFF SHAは副証拠とする。
- 1回目のtrain/inference outputをローカルのignored artifactsへ退避してから、コードを変えずに2回目をpushする。比較scriptで入力SHAと必須content SHA/CVを検証し、結果を`metrics.json`へ記録する。
- 実験固有testは親から変えてよい箇所を決定論化patchと記録項目だけに限定し、entropy-seeded RNG、shuffle DataLoader、旧kernel dependency、旧exp005比較が残らないことを検査する。

## 探索幅とpivot判定

- 変更class: `mechanism`。予測能力の改善ではなく、同じ学習内容へ到達できる乱数・runtime機構の変更。
- active variant 1、model config 1、outer fold 2、authoritative rerun 2、booster 0。親controlや追加seedは学習しない。
- 追加GPUコスト: exp006実測を基準にtrain約6.4時間×2、held-out inference約5.3時間×2で約23.4時間を見込む。各push直前にquotaを再確認し、1Notebook 12時間制約と各gateを守る。
- `kaggle-idea-forge`は使わない。これは新規手法探索ではなく、ユーザーが指定した再現性監査である。

## 再現性・リスク

- seed policy: Python、NumPy、PyTorch、CUDAを314159へ固定し、shuffleとaugmentationはstable keyで再構成する。
- stochastic処理: 初期化、shuffle、brightness/flip、CUDA計算は存在するが、各乱数と決定論的algorithmを固定する。
- 並列処理: DataLoader worker schedulingへ依存しないitem keyを使う。DataParallelは同じ2基のT4と同じdevice構成で2回実行する。
- SHA方針: serialization metadataに左右されるfile SHAだけで判定せず、tensor/array/payloadのcanonical content SHAを主証拠とする。
- リークリスク: 外側評価胚は全prediction固定後の採点だけに使用し、学習、selection、閾値選択へ使わない。
- CV/LBリスク: 胚holdout CVの再現性実験でありLBは取得しない。CV一致をLB改善の根拠にしない。
- runtime/メモリリスク: deterministic kernelで遅くなる可能性がある。version 2 smokeの保守的推定9.071時間を通すためgateだけを9.25時間へ最小修正し、構成は縮小しない。
- 再現性リスク: PyTorchが非決定的operationを検出した場合はerrorで停止する。2回一致しても固定した環境外まで保証しない。
- 手法忠実性リスク: poolのforward領域とfirst-max tie処理を維持するが、PyTorch内部kernelは変わる。source patchとSHAを記録し、その他のmodel/loss/decode差分をtestで禁止する。

## 受け入れ基準

- [x] 直接承認、親実験、固定事項、変更点、成功条件、停止条件、実行しないことを実装前に記録した。
- [x] `input / target / output / loss / decode / context unit`を一意にした。
- [x] `config.yaml`のlineageと再現性方針が本書と一致する。
- [x] train/inference source、Notebook、comparison script、実験固有testを実装する。
- [x] `validate-exp`、`check-exp`、`test-exp`を通す。
- [ ] 同一train NotebookをKaggleで2回フル実行し、各runのoutputを保存する。
- [ ] 各train runに対応する同一inference NotebookをKaggleで2回実行し、199動画を評価する。
- [ ] 必須canonical SHAとCVが全て一致することを比較scriptで確認する。
- [ ] 一致確認後だけ`reproducibility.deterministic_anchor`を`true`へ更新する。
- [ ] 実験の完了、採用、不採用は結果提示後にユーザーが判断する。
