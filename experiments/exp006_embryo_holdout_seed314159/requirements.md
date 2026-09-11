# exp006_embryo_holdout_seed314159 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: 直接承認。2026-09-11に、exp005と同一条件で第2seedをフル学習する提案に対し、ユーザーが「それぞれ実験化してください。実装はまだです。」と実験化だけを承認した。
- 移行元backlog: `N/A`。直接承認のため形式的なbacklog候補は作らない。
- 対応する上位仮説: `N/A`。既存の検証中仮説へ根拠なく紐づけない。
- 上位仮説のうちこの実験が検証する範囲: exp005と同じ胚分離構成を別の固定seedで再学習し、保存予測の精度と誤りがどの程度変わるかを測る。
- この実験だけで上位仮説を判断できるか: いいえ。単一の追加seedだけではseed分布全体やensembleの改善を判断しない。
- 上位仮説の判断に残る検証: exp005との予測差、動画別誤差の非相関性、2-seed融合を別比較として評価すること。
- 親実験: [`exp005_embryo_holdout_batch8`](../exp005_embryo_holdout_batch8/)。同実験の2方向の胚holdout、batch size 8、モデル、教師、損失、3 epochs、推論、候補保存、公式評価を固定する。
- 根拠 / 一次資料 / 参照実装: [exp005の要件](../exp005_embryo_holdout_batch8/requirements.md)、[exp005の設定](../exp005_embryo_holdout_batch8/config.yaml)、[exp005のmetrics](../exp005_embryo_holdout_batch8/metrics.json)。exp005 train version 1はT4 2基で23046.864秒を要し、2foldのcheckpointを保存した。
- 固定するもの: exp005の固定source、外側2fold、内部split seed 0、学習・評価動画、`TemporalUNet3D_SimpleNodeTransformer`、batch size 8、3 epochs、AdamW、learning rate 0.0001、augmentation、損失、内部checkpoint選択、推論、decode、公式評価を維持する。
- 変更するもの: model初期化、PyTorch、DataLoaderに渡す実験seedを42から314159へ変更する。固定source内で引数なしに生成されるaugmentation用NumPy乱数は変更せず、そのため本実験をseedだけの厳密な因果比較とは扱わない。
- 最小の反証可能な検証: 1構成、外側2fold、各3 epochsをscratchから学習し、外側評価胚の199動画を推論する。exp005と同じ公式指標を全体・胚別・動画別に比較し、prediction差と誤りの非相関性を保存する。
- 成功条件: 2fold学習と199動画の推論が欠落なく成立し、exp005と同じ評価処理で比較できることを実行成功とする。追加seedが有用という仮説の支持には、両胚で公式指標が悪化せず、かつexp005と異なる誤りが観測されることを要求する。微小差の閾値は実測前に設定しない。
- 停止条件: 外側評価胚の混入、split不一致、設定差がseed以外へ拡大すること、199動画の欠落、再採点不一致、学習が7時間の実行枠を超える見込み、または1Notebook 12時間制約超過があれば停止する。
- 実行しないこと: ensemble、logit融合、閾値調整、追加seedの探索、epoch数変更、親controlの再学習、hidden test推論、competition submission、Public LB取得を本実験へ含めない。
- 未決事項: なし。seedは314159、学習上限は7時間、親との比較方法は上記で固定する。実装と実行は未承認であり開始しない。
- backlog記録から解釈を変更した箇所とユーザー承認: `N/A`。

## 判断履歴

- 2026-09-11: exp005 inferenceと並行して、約7時間の第2seedフル学習を行い、後続の2-seed融合にも使える独立モデルを作る案を提示した。
- 2026-09-11: ユーザーが3案すべての実験化を承認し、実装はまだ行わないよう指定した。seed 314159はexp005の42と重ならない固定値として契約した。

## 手法契約

実装区分は`docs/glossary.md`に定義したこのリポジトリ内の管理用ラベルを使う。処理内容と限界を先に記録する。

- 依頼原文: 「1つ目もある程度長い実験でいいと思いますけど」「それぞれ実験化してください。実装はまだです。」
- 期待する成果: exp005と同一構成のseed 314159モデル2個、外側評価胚199動画の予測、候補cache、公式評価、exp005との予測差を得る。
- input: 外側学習胚のZarr画像とGEFF注釈。外側評価胚は推論時に画像と利用可能なmetadataだけを読み、GEFFは予測固定後の採点だけに使う。
- target / objective: exp005と同じ細胞中心検出と隣接時刻間の接続予測。新しい教師、損失、出力は導入しない。
- output: foldごとの選択済みcheckpoint、分割・モデルmanifest、199動画の予測graphと候補cache、公式評価、exp005との動画別prediction差。
- loss: exp005と同じ重み付き検出損失と接続損失。
- decode: exp005と同じ検出時augmentation、中心抽出、接続softmax、閾値、整数線形計画を使う。
- context unit: モデル入力は2時刻の3D画像窓、推論と保存は1動画、外側分割は1胚。
- 実装区分: `faithful`。exp005の処理を保ち、学習seedだけを承認済みparameterとして変える。
- 省略する機構と理由: 2-seed融合は別のpostprocess比較であり、本実験では第2モデルの生成と独立評価までに限定する。
- proxyで検証できない主張: `N/A`。
- proxyの場合のユーザー承認: `N/A`。
- この実験が支持 / 棄却できる主張: exp005の固定構成で別seedのモデルを作れるか、1追加seedで精度と誤りがどの程度変わるか。
- この実験では判断できない主張: seed分布全体、2-seed融合の改善、hidden test一般化、Public LB改善。

## 実装方法

- アプローチ: 実装時にexp005のcompact self-contained train/inference Notebookを構成参照元とし、設定seedだけを314159へ変更する。親のNotebookやsourceはこの実験化時点ではコピーしない。
- inputの実装箇所と変換: exp005と同じ動的Zarr/GEFF列挙と、同じouter fold・内部split manifestを使う。
- target / objectiveの構築箇所: exp005で固定した公式sourceの教師作成と損失を変更しない。
- outputの生成箇所と表現: 実装時に`artifacts/models/`、`artifacts/predictions/`、`artifacts/candidates/`、各manifestと公式評価summaryを作る。
- lossの実装箇所: exp005と同じ学習関数へ同じ引数を渡し、RNG seedだけ314159とする。
- decode / postprocessの実装箇所: exp005 inferenceを変更せず、比較用prediction差の集計だけを追加する。
- context unitを保つ処理箇所: outer splitは胚単位、学習は2時刻窓、推論・解放は動画単位を維持する。
- 変更するファイル / component: 実装時にtrain/inferenceのJupytext sourceとNotebook、実験固有testを追加する。今回変更するのは実験契約、設定、未実装状態の記録だけである。
- 固定事項を保つ確認方法: exp005 configとの差分が実験名、系譜、seed、実行枠、出力先だけであることをtestで固定する。
- 参照sourceとの一致を確認するテスト: exp005のsource manifest SHA、モデル、loss、decode、split件数、閾値を照合する。
- 承認済み差分を確認するテスト: `reproducibility.seed`と学習seedが314159、内部split seedが0、1構成・2fold・3 epochs・モデル2個であることを確認する。

## 探索幅とpivot判定

- 変更class: `parameter`。seedだけを変える1回の独立rerunとして管理する。
- 同じ親 / familyで連続した小改善実験数: 2。exp005のbatch size変更に続くparameter差分だが、精度調整ではなく予測分散を測るための一度限りのrerunである。
- positiveなoracle headroom / coverage / 誤差非相関性: 非相関性は未測定で、本実験が直接測る。
- 比較したtarget、output、decode、context unitを変える案: 同時に実験化するexp008は既知3D変形による対応特徴を追加し、targetと中間outputを変える。
- 小改善の継続またはpivotを選ぶ根拠: exp005 1seedだけではseed変動と融合余地を評価できないが、3つ以上のseed探索は無料枠に対して費用が大きい。追加は1seedで止める。
- `kaggle-idea-forge`の実行要否と根拠: 追加実行不要。直前の次実験検討で異なる問題表現を含む案を生成し、ユーザーがparameter案1件、selector案1件、representation案1件の実験化を選んだ。

## 再現性・リスク

- 実行予定: active variant 1、model/config 1、outer fold 2、booster 0、選択済みモデル2個。親controlは再学習せず、exp005の保存済みmetrics・予測を比較対象にする。
- 追加GPUコスト: exp005実測を根拠にtrain 6.4〜7時間を見込み、7時間gateを置く。推論は別Notebookでsmokeから12時間以内を再確認する。
- seed policy: model、PyTorch、DataLoaderは314159、内部splitは0。処理単位ごとの乱数は実装可能な箇所でstable keyから生成する。
- stochastic 処理の有無: random initialization、DataLoader shuffle、brightness/flip augmentation、CUDA kernelがある。
- stochastic feature generation / augmentation / seed bagging の有無: augmentationあり、seed baggingなし。追加seedは1個だけである。
- 並列処理と乱数の関係: DataLoader generatorを固定する。親source内の引数なしNumPy RNGはbitwise固定できないため、並列順序を含む制約をmanifestへ記録する。
- CPU/GPU runtime と deterministic flags: Kaggle T4 2基、DataParallel、AMP、internet無効。bitwise deterministic anchorとは扱わない。
- train cache / test feature regeneration の SHA 記録方針: exp005と同じcanonical array content SHAを記録し、NPZ file SHAは副証拠とする。
- model manifest / prediction / submission SHA 記録方針: split、checkpoint、model manifest、OOF相当の外側胚prediction、候補、graph、評価summaryのSHAを記録する。submissionは作らない。
- Kaggle package bootstrap 確認方針: 実装後のprepareとpush前validatorで正のNotebook、config、metrics、source manifestとの一致を確認する。
- リークリスク: 外側評価胚を学習、checkpoint選択、閾値・費用選択に使わない。exp005との差を見て追加seedや設定を選ばない。
- CV/LB 不一致リスク: 胚holdoutの結果でありPublic LBではない。LB改善を推定しない。
- ランタイム/メモリリスク: 7時間gateを超える場合はepoch、動画数、解像度を縮小せず停止し、契約変更をユーザーへ確認する。
- 再現性リスク: 親augmentationとGPU kernelがbitwise固定されないため、観測差をseedだけの効果と断定しない。
- 手法忠実性リスク: seed以外の差が入ると比較不能になるため、config差分とsource SHAをtestで固定する。
- 過度な縮小 / proxy化リスク: 時間超過時に1foldや短縮epochへ黙って変更しない。

## 受け入れ基準

- [x] 実験化承認、親実験、seed 314159、固定事項、成功条件、停止条件、実行しないことを記録した。
- [x] 手法契約の `input / target / output / loss / decode / context unit` が実装前に一意である。
- [x] `config.yaml`のlineageと本書が一致し、statusは`planned`である。
- [ ] train/inference Notebookと実験固有testを実装する。
- [ ] `validate-exp`、`check-exp`、`test-exp`を実装後に通す。
- [ ] Kaggle T4 2基で2fold学習と199動画推論を実行し、比較可能なSHAと公式評価を記録する。
- [ ] 実験の完了、採用、不採用は結果提示後にユーザーが判断する。
