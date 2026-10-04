# exp008_deformation_pair_aux 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog `deformation_pairs`からの移行。2026-09-11に、学習画像へ既知の滑らかな3D変形を適用し、対応する特徴を補助lossで学習する案を提示し、ユーザーが「それぞれ実験化してください。実装はまだです。」と実験化だけを承認した。
- 移行元backlog: `backlog/deformation_pairs.md`。本書と`config.yaml`への移行確認後、元候補と未着手行を削除する。
- 対応する上位仮説: `HYP-20260910-07`。
- 上位仮説のうちこの実験が検証する範囲: 関係が既知の滑らかな3D画像変形から密な特徴対応targetを作り、疎い実注釈だけのexp005より外側の胚で細胞中心と接続を回収できるかを測る。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: 実画像への移行、合成痕跡の識別可能性、分裂の生成と系譜整合、`synthetic_divisions`、`division_lookalikes`、`masked_video_pretrain`の比較。
- 親実験: [`exp005_embryo_holdout_batch8`](../exp005_embryo_holdout_batch8/)。batch size 8で成立したouter fold、モデル、基本教師、基本loss、3 epochs、推論、decode、公式評価を比較基準にする。
- 根拠 / 一次資料 / 参照実装: 移行前の`backlog/deformation_pairs.md`の内容を本書へ移した。[未決事項調査](../../docs/surveys/biohub-backlog-readiness_20260910.md#deformation_pairs)、[仮説調査I07](../../docs/surveys/biohub-accuracy-hypotheses_20260910.md#i07-対応関係が分かる画像変形と合成分裂で教師を増やす)、[exp005要件](../exp005_embryo_holdout_batch8/requirements.md)、[exp005設定](../exp005_embryo_holdout_batch8/config.yaml)、[exp003で照合した公式評価](../exp003_official_metric_audit/result.md)を継続参照する。
- 固定するもの: exp005のouter fold、内部split、seed 42、モデル本体、細胞中心・接続の教師とloss、batch size 8、3 epochs、checkpoint selector、推論、decode、閾値、整数線形計画の費用、公式評価を固定する。
- 変更するもの: outer学習胚の勾配更新用windowの50%へ、物理座標上の滑らかで折り返しのない3D変形を1個生成する。元画像と変形画像の対応する特徴vectorへcosine距離の補助lossを重み0.1で追加する。
- 最小の反証可能な検証: active variant 1、outer 2fold、各3 epochsを学習し、exp005の保存済み結果をcontrolとして外側199動画の公式指標を比較する。変形なしの別controlは再学習しないため、run間の確率差を限界として併記する。
- 成功条件: 実行成功には、変形の正Jacobian、valid mask、対応target、補助loss、2foldモデル、199動画予測、公式評価とSHAが揃うことを要求する。仮説の支持には、公式指標が両胚でexp005より改善し、実行失敗が増えず、train/inferenceの各Notebookが12時間以内であることを要求する。
- 停止条件: outer評価胚から変形分布を作る、Jacobianが正でない変形を使用する、crop外対応を正例にする、無効変形率が1%を超える、base lossまたはdecodeが変わる、または各Notebookの11.5時間gateを超える見込みなら停止する。
- 実行しないこと: 人工分裂の生成、娘細胞の追加、実GEFFにないedgeの正例化、外側評価胚による変形量・loss重み選択、複数の変形強度やloss重みのsweep、閾値調整、hidden test推論、competition submissionを行わない。
- 未決事項: なし。D1は下記の変形とcosine feature lossの1設定、D2は無料GPU枠かつ各Notebook 11.5時間gate、D3は両胚改善・失敗増加なし・推論12時間以内の推奨条件として承認済みである。実装と実行は未承認であり開始しない。
- backlog記録から解釈を変更した箇所とユーザー承認: 候補作成時はexp002を構成参照先としていた。2026-09-11の実験案と実験化承認に基づき、同じモデルの胚holdoutをbatch size 8で完走したexp005を親・controlへ更新する。滑らかな可逆3D変位場、領域外を未知にすること、人工対応だけを追加する範囲は維持する。

## 判断履歴

- 2026-09-10: 画像と中心座標を同時に変形すれば対応を作れる一方、任意の変形では分裂を生成できないことを確認し、滑らかな可逆3D変位場による最初の比較案をbacklogへ記録した。
- 2026-09-11: ユーザーの「すべて推奨でいいです」により、無料枠、両胚改善、実行失敗増加なし、推論12時間以内という共通方針が承認された。
- 2026-09-11: exp005を親に、既知3D変形の対応特徴だけを追加する長時間GPU案を提示した。
- 2026-09-11: ユーザーが本候補の実験化を承認し、実装はまだ行わないよう指定した。

## 手法契約

- 依頼原文: 「GPUの使用時間は日本時間の明日9時にリセットされます」「3つ教えてください」「それぞれ実験化してください。実装はまだです。」
- 期待する成果: fold-safeな既知3D変形pair、密なfeature対応target、補助lossを持つ2foldモデル、外側199動画の予測と公式評価を得る。
- input: outer学習胚の2時刻3D画像windowと、同じwindowから生成した滑らかな3D変形画像。outer評価胚は通常の画像推論と予測固定後の採点だけに使う。
- target / objective: exp005の細胞中心検出・接続objectiveに、既知変形で対応する元画像と変形画像の中間featureを近づけるobjectiveを追加する。
- output: exp005と同じ中心・edge予測に加え、学習時だけ対応評価へ使うL2正規化済み中間featureとvalid correspondence maskを出す。
- loss: exp005のbase lossに、valid対応点における平均`1 - cosine_similarity`を0.1倍して加える。変形pairを使わないminibatchでは補助項を0とする。
- decode: exp005と完全に同じ。変形画像、対応feature、補助headを推論入力やgraph選択へ使わない。
- context unit: 対応targetは1つの2時刻3D画像window内のfeature voxel、推論と評価は1動画graph、outer独立性は1胚。
- 実装区分: `faithful`。候補で定めた既知の滑らかな3D変形と密な対応の補助学習を実装し、人工分裂は別候補として除外する。
- 省略する機構と理由: 人工分裂、分裂に似た通常事象、masked video事前学習は同じ上位仮説の別候補であり、原因を分けるため含めない。
- proxyで検証できない主張: `N/A`。
- proxyの場合のユーザー承認: `N/A`。
- この実験が支持 / 棄却できる主張: 指定した変形分布とfeature cosine lossが、exp005の学習条件で外側2胚の中心・接続予測を改善するか。
- この実験では判断できない主張: 任意のdeformation augmentation、人工分裂全般、別feature layer・loss・重みの優劣、hidden testやPublic LBの改善。

## 実装方法

- アプローチ: 同じnetworkへ元windowと変形windowを通す。細胞中心をsampleする直前の共有3D feature mapから、既知変形で対応する位置をtrilinear interpolationで取り出し、valid mask上のcosine距離をbase lossへ追加する。
- inputの実装箇所と変換: outer学習胚の勾配更新用windowだけを対象に、4×4×4 control gridの変位を物理座標でsampleし、tricubic interpolationで密な変位場へする。最大変位は同じ勾配更新subsetの正しい隣接edgeにおける物理移動距離95 percentileと7 µmの小さい方とする。
- 変形の有効性: 変位を必要に応じて半減し、離散Jacobian determinantが全valid voxelで正になるまで最大8回調整する。sourceとmapped pointがcrop内で、interpolation境界から2 voxel以上内側の点だけをvalidとする。8回で成立しないpairはエラーとして記録し、無効率が1%を超えたrunは停止する。
- target / objectiveの構築箇所: 変形前feature座標を既知の変位場で変形後feature座標へ写し、両feature vectorをL2正規化して対応targetを構築する。GEFFに存在しないnodeやedgeを追加教師にしない。
- outputの生成箇所と表現: foldごとのcheckpointと、変形設定、最大変位、valid率、Jacobian最小値、base/aux loss、model manifest、外側prediction・candidate・評価summaryを保存する。
- lossの実装箇所: exp005のtrain stepへ補助lossを加える。pair probability 0.5、aux weight 0.1を固定し、outer結果を見て変更しない。
- decode / postprocessの実装箇所: exp005 inferenceをそのまま使い、training-onlyの変形処理をimportしないことをtestする。
- context unitを保つ処理箇所: stable seedをexperiment、fold、sample、epoch、window indexからSHA-256で生成し、windowごとに独立した変形を作る。変形量の統計はfoldの勾配更新subsetだけから計算する。
- 変更するファイル / component: 実装時にtrain/inferenceのJupytext sourceとNotebook、変形・feature sampling helper、実験固有testを追加する。今回変更するのは契約・設定・未実装状態の記録だけである。
- 固定事項を保つ確認方法: exp005 configとの差分をtraining-only deformation、feature loss、runtime、出力先に限定するtestを作る。
- 参照sourceとの一致を確認するテスト: exp005 source manifest SHA、outer/internal split、base model、base loss、decode、閾値、公式評価器を照合する。
- 承認済み差分を確認するテスト: pair probability 0.5、aux weight 0.1、4×4×4 control grid、training-subset由来の最大変位、正Jacobian、crop外mask、stable per-window seedを検査する。

## 探索幅とpivot判定

- 変更class: `representation`。既知変形に対する密なfeature対応を新しいtraining targetと中間outputとして追加する。
- 同じ親 / familyで連続した小改善実験数: 0。parameterやpostprocessではなくtraining representationを変える最初の比較である。
- positiveなoracle headroom / coverage / 誤差非相関性: exp005の段階別headroomは未集計である。本実験は期限前の余剰GPUを使う独立した高upside比較として承認された。
- 比較したtarget、output、decode、context unitを変える案: 本実験がtargetと中間outputを変える。decodeと動画単位の評価は比較可能性のため固定する。
- 小改善の継続またはpivotを選ぶ根拠: 注釈nodeが推定総nodeの一部に限られるという既存調査に対し、画像変形から正解既知の対応を増やすことで、閾値調整とは異なる原因を検証できる。
- `kaggle-idea-forge`の実行要否と根拠: 追加実行不要。直前の次実験検討でparameter tuningに限定しない候補として生成され、ユーザーが実験化を選んだ。

## 再現性・リスク

- 実行予定: active variant 1、model/config 1、outer fold 2、booster 0、選択済みモデル2個。親controlは再学習せず、exp005の保存済みmetrics・predictionを比較対象にする。
- 追加GPUコスト: paired forwardによりtrain 9〜11.5時間を見込む。smokeの保守係数込みで11.5時間を超える場合は設定を縮小せず停止する。inferenceはexp005と同じで別Notebookとする。
- seed policy: base trainingはseed 42、内部splitは0。変形はexperiment、fold、sample、epoch、window indexからstable SHA-256 seedを作る。
- stochastic 処理の有無: random initialization、DataLoader shuffle、既存augmentation、3D変形、pair sampling、CUDA kernelがある。
- stochastic feature generation / augmentation / seed bagging の有無: 変形とpair samplingあり、seed baggingなし。
- 並列処理と乱数の関係: global RNGを使わず、各windowのstable seedから局所generatorを作る。worker順序で変形内容を変えない。
- CPU/GPU runtime と deterministic flags: Kaggle T4 2基、DataParallel、AMP、internet無効。GPUと既存augmentationのためbitwise deterministic anchorとは扱わない。
- train cache / test feature regeneration の SHA 記録方針: 変形設定schema、fold別最大変位、sampled fieldの診断content SHA、candidate array content SHAを記録する。推論では変形を再生成しない。
- model manifest / prediction / submission SHA 記録方針: split、checkpoint、model manifest、外側prediction、candidate、graph、評価summaryのSHAを記録する。submissionは作らない。
- Kaggle package bootstrap 確認方針: 実装後に正のNotebook、config、metrics、helper、source manifestの一致をpush前validatorで確認する。
- リークリスク: 外側評価胚のGEFF、画像統計、移動分布を変形設計やloss選択へ使わない。
- CV/LB 不一致リスク: 2胚holdoutの結果でありPublic LBではない。外側結果を見て同じ実験内の変形強度を再選択しない。
- ランタイム/メモリリスク: paired forwardで時間とactivation memoryが増える。batch size 8を固定したままsmokeし、OOMまたは11.5時間超過なら停止して契約変更を相談する。
- 再現性リスク: 既存augmentationとCUDAは完全固定されない。変形fieldだけはstable keyで固定し、controlとのrun間差を限界として記録する。
- 手法忠実性リスク: 画像だけ変形してfeature対応を誤らせること、voxel座標のまま異方性を無視すること、crop外を正例化することをtestで防ぐ。
- 過度な縮小 / proxy化リスク: sparseな中心点だけの一致lossへ置換したり、1foldだけで結論を出したりしない。

## 受け入れ基準

- [x] backlogの仮説、検証範囲、残る検証、根拠、差分、固定事項、実装方法、成功条件、停止条件、禁止事項、判断履歴を移行した。
- [x] 手法契約の `input / target / output / loss / decode / context unit` が実装前に一意である。
- [x] `config.yaml`の`HYP-20260910-07`と`deformation_pairs`が本書と一致する。
- [ ] train/inference Notebook、変形helper、実験固有testを実装する。
- [ ] `validate-exp`、`check-exp`、`test-exp`を実装後に通す。
- [ ] Kaggleで2fold学習、199動画推論、変形診断、公式評価とSHAを取得する。
- [ ] 実験の完了、採用、不採用は結果提示後にユーザーが判断する。
