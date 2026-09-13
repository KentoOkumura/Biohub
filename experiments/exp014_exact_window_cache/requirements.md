# exp014_exact_window_cache 要件と実装方法

この文書を、実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は`SESSION_NOTES.md`へ記録し、この文書へ重複させない。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlogの状態は`設計可能・実験化未承認`。2026-09-13のユーザー指示「それではこれで進めてください」により、exp010を停止して`exact_window_cache`へ進むことが承認された。
- 移行元backlog: `backlog/exact_window_cache.md`。本契約への移行確認後、元ファイルと未着手行を`kaggle-strategy`で削除する。
- 対応する上位仮説: `HYP-20260910-12`
- 上位仮説のうちこの実験が検証する範囲: 固定公開モデルの同一候補点特徴を再利用しても下流出力が変わらず、画像からの再抽出より反復費用を下げられるか。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: `oracle_stage_limits`による制限箇所の診断と、固定特徴で学習するtrackerの精度・学習再現性。
- 親実験: `exp013_public_notebook_replay`。公開モデル選定と特徴契約は`exp011_public_detector_selection`から継承する。
- 根拠 / 一次資料 / 参照実装: `backlog/exact_window_cache.md`、`docs/surveys/biohub-backlog-readiness_20260910.md#exact_window_cache`、exp011のmodel manifest、exp013の2-run一致証拠、保存済み公開Notebook。
- 固定するもの: 公開Notebook、3 checkpoint、入力manifest、2-frame window、前処理、8-view D4 detection/primary association feature TTA、secondary original-view feature、候補生成、候補座標、tracker、edge fusion、ILP、graph repair、dtype。
- 変更するもの: primary/secondary候補点特徴と対応情報をwindow単位のNPZへ保存し、再読込した値をtrackerへ渡す。時間・容量・peak memoryと完全一致を記録する。
- 最小の反証可能な検証: 公開test全4動画の全ての有効な連続frame pairで直接抽出値とNPZ再読込値を完全一致比較し、直接scoreと再読込scoreを完全一致比較する。最終graph/submissionをexp013のauthoritative SHAと比較する。
- 成功条件: 全cache配列とprimary/secondary edge logitsが完全一致し、最終graph topologyとraw `submission.csv`がexp013に一致する。cache書込・読込時間、容量、最大GPU memoryを取得し、再読込時間が画像特徴抽出時間より短い。
- 停止条件: window identity、schema、配列、edge logits、最終graph/submissionのいずれかが不一致、必要なcacheが欠落、または保存量・読込費用が再利用の利点を失わせる。
- 実行しないこと: 学習、特徴量圧縮・量子化、frame単独特徴への置換、異なるwindow・重み・座標の混用、全voxel特徴または全候補pair中間値の保存、Kaggle submission。
- 未決事項: なし。
- backlog記録から解釈を変更した箇所とユーザー承認: 親の基準予測を、exp011選定時点の未実行予測ではなく、2026-09-13に2-run一致を確認したexp013の公開test全件予測へ具体化した。ユーザーはexp013で再現性を確認する方針と本候補への移行を承認した。

## 判断履歴

- 2026-09-10: 調査候補I12から、固定特徴の等価な再利用を候補化した。
- 2026-09-12: 公開モデルをexp011で固定し、元の時間窓と候補点特徴を元dtypeで保存する比較を設計可能と確認した。
- 2026-09-13: exp013で同じ公開推論の候補座標、graph topology、run統計、raw submissionが2 run一致した。
- 2026-09-13: ユーザーがexp010を停止し、`exact_window_cache`、`oracle_stage_limits`、tracker学習の順に進むことを承認した。

## 手法契約

- 依頼原文: 「それではこれで進めてください」
- 期待する成果: 固定公開画像特徴を後続tracker学習で再利用できるcacheの等価性、保存量、読込費用を確認する。
- input: canonical public test 4動画、固定primary/secondary model、元の2-frame window、候補ID、downsample grid座標、物理座標、検出score、位置埋め込み、候補mask。
- target / objective: 直接抽出と保存済み特徴を使った下流計算の等価性を保ち、反復費用を減らす。
- output: windowごとのNPZ、window manifest、cache集計、直接/再読込score差、exp013との最終graph/submission比較を含むreceipt。
- loss: なし。全model weightを固定する。
- decode: exp013と同じprimary/secondary association、bidirectional fusion、ILP、graph repair、submission生成。
- context unit: 1動画内の元の2-frame windowと、その両frameの候補点集合。
- 実装区分: `faithful`。保存・再読込以外の予測処理を変更しない。
- 省略する機構と理由: 全voxel特徴と全候補pair中間値は、後続tracker学習に不要で容量を増やすため保存しない。
- proxyで検証できない主張: N/A。
- proxyの場合のユーザー承認: N/A。
- この実験が支持 / 棄却できる主張: 固定候補点特徴をNPZで可逆に再利用でき、同じtracker出力を保ちながら特徴再抽出より安く読み込めること。
- この実験では判断できない主張: cacheを使う新しいtrackerがCV/LBを改善すること、tracker学習自体の再現性、hidden testでの保存容量。

## 実装方法

- アプローチ: exp013の推論sourceへ、候補点特徴の保存・再読込と完全一致guardだけを追加する。
- inputの実装箇所と変換: 公開sourceの`predict_video`内で、候補抽出後のID、grid/物理座標、検出確率、位置特徴、mask、primary/secondary 32-channel featureを連続配列へ変換する。
- target / objectiveの構築箇所: `window_cache.py`がmetadataと配列schema/content SHAをNPZに保存し、同じwindow identityで再読込する。
- outputの生成箇所と表現: `/kaggle/working/window_cache/<dataset>/<t_src>_<t_tgt>.npz`とGPU shard別JSONL manifest。最終cellでmanifestを集約して`window_cache_summary.json`とreceiptへ記録する。
- lossの実装箇所: なし。
- decode / postprocessの実装箇所: 再読込した特徴・座標・位置特徴・maskからedge logitsを再計算し、以後はexp013と同じ処理へ渡す。
- context unitを保つ処理箇所: cache metadataのdataset、`window_frames`、window size、model SHA、feature contractで別windowの混用を拒否する。
- 変更するファイル / component: inference Jupytext source/notebook、`window_cache.py`、実験固有test、configと記録文書。
- 固定事項を保つ確認方法: 参照Notebook SHAと3 checkpoint SHAをpinし、予測parameterを変更せず、exp013のcandidate coordinate、graph topology、submission SHAへ照合する。
- 参照sourceとの一致を確認するテスト: 保存公開Notebook SHA、許可したexp014 marker、禁止した決定論flag変更を静的検査する。
- 承認済み差分を確認するテスト: NPZ round-tripのdtype/shape/value完全一致、wrong-window拒否、cache/score/final-output guard marker、lineageを検査する。

## 探索幅とpivot判定

- 変更class: `representation`
- 同じ親 / familyで連続した小改善実験数: 0。exp014は精度変更ではなく、後続比較の計算基盤を検証する。
- positiveなoracle headroom / coverage / 誤差非相関性: exp012で誤差診断は済んだが、本実験の成功判定は精度headroomではなく等価性と費用で行う。
- 比較したtarget、output、decode、context unitを変える案: 後続のtracker学習でtarget/lossを扱う。本実験では変えない。
- 小改善の継続またはpivotを選ぶ根拠: 固定公開検出器路線の反復費用を先に測定し、`oracle_stage_limits`とtracker学習へ進む。
- `kaggle-idea-forge` の実行要否と根拠: 不要。承認済み候補の実装であり、新しい候補探索ではない。

## 再現性・リスク

- seed policy: 明示的乱数samplingなし。exp013で同一Kaggle T4構成の2-run一致を確認済み。
- stochastic 処理の有無: CUDA kernel、SCIP solver、2 process shardingは決定論flagを強制していないが、親の2-run実測は一致した。
- stochastic feature generation / augmentation / seed bagging の有無: 学習augmentationとseed baggingなし。推論時D4 TTAは固定8変換。
- 並列処理と乱数の関係: sorted datasetを2 GPUへ交互に固定分割し、shard別manifestへ書く。
- CPU/GPU runtime と deterministic flags: exp013と同じT4 2基。新しいdeterministic flagは追加しない。
- train cache / test feature regeneration の SHA 記録方針: 各windowのschema/content/file SHAと集約content SHAを記録する。train cacheは本実験では作らない。
- model manifest / prediction / submission SHA 記録方針: exp011のmodel SHA、exp013のbaseline graph/submission SHA、exp014のcache集約、graph、submission SHAを記録する。
- Kaggle package bootstrap 確認方針: exp013と同じcanonical dataset manifest、offline wheel manifest、support source manifestを検査する。
- リークリスク: 公開testに教師は使わない。公開modelの学習来歴に由来する条件付き比較である。
- CV/LB 不一致リスク: 精度比較や提出をしない。exp013出力との等価性だけを判定する。
- ランタイム/メモリリスク: NPZ I/Oと直接/再読込scoreの二重計算が増える。cache容量とpeak memoryを必須記録する。
- 再現性リスク: NPZ file SHAはcontainer metadataの影響を受けうるため、配列schema/content SHAを主証拠にする。
- 手法忠実性リスク: feature、座標、位置特徴、maskのいずれかを元dtypeから変えるとedge scoreが変わりうるため完全一致guardで停止する。
- 過度な縮小 / proxy化リスク: 小規模例だけでなくpublic test全4動画を実行し、最終graphまで照合する。

## 受け入れ基準

- [ ] 手法契約の `input / target / output / loss / decode / context unit` がコードと一致する。
- [ ] 公開test全4動画の有効な連続frame pairがwindow manifestへ一意に記録される。
- [ ] 保存前後の全配列がdtype、shape、値について完全一致する。
- [ ] primary/secondary edge logitsが直接経路とcache経路で完全一致する。
- [ ] exp013とcandidate coordinate、graph topology、raw `submission.csv`が一致する。
- [ ] cache抽出、書込、読込の所要時間、総容量、最大GPU memoryが記録される。
- [ ] `config.yaml`の`lineage.hypothesis_id`と`lineage.backlog_candidate`がこの文書と一致する。
- [ ] 実験固有テストと静的検証が通る。
- [ ] gzip生成物を比較する場合は、decompressed content SHAを主証拠として記録する。
