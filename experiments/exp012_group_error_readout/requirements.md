# exp012_group_error_readout 要件と実装方法

この文書を実装前の契約、実装方法、受け入れ条件の正とする。実行中の進捗は `SESSION_NOTES.md` へ記録する。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: backlog 候補 `group_error_readout` は `設計可能・実験化未承認`、未決事項は `なし`。2026-09-12 のユーザー依頼「group_error_readoutを実装してください」を実験化承認として扱う。
- 移行元 backlog: `backlog/group_error_readout.md`。
- 対応する上位仮説: `HYP-20260910-14`。
- 上位仮説のうちこの実験が検証する範囲: 胚、画像輝度、候補密度、候補の画像境界距離、既知分裂の有無ごとに、固定した予測の改善と悪化を分けて再現可能に集計できるかを検証する。
- この実験だけで上位仮説を判断できるか: いいえ。
- 上位仮説の判断に残る検証: exp003 の評価器一致、exp007 の重み選択、`oracle_stage_limits` の段階別上限、採用済み exp011 構成の固定予測と独立性を確認する必要がある。
- 親実験 / 比較対象: 採用済み [`exp011_public_detector_selection`](../exp011_public_detector_selection/) を公開構成の正とし、現時点で取得済みの [`exp005_embryo_holdout_batch8`](../exp005_embryo_holdout_batch8/) を固定基準、[`exp006_embryo_holdout_seed314159`](../exp006_embryo_holdout_seed314159/) を比較対象にする。
- 根拠 / 一次資料 / 参照実装: 移行元候補、exp003 の固定公式評価器、exp005・exp006 の保存済み `prediction_manifest.json` と `per_sample_metrics.json`、exp005 の保存済み画像診断表、exp011 の採用済み manifest。
- 固定するもの: 199 動画の対象一覧、2 方向の胚 holdout、各 route の checkpoint・候補・最終 graph、公式評価式、検出と接続の復号、画像 sampling 値を固定する。
- 変更するもの: 固定結果へ条件列を結合し、条件別集計と exp005 を基準にした exp006 の対応動画比較を追加する。予測値は変更しない。
- 最小の反証可能な検証: 2 route の同じ 199 動画を、胚、輝度分位点、候補密度分位点、候補の境界距離分位点、既知分裂の有無で集計する。失敗、NaN、有効件数、入力・出力 SHA を保持する。
- 成功条件: 対象一覧、条件、失敗、NaN、有効件数を保持した表を再計算できる。異なる学習来歴を route 列で分離し、診断の成立をモデル精度改善と呼ばない。
- 停止条件: 対象集合、予測 manifest、動画別評価、candidate cache、画像条件の対応、または SHA が一致しない場合は停止し、欠損を推測で補わない。
- 実行しないこと: 追加学習、再推論、閾値・ILP 費用の選択、正解を使う予測置換、hidden 入力参照、competition submission、Public LB 取得を行わない。
- 未決事項: なし。exp011 の公開構成予測は未取得であり、今回の自前予測の実装を妨げる未決事項ではない。取得後に同じ入力 schema へ追加する。
- backlog 記録から解釈を変更した箇所とユーザー承認: N/A。保存済み自前予測を先に集計し、公開構成予測を取得後に同じ集計へ載せる順序を維持する。

## 観測事実と依存

- exp005 と exp006 には同じ 199 動画の胚 holdout 予測、動画別公式評価、candidate manifest が保存されている。exp005 の画像診断表には同じ対象の輝度統計と物理 scale がある。
- exp005 の全体公式 score は `metrics.json.cv.overall.score`、exp006 の値は同じキーを正とする。本実験はこれらを基準結果として更新せず、保存済み動画別成分を条件別に再集計する。
- exp005 と exp006 の予測は学習 seed と checkpoint が異なる。学習来歴を混ぜず route 列で分け、対応動画の差を別表へ保存する。
- exp011 の公開 detector、画像 encoder、tracker 初期値は採用済みだが、公開構成の固定予測は未取得である。取得前に存在すると仮定しない。
- Assumption: 画像輝度は入力動画に固有なので、exp005 の固定画像 sampling 値を両 route の共通条件として利用できる。候補密度と境界距離は予測 route に依存するため各 manifest から計算する。
## 判断履歴

- 2026-09-10: 調査 I14 の候補として backlog 化した。
- 2026-09-12: 設計上の未決事項と測定待ちを分離し、`設計可能・実験化未承認`、未決事項 `なし` へ更新した。
- 2026-09-12: exp011 の公開構成採用により選定待ちを解除した。
- 2026-09-12: 公開 detector を固定し tracker を学習する方針に合わせ、診断範囲を更新した。
- 2026-09-12: ユーザー依頼「group_error_readoutを実装してください」により実験化を承認された。

## 手法契約

実装区分は `docs/glossary.md` に定義したこのリポジトリ内の管理用ラベルである。

- 依頼原文: 「group_error_readoutを実装してください」。
- 期待する成果: 胚・画像条件ごとの誤りを測る実行可能な diagnostic Notebook、条件別表、対応動画の route 比較、入力と出力の SHA。
- input: exp005・exp006 の固定 `per_sample_metrics.json`、`prediction_manifest.json`、candidate NPZ、exp005 の固定画像診断 CSV。公開構成予測は取得後に同じ schema で追加する。
- target / objective: 新しい学習目的はない。公式評価の動画別成分と条件を結合し、誤りが集中する条件と route 間の改善・悪化を記述する。
- output: `per_sample_readout.csv`、`group_error_summary.csv`、`paired_route_comparison.csv`、`group_error_summary.json`、`readout_manifest.json`。
- loss: なし。
- decode / 推論方法: 保存済み graph・candidate を変更せず、exp003 の公式 `summarise` と同じ集計式を適用する。正解由来の division 有無は診断列に限定する。
- context unit: 条件値と公式評価は 1 動画、外側評価の独立性と分位点境界は 1 胚、集計は route と条件 group を単位にする。
- 実装区分: `staged-faithful`。保存済み自前予測に対する指定済み診断をすべて実装し、未取得の exp011 公開構成予測だけを後段に残す。
- 省略する機構と理由: exp011 公開構成の予測集計は入力が存在しないため省略する。予測取得後は `diagnostic.prediction_sources` へ固定 SHA とともに追加できる。
- proxy で検証できない主張: N/A。proxy ではないが、現在の 2 route だけでは公開構成の精度、未知胚一般化、下流 tracker 改善を判断できない。
- proxy の場合のユーザー承認: N/A。
- この実験が支持 / 棄却できる主張: 固定自前予測について条件別の誤差を再現可能に分離でき、exp006 が exp005 より改善・悪化する動画条件を対応比較できるか。
- この実験では判断できない主張: 条件を使った補正が hidden test を改善するか、公開 detector 固定 tracker の改善幅、2 胚を越えた一般化。

## 実装方法

- アプローチ: Jupytext の `exp012_group_error_readout_diagnostic.py` を正の編集対象にし、同名 `.ipynb` を生成する。Kaggle では 3 個の固定 kernel output を CPU Notebook の入力にする。
- input の実装箇所と変換: `resolve_prediction_source` が route と固定 SHA で manifest・動画別評価を解決する。`read_feature_rows` が sample を一意 key として画像条件を読む。`readout_rows_for_route` が candidate cache の SHA、候補数、境界距離を検査して結合する。
- target / objective の構築箇所: `derive_bucket_thresholds` が exp005 の反対側の胚だけから各分位点境界を作り、`assign_condition_buckets` が両 route へ同じ境界を適用する。誤差値や正解は境界選択に使わない。
- output の生成箇所と表現: `make_group_summaries` が route ごとの条件別公式集計、`make_paired_comparisons` が exp005 の同じ動画集合を基準に exp006 の score 差、改善・悪化件数を作る。
- loss の実装箇所: N/A。
- decode / postprocess の実装箇所: 予測の decode は行わない。`summarise_official` が exp003 と同じ micro 集計、動画サイズ重み付き adjusted edge Jaccard、division 項を計算する。
- context unit を保つ処理箇所: sample 集合の完全一致、胚別件数、sample key の一意性を実行時に検査する。分位点境界は評価胚ごとに反対側の胚から作る。
- 変更するファイル / component: exp012 の config、契約・記録、diagnostic source・Notebook、実験固有 test、移行後の backlog 索引。
- 固定事項を保つ確認方法: source SHA、199 件、胚別 71/128 件、route 間 sample 集合、candidate file SHA、画像 CSV SHA、公式集計式を検査する。
- 参照 source との一致を確認するテスト: 合成行で exp003 の重み付き集計を検査し、ローカルに保存済み exp005 動画別評価を用いる test は保存証拠が存在する場合だけ全体値との一致を検査する。
- 承認済み差分を確認するテスト: model・loss・submission がなく CPU diagnostic のみであること、5 条件、反対側胚での境界決定、出力 schema、公開構成未取得の明記を静的に検査する。

## 最小検証と計算量

- variant / config / fold / booster 数: 固定予測 2 route、外側 2 胚、model config 0、fold 学習 0、booster 0。
- control 再学習: なし。exp005・exp006 の保存済み結果を読む。
- 想定 runtime / resource: Kaggle CPU、internet 無効。candidate NPZ は動画単位で読み、境界統計を計算後に解放する。GPU は使わない。
- 分位点: 0.25、0.5、0.75。各評価胚について exp005 の反対側胚にある条件値から決める。同じ境界を exp005・exp006 へ適用する。
- 胚: `44b6` と `6bba` を別々に報告する。
- 輝度: 固定画像 sampling の `image_normalized_mean` を使う。
- 候補密度: route ごとの candidate node 数を frame 数で割る。
- 境界距離: candidate の z・y・x 座標から画像の最近傍面までの物理距離を求め、動画内 median と 7 µm 以下の割合を保存する。
- 既知分裂: 動画別の `division_tp + division_fn` が正なら既知分裂ありとする。外側評価の正解由来なので診断だけに使用する。
- 必須 guard: 未注釈を誤検出・負例と呼ばない。条件別結果から閾値や補正規則を選ばない。2 胚だけの差を一般化と呼ばない。

## 探索幅とpivot判定

- 変更 class: `postprocess`。このリポジトリ内では固定予測の診断的な集計追加を表し、提出予測の後処理変更を意味しない。
- 同じ親 / family で連続した小改善実験数: 0。モデル精度を変える実験ではない。
- positive な oracle headroom / coverage / 誤差非相関性: exp005 の候補 coverage は保存済みだが、本実験は条件別に再整理する。結果取得前に新しい改善機構を選ばない。
- 比較した target、output、decode、context unit を変える案: N/A。承認済み診断の実装であり、次の学習機構は結果提示後に別候補として判断する。
- 小改善の継続または pivot を選ぶ根拠: 本実験では選ばない。
- `kaggle-idea-forge` の実行要否と根拠: 不要。新規アイデア生成ではなく、設計済み診断の実装である。

## 再現性・リスク

- seed policy: 乱数を使わない。source と artifact SHA、sample 順、分位点、境界定義を固定する。
- stochastic 処理の有無: なし。
- stochastic feature generation / augmentation / seed bagging の有無: なし。画像条件は exp005 の保存済み決定的 sampling を読む。
- 並列処理と乱数の関係: 逐次処理、乱数なし。
- CPU/GPU runtime と deterministic flags: CPU のみ、GPU 無効、internet 無効。
- train cache / test feature regeneration の SHA 記録方針: candidate file SHA を実ファイルで検査し、manifest に記録済みの candidate content SHA を入力 bundle SHA に含める。
- model manifest / prediction / submission SHA 記録方針: 2 prediction manifest、動画別評価、画像条件、出力表の SHA を記録する。model と submission は生成しない。
- Kaggle package bootstrap 確認方針: diagnostic Notebook、config、metrics を prepare し、固定 kernel source 3 件と CPU/internet metadata を検査する。
- leakage リスク: 条件と結果を同じ 2 胚で繰り返し見て補正を選ぶと外側評価も開発集合になる。分位点境界は反対側胚の exp005 条件だけで決め、今回の出力は説明的診断に限定する。
- CV/LB 不一致リスク: training data の胚 holdout 診断であり LB ではない。Public LB を推定しない。
- runtime / memory リスク: candidate cache は大きい。1 動画ずつ読み、全候補配列を route 全体で保持しない。
- 再現性リスク: Kaggle kernel version を取得するまでは正式な実行証拠が完成しない。実装時点の `metrics.json` status は `scaffold_completed` とする。
- 手法忠実性リスク: 動画単位条件は局所的な誤り位置そのものではなく動画の条件要約である。境界については candidate 位置の分布を明示し、局所 edge error の距離別評価と呼ばない。
- 過度な縮小 / proxy 化リスク: public 構成予測が未取得であることを 2 自前 route で代用せず、後続入力として残す。

## 受け入れ基準

- [x] 手法契約の input / target / output / loss / decode / context unit を実装前に固定した。
- [x] 実装区分と実験名が、保存済み予測の条件別診断を表している。
- [x] backlog の根拠、検証範囲、残る検証、親、差分、固定事項、最小検証、成功条件、停止条件、実行しないこと、判断履歴を移行した。
- [x] `config.yaml` の `lineage.hypothesis_id` と `lineage.backlog_candidate` が本書と一致する。
- [x] Jupytext round-trip、`validate-exp`、`check-exp`、実験固有 test が通る。
- [ ] Kaggle CPU で candidate cache を含む 2 route・199 動画の diagnostic が完走する。
- [ ] 5 条件の表、対応動画比較、失敗・NaN・有効件数、入力・出力 SHA、kernel version、Notebook 実行時間が揃う。
- [ ] 結果と限界をユーザーへ提示し、採否・完了の判断を受ける。
