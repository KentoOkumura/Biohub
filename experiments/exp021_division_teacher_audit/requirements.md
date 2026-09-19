# exp021_division_teacher_audit 要件と実装方法

## 実験化の入口・引き継ぎ・承認

- 2026-09-19: ユーザーはexp020の診断を完了と判断し、division_tripletsの教師を調査する方針を選択した。主催者の疎注釈仕様、baseline loss、評価器の分裂FP条件を確認した。
- 2026-09-19: ユーザーは「推奨：負例候補を追加診断し、教師を確定する」を選択した。本実験はその直接承認による追加診断であり、元のbacklog候補自体は未移行とする。
- 親実験: [exp020](../exp020_division_triplet_candidates/)。関連する上位仮説はHYP-20260910-03だが、直接診断のためconfigのlineage.hypothesis_idとbacklog_candidateはN/Aとする。この実験だけで上位仮説を支持・棄却できない。
- 根拠: [主催者のデータ仕様](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data)、[主催者の評価仕様](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)、[保存した主催者baseline loss](../exp002_unet3d_expandable_segments/official_source/scripts/train_unet_transformer.py)、[exp020結果](../exp020_division_triplet_candidates/result.md)。
- 固定するもの: exp015のcache SHA、199動画・19,701 window、GTへの7 µm一対一対応、母娘9 µm・娘間14 µmの候補生成、公開検出器の重み、両胚別集計。幾何範囲をこの診断のGTで選び直さない。
- 変更するもの: 候補のうち、GTで母が対応した組について、既知分裂正例、既知分裂母のGT対応誤組、別母から注釈edgeがある娘を含む組、単一出力edgeの母に既知娘を含む組を別々に数える。
- 追加readout: 誤組をGT母の出力次数0・1・2で分け、100正例母のうち同じ母に誤組がある数も報告する。
- 実装上の未決事項: なし。学習の教師mask、loss、decode、候補上限は診断後にユーザーと決める。

## 手法契約

- input: 固定中心候補のID・物理座標、train GEFFのnode・edge、事前固定した幾何条件。
- target / objective: モデル学習のtargetはない。候補をGT対応の有無と既知edgeの関係で分類し、教師に利用できる件数と重複を測る。
- output: 199動画ごとの各分類件数、両胚別総数、候補数、入力と出力のSHAを含むJSON・CSV。
- loss / decode: 実装しない。学習による公式指標の改善はこの診断では判断しない。
- context unit: 1動画の隣接2時刻と、その母・2娘の順序に依存しない1組。
- 実装区分: このリポジトリ内の管理用語ではproxy。組スコアの学習と既存接続との競合を省略するため、division_tripletsの効果は検証できない。診断範囲はユーザーが承認済み。
- この実験が判断できること: 現行固定候補で、GTの既知edgeから直接支持できる正例と誤組の件数、単一edge母における未確定な組の件数。
- この実験が判断できないこと: 注釈されていない第2娘の実在、推論時の真の偽陽性率、公式scoreの改善、独立CV、学習に使う最適な損失・復号。

## 候補の分類

1. 既知分裂正例: GT母に異なる2娘の注釈edgeがあり、3中心が一対一対応し、その組が候補集合にある。
2. 既知分裂母のGT対応誤組: 同じ母に既知の2娘があり、候補の両娘がGT nodeに対応するが、正しい娘2点の組と異なる。exp020の4件を再現する。
3. 別母の注釈edgeがある娘を含む組: GTへ対応した母とは異なるGT母から注釈edgeを受ける娘が候補の少なくとも1点にある。GT edgeの正しさと1娘1母・mergeなしの仮定の下で、その組は不整合。既知分裂母の誤組との重複、両娘GT対応の有無を別記する。
4. 単一edge母の既知娘を含む組: 母のGT出力が1本で、その既知娘に対応する候補を含む組。公式評価で予測forkがFPとなる可能性があるが、別の真の娘が未注釈の可能性を除外できない。生物学的な負例として扱わない。

分類2と3はGTの正しいnode/edgeと7 µm対応に条件付けた誤組であり、分類4は未確定とする。GTに対応しない母・娘、GTの境界、非隣接edgeは一律負例にしない。1候補が分類2と3に重なる場合はunionを重複なく数える。GTの複数入力edge、非隣接edge、同一時刻の重複IDを見つけたら停止する。

## 実装方法

- exp020のself-contained Jupytext診断Notebookを基に、9/14 µmのみを実行する。GEFFのincoming edgeを作り、候補ごとに上記分類の条件を判定する。199動画の各sampleで件数を出し、両胚別に合算する。
- 実験固有テストで娘順序不変、正例と誤組の排他、別母edgeの条件、単一edge母を確定負例にしないこと、分類重複のunion、merge/非隣接GT edgeの停止を検証する。
- Kaggle private CPU Notebookで199動画・19,701 windowを完走し、exp020と同じGT bundle SHA、候補8,807,617件、既知分裂151件、正例100件、厳格な誤組4件を再現する。差異が出たら入力・実装を調査して停止する。
- 対照の公開検出器、画像モデル、トラッカーは再学習しない。実行予定はCPU diagnostic Notebook 1本、active variant 1、model/config 0、fold学習0、booster 0、control再学習なし。submissionはしない。
- 成功条件: 各分類と重複の件数、両胚差、SHA、教師として使う前提と未知部分を記録できること。停止条件: 入力SHA不一致、GT構造の不整合、exp020再現失敗、Kaggleでの実行失敗。

## 再現性・リスク

- 乱数と並列処理を用いず、sample・window・candidate IDをソートする。cache identity、GT bundle、表・summaryのSHAを保存する。deterministic anchorは予測モデル・submissionの意味では主張しない。
- 公開画像モデルはtrain胚由来なので本診断は独立CVではない。GTは診断集計だけに使い、推論用の候補生成や条件選択には使わない。
- 候補8.8百万組の走査はCPUで行う。1動画ずつ処理し、候補全体を保存しない。
- 公式評価器は局所の前後時刻も使うため、単一edge母の組を直ちに確定FPと呼ばない。別母edgeに基づく誤組もGTの誤注釈、中心対応誤り、mergeなしの仮定に依存する。

## 受け入れ基準

- [x] Jupytext一致、validate-exp、check-exp、test-expが通る。
- [x] Kaggle CPUで199動画・19,701 windowが完走し、exp020の候補数・正例・厳格誤組を再現する。
- [x] 別母edgeによる誤組の件数、厳格誤組との重複、単一edge母の未確定な組を両胚別に記録する。
- [x] 生成物SHA、実行証拠、教師の前提と限界を記録し、ユーザーへ次の学習設計の判断材料を示す。

## 判断履歴

- 2026-09-19: ユーザーはexp020の完了を判断し、教師の根拠を調べる方針を選択した。
- 2026-09-19: ユーザーは負例候補の追加診断を選択した。未注釈組の一律負例化と単一edge母の確定負例化は承認されていない。

- 2026-09-19: ユーザーはexp021の診断完了を判断し、元のdivision_tripletsは追加の教師を探してから学習する方針を選択した。学習方式・実験化は未承認。

## 探索幅とpivot判定

- この実験ではexp020の9/14 µm条件だけを再利用し、GTを見た候補範囲の探索をしない。分類ごとの件数不足なら教師を推測で補わず、学習設計の選択肢をユーザーに提示する。
- 親exp020も診断であり、連続した小改善学習ではない。元のdivision_tripletsはtarget・output・decodeを変える候補で、その実装可能性を調べている。
