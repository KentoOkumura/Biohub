# exp020_division_triplet_candidates 要件と実装方法

この文書を実装前の契約とする。実行経過は SESSION_NOTES.md、実測値と解釈は result.md と metrics.json に記録する。

## 実験化の入口・引き継ぎ・承認

- 2026-09-19、ユーザーの「division_tripletsを実装してください」に対し、候補範囲・教師mask・loss・復号が未決であることを確認した。ユーザーは「推奨：候補と教師の診断から進める」を選択した。この実験はその診断段階の直接承認である。
- 元の候補は [division_triplets](../../backlog/division_triplets.md)。学習する組のスコアと復号は未実装なので、候補の契約を移行せず、未着手候補として残す。
- 関連する上位仮説は HYP-20260910-03。元候補を移行しない直接診断なので、configのlineageは両項目ともN/Aとする。この実験だけで支持・棄却は判断できない。次に教師mask、loss、組の得点と通常接続の競合方法を決める必要がある。
- 親実験は [exp015_oracle_stage_limits](../exp015_oracle_stage_limits/)。固定公開検出器と保存済み19,701 window cacheを使う。公開トラッカーとの学習比較基準は [exp016_frozen_image_encoder](../exp016_frozen_image_encoder/)。
- 根拠は [現在の学習方針](../../backlog/KAGGLE_DIRECTION.md#今後の学習方針)、exp015の [metrics](../exp015_oracle_stage_limits/metrics.json) と [requirements](../exp015_oracle_stage_limits/requirements.md)、[データ仕様](../../docs/04_data.md)。
- 固定するもの: 公開モデルの重み、検出候補、画像特徴、train 199動画、7 µmの時刻別一対一GT対応。検出・追跡・復号・評価器を更新しない。
- 変更するもの: exp015の固定中心候補から、隣接時刻の母と2娘の幾何学的な組候補を列挙して回収と教師候補を測る。
- 最小検証: GTを候補形成に使わず、全199動画・19,701 windowで3条件の組候補数と既知151分裂の回収を測る。胚別結果、対応中心3つの回収、既知分裂母での誤組、疎い注釈で確定できない通常継続を別々に記録する。
- 成功条件: cacheのsummaryとidentity SHA、全sample/window/frame coverageが一致し、199動画の3条件の表とmanifestを保存できること。これは精度改善の成功条件ではない。
- 停止条件: 入力SHA、sample/window/frame/GT件数、重複frameの候補、候補数guardが一致しない、NaN、GEFF読込失敗があれば停止する。
- 実行しないこと: 検出器・画像特徴抽出器・trackerの学習、疎い注釈の未対応候補を負例にすること、未知の第2娘を「いない」と扱うこと、GTによる候補選択、公式スコアやPublic LBの改善主張、Kaggle submission。
- この診断の実装上の未決事項はなし。後続の学習実験の候補上限、確定負例、loss、decodeは、この結果を確認してからユーザーと決める。

## 手法契約

- input: exp015の2-frame cacheにある候補IDと物理座標。GT対応と集計に限り主催者GEFFを読む。
- target / objective: 学習targetはない。既知分裂に対応する母と2娘の候補が、GT非依存の候補集合に含まれるかを診断する。
- output: 娘の順序に不変な組候補の件数、既知分裂回収、GTに対応した誤組数、胚別集計、入力・出力SHA。
- loss: なし。
- decode: なし。公開graphを変更しない。
- context unit: 候補形成は同一動画の隣接2 frame、GT対応は時刻単位、集計は動画と胚単位。
- 実装区分: proxy。この段階では組スコアの学習と通常継続との競合を省略するため、division_tripletsの精度仮説は検証できない。診断段階について2026-09-19のユーザー選択で承認済み。
- 変更class: representation。このリポジトリ内の管理用区分として、固定中心から母・娘1・娘2の組を表現する。
- この実験が判断できること: 固定中心の幾何範囲で既知分裂を何件回収し、候補数とGTから安全に利用できる組が何件あるか。
- この実験が判断できないこと: 組スコア学習による公式指標の改善、通常継続の確定負例数、独立CV、hidden testの性能。

## 候補と教師の診断方法

- 既存の公開safe division repairの9/14 µm、最終division geometry filterの10.5/8 µm、出力edge上限に対応する広い14/14 µmの3条件を、実行前にconfig.yamlへ固定する。各条件は母から各娘への最大距離、2娘間の最大距離を表す。条件をGTで選ばず、感度表として並列報告する。
- frame tの各母候補からframe t+1の候補を距離で抽出し、異なる2娘を順序不変の組として1回だけ列挙する。正解、現行candidate edge、学習済みscoreは候補形成に使わない。1母10,000組超は計算guardとして停止し、黙って上位だけ残さない。
- 既知中心との対応はexp015と同じ7 µm以内の時刻別linear sum assignmentを使う。母と2娘すべて対応した事象と、幾何候補まで回収した事象を分ける。151件との一致をassertする。
- 既知分裂母に限り、両娘候補が既知GTへ対応した別の組を確定した誤組として数える。GTに未対応の候補を誤組としない。
- GEFFは疎く完全注釈領域の仕様がない。出力edgeが1本の注釈を「第2娘なし」の確定継続と扱わず、注釈数だけを別記する。confirmed_continuationsは0と記録し、後続学習で使える負例が0と断定するものではない。
- cacheの全window identityを検証し、重複するframeの候補IDと座標の一致をassertする。per_sample.csv、summary.json、manifest.jsonを保存する。

## 再現性・リスク

- 乱数は使わず、sample/window/候補を固定順序で処理する。cache summary self-SHAとsorted identity SHA、GT内容bundle SHA、出力SHAを残す。
- exp015の公開モデルの学習来歴にtrain胚が含まれるため、結果は固定公開モデル下の条件付き診断であり独立CVではない。
- cacheはKaggle Notebook outputで約4.15GB。GPU学習0、model/config 0、fold 0、booster 0、control再学習なし。CPU Notebook 1本で順次読み、全配列を一括展開しない。
- Kaggle package bootstrapではsupport dataset内のoffline wheelを用い、internetを無効にする。
- 参照元の回収上限61/151はILP前edge graphによる値であり、今回の固定中心から作る組候補の結果と区別する。

## 受け入れ基準

- [x] Jupytext変換、validate-exp、check-exp、test-expが通る。
- [x] Kaggle CPUで199動画・19,701 windowの診断が完走する。
- [x] 既知分裂151件、各条件の回収と件数、両胚別の集計、SHA、失敗0件を記録する。
- [x] 結果と残る教師・loss・decodeの判断をユーザーへ提示し、診断実験の完了判断を得る。

## 判断履歴

- 2026-09-19: 元のdivision_triplets候補は設計不可のまま。ユーザーは候補と教師の診断を先に行う方針を選択した。学習候補そのものは未移行とする。
- 2026-09-19: ユーザーはexp020の診断完了を判断し、元のdivision_tripletsの教師根拠を調査してから学習を設計する方針を選択した。

## 実装方法

- Jupytext percent形式のdiagnostic sourceを正にしてNotebookへ変換する。imports、設定確認、入力検証、cache・GEFF読込、候補生成、集計、保存をセルで追える構成にする。
- GT非依存の組候補生成、7 µm一対一対応、既知分裂と安全な誤組の計数を同じNotebook sourceに実装する。全件coverageとSHAが不一致なら出力を結果として使わない。
- 対象実験テストでは娘順序、距離gate、1母の候補guard、matchingの一対一、疎い注釈の扱いを検証する。

## 探索幅とpivot判定

- 3つの事前固定した幾何範囲を感度表として測る。どれかをGTで学習・推論用に選択しない。
- 既知分裂の回収または安全な教師が不足した場合は、モデル学習へ進まず、候補範囲と教師の成立条件をユーザーへ示す。候補数・回収が足りても学習精度は実証されない。
- 元の分裂予測はtarget、output、decodeを変える高upside案である。この診断だけではその機構を実装しないため、別の小改善案へ無断でpivotしない。
