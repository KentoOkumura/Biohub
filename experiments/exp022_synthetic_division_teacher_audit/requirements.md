# exp022_synthetic_division_teacher_audit 要件と実装方法

## 実験化の入口・引き継ぎ・承認

- 2026-09-19: ユーザーはexp021の完了を判断し、元のdivision_tripletsは追加の教師を探してから学習する方針を選択した。公開資料調査後、提示した「推奨：合成系譜を小規模に診断する」を選択した。本実験は直接承認された教師診断であり、元のbacklog候補の学習実験化ではない。
- 親実験: [exp021](../exp021_division_teacher_audit/)。関連する上位仮説はHYP-20260910-03だが、直接診断のためconfigのlineage.hypothesis_idとbacklog_candidateはN/Aとする。本実験だけで上位仮説を支持・棄却できない。
- 一次資料: [参加者の公開説明](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103)、[公開Notebook](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset)、[コンペ規則](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules)。2026-09-19にKaggle CLIで取得したNotebook source SHA、manifest SHA、metadata SHAをconfigに固定した。公開Notebookのcodeは生成処理の参照であり、再実行はしない。
- 固定するもの: source版、manifest先頭32時系列、母娘9 µm・娘間14 µm、native座標の物理尺度、娘順序に不変な組、CPU Notebook、internet無効。exp021との実データ比較は既存の件数と条件だけを用い、実データのGTを合成データの候補制御に使わない。
- 変更するもの: exp021の疎GTを、公開合成系譜の完全なnode・edge・division配列へ置き換え、正例、同じ分裂母の誤組、通常継続母からの誤組を区別する。
- 実装上の未決事項: なし。合成教師を学習へ採用するか、loss、decode、固定検出器との接続は本診断後に判断する。

## 手法契約

- input: 公開Notebook outputのmanifest、metadata、先頭32個の`seq_XXXX.npz`。各配列は`nodes`（時刻・native zyx座標・継承されるlineage id）、`edges`（node配列の行index）、`divisions`（分裂母node index）、`voxel_um_pooled`。`volumes`はschemaだけ確認し、画素は読まない。
- target / objective: 学習targetはない。完全な生成系譜に条件付け、幾何候補が正しい分裂組を含む率、同じ母の誤組、非分裂母の誤組の件数を測る。
- output: 時系列ごとのCSV、合計・分布・SHAを持つJSONとmanifest。
- loss / decode: 実装しない。組スコア、接続選択、公式指標は測らない。
- context unit: 1時系列の隣接2時刻で、母1点・順序なしの娘2点からなる組。
- 実装区分: このリポジトリ内の管理用語ではfaithful。承認された完全合成系譜の小規模教師診断を実装する。32時系列以外、固定公開検出器・画像特徴の回収、実画像への転移と学習効果は診断対象外である。
- この実験が判断できること: 合成系譜の配列が正しい構造を持つか、9/14 µmでの教師件数と同一母内の比較がどの程度成立するか。
- この実験が判断できないこと: 実画像での真の非分裂ラベル、固定公開検出器での候補回収、合成画像の見た目の一致、公式scoreの改善、独立CV。

## 実装方法

- manifestとmetadataのSHA、sourceの公表件数、先頭32ファイルのsizeとSHAを照合する。配列のshape/dtype、node時刻、隣接edge、一娘一母、時刻末端以外の各母の出力1または2本、`divisions`と出力2本の完全一致をassertする。lineage idは娘の識別に使わない。
- native座標を`[1.625, 0.40625, 0.40625]` µm/voxelで物理座標へ変換する。全nodeを候補中心として各母から9 µm内の娘を列挙し、娘間14 µm以内の順序なし組を生成する。最大組数guardを設定する。
- 母の出力2本なら真の娘2点の組を正例、それ以外を同じ母の誤組とする。出力1本の母の組は合成世界内の非分裂誤組とする。実データの単一edge母へこの定義を流用しない。
- 真の分裂すべての幾何回収、正例母のうち同母誤組を持つ数、通常継続母の候補組数、母娘距離と娘間距離の分布、両胚を含むexp021の既存件数との単位差を記録する。
- NotebookはJupytext percent形式から生成するself-containedなCPU診断とし、公開生成コードを実行せずoutputを読む。CPU diagnostic Notebook 1本、active variant 1、model/config 0、fold学習0、booster 0、control再学習なし、submissionなし。

## 最小検証と停止条件

- 成功条件: Kaggle CPUで32時系列の構造検証と教師件数が完走し、source版、入力・出力SHA、正例・誤組・通常継続、分裂回収、母内の比較可能件数を記録できること。
- 停止条件: source SHAやschemaの不一致、GT構造の矛盾、Kaggle inputが取得不可、幾何候補の爆発、CPU実行失敗。値が乏しい場合は教師を推測で増やさず、ユーザーへ設計判断を示す。
- 実行しないこと: 合成組をBiohub実画像の確定負例と呼ぶ、合成sourceを再生成・再学習する、GPUで固定検出器を走らせる、loss・decode・submissionを実装する、未注釈実組を一律負例にする。

## 再現性・リスク

- manifest順の最初の32時系列を機械的に選び、乱数・並列処理なし。source metadata・manifest、選択ファイル、CSV・JSONのSHAを保存する。deterministic anchorは学習modelやsubmissionの意味では主張しない。
- 公開合成データはBiohubの実画像から一部幾何・運動・撮像統計を校正し、分裂率を意図的に上げている。作者もtexture・contrast差を明示した。32時系列の教師数は転移可能性の証明にならず、実画像の未知部分の教師へ流用しない。
- 全nodeは生成系譜上の正解だが、固定公開検出器が見つける候補とは異なる。仮に9/14 µmで高回収でも、実際の固定特徴入力と学習の成立は未確認。

## 受け入れ基準

- [x] Jupytext一致、validate-exp、check-exp、test-expが通る。
- [x] Kaggle CPUで32時系列を完走し、公開sourceのschema・manifest・metadata SHAと完全系譜の構造を検証する。
- [x] 正例・同母誤組・通常継続誤組、分裂回収、距離分布、source版、実行証拠と限界を記録する。
- [x] ユーザーへ診断結果と次の学習設計の判断材料を示す。

## 判断履歴

- 2026-09-19: exp021の結果を踏まえ、ユーザーは追加教師を探す方針を選択。
- 2026-09-19: ユーザーは公開合成系譜の小規模診断を選択。元のdivision_triplets学習方式は未承認のまま。

## 探索幅とpivot判定

- 探索幅: 先頭32時系列・母娘9 µm・娘間14 µmの1条件のみ。閾値調整、再重み付け、学習器探索は行わない。
- pivot判定: 完全系譜の教師件数と分裂回収を報告し、合成教師を実画像学習へ試すかはユーザーが後続実験で判断する。分布差や固定検出器との接続は残る問いとして提示する。

- 2026-09-20: ユーザーはexp022の完了と、固定公開検出器の候補・特徴に合成教師を接続できるかを小規模に診断する次の作業を承認。
