# exp023_synthetic_detector_teacher_audit 要件と実装方法

## 実験化の入口・引き継ぎ・承認

- 2026-09-20: ユーザーは[exp022](../exp022_synthetic_division_teacher_audit/)の完了を判断し、「固定公開検出器の候補・特徴に合成教師を接続できるか、小規模に診断」に進むことを承認した。本実験は直接承認の診断であり、元のdivision_triplets候補の学習実験化ではない。
- 親実験: exp022。先頭32時系列のGT中心では2,266分裂中2,193組を回収し、正例と同じ母の誤組が1,385/2,193母にあった。合成画像を固定検出器へ入力する部分は未検証だった。
- 根拠: [exp011の選定manifest](../exp011_public_detector_selection/assets/public_detector_selection.json)、[exp015の固定推論・特徴cache契約](../exp015_oracle_stage_limits/requirements.md)、[exp022の結果](../exp022_synthetic_division_teacher_audit/result.md)、[公開合成source](https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset)。公開モデル2つのcheckpoint SHAとTemporalUNet3D source SHAをconfigへ固定した。
- 固定するもの: exp011が採用した2モデルの重み、検出閾値0.965、8視点D4、secondary検出weight 0.80、primary特徴TTA・secondary原視点特徴、2時刻窓、母娘9 µm・娘間14 µm、GT対応7 µm。GTを見て閾値を調整しない。
- 変更するもの: exp022の正解node中心を、公開モデルが合成画像から実際に検出した中心に置き換える。各候補の固定画像特徴とGTの一対一対応を診断する。
- 実装上の未決事項: なし。学習に使う教師mask、loss、decode、実画像との混合は結果を見て判断する。

## 手法契約

- input: 公開合成sourceのmanifest・metadata・先頭2時系列の6×64³ uint16画像と完全なnode・edge・division。加えて採用済み公開checkpoint2つとTemporalUNet3D source。2時系列の選択はmanifest順で固定し、1つは低密度、1つは高密度の例になる。
- target / objective: 学習targetなし。検出中心と32チャネル×2系統の画像特徴が、完全合成系譜のどの正誤分裂候補へ対応するかを数える。
- output: 10窓の候補数、GT中心・分裂・組の回収、教師分類、候補特徴のschema・有限値・SHA、時系列別CSVとsummary/manifest。小さな窓別特徴NPZを保存する。
- loss / decode: 実装しない。固定モデルのtransformer、ILP、repair、公式scoreは実行しない。
- context unit: 同一合成時系列の隣接2時刻窓と、母候補・順序なしの娘候補2点。
- 実装区分: このリポジトリ内の管理用語ではfaithful。承認された「検出候補と特徴への教師接続診断」の範囲に従い、公開構成の検出・画像特徴抽出を固定して実行する。公開Notebook全体の追跡・復号を再現する実験ではない。
- この実験が判断できること: 合成画像で固定検出器の候補・特徴を得られるか、9/14 µmの組に完全系譜から信頼できる教師がどれだけ残るか。
- この実験が判断できないこと: 実画像での学習効果、独立CV、合成教師の重み、損失と復号、公式score、32時系列全体への一般化。

## 実装方法

- Kaggle T4で2時系列・10窓を実行する。画像は公開sourceですでにXY方向4倍縮小済み。実データのZarr `image_statistics.quantiles`はないため、同じ0.001/0.999 quantileを各6時刻全体から計算し、下限0でclampする。この前処理差を実行証拠に残す。
- 公開support datasetからTemporalUNet3Dだけをimportする。model checkpointの`unet.*`と`detect_head.*`をstrictに読み、両checkpoint/source SHAを照合する。学習済みtracker部は実行しない。
- 2時刻窓に8視点D4を適用し、両モデルの検出logitを各視点の逆変換後に平均する。secondary logitの平均・標準偏差をprimaryへ整合し、weight 0.80で混合する。確率0.965超のlocal maximaを3 µmのpool kernelで抽出する。primary特徴は同じ8視点の逆変換平均、secondary特徴は元視点から、候補座標を整数indexして得る。
- 各窓で検出候補と合成GT nodeを7 µm以内の一対一Hungarian対応で結ぶ。候補生成・正規化・検出閾値にGTを使わない。9/14 µmの順序なし組を生成し、3中心すべてがGTに対応した候補だけを、正しい分裂組、同母の誤組、通常継続母の組に分ける。未対応候補を含む組は別計数し、実データの教師と混同しない。
- GT中心・分裂・母内の正誤組をexp022の同じ2時系列の正解中心条件と比較する。候補特徴を窓別NPZへ保存し、有限値、shape、SHAを検証する。入力と出力のSHA、GPU推論秒数、peak memoryを記録する。
- CPU補助処理とT4 forwardのみ。active variant 1、model config 0、fold学習0、booster 0、control再学習なし、submissionなし。

## 探索幅とpivot判定

- 探索幅: 2時系列、固定0.965閾値、固定9/14 µm、GT対応7 µmの1条件。密度別・閾値別の選択をこのGTで行わない。
- pivot判定: 候補がほぼ出なければ、合成画像と固定検出器の入力・分布差を報告し、勝手に閾値を下げて学習へ進めない。候補と特徴が得られても、まず同母の正誤組数と実画像への転移課題を報告する。

## 再現性・リスク

- source manifest/metadata・各選択ファイル・checkpoint・TemporalUNet3D source・NPZ/CSV/summaryのSHAを照合。入力順と候補IDを固定し、T4の非決定性は同値性を主張しない。
- 合成sourceは分裂を意図的に多く生成し、texture・contrastに実画像との差がある。quantile属性がない前処理差も含む。公開モデルはtrain胚で学習しており、実データとの比較は独立CVではない。
- GT完全性は合成系譜内に限る。detectorがGTを見ずに作った中心の7 µm対応にも誤対応余地がある。未対応候補の組を一律に実データの確定負例へ転用しない。

## 受け入れ基準

- [x] Jupytext一致、validate-exp、check-exp、test-expが通る。
- [x] Kaggle T4で2時系列・10窓を完走し、公開checkpoint/source SHAと合成source SHAを照合する。
- [x] 検出候補・画像特徴の有限値、GT対応、正誤分裂組と通常継続組、入力・出力SHAと計算費用を記録する。
- [x] 結果と限界をユーザーへ示し、完了・学習設計の判断材料にする。

## 判断履歴

- 2026-09-20: ユーザーはexp022の完了と、本実験の小規模診断を承認した。

## 2026-09-20の規模拡張契約

- ユーザーは、検出後に残る教師量をより多くの合成時系列で確認し、実データの疎なGTで評価できる範囲を定めることを依頼した。新しい学習方式の承認ではない。初回version 1の2時系列結果は履歴として残し、同じ診断Notebook・同じ固定重み・閾値・幾何条件・GT対応条件で、exp022と同じmanifest先頭32時系列へ拡張する。合計160窓。分裂母数、幾何条件を満たすGT分裂、検出後の正例、同母の誤組を持つ正例母、誤組、通常継続組、未対応組を時系列別に集計する。
- 分布は時系列単位の合計と範囲・中央値で報告する。先頭2時系列と残り30時系列の差も記録し、単純な線形外挿を結果と呼ばない。exp022の同じ32時系列のGT中心候補と比較し、分裂ごとの正例回収が検出でどれだけ減るかを測る。GTは検出・特徴抽出・閾値選択に使わない。
- 実データ側はexp020/021の199動画・151既知分裂の集計から、既知分裂の候補回収、注釈から確定できる誤組、同母内の比較可能件数を確認する。未注釈組、第2娘不明の単一edge母、通常継続母の有無は未知とする。scoreの精度や公式指標を、この診断から推定しない。
- 追加実行は固定検出器の推論のみで、学習variant 0、model config 0、fold 0、booster 0、control再学習なし、submissionなし。初回2時系列の289.47秒から診断処理は約77分を見込むが、実測を正とする。Kaggle T4の週30時間以内・1回12時間以内を確認してからpushする。
- 追加の受け入れ条件: 32時系列・160窓完走、出力SHA・特徴有限値・分類内訳の検証、時系列別分布とexp022/020/021との条件差を明記し、ユーザーへ完了判断材料として示す。

- 2026-09-20: ユーザーは32時系列の実行と実データGT評価範囲の確認後、exp023の完了とcommit・pushを明示した。
