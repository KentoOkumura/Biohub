# 同じ評価単位の結果を比較する

`compare-evaluations` は、課題固有の評価器で計算済みの2つの指標表を照合し、同じ評価単位の差を出す。分類、回帰、検出、graphなどの予測形式には依存しない。入力は小規模なCSVとJSONで、Python標準ライブラリだけを使う。指標の計算、正解との対応付け、未知注釈の補完は行わない。

実験全体の代表CV・LB一覧には既存の `compare-exp` を使う。このコマンドは、例えば同じ動画や同じfoldにおける候補と対照の差を詳しく確認するときに使う。独立した交差検証をCVと呼び、公開モデルの学習来歴を含む条件付き評価や補助診断とは区別する。

## 入力の準備

各条件の評価結果を、1評価単位につき1行のCSVにする。IDは文字列として扱い、先頭のゼロも保持する。複数列を合わせたIDが必要なら、重複しないID列を評価器側で作る。複数指標を同じ行に置ける。

```csv
unit_id,group,score,error
sample001,group_a,0.8,1.2
sample002,group_b,0.7,1.5
```

各CSVと同じ実行から得た情報でmanifestを作る。以下がschema version 1の例であり、SHAと固定条件の値は実データの値へ置き換える。

```json
{
  "schema_version": 1,
  "table": {
    "path": "evaluation.csv",
    "sha256": "CSVファイル全体のSHA256を小文字64桁で記録する"
  },
  "unit_column": "unit_id",
  "group_column": "group",
  "metrics": {
    "score": "max",
    "error": "min"
  },
  "evaluation_kind": "conditional",
  "expected_unit_count": 2,
  "fixed_conditions": {
    "evaluation_set": "評価対象ID一覧のSHA256または固定した集合の識別子",
    "labels": "正解データと注釈の利用範囲を固定した識別子",
    "split": "分割定義のSHA256または固定した分割の識別子",
    "evaluator": "評価器コード・設定のSHA256または固定版",
    "aggregation_unit": "sample単位、指標の計算方法と重みを含む固定定義",
    "prediction_stage": "指標を測る出力段階の固定定義",
    "fixed_inputs": "比較で固定する入力のmanifest SHA256または固定定義"
  }
}
```

- `table.path` はmanifestからの相対パス。UTF-8のCSVを読み、実ファイルのSHA256を検証する。manifestそのもののSHAも出力に残す。
- `unit_column`、`group_column`、`metrics` に指定する列はそれぞれ異なる列とする。集団別比較が不要なら `group_column` を明示的に `null` にする。指標の方向は、大きい方がよい場合 `max`、小さい方がよい場合 `min`。
- `evaluation_kind` は `cv`、`conditional`、`diagnostic` など評価の意味を表す非空文字列。両側で一致させる。この文字列によって独立性を認証するわけではない。
- `expected_unit_count` は計画した評価単位数の正整数。両条件で同じ途中結果だけが残っていても、件数不足を検出するため実行前の計画値を記録する。
- `fixed_conditions` の上記7項目はすべて必須の非空文字列。追加項目も比較する。両側で欠けている、空欄、`null` の場合を一致とみなさない。`TODO`、`unknown`、`N/A` 等の未確認placeholderも拒否する。非該当の項目には、例えば `not_applicable: no split in this diagnostic` のように理由を明記する。
- 変更対象のモデル、学習法、候補生成などを固定条件へ無条件に入れない。生画像を固定して検出器を比べる場合と、同じ検出候補から復号処理を比べる場合では `fixed_inputs` の範囲が異なる。契約で固定すると決めたものだけを指定し、変更するものは実験の `requirements.md` に記録する。

正解、分割、入力等の固定条件は宣言である。このコマンドが内容を読み直して正しさを保証するのはCSVファイルとそのSHAの対応だけであり、列挙した別のSHAが実処理に使われたことまでは検証しない。元の入力manifestと評価器の実行証拠を併せて残す。件数と両側のID集合が一致しても、本来の評価集合から正しいIDを選んだ保証にはならない。失敗した単位を除いて件数を減らし、完了した比較に見せない。

## 実行

```bash
task compare-evaluations EXTRA_ARGS='--baseline experiments/expXXX_model/artifacts/control/manifest.json --candidate experiments/expXXX_model/artifacts/candidate/manifest.json --output-dir experiments/expXXX_model/artifacts/comparison'
```

`task` がない環境では同じ引数で `make compare-evaluations` を使う。`--output-dir` を省くとJSONの要約を標準出力に出し、ファイルは作らない。入力の指標表は書き換えない。

固定条件、評価種別、列の意味、指標の方向が一致し、CSVのSHA、IDの一意性、予定件数、両側のID集合、IDごとの集団、有限な指標値を確認できたとき、`comparison: comparable` として終了コード0を返す。条件不一致、欠損、重複、非有限数などがあると `comparison: not_comparable`、理由一覧、終了コード2を返し、指標差は出さない。これらは比較コマンドの判定値であり、実験statusではない。

出力先を指定した場合は次を作る。

- `comparison.json`: 照合結果、入力パスとSHA、評価種別、固定条件、指標別・集団別の診断集計。
- `paired.csv`: 比較成立時だけ、評価単位×指標ごとの対照値、候補値、差、改善方向に揃えた差、改善・悪化・同値の判定。

同じ出力先で再実行するとこの2ファイルを更新する。不成立時は前回の `paired.csv` を削除し、古い成立結果と混在させない。比較履歴を残す場合は実行ごとに別のサブディレクトリを使う。

例外として、manifestの破損・JSONキー重複等で入力表の場所を安全に特定できない場合は、入力を誤って削除しないよう出力先を一切変更しない。標準出力に `output not written` と比較不能の理由を示して終了コード2を返す。この場合、出力先に残るファイルは今回の結果ではない。また、出力ファイルが特定できた入力manifestまたはCSVと同じ場所になる指定も拒否する。

## 解釈と記録

`delta` は候補値から対照値を引いた差。`oriented_delta` は大きい方がよい指標ではそのまま、小さい方がよい指標では符号を反転する。`improved`、`worsened`、`unchanged` はその評価単位の数値の増減であり、統計的有意性や採否の判断ではない。`--tolerance` に非負の有限値を指定すると、その絶対差以下を同値に数える。既定は0で、実際の数値差や平均差は丸めない。指標間で単位が異なる場合、1つの許容値を一律に使う意味を確認する。

`unit_mean_*` は全単位を同じ重みで平均した診断値。集団別の件数や改善・悪化件数と併せて読む。公式評価の重み、非線形な集計、対応付け、全体での再計算が必要な場合、この平均は公式集計と一致しない。公式スコアを上書きせず、公式集計は元の評価器で別に保存する。部分注釈の未知部分を負例にする処理も加えない。

比較の実行証拠は既存の `record-exp --evidence` で `metrics.json.evidence` に参照できる。`record-exp` の使い方に合わせ、例えば次のように保存先を記録する。必要に応じ、実際に計算した `comparison.json` のSHAを別のevidenceへ記録する。

```bash
task record-exp EXP=expXXX_model EXTRA_ARGS='--evidence evaluation_comparison.path=artifacts/comparison/comparison.json'
```

比較コマンドは `metrics.json`、`result.md`、実験statusを変更しない。解釈、採否、次アクションの記録責務は [AGENTS.md](../AGENTS.md) に従う。
