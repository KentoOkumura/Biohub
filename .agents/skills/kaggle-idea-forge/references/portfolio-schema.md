# Idea portfolio schema

新しい記録はUTF-8 JSONの`schema_version: "3"`を使う。field名はこのリポジトリ内の管理用語。候補数や分類枠を満たすために案を増やさず、未選択案は短く、選択した本命だけを具体化する。

```json
{
  "schema_version": "3",
  "task_summary": "課題、指標、推論時に使える情報",
  "evidence_cutoff": "参照できる証拠の時点",
  "allowed_sources": ["許可された資料またはデータの参照"],
  "data_exploration": "実施・既存結果の再利用・未実施の別、対象範囲、根拠への参照、限界",
  "selection_notes": "比較する案を選んだ理由と、他案を選ばない理由",
  "idea_cards": [
    {
      "id": "I01",
      "title": "案の名前",
      "hypothesis": "何を変えると何が改善すると考えるか",
      "basis": "観測・文献への参照、または明示した仮定",
      "changed_mechanism": "変更する入力、教師、出力、損失、推論方法など",
      "key_unknown": "この案について主要な未知事項"
    }
  ],
  "portfolio": [
    {
      "idea_id": "I01",
      "why": "この比較を優先する理由",
      "experiment": {
        "input_target_output_loss_decode": "入力、教師・予測対象、出力、損失、推論方法",
        "comparison": "比較対象、共通の検証条件、対象範囲",
        "preserved_mechanism": "初回比較で実装し、縮小しても省略しない仕組み",
        "evaluation_limitations": "この比較では判別できない効果や対象範囲",
        "decision_rules": "支持・反証・判断不能の判定と、それぞれの次の行動",
        "inference_contract": "推論時に存在する入力から必要な出力を生成する方法"
      }
    }
  ]
}
```

- 上記のroot fieldと各objectのfieldは必須。文字列は空にしない。
- `allowed_sources`は空でない文字列の配列。資料未提供なら空配列を許し、仮定からの発想であることを`basis`と`data_exploration`で説明する。
- `idea_cards`は1件以上でIDは一意。件数の上限、分類、初期案の件数、パラメータ変更案の数を機械的に制限しない。内容の幅と選択理由を別途レビューする。
- `portfolio`は件数を固定しない。IDは`idea_cards`内に存在し、重複しない。選択できない場合は空配列とし、`selection_notes`に理由と足りない情報を記す。
- 未選択案に`experiment`は不要。主要な方針が未定なら選択を保留し、未測定の精度だけを理由に実験契約を未定扱いにしない。
- 選択案の`experiment`には、必要な場合だけ`dependencies`（空でない文字列の配列、空配列可）と`resource_limits`（確定した制約を示す文字列）を追加できる。所要時間の予測値は要求しない。
- 予備診断を加える場合だけ、`experiment.diagnostics`へ`check`、`decision_if_pass`、`decision_if_fail`、`cannot_refute`の空でない文字列を持つobjectを入れる。空配列または省略を許す。診断そのものを必須にしない。
- 必要ならrootに`assumptions`（文字列の配列）、`closure_ledger`、`rejected`（objectの配列）を追加できる。全案に非該当項目を埋めない。
- validatorは型、必須項目、ID参照を確認する。データ観測の真偽、仮説の重要性、比較で中核を保てるか、科学的な反証の妥当性は判定しない。

## 旧版との互換性

validatorはv1（version省略を含む）とv2の既存形式も、その当時の必須項目・件数・分類条件で検証する。旧記録の`compute_estimate`などは互換性のために読み取るだけで、新しい案の記録には使わない。凍結した評価の入力・出力をv3へ自動変換しない。旧版の検証条件は同梱validatorの`validate_legacy`に保持する。
