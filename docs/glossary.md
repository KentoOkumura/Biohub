# 用語集

| 用語 | 意味 | メモ |
| --- | --- | --- |
| CV | 交差検証スコア | ローカル検証の結果。 |
| LB | リーダーボードスコア | Kaggle の public/private スコア。 |
| OOF | Out-of-Fold prediction | 各行を、その行が属するfoldを学習に使っていないモデルで予測した値。 |
| Zarr v3 | chunk化したN次元arrayとmetadataをディレクトリ構造で保存する形式 | このコンペでは3D+time画像とGEFFの基盤形式。 |
| GEFF | Graph Exchange File Format | このコンペの正解tracking graphを保存する形式。nodeの時刻・重心座標とedgeを持つ。 |
| node | あるtimepointで検出された細胞 | `node_id,t,z,y,x`で表す。 |
| edge | 時間方向に同一細胞または分裂後の娘細胞を結ぶ有向接続 | `source_id,target_id`で表す。 |
| division | 1つの細胞から2つ以上の娘細胞へ分かれる事象 | 公式metricでは2本以上の出力edgeを持つ予測nodeを分裂候補として扱う。 |
| SHA | Secure Hash Algorithmによるhash値 | 内容同一性の証拠。algorithmと圧縮前後のどちらをhashしたかを明示する。 |
| `planned` / `running` / `debug_completed` / `scaffold_completed` / `failed` | 実験の実行状態 | `metrics.json`のstatusに記録する。 |
| `usable` | 派生実験や提出に使える状態 | ユーザーが判断する実験status。 |
| `completed` | 必要な検証と記録を終えた状態 | ユーザーが判断する実験status。 |
| `deprecated` | 履歴として残すが再利用しない状態 | ユーザーが判断し、可能なら代替実験を併記する。 |
| `discarded` | 候補として使わない状態 | ユーザーが判断する実験status。 |
| `leak-risk` | 検証リークの注意表示 | 採用・不採用・完了の判断ではない。 |
| `検討メモ・設計不可` | 結果や実装方針に影響する未決事項が残るbacklog状態 | このリポジトリ内の管理用語。 |
| `設計可能・実験化未承認` | 実験境界が揃い未決事項がないが、実験化は承認されていないbacklog状態 | このリポジトリ内の管理用語。 |
| `faithful` / `staged-faithful` / `proxy` | 参照手法をどこまで保持したかを記録する実装区分 | このリポジトリ内の管理用語。先に具体的な処理と省略点を説明する。 |
