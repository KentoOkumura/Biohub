---
name: kaggle-strategy
description: "ローカルの実験メモ、`experiment_summary.md`、`backlog/KAGGLE_DIRECTION.md`、`backlog/`、metrics、提出履歴、保存済み資料から Kaggle コンペ戦略を整理する。複数候補・実験を束ねる上位仮説の追跡、壁打ち結果や他skillが生成した候補のバックログ作成・更新・削除、次の実験、ロードマップ、CV/LB の一貫性、失敗パターン、優先順位付け、実験横断の整理を求められたときに使う。`backlog/KAGGLE_DIRECTION.md` の検証中の仮説・アイデアバックログ節と `backlog/` を変更する唯一のskillとする。新しい外部文献や過去解法の調査が必要なら、先に `kaggle-survey-papers` を使う。"
---

# Kaggle 戦略整理

単発の結果ではなく、複数実験から見える「流れ」を整理する。まずローカルの証拠を使う。新しい外部調査が必要な場合は、戦略を確定する前に `kaggle-survey-papers` を使うべきだと明示する。

実装区分と変更classには`docs/glossary.md`で定義したこのリポジトリ内の管理用語を使う。skill内で別の定義や表記を作らず、ユーザーへの説明では実際に変更する処理を先に具体的に示す。

## 手順

対象候補を指定した「実装してください」という依頼では、全体戦略の収集手順を先に実行しない。まず`backlog/<candidate>.md`だけを読み、直接参照される根拠、親実験の`requirements.md`、親実験の`config.yaml`、変更対象コードの順で`kaggle-review-exp`へ引き渡す。実験作成後に候補を削除するときだけ、移行先の`requirements.md`と`config.yaml`を確認し、`backlog/KAGGLE_DIRECTION.md`では対象候補と上位仮説の該当箇所だけを検索して更新する。全体索引、他候補、全体summary、最近の実験記録を事前に横断しない。

戦略全体の整理、次実験の提案、優先度の再評価を依頼された場合は、以下の収集手順を使う。

1. ローカル文脈を集める。

```bash
uv run python .agents/skills/kaggle-strategy/scripts/collect_strategy_context.py --root .
```

2. 収集結果から、情報量の多いファイルを読む。
   - `docs/surveys/README.md`の上位仮説・実験番号・種類・トピック別索引と、そこから選んだ関連レポート
   - コンペ概要、または存在する場合は `backlog/KAGGLE_DIRECTION.md`
   - `experiment_summary.md`
   - `SUBMISSIONS.md`
   - `**/SESSION_NOTES.md`
   - `backlog/KAGGLE_DIRECTION.md` の未着手索引から選んだ候補の `backlog/<candidate>.md`
   - legacyの`docs/experiment/*.md`、`docs/experiments/*.md`、または類似の実験ドキュメントが残る場合は、`docs/surveys/`に正がないか先に確認する
   - `**/metrics.json`

次の実験を提案するときは、少なくとも `experiment_summary.md`、`backlog/KAGGLE_DIRECTION.md`、`SUBMISSIONS.md`、最近の `experiments/*/SESSION_NOTES.md` を読む。

同梱collectorは正本を先に収集し、未着手索引では`P0`から`P2`の候補詳細を優先する。残りの枠は、実験番号が新しい順の`SESSION_NOTES.md`、`metrics.json`、`result.md`で埋める。優先度の定義と記録形式は`AGENTS.md`に従い、`P3`と`P4`の詳細は対象候補を検討するときだけ個別に読む。

3. 作業を提案する前に、安定したベースラインと信頼できるCVの有無、探索の進み具合、停滞の証拠、締切までの時間を具体的に整理する。固定した段階名へ当てはめず、これらの状態から次の作業を説明する。

### 壁打ち結果をバックログ化する場合

担当範囲、ID、優先度、設計状態、ユーザー判断、実験への移行・終了の規約は、[AGENTS.md の引き継ぎ規約](../../../AGENTS.md#仮説とアイデアバックログの引き継ぎ)を正とする。追加・更新・削除の前に該当する規約を確認する。

1. 同名候補、実装済み実験、閉じたbranchを検索し、関連する上位仮説と証拠を読む。
2. 他skillから受け取る場合は、候補本文に加えて根拠ファイル、採らなかった案、未決事項、追加または削除の理由を確認する。
3. `backlog/_TEMPLATE.md`を使って候補固有の比較を記録し、上記規約に従って候補詳細と索引を対応付ける。共通規約と複数候補に共通する根拠は正本への参照に留め、候補固有の観測と入力・教師・出力・損失・固定事項・停止条件を具体化する。
4. 実験への移行では、移行元の契約と判断履歴を移行先の`requirements.md`・`config.yaml`と照合してから、規約に従って索引と候補詳細を更新する。
5. 上位仮説を閉じる場合の調査レポート作成・索引更新は、[調査レポートの手順](../../../docs/surveys/README.md#作成完了手順)を使う。
6. 変更後に`task check-strategy-docs`を実行する。調査索引も更新した場合は`task validate-surveys`を実行する。

### 戦略メモをまとめる場合

収集した証拠に基づき、次を簡潔に整理する。

- 現在のフェーズと、フェーズ認識のずれ。
- route別の基準と現時点のベスト結果、信頼度、根拠ファイル。
- CV/LB の一貫性評価。
- 主な失敗パターンと、明確に効かなかったこと。
- 次に試す手堅い実験。
- 当たれば大きい高リスク実験。
- もう少し長いロードマップが有用なら、期待値の高い次の 2-4 実験。
- リスク管理: leakage、CV/LB 乖離、実行時間、提出回数制限。
- `backlog/KAGGLE_DIRECTION.md` の「アイデアバックログ」に、完了済み・実装済みの候補が残っていないか。
- 「検証中の仮説」の対応候補・対応実験と、各実験の`config.yaml`の系譜が一致しているか。
- 実験結果から出た次候補が backlog に反映され、既存候補も含めて優先度が見直されているか。

数式を含む文書を作成・変更した場合は、[AGENTS.md の数式規約](../../../AGENTS.md#markdown-と-notebook-の数式)に従い、変更したファイルの検証を行う。

## ルール

- すべての提案は、ローカルファイルのパス、Kaggle 公開情報、論文、または明示した仮定に結び付ける。
- 仮定は `Assumption:` としてラベルを付ける。
- 手法の中核機構を保った最小の反証可能実験を優先する。実装区分は`docs/glossary.md`に従い、`proxy`を忠実実装より優先しない。
- ユーザーが特定手法またはrepresentation changeを求めた場合は、既存コードの再利用率、GPUコスト、実装容易性より手法忠実性を優先する。忠実実装のコストが大きい場合は、無断で縮小せず選択肢としてユーザーに示す。
- 同じ親実験または機構familyの`parameter`、`add-only`、`selector-only`、`postprocess`変更が2件連続した場合、またはpositiveなoracle headroom / coverage / 誤差非相関性に対しend-to-end改善が得られない場合は、次の実験を確定する前に `kaggle-idea-forge` を使い、target、output、decode、context unitを変える案まで見直す。
- 次実験の提案には、少なくとも1件の target / output / decode / context unit を変える高upside案を含める。採用しない場合は、safe案を選ぶ根拠を証拠とともに書く。
- 実験ディレクトリは作らない。バックログを追加・更新する場合の `backlog/<candidate>.md` はこの制限の例外とする。ログやメモが不足している場合は、不足している証拠を明示しつつ、最小限の次アクションを示す。
- 次候補の追加と同時に既存候補の優先度も再評価する。候補の記録形式・設計状態・実験への移行は上記の引き継ぎ規約に従う。
