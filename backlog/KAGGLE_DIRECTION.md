# Kaggle 方針

## 参照する前提

- 機械可読なコンペ設定、データパス、validation、submission、Kaggle runtimeは[`project.yml`](../project.yml)を正とする。
- コンペの公式情報と制約は[`docs/01_competition.md`](../docs/01_competition.md)、評価指標は[`docs/02_metric.md`](../docs/02_metric.md)、CV設計とリーク確認は[`docs/03_validation.md`](../docs/03_validation.md)、データ仕様は[`docs/04_data.md`](../docs/04_data.md)を参照する。
- 保存場所、記録責務、ユーザー判断、用語の規則は[`AGENTS.md`](../AGENTS.md)、コンペ固有の略語とリポジトリ内の管理用語は[`docs/glossary.md`](../docs/glossary.md)を正とする。
- このファイルは、現在の重点、比較基準、検証中の上位仮説、未着手候補の索引だけを管理する。

## 現在の重点

対象コンペを設定した後、公式評価指標とデータ仕様を確認し、再現可能な最初のbaselineとvalidationを確立する。

### 現行の比較基準

実験開始前のため、比較基準はまだない。数値は各実験の`metrics.json`、証拠への参照と解釈は`result.md`を正とする。

## アイデアバックログ

この節と`backlog/`の作成・更新・削除は`kaggle-strategy`の手順で行う。上位仮説は複数の未着手候補・実験を束ね、各`backlog/<candidate>.md`は1回の実験として実施できる具体的な設計を保持する。

### 検証中の仮説

| 仮説ID | 仮説 | 対応する未着手候補 | 対応する実験 | 残っている問い |
| --- | --- | --- | --- | --- |

### 未着手バックログ

未着手候補が0件の状態は正常であり、検証用のダミー候補は追加しない。

| 優先度 | 対応仮説 | アイデア | 短い要約 | 主な先行条件 / 依存 | 状態 |
| --- | --- | --- | --- | --- | --- |
