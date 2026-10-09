# {{ EXPERIMENT_NAME }} 結果

## 仮説

TODO

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV/LB、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

設定とSHAをこのファイルへ独立した記録として転記しない。CV・Public LB・Private LBは`metrics.json`を数値の正本とし、比較や乖離の説明に必要な場合は、参照キーを添えて同じ値を本文で使用してよい。

## 実行証拠

- 比較対象: TODO
- `metrics.json` の参照キー: TODO
- `SESSION_NOTES.md` の実行記録: TODO
- 参照する生成物パス: TODO
- 評価の実施範囲と欠落: TODO（長い評価を分割した場合は予定・完了・失敗・未実施の単位と、比較条件が揃う範囲への参照）

## 解釈

TODO

## ユーザー判断

- 判断: 未判断
- 確認日時 / 依頼メッセージ: TODO
- 理由: TODO
- 今後の比較対象: TODO（据え置き / 更新 / 未決、対象パイプラインと理由。採用との区別は`docs/05_workflow.md`を参照）

## 終了後の結果照合（該当する場合）

終了後の結果照合を行った場合だけ記入し、未実施ならこの節は省略してよい。

- 観測日・対象submission ref・`metrics.json`の参照キー: TODO
- 当時の選択とPrivate LB公開後に分かったこと: TODO（当時の採否を上書きしない。未取得の対照と比較上の限界も示す）

## 次

後続の結果を受けた案内の更新と当時の判断の保存は、[AGENTS.mdの運用ルール](../../AGENTS.md#運用ルール)に従う。

TODO
