# exp035_velocity_features 結果

## 仮説

対象sampleを学習に使わないfixed first-pass trackerの予測履歴から直前の3次元移動ベクトルを求め、次候補とのずれと履歴品質をpair logitへ追加すると、固定画像特徴と現在位置だけを使う保存済みexp016より両胚の接続診断が改善する。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- CV、実験status、kernel情報、Kaggle Notebook実行時間、生成物SHA: [`metrics.json`](metrics.json)
- 実行コマンドと途中経過: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行証拠

- 比較対象: 保存済みexp016 fold model。controlは再学習しない。
- 比較単位: 同じ外側胚の隣接2-frame window、固定閾値0.5。
- 確認する指標: 既知edge recall、active pair内の教師負例予測、正解親1位率、分裂回収数、履歴coverage。
- 全graphへの進行条件: 両胚で既知edge recallが改善し、教師負例予測の増加が各胚5%以内、分裂回収数が減らないこと。
- Kaggle実行: `kentookumura/exp035-velocity-features-train` version 3、Tesla T4、internet無効。
- Notebook実行時間: 8,629.124秒。学習・pair診断本体は8,006.298秒。
- 生成物: velocity model 2個、対象sampleを除外した履歴生成model 4個。
- model manifest SHA-256: `20ea0e29df1de080764704e5a3f24433fa22c807e0a681c2d40cbdec33325842`。
- history manifest SHA-256: `3119186560b6a989b4b00bcb00b963a2c417012f2f9a20b7baad84efc92d8670`。
- 回収先: [`artifacts/train_v3/`](artifacts/train_v3/)。6個のmodel fileと2個のmanifestは記録済みSHA-256に一致した。
- 公式score: 未計測。pair診断gateが0/2胚で未達だったため、全graph推論を開始しなかった。
- submission: 作成しない。

## exp016との比較

| 評価胚 | 指標 | exp016 | velocity features | 差 |
|---|---|---:|---:|---:|
| 6bba | 既知edge recall | 96.9408% | 96.8837% | -0.0571ポイント |
| 6bba | active pair内の教師負例予測数 | 4,355 | 4,349 | -6（-0.14%） |
| 6bba | 正解親1位率 | 97.9235% | 97.8664% | -0.0571ポイント |
| 6bba | 分裂親回収数 | 48 / 108 | 41 / 108 | -7 |
| 44b6 | 既知edge recall | 94.7754% | 94.2794% | -0.4961ポイント |
| 44b6 | active pair内の教師負例予測数 | 1,341 | 1,255 | -86（-6.41%） |
| 44b6 | 正解親1位率 | 96.8442% | 96.8653% | +0.0211ポイント |
| 44b6 | 分裂親回収数 | 6 / 22 | 3 / 22 | -3 |

詳細な分母、loss、履歴coverage、予測SHAは [`metrics.json`](metrics.json) と各foldの [`fold_0_summary.json`](artifacts/train_v3/fold_0_summary.json)、[`fold_1_summary.json`](artifacts/train_v3/fold_1_summary.json) を正とする。active pair内の教師負例予測は疎な注釈に基づく診断であり、注釈にない実際の接続をすべて誤接続と断定する値ではない。

## 解釈

今回合意した判断基準では、velocity featuresはexp016より精度が上がらなかった。両胚で教師負例予測は減ったが、主条件の既知edge recallが低下し、分裂親回収数も減った。44b6の正解親1位率だけは0.0211ポイント上がったものの、既知edge recallの0.4961ポイント低下と分裂回収の半減を相殺しない。

したがって、事前契約どおり全graph推論と公式評価へ進まない。これは今回のfixed first-pass履歴、直前2点の移動、zero-initialized delta branchという表現に対する結果であり、時間情報を使う手法全体の否定ではない。pair診断は長い軌跡、最終復号、公式combined scoreの代用にはしない。

## ユーザー判断

- 判断: 未判断。
- 確認日時 / 依頼メッセージ: 2026-09-22の結果提示後に確認する。
- 推奨: 今回のvelocity表現は採用せず、全graphへ進めずに終了する。
- 理由: 2胚とも事前gateの既知edge recall改善と分裂回収数非減少を満たさず、通過は0/2胚だった。

## 次

ユーザーに今回の表現を不採用として実験を終了するか判断してもらう。判断までは `metrics.json` のstatusを `running` のまま維持し、commit・pushしない。
