# exp053_x138_postlink_division_score 結果

## 仮説

exp043の再接続・欠損補完後に残る第2娘候補について、母の直前と2娘の直後の接続・座標を採点に使うと、中央2時点だけでの判定と最終graphが改善するかを検証する。元の全動画ILPで前後edgeを同時選択する効果は扱わない。

## 記録の参照先

- 設定、系譜、再現性方針: [`config.yaml`](config.yaml)
- 候補診断、実験status、kernel、成果物SHA、CV/LB: [`metrics.json`](metrics.json)
- 実行コマンド、失敗と修正、提出時系列: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- 実装前の比較契約: [`requirements.md`](requirements.md)

## 実行証拠

- 比較対象: exp043の固定推論経路とPublic LB 0.950。exp053では分裂候補の採点と合法な追加edgeだけを変更した。
- Kaggle CPUでGEFFから接続・座標の候補表を生成し、Colab CLI CPUで4時点と前後遮断の対照を同じ候補・分割・学習設定で再学習した。`metrics.json`の`evidence.kaggle_train`、`evidence.colab_train`、`evidence.artifacts`に実行時間、件数、入力・モデルSHAを記録した。
- 外側の各胚10動画は学習・閾値選択から除外した。胚別のAverage Precision、閾値でのprecision・recall・予測件数は`metrics.json`の`evidence.candidate_diagnostics.outer`を正とする。
- ColabモデルはNotebook同梱の[`assets/division_model.json`](assets/division_model.json)。Kaggle公開test推論v1の[Notebook](https://www.kaggle.com/code/kentookumura/exp053-x138-postlink-division-score-inference)は完了。提出CSVは225842行で、提出前検証を通過した。competition submission refは`56663354`。Kaggle CLIでscoring complete、Public LB 0.953を確認した。比較対象のexp043は0.950で、差は+0.003。既存のexp042も0.953で同点。

## 解釈

外側の既知正例は両胚で合計12件と少ない。4時点採点器のAverage Precisionは44b6で前後遮断対照をわずかに下回り、6bbaでわずかに上回った。内部集合で固定した閾値では4時点採点器だけが両胚で既知正例を選んだが、これは疎いGEFF候補の局所診断であり、最終graphの公式指標ではない。公開画像モデルと座標headの学習来歴もあるため、外側動画を独立CVとは呼ばない。両胚のAverage Precision改善という事前の成功条件は満たしていない。

推論時は予測graphの親なし第2娘だけを対象にする。既に別の親へ接続された娘、欠けた娘点、前後edgeの同時付け替えは回収できない。追加後の短軌跡除去などはexp043の後処理が続くため、採点器が選んだ全edgeが提出graphへ残るとは限らない。 保存済みexp043 v2の公開test CSVとはnode・edgeの集合が完全一致しない。したがってPublic LB 0.953は提出パイプライン全体の比較として扱い、CSV差分だけから4時点情報の寄与を分離しない。exp043比の改善は確認できたが、元の全動画ILP共同選択や画像特徴の効果は未検証である。

## ユーザー判断

- 判断: 採用・完了。このリポジトリの実験statusは`completed`。採用対象はexp043の再接続後に4時点の接続・座標特徴で第2娘を採点する今回の提出パイプライン。
- 理由: 2026-09-29にユーザーが完了および採用を明示した。Public LB 0.953はexp043の0.950を0.003上回り、exp042の0.953と同点。胚別候補診断の混在、画像特徴と元の4時点共同選択の未検証は残る。

## 次

元設計の前後edge同時選択や画像特徴の寄与を検証する場合は、今回の採用結果とは別の比較として設計する。
