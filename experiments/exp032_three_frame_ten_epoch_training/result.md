# exp032_three_frame_ten_epoch_training 結果

## 仮説

exp027と同じ3時点trackerを2 foldで各10 epoch学習し、同じrunのepoch 1、保存済みexp016の2時点tracker、保存済みexp027の予測と、事前選択した各胚64窓で比較した。Colab CLIのTesla T4で学習と診断を実行した。画像特徴と候補は固定し、2時点対照は再学習していない。

## 実行証拠

- 学習: 2 fold × 10 epoch、計20 epoch。内部指標による選択は6bba評価のfold 0がepoch 7、44b6評価のfold 1がepoch 9。epoch計算時間の合計は20,735秒。学習用の入力cache 19,701窓のうち、正解nodeがない994窓を除外して18,707窓を使用した。
- 学習済みmodel manifest SHA-256: `5516f50f5787c1fc993bb7ec9ac958f30cf903c3e98a5358d39667d95c88b742`。診断summary SHA-256: `49b9a721c48dad0f7963905a5f84a361c297d7e06b79c25871ca0edb7abc4855`。
- SHA検証済み回収物: [training_summary.json](artifacts/colab_runs/ten_epoch_v1_cli/training_summary.json)、[model_manifest.json](artifacts/colab_runs/ten_epoch_v1_cli/model_manifest.json)、[diagnostic_summary.json](artifacts/colab_runs/ten_epoch_v1_cli/ten_epoch_diagnostic/diagnostic_summary.json)、[metrics.json](metrics.json)。train archiveは9,841,761 byte、diagnostic archiveは470,991 byte。大容量の予測配列はローカルへ保存しない。
- exp015の約4.15 GB cache、GEFF ZIP、exp027の保存済み予測136,560,816 byteはKaggleからColabへ直接取得した。GEFF exportは非公開CPU kernel `kentookumura/exp032-train-geff-export` version 1を使用した。Google Driveへの手動配置とNotebookの手動実行は行っていない。
- CV、公式graph score、LB、submissionは未計測。以下の数値は、選択済み各胚64窓の候補接続に対する部分診断である。

## 固定閾値0.5の部分窓結果

| 評価胚 | weight | 既知edge recall | 教師上の負例予測数 | 正解親1位率 | 有効な分裂母数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 6bba | 同じrunのepoch 1 | 91.76% | 21 | 97.51% | 1 |
| 6bba | 選択epoch 7 | 94.83% | 22 | 97.51% | 1 |
| 6bba | 保存済みexp016 | 97.32% | 23 | 98.47% | 1 |
| 44b6 | 同じrunのepoch 1 | 84.38% | 18 | 94.27% | 0 |
| 44b6 | 選択epoch 9 | 90.10% | 19 | 95.31% | 0 |
| 44b6 | 保存済みexp016 | 94.79% | 18 | 97.40% | 0 |

- epoch 1から選択weightへの既知edge recall差は6bbaで+3.07ポイント、44b6で+5.73ポイント。動画単位のpaired bootstrap 95%区間はそれぞれ+1.06〜+5.16ポイント、+2.29〜+9.41ポイント。
- 教師上の負例予測は両胚で1件増えたため、事前に定めた予備的改善条件は満たさない。選択weightはexp016より両胚で既知edge recallと正解親1位率が低い。教師上の負例は真の誤接続件数ではない。
- exp027のepoch 1とのstate SHAは6bba評価foldで一致、44b6評価foldで不一致。44b6の固定窓recall、教師上の負例予測数、正解親1位率は一致し、平均正解親確率の差は約2.64×10⁻⁹だった。ColabとKaggleの重み同一性は両foldで主張しない。
- 有効な分裂母数は6bbaで1、44b6で0であり、分裂性能は判断できない。10 epochすべての学習・内部指標は保存したが、固定128窓での推論はローカル回収したepoch 1と選択weightで実施した。

## 解釈

この設定では学習延長により既知edge recallは増えたが、教師上の負例予測数を増やさない条件を満たさず、保存済みexp016を上回らなかった。公式graph評価へ自動進行しない。公開画像weightの学習来歴を含むため、この部分窓結果を独立CVとして扱わない。

## ユーザー判断

実験の完了・採用・不採用はユーザー判断待ち。判断には [SESSION_NOTES.md](SESSION_NOTES.md) の実行時系列、[requirements.md](requirements.md) の事前条件、上記の比較と未計測事項を用いる。
