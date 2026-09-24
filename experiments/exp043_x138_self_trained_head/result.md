# exp043_x138_self_trained_head 結果

## 仮説

公開x138の画像モデルとトラッカーを固定し、検出点の周辺特徴から自前学習した3軸の座標補正headで、既知中心との距離を縮められるか確認した。作者の追加checkpointの再現ではない。

## 記録の参照先

- 設定と入力版: [config.yaml](config.yaml)
- 数値、SHA、実行status: [metrics.json](metrics.json)
- 実行と失敗復旧の経過: [SESSION_NOTES.md](SESSION_NOTES.md)
- train version 2の[動画別検証](artifacts/train_v2/coordinate_validation.json)、[head manifest](artifacts/train_v2/self_trained_head_manifest.json)、[学習receipt](artifacts/train_v2/self_trained_head_train_receipt.json)、推論version 2の[receipt](artifacts/inference_v2/self_trained_x138_receipt.json)、[run stats](artifacts/inference_v2/run_stats.csv)

## 実行証拠

固定公開画像モデルの特徴をtrain 20動画から取得し、主催者GEFFの既知中心と10,894点を一対一に対応付けた。5分割では各foldの検証動画をhead学習から除外し、補正なしと補正ありの平均物理距離を比較した。

| 胚 | 動画 | 対応点 | 補正前 | 補正後 | 改善した動画 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `44b6` | 10 | 3,367 | 1.563 µm | 1.452 µm | 7/10 |
| `6bba` | 10 | 7,527 | 1.937 µm | 1.545 µm | 10/10 |

全対応点を重み付けした平均距離は1.821→1.516 µmで16.8%減少した。一方、`44b6`の3動画では悪化し、最大の悪化は`44b6_66f9292d`の2.063→2.369 µmだった。

最終headは20動画すべてで学習し、`state_dict`、特徴平均、特徴標準偏差を保存した。SHA256は`metrics.json`の`evidence.artifacts.model_shas`を正とする。private Kaggle Dataset `kentookumura/exp043-self-trained-coordinate-head`からSHAを検査して読み込んだ推論Notebook version 2は、T4・internet無効で公開test 4動画を完走した。`submission.csv`は225,875行で、正のCSV validatorが重複ID0、欠損0、無限大0を確認した。CSV SHAは`metrics.json`の`evidence.artifacts.submission_sha`を正とする。2026-09-24に推論Notebook version 2をcompetition submissionとして提出した。ref `56508119`は`COMPLETE`となり、Public LBは0.950、Private LBは未表示。提出履歴は[SUBMISSIONS.md](../../SUBMISSIONS.md)を参照する。

train Notebook version 1はhead学習と公開test CSV生成後、train動画の診断ログが公開test用の診断セルへ混入して停止した。学習Notebookをhead保存と検証までに修正したversion 2はKaggleで正常終了し、20動画・10,894対応点の評価JSON、manifest、重みのSHAがversion 1の学習成果物と一致した。train version 2の重みSHAは、正常終了した推論Notebook version 2が使用した重みSHAとも一致する。version 1の失敗は実行履歴として残す。

## 解釈

提出ref `56508119`のPublic LB 0.950は、過去の[exp013](../exp013_public_notebook_replay/result.md)の0.944より0.006高い。ただし推論構成全体が異なり、自前headだけの寄与を切り分ける対照提出はない。採点監視開始から初回の完了確認までは384分で、Notebook自体の実行時間は取得できていない。

既知中心との位置誤差は両胚で平均改善したが、公開画像モデルはtrain集合を学習に使っている。この動画分割はheadに対する条件付き評価であり、画像モデルを含む独立した交差検証ではない。中心距離の改善だけから、接続・分裂を含む公式スコアの改善は判断できない。公開testの4動画も公式のhidden test性能を示さない。作者の重みと学習条件は未取得なので、作者の0.953を再現したとは扱わない。

## 今後の実験との比較

[metrics.json](metrics.json)の`evidence.coordinate_validation`に、20動画・10,894対応点の全体、胚別、5-fold別、動画別の件数と補正前後の平均中心距離を記録した。全体では1.82144→1.51618 µm、対応点加重平均で16.76%改善し、20動画中17動画で改善した。fold別値は保存済み動画別平均と対応点数から再計算した値で、Notebookの再実行結果ではない。選択動画、fold割当、教師対応半径7 µm、入力の固定画像モデル、checkpointと予測のSHAも`metrics.json`と[config.yaml](config.yaml)に記録されている。

将来の座標補正headを直接比較する場合は、同じ20動画、固定画像モデルと検出候補、教師との一対一対応規則、fold割当を維持して、対応点数と動画別・胚別・全体平均を比較する。検出候補や対応点が変わる実験では、評価対象自体が変わるため中心距離の平均だけで優劣を決めない。Public LB 0.950は推論全体の比較指標であり、head単体の効果を示すものではない。

点ごとのOOF予測はファイルとして保存されておらず、内容SHAのみ残る。点ごとの誤差分布、同じ対応点での差分、接続・分裂の動画別指標、採点時の動画別scoreは未取得である。これらを使う将来の比較には、同じ入力条件で新たに保存・評価する必要がある。

この実験では公開トラッカーの重みを更新していない。従来のトラッカー学習評価である[exp016](../exp016_frozen_image_encoder/result.md)や[exp025](../exp025_frame_self_attention/result.md)の隣接2フレームの既知edge recall、positive pair precision、false-positive pair数、分裂親の回収率、およびexp016の全199動画graphに対する公式指標とその胚別成分は、exp043では測定していない。したがって、この実験の採用は自前の座標補正headを組み込んだx138推論に対する判断であり、トラッカー学習結果の比較基準ではない。今後exp043の構成でトラッカーを学習する場合は、同じx138入力・候補・後処理で公開トラッカーを評価して対照とする。exp016は指標と評価手順の参考に留め、旧構成の数値を直接の対照にはしない。

## ユーザー判断

- 判断: 採用・完了。座標補正headを組み込んだx138推論の比較結果として扱う（このリポジトリの実験status `completed`）。トラッカー学習の比較基準とはしない。
- 理由: ユーザーが2026-09-24に採用と完了を明示した。条件付き中心誤差とPublic LB 0.950を確認した。Private LBとhead単独の寄与は未確認のまま残す。

## 次

別実験では上記の入力・評価条件を明示して比較し、条件が異なる指標は同じ基準値として扱わない。
