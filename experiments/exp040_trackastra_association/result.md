# exp040_trackastra_association 結果

## 仮説

固定した検出候補と32次元画像特徴を使い、6時点のTrackastra対応モデルが保存済みexp016 primary単体より、両胚の既知接続を多く回収できるかを調べた。両runは同じ候補、同じ部分注釈、同じ外側胚の分割、4,096学習窓/epoch、10 epochs×2fold、同じ窓抽出スケジュールを使う。checkpointと確率閾値は学習側の内部検証だけで選び、外側胚では調整していない。固定公開画像重みの学習来歴にtrain動画が含まれる可能性があるため、外側胚の診断を独立したCVとは呼ばない。

実験契約は[`requirements.md`](requirements.md)、設定は[`config.yaml`](config.yaml)、数値・実験状態・SHAの正本は[`metrics.json`](metrics.json)、実行経過は[`SESSION_NOTES.md`](SESSION_NOTES.md)に記録した。

## 実行証拠

2026-09-24にColab CLIのTesla T4で、確率clampの数値上の不具合を修正したmasked BCEを用いて2foldを最初から学習した。旧重みは再開に使っていない。各foldの教師あり学習窓は10 epochs中に少なくとも1回選ばれ、全20 epochのSHA付き受領記録をローカルに回収した。内部検証損失で選んだcheckpointはfold 0がepoch 3、fold 1がepoch 8。Notebook本体の実測時間は27,072.53秒（約7時間31分）で、12時間gate内だった。

最終archiveは`artifacts/colab_runs/trackastra_logbce_v2/train_final.zip`（1,377,082,588 bytes、SHA-256 `25c10aa3fad768b75aa212940d5bd5d304fbda3ae9fcd22b04b641a7354b10e4`）。Colabの受領票とarchive全体、完了マーカー、model manifest、2モデル、20件のepoch受領記録をローカルで照合した。両胚のラベル付き予測ファイルも独立にSHA-256を再計算し、`metrics.json.train_stage.fold_evaluation`の値と一致した。Colabセッションは成果物のACK後に停止した。大きな予測ファイルと重みはGitへ保存しない。

旧run `trackastra_cap4096_v1`は確率clampを含む損失で25,447.14秒かけて学習したもので、archiveは`artifacts/colab_runs/trackastra_cap4096_v1/train_final.zip`（SHA-256 `e0f7bc1ac9a00c0fa25843255685ccaabc8d727f322c862009ce8a701d69ed98`）。旧runの数値と証拠は`metrics.json.evidence.reruns`に保持し、修正版の現行結果と区別する。Kaggle train version 1は費用gateで学習前に停止し、version 2は窓抽出のための時間計測のみを行った。Colabで本学習した経緯は`metrics.json.preflight`、`metrics.json.runtime_redesign`、`SESSION_NOTES.md`に残した。

## 外側胚の隣接ペア診断

Trackastraの閾値はfold 0の旧runが0.973269、修正版が0.979841、fold 1の旧runが0.985532、修正版が0.985548。いずれも学習側の内部検証で固定した。exp016 primary単体の閾値は0.5。分裂親回収は娘2個の接続をともに選んだ親の数。legacy mask負例予測には未注釈の候補も含まれ、真の誤接続件数とは解釈しない。既知負例予測は注釈に対応した誤った親候補を選んだ数である。

| 評価胚 | モデル | 既知接続回収 / 全件（再現率） | 正解親1位率 | 分裂親回収 / 全件 | legacy mask負例予測 | 既知負例予測 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 6bba（fold 0） | exp016 primary | 100,230 / 103,393（0.969408） | 0.979235 | 48 / 108 | 4,355 | 14 |
| 6bba（fold 0） | Trackastra旧run | 67,571 / 103,393（0.653536） | 0.886501 | 21 / 108 | 6,314 | 91 |
| 6bba（fold 0） | Trackastra修正版 | 63,799 / 103,393（0.617053） | 0.873492 | 17 / 108 | 6,090 | 96 |
| 44b6（fold 1） | exp016 primary | 17,959 / 18,949（0.947754） | 0.968442 | 6 / 22 | 1,341 | 4 |
| 44b6（fold 1） | Trackastra旧run | 8,958 / 18,949（0.472743） | 0.900100 | 2 / 22 | 611 | 1 |
| 44b6（fold 1） | Trackastra修正版 | 8,892 / 18,949（0.469260） | 0.900945 | 2 / 22 | 596 | 1 |

事前指定の進行条件は、両胚で既知接続再現率が改善し、正解親1位率と分裂親回収が非減少、負例予測の増加が5%以内であること。修正版の6bbaでは5条件すべて未達。44b6では負例予測の2条件のみ達成し、接続再現率・親順位・分裂親回収は未達。よって`metrics.json.train_stage.graph_eligible`は`false`であり、全graph推論へ進まない。公式graph scoreとPublic/Private LBは未測定で、Kaggle submissionはしていない。

## 解釈

旧runの低精度に対し、学習・教師・推論・評価の実装と保存済みの全ラベル付き予測を監査した。Notebookの共通62関数・クラスはテスト対象モジュールと構文上で一致し、胚の分割、候補ID、座標、親・子の行列方向、対照の再計算に取り違えは見つからなかった。旧予測を独立に集計した正例件数と回収件数も保存済みmetricsと一致した。

一方、旧masked BCEは確率を`1e-7`以上へclampしてから計算し、正しい親に極小確率を付けた場合に主損失の勾配が消える不具合があった。合成例で再現し、log確率とlog補確率から同じ損失を数値安定に計算するよう修正した。旧runの外側胚予測でclamp域より小さい既知正例は6bbaで95 / 103,393件、44b6で4 / 18,949件。学習中の該当件数は記録されていない。旧runを診断用の閾値0.5で読み直しても、既知接続再現率は6bbaで0.869798、44b6で0.891709と対照を下回り、負例予測は増加した。この再読出しは旧runの閾値だけが原因かを調べる監査であり、修正後モデルの精度測定ではない。

今回、損失を修正した新規学習では、既知接続再現率が旧runより6bbaで3.65ポイント、44b6で0.35ポイント低下した。44b6の正解親1位率は旧runからわずかに上がったが、対照を大きく下回る。したがって、確認した数値上の不具合を直すだけでは、この設定の低い隣接ペア精度は解消しなかった。学習窓抽出の影響や、モデル・部分教師・損失・候補確率の設計に残る問題はこの比較だけでは切り分けられない。

## ユーザー判断

この設定を現時点で採用することは勧めない。全graph推論は進行条件未達のため実施せず、公式scoreは未測定。実験の完了・採否はユーザー判断待ちで、`metrics.json`のstatusは`debug_completed`のままにする。
