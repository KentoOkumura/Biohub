# exp048_x138_primary_past_feature_attention 結果

## 仮説

直前フレームの候補画像・位置特徴を親・娘の組ごとにprimary pair MLP内部で参照し、primaryとattentionを共同学習すると、公開primaryおよび同条件でprimaryのみ再学習した対照より両胚の既知接続が改善するかを調べた。

固定画像モデル・座標head・secondary・pair入力は親実験と同じ。外側は胚別2foldで、各foldの同胚8動画を学習、2動画をcheckpoint選択、反対胚10動画を評価に使った。各学習条件3 epochで、判定閾値は事前設定の0.48に固定した。公開画像モデルの学習来歴があるため、この外側評価を独立CVとは呼ばない。

## 実行証拠

[train Notebook](https://www.kaggle.com/code/kentookumura/exp048-x138-primary-past-feature-attention-train) のversion 2は `KernelWorkerStatus.COMPLETE`。20動画のcapture、2条件×2foldの学習とpair評価を完了し、4 checkpointを保存した。取得したmodel manifest・pair指標と4 checkpointのSHAはすべて一致した。captureは2,502秒、学習とpair評価は9,023秒、Notebook全体はKaggleログ最終イベントまで12,131秒（約3時間22分）。アカウントの週GPU残量は実行前35.84時間、実行後32.47時間で、差は3.37時間。ただし割当残量はアカウント共通である。

K=8の既知親保持率は44b6で3,275/3,275、6bbaで7,181/7,183。公開primaryとの初期logit最大絶対差は8.58e-6で、事前条件1e-4以内。version 1は最大pairを全窓の代理にした費用見積もりで学習前に停止した。version 2では窓サイズ別の実測に修正し、モデル・教師・分割・epoch・閾値は変更していない。出力の数値とSHAは [metrics.json](metrics.json)、時系列は [SESSION_NOTES.md](SESSION_NOTES.md) を正とする。

## 胚別のpair評価

既知接続の回収数は分母に対する値。active-pair errorsは同じ候補pair集合での誤り数。分裂親の全娘回収数は既知の分裂親に対する値。

| 外側胚 | 条件 | 既知接続の回収 | active-pair errors | 分裂親の全娘回収 |
| --- | --- | ---: | ---: | ---: |
| 44b6 | 公開primary | 3,162 / 3,302 (95.76%) | 383 / 2,215,209 | 1 / 2 |
| 44b6 | primaryのみ再学習 | 3,144 / 3,302 (95.22%) | 363 / 2,215,209 | 1 / 2 |
| 44b6 | 履歴attention付き共同学習 | 3,066 / 3,302 (92.85%) | 408 / 2,215,209 | 1 / 2 |
| 6bba | 公開primary | 7,016 / 7,275 (96.44%) | 627 / 3,243,792 | 1 / 11 |
| 6bba | primaryのみ再学習 | 6,962 / 7,275 (95.70%) | 637 / 3,243,792 | 1 / 11 |
| 6bba | 履歴attention付き共同学習 | 6,944 / 7,275 (95.45%) | 658 / 3,243,792 | 0 / 11 |

両胚とも分母は3条件で一致した。履歴attention付き共同学習はprimaryのみ再学習と比べ、44b6で既知接続が78件減り、active-pair errorsが45件増えた。6bbaでは既知接続が18件減り、active-pair errorsが21件増え、分裂親の全娘回収も1件減った。公開primaryと比べても両胚で回収数が少なく、active-pair errorsが多い。事前に定めた両胚の全graph進行条件は満たしていない。

同じ共同学習重みで過去入力を全maskした診断では、44b6が3,073件回収・404 errors・分裂1/2、6bbaが6,943件回収・653 errors・分裂1/11だった。通常の履歴入力との差は小さく、少なくとも今回の設定では履歴入力による改善を確認できない。この全maskは独立に学習したprimaryのみ対照の代わりではない。

## 解釈

今回の結果は、確定した直前1時点・8候補・pair MLP内部attention・共同学習の初回設計が、同一pairの早期診断で対照を改善しなかったことを示す。公開primaryからprimaryのみ再学習した時点でも既知接続は両胚で減っており、共同学習による変化と履歴入力の寄与を分けて読む必要がある。GEFFは部分注釈で、active-pair errorsや教師負例予測を真の誤接続総数とはみなせない。

全graph推論と公式評価は実行しておらず、公式scoreは未計測。pair評価だけでは候補選択、ILP、後処理との相互作用は測れない。事前の進行条件に従い、今回の実行から全graphへは進まない。

## ユーザー判断

2026-09-26にユーザーが本実験の完了・不採用を判断した。対象は直前1時点・8候補・pair MLP内部attention・共同学習の今回の設計と評価条件であり、多時点情報を使う手法全体の棄却ではない。全graphへの進行は行わず、公式scoreは未計測。competition submissionも行っていない。
