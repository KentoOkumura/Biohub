# exp044_x138_past_candidate_knn_attention 要件と実装方法

実装前の契約、実装方法、受け入れ条件の正本。進捗は `SESSION_NOTES.md`、設定は `config.yaml` に置く。

## 実験化の入口・引き継ぎ・承認

- 直接承認: ユーザーが `past_candidate_attention` の実行を依頼し、その後「exp043をベースとした実装に変更」を依頼した。確認への回答で「exp043の入力と処理を使う」「primary tracker＋attention」を選択した。
- 親実験: `exp043_x138_self_trained_head`。exp038のK=8 attentionは機構の参照元だが、その旧cache・CV値を対照としない。
- backlog候補: N/A（会話からの直接承認）。上位仮説ID: N/A。exp038と同じK=8機構だが、既存の仮説索引へ新実験を推測で紐づけない。
- 参照実装: exp043の推論source、x138公開Notebook、公開 `SimpleNodeTransformer` checkpoint、exp038の13特徴・K=8 attention。
- 変更するもの: primary trackerの重みと、直前frameの全候補から物理距離で近い8点を参照するattention。追加branchはzero deltaで初期化する。
- 固定するもの: exp043の2つの画像encoder、検出候補、self-trained座標head、secondary tracker、双方向harmonic融合、低margin時のsecondary融合、および後処理。単体評価段階では後処理を実行せず、同じpair cacheで固定exp043対照と比較する。
- 未決事項: なし。全graph推論へ進むかは単体評価の実測後に判断する。
- 実行しないこと: exp038の古いcacheで学習・対照評価すること、公開画像encoderや検出器を再学習すること、pair指標を公式scoreと呼ぶこと、ユーザー承認なしのsubmission。

## 判断履歴

- 2026-09-24: exp038の履歴を保持し、exp043の入力と処理へ変更する実装は別実験exp044とした。
- 2026-09-24: ユーザー回答で、固定trackerにattentionだけを学習する案ではなく、primary trackerとattentionの同時学習を選択した。

## 手法契約

- input: exp043と同一の公開2 encoderによる検出候補、self-trained headで補正済みの候補座標、primary画像特徴と位置特徴、固定secondary logit、直前frameの候補座標・ID。20 train動画はexp043と同じseed 42による胚別10本選択。
- target: GEFFの既知nodeを7 µm以内で検出候補へ1対1対応付けし、既知edgeのpair行列を構築する。注釈が疎なため、負例解釈には制限がある。
- output: primary `SimpleNodeTransformer` のedge logitにK=8 attentionのdeltaを加え、exp043の双方向harmonic・secondaryのlow-margin consensusを再適用したlogit。候補ごとにsource軸softmaxする。
- loss: 既知edgeのある行または列に限定したsource軸softmax focal BCE、gamma 2。primary trackerと追加attentionを同時更新する。secondaryと画像encoderは固定。
- decode: 2-frame診断ではexp043と同じ0.48確率閾値。全動画graph推論、ILP、後処理は単体診断を通過した後の別段階とし、今回のtrain Notebookには含めない。
- context unit: 隣接2-frameの全候補。各sourceのattentionは、その直前frameの候補を最大8点選択する。教師node IDはモデル入力にもKNN選択にも使わない。
- 実装区分: `staged-faithful`（このリポジトリ内の管理用ラベル）。pair学習は忠実に実装し、全graph経路は段階的に進める。
- 検証できる主張: 同じexp043入力上で、既知edgeの回収、active pairの誤接続、既知分裂親の回収が改善するか。
- 判断できない主張: 全graphでの公式score、未注釈edgeの真の誤接続、public testでの改善。

## 実装方法

1. exp043 inference sourceをSHAで固定してtrain Notebookを組み立て、self-trained座標headの入力SHAも検証する。推論scriptにfail-closedのcapture patchを当て、graph作成の前にframe pairごとの候補・特徴・logitを保存する。
2. 捕捉したprimary forward/reverse、secondary、最終融合logitから、公開checkpointと融合計算の一致を $10^{-4}$ 以内で検証する。候補ID・座標・特徴shapeの異常があれば停止する。
3. GEFFに存在する隣接frame pairのみを教師化し、既知の直前親候補がK=8に入る率を胚別に調べる。どちらかが99%未満なら学習前停止する。
4. 片方の胚で学習し、同じ胚内2動画をinternal selectionに、反対の胚をouter evaluationに用いる。胚を入れ替えて2 checkpoint作る。対照は同じpair cacheの保存済みexp043 logitを使う。
5. 両胚それぞれのknown-edge recall、active-pair error rate、false edges、division-parent recall、教師対象数を記録する。既知edge recallとactive-pair error rateの両方が悪化しない場合だけ全graph推論を検討する。単体評価で測れないILP・後処理との相互作用が残る。

## 探索幅とpivot判定

- 変更class: `representation`。exp038から入力候補と画像特徴をexp043へ変えるため、単なるparameter tuningではない。
- 既存のK=8機構は流用するが、対照を同じx138候補上で再測定する。正のheadroomはK=8既知親候補retentionと2-frame誤差で実測する。
- その結果が届かない場合は全graph推論へ進まず、候補選択または情報源の変更を検討する。今回の変更には新しいアイデア生成を含めない。

## 再現性・リスク

- 公開画像モデルのtrain由来が不透明なので、CVは公開事前学習済みモデルに条件付けられる。exp043のLBと直接同じ尺度のCVではない。
- GEFFは部分注釈。active行・列中の未注釈edgeを負例とみなす既存lossは誤負例を含み得る。両胚別の数値とこの限界を一緒に報告する。
- 全20動画のx138特徴captureは計算費用が大きい。2動画pilotから180分上限を予測し、超えると残りを開始しない。tracker学習は240分上限。Kaggle GPU quotaを実行直前にも確認する。
- 選択動画、checkpoint SHA、head SHA、cache内容、モデルSHA、Kaggle kernel versionと実行時間を証拠として保存する。非決定的なCUDA処理があるので、完全bitwise再現は主張しない。

## 受け入れ基準

- [ ] captureがexp043と同じ候補・座標head・画像特徴・fusionを使い、graph推論前で停止する。
- [ ] 公開primary tracker、fusion、zero-initialized attentionのbaseline logit parityが閾値内。
- [ ] K=8既知親候補retentionが両胚で99%以上。
- [ ] optimizerにprimary trackerとattentionだけが入り、secondary・画像encoder・座標headが入らない。
- [ ] 胚別の対照と学習枝のpair指標、件数、モデルSHAを記録する。
- [ ] `make check-exp`、`make test-exp` を通す。
