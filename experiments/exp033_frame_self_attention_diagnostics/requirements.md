# exp033_frame_self_attention_diagnostics 要件と実装方法

## 実験化の入口・引き継ぎ・承認

- 2026-09-20: `backlog/frame_self_attention_diagnostics.md` を P1 の検討メモとして作成。PR比較の基準とbucket定義が未決。
- 2026-09-21: ユーザーが「frame_self_attention_diagnosticsを実装してください」と依頼し、実験化を承認。
- 2026-09-21: ユーザーが precision 0.95、同点scoreの一括処理、到達不能時の未達記録を選択。混雑度に子候補の最近傍距離を用い、候補数と移動距離のbucket境界を学習側の分布から固定することを選択。
- 移行元: `backlog/frame_self_attention_diagnostics.md`。本書とconfigへ移行後に候補詳細と未着手行を削除する。
- 上位仮説: `HYP-20260920-02`。関連仮説: `HYP-20260910-14`。
- この実験が検証する範囲: Self-Attention追加の効果が、0.5閾値の指標だけでは見えない順位や局所的な誤りの差として残るか。
- この実験だけで上位仮説を判断できるか: いいえ。距離bias等の学習比較と公式graph評価が残る。
- 親と対照: `exp025_frame_self_attention` のModel A、Model B、model_b_identity_init、および `exp016_frozen_image_encoder` の保存済み現行tracker。
- 根拠: 両実験のrequirements/config/metrics/result、exp025の `frozen_tracker.py` と `simple_node_transformer.py`、exp015の固定window cache。exp025の保存manifest SHAはA `c0e70224a14de199119c0104e51c002f7b6121bc562dc4727f52703998e80e3c`、B `123121e4e882220bfe86af249b6712d6cfe12fa64fed1fa415bdc50d2a21e6ea`、identity `442c25536a64419d2460755a13513edd0f4aa76b39df29d954af5a92359faa4c`。exp016は `6f9548739ae6f58d3c8beb7ccb78464ddc10243841b1fc7e071ed876f757ea56`。
- 観測事実: identity Bは通常Bより両胚でprecisionが上がりrecallが下がった。0.5以外の順位比較と条件別誤りは未測定。
- 仮定: 固定閾値での差には確率の出方の変化が含まれる可能性がある。距離biasの効果を直接示す証拠はまだない。
- 未決事項: なし。四分位境界と小標本表示の細部は以下で固定する。

## 手法契約

- input: exp015の固定画像・位置特徴cache、候補ID・物理座標、GEFF教師、exp025/016のfold別checkpointと対応source。
- target / objective: 既知親子edgeと分裂親。既存のgreedy 5 µm候補対応、教師mask、fold分割を維持し、再学習しない。
- output: 胚・構成・split別PR曲線、precision 0.95以上で到達できる最大recall、0.5指標の再現、候補数・混雑度・移動距離・分裂別の誤り、有効件数と未知endpoint件数。pair ID・教師・物理座標・確率をsample別shardへ保存する。
- loss: なし。保存済みcheckpointをevalで再推論し、勾配更新しない。
- decode: logitsのsource軸softmaxを従来どおり計算する。graph復号・実運用閾値の選択はしない。
- context unit: 隣接2-frame window。内部検証と外側胚を分け、胚別に集計する。
- 実装区分: 新しい参照手法を実装しない診断のためfaithful / staged-faithful / proxyの分類対象外。
- 変更class: `postprocess`。このリポジトリ内の管理用語で、学習済み出力の保存と評価を加える。tracker、教師、lossを変えない。
- 支持・棄却できる範囲: 固定4checkpoint構成の条件付きpair順位と誤りの差。公式score、独立CV、距離biasの有効性、他設定のSelf-Attention全体は判断できない。

## 集計定義

- 主PR母集団は既存lossと同じactive pair mask。未知endpointを含むactive pair件数を別記し、mask内の未注釈pairを真の負例と断定しない。既知endpointだけの感度分析も出す。
- scoreはfloat32のsource軸softmax確率。全同点scoreを一度に取り込み、累積TP/予測陽性と累積TP/全既知正例を計算する。precisionが0.95以上の点の最大recallを記録し、非空の予測集合で到達不能ならnullとする。外側胚で得た閾値は運用に使わない。
- 固定0.5の再現は `probability > 0.5` とし、既存のpositive-edge recall、positive-pair precision、教師mask内のfalse-positive pair、分裂親回収と分母を照合する。一致しない場合は比較を停止する。
- 候補数はwindowの子候補数。混雑度は同じ子frame内での他候補との最近傍物理距離 µm。候補1個なら未定義と別件数にする。移動距離は親・子候補の物理距離 µm。
- 各foldのgradient-update windowのみから、候補数はwindow、最近傍距離は子候補、移動距離は既知正例pairを対象に25/50/75パーセンタイルを計算する。重複境界は結合し、境界と各bucket件数を保存する。外側胚・内部検証の値で境界を再調整しない。
- bucket別は0.5固定のTP/FP/FN、precision、recallと件数を示す。正例30件未満ならprecision/recallをnullとして件数だけ示す。分裂親の回収は分母10件未満なら割合をnullとする。
- split: 学習側の内部検証を診断と境界確認に使い、外側胚は記述的評価のみ。公開画像モデルの学習来歴にtrain動画が含まれるため独立CVと呼ばない。

## 実装と検証

- 4構成×2fold=8保存checkpoint、診断config 1、tracker学習0、booster 0、control再学習なし。Kaggle Notebookで小規模windowの時間・メモリを測り、週30 GPU時間・単Notebook12時間の残量を確認してから全件へ進む。
- exp025の教師・cache検証・model sourceをバイト一致で引き継ぐ。各Kaggle outputのmanifest、model file、state、source SHAを照合し、fold・variant・評価胚を固定する。exp016現行も同じwindowと教師で推論する。
- active pairのID、ラベル、未知endpoint flag、座標、4確率はsample単位に分割保存し、file SHAを記録する。全pairを一度にGPU/CPUへ保持しない。
- 最小検証: 入力SHAと0.5既存指標を先に再現し、PR曲線と条件別誤りを両胚で得る。改善観測を診断の完了条件にしない。0.5再現失敗なら比較停止。
- リスク: trainの部分注釈、公開重みの来歴、推論費用、全pair保存容量。出力サイズ・実推論時間・peak memoryは実行時に測る。
- 実行しないこと: 集約済み指標からPR曲線を復元、未知pairを確定負例扱い、外側胚で閾値やbucketを最適化、再学習、全graph推論、submission。
- 後続: 診断を受けて `frame_self_attention_spatial` の距離biasまたは近傍制限を検討する。局所誤りだけで方式を確定しない。

## 受け入れ基準

- [x] 4構成×2foldのmanifest/source/checkpoint/cache/teacherを照合する。
- [x] 同じwindow・候補・教師で既存0.5指標を再現する。
- [x] PR曲線とprecision 0.95でのrecall、条件別誤り、有効・未知件数を保存する。
- [x] 内部検証と外側胚を分離し、bucket境界は学習側のみから得る。
- [x] Jupytextのround-trip、validate-exp、check-exp、test-expが通る。
- [x] Kaggle実行時は時間・メモリ・生成物SHAをmetricsへ記録し、公式score未計測を明記する。
- [x] 2026-09-21にユーザーが実験完了を判断した。Self-Attention構成の採否は未判断。

## 実装方法

- `frozen_tracker.py` はexp025のcache、GEFF教師、split処理を同じsourceとして保持する。`diagnostic_runner.py` はmanifestとcheckpointをSHA照合し、sample別pair shard、学習側四分位境界、0.5再現、条件別集計を行う。
- `diagnostic_metrics.py` は同点をまとめるPR曲線、固定閾値の件数、子候補の最近傍距離とbucket集計を計算する。Notebookは入力、実行順、生成物と数値をセルごとに示す。
- 親のsource一致はSHA、承認した集計差分は実験固有testとKaggle上の0.5再現gateで確認する。

## 探索幅とpivot判定

- 既存checkpointの診断のみ。新たなパラメータ、target、output head、decode、context unitを選ばない。
- exp025のpair指標は混在しており、後続の距離biasや近傍制限をこの実験内で自動追加しない。診断後に方式を検討する。

## 再現性・リスク

- sample・window・pairを固定順で処理する。新しい乱数、特徴生成、augmentation、学習、optimizerを使わない。CUDAの浮動小数点演算がbitwise同一とは仮定しない。
- cache summaryと全window identity、GEFF内容、model manifest/source/checkpoint/state、pair shard/PR曲線/summaryのSHAを記録する。Kaggle kernel versionは実行後に記録する。
- 公開画像モデルがtrain動画由来のため独立CVではない。部分GEFFで未知pairを真の負例とみなさない。混雑度と距離の境界はfoldの学習側からのみ決める。
- 4構成の再推論、全pairの分割保存、PRソートには時間とstorageを要する。GPU・Notebookの残量と実測を事前gateにする。

## 判断履歴

- 2026-09-20: ユーザーはPR曲線・誤り分析を空間的な構造変更より先に行う案を受け、候補のバックログ化を依頼した。
- 2026-09-21: ユーザーが実装を依頼し、precision 0.95と学習側分布のbucket境界を選択した。
- 2026-09-21: ユーザーがexp033を完了と判断した。後続の学習とSelf-Attention構成の採否は未判断。
