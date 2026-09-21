# exp032_three_frame_ten_epoch_training 要件と実装方法

本書は実装契約です。2026-09-20の「three_frame_longer_trainingを実装してください。colabで実行する前提としてください。10エポックに変更してください。バックログ名など関連するものはすべて変更してください」を実験化と10 epoch・Colabへの変更の承認として記録します。候補名は内容に合わせて `three_frame_ten_epoch_training` とし、上位仮説は `HYP-20260910-10` を引き継ぎます。

## 実験化の入口・引き継ぎ・承認

元候補の設計可能・未決事項なしの契約をユーザーの実装依頼により実験化した。今回の10 epochとColab指定が元の3 epoch・Kaggle予定に優先する。

## 根拠と検証範囲

- 親実験: [exp027](../exp027_multi_frame_tracker/requirements.md)、[設定](../exp027_multi_frame_tracker/config.yaml)、[結果](../exp027_multi_frame_tracker/result.md)、[実行証拠](../exp027_multi_frame_tracker/metrics.json)、[原因考察](../../docs/surveys/biohub-exp027-three-frame-analysis_20260920.md)。
- 保存済み対照: [exp016](../exp016_frozen_image_encoder/result.md)のtrain v3内部選択済み2 weight。両foldとも2 epoch目が選択された。exp027の2時点1 epochと3時点1 epochは補助対照にする。[exp024](../exp024_tracker_six_epochs/result.md)は構造が異なるため、重みの再選択や直接の共通窓対照に使わない。
- 観測事実: exp027 train v4は両条件を各foldで1 epoch学習し、各胚64窓を評価した。後のepochでの改善や収束は未測定。exp027のtrain manifest SHAは `da33cd9df862ed0e86f0ec5539e9d0e0befeda78e277e37f3e4d88a05a92ee52`、split SHAは `401b073ee952514982e40f69f116dca08ff0aee02c4d0a81b7454b4b71112e61`、exp016 train v3 manifest SHAは `6f9548739ae6f58d3c8beb7ccb78464ddc10243841b1fc7e071ed876f757ea56`。
- 仮説: exp027と同じ3時点attentionを同じ初期値から合計10 epoch学習すると、1 epoch時点と保存済みexp016の接続判断より改善する可能性がある。この比較だけで多時点手法全体を判断しない。
- この実験の限界: exp016とは構造と入力時点数も異なり、過去情報だけの効果を分離できない。各胚64窓は分析済みの部分集合で、独立CV、全件、公式graph scoreではない。分裂母はexp027の外側窓で0件／1件のため分裂効果を判定できない。残る検証は同構造2時点10 epoch、複数seed、外側全件、分裂、公式graph、位置履歴であり、自動追加しない。

## 手法契約

- input: exp027と同じ動画内の `t−1,t,t+1` 固定候補、候補ID、画像特徴32次元、位置特徴32次元、物理座標、有効mask、相対時刻。前時点は直前のexp015保存cacheのsource側を使い、重複時点の候補IDと座標を照合する。
- target / objective: 中央 `t→t+1` の主催者GEFF既知接続と母娘接続。画像特徴、候補、教師、分割を変更しない。
- output: 中央ペアの候補対logit。座標、追加node、教師を新たに作らない。
- loss: exp027と同じsource軸softmax、focal weighted binary cross entropy、active pair mask。疎い教師の未知部分を真の負例と断定しない。
- decode: 本段階ではgraphを作らず、中央ペアの確率、正解親順位、固定0.5と内部validationだけで選んだ閾値を比較する。公式評価は実行しない。
- context unit: 1動画の3時点窓。評価は事前選択した各胚64窓で胚別に保持する。
- 実装区分: このリポジトリ内の管理用語ではexp027と同じ `staged-faithful`。Trackastraのencoder–decoder、全時点教師、窓間平均を再現したとは主張しない。
- 変更class: このリポジトリ内の管理用語では `parameter`。学習epoch上限と実行環境を変える。モデル構造、教師、loss、候補、復号は固定する。

## 実装方法

- 新規学習は `three_frame_local` 1構成×外側2fold、計2 tracker、各10 epoch、booster 0。2時点、exp016、公開trackerは再学習しない。exp027の1 epoch weightはoptimizerと乱数・DataLoader状態がないため再開元に使わず、同じ公開初期値とseedから10 epoch通して学習する。
- 固定: exp027の公開画像weightと正規化統計、exp015 cacheと候補、15 µm近傍、4層attention、時刻embedding、中央教師、loss、split、初期化、seed 42、AdamW、学習率0.0001、weight decay 0.01、batch size 2、mixed precision無効、内部 `selection_score` 最大・同点なら後のepochの規則。外側の結果でepochや学習率を選ばない。
- CLI workerは入力ZIPをSHA検証し、exp015のwindow cacheをKaggleの期限付きURLから /content に取得する。competition train GEFFはKaggleのprivate export kernelが作るZIPの期限付きURLからColabへ直接取得し、128件の保存済み予測もexp027のKaggle outputから直接取得する。公開supportと保存済みmodelは小さなZIPから展開する。学習コードはcache summary、全窓identity、公開sourceとcheckpoint SHA、GEFF内容、固定128窓の選択SHAを検証する。
- 各epochの重み、optimizer、scaler、Python・NumPy・PyTorch・CUDA乱数、DataLoader generator、内部指標をColab内へ原子的に保存し、内部選択weightの外側64窓指標を記録する。CLIは各epochのreceiptと最新の再開stateをローカルへ回収してSHAを照合し、古いstateを削除する。epoch 1と内部選択されたweightのみ結果として保管する。session切断後は完了epochの直後から次のCLI sessionで再開する。同じrun IDでconfigが変われば停止する。
- 診断コードはCLIで回収したtrain生成物を新しいColab sessionへ渡し、train完了markerとmanifest、親のsplit・選択・予測manifest、exp016の重みSHAを照合する。10 epochの学習履歴を保存し、回収したepoch 1と内部選択weightの外側固定0.5、選択weightの内部閾値、exp016とexp027の保存済み対照を同じ窓・候補ID・教師で評価する。
- exp027保存予測の1 epochと新runの1 epochはstate SHAと外側指標を照合する。Colabとの差は再現性差として記録し、同じrun内の1→10 epoch差と混同しない。cache、教師、分割、候補IDが違えば停止する。
- 内部閾値はfold 0で保存済み2時点0.5の教師負例予測151件、fold 1で288件を上限とし、strict greater判定で上限以下となる最小値を内部窓だけから選ぶ。exp016と内部選択された新3時点weightに同じ規則を適用し、外側へ固定適用する。保存済みexp027の閾値は元の内部診断結果を使う。
- 実行環境の例外: 現行の [学習方針](../../backlog/KAGGLE_DIRECTION.md#今後の学習方針) はKaggle Notebookを標準とするが、本実験の学習と部分窓診断は今回のユーザー指示によりColabを使う。公式graph評価を実施する際はKaggle上で別途行う。
- CLIを正の実行入口とする。runtime.colab.cliの入力・実行先を使用し、Google Drive mountとColab Secretsに依存しない。コード・公開support・保存済みmodelをZIPで、cache・GEFF・保存済み予測をKaggleの期限付きURLでColabへ直接渡す。大容量入力をローカルへダウンロードしない。各種転送はcolab-notebook-runnerの明示承認を得てから行う。資格情報はColabへ転送しない。実行結果をローカルへ回収して確認後、metrics.json、SESSION_NOTES.md、result.mdへ証拠を反映する。従来のDrive手動notebookは予備経路とする。

## 受け入れ基準

- 各epochのtrain loss・内部指標・更新回数・所要時間と、事前選択128窓の既知edge recall、教師負例予測数、正解親1位率、平均順位・確率、競合親との差、有効分裂母数を胚別に報告する。動画単位のpaired bootstrapで差の不確実性を示す。
- 主要比較は内部選択された3時点weight対exp016保存weight。学習量の比較は同じrunの1 epoch対内部選択weight。exp027保存2時点と3時点1 epochを補助対照として別表示する。固定0.5と内部閾値を混ぜない。
- 予備的改善は内部閾値で両胚とも対象基準より既知edge recallが増え、教師負例予測数が増えず、正解親1位率が下がらない場合。選択epochが後になっただけ、または内部指標同点だけでは学習延長の有効性としない。exp016を超えない場合は学習量の効果と既存方式への優位性を分けて判断する。
- 停止: 入力・SHA・教師・分割の不一致、NaN、OOM、Drive容量不足、Colab GPU不足。実行時間はbenchmarkとepochごとに記録し、12時間見込みを超えても保存した状態から別sessionへ再開できる。前段条件に届かなければ全graphへ自動進行しない。
- 実行しないこと: 2時点再学習、画像側の学習、外側での閾値・epoch・窓の選択、構造やlossの同時変更、全graph推論、submission。比較後の実験完了・採用・不採用はユーザー判断を待つ。

## 探索幅とpivot判定

- 変更する学習条件はepoch上限だけ。外側の結果を使ったepoch・seed・閾値・半径・学習率探索へ拡張しない。
- 2foldの10 epochで同じrunの1 epochと保存済みexp016に対する改善を切り分ける。改善しない場合は、この設定の結果を残し、入力表現・出力・教師の別案を検討する。

## 再現性・リスク

- ColabのGPU型とライブラリ版でexp027のKaggle 1 epoch state SHAが変わり得る。入力・教師・splitのSHA不一致は停止し、数値差は学習量の効果と別に記録する。
- CLI入力を読み込む前に全ファイルの存在を確認し、cache・公開source・model・予測manifestのSHAを照合する。Colabのローカル容量、期限付きURL、GPU runtimeの期限を実測し、epoch境界で再開する。
- 既知接続の部分注釈、公開画像weightの学習来歴、分析済み128窓、分裂母の少なさを結果の限界として保持する。教師負例予測数を真の誤接続数とは呼ばない。

## 判断履歴

- 2026-09-20: 元候補 `three_frame_longer_training` は3時点のみ3 epoch、Kaggle実行を予定していた。今回のユーザー指示を優先し、候補名・実験名を10 epochへ更新、実行環境をColabに変更した。
- 2026-09-20: 保存済み2時点対照を使う元の判断を維持し、新規学習は3時点のみとした。元候補の内容と根拠を本書へ移し、バックログの未着手行を削除する。
