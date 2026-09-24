# exp042_public_x138_replay 要件と実装方法

この文書は実装前の契約と受け入れ条件の正とする。2026-09-23のユーザー依頼「public_x138_replayを実装してください」を、未決事項なしの候補の実験化承認として扱った。元の候補の契約・根拠・判断履歴は後半へ移した。進捗と実行時系列は `SESSION_NOTES.md` に記録する。

## 実験化の入口・引き継ぎ・承認

- 入口: 設計可能・未決事項なしの `public_x138_replay` 候補。
- 承認: 2026-09-23の「public_x138_replayを実装してください」。Notebook実装を承認し、submission承認とは扱わない。
- 親実験: `exp013_public_notebook_replay`。対応仮説は `HYP-20260910-12`。
- 候補からの解釈変更: なし。公開構成の予測機構は省略しない。
- 判断履歴: 後半の候補移行記録に初回提案、追加座標補正の訂正、バックログ承認、今回の実験化承認を記載。

## 手法契約

- input: SHA256を固定した公開x138 V1 Notebook、公開3 checkpoint、追加のV1284座標補正checkpoint、offline wheels、競技test Zarr。Notebook冒頭と `replay_input_manifest.json` で重み・入力・環境を検査する。
- target / objective: 参照構成で細胞中心・接続・分裂を予測し、同じ入力・環境から同じ予測が得られるか調べる。
- output: 参照sourceの `submission.csv` と `run_stats.csv`、座標補正前後・graph・入力のmanifest、`replay_receipt.json`、2実行の比較report。
- loss: なし。全モデルを固定して推論する。
- decode / 推論: 取得したNotebookの2画像モデル、特徴の反転・回転平均、V1284による中心補正と補正位置での特徴補間、正逆対応確率、ILP、近傍移動量、未使用検出点の再追加、低得点候補による欠落補完、分裂・軌跡修復を維持する。推論コードは `assets/reference_notebook/biohub-x138.ipynb` から `build_notebook.py` がセル順に複写する。変更は入力検査、座標SHA記録、実行後のread-only receiptだけ。
- context unit: 画像特徴と対応は隣接2フレーム、座標補正は検出点、graphと出力は動画、再現性比較はtest全件。
- 実装区分: `faithful`（このリポジトリの管理用語）。予測機構を省略しない。Public LBだけは提出承認後の測定であり、この実装に含まない。
- 変更class: `mechanism`。exp013の旧公開構成から、V1284補正と候補回収・再接続を含むx138全体へ変える。小さな設定探索の連続には当たらず、今回の具体的な公開構成の再現には `kaggle-idea-forge` を使わない。
- 検証可能な主張: 公開testの同一条件で補正前後の座標・graph・raw提出ファイルが2回一致し、実測費用が予算内か。未提出のPublic LB、hidden test全体の時間、独立CV、個別機構の寄与は判断できない。
- 実装ファイル: `build_notebook.py` がJupytext percent形式の `exp042_public_x138_replay_inference.py` を作り、Jupytextが `.ipynb` へ変換する。`compare_replays.py` が2回の出力を照合する。
- source一致の確認: reference Notebookの固定SHA、code cellのAST、および追加cellを除いた実行Notebookのcode cell順序を実験テストで確認する。公開sourceのprediction設定、V1284の `candidate` mode、float座標を維持する。
- 入力guard: 追加checkpointのdataset ref・version ID・SHA256を `config.yaml` へ固定するまでNotebook冒頭で失敗する。3公開artifact manifestとT4 2基、test列挙、wheelを検査する。追加重みの別モデル・ゼロ補正・再学習で代用しない。
- 再現性: sourceと全重みSHA、test入力構造とmetadata SHA、補正前後のfloat32座標SHA、detector座標manifest、graph topology、決定的なrun統計、raw提出SHAを記録する。推論時間、Kaggle version、GPU割当消費は別に記録する。CUDAとILPの非決定性は2 runで実測する。
- 実行gate: 追加重みが未取得なのでKaggle push・フル実行は保留。入手後、quotaを確認し、同じT4 2基のcleanな2 runと提出前検証を行う。実際のsubmissionには別途ユーザーの明示承認が必要。
- 受け入れ条件: 静的検証が通り、追加重みを固定した後の2 runで入力manifest、座標補正前後、graph、決定的な統計、raw提出ファイルが一致すること。入力欠落、SHA不一致、OOM、時間・GPU予算超過、不一致では依存する実行を止め、原因と証拠を残す。
- GPU学習コスト: active variant 1、config 1、fold 0、booster 0、control再学習なし。GPUは推論2実行のみ。公開20分28秒は参考値で、hidden testの保証ではない。
- 現在の阻害条件: 公開実行の生metadataから `anvithpothula/biohub-v1284-head-s075` とDataset Version ID `19822532` を特定した。Kaggle APIのfiles/downloadは403で、`v1284_head.pt` の本体とSHAは未取得。これは実行の先行条件で、実装方針の未決事項ではない。

## 実装方法

- 固定するもの: x138 V1の参照source、3つの公開checkpoint、追加V1284 checkpoint、依存物、推論順序と設定、T4 2基。
- 変更するもの: exp013の旧公開構成をx138全体へ置き換え、入力検査と座標・graph・提出の再現性記録を加える。
- source検査: 固定Notebook SHAと全code cellのASTを比較し、追加cell以外の予測コードが一致することを実験固有テストで確認する。
- Notebook: setup、入力、source、推論、graph、出力をMarkdown見出しで追えるJupytext sourceから生成する。Kaggle packageには正の.ipynbとconfigを含める。
- 比較: 追加checkpoint取得後に公開test全件をcleanに2回実行し、`compare_replays.py`で照合する。

## 探索幅とpivot判定

- variant/config/fold/booster: 1/1/0/0。親の再学習はない。
- parameter探索: なし。公開x138の固定版だけを調べる。
- pivot: 今回は公開構成の再現であり、targetやdecodeを変える候補選定ではない。追加重みが取得できなければ代用せず停止する。

## 再現性・リスク

- CUDA、SCIP、2 GPU shardには非決定性が残り得るため、同じ環境の2 runで測定する。決定性設定を追加して参照sourceを変えない。
- 公開重みの学習来歴は独立CVを保証しない。trainの独自指標を公式scoreと呼ばず、作者の0.953を自身の結果として記録しない。
- 公開testで一致してもhidden testの費用と精度は未測定。ILP 1,200秒/動画、7.5時間後の処理縮小、Notebook 12時間の条件を監視する。
- sourceやcheckpointのSHA不一致、欠落、OOM、費用超過、再実行不一致では停止する。予測設定・重みを探索して救済しない。

## 受け入れ基準

- [x] 上位仮説ID、候補契約、根拠、差分、実装方法、成功・停止条件、判断履歴を移行した。
- [x] 参照NotebookをSHAで固定し、V1284必須guardと補正前後の診断、2 run比較を実装した。
- [ ] 正確な追加checkpointのdataset ref、version、SHAを固定する。
- [ ] Kaggleで公開test全件をcleanに2回実行し、座標・graph・raw提出の一致を確認する。
- [ ] 提出前検証を行う。Public LBは別途承認されたsubmissionまで未測定とする。

## 候補から移した契約・根拠・判断履歴

# public_x138_replay

- 候補名: `public_x138_replay`
- 状態: `設計可能・実験化未承認`
- 対応する上位仮説: `HYP-20260910-12`
- 関連する上位仮説: `HYP-20260910-02`（中心補正）、`HYP-20260910-04`（近傍の移動）、`HYP-20260910-14`（比較の信頼性）。各機構単独の効果はこの候補では判断しない。
- 作成日: 2026-09-23
- 最終更新日: 2026-09-23
- 依頼原文: 「バックログ案にするとどうなりますか？」「先ほどのバックログ案2件をバックログに追加してください」
- 期待する成果: 公開biohub x138 V1を追加の座標補正モデルまで含めて忠実に再現し、再実行一致・実測費用・提出前検証の証拠を作る。提出を別途承認された場合に、Public LB 0.953の再現性とexp013の0.944との差を測る。
- 親実験 / 比較対象: 再現手順と提出基準は[`exp013_public_notebook_replay`](../exp013_public_notebook_replay/)、既存公開重みは[`exp011_public_detector_selection`](../exp011_public_detector_selection/)、再現対象は[anvithpothula/biohub-x138](https://www.kaggle.com/code/anvithpothula/biohub-x138) V1。
- 優先度: P1
- 優先度の理由: 新しい公開0.953構成が手元の条件でも再現できるかを先に確かめる。後続の重み比較の入力を確定するため、未着手候補では最初に扱う。既に動いている実験を中断する指示ではない。
- `backlog/KAGGLE_DIRECTION.md` の対応箇所: 「検証中の仮説」の`HYP-20260910-12`、未着手バックログの本候補。

## 観測事実と根拠

- 実測済みの事実: 2026-09-23取得のKaggle APIではx138のBest Public Scoreは0.953、公開版はV1。作者の公開test 4動画の実行は正常終了し、Notebook全体1,228.45秒、最終出力238,260行だった。私たちの再実行・提出結果ではなく、hidden test全体の時間でもない。exp013は同じ環境での2回一致と自身の提出0.944を確認済み。
- 根拠ファイル / 一次資料: [訂正済み調査](../../docs/surveys/biohub-public-0953-differences_20260923.md)、[取得Notebook](../../docs/notebooks/biohub-cell-tracking-during-development/anvithpothula__biohub-x138/biohub-x138.ipynb)、[取得metadata](../../docs/notebooks/biohub-cell-tracking-during-development/anvithpothula__biohub-x138/kernel-metadata.json)、[APIの版とscore](../../studies/biohub_public_notebooks_20260923/public_scores.json)、[公開ログ](../../studies/biohub_public_notebooks_20260923/x138_run.log)、[exp013要件](../exp013_public_notebook_replay/requirements.md)と[設定](../exp013_public_notebook_replay/config.yaml)。
- 利用する保存済み生成物とSHA: x138 V1のNotebook SHA256は`6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d`。[Notebook取得一覧](../../studies/biohub_public_notebooks_20260923/notebook_inventory.json)を正とする。既存primary・secondary・DeepCenterの3 checkpointは[公開実行の記録](../../studies/biohub_public_notebooks_20260923/x138_output/bidirectional_production_runtime_integrity.json)と[exp011選定記録](../exp011_public_detector_selection/assets/public_detector_selection.json)でSHAが一致する。追加の`biohub-v1284-head-s075/v1284_head.pt`は未取得であり、取得・版固定・SHA計測を実行の先行条件とする。
- 仮定: Assumption: 作者と同じ追加座標補正checkpointと依存物を取得できる。成立しない場合は忠実再現を停止し、別モデルや再学習で代用しない。再現成功・score一致・性能向上を観測前に仮定しない。

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: 公開構成の保存・再利用によって、追加座標補正と候補回収を含む新しい推論をKaggleの計算予算内で再現可能な比較基準にできるか。
- この候補の具体的な仮説: x138 V1の全モデル・推論順序・数値設定を固定すれば、公開testのcleanな2回実行で同じ座標・graph・提出ファイルが得られ、別途承認された提出でも公開0.953の基準を再現できる。
- 仮説が正しい場合に期待する観測: 同一入力・環境の2回で予測が一致し、通常実行の時間内に完走する。提出後には0.953との差とexp013の0.944との差を数値で示せる。
- 仮説を棄却する観測: 必要重みを同一条件で用意できない、予測が再実行で一致しない、予算・実行時間を超える、または同一構成の提出で0.953を再現しない。原因を切り分け、設定探索によって再現失敗を隠さない。
- この候補だけで上位仮説を判断できるか: いいえ
- 上位仮説の判断に残る検証: 学習済みtrackerの交換による順位と費用は[`public_x138_tracker_comparison`](../../backlog/public_x138_tracker_comparison.md)、座標補正・近傍移動量・検出点回収・欠落補完それぞれの寄与は将来の別比較。hidden test全体の費用とLBは実際の提出まで未確認。

## 入力・予測対象・出力・推論方法

- input: 固定したx138 V1 source、既存3 checkpoint、追加座標補正checkpoint、offline依存物、competition実行時のtest画像。test動画は動的に列挙する。
- target / objective: 全動画の細胞中心、時刻間接続、分裂を参照構成と同じ計算で予測し、再現性と費用を確認する。
- output: 検出候補・補正座標・graph・`submission.csv`、source / artifact / environment manifest、再実行比較結果と時間・メモリの記録。
- loss: なし。すべての学習済み重みを固定して推論する。
- decode / 推論方法: x138 V1を一体として保持する。2つの画像モデルとトラッカー、特徴の反転・回転平均、追加モデルによる中心補正と補正位置での特徴補間、正逆方向の対応確率統合、ILP、近傍移動量による再接続、未使用検出点の再追加、低得点候補による欠落補完、既存の分裂・軌跡修復を含む。
- 処理単位: 画像特徴と対応は隣接2フレーム、座標補正は検出点ごと、最終graphは動画単位、再現性確認は公開test全件。
- 実装区分: 参照sourceの予測機構と設定を省略しない`faithful`。このリポジトリ内の管理用語であり、パス置換と記録追加だけを許容する。

## 親実験からの差分

- 変更するもの: exp013の旧公開構成をx138 V1全体に置き換える。主な差は追加の学習済み座標補正、secondary画像特徴とDeepCenterの反転・回転平均、候補回収・欠落補完・再接続と実行時間制御。後処理だけの変更とは記録しない。
- 固定するもの: 取得したx138 V1 source、使用する全checkpoint、特徴正規化、推論設定、処理順序、依存物、Kaggle T4環境。exp016の再学習重みは使わず、x138の公開トラッカーを使う。
- 再利用するコード / config / 生成物: exp013の入力manifest・run receipt・中間出力比較・提出前検証の方法、exp011の既存3重みの照合、保存したx138 sourceと公開実行証拠。旧exp015の特徴cacheを新構成の入力と同一とは扱わない。
- 新しく作るもの: 実験化後に、x138の忠実な実行Notebook、追加モデルを含むartifact manifest、cleanな2回実行の比較記録と提出前検証結果を作る。

## 最小の反証可能な検証

- 検証方法: まず追加座標補正モデルの取得可能性・使用版・来歴を確認し、全sourceと重みのmanifestを作る。次にKaggleで公開test全件を同一条件から2回実行し、補正前後の座標、graph topology、決定的な統計、提出ファイルを比較する。作者の公開ログ・出力も参照するが、自身の実行証拠と混同しない。提出は別途の明示承認後に固定したversionで行う。
- variant / config / fold / booster数: x138 V1の1構成、cleanな2実行。追加学習・fold選択・設定探索は0件。
- control再学習: なし。exp013の保存済み再現結果と提出0.944を旧基準として使う。
- 想定runtime / resource: Kaggleのみ、週45 GPU時間以内、課金なし。作者の公開test実行20分28秒は参考値であり、自分の実行やhidden testの保証にはしない。T4 2基を使う場合の割当消費をNotebook経過時間と別に記録する。x138の動画別ILP上限1,200秒・7.5時間後の処理縮小を維持し、Notebook全体12時間以内と利用可能GPU残量を確認する。

## 成功条件と停止条件

- primary指標: 再実行一致。提出承認後はPublic LBとexp013との差、作者の0.953との差。
- 成功条件: 全source・重み・入力・環境manifestが一致し、2回の候補座標とgraph topologyが一致、raw提出ファイルがbyte-identical、形式検証が通り、時間・GPU予算内で完走する。これで再現性確認は成立するが、LB再現は未提出なら未検証と記録する。提出後に0.953との一致を判定し、0.944を上回るかも別に示す。
- 必須guard: 追加座標補正が`candidate`モードで実行され、既存3重みだけの検証で済ませていないこと。補正座標を整数へ戻さないこと。比較用のSHA計算やcache保存で推論値を変えないこと。test固定ID・4動画・238,260行を推論の条件に使わないこと。trainの独自指標を公式scoreと呼ばないこと。
- 成功時の次段階: 証拠をユーザーへ提示し、基準の採用と提出を別々に判断してもらう。2件目の実験化・実行もこの候補から自動開始しない。
- 失敗時の停止範囲: 追加重み未取得、SHA不一致、非再現、入力漏れ、OOM、時間超過、GPU残量不足があれば依存する実行を停止し、原因と取得済み証拠を残す。実行時間縮小が作動した場合は通常実行と区別する。

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: 追加座標補正モデルの省略・ゼロ補正・別重み・独自再学習、exp016重みへの交換、閾値調整、独自指標の探索、後処理だけの移植をx138再現と呼ぶこと、作者scoreを自身のscoreとして記録すること、未承認submission。
- 壁打ちで採らなかった案と理由: すぐ既存baselineを置き換える案は、自身の再現・提出証拠がないため採らない。exp016がより良い提出重みだという前提は実測で支持されないため採らない。個別機構の除去比較は、全体再現前に候補数を増やさないため別途とする。

## リスク

- leakage / validation: 既存公開モデルのtrain重複に加え、追加座標補正モデルは作者コメントではtrain 20動画・4,136対応点で学習。正確な学習コード・対象一覧は未確認であり、独立CVとは呼ばない。
- hidden test: 入力一覧は実行時に取得し、公開testでの一致をhidden testの精度・費用保証にしない。
- runtime / memory: 低得点候補の保存、補正座標での特徴補間、動画サイズとILP上限への到達で費用・出力が変わり得る。
- 再現性: 取得metadataのdataset_sourcesに空欄があり、追加モデルの正式な取得先と版は記録だけでは特定できていない。実行前に確認する。モデルが取得できなければ忠実再現できない。sourceの実行時patch、CUDA、ILPの非決定性も2回比較で検査する。

## 先行条件 / 依存

- x138 V1と同じ追加座標補正checkpoint・offline依存物を取得でき、版とSHAを固定できること。入手できない場合の代替モデルは本候補の範囲外。
- 学習方針は維持する。既存の公開検出器・画像encoderを更新せず、公開済みの追加モデルも固定して使用する別比較である。

## 調査・実行時に確認する事項

- 追加checkpointの取得先・SHA・学習来歴、実行環境の一致、時間・メモリ、座標とgraphの再実行一致。これらは取得・測定対象であり、手法の未決定事項とは分ける。
- Public LBは明示承認された提出後にのみ取得し、score未測定のまま採用を確定しない。

## 未決事項

- なし

## 判断履歴

- 2026-09-23: 公開0.953構成をP1で忠実再現し、その後に重み比較を行う2候補を提案した。
- 2026-09-23: 初回調査の「後処理だけ」という見落としを訂正。追加の学習済み座標補正と、トラッカーへの入力変更を含む候補にした。
- 2026-09-23: ユーザーが先ほどの2候補のバックログ追加を依頼。実験化・実装・Kaggle実行・submissionの承認は含まない。

- 2026-09-23: 既存66候補の優先度を再点検し、他候補の順位と進行中実験は維持。x138再現をP1、依存する重み比較をP2とした。

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: はい。x138 V1全体と全checkpoint、推論設定を固定する。
- 変更するものを一意に説明できる: はい。旧公開構成からx138構成への比較であり、実行上はパスと記録の追加だけ。
- 最小検証と停止条件を一意に説明できる: はい。必要重み取得後のcleanな2回実行と一致検査。取得不能・不一致・予算超過で停止。
- 実行しないことを一意に説明できる: はい。追加学習・重み交換・設定探索・未承認提出は行わない。
- 未決事項が明示されている: はい。なし。追加重みの取得等は実行の先行条件に明記した。
