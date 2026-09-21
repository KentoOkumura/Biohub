# exp032_three_frame_ten_epoch_training セッションノート

## 目的

元候補 `three_frame_longer_training` をユーザー指定の10 epoch・Colab実行へ変更し、exp027の3時点trackerを保存済みexp016と同じ128窓で比較できるようにする。

## 2026-09-20 実装

- ユーザーが実装、Colab前提、10 epoch、関連名称の変更を依頼した。候補名を `three_frame_ten_epoch_training`、新実験を `exp032_three_frame_ten_epoch_training` とした。
- 親exp027から新実験を作成し、3時点だけを2fold×10 epoch学習するconfigへ変更した。control再学習は0構成、boosterは0。
- Colab学習notebookはDrive入力を `/content` にコピーし、各epochのweightとresume state、内部・外側窓の指標をDriveに保存する。診断notebookは保存済みexp016・exp027のSHAと候補ID・教師を照合し、固定0.5と内部閾値を比較する。
- 現時点でColab実行、データ転送、公式graph評価、submissionは未実施。数値の採否判断は未実施。

## 2026-09-20 Colab入力のZIP化

- ユーザーの指示でアップロード対象のコードと保存済みexp016・exp027対照を1つのZIPにまとめ、Colab notebookがDrive上でSHA検証・展開するよう変更した。
- exp015 window cacheは前回のColab実行と同じく、Colab SecretsのKAGGLE_API_TOKENを使ってKaggle kernel outputから /content にダウンロードする。Driveにcache本体を置く必要はない。
- 当初はcompetition train GEFFと公開supportがローカルに存在せずZIPへ含めていなかった。CLI化の初期案ではKaggleから取得してZIPへ同梱する予定だったが、後にKaggleからColabへの直送へ変更した。

## 2026-09-21 CLI方式への変更

- ユーザーは前回exp016の最終実行と同じColab CLI方式を指定した。Drive mountとColab Secretsに依存する手動Notebookは予備経路に変更した。
- 公開support datasetをローカルへ取得し、train script、model source、checkpointのSHAがconfig.yamlと一致することを確認した。必要なrepo/src、script、checkpointのみZIPへ同梱する。当初はcompetition train GEFFをローカル取得してZIPへ同梱する予定だったが、後述のexp016方式に変更した。
- CLIの入力はZIPの分割転送と、ローカルKaggle認証で発行するexp015 cache・GEFF・保存済み予測の短命URLとする。Kaggle credentialは転送しない。
- 学習中は各epochのreceiptと最新の再開state、epoch 1 weightをローカルへ回収する。終了時はtrainまたはdiagnosticのarchiveを分割回収し、SHAと完了markerを照合する。
- 現時点で新しいCLI実行、URL生成と転送、model出力回収は未実施。転送にはskillの明示承認が必要。

## 実行予定

1. 静的検証とZIP再生成、ローカルでの転送・再開ロジック検証を行う。
2. ZIP、短命URL、生成物回収、必要な再開seedの各転送についてユーザーの明示承認を得る。
3. Colab CLIでstage=trainを実行する。session切断時は確認済みepochから再開する。
4. Colab CLIでstage=diagnosticを実行し、固定128窓の比較と完了markerを回収する。
5. metrics.json、本書、result.mdへ証拠を記録してユーザー判断を求める。

## 検証コマンド

`task` がこの環境にないため、対応する `make` targetを使う。

```bash
make validate-exp EXP=exp032_three_frame_ten_epoch_training
make check-exp EXP=exp032_three_frame_ten_epoch_training
make test-exp EXP=exp032_three_frame_ten_epoch_training
make check-strategy-docs
```

## ローカル検証

- `make validate-exp`: strict validation通過。
- `make check-exp`: Ruff lint・format通過。
- `make test-exp`: 実験固有testsなし。新Notebookの実データ実行は未検証。
- `make check-strategy-docs`: バックログと仮説の索引整合を確認。
- Jupytextの `--to ipynb --test`: Colab学習・診断の2本ともround-trip通過。
- 保存済みexp016・exp027のmanifest、split、窓選択、予測、内部閾値の設定SHAをローカル証拠と照合した。
- 親のself-containedな学習sourceは2,141行、診断sourceは1,794行。新sourceは学習2,172行、診断1,926行で、設定・入力確認・特徴と教師・モデル・実行・出力の章を保持する。最終の行数はformat後の値で変わり得る。

## 2026-09-21 CLI取得経路の確認

- Colab CLIのCPU preflightで、rootへのupload、`colab exec`中のdownload、終了後のsession stopを確認した。実験データは転送していない。
- Kaggle competition train GEFFの一覧は199サンプル・4179ファイル・2,353,863 byte。個別APIは373ファイル取得後に404となり、既取得ファイルの再リクエストも404を返した。この取得経路をfull-runに用いない。
- private Kaggle CPU kernelでcompetition mountからGEFFだけをZIP化する経路を用意した。直送方式への変更後はColabで一覧・byte数・SHAを検証する。kernel push、GEFF ZIPの生成、Colabへのデータ・weight・期限付きURL転送は未実施。
- 初期版Colab ZIPは削除した。GEFFをKaggle直送とする小さいZIPへ再生成した。runnerはZIPの全memberをローカル入力と照合し、古いZIPでGPU sessionを作らない。

- CLI構成の検証: `make validate-exp`、`make check-exp`、`make test-exp`（7件）、`make check-strategy-docs`、2 notebookのJupytext round-tripが通過した。古いZIP拒否、bundle展開SHA、GEFF ZIP検証、再開seed、転送承認フラグをローカルで検査した。
- Kaggle GEFF export ZIPと10 epoch結果は未生成。Colab用の小さいZIPはローカルの既存入力から生成した。GPU実行と対外転送は承認後に実施する。

## 2026-09-21 exp016方式への修正

- ユーザー指定により大容量入力をローカルへダウンロードしない。exp016 full replayと同じく、Kaggle APIで期限付きURL manifestだけを作成してColabへ渡す。exp015 cache約4.15 GB、Kaggle GEFF export ZIP、exp027保存済み予測128件・136,560,816 byteはColabがKaggleから直接取得する。
- 先にローカルへ取得していた公開support packの不要なwheel、checkpoint等347,170,775 byteと途中取得のGEFF 373ファイルを削除した。公開train sourceと必要なcheckpoint等8,475,789 byteだけを残した。今後GEFF ZIPや保存済み予測をローカルへダウンロードしない。
- Colab用ZIPはローカルに既存のmodelとコード・manifestだけから生成する。GEFF export kernelをpush・実行してもoutput ZIP本体はローカルへ取得せず、URLだけを使用する。
- ローカルへ回収する学習成果は各epochの小さなreceiptと最新の再開state、epoch 1と内部選択weight、指標・診断結果に限定する。古い再開stateは新しい状態のSHA確認後に削除する。

- 非公開CPU kernel `kentookumura/exp032-train-geff-export` version 1が完了した。199 sample・4179 fileを含むGEFF ZIPは2,387,014 byte、SHA-256は `917b7d354bab6346cfb94dff6ccc478ad52f94a82effafeb2b7e543b5d344663`。Kaggle出力にHTTPS URLがあることを確認し、ZIP本体はローカルへ取得していない。

## 2026-09-21 Colab CLI実行結果

- ユーザーはZIP、Kaggle短命URL、Colabへの転送、非公開Kaggle CPU kernel、結果回収、再開seedを含む1〜7の作業を許可した。`kentookumura/exp032-train-geff-export` version 1を登録・実行し、competition train GEFFの199 sample・4179 fileだけを2,387,014 byteのZIPとして出力した。SHA-256は `917b7d354bab6346cfb94dff6ccc478ad52f94a82effafeb2b7e543b5d344663`。GEFFはexp015 cacheに含まれない正解データである。出力は非公開とし、ColabがKaggleから直接取得した。CPU kernelはZIP作成だけに使い、GPU学習には使っていない。
- Colab CLIのT4で2 foldを各10 epoch学習した。20件のepoch receiptと最新の再開stateをSHA照合してローカルに回収し、古い再開stateを次の検証後に削除した。学習入力の19,701 cache窓から正解nodeが空の994窓を除外し、18,707窓を対象とした。fold 0の最良はepoch 7、fold 1の最良はepoch 9。モデルmanifest、学習summary、完了markerと9,841,761 byteのtrain archiveを検証した。
- 長時間実行中、Colab CLIが保存したruntime proxy tokenの期限が切れ、計算は進む一方でファイルAPIが404を返した。Colabのassignment一覧から新しいtokenを取得してローカル接続情報を更新すると、epoch 4・5のreceipt回収が再開した。その後は期限切れ前に接続情報を更新した。学習の重みやconfigは変更していない。
- 診断の初回実行は、archiveに含めない選択epochの個別weightを読もうとして停止した。model manifestに記録された `primary_tracker_best.pth` をSHAとepoch情報で確認して使うよう修正した。
- 2回目は正解nodeが空のcache窓を内部閾値計算に含めて停止した。学習で記録した `gt_window_filter_audit.json` をSHA照合し、同じ994窓を内部診断から除外した。事前選択128窓とこの除外集合の重複は0件だった。
- 3回目はローカルへ回収しない中間epochのweightを固定窓診断で要求して停止した。診断対象を保存済みepoch 1と各foldの内部選択weightに合わせた。10 epochすべての学習・内部指標はtraining summaryに残し、固定窓の経時比較はこの2点とした。
- 4回目の診断計算は完了したが、結果回収で更新版 `metrics.json` と学習時の同名ファイルが衝突して停止した。この失敗時に約137 MBの診断archiveを一時的にローカルへ転送したが、一時ディレクトリ終了時に削除され、永続保存はしていない。ユーザー指定を守るため、回収対象から大容量の予測配列を除き、診断archiveのローカル転送を20 MB以下に制限した。学習証拠が変わらないことを確認した上で診断metricsを更新する処理と回帰テストを加えた。
- 5回目の診断は完了し、470,991 byteのarchive、完了marker、診断summary SHA-256 `49b9a721c48dad0f7963905a5f84a361c297d7e06b79c25871ca0edb7abc4855` を検証して回収した。exp015 cache約4.15 GB、GEFF、exp027保存済み予測136,560,816 byteは各sessionでKaggleからColabへ直接取得した。Google Driveの手動配置とNotebookの手動実行は行っていない。
- 固定0.5では選択weightの既知edge recallがepoch 1より6bbaで+3.07ポイント、44b6で+5.73ポイント。教師上の負例予測は各胚で1件増えた。保存済みexp016より両胚のrecallと正解親1位率が低い。予備的改善条件は満たさず、全graph推論へ自動進行しない。CV、公式graph score、LB、submissionは未計測。採用・不採用と実験完了はユーザー判断待ち。
- exp027の1 epochと新runの1 epochのstate SHAは6bba評価foldで一致、44b6評価foldで不一致。後者も固定窓のrecall、教師上の負例予測数、正解親1位率は一致した。ColabとKaggleで完全な重み再現は両foldでは確認できなかった。
