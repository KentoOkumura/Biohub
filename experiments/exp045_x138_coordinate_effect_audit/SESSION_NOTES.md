# exp045_x138_coordinate_effect_audit セッションノート

## 目的

座標head適用有無の作用を、同じ検出IDの接続得点から公式graph評価まで追跡する。

## 開始時の計画と経過

- 2026-09-24: ユーザーが設計確定後に実装を承認。exp043の採用済み最終headを固定し、SHA256順の学習外20動画を対象に実装中。
- 実行予定: 2 arm、head 1個、モデル/config 1組、fold 0、booster 0。controlの再学習なし。
- 2026-09-24時点のKaggle実行: 両胚各1本・2 armの予備確認はversion 4で完了。全20本の主実行はGPU quota不足で未開始。
- 2026-09-24 11:49 UTCごろ `uv run kaggle quota --format json` でGPU残り2.44時間、週45時間中42.56時間使用、refreshは2026-09-26 00:00 UTCと確認。生成metadataはGPU true、TPU false、T4、internet false。exp043推論の公開4本でprediction 539秒に加えgraph後処理が必要で、20本×2 armは2.44時間内で25%の余裕を持って完走できると保証できない。このため本実行はpushせず、同じsourceを2本へ限定したpilot Notebookを先に実行する。

## コマンドログ

- `make new-exp EXP=exp045_x138_coordinate_effect_audit`: 実験雛形作成。
- `make validate-exp EXP=exp045_x138_coordinate_effect_audit`: 要件・config移行後に成功。
- `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile experiments/exp045_x138_coordinate_effect_audit/exp045_x138_coordinate_effect_audit_inference.py`: Notebookソース構文検査に成功。
- `make test-exp EXP=exp045_x138_coordinate_effect_audit`: 選択・対応・既知edge集計の5件に成功。
- Jupytextで `.py` から `.ipynb` へ変換し、`--to py:percent --test` に成功。
- `make check-exp EXP=exp045_x138_coordinate_effect_audit`: 初回はテストのimport順を修正後、不要な雛形 `settings.py` のformat指摘が残った。雛形を削除して再検証予定。

## 2026-09-24 Kaggle pilot version 1

- `make prepare-kaggle-notebooks EXP=exp045_x138_coordinate_effect_audit EXTRA_ARGS='--notebook pilot --run-on-push'` を実行した。metadataはprivate、T4、GPU有効、TPU無効、internet無効、competition trainと固定Dataset 4件を入力に持つ。
- push直前の`uv run kaggle quota --format json`でGPU残り2.44時間とrefresh 2026-09-26 00:00 UTCを再確認した。
- `make push-kaggle-notebook EXP=exp045_x138_coordinate_effect_audit NOTEBOOK=pilot`で `kentookumura/exp045-x138-coordinate-effect-audit-pilot` version 1をpush。`uv run kaggle kernels pull ... -m`でKaggle側metadataを確認し、`machine_shape=NvidiaTeslaT4`、`id_no=135662742`、同じ入力4件を確認した。version 1は推論前の `ModuleNotFoundError: biohub_tracking` で停止。公開repoの `src` をimport前に追加し、推論入力からGEFF symlinkを除いて同じkernel IDのversion 2をpushし、実行ログ監視中。version 2 push直前のGPU残りは2.34時間。competition submissionは行っていない。

- pilot version 2は両胚2 armの推論と後処理まで完了したが、同一ID guardがGEFF node IDは`0..N-1`の連番だと誤認し停止。失敗versionから候補cacheと元GEFFを限定回収し、`44b6_7a302da0`の54,942候補に対して元graphは46,265個の疎なIDを持ち、全46,265行の`[t,z,y,x]`が同じ添字の候補cacheと一致することを確認した。IDの連番要求を外し、保存済み元graphのID集合だけを後処理段階で元候補とみなすよう修正。実測データで新guardを確認し、6件の単体テストが成功。
- pilot version 3のpush前 `uv run kaggle quota --format json`: GPU残り1.84時間、refreshは2026-09-26 00:00 UTC。2動画pilotの再実行に必要な時間はversion 2で約0.5時間、残りに収まる見込み。全20動画実行には不足する。`metrics.json`変更でpackageが一度staleとなったため再prepareし、同じkernel IDのversion 3をpushした。

- pilot version 3は疎なID guardを通過し、両胚2 armの全graph後処理まで実行した。2 µm移動量の検査で停止。失敗versionの候補cacheを調べると、cache座標のy/xはfeature gridではなくnative voxelである。全軸1.625 µmを掛けると見かけの最大6.32 µmだが、正しいnative間隔`[1.625,0.40625,0.40625]` µmでは最大1.83 µm、2 µm超過0件。GT対応と補正量の両方をnative間隔へ修正し、単体テスト6件を再検証した。
- pilot version 4のpush前quotaはGPU残り1.22時間、refreshは2026-09-26 00:00 UTC。version 3の約0.62時間を25%増しで見ても残りに収まる見込み。全20動画実行には不足する。`make push-kaggle-notebook ... NOTEBOOK=pilot`で同じkernel IDのversion 4をpushした。

## 2026-09-24 Kaggle pilot version 4 完了

- Kaggle Notebook `kentookumura/exp045-x138-coordinate-effect-audit-pilot` version 4はT4、internet無効で`KernelWorkerStatus.COMPLETE`。2動画の両armで最終graphと公式評価器を完走した。予測処理にGEFFを渡していない。
- 対象は事前のSHA256順位で固定した`44b6_7a302da0`と`6bba_07e24132`。両armの元検出候補は各54,942件、36,330件で同一、最大補正量はnative spacingで1.83 µm、1.91 µm。
- 2動画の公式評価器集計: adjusted edge Jaccardは補正なし0.889538、補正あり0.916229。edge Jaccardは0.885428と0.899103。division true positiveは両armとも0/3。pilot指標を20本の主評価と混同しない。
- Notebook実測1,360.97秒。残り18本を25%余裕込みで15,310.91秒、全20本で16,671.88秒（約4.63時間）と推定。予備実行生成物は663ファイル、352,450,168 bytes。receiptを`artifacts/pilot/exp045_pilot_receipt.json`に保存し、SHA256は`b92b988ab3002dc295c038bf5d5ede22461f4f471a63536d2ca39cb676b42e51`。
- 完了直後の`uv run kaggle quota --format json`はGPU使用44.39/45.00時間、残り0.61時間、更新`2026-09-26T00:00:00 UTC`。全件見積もりに不足するため本実行をpushしない。予備2本の結果に応じて対象を縮小しない。
- 全件Notebookの大量pair順位処理は既知母IDだけをstream集計するよう省メモリ化した。これは未実行であり、全20本のKaggle実行で検証する。
- pilot Notebookを全件ソースから再生成した際のversion 4との差分は、pilotで呼ばない既知edge順位関数の省メモリ化のみ。Kaggleで実行したversion 4は同kernelの履歴として保持される。
- 最終確認: `make check-exp`、`make test-exp`（7件）、`make validate-exp`、`make check-strategy-docs`、両NotebookのJupytext `--test`、`git diff --check`に成功。全件Kaggle packageをprepareし、metadata検証に成功したが、quota不足のためpushしていない。

## 実装上の確認

- exp043の選択と同じ `SHA256(42:動画名)` 順の各胚11～20位を採り、head manifestの学習20本が各胚1～10位であることを実行時に検証する。
- 推論中はGEFF注釈を読まず、2 arm終了後の固定ID診断と公式評価だけで読む。
- head checkpointとmanifest、公開support pack、公式評価器のSHAを照合する。
- 同じ検出IDに対する元座標を比較し、補正移動量の上限2 µmと時刻・候補数を確認する。
- 旧公開Notebookに由来する長い推論・後処理セルを同じNotebookへ持ち込み、診断関数と段階保存を追加した。親実験には `*_compact_selfcontained_inference.py` がないため、親の正規Notebookと章立て・長さを比較した。親は公開モデル準備・head確認・公開graph後処理、今回はこれに選択、2 arm実行、固定ID診断、公式評価、SHA保存を追加した。薄いhelper呼び出しNotebookではない。
- 座標と特徴取得位置を分けた追加pair診断は、既知辺の得点または候補残存とILP以降の回収に同一辺の差が出た場合のみ対象となる。主実行後に条件と追加GPU見積もりを判定する。

## 2026-09-25 全件実行の開始判断

- ユーザーが「quotaがなくなってもジョブは途中で止まらないので実行してください」と明示。前日のGPU残枠不足による保留を解除し、用意済みの全20本・2 armをKaggleへpushする。動画数や比較条件は変更しない。
- 2026-09-25 09:47 UTCごろの`uv run kaggle quota --format json`: GPU使用44.39/45.00時間、残り0.61時間、更新`2026-09-26T00:00:00 UTC`。想定4.63時間より残枠は短い。ユーザーの指示に従い起動し、Kaggleが受理するかと実行継続を監視する。
- push対象metadata: GPU true、TPU false、NvidiaTeslaT4、internet false、private、`run_on_push=true`。`make validate-exp`、`make check-exp`、`make test-exp`（7件）は再確認済み。competition submissionの承認はない。

- `make push-kaggle-notebook EXP=exp045_x138_coordinate_effect_audit NOTEBOOK=inference` が成功し、`kentookumura/exp045-x138-coordinate-effect-audit-inference` version 1を起動。`uv run kaggle kernels pull ... -m`でid_no `135800352`、T4、GPU true、TPU false、internet false、入力Dataset 4件を確認した。live logsでoffline dependency install開始を確認。

## 2026-09-25 全件version 1の失敗と再開

- Kaggle `kentookumura/exp045-x138-coordinate-effect-audit-inference` version 1は`KernelWorkerStatus.ERROR`。ログではzero/refinedの各20動画が最終graphまで完了し、固定ID診断の最初のGT GEFF読込で`AttributeError: 'tuple' object has no attribute 'node_attrs'`となった。`tracksdata.IndexedRXGraph.from_geff()`がtupleを返す版であり、同Notebookの既存`graph_from_geff` helperは対応済みだったが、GT読込だけ直接呼んでいた。
- 元NotebookのGT読込を同helperへ修正。version 1の`exp045_selection.json`と`runtime_gate.json`をKaggle outputから回収し、SHA256はそれぞれ`ba7eb44348be9ee18a0a15180276dbb92399a5c5995e26ae3b3d667b8cc30daa`、`b5b53aacfae8c67a81bb75be00aaeda6c1809ca2faceaf80f1a3bc75b7ebd85c`。
- 40 graphの再推論を避けるため、同じ実験内にCPU `audit_resume` Notebookを追加した。20動画・2 arm・stage生成物・manifest SHAをguardした後、固定ID診断と公式評価器だけを実行する。新しい予測やcompetition submissionは作らない。
- `make validate-exp`、`make check-exp`、`make test-exp`（7件）、両NotebookのJupytext整合とresumeソースの未定義名検査が成功。GPU quota残量は0.00時間、refreshは2026-09-26 00:00 UTC。CPU NotebookなのでGPU quotaは消費しない。
- `audit_resume` version 1をpushしたが、Kaggleは失敗statusのNotebookを`kernel_source`として受理せず、metadataからsourceを除外した。CPU実行は生成物をmountできず停止した。そこで失敗Notebookのversion 1 outputをKaggle APIから回収し、SHA付きの非公開Datasetへ保存してCPU監査の入力とする。公開設定にはしない。

## 2026-09-25 保存済み出力の回収と診断分割

- Kaggle output APIでversion 1の6,604ファイルをすべて回収し、未完了転送0件を確認。選択manifest・runtime gateのSHAが既記録と一致した。両arm各20動画で候補cache、raw ILP graph、最終graph、各9段階の保存物、raw detector 2,000 frameを確認した。全回収archiveは2,633,840,640 bytes、SHA256 `d0e1a679e91e6f730fcf9e6eeb7d3ab452600368658e1f1df36948e8b77d5b36`。
- 全archiveの非公開Kaggle Dataset転送は自動承認審査で一度拒否された。生成物2.6 GBの外部転送についてユーザーの明示承認を受けて開始したが、速度が数百kB/sで数時間を要する見込みとなり中断。Datasetは作成されなかった。
- 公式評価器に必要な両arm計40最終graph、選択manifest、runtime gateだけを11,776,000 bytesのtarに梱包。SHA256 `8315c5b394c524ffee283561effc231a3b88aa1cac8f7954ec3a71837115f2d1`。ユーザーが承認した非公開Dataset `kentookumura/exp045-x138-coordinate-audit-v1-artifacts`へアップロードした。Kaggleはtarを842ファイルへ自動展開し、全ファイルのmanifest SHA256 `868282ef3065bfb7d30249a38d80234d3681856d9284559d82f2af5ab3b788a2`をNotebookで照合する。
- CPU `official` Notebook version 1はKaggle標準Polarsが公開`tracksdata`の必要な`Float16`を持たず停止。公開support packの固定Polars wheelを強制適用したversion 2は、公開評価器の`open_dataset()`が不要な画像をCUDAへ転送しようとして全動画を採点できず停止。公開評価器が使うGT graphとscaleだけを`open_dataset(load_image=False)`で渡すversion 3をpushした。公式評価器sourceのSHAと`evaluate_run`の指標計算は維持する。
- 段階別固定ID診断には全cacheが必要なため、競技から選択20動画のGT GEFF member 420件だけをローカル取得し、Zarrとして読込確認した。`run_local_fixed_id.py`は保存済み出力と同じ`audit_core.py`の対応・段階集計を使い、初期20動画の同一候補ID guardを通過した。実行中。これは公式評価ではない。

## 2026-09-25 主評価と固定ID診断の回収

- Kaggle CPU `official` version 3は`COMPLETE`。40最終graphの公開評価器による20動画全行、両胚別集計、二度目の同じ評価との一致、receipt SHAを回収した。公開評価器はGT graphとscaleだけを使うため、CPU環境で不要な画像の読込を`load_image=False`にした。scoreは補正なし0.879239、あり0.890348。`metrics.json`を数値の正とする。
- 競技から選択20動画のGT GEFF member 420件をローカル取得した。全memberのmanifest SHAを記録。`run_local_fixed_id.py`は公開GEFFのZarr node・edge配列を直接読み、既存の`audit_core.py`で同一IDの対応と段階集計を実行した。数百万pairの得点は、結果に使う既知母IDだけNumPyで先にfilterしてから同じ順位関数へ渡した。20/20動画のidentity guardと固定ID診断を完了し、receipt・胚別集計・動画別診断のSHAを記録した。
- 公式評価は両胚で上昇したが、元座標で固定した既知edgeの最終回収は44b6で2787→2759、6bbaで7413→7333へ減った。融合後順位または候補残存が変わり、同じedgeのILP以降の回収差が出たものは183件、18動画。追加pair診断の条件は成立した。
- `uv run kaggle quota --format json`でGPU残り0.00時間、更新`2026-09-26T00:00:00 UTC`を確認。要件のquota gateに従い、座標と特徴取得位置を分ける追加pair診断は今回は起動しない。GPU quota更新後の費用条件とユーザーの実験判断を待つ。competition submissionは行っていない。

## 2026-09-26 完了判断

- ユーザーが20動画のscore改善とexp043との差を確認し、「実験は閉じてcommitとpushしてください」と明示した。
- 主評価は補正なし0.879239、補正あり0.890348。固定した公開画像モデル・trackerではhead使用を支持する結果として記録し、独立CVやPublic LBの改善とは扱わない。
- 座標入力と画像特徴の取得位置を分離する追加pair診断は実施せずに閉じる。発動条件183 edge / 18動画と未分離の問いはmetricsとresultに残す。
- 実験statusを`completed`へ更新し、exp045に関係するファイルだけをcommit・pushする。competition submissionは行わない。
