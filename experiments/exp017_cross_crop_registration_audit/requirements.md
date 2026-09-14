# exp017_cross_crop_registration_audit 要件と実装方法

この文書を実装前の契約、実装方法、受け入れ条件の正とし、実行中の進捗は `SESSION_NOTES.md` へ記録する。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: 直接承認。2026-09-14 のユーザー依頼「これはGPUなしで調べられますか？」へのCPU監査案に続く「それで調査を進めてください」を実験化とKaggle CPU実行の承認として扱う。
- 移行元 backlog: N/A。
- 対応する上位仮説: N/A。既存の検証中仮説への所属を推測せず、今回の調査を独立した診断として記録する。
- 上位仮説のうちこの実験が検証する範囲: N/A。
- この実験だけで上位仮説を判断できるか: N/A。
- 上位仮説の判断に残る検証: N/A。
- 親実験: N/A。予測性能を比較するモデル実験ではない。
- 根拠 / 一次資料 / 参照実装: 公式dataページの `{embryo_id}_{field_of_view}`、Kaggle Discussion [Beware of jumps in ground truth track](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724283) のcrop再構成案と不足metadataの指摘、公式ホストがsuffixをcrop IDと説明した [Can we know which embryo the test data belongs to?](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723694)、既存のZarr frame readerとGEFF readerを参照する。Discussionの再構成可能性は参加者の仮説であり、公式確認済みの事実として扱わない。
- 固定するもの: train 199 crop、同一胚内比較、画像値、GEFF注釈、2D/3Dの縮小率、参照時刻、負例対照、判定閾値を固定する。学習済みdetectorやtrackerは読み込まない。
- 変更するもの: crop対の候補生成、画像登録、負例対照、3D・GEFF確認を新しく追加する。
- 最小の反証可能な検証: 同一胚内の全crop対を5時刻の低解像度画像で探索し、上位対を異時刻・異胚の対照、低解像度3D、GEFF注釈で確認する。
- 成功条件: 同じD4変換と平行移動が複数時刻で安定し、2Dと3Dの正規化相互相関が固定閾値と異胚null分布を上回り、異時刻対照より高い候補を再現可能に列挙できる。GEFF注釈が重なる場合は同じ変換で座標・node ID・edgeの整合性も報告する。
- 停止条件: 199 cropまたは期待shapeが揃わない、frame chunkを正しく読めない、相関の合成データテストが通らない、metadataがGPU/internet無効でない場合は停止する。登録候補が0件でも監査結果として正常終了する。
- 実行しないこと: GPU利用、モデル学習、test推論、提出、画像や注釈からの絶対時刻・crop原点の捏造、同じ対象が写らない隣接cropの接続、非剛体登録、結果を使うtracker変更を行わない。
- 未決事項: なし。crop原点、絶対時刻、crop間の重複保証が公開metadataにないこと自体を監査の限界として扱う。
- backlog記録から解釈を変更した箇所とユーザー承認: N/A。

## 判断履歴

- 2026-09-14: ユーザーが領域ごとの軌跡をつなげて一つの軌跡として学習する可能性を提示した。
- 2026-09-14: 先に同様のDiscussionを確認するよう依頼され、cropをjigsawのように再構成する提案と、絶対時刻・crop origin・transform・永続ID等が不明という指摘を確認した。
- 2026-09-14: ユーザーがGPUなしの確認方法を問い、CPUでの画像登録とGEFF照合を提案した後、「それで調査を進めてください」と実行を承認した。

## 手法契約

- 依頼原文: 「それで調査を進めてください」。直前に示した、同じ停止frameを持つcrop群を手掛かりに、同一時刻の画像が平行移動・回転で重なるかをCPUで調べる案を指す。
- 期待する成果: crop間の共通視野を示す候補対、変換、複数時刻の一致、負例対照、3DとGEFFの確認結果、限界を含む監査結果。
- input: train Zarr画像、各sampleのGEFF `labels`、sample名から得る胚IDとcrop ID。
- target / objective: 学習targetはない。同一胚のcrop対に、同じ空間変換で説明できる共通画像領域があるかを検定可能な証拠として測る。
- output: `sample_inventory.csv`、`freeze_runs.csv`、`phase_screen.csv`、`registration_pairs.csv`、`null_pairs.csv`、`volume_confirmation.csv`、`geff_pair_evidence.csv`、`audit_summary.json`、`artifact_manifest.json`、`top_registration_examples.png`。
- loss: なし。
- decode: 5時刻のZ最大値投影を4倍縮小しFFT位相相関で候補shiftを得る。重なり部分の正規化相互相関で候補を採点し、上位対と完全に同じ非空の停止frame scheduleを持つ全crop対を8種類のD4変換で再評価する。低解像度3Dでは2Dのy/x変換を固定し、z shiftを探索する。
- context unit: 候補生成は同一胚内のcrop対、安定性判定は同じcrop対の複数時刻、3D・GEFF確認もcrop対を単位にする。
- 実装区分: `staged-faithful`。承認されたCPU監査を2D探索、D4絞り込み、3D・GEFF確認に分け、計算量を抑えながらすべて実装する。
- 省略する機構と理由: 非剛体登録と任意の時間offset探索は組合せ数と誤一致が大きく、公開metadataがない初回監査では識別できない。完全に非重複な隣接領域は画素対応がないため画像だけでは接続しない。
- proxyで検証できない主張: N/A。省略範囲を除き、共通視野の有無を直接画像で測る。ただし軌跡統合後のscore改善は本実験では検証しない。
- proxyの場合のユーザー承認: N/A。
- この実験が支持 / 棄却できる主張: 同一胚のcropの一部に、同じ時間基準と剛体的なD4変換・平行移動で再現できる共通視野が存在するか。
- この実験では判断できない主張: 重複しないcropの相対配置、任意の絶対時刻差、非剛体変形、全cropを一意な大域座標へ置けるか、領域をまたぐ全軌跡が復元できるか、tracker学習やLBが改善するか。

## 実装方法

- アプローチ: Jupytext percent形式の `exp017_cross_crop_registration_audit_audit.py` を正の編集対象にし、同名Notebookを生成する。Kaggle CPU、internet無効でcompetition train inputを読む。
- inputの実装箇所と変換: sample探索で胚ID・crop ID・Zarr shapeを検査する。frame chunkはBlosc2で直接展開し、Z最大値投影とblock mean縮小を行う。連続frameの圧縮chunk署名が同じ場合だけbyte比較して停止frameを記録する。
- target / objectiveの構築箇所: `screen_pair` が位相相関候補と重なり相関を5時刻で集計し、`identical_freeze_schedule_pairs` が完全一致する停止frame scheduleの対を選び、`refine_pair` が全shiftとD4変換を比較する。異胚null対と同じcrop対の異時刻相関を判定基準にする。
- outputの生成箇所と表現: 段階ごとのCSV、要約JSON、上位登録例のPNG、全生成物のSHA256 manifestを `artifacts/audit_v1/` へ保存する。
- lossの実装箇所: N/A。
- decode / postprocessの実装箇所: `phase_shift`、`overlap_ncc`、`best_ncc_shift`、`apply_d4` が2D登録を行い、`confirm_volume_pair` が3Dのz shiftを選ぶ。
- context unitを保つ処理箇所: pair keyをソートして一意化し、胚をまたぐ対はnullだけに使う。同一対の時刻別結果を保持してmedianとshift MADを出す。
- 変更するファイル / component: exp017のconfig、契約・記録、audit source・Notebook、実験固有テストだけを変更する。
- 固定事項を保つ確認方法: sample数、胚別件数、shape、frame範囲、CPU/internet metadata、seed、出力schemaを実行時または静的検証で確認する。
- 参照sourceとの一致を確認するテスト: 合成した部分重複画像で既知の平行移動とD4変換を回収し、重なりのないノイズ対の相関が低いことを確認する。
- 承認済み差分を確認するテスト: notebook種別がauditだけ、GPU/internetがfalse、model/loss/submission生成がないこと、3D・GEFF確認と負例対照を含むことを検査する。

## 探索幅とpivot判定

- 変更class: `representation`。cropを独立画像だけで扱う前提を監査し、共通座標へ置ける証拠を探す診断である。
- 同じ親 / familyで連続した小改善実験数: 0。モデル改善実験ではない。
- positiveなoracle headroom / coverage / 誤差非相関性: 実行前は不明。登録を満たすcrop対数とGEFF確認数をcoverageとして測る。
- 比較したtarget、output、decode、context unitを変える案: 登録証拠が成立した場合だけ、次の別実験でsample単位から胚内の接続可能なcrop集合へcontextを広げる。
- 小改善の継続またはpivotを選ぶ根拠: 本実験では選ばない。登録可能性の有無をユーザーへ提示してから判断する。
- `kaggle-idea-forge` の実行要否と根拠: 不要。ユーザーが指定した具体的な検証を実装するため、新規案の発散は行わない。

## 再現性・リスク

- seed policy: null対の選択だけ固定seed 42のlocal generatorを使い、sample順を先にsortする。
- stochastic 処理の有無: null対の抽出だけ。固定seedにより決定的である。
- stochastic feature generation / augmentation / seed bagging の有無: なし。
- 並列処理と乱数の関係: 逐次処理。並列乱数は使わない。
- CPU/GPU runtime と deterministic flags: Kaggle CPUのみ、GPU false、internet false。NumPy/SciPyの決定的な配列演算を使う。
- train cache / test feature regeneration の SHA 記録方針: trainの全87GBをhashせず、sample名・Zarr metadata・対象chunk署名をsortした入力manifest SHAを記録する。出力CSV/JSON/PNGは各file SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: model、prediction、submissionは生成しない。
- Kaggle package bootstrap 確認方針: audit Notebook、config、metricsをprepareし、competition source、GPU false、TPU false、internet false、run-on-pushを確認する。
- リークリスク: train画像とtrain注釈だけの診断である。得た配置をhidden testへ存在すると仮定したり、同じデータで閾値を細かく調整してscore改善と呼ばない。
- CV/LB 不一致リスク: CV/LBを測らない。trainの2胚に登録可能なcropがあってもtestに同じ関係があるとは限らない。
- ランタイム/メモリリスク: 199 crop全体は約87GB。frameを逐次展開し、5+5時刻の64x64 thumbnailだけ保持する。全対の高解像度・3D比較は行わず、3Dはscore上位30対と完全一致する停止frame scheduleの全対に限定する。
- 再現性リスク: 圧縮chunkの署名は停止候補の高速探索であり、候補はbyte比較で確定する。Kaggle kernel versionと生成物SHA取得前は正式な実行証拠が完成しない。
- 手法忠実性リスク: 位相相関は重複が小さい場合や細胞配置が大きく変形する場合に失敗する。正規化相互相関、複数時刻、3Dで誤一致を抑えるが、負の結果は非重複の証明ではない。
- 過度な縮小 / proxy化リスク: 4倍のxy縮小と2倍のz縮小により微細構造は失う。上位候補のGEFF座標確認を併用するが、画像候補がscreeningで落ちた対は高解像度では再確認しない。

## 受け入れ基準

- [x] 手法契約の `input / target / output / loss / decode / context unit` を実装前に固定した。
- [x] 実装区分と実験名がcrop間画像登録のCPU監査を正確に表す。
- [x] `config.yaml` の `lineage.hypothesis_id` と `lineage.backlog_candidate` がN/Aで一致する。
- [x] 既知shiftとD4変換を使う合成データテスト、Jupytext round-trip、`validate-exp`、`check-exp`、実験固有testが通る。
- [x] Kaggle metadataでcompetition source、CPU、TPU無効、internet無効、auditだけを確認する。
- [x] Kaggle CPUで199 cropのauditが完走し、段階別の表、要約、入力・出力SHA、kernel version、Notebook実行時間が揃う。
- [x] 正例判定に2D、異胚null、異時刻対照、shift安定性、3Dを使用し、完全一致する非空の停止frame scheduleを持つ全対をD4・3Dで確認し、GEFFが存在する範囲の一致を別列で報告する。
- [x] 正例と負例の解釈限界をユーザーへ提示し、実験の完了・採否判断を受ける。
