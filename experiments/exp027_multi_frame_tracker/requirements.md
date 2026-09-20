# exp027_multi_frame_tracker 要件と実装方法

本書を実装前の契約とする。進捗は `SESSION_NOTES.md`、数値と実行証拠は `metrics.json` に記録する。

## 実験化の入口・引き継ぎ・承認

- 入口と承認: 2026-09-20にユーザーが `multi_frame_tracker` の実装を依頼し、方式の確認後「近傍に限った3時点attentionでお願いします」と選択した。
- 移行元候補: `multi_frame_tracker`。上位仮説 `HYP-20260910-10` のうち、予測軌跡を入力せずに前後3時点の固定候補を同時参照する部分だけを検証する。
- この実験だけで判断できない範囲: 位置履歴、長い画像窓、欠測node、軌跡断片の接続。3時点の結果を多時点手法全体へ一般化しない。
- 親と比較対象: [exp016 requirements](../exp016_frozen_image_encoder/requirements.md)、[config](../exp016_frozen_image_encoder/config.yaml)、[result](../exp016_frozen_image_encoder/result.md)。入力cacheは [exp015](../exp015_oracle_stage_limits/) の保存済み窓。公開trackerを外部対照として保持する。
- 参照論文: Gallusser / Weigert, [Trackastra: Transformer-based cell tracking for live-cell microscopy](https://arxiv.org/html/2405.15700v2)。複数時点の検出点をtokenとして扱う着想のみ参照する。論文のencoder–decoder、parental softmax、全時点の教師、窓間平均を再現する実験ではない。
- 観測事実: exp016は隣接2フレームからlogitを作り、逆方向にも同じペアを採点する。公式combined scoreは0.9120545013で、胚別の両方の改善は未確認。cacheは199動画・19,701窓。3時点の効果と費用は未測定。
- 仮定: 前フレームの近傍点を参照すれば、中央ペアだけでは曖昧な接続を減らせる。候補にない細胞は回収しない。

## 判断履歴

- 判断履歴: 2026-09-20のバックログ追加時は方式・特徴の出所・境界処理・診断閾値が未決だった。同日の実装依頼後、ユーザーは近傍に限る3時点attentionを選択した。下記の固定設定は初回比較のための実装値であり、評価胚の結果によって変更しない。
- 2026-09-20のKaggle version 3で、4モデル・3epoch・外側全窓評価の保守的推定が34.52時間となり、12時間gateで停止した。ユーザーは次の実行として1epoch・各胚64窓評価の予備実験を選択した。全件比較とは区別して記録する。

## 手法契約

- 入力: 各動画の `t−1,t,t+1` にある検出点の固定画像特徴32次元、位置特徴32次元、物理座標、有効mask、相対時刻。中央 `t,t+1` の特徴はその中央ペアのcacheを使う。追加する `t−1` は直前の `t−1,t` cacheのsource側を使い、元の画像窓を保持する。重なる `t` の候補IDと座標が両cacheで一致しなければ停止する。
- 予測対象: 中央 `t→t+1` の通常接続と母娘接続。追加時点のGEFFを教師や入力として使わず、未知の候補を新しい負例にしない。
- 出力: exp016と同じ中央ペアの全候補対logit。位置や欠測nodeを生成しない。
- 損失: exp016のsource軸softmax、focal weighted binary cross entropyと同じ中央ペアのactive pair mask。部分注釈で未知の接続を負例に含みうる限界を継承する。
- モデル: 3時点の点を1つの集合にし、物理距離15 µm以内の全近傍に各点がattentionする。近傍選択は教師や予測接続を使わない。同じ時刻の点も対象に含め、相対時刻は学習可能なembedding、位置は既存32次元特徴と物理座標で表す。attentionを4層通した後、exp016のpair MLPで中央ペアの全組を採点する。計算上の安全上限は1024件とし、実際のattention件数は各窓の半径内点数に合わせる。半径は `config.yaml` に固定し、評価胚で調整しない。
- 初期化: 公開primary trackerのprojection、normalization、attention block、pair MLPを形状一致で移植し、新しい時刻embeddingは0で初期化する。2時点対照と3時点条件に同じ初期化を使う。attentionの通信範囲は親モデルから変わるため、親モデルと同じ関数として扱わない。
- 比較: 同じ新構造で `t−1` をmaskした2時点条件と、3時点条件を各2foldで比較する。中央ペアの候補・教師・分割・loss・学習量を同一にする。全件比較は3epochを想定したが、Kaggle version 3のruntime gate超過を受け、ユーザーが1epoch・評価胚ごと64窓の予備実験を承認した。exp016保存済みモデルと公開trackerを再学習しない。
- 推論: 中央ペアにつき1回だけ窓を作る。逆方向のlogitは同じ3時点の埋め込みから中央ペアを逆順にpair MLPへ渡し、従来のbidirectional fusionへ接続する。最初のペアでは前フレームをmaskし、短い動画・動画境界を越えない。重複得点は生成しない。secondary tracker、閾値、ILP、graph repair、公式評価をexp016と同じに保つ。
- 処理単位: 1動画内の3時点窓。分割・結果は胚別。公開画像重みの学習来歴が評価胚を含む可能性があるため、独立CVとは呼ばない。
- 実装区分: 本リポジトリ内の管理上は `staged-faithful`。Trackastraの全方式ではなく、複数時点の検出点を同時に扱う機構だけを固定特徴・既存教師と復号で比較する。

## 実装方法

- 固定: 公開検出器・画像encoder・正規化統計、2-frame画像入力窓、候補IDと座標、中央ペア教師とmask、胚split、secondary tracker、graph復号、公式評価。
- 変更: primary trackerの点間attentionと入力時点数。前フレームだけをmaskした同構造対照で時点追加の効果を分離する。
- 予備実験: active variant 2、model/config 1、outer fold 2、計4 tracker、booster 0。control再学習なし。全学習窓を1epoch使い、内部validationは全対象を使う。外側評価は各胚64窓を、eligibleな動画から各1窓ずつ、seed 42のSHA-256順位で教師のedgeやモデル予測を見ずに事前選択する。両条件は同じ選択窓を使い、選択manifestとSHAを保存する。
- 実行前: exp015 cache summary・identity・window content SHA、公開source・checkpoint SHA、窓間の候補IDと座標、有効窓数、近傍件数、メモリ、64窓/fold/variantのruntime benchmarkを検査する。Kaggle GPU残量とActive Sessionを確認し、週30 GPU時間とNotebook 12時間gateを守る。
- 予備実験の診断: 同じ中央ペアのknown positive edge recall、active pair内の誤接続数、division parent recallと有効件数を、固定した各胚64窓で測る。両胚・両条件を比較するが、1epochと部分評価のため全件の改善や全graphへ進む条件の成立とは扱わない。有効な分裂親がない胚では分裂条件は未判定として保留する。全件学習と全件前段診断を行うかは予備結果と費用を示してユーザーに判断を仰ぐ。
- 全graph後の成功条件: 両胚の公式combined scoreが同構造2時点対照より改善し、division Jaccardが悪化せず、全対象を予算内で処理する。adjusted edge Jaccardとexp016との差も別に示す。
- 停止: cacheや候補の不整合、画像側の更新、fold間の漏れ、NaN/OOM、12時間または週30時間のgate超過、単体診断の不成立。予算で停止した場合は情報の有効性を否定しない。条件未達なら公式score未測定と記録する。
- 単体評価の限界: 長い軌跡、ILP後の選択、公式combined scoreは分からない。公式scoreを主張するには全graphと公式評価器をKaggleで実行する。

## 探索幅とpivot判定

- 今回は入力時点数とattentionの通信範囲を変え、2時点対照を同じ新構造で学習する。半径は探索せず、近傍数は半径内の実測点数を使う。予備実験の1epochと部分評価はruntime gate超過後のユーザー承認による縮小であり、精度を見て選んだ設定ではない。
- exp024のepoch変更やexp025の後処理変更を混ぜない。2回連続の小変更だけで多時点情報の有効性を判断せず、必要な場合はtarget・出力・復号・処理単位を変える候補と比較する。
- この候補は処理単位を2時点から3時点へ変える比較であり、追加時点を隠した対照と保存済みexp016を区別して解釈する。

## 再現性・リスク

- 正解軌跡を入力へ使うこと、別動画の特徴混入、画像側の再学習、未知点の一律負例化、外側胚での半径・設定選択、時間を飛ばす提出辺、候補点の削減、検出器の再学習、Kaggle submissionは行わない。
- 前フレームは前の2-frame画像窓から取るため、単一frame特徴と読み替えない。窓依存特徴の差、時間端、空候補、密な領域の近傍数と計算量、GPU費用を記録する。
- 公開重みの来歴、疎いGEFF教師、親から変わるattention構造、二つのcacheを読む費用が主な解釈上の制限である。
- 学習重み・予測・graph、入力cache、featureのSHAと有効件数を `metrics.json` に記録する。途中の実行経緯は `SESSION_NOTES.md` に残す。
- 成功・失敗にかかわらず、実験の完了・採否はユーザーが判断する。後続の4時点化、位置履歴、分裂組モデルを自動で始めない。

## 受け入れ基準

- [ ] 候補のID・根拠・比較・固定事項・停止条件を移行し、backlog索引を更新する。
- [ ] 同構造2時点対照と3時点条件のtrain Notebookとmodel manifestを実装する。
- [ ] cache境界、教師の一致、mask、局所attention、逆方向出力のテストを通す。
- [ ] `validate-exp`、`check-exp`、`test-exp`とJupytext変換検査を通す。
- [ ] Kaggle runtime・quota確認後に初回full trainを実行し、両胚の前段診断を記録する。
- [ ] 進行条件成立時だけ固定graph推論と公式評価を実行する。
- [ ] 結果と未解決事項をユーザーへ提示し、完了・採否の判断を得る。

## 保存済みモデルで過去入力を除く追加診断（2026-09-20承認）

- ユーザーは、保存済みの同じ3時点モデルで過去入力あり・なしを比較し、親順位・確率・正誤変化を確認する提案について「これに進んでください」と依頼した。同じ実験内のdiagnostic Notebookで行う。
- 入力: train version 4の4重み、model manifest、事前選択manifest、training summaryとexp015 cache、同じGEFF教師。manifestと各重みのSHAを固定し、各胚64窓の選択を変えない。画像特徴・候補・教師・座標・重みは固定する。
- 比較: 保存済み2時点モデル、保存済み3時点モデルの過去入力あり、同じ3時点重みで過去点だけをmaskした推論の3条件。2foldで計4重みを読み、学習・optimizer更新は0回。
- 出力: 各既知正例edgeの親順位（同点は元の候補index順）、親確率、最大競合親との差、予測親ID、確率0.5超の回収、分裂の有無、変位と候補数を保存する。窓別の全logits・候補ID・教師も保存し、事後集計のための再推論を避ける。
- 診断: 既知edge recall、正しい親の1位率、閾値ごとのrecallとactive pair内の教師負例予測数、同じedgeの救済／悪化、1位のまま0.5を跨いだ件数を両胚別に比較する。動画を単位としたpaired bootstrapで差の不確実性も示す。閾値の選択・再学習・全graph評価には使わない。
- 受け入れ: 2時点と3時点の元条件が、version 4の既知edge・誤予測・分裂件数とloss許容誤差を再現することを先に検査する。入力SHA不整合、重み変更、非有限値、再現不一致、runtime gate超過では停止し、不一致を記録する。
- 実行: Kaggle T4で最初の診断を行う。各batchの実測から保守的な残時間を計算し、診断Notebookを1時間以内、既存の週30 GPU時間以内に制限する。
- 解釈: maskは学習時の入力分布と異なり得る。過去入力への依存と有用性を調べる介入であり、別々に学習した2時点／3時点モデルの優劣とは区別する。部分注釈の教師負例を真の誤接続と断定しない。公式score・全件への有効性・実験採否は確定しない。

## 内部validationで確率と閾値を確認する追加診断（2026-09-20承認）

- ユーザーの「次に進んでください」を受け、保存済み4重みと同じ3推論条件で学習側の内部validation全窓を評価する。分割manifestを固定し、gradient update・外側評価の動画と混ざらないことを検査する。学習・重み更新は0回。
- 入力・教師・batch順・既定閾値0.5を元の内部validationに揃え、2時点と3時点の保存済み内部指標の再現を確認する。内部validationは7動画693窓と12動画1,156窓。空GT除外は元と同じ中央ペアの2フレームに適用する。
- 閾値選択はfoldごとに、2時点モデルの内部validationで0.5を超えるactive pair内教師負例数を上限とする。各条件で、この上限を超えない最小閾値を内部教師負例確率の順序統計から一意に決める。判定は確率が閾値を厳密に超える場合とし、同点をまとめて扱う。正例recallや外側評価による選択をしない。2時点モデルにも同じ規則を適用する。
- 内部で確定した閾値を保存済みdiagnostic v1の外側128窓へ適用する。元の0.5と各条件の内部選択閾値を併記し、既知edge回収数・教師負例予測数・親1位率・分裂回収数を両胚別に比較する。
- 確率分布は既知親の確率と、既知親が一つの子に限った最大確率と正答率を示す。未知接続を真の負例と呼ばず、内部の教師負例数を揃えても外側で同数になるとは仮定しない。内部validationでcheckpoint選択済みという制限も残る。
- Kaggleで内部推論を実行し、元の診断と同じ1時間gateを使う。教師正例・active教師負例の確率をlossless NPZで保存する。外側の集計は保存済み予測からCPUで行う。公式score・全graph復号の閾値採用は判断しない。
