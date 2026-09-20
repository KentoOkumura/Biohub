# exp030_frame_self_attention_diagnostics 要件と実装方法

実装前の契約と受け入れ条件の正。進捗は`SESSION_NOTES.md`に記録する。

## 実験化の入口・引き継ぎ・承認

- 実験化の入口と承認: 2026-09-20の「frame_self_attention_diagnosticsを実装してください」。同日の確認への回答は「診断だけ先に実装」。
- 移行元backlog: N/A。この診断は`frame_self_attention`の学習・公式graph評価契約を引き受けない。後続の契約は`exp031_frame_self_attention_spatial`へ移行した。
- 対応する上位仮説: `HYP-20260920-02`。この実験は機構の妥当性と計算費用を事前に調べる。
- 上位仮説のうちこの実験が検証する範囲: 旧checkpoint互換性、フレーム内Self-Attentionのshape・mask・勾配、実cache windowでのGPU時間とメモリ。
- この実験だけで上位仮説を判断できるか: いいえ。予測精度は測定しない。
- 上位仮説の判断に残る検証: 固定cache・教師・2方向胚holdoutでのModel A/B/現行の3エポック学習と、同じ候補・復号による胚別公式graph評価。
- 親実験: [exp016](../exp016_frozen_image_encoder/requirements.md)。実装根拠は[フレーム内Self-Attentionの設計](../../docs/surveys/biohub-node-self-attention-design_20260920.md)と`exp031_frame_self_attention_spatial`の実験契約。
- 固定するもの: 公開検出器・画像encoder・正規化、exp015 cache、64次元入力、座標の尺度、旧Cross-Attentionの逐次更新、Pair MLPとchunking、旧tracker checkpoint。
- 変更するもの: 共有2層TransformerEncoderを両フレームへ個別に適用する選択肢と、Cross-Attentionを使うかのflag。
- 最小の反証可能な検証: 旧checkpointを現行モードへ厳密に読み込み、同一logitsを得る。Model A/BはSelf Encoder初期値を共有して保存復元できる。代表的なwindowと最大級windowで3構成のforward、backward、peak GPU memoryを測る。
- 成功条件: 契約テストと旧モデルの出力一致を満たし、Kaggle T4で3構成の代表・最大windowが有限なlogitsと勾配を返す。費用は後続実験の週30 GPU時間・推論12時間gateの判断材料として提示する。
- 停止条件: 旧checkpoint不一致、mask/空集合/順序/勾配の破綻、Kaggle OOMがあれば記録して後続のフル学習へ進めない。
- 実行しないこと: 3エポック学習、checkpoint選択、graph復元、公式score計算、Kaggle submission。synthetic lossの値を精度と解釈しない。
- 未決事項: なし。
- backlog記録から解釈を変更した箇所とユーザー承認: 元候補全体の実験化ではなく先行診断のみ。2026-09-20の確認回答で範囲を確定。

## 判断履歴

- 2026-09-20: `frame_self_attention`のModel Bは現行の逐次Cross-Attentionを維持する、とユーザーが設計時に確認。
- 2026-09-20: 候補は全体バックログへ追加されたが、学習・評価は未承認。
- 2026-09-20: ユーザーは`frame_self_attention_diagnostics`の実装を依頼し、診断だけ先に行うと回答。

## 手法契約

- 依頼原文: 「frame_self_attention_diagnosticsを実装してください」
- 期待する成果: モデルの実装互換性とKaggle上の計算費用を、後続の学習判断に使える形で測る。
- input: exp015の固定公開画像特徴32次元、既存の位置特徴32次元、両時刻のcell座標と実cell mask。
- target / objective: 学習教師は作らない。backward費用測定のみlogitsの二乗平均を微分する。
- output: 両フレームのcell pairに対する活性化前edge logits。diagnostic JSONには互換性、時間、GPU memoryを保存。
- loss: 診断ではsynthetic gradient用の二乗平均のみ。後続学習で使うexp016のfocal weighted BCEを変更しない。
- decode: 実施しない。後続の本比較ではexp016の固定secondary tracker、ILP、graph repair、公式評価器を使う。
- context unit: 隣接2フレームのcell集合を別々にSelf-Attentionし、必要な構成のみCross-Attentionへ渡す。
- 実装区分: このリポジトリ内の管理用語で`staged-faithful`。モデル機構は設計どおり実装し、精度比較段階は後続に残す。
- 省略する機構と理由: 予測graph生成・評価は計算費用gate前なので省略する。
- proxyで検証できない主張: N/A。診断のsynthetic gradientから精度は判断できない。
- proxyの場合のユーザー承認: N/A。
- この実験が支持 / 棄却できる主張: 実装の旧重み互換性、Self Encoderの機能、代表・最大windowでのGPU実行可能性。
- この実験では判断できない主張: Model A/Bの接続・分裂精度、両胚の公式score、フル学習時間の確定値。

## 実装方法

- アプローチ: `frame_attention_model.py`に旧module名を保つ変更版を置き、Notebook内にも同じsourceを展開する。configのflagで現行、Model A、Model Bを作る。
- inputの実装箇所と変換: diagnostic Notebookがexp015 cacheのidentityを検証し、画像・位置特徴を連結、grid座標へ既存downsample倍率を掛ける。
- target / objectiveの構築箇所: diagnostic Notebookのbackward benchmarkだけでlogits二乗平均を計算する。
- outputの生成箇所と表現: `SimpleNodeTransformer.forward`がcell pair logitsを返し、Notebookが時間・memoryとともにJSONへ書く。
- lossの実装箇所: diagnostic Notebookの`benchmark`。exp016の学習lossは変更しない。
- decode / postprocessの実装箇所: N/A。診断JSONの集計のみ。
- context unitを保つ処理箇所: `SimpleNodeTransformer._self_encode`で共有Encoderを各フレームへ個別適用する。
- 変更するファイル / component: この実験のconfig、model source、Jupytext source・Notebook、契約test、記録。
- 固定事項を保つ確認方法: 公開sourceとcheckpointのSHA、cache identity、旧checkpointの`strict=True`復元、現行モードのlogits完全一致を確認。
- 参照sourceとの一致を確認するテスト: 旧moduleのstate keyとeval出力一致。Notebookの埋め込みmodel source一致。
- 承認済み差分を確認するテスト: 3構成のflag、mask・padding・空集合・順序・gradient・保存復元を検証。

## 探索幅とpivot判定

- 変更class: `mechanism`。このリポジトリ内の管理用語で、フレーム内cell間の注意機構を追加する。
- 同じ親 / familyで連続した小改善実験数: 診断のみのため該当しない。
- positiveなoracle headroom / coverage / 誤差非相関性: この診断では測定しない。
- 比較したtarget、output、decode、context unitを変える案: 後続の精度比較は既存の入力・教師・復号を固定する。
- 小改善の継続またはpivotを選ぶ根拠: まず互換性とGPU費用を測り、結果からユーザーがフル比較を判断する。
- `kaggle-idea-forge` の実行要否と根拠: 不要。今回の依頼は新しい案の発想ではなく既存設計の診断。

## 再現性・リスク

- seed policy: 42を固定し、Model A/Bの新規Encoderを同じ初期stateにする。
- stochastic 処理の有無: backward benchmarkではdropoutがある。CUDAでbitwise再現を主張しない。
- stochastic feature generation / augmentation / seed bagging の有無: なし。固定cacheのみ。
- 並列処理と乱数の関係: benchmarkは単一processで順次実行。
- CPU/GPU runtime と deterministic flags: Kaggle T4を正とし、時間はCUDA synchronizeで測る。deterministic algorithm強制はしない。
- train cache / test feature regeneration の SHA 記録方針: exp015 cache summaryとidentityを再計算・照合。test特徴は作らない。
- model manifest / prediction / submission SHA 記録方針: checkpointは生成しない。公開初期checkpointと変更sourceのSHAを記録。
- Kaggle package bootstrap 確認方針: Notebook、config、埋め込みsourceと公開dataset・exp015 cacheの参照をprepare後に確認。
- リークリスク: 外側胚の正解・指標を設定選択に使わない。公開重みの学習来歴は独立CVではない。
- CV/LB 不一致リスク: 診断でCV・LBは測らない。
- ランタイム/メモリリスク: Self-Attentionはcell数の二乗の費用を加える。最大windowのOOMを記録して停止判断へ使う。
- 再現性リスク: GPU時計・他process負荷とdropoutで時間やgradientは変動する。複数回の中央値を記録する。
- 手法忠実性リスク: 診断のsynthetic gradientは実教師での学習時間・精度を直接示さない。
- 過度な縮小 / proxy化リスク: 最大windowでcell数を減らさない。OOM時は明記する。

## 受け入れ基準

- [x] 旧checkpointの厳密読み込みと現行logits一致をKaggleで確認した。
- [x] Model A/BのSelf Encoder初期state一致、厳密な保存復元をKaggleで確認した。
- [x] 3構成のlogits・有限gradientと、A/Bのmask・空集合・順序をKaggle上で確認した。ローカルPyTorch testは環境不足でskip。
- [x] 変更版モデルの実装とNotebook埋め込みが一致する。
- [x] Jupytext同期、validate-exp、check-exp、test-expが通る（PyTorchを要する動作テストはローカル環境でskip）。
- [x] Kaggle T4で代表・最大windowのforwardとbackwardの時間・peak memoryを記録した。
- [x] 2026-09-20にユーザーが診断実験の完了を判断した。Model A/Bの採否と公式graph評価はこの診断の対象外とし、後続のexp031で扱う。
