# exp015_oracle_stage_limits

## 概要

- 仮説要約: 固定公開モデルの既知中心、既知辺、母と2娘、最終graph選択を分け、改善余地がどの処理に残るかを診断する。
- 変更点要約: train 199動画の同じforwardから19,701 windowの再利用用feature cache、ILP前候補graph、元candidate ID付き最終graphを保存し、exp012の悪化16件と全件を同じ4段階・5条件で集計する。
- 結果要約: Kaggleでinference version 1とdiagnostic version 6が完走した。中心候補99.286%、候補edge 94.829%、最終edge 91.602%、候補division 40.397%、最終division 9.934%。cacheは19,701 window・4.15GBで完全一致検査を通過した。
- リスク: 公開modelはtrainを学習に含むため独立CVではない。疎いGEFFに未対応の候補をfalse positiveとしない。divisionの既知件数は151件と少ない。
- 次: ユーザー判断によりdiagnosticとして完了した。段階別結果を反映した全体バックログから後続実験を選ぶ。submissionは行わない。

## 正の記録

- 数値、実験status、構造化された実行証拠: [metrics.json](metrics.json)
- route、設定、系譜: [config.yaml](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [requirements.md](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [result.md](result.md)
- 実行中の作業ログ: [SESSION_NOTES.md](SESSION_NOTES.md)

## 実行入口

- 固定公開modelのtrain推論とwindow cache生成: exp015_oracle_stage_limits_inference.ipynb
- 段階別診断: exp015_oracle_stage_limits_diagnostic.ipynb
- 正式なfull runはKaggle Notebookを正とする。
- inferenceはGPU、diagnosticはCPUで別々に実行し、inference outputのcandidate/final graphをdiagnosticへ、window cacheを後続学習実験へ渡す。
- 大容量inference outputはKaggle側に保持し、ローカルartifactへはdownloadしていない。
