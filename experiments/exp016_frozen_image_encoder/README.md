# exp016_frozen_image_encoder

## 概要

- 仮説要約: 公開画像encoderの出力と検出候補を固定し、primary `SimpleNodeTransformer`だけを再学習すれば、画像forwardを繰り返さずにtracker変更を比較できる。
- 変更点要約: exp015の固定feature cacheを入力にし、公開sourceの5 µm greedy one-to-one対応と既存lossを保った2方向の胚holdout学習を行う。後半も同じcacheから全候補のprimary/secondary edgeを再計算し、公開trackerのILP前graphがexp015と199件すべて一致した場合だけ、fold別primary trackerと固定ILP・graph repairによる公式指標を比較する。main画像encoderは再実行しない。
- リスク: 公開checkpointの学習来歴にtrain 199動画が含まれるため独立CVではない。train段階の候補間edge指標だけでは公式combined scoreを判断できない。
- 次: version 3で3 epoch × 2foldのtrain段階、Colab T4で199動画・19,701 windowのcache replay、Kaggle CPUで固定graph repairと公式評価を完了した。公開trackerのcandidate graphは全件完全一致した。公式combined scoreは全体で0.000274上昇したが、44b6で0.002446低下し、今回の再学習だけによる両胚での精度改善は確認できなかった。ユーザー判断により比較ベンチマーク実験を完了し、fold別再学習trackerと評価結果を後続のtracker改善の比較基準として残す。提出用モデルは未選択。

## 正の記録

- 数値、実験status、構造化された実行証拠: [`metrics.json`](metrics.json)
- route、設定、系譜: [`config.yaml`](config.yaml)
- 実装前の要件、実装方法、受け入れ条件: [`requirements.md`](requirements.md)
- 証拠への参照、結果の解釈、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)

## 実行入口

- 学習 notebook: `exp016_frozen_image_encoder_train.ipynb`
- Kaggle正規ベンチマーク: `exp016_frozen_image_encoder_inference.py` / `exp016_frozen_image_encoder_inference.ipynb`。199動画のcache replay、固定graph repair、公式評価までの通し再実行用として保持する。今回の比較値はこのNotebookの通し実行ではなく、Colab replayとKaggle repair・公式評価の分割実行から得た。
- Kaggle評価専用Notebook: `exp016_frozen_image_encoder_official_eval.py` / `exp016_frozen_image_encoder_official_eval.ipynb`。SHA検証済みの回収graphから、graph repairまたは公式評価だけをCPUで再開する。
- Colab代替実行: `exp016_frozen_image_encoder_colab_inference.py` / `exp016_frozen_image_encoder_colab_inference.ipynb`。tracker・ILP replayの実行環境を移すための経路であり、Kaggle正規ベンチマークを置き換えない。
- Colab CLI smoke helper: `colab_smoke_execute.py`
- Colab CLI 2動画benchmark helper: `colab_benchmark_execute.py`
- Colabへ手動配置するbundle: `artifacts/colab_bundle/exp016_colab_bundle.zip`
- Kaggle 準備と実行: [`SESSION_NOTES.md`](SESSION_NOTES.md)の予定を埋め、`kaggle-review-exp`と`kaggle-platform`の手順に従う
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 表記

用語は`AGENTS.md`の規則を正とし、公式資料、参加者の説明、論文、既存コードで実際に使われている専門用語を優先する。コンペ固有の略語とリポジトリ内の管理用語は`docs/glossary.md`で定義する。
