# Kaggle 実験テンプレート

Kaggle コンペの調査、実験、検証、提出、記録を一貫して管理するためのテンプレートです。

実験化前の未着手候補と検証中の上位仮説は`backlog/`、実験化を承認した作業は`experiments/expXXX_title/`で管理します。実装前の要件と実装方法は同じ実験の`requirements.md`に残します。コンペ単位の設定は`project.yml`、現在の戦略は`backlog/KAGGLE_DIRECTION.md`、実験比較は`experiment_summary.md`、完了した調査・判断履歴の検索入口は`docs/surveys/README.md`に集約します。

この README は人間向けの入口です。エージェント向けの詳細ルールは `AGENTS.md`、作業別の参照入口は `docs/agent-playbooks.md`、実際の手順は各 `.agents/skills/*/SKILL.md` を参照してください。

## クイックスタート

必要なもの:

- Python 3.11 以上
- `uv`
- `task`。未導入の場合は `make` で代替できます
- Kaggle を操作する場合は `uv run kaggle auth login`、`~/.kaggle/access_token`、実行環境のsecret store、またはlegacy `~/.kaggle/kaggle.json`による認証。client別の対応方式は`kaggle-platform`の認証設定を参照します

初回セットアップ:

```bash
uv sync --locked --extra dev
task validate-template
```

検証範囲の判断は`AGENTS.md`の運用ルールを正とします。

実験単位の確認コマンド:

```bash
task check-exp EXP=expXXX_name
task test-exp EXP=expXXX_name
```

skillの確認コマンド。`check-skills`は各`SKILL.md`のfrontmatter・名前・本文、存在する`agents/openai.yaml`のUI metadata、skill内のPythonコードを検査します。`agents/openai.yaml`自体は推奨ファイルなので、存在しないskillも許可します。

```bash
task check-skills
```

ルートテストのファイル指定とテンプレート検証のコマンド例:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev --extra notebook pytest -q tests/test_target.py
task validate-template
```

共通テストと全件検証のコマンド。全件検証では、共通テストと実験固有テストを収集する前にNotebook依存も同期します。

```bash
task test-common
uv sync --locked --extra dev --extra notebook
task test
```

Notebook や Streamlit アプリを使う場合:

```bash
uv sync --locked --extra dev --extra notebook
uv sync --locked --extra dev --extra app
uv sync --locked --extra dev --extra notebook --extra app
```

`kagglehub` を使う Kaggle Platform の操作が必要な場合:

```bash
uv sync --locked --extra dev --extra kaggle-platform
```

`task`コマンドが使えない環境では、失敗する`task`を先に試さず、同名の`make`ターゲットを使います。Makefileは原則として`.venv/bin/...`を直接呼び、`uv`を使うターゲットには書き込み可能な`UV_CACHE_DIR=/tmp/uv-cache`を渡します。先に`uv sync --locked`を実行してください。

```bash
make validate-template
make validate-exp EXP=expXXX_name EXTRA_ARGS="--allow-todo"
```

## このテンプレートで管理するもの

| 領域 | 管理する内容 |
| --- | --- |
| コンペ設定 | `project.yml` に competition、data、defaults、submission、metadata、runtime.kaggle を記録。データ保存先は`data.raw_dir`などの`data`設定、`paths`はautomationが参照するexperiments、docs、submission historyの保存先 |
| 検証中の上位仮説 | `backlog/KAGGLE_DIRECTION.md` で複数の未着手候補・実験と残っている問いを対応付ける |
| 未着手候補 | `backlog/KAGGLE_DIRECTION.md` を索引、`backlog/<candidate>.md` を候補詳細の正として管理 |
| 実験計画 | `experiments/<exp>/requirements.md` に要件、実装方法、受け入れ条件を記録 |
| 実験コード | `experiments/<exp>/` に `config.yaml`、`settings.py`、実験に必要なtrain/inference/audit/diagnostic notebook、記録ファイルを配置 |
| 共通コード | 複数実験で使う処理を `src/` に集約 |
| 公開Notebook資料 | 取得した公開Notebookとmetadataを `docs/notebooks/` に保存 |
| 調査レポート | 完了した実験調査、モデル説明、OOF／結果EDA、外部調査を `docs/surveys/` に集約し、上位仮説・実験番号・種類・トピック別の生成索引で検索 |
| 調査コード | その場限りの分析コードと生の表・図を `studies/` に保存 |
| 提出管理 | `submission.csv` の形式検証、Kaggle Notebook 実行、提出履歴の記録 |
| 実験比較 | `metrics.json`、`experiment_summary.md`、`SUBMISSIONS.md` で結果を追跡 |

## 機能一覧

- 新規実験ディレクトリをテンプレートまたは既存実験から作成できます。
- 複数の未着手候補・実験を共通の上位仮説IDで追跡できます。
- 実験の`requirements.md`に仮説、実装方法、成功条件を明文化できます。
- `project.yml`、各実験の`config.yaml`、新規形式の`requirements.md`を検証し、TODOや設定漏れを検出できます。
- 実験契約に必要な Kaggle Notebook の実行、提出形式チェックを `task` で統一して実行できます。
- Kaggle 実行用の notebook ディレクトリと `kernel-metadata.json` を生成できます。
- 実験結果と提出履歴を Markdown と JSON に記録し、比較表を更新できます。
- 完了した調査レポートを上位仮説・実験番号・種類・トピック別に自動索引化できます。
- Streamlit アプリで実験結果や OOF 分析を確認できます。
- Codex 用の repo-local skills を `.agents/skills/` に置き、Kaggle 作業の手順を共有できます。

## エージェント用 skills

このテンプレートには、Kaggle 作業で使う repo-local skills を `.agents/skills/` に同梱しています。Codex に依頼するときは、必要に応じて skill 名をそのまま指定できます。

| Skill | 使う場面 |
| --- | --- |
| `kaggle-platform` | Kaggle API、認証、データ取得、`project.yml` 設定、コンペ資料の準備 |
| `colab-notebook-runner` | Kaggle GPU quota が限られる場合の Colab notebook、Google Drive、runtime/session 対応 |
| `kaggle-review-exp` | 承認済み実験の作成、requirementsへの契約移行・実装・実行・記録・レビュー。backlog経由では仮説・backlog更新をStrategyへ引き渡す |
| `kaggle-review` | 学習/推論コード、notebook、OOF、失敗実行のレビュー |
| `kaggle-oof-readout` | OOF、feature importance、feature cache、group別metricsを結合した誤差分析。仮説・候補の保存はStrategyへ引き渡す |
| `kaggle-idea-forge` | 反証可能な候補の独立生成。検証中の仮説やbacklogへ直接保存しない |
| `kaggle-strategy` | 上位仮説、戦略、優先順位を整理し、検証中の仮説とアイデアbacklogを作成・更新・削除する唯一のSkill |
| `kaggle-submit-check` | `submission.csv`、Kaggle Notebook、`kernel-metadata.json` の提出前検証 |
| `kaggle-submit-monitor` | `kaggle competitions submit` 後の scoring 監視と LB 記録 |
| `kaggle-notebook-fetch` | 上位公開 notebook をメタデータ付きでローカル保存 |
| `kaggle-survey-papers` | 関連論文、過去解法、公開 notebook、ディスカッションの調査 |
| `kaggle-discussion-archive` | Kaggle discussion の HTML や本文を Markdown として保存 |

コンペ固有の観点が増えたら、対応する `SKILL.md` に追記します。汎用テンプレートとして使う場合は、コンペ名やドメイン固有のチェック項目を置き換えてください。

## 初期セットアップ

初回は`project.yml`のコンペ固有項目を埋めてtemplate validationを行い、Kaggle認証後にコンペデータを取得します。`data.train_dir`、`data.test_dir`、`submission.sample_file`は`data.raw_dir`内に設定します。`dl-kaggle-comp`は取得したzipを`data.raw_dir`へ安全に展開し、設定した`submission.sample_file`が存在することまで確認します。その後にstrict config validationを行います。詳しい設定項目は`kaggle-platform`の「Repository Template Setup」を正とします。

```bash
task validate-template
task dl-kaggle-comp
task validate-config VALIDATE_ARGS="--expected-competition <competition-slug>"
```

`submission.sample_file`を別の方法で配置済みなら、データ取得は省略できます。初期設定の検証後は、次の「壁打ちから実験まで」に従って、検討を続けるか、未着手候補として保存するか、実験化するかを決めます。

## 壁打ちから実験まで

チャットで仮説や実験案が出ただけでは、リポジトリを変更しません。壁打ち後は、ユーザーの意図に応じて次のいずれかへ進みます。

| ユーザーの意図 | 依頼例 | リポジトリで行うこと |
| --- | --- | --- |
| 検討を続ける | 「もう少し壁打ちしたい」 | ファイルを変更しない |
| 今は実験しないが候補を残す | 「この案をバックログ化してください」 | 上位仮説と未着手候補を記録する |
| 今すぐ実験する | 「この案を実験化して実装してください」 | 形式的なbacklogを作らず、直接実験を作成する |
| 保存済み候補を実験する | 「`backlog/<candidate>.md`の候補を実装してください」 | 候補の契約を実験へ移行する |

### 仮説と未着手候補

「上位仮説」と「未着手候補」は、このリポジトリ内の管理用語です。

- チャット内で出た仮説: 壁打ち中の案であり、保存を依頼されるまではファイルに記録しません。
- 上位仮説: 複数の未着手候補や実験で検証する反証可能な問いです。`backlog/KAGGLE_DIRECTION.md`で対応候補、対応実験、残っている問いを追跡します。
- 未着手候補: 上位仮説の一部を1回の実験として検証できる具体案です。`backlog/KAGGLE_DIRECTION.md`を索引、`backlog/<candidate>.md`を候補詳細の正とします。
- 実験固有の仮説: 1件の実験で検証する主張です。実験化後は`experiments/<exp>/requirements.md`へ記録します。

### バックログ化する場合

ユーザーが「バックログ化」「バックログへ追加」と依頼した場合は、`kaggle-strategy`が同名候補、実装済み実験、終了済みの検証との重複を確認します。そのうえで、既存の上位仮説に属する候補なら同じIDを使い、新しい反証可能な問いなら未使用の`HYP-YYYYMMDD-NN`を発行します。

同じ変更で、`backlog/KAGGLE_DIRECTION.md`の「検証中の仮説」と「未着手バックログ」を更新し、`backlog/_TEMPLATE.md`から`backlog/<candidate>.md`を作成します。候補詳細には、根拠、上位仮説のうち直接検証する範囲、変更するもの、固定するもの、最小検証、成功条件、停止条件、実行しないこと、未決事項を記録します。追加後は既存候補を含めて優先度を見直します。

結果や実装方針に影響する未決事項が残る候補は`検討メモ・設計不可`、必要項目が埋まり未決事項が`なし`の候補だけを`設計可能・実験化未承認`として管理します。

バックログ化だけでは、実験番号の採番、実験ディレクトリ作成、コード実装、Kaggle実行を行いません。

### 直接実験化する場合

ユーザーが実験化を直接承認した場合は、形式的なbacklogを作りません。`kaggle-review-exp`が次に利用できる実験番号を採番し、`experiments/<exp>/`を作成または親実験からコピーします。依頼原文と承認、親実験、根拠、変更するもの、固定するもの、実装方法、最小検証、成功条件、停止条件、実行しないことを同じ実験の`requirements.md`へ記録します。

明示的に紐づける既存の上位仮説がなければ、`config.yaml`の`lineage.hypothesis_id`と`lineage.backlog_candidate`は`N/A`とします。上位仮説を発行するためだけの形式的なbacklogは作りません。

### バックログ候補を実験化する場合

候補が`設計可能・実験化未承認`で未決事項が`なし`なら、対象候補を指定した「実装してください」を実験化承認として扱います。重要な解釈差がある場合だけ、コード作成前にユーザーへ確認します。`検討メモ・設計不可`の候補は、未決事項を解消してから実験化します。

実験化時は、候補詳細の上位仮説ID、根拠、具体的な仮説、固定事項、変更事項、実装方法、最小検証、成功条件、停止条件、実行しないこと、判断履歴を`requirements.md`へ移し、`config.yaml`のlineageと一致させます。移行確認後に、元の候補詳細と未着手バックログの行を削除し、「検証中の仮説」の対応先を未着手候補から実験へ変更します。以後は`experiments/<exp>/`だけを正とします。

### 実験の実行と判断

実装、実験契約に必要な Kaggle Notebook の実行、記録、レビューは`kaggle-review-exp`、Kaggle CLIとkernel操作は`kaggle-platform`を使います。提出物の実ファイルを検証する場合は`kaggle-submit-check`、submit後の監視は`kaggle-submit-monitor`を使います。ライフサイクル全体は`docs/05_workflow.md`、作業別の入口は`docs/agent-playbooks.md`を参照してください。

実験結果だけで、完了、採用、不採用を自動決定しません。比較対象、CV・LB、実行証拠、未解決事項を整理し、ユーザーの判断後に確定します。1件の実験が終了しても残っている問いがあれば、上位仮説は検証中のままです。関連実験の証拠が揃った後、ユーザーが支持、棄却、保留を判断し、結論を`docs/surveys/`へ保存して上位仮説別索引へ反映してから「検証中の仮説」から外します。

## 記録と判断

保存場所、各記録ファイルの役割、実験status、完了・採用・不採用の判断規則は`AGENTS.md`を正とし、このREADMEでは別定義しません。人間向けの横断入口は`experiment_summary.md`と`SUBMISSIONS.md`です。

## データと提出

データ配置は `AGENTS.md`、コンペ設定は `project.yml` を正とします。Kaggle Notebook のフル実行と公式評価を基準にし、local smoke だけで公式スコアや Kaggle 実行完了を判断しません。

Kaggle outputを取得する条件とNotebook-only code submissionの操作手順は`kaggle-platform`を正とします。実際のsubmissionに必要なユーザー承認は`AGENTS.md`を参照してください。

## リポジトリ構成

以下は既定の配置です。`project.yml.paths`を変更する場合、automationは設定後のパスを参照します。既存ファイルの移動、Markdownリンク、`.gitignore`も同じ変更で更新してください。

- `.agents/skills/`: このリポジトリ固有の Codex skills。Kaggle 系スキルはここで管理します
- `app/`: 実験や OOF を確認する Streamlit アプリ
- `backlog/`: 検証中の上位仮説、未着手候補の索引、候補ごとの設計
- `data/`: ローカルデータキャッシュ。Git には入れません
- `docs/`: 公式情報、保存資料、調査レポート。保存先の一覧は [docs/README.md](docs/README.md)
- `experiments/`: 実験ごとのコード、設定、出力、記録
- `experiments/<exp>/artifacts/`: その実験が生成した出力。必要な分類はこの下のサブディレクトリで表現
- `experiments/<exp>/assets/`: その実験で参照する小規模な固定データ
- `experiments/<exp>/tests/`: その実験だけに属するテスト
- `scripts/`: テンプレート作成、検証、提出準備、記録更新用スクリプト
- `src/`: 複数実験で再利用する共通コード
- `studies/`: その場限りの EDA・調査コードと生の表・図。完了した結論は置かない
- `templates/`: 新規実験、survey用テンプレート
- `tests/`: 複数実験やリポジトリ全体に関わる共通テスト
- `tools/`: Git で追跡しない外部ツールの clone やローカル配置

主要ファイル:

- `AGENTS.md`: エージェント向けの最優先運用ルール
- `Taskfile.yml`: 推奨コマンド定義
- `Makefile`: `task` が使えない環境向けの代替コマンド
- `project.yml`: コンペ単位の competition、data、defaults、submission、metadata、runtime.kaggle の正。データ保存先は`data.raw_dir`などの`data`設定、`paths`はautomationが参照するexperiments、docs、submission historyの保存先
- `SUBMISSIONS.md`: submission refを専用列に持つ提出履歴
- `backlog/KAGGLE_DIRECTION.md`: 現在の重点、比較基準、検証中の上位仮説、未着手候補の索引
- `experiment_summary.md`: 実験比較と自動更新される要約
- `docs/surveys/README.md`: 完了した調査レポートの生成索引
- `docs/agent-playbooks.md`: 実験、提出、レビュー、分析の作業別参照索引

## エージェントへの依頼例

壁打ち中の案を残すだけなら、次のように依頼します。この時点では実験化しません。

```markdown
この案をバックログ化してください。

仮説:
カテゴリ単位の集約情報が未知カテゴリへの予測を改善する。

候補:
親実験のvalidationとモデルを固定し、カテゴリ単位のmean/count/std特徴量だけを追加する。

根拠:
カテゴリ別のOOF誤差に偏りがある。

未決事項:
対象カテゴリ列の範囲を決める必要がある。
```

直接実験化を依頼するときは、実験番号を指定せず、仮説、親実験、変更するもの、固定するもの、検証方法、成功条件、停止条件を伝えてください。不足している重要な選択肢がある場合は、実装前にエージェントが確認します。

```markdown
この案を直接実験化して実装してください。

親実験:
expYYY_parent

仮説:
カテゴリ単位の集約特徴量を追加すると、同じvalidation条件でOOFが改善する。

変更するもの:
- カテゴリ単位のmean/count/std特徴量を追加する

固定するもの:
- validation split
- モデルと主要ハイパーパラメータ
- 評価指標

成功条件:
- 親実験よりOOFが改善する
- group別評価に重大な悪化がない

停止条件:
- OOFが改善しない場合、この特徴量構成の追加探索を止める

実行しないこと:
- 同じOOFを見ながら特徴量を追加選択しない
```

保存済み候補を実験化するときは、対象の詳細ファイルを指定します。

```markdown
backlog/<candidate>.mdの候補を実装してください。
```

レビューを依頼するときは、対象実験、見てほしい観点、直近の実行コマンド、エラーやスコアを添えると確認が速くなります。

## 人間が確認すること

- 公式ルール、外部データ可否、提出回数、Notebook の internet/GPU 制約
- `project.yml` の competition、data、defaults、submission、metadata、runtime.kaggle 設定。リポジトリ構成を変える場合は paths も確認
- Kaggle API token や秘密情報が Git に入っていないこと
- CV と LB がずれた場合の原因調査
- 採用する実験、提出する実験、public/private LB の確認
- 大きなデータ、モデル重み、生成物が Git に含まれていないこと
