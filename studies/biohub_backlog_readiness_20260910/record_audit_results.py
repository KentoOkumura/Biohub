import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

exp = Path('experiments/exp003_official_metric_audit')
output = exp / 'artifacts/kaggle_v1'
artifacts = output / 'artifacts'
verification = json.loads((exp / 'artifacts/kaggle_v1_verification.json').read_text())
samples = json.loads((artifacts / 'sample_results.json').read_text())
fixtures = json.loads((artifacts / 'fixture_results.json').read_text())
summary = verification['summary']
assert summary['failed_count'] == 0 and summary['sample_count'] == 4
assert len(fixtures) == 9
now = datetime.now().astimezone().isoformat()
current = json.loads((exp / 'metrics.json').read_text())
received = json.loads((output / 'metrics.json').read_text())
for field in ['kernel_id', 'kernel_version', 'url']:
    received['evidence']['kaggle'][field] = current['evidence']['kaggle'][field]
received['evidence']['kaggle']['kernel_source_ids'] = ['kentookumura/exp002-unet3d-expandable-segments-inference']
received['evidence']['kaggle']['log_last_event_seconds'] = verification['kaggle_log_last_event_seconds']
received['evidence']['kaggle']['runtime_measurement'] = 'notebook_runtime_seconds: notebook monotonic timer from config-loaded audit setup to final aggregation; log_last_event_seconds: Kaggle saved log through nbconvert'
received['evidence']['artifacts']['kaggle_output_directory'] = str(output)
received['evidence']['artifacts']['fixture_sha'] = verification['environment']['fixture_sha256']
received['evidence']['artifacts']['notebook_source_sha'] = hashlib.sha256((output / 'exp003_official_metric_audit_audit.py').read_bytes()).hexdigest()
received['evidence']['artifacts']['manifest_verified_file_count'] = verification['verified_manifest_file_count']
received['diagnostic_validation']['official_metric_audit']['per_sample'] = samples
received['diagnostic_validation']['official_metric_audit']['fixture_counts_and_scores'] = verification['fixtures']
received['diagnostic_validation']['official_metric_audit']['fixture_score_difference_count'] = sum(row['difference'] != 0 for row in verification['fixtures'])
received['evidence']['artifacts']['node_and_edge_count_agreement'] = all(
    [r['official_counts'][f'edge_{k}'] for k in ['tp', 'fp', 'fn']] == r['proxy_edge_tp_fp_fn']
    and [r['official_counts'][f'division_{k}'] for k in ['tp', 'fp', 'fn']] == r['proxy_division_tp_fp_fn']
    and r['node_matching_symmetric_difference_count'] == 0
    for r in samples
)
assert received['cv'] is None and received['public_lb'] is None and received['private_lb'] is None
(exp / 'metrics.json').write_text(json.dumps(received, ensure_ascii=False, indent=2) + '\n')
subprocess.run(['make', 'record-exp', 'EXP=exp003_official_metric_audit', 'STATUS=debug_completed', 'NOTES=CPU audit v1 executed: 9 fixtures and 4 fixed samples passed; all real-sample scores and counts agree, 3 synthetic scores differ. Diagnostic only; user completion judgement pending.'], check=True)
rows = ['| 動画 | 公式 | 公開実装 | 差（公式−公開） |', '| --- | ---: | ---: | ---: |']
rows += [f'| `{r["name"]}` | {r["official_summary"]["score"]:.9f} | {r["proxy_score"]:.9f} | {r["score_delta_official_minus_proxy"]:.9f} |' for r in samples]
result = '''# exp003_official_metric_audit 結果

## 結論

固定した公式評価コードと公開DCTTA Notebookの評価関数をKaggle CPUで実行した。人工例9件とexp002の保存済み予測4動画を処理し、失敗・対象の脱落はない。実予測4動画は点の対応、接続と分裂の正解・誤検出・見逃し件数、各スコアがすべて一致した。

人工例では3件でスコア差、さらに重複辺の例で正しい接続の件数差を再現した。したがって、今回の実予測で一致したことを、公開評価実装と公式があらゆる入力で同等だという根拠にはできない。今後の評価に完全な公式処理を使うことを推奨する。

## 実行証拠

- [Kaggle Notebook version 1](https://www.kaggle.com/code/kentookumura/exp003-official-metric-audit-audit?scriptVersionId=1) は非公開、CPU、internet無効で実行。versionと環境の正は[metrics](metrics.json)および[取得metadata](artifacts/kaggle_metadata_v1.json)。
- [取得した実行集計](artifacts/kaggle_v1/artifacts/audit_summary.json)、[動画別比較表](artifacts/kaggle_v1/artifacts/sample_comparison.csv)、[人工例の全結果](artifacts/kaggle_v1/artifacts/fixture_results.json)、[失敗一覧](artifacts/kaggle_v1/artifacts/failures.json)。
- source・fixture・予測CSVのSHAを実行時に照合し、取得後にmanifest内6ファイルのSHAを再照合した。[検証結果](artifacts/kaggle_v1_verification.json)。
- 実行時間はNotebook内計測が約7分36秒、Kaggle保存ログの最終イベントまで約7分58秒。大半はinput探索と依存環境の準備。正確な数値と計測範囲はmetricsを参照する。GPU使用なし。
- 固定例の分裂なしでは公式の分裂指標がNaNとなるため、JSONではnullとして理由のwarningを保存した。有効な接続指標を使い、動画を集計から落とす処理はない。

## 実予測4動画の比較

''' + '\n'.join(rows) + '''

公式summariseによる集計値はmetricsの `diagnostic_validation.official_metric_audit.official_summary` を参照する。個別動画の得点順も一致した。比較したのは一つのモデルの固定予測であり、モデル間の順位の比較ではない。公開Notebook全体の集計処理の互換性まで検証したとは主張しない。

## 人工例で分かった差

| 入力 | 公式の処理・結果 | 公開実装の処理・結果 | 解釈 |
| --- | --- | --- | --- |
| 離れた分岐で両娘の系譜へ到達する例 | 正しい分裂0、見逃し1、得点0 | 正しい分裂1、見逃し0、得点0.1 | 同じ連結成分に母・娘・分岐があることだけでは公式の局所的な分裂条件を満たさない |
| 重複した接続 | 正しい接続6 | 正しい接続7 | この例は誤接続・見逃し0なので得点差は0だが、件数は一致しない |
| 時刻を飛ばす接続 | 接続評価では当該辺を除外。分裂は正解0、誤検出1、見逃し1。得点1 | 当該辺を誤接続1と数え、分裂は正解1。得点約0.957143 | 接続と分裂で処理差が同時に出る。公式の全処理を使う必要がある |
| 子への接続が3本ある例 | 接続評価で2本だけ残し、得点約1.085714 | 余分な接続を誤接続として残し、得点約0.944898 | 追加辺をどう扱うかが異なる |

正しい分裂、合流を含む今回の例、未注釈成分、分裂のない継続、辺のない例では得点が一致した。辺のない例の点対応差7は、公式evaluateが辺なしで早期終了して対応付けを行わないためであり、同じmatching処理の誤差とは解釈しない。

## 制約と残る検証

- 対象の4動画は公開testとtrainに共通し、学習から除いた胚の予測ではない。この得点を交差検証（CV）やKaggleの公開ランキング（Public LB）の値として記録しない。
- 公開例では正しい分裂の回収が0件だったため、難しい分裂を回収した予測に対する実データの評価差は未検証。
- この監査は固定した公式commitに対する照合であり、将来の公式更新や全入力に対する完全な同等性を保証しない。
- 本実験でHYP-20260910-14全体を終了しない。学習・評価を胚で分けた基準予測、checkpoint選択、誤差・段階別回収上限の検証が残る。

## ユーザー判断と次の作業

送信・CPU実行は目的説明後の「実行してください」で明示承認済み。受け入れ基準を満たす実行証拠が揃ったので、この監査実験を完了扱いにすることを推奨する。完了・採否のユーザー判断は未記録であり、実行状態はmetricsを正とする。

既に合意した順序の次は、胚を分けた基準予測の準備。その後に誤差と段階別回収上限の分析を行い、未注釈の子への接続損失を比較する。今回の採点差を理由に追加GPU学習やsubmissionを実行してはいない。
'''
# Version ordinal is not Kaggle scriptVersionId; link to canonical notebook.
result = result.replace('?scriptVersionId=1', '')
(exp / 'result.md').write_text(result)
(exp / 'README.md').write_text('''# exp003_official_metric_audit

固定した人工graphと既存予測に、完全な公式評価と公開Notebookの評価関数を適用し、点の対応・接続・分裂・集計の差を確認する実験。

Kaggle CPU実行と結果回収を終えた。実予測では一致し、人工例で差を再現した。解釈・制約・推奨する判断は[result](result.md)、数値と実行状態は[metrics](metrics.json)を参照する。

- [要件と承認](requirements.md)
- [設定・系譜](config.yaml)
- [実行経緯](SESSION_NOTES.md)
- [監査Notebook](exp003_official_metric_audit_audit.ipynb)

次は合意済みの胚を分けた基準予測を準備する。
''')
notes = exp / 'SESSION_NOTES.md'
text = notes.read_text().replace('CPUで監査中。', 'CPUで監査を実行し結果回収済み。')
text = text.replace('Kaggleの監査ログを確認し、結果回収と記録を行う。', '監査の実行と記録は終了。実験の完了判断はユーザーへ結果を提示する。')
text = text.replace('Kaggle実行値と実験結果の解釈を確認する。', '結果と解釈はresultに記録済み。実験の完了判断はユーザー未回答。上位仮説の後続検証は別作業として残る。')
text += f'\n- {now}: version 1のAUDIT_SUMMARYとNotebook保存ログで実行終了を確認。`make kaggle-output KERNEL=kentookumura/exp003-official-metric-audit-audit/1 OUT=experiments/exp003_official_metric_audit/artifacts/kaggle_v1`で監査成果物を取得。全9人工例・4動画、失敗0。\n'
text += '- 取得後に `inspect_audit_outputs.py kaggle_v1` を実行し、manifest内6ファイルSHA、入力予測SHA、全対象件数を照合。実予測の対応・接続・分裂・得点が一致し、人工例の3スコアと重複辺の件数差を確認。CV/LBはnullを維持した。\n'
text += '- Kaggle内のmetricsと実行IDを突き合わせ、record-expで実行結果を記録。完了・採用・不採用は確定せずdebug_completedとして結果提示する。\n'
notes.write_text(text)
req = exp / 'requirements.md'
text = req.read_text().replace('## 判断履歴\n', '## 判断履歴\n\n- 2026-09-11: 具体的な非公開Kaggle宛先とpayloadの確認後、目的説明を受けたユーザーが「実行してください」と明示。CPU実行を承認。\n', 1)
req.write_text(text)
direction = Path('backlog/KAGGLE_DIRECTION.md')
text = direction.read_text()
marker = '### 現行の比較基準\n'
text = text.replace(marker, marker + '\n公式評価の前提確認は[exp003の結果](../experiments/exp003_official_metric_audit/result.md)を参照する。人工例で公開実装との差を再現し、固定した実予測では一致した。実行証拠を回収済みで、完了判断は未記録。合意済みの次段階は学習・評価の胚を分けた基準予測の準備。\n', 1)
text = text.replace('| 評価器の一致、基準予測の独立性、順位差と候補上限 |', '| 固定公式と公開実装の差はexp003で確認。基準予測の独立性、モデル順位差と候補上限は未検証 |')
direction.write_text(text)
print('Recorded successful CPU audit and conclusions; completion judgement remains pending.')
