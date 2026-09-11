import hashlib
import json
from datetime import datetime
from pathlib import Path

exp = Path('experiments/exp003_official_metric_audit')
package = exp / 'kaggle/audit'
now = datetime.now().astimezone().isoformat()
metadata = json.loads((package / 'kernel-metadata.json').read_text())
assert metadata['is_private'] and not metadata['enable_gpu'] and not metadata['enable_tpu']
assert not metadata['enable_internet']
manifest = {
    'observed_at': now, 'metadata': metadata,
    'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in sorted(package.iterdir()) if p.is_file()},
    'validation': {'experiment_tests_passed': 5, 'validator_tests_passed': 21,
                   'ruff': 'passed', 'strict_experiment_validation': 'passed',
                   'jupytext_roundtrip': 'passed'},
}
(exp / 'artifacts').mkdir(exist_ok=True)
(exp / 'artifacts/local_preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')
with (exp / 'SESSION_NOTES.md').open('a') as stream:
    stream.write(f'\n- {now}: ローカル検証は対象5 tests、共通validator 21 tests、Ruff、strict validation、Jupytext roundtripすべて通過。\n')
    stream.write('- Notebookは7節・1753行。親の正規self-contained trainは683行で、compact別版はない。監査に必要な公式関数、公開関数、入力検査、実行、集計をNotebook内に展開した。\n')
    stream.write('- prepareの初回要求は自動承認レビューが外部送信と解釈して拒否。prepare実装に通信がなくローカル生成のみであることを確認し、その証拠を示した再実行は成功した。\n')
    stream.write(f'- {now}: push前metadataを確認。宛先 `{metadata["id"]}`、private、CPU、TPUなし、internetなし。CPUなので直前のGPU quota再照会は不要。packageは監査Notebook、実験設定・固定fixture・保存した公式/公開評価コードからなり、送信内容のSHAを `artifacts/local_preparation.json` に記録。学習0、model 0、fold 0。\n')
print(json.dumps({'observed_at': now, 'id': metadata['id'], 'files': len(manifest['files'])}))
