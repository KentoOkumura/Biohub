"""Compare source syntax, attachments and public output receipts without executing notebooks."""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT/'docs/notebooks/biohub-cell-tracking-during-development/20260925'

def notebook(ref):
    folder = SOURCE/ref.replace('/', '__')
    meta = json.loads((folder/'kernel-metadata.json').read_text())
    return json.loads((folder/meta['code_file']).read_text()), meta

def syntax(nb):
    cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if source.strip():
                cells.append(ast.dump(ast.parse(source), include_attributes=False))
    return cells

x, xm = notebook('anvithpothula/biohub-x138')
k, km = notebook('kunaldesale2408/biohub-cell-tracking')
x_ast, k_ast = syntax(x), syntax(k)
meta_keys = ('enable_gpu','enable_internet','dataset_sources','competition_sources','kernel_sources','model_sources')
public = json.loads((OUT/'public_scores.json').read_text())
def attachments(ref):
    row = next(r for r in public if r['ref'] == ref)
    return sorted({(a['sourceId'], a.get('datasetId'), a['mountSlug']) for a in row['execution_and_attachment_evidence'] if a.get('sourceType') == 'DATA_SOURCE_TYPE_DATASET_VERSION'})
xr = ROOT/'studies/biohub_public_notebooks_20260923/x138_output'
kr = OUT/'kunal_output'
x_receipt = json.loads((xr/'bidirectional_production_runtime_integrity.json').read_text())
k_receipt = json.loads((kr/'bidirectional_production_runtime_integrity.json').read_text())
xstats = list(csv.DictReader((xr/'run_stats.csv').open()))
kstats = list(csv.DictReader((kr/'run_stats.csv').open()))
comparison = {
    'scope': 'x138 V1 vs Kunal Biohub Cell Tracking V10',
    'nonempty_code_cells': {'x138':len(x_ast),'kunal':len(k_ast)},
    'code_cell_ast_equal':x_ast == k_ast,
    'code_cell_ast_sha256': hashlib.sha256(json.dumps(x_ast).encode()).hexdigest(),
    'execution_metadata_equal': {key: xm.get(key) == km.get(key) for key in meta_keys},
    'dataset_version_attachments_equal': attachments('anvithpothula/biohub-x138') == attachments('kunaldesale2408/biohub-cell-tracking'),
    'primary_secondary_deepcenter_sha_equal': x_receipt['checkpoint_sha256'] == k_receipt['checkpoint_sha256'],
    'checkpoint_comparison_scope': ['primary','secondary','deepcenter'],
    'v1284_comparison_scope': 'same dataset ID 12103746 and version source ID 19822532; neither old receipt records head SHA',
    'visible_test_run_stats_equal': xstats == kstats,
    'visible_test_run_stats':kstats,
}
(OUT/'source_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
print(json.dumps({key:value for key,value in comparison.items() if key != 'visible_test_run_stats'}, indent=2))
