import json
from pathlib import Path

path = Path('/home/kento/work/kaggle/Biohub/studies/biohub_source_hidden_eval_20261002/run_a/idea_portfolio.json')
data = json.loads(path.read_text(encoding='utf-8'))
required = {'id', 'title', 'mechanism_family', 'origin_pass', 'roles', 'information_sources', 'hypothesis', 'evidence_ids', 'changed_mechanism', 'input_target_decode', 'deployment_error_simulated', 'preserved_invariants', 'nearest_prior_attempt', 'exact_difference', 'counterevidence', 'cheap_test', 'full_test', 'kill_criterion', 'reopen_criterion', 'coverage_test', 'selectability_test', 'hidden_inference_contract', 'compute_estimate', 'is_parameter_only', 'novelty_level', 'confidence'}
assert data['schema_version'] == '2'
assert len(data['idea_cards']) == 10
assert len(data['portfolio']) == 5
ids = [c['id'] for c in data['idea_cards']]
assert len(set(ids)) == len(ids)
for card in data['idea_cards']:
    assert required <= card.keys(), (card['id'], required - card.keys())
    for field in required:
        value = card[field]
        if isinstance(value, (str, list)):
            assert value, (card['id'], field)
        if isinstance(value, list):
            assert all(isinstance(x, str) and x for x in value)
    assert isinstance(card['is_parameter_only'], bool)
    assert card['confidence'] in {'A', 'B', 'C'}
    assert card['origin_pass'] in {'task_first', 'evidence_inversion', 'cross_pollination'}
    assert card['novelty_level'] in {'incremental', 'role_change', 'representation_change'}
families = {c['mechanism_family'] for c in data['idea_cards']}
assert len(families) >= 4
by_id = {c['id']: c for c in data['idea_cards']}
selected = [by_id[p['idea_id']] for p in data['portfolio']]
assert len({p['idea_id'] for p in data['portfolio']}) == 5
sf = {c['mechanism_family'] for c in selected}
assert {'representation', 'information', 'data_generation'} <= sf
assert sf & {'candidate_generation', 'fusion_uncertainty'}
assert sf & {'validation', 'compute_enabler'}
assert any(c['origin_pass'] == 'task_first' for c in selected)
assert any(c['novelty_level'] == 'representation_change' for c in selected)
assert sum(c['is_parameter_only'] for c in data['idea_cards']) <= 2
assert sum(c['is_parameter_only'] for c in selected) <= 1
assert all(p['slot'] in {'safe', 'exploration', 'orthogonal', 'compute_enabler'} for p in data['portfolio'])
print(json.dumps({'status': 'PASS', 'cards': len(ids), 'selections': len(selected), 'families': sorted(families), 'scientific_validation': False}))
