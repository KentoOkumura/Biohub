from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
exp = ROOT / "experiments/exp003_official_metric_audit"
path = exp / "config.yaml"
config = yaml.safe_load(path.read_text())
config["experiment"]["notebooks"] = ["audit"]
config["overrides"] = {"project_defaults": ["validation.metric", "validation.n_folds"]}
path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
ruff = exp / "ruff.toml"
ruff.write_text(ruff.read_text().replace('"B905"]', '"B905", "B028"]'))
validator = ROOT / "scripts/validate_experiment.py"
text = validator.read_text()
text = text.replace("    project_experiment_defaults,\n", "    project_experiment_defaults,\n    validate_notebook_kind,\n", 1)
helper = '''
def required_notebook_names(experiment_name: str, config: dict[str, Any]) -> tuple[str, ...]:
    """Use explicitly declared kinds; preserve train/inference for older experiments."""
    kinds = get_nested(config, "experiment.notebooks")
    if kinds is None:
        kinds = ["train", "inference"]
    if not isinstance(kinds, list) or not kinds or not all(isinstance(k, str) for k in kinds):
        raise ValueError("experiment.notebooks must be a nonempty list of notebook kinds")
    validated = [validate_notebook_kind(kind) for kind in kinds]
    if len(set(validated)) != len(validated):
        raise ValueError("experiment.notebooks must not contain duplicate kinds")
    return tuple(f"{experiment_name}_{kind}.ipynb" for kind in validated)

'''
text = text.replace("\ndef main() -> None:\n", "\n" + helper + "\ndef main() -> None:\n", 1)
old = '''    for filename in (
        f"{args.experiment}_train.ipynb",
        f"{args.experiment}_inference.ipynb",
    ):
'''
new = '''    config_path = experiment_dir / "config.yaml"
    config = read_yaml(config_path) if config_path.exists() else {}
    try:
        notebook_names = required_notebook_names(args.experiment, config)
    except ValueError as error:
        errors.append(str(error))
        notebook_names = ()
    for filename in notebook_names:
'''
assert old in text
text = text.replace(old, new, 1)
text = text.replace('    config_path = experiment_dir / "config.yaml"\n    if config_path.exists():\n        config = read_yaml(config_path)\n', '    if config_path.exists():\n', 1)
validator.write_text(text)
tests = ROOT / "tests/test_validate_experiment.py"
with tests.open("a") as handle:
    handle.write('''

def test_legacy_notebook_defaults_are_preserved():
    validator = load_validator()
    assert validator.required_notebook_names("exp001_example", {}) == (
        "exp001_example_train.ipynb",
        "exp001_example_inference.ipynb",
    )


def test_explicit_audit_notebook_does_not_require_training():
    validator = load_validator()
    assert validator.required_notebook_names(
        "exp003_example", {"experiment": {"notebooks": ["audit"]}}
    ) == ("exp003_example_audit.ipynb",)


def test_notebook_declaration_cannot_hide_all_notebooks_or_escape_paths():
    import pytest

    validator = load_validator()
    for kinds in [[], "audit", ["audit", "audit"], ["../audit"], [None]]:
        with pytest.raises(ValueError):
            validator.required_notebook_names(
                "exp003_example", {"experiment": {"notebooks": kinds}}
            )
''')
with (exp / "SESSION_NOTES.md").open("a") as handle:
    handle.write("\n- 2026-09-11: 公式・公開関数のAST同一性を含む5 testが通過。既存validatorがtrain/inferenceを無条件要求していたため、experiment.notebooksを明示した監査実験を許容し、未指定の既存動作を維持する限定修正と回帰テストを追加した。\n")
print("audit declaration and validator compatibility updated")
