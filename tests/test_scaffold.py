from pathlib import Path

import yaml

from scripts import run_tests
from scripts.config_utils import (
    ROOT as CONFIG_ROOT,
)
from scripts.config_utils import (
    configured_project_path,
    get_nested,
    load_project_config,
    project_experiment_defaults,
    project_path,
)

ROOT = Path(__file__).resolve().parents[1]


def test_template_files_exist() -> None:
    template_dir = ROOT / "templates" / "experiment"
    required = [
        "README.md",
        "requirements.md",
        "config.yaml",
        "settings.py",
        "{{EXPERIMENT_NAME}}_train.ipynb",
        "{{EXPERIMENT_NAME}}_inference.ipynb",
        "SESSION_NOTES.md",
        "result.md",
        "metrics.json",
    ]

    for filename in required:
        assert (template_dir / filename).exists()
    assert (template_dir / "artifacts" / ".gitkeep").is_file()
    assert not (template_dir / "features").exists()
    assert not (template_dir / "variants").exists()


def test_repository_control_files_exist() -> None:
    required = [
        "AGENTS.md",
        "backlog/KAGGLE_DIRECTION.md",
        "Taskfile.yml",
        "Makefile",
        "uv.lock",
        "experiment_summary.md",
        "project.yml",
        "SUBMISSIONS.md",
        "docs/agent-playbooks.md",
        "app/streamlit_app.py",
        "app/oof_analysis_app.py",
        "docs/official/evaluation.md",
        "scripts/validate_project.py",
        "scripts/validate_experiment.py",
        "scripts/validate_submission.py",
        "scripts/execute_experiment_notebook.py",
        "scripts/prepare_kaggle_notebooks.py",
        "scripts/validate_kaggle_metadata.py",
        "scripts/new_survey_report.py",
        "scripts/update_survey_index.py",
        "scripts/kaggle_download.py",
        "scripts/project_value.py",
        "scripts/archive_kaggle_discussions.py",
        "scripts/record_submission.py",
        "scripts/record_experiment.py",
        "scripts/compare_experiments.py",
        "scripts/update_experiment_summary.py",
        "scripts/check_markdown_links.py",
        "scripts/check_strategy_docs.py",
        "templates/experiment/requirements.md",
        "templates/survey/report.md",
        "docs/surveys/README.md",
        "docs/01_competition.md",
        "docs/02_metric.md",
        "docs/03_validation.md",
        "docs/04_data.md",
        "docs/05_workflow.md",
        "docs/glossary.md",
    ]

    for filename in required:
        assert (ROOT / filename).exists()


def test_docs_readme_indexes_managed_subdirectories() -> None:
    docs_readme = (ROOT / "docs" / "README.md").read_text()
    managed_directories = sorted(
        path.name
        for path in (ROOT / "docs").iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__"
    )

    assert managed_directories
    assert all(f"`{name}/`" in docs_readme for name in managed_directories)


def test_docs_readme_indexes_managed_root_documents() -> None:
    docs_readme = (ROOT / "docs" / "README.md").read_text()
    managed_documents = {
        "01_competition.md",
        "02_metric.md",
        "03_validation.md",
        "04_data.md",
        "05_workflow.md",
        "06_reproducibility.md",
        "agent-playbooks.md",
        "glossary.md",
    }

    assert all(f"`{name}`" in docs_readme for name in managed_documents)


def test_root_readme_delegates_docs_directory_index() -> None:
    root_readme = (ROOT / "README.md").read_text()

    assert "[docs/README.md](docs/README.md)" in root_readme


def test_no_repository_local_codex_skills() -> None:
    skills_dir = ROOT / "skills"
    if not skills_dir.exists():
        return
    assert not any(skills_dir.glob("*/SKILL.md"))


def test_project_yml_supplies_experiment_defaults() -> None:
    project = load_project_config()
    defaults = project_experiment_defaults(project)

    assert get_nested(defaults, "validation.metric") == get_nested(project, "defaults.metric")
    assert get_nested(defaults, "validation.seed") == get_nested(project, "defaults.seed")
    assert get_nested(defaults, "validation.secondary_strategy") == get_nested(
        project,
        "defaults.secondary_validation",
    )
    assert get_nested(defaults, "validation.secondary_sample_ids") == get_nested(
        project,
        "defaults.secondary_validation_samples",
    )
    assert get_nested(defaults, "reproducibility.seed") == get_nested(
        project,
        "defaults.seed",
    )
    assert get_nested(defaults, "data.id_column") == get_nested(project, "submission.id_column")
    assert get_nested(defaults, "data.sample_submission") == get_nested(
        project,
        "submission.sample_file",
    )
    assert (
        get_nested(defaults, "data.submission_target_column")
        == get_nested(
            project,
            "submission.target_columns",
        )[0]
    )


def test_project_yml_supplies_repository_paths() -> None:
    project = load_project_config()

    assert project_path(project, "paths.experiments_dir") == CONFIG_ROOT / "experiments"
    assert project_path(project, "paths.submissions_file") == CONFIG_ROOT / "SUBMISSIONS.md"


def test_configured_project_path_uses_the_selected_repository_root(tmp_path: Path) -> None:
    (tmp_path / "project.yml").write_text(
        "paths:\n  experiments_dir: runs\n  docs_dir: knowledge\n"
    )

    assert configured_project_path("paths.experiments_dir", "experiments", root=tmp_path) == (
        tmp_path / "runs"
    )
    assert configured_project_path("paths.docs_dir", "docs", root=tmp_path) == (
        tmp_path / "knowledge"
    )


def test_test_groups_use_configured_experiments_directory(tmp_path: Path) -> None:
    (tmp_path / "project.yml").write_text("paths:\n  experiments_dir: runs\n")
    common = tmp_path / "tests"
    experiment = tmp_path / "runs/exp001_example/tests"
    common.mkdir()
    experiment.mkdir(parents=True)
    (tmp_path / "experiments/exp002_unused/tests").mkdir(parents=True)
    assert run_tests.test_groups(tmp_path) == [common, experiment]


def test_experiment_template_does_not_duplicate_project_defaults() -> None:
    config_path = ROOT / "templates" / "experiment" / "config.yaml"
    template_config = yaml.safe_load(config_path.read_text())

    assert template_config["validation"] == {}
    assert template_config["data"] == {}
    assert "seed" not in template_config["reproducibility"]

    settings = (ROOT / "templates" / "experiment" / "settings.py").read_text()
    assert '"reproducibility": {"seed": seed}' in settings


def test_experiment_template_declares_route() -> None:
    config_path = ROOT / "templates" / "experiment" / "config.yaml"
    template_config = yaml.safe_load(config_path.read_text())

    assert template_config["experiment"]["route"] == "TODO"


def test_experiment_template_keeps_generated_outputs_under_artifacts() -> None:
    settings = (ROOT / "templates" / "experiment" / "settings.py").read_text()

    assert 'return self.artifacts_dir / "features"' in settings
    assert 'return self.experiment_dir / "features"' not in settings
    assert 'return self.experiment_dir / "variants"' not in settings
