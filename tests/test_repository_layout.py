from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_records_and_submission_history_use_current_layout() -> None:
    experiments = ROOT / "experiments"

    assert (experiments / "README.md").is_file()
    for path in experiments.iterdir():
        if path.is_dir():
            assert path.name.startswith("exp")
            assert (path / "config.yaml").is_file()
            assert (path / "metrics.json").is_file()
    assert not (ROOT / "submissions").exists()
    assert (ROOT / "SUBMISSIONS.md").is_file()


def test_retired_layout_is_absent() -> None:
    retired_paths = [
        ".steering",
        "KAGGLE_DIRECTION.md",
        "scripts/new_steering.py",
        "templates/steering",
        "docs/analysis",
        "docs/legacy",
    ]

    assert all(not (ROOT / path).exists() for path in retired_paths)


def test_expected_repository_skills_exist() -> None:
    expected = {
        "colab-notebook-runner",
        "kaggle-discussion-archive",
        "kaggle-idea-forge",
        "kaggle-notebook-fetch",
        "kaggle-oof-readout",
        "kaggle-platform",
        "kaggle-review",
        "kaggle-review-exp",
        "kaggle-strategy",
        "kaggle-submit-check",
        "kaggle-submit-monitor",
        "kaggle-survey-papers",
    }
    skills_dir = ROOT / ".agents" / "skills"
    actual = {path.name for path in skills_dir.iterdir() if (path / "SKILL.md").is_file()}

    assert actual == expected


def test_project_is_configured_for_biohub_cell_tracking() -> None:
    project = yaml.safe_load((ROOT / "project.yml").read_text())

    assert project["competition"] == {
        "name": "Biohub - Cell Tracking During Development",
        "platform": "kaggle",
        "slug": "biohub-cell-tracking-during-development",
        "url": ("https://www.kaggle.com/competitions/biohub-cell-tracking-during-development"),
        "is_code_competition": True,
    }
    assert project["data"]["group_column"] == "embryo_id"
    assert project["defaults"]["primary_validation"] == "leave-one-embryo-out"
    assert project["defaults"]["n_folds"] == 2
    assert project["defaults"]["secondary_validation"] == (
        "fixed 8-sample holdout from Clean Approach + Lightweight Local CV"
    )
    assert project["defaults"]["secondary_validation_samples"] == [
        "44b6_0113de3b",
        "44b6_0b24845f",
        "44b6_341df25f",
        "44b6_e57ff5c6",
        "6bba_05b6850b",
        "6bba_05db0fb1",
        "6bba_969618f6",
        "6bba_fc83837d",
    ]
    assert project["submission"]["id_column"] == "id"
    assert project["submission"]["target_columns"] == [
        "node_id",
        "t",
        "z",
        "y",
        "x",
        "source_id",
        "target_id",
    ]
    assert project["runtime"]["kaggle"] == {
        "enable_gpu": False,
        "enable_internet": False,
        "time_limit_hours": 12,
    }


def test_colab_generator_is_generic() -> None:
    scripts_dir = ROOT / ".agents" / "skills" / "colab-notebook-runner" / "scripts"

    assert (scripts_dir / "create_colab_notebook.py").is_file()
    assert not (scripts_dir / "create_colab_train_notebook.py").exists()
