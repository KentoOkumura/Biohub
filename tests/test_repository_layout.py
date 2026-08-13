from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_template_has_no_sample_experiment_or_submission_directory() -> None:
    experiments = ROOT / "experiments"

    assert {path.name for path in experiments.iterdir()} == {"README.md"}
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
        "url": (
            "https://www.kaggle.com/competitions/"
            "biohub-cell-tracking-during-development"
        ),
        "is_code_competition": True,
    }
    assert project["data"]["group_column"] == "embryo_id"
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
