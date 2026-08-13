---
name: colab-notebook-runner
description: Create Colab-first notebooks for running Kaggle experiments from this repository when Kaggle GPU quota is limited. Use when Codex needs to generate or adapt a notebook for Google Colab or Colab CLI execution, mount Google Drive, validate repository layout, copy large artifacts from Drive to /content, write persistent logs and status files, or troubleshoot Colab-specific path, RAM, runtime, and session issues.
---

# Colab Notebook Runner

Use this skill when moving an existing `experiments/expXXX_name/` workflow to Colab.

## Core workflow

1. Keep the original experiment ID. A runtime change alone does not create a new experiment.
2. Prefer `<exp>_colab_<kind>.ipynb` instead of overwriting the canonical Kaggle notebook.
3. Store persistent logs and run metadata under `experiments/<exp>/artifacts/colab_runs/` on Google Drive.
4. Copy large input artifacts from DriveFS to `/content` before heavy processing.
5. Treat short CUDA, dependency, and data-preview checks as preflight only. A run is complete only when the experiment's real metrics and completion marker exist.

## Drive layout

Do not assume a repository name or a fixed Drive path. Ask for or derive a project root such as:

```text
/content/drive/MyDrive/Kaggle/<project-name>/
  project.yml
  data/
  experiments/<exp>/
```

Validate every experiment-specific input declared by its `config.yaml` or runner before starting expensive work. Do not copy paths from another experiment.

## Notebook generation

For a simple runner, use the bundled generator:

```bash
uv run python .agents/skills/colab-notebook-runner/scripts/create_colab_notebook.py \
  --experiment expXXX_name \
  --output experiments/expXXX_name/expXXX_name_colab_train.ipynb \
  --drive-root /content/drive/MyDrive/Kaggle/<project-name> \
  --run-command "uv run python experiments/expXXX_name/train.py"
```

Add `--cache-source <repository-relative-path>` for each large file that should be copied to `/content/kaggle_cache/<exp>/` first. The generated notebook mounts Drive, checks `project.yml` and the experiment directory, reports CPU/RAM/GPU state, copies declared caches, and runs the explicit command with Drive-backed logs.

For substantial notebook logic, prefer a Jupytext percent `.py` source with `# %%` and `# %% [markdown]`, then convert it to `.ipynb`. Keep only functions needed for the Colab path. Notebook cells do not define `__file__`; use an explicit project root or `Path.cwd()`.

```bash
UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb experiments/<exp>/<exp>_colab_train.py
UV_CACHE_DIR=/tmp/uv-cache JUPYTER_DATA_DIR=/tmp/jupyter-data uv run --extra notebook jupytext --to ipynb --test experiments/<exp>/<exp>_colab_train.py
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev ruff check experiments/<exp>/<exp>_colab_train.py --select F821
rg -n "__file__|Path\\(__file__\\)" experiments/<exp>/<exp>_colab_train.py
```

## Colab CLI guidance

Use a named session for URL retrieval and short checks:

```bash
colab new -s <session-name> --gpu L4
colab url -s <session-name>
colab status -s <session-name>
```

Mount Drive with `colab drivemount -s <session-name>`. Distinguish the Colab notebook URL from the Google Drive authorization URL. If authorization is requested, show the user the `accounts.google.com` URL as the Drive authorization URL.

`colab exec -f FILE` reads a local file, not a remote `/content` file. For remote scripts, send a small local wrapper through standard input.

## Long-running execution

- Write stdout and stderr to Drive-backed logs.
- Record a start timestamp, command, configuration, process ID when applicable, and completion or failure marker.
- Poll no more often than needed and report progress to the user during long runs.
- Check both the process state and expected artifacts; an exited process without metrics is not success.
- Copy only the minimal outputs needed for local review after completion.

When a Colab run produces official experiment evidence, record it in the same `metrics.json`, `SESSION_NOTES.md`, and `result.md` roles defined by `AGENTS.md`. Colab preflight results are not CV or Kaggle execution evidence.

## Validation

Before handing off a generated or adapted notebook:

```bash
task validate-exp EXP=<exp>
task check-exp EXP=<exp>
task test-exp EXP=<exp>
```

Use the same-named Make targets only when `task` is unavailable.
