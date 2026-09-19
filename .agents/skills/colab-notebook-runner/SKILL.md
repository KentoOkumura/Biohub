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

### Direct-upload smoke runs

For a small, explicitly authorized, non-secret payload, Drive is not required for a short smoke run. This is useful for validating the Colab runtime and experiment code before arranging persistent full-run inputs.

```bash
colab new -s <session-name> --gpu T4
colab upload -s <session-name> <local-input> <input-name>
colab upload -s <session-name> <local-code> <code-name>
colab ls -s <session-name> .
colab exec -s <session-name> -f <local-wrapper.py> --timeout <seconds>
colab stop -s <session-name>
```

- `colab upload` addresses the Jupyter contents API, not an assumed `/content` filesystem path. Upload to a simple CLI-root name, inspect it with `colab ls`, and have the executed wrapper resolve or move the uploaded file into `/content`. In the observed CLI runtime, a root upload named `payload.zip` appeared as `/payload.zip`; a nested `content/...` upload returned 404.
- Obtain explicit authorization before transferring experiment data or model weights. Authorization to run an experiment does not by itself authorize transfer to Colab.
- Treat materially different external payloads separately when obtaining authorization: support/model archives, signed download manifests, resume seeds or generated outputs, and large experiment caches are distinct transfers.
- Never transfer Kaggle, Google, or other account credentials with `colab upload`. Use Colab Secrets or the provider's interactive authorization flow.
- Treat signed URL manifests as temporary sensitive transport material. Create them immediately before use, avoid logging their contents, and delete both local and remote copies after the download completes.
- Do not infer liveness from `colab status` alone; it can be stale. Prove session access with an actual `exec`, `ls`, upload, or download operation.
- Do not treat the CLI process exit code alone as notebook success. `colab exec` can return exit code 0 while the captured IPython output contains a traceback. Require an explicit success marker and validate its content.
- Verify the completion marker, expected artifacts, archive integrity, byte count, and SHA locally before reporting success. Confirm that the named session is stopped afterward.
- Treat direct upload as a smoke/debug route. For large inputs, long execution, or resumable batches, use Drive-backed inputs, logs, receipts, and outputs so a lost Colab VM does not erase the run.

## Long-running execution

- Write stdout and stderr to Drive-backed logs.
- Record a start timestamp, command, configuration, process ID when applicable, and completion or failure marker.
- Poll no more often than needed and report progress to the user during long runs.
- Check both the process state and expected artifacts; an exited process without metrics is not success.
- Copy only the minimal outputs needed for local review after completion.
- Do not assume that outputs remain downloadable after `colab exec` returns. A runtime can expire immediately after successful computation. For material non-Drive outputs, emit bounded batches while the process is live, verify each batch locally by byte count and SHA, and keep resumable receipts; use Drive persistence when available.
- When using time-limited download manifests, prepare them before allocating the GPU session or keep the session active, then begin continuous work immediately so URL expiry and idle teardown do not consume the execution window.

When a Colab run produces official experiment evidence, record it in the same `metrics.json`, `SESSION_NOTES.md`, and `result.md` roles defined by `AGENTS.md`. Colab preflight results are not CV or Kaggle execution evidence.

## Validation

When creating or changing formulas in documents or Notebook Markdown cells, follow the [math conventions and verification steps in AGENTS.md](../../../AGENTS.md#markdown-と-notebook-の数式) for the changed files.

Before handing off a generated or adapted notebook:

```bash
task validate-exp EXP=<exp>
task check-exp EXP=<exp>
task test-exp EXP=<exp>
```

Use the same-named Make targets only when `task` is unavailable.
