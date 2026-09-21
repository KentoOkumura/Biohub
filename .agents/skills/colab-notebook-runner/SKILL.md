---
name: colab-notebook-runner
description: Run Kaggle experiments from this repository on Google Colab through notebooks or Colab CLI. Use for Drive-backed runs, direct Kaggle-to-Colab downloads through signed URLs, small ZIP transfers, resumable CLI sessions, artifact verification, and Colab runtime troubleshooting.
---

# Colab Notebook Runner

Use this skill when moving an existing `experiments/expXXX_name/` workflow to Colab. Follow the experiment contract and the user's chosen execution route.

## Core workflow

1. Keep the original experiment ID. A runtime change alone does not create a new experiment.
2. When creating a notebook, prefer `<exp>_colab_<kind>.ipynb` instead of overwriting the canonical Kaggle notebook.
3. Choose persistence before starting: use Drive-backed files for a Drive run, or collect verified receipts and resumable checkpoints during a CLI run without Drive. Do not require a Drive mount or manual notebook execution for the CLI route.
4. Put large inputs in `/content` for computation. Obtain them directly from their remote source when the user wants to avoid a local download; transfer only small code, model, and manifest artifacts through the local machine.
5. Treat short CUDA, dependency, and data-preview checks as preflight only. A run is complete only when the experiment's real metrics and completion marker exist and have been verified outside the Colab VM.

## Drive-backed layout

Do not assume a repository name or a fixed Drive path. Ask for or derive a project root such as:

```text
/content/drive/MyDrive/Kaggle/<project-name>/
  project.yml
  data/
  experiments/<exp>/
```

Validate every experiment-specific input declared by its `config.yaml` or runner before starting expensive work. Do not copy paths from another experiment.

## Notebook generation

For a simple Drive-backed runner, use the bundled generator:

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

Use a named session for preflight and full CLI runs. Select the GPU type from the experiment contract:

```bash
colab new -s <session-name> --gpu <gpu-type>
colab url -s <session-name>
colab status -s <session-name>
```

For a Drive-backed run, mount it with `colab drivemount -s <session-name>`. Distinguish the Colab notebook URL from the Google Drive authorization URL. If authorization is requested, show the user the `accounts.google.com` URL as the Drive authorization URL.

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

### Transfer and verification rules

- Check the current conversation for authorization before transferring experiment data, model weights, or generated outputs. Obtain authorization for a transfer that is not yet covered; do not ask again for a transfer the user has already authorized.
- Distinguish materially different payloads when checking authorization: support/model archives, signed download manifests, resume seeds or generated outputs, and large experiment caches.
- Never transfer Kaggle, Google, or other account credentials with `colab upload`. Keep Kaggle credentials local when generating signed URLs; use Colab Secrets or the provider's interactive authorization flow only for routes that require them.
- Treat signed URL manifests as temporary sensitive transport material. Create them immediately before use, avoid logging their contents, and delete both local and remote copies after the download completes.
- Do not infer liveness from `colab status` alone; it can be stale. Prove session access with an actual `exec`, `ls`, upload, or download operation.
- Do not treat the CLI process exit code alone as notebook success. `colab exec` can return exit code 0 while the captured IPython output contains a traceback. Require an explicit success marker and validate its content.
- Verify the completion marker, expected artifacts, archive integrity, byte count, and SHA locally before reporting success. Confirm that the named session is stopped afterward.

### Full CLI runs without Drive

1. Package the small code, configuration, required public support, and saved control weights into a ZIP with a member manifest. Verify file names, byte counts, and SHA-256 before upload and after extraction. Split the ZIP into bounded parts if the contents API cannot handle it as one upload.
2. For large Kaggle kernel outputs, create short-lived signed HTTPS URL manifests with local Kaggle authentication. Upload only the manifests; download the files directly from Kaggle into Colab `/content`. Never place Kaggle credentials or large cache files in the uploaded ZIP or local staging area. If required competition files are available only inside a Kaggle mount, use a private CPU export kernel to produce the smallest needed archive, then fetch that output directly in Colab.
3. Verify each downloaded file against the expected size and SHA-256 when available, then validate the cache schema, sample identities, and experiment-specific manifests before GPU work. Keep the signed URLs out of logs and remove local and remote URL manifests after the download.
4. Make long work resumable at a declared boundary such as an epoch. Save the model, optimizer, random-number-generator state, configuration identity, and a small receipt. Collect receipts and the newest resume state while the session is live. Verify the new state before removing an older local copy. On a new session, restore only a verified state from the same run and configuration.
5. For multi-stage runs, verify training completion and artifact identities before starting a diagnostic stage. Produce a bounded completion archive containing metrics, manifests, selected weights, and completion marker. Exclude large prediction arrays unless explicitly needed. Check the archive size and expected member list from a remote receipt before local download, then verify part sizes, SHA-256, extracted members, and completion marker. Honor any user limit on local download size.

Use an experiment-specific runner for these steps; the bundled notebook generator targets the Drive route. `experiments/exp032_three_frame_ten_epoch_training/colab_cli_run.py` and `colab_cli_worker.py` show one implemented CLI route, not a required layout for other experiments.

## Long-running execution

- Write stdout and stderr to Drive-backed logs, or preserve bounded logs and progress receipts outside the VM during a CLI run.
- Record a start timestamp, command, configuration, process ID when applicable, and completion or failure marker.
- Poll no more often than needed and report progress to the user during long runs.
- Check both the process state and expected artifacts; an exited process without metrics is not success.
- Copy only the minimal outputs needed for local review after completion.
- Do not assume that outputs remain downloadable after `colab exec` returns. A runtime can expire immediately after successful computation. For material non-Drive outputs, emit bounded batches while the process is live, verify each batch locally by byte count and SHA, and keep resumable receipts; use Drive persistence when available.
- If file operations begin returning 404 after earlier successful access while computation continues, check the current session assignment and refresh its connection or proxy token through a supported CLI mechanism. Confirm liveness with an actual operation and resume collection; do not restart the computation merely because an old connection expired. Do not print or persist access tokens in logs.
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
