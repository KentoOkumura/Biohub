"""Build the two-video pilot from the fixed full-audit notebook source."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "exp045_x138_coordinate_effect_audit"
source = (ROOT / f"{NAME}_inference.py").read_text()
marker = "if _elapsed_pilot + _estimated_remaining_seconds >= _notebook_limit_seconds:\n"
if source.count(marker) != 1:
    raise RuntimeError("runtime-gate boundary changed")
prefix = source.split(marker, 1)[0]
prefix = prefix.replace(
    "# # exp045: fixed-head coordinate effect audit",
    "# # exp045: two-video coordinate effect pilot",
    1,
)
suffix = """# %% [markdown]
# ## 7. Pilot receipt and resource decision
#
# This notebook intentionally stops after one video from each embryo in each
# arm. The full 20-video graph audit is a separate notebook version.

# %%
_PILOT_FILES = sorted(
    path for path in AUDIT_ROOT.rglob("*")
    if path.is_file() and path.name != "exp045_pilot_receipt.json"
)
_PILOT_RECEIPT = {
    "experiment": "exp045_x138_coordinate_effect_audit",
    "scope": "pilot_only",
    "selected_videos": test_stems,
    "pilot_videos": _pilot,
    "arms": list(AUDIT_ARMS),
    "head_sha256": _SELF_HEAD_SHA,
    "head_manifest_sha256": _expected_manifest_sha,
    "public_evaluator_sha256": AUDIT_EXPECTED_EVALUATOR_SHA,
    "pilot_runtime_seconds": _elapsed_pilot,
    "estimated_remaining_seconds_with_25pct_margin": _estimated_remaining_seconds,
    "estimated_full_seconds_with_25pct_margin": _elapsed_pilot + _estimated_remaining_seconds,
    "pilot_identity": _pilot_identity,
    "pilot_official": _pilot_official,
    "files": [
        {"path": path.relative_to(AUDIT_ROOT).as_posix(),
         "bytes": path.stat().st_size, "sha256": _audit_sha256(path)}
        for path in _PILOT_FILES
    ],
    "ground_truth_accessed_during_prediction": False,
    "competition_submission_created": False,
}
(AUDIT_ROOT / "exp045_pilot_receipt.json").write_text(
    json.dumps(_audit_jsonable(_PILOT_RECEIPT), indent=2, sort_keys=True) + "\\n"
)
print("Pilot complete:", AUDIT_ROOT / "exp045_pilot_receipt.json", flush=True)
"""
(ROOT / f"{NAME}_pilot.py").write_text(prefix + suffix)
