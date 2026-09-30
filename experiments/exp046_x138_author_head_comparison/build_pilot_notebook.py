"""Build the two-video pilot from the paired head-comparison notebook source."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "exp046_x138_author_head_comparison"
source = (ROOT / f"{NAME}_inference.py").read_text()
marker = "if _elapsed_pilot + _estimated_remaining_seconds >= _notebook_limit_seconds:\n"
if source.count(marker) != 1:
    raise RuntimeError("runtime-gate boundary changed")
prefix = source.split(marker, 1)[0]
prefix = prefix.replace(
    "# # exp046: x138 author and self-trained coordinate head comparison",
    "# # exp046: two-video head comparison pilot",
    1,
)
prefix = prefix.replace(
    "# 7. Fixed-ID known-edge and division readout\n"
    "# 8. Official evaluator on both sets of complete graphs\n"
    "# 9. Coverage, content hashes and execution receipt\n",
    "# 7. Pilot receipt and resource decision\n",
    1,
)
suffix = """# %% [markdown]
# ## 7. Pilot receipt and resource decision
#
# This notebook intentionally stops after one video from each embryo in each
# arm. The full 20-video head comparison is a separate notebook version.

# %%
_PILOT_FILES = sorted(
    path for path in AUDIT_ROOT.rglob("*")
    if path.is_file() and path.name != "exp046_pilot_receipt.json"
)
_PILOT_RECEIPT = {
    "experiment": "exp046_x138_author_head_comparison",
    "scope": "pilot_only",
    "selected_videos": test_stems,
    "pilot_videos": _pilot,
    "arms": list(AUDIT_ARMS),
    "head_bundles": _HEAD_BUNDLES,
    "self_head_manifest_sha256": _expected_manifest_sha,
    "selection_manifest_sha256": _audit_sha256(WORKING_DIR / "exp046_selection.json"),
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
(AUDIT_ROOT / "exp046_pilot_receipt.json").write_text(
    json.dumps(_audit_jsonable(_PILOT_RECEIPT), indent=2, sort_keys=True) + "\\n"
)
print("Pilot complete:", AUDIT_ROOT / "exp046_pilot_receipt.json", flush=True)
"""
(ROOT / f"{NAME}_pilot.py").write_text(prefix + suffix)
