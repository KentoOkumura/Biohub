"""Record primary-source corrections without changing the original rubric."""

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
original = json.loads((BASE / "withheld_rubric.json").read_text())
effective = copy.deepcopy(original)
effective["version"] = "2"
corrections = {
    "R07": (
        "Use a provisional graph to revise division costs and rerun ILP (rank 7), or iteratively relinearize metric-surrogate coefficients from predicted/expected graph counts and rerun constrained selection (rank 3). One-time use of graph features alone is partial.",
        "Rank 3's metric-surrogate iteration is not necessarily a topology-feature classifier."
    ),
    "R08": (
        "Run high-confidence graph selection first and include weaker detections in a second ILP; cancel divisions absent from the first pass while keeping extra independent cells, then re-solve links. Hard fixation of all first-pass paths is not required.",
        "Rank 5 does not explicitly freeze all reliable first-pass paths."
    ),
    "R11": (
        "Train final-graph edit candidates using signed utility based on changes in officially evaluated TP/FP and node-count cost, or an exact official-score difference. Exact whole-score delta is not required; unknown-label handling and feature recomputation are separate safety/implementation checks.",
        "Rank 17 uses a local surrogate utility rather than exact official-score delta; the other two details were undocumented as source mechanisms."
    ),
    "R12": (
        "After provisional division adoption, repair/extend daughter tracks and then perform a final division re-evaluation on repaired topology while considering competing ordinary edges. Initial provisional decisions before repair are allowed.",
        "Rank 18 makes provisional decisions before daughter repair; only the final recheck must follow repair."
    ),
    "R13": (
        "Feed final predicted track nodes after ILP and postprocessing back as weighted pseudo-labels for a new detector-training pass, distinguish inferred labels and confirmed annotation. A special reliability filter is not required.",
        "Rank 16 documents Silver final nodes with a separately weighted loss, not a special reliability filter."
    ),
    "R14": (
        "Apply final-graph coordinate correction while holding identities/links fixed and/or restore valid divisions after downstream processing deletes them; distinguish final geometry/repair from pre-tracker changes. Graph-only or frozen-feature variants fit the policy; rank 12's own learned raw-image recentring requires a policy change.",
        "Policy feasibility depends on the correction's input/model; rank 12's exact CNN variant is outside the frozen-image policy."
    ),
    "R16": (
        "Learn explicit birth/no-parent and mother-division outputs and use their probabilities for graph start/division costs (rank 5). Partial-annotation-aware division supervision is independently documented in rank 11, which has no documented learned birth head. State how unknown birth/division targets are treated; dense joint image learning requires a policy change, but separate heads on frozen features can be within policy.",
        "Rank 5 and rank 11 do not each implement the whole conjunction; downstream heads can be studied on fixed image features."
    ),
}
amendments = []
for criterion in effective["criteria"]:
    if criterion["id"] in corrections:
        text, reason = corrections[criterion["id"]]
        amendments.append({"criterion": criterion["id"], "old": criterion["mechanism"], "new": text, "reason": reason})
        criterion["mechanism"] = text
        if criterion["id"] == "R16":
            criterion["policy"] = "within_policy"
effective["amendment_basis"] = "Independent primary-source audit before any generator content was read by parent or judge. Original remains immutable."
destination = BASE / "effective_rubric.json"
destination.write_text(json.dumps(effective, ensure_ascii=False, indent=2) + "\n")
sha = hashlib.sha256(destination.read_bytes()).hexdigest()
now = datetime.now(timezone.utc).isoformat()
(BASE / "rubric_errata.json").write_text(json.dumps({"recorded_at_utc": now, "effective_rubric_sha256": sha, "amendments": amendments}, ensure_ascii=False, indent=2) + "\n")
protocol_path = BASE / "protocol.json"
protocol = json.loads(protocol_path.read_text())
protocol["effective_rubric_sha256"] = sha
protocol["source_correction_recorded_at_utc"] = now
protocol["source_correction_before_generator_content_read"] = True
protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"effective_rubric_sha256": sha, "amendment_count": len(amendments), "recorded_at_utc": now}))
