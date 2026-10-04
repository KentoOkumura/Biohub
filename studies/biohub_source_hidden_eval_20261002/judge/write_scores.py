"""Persist the judge's fixed ratings and verify each cited quote and frozen input."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
rubric = json.loads((ROOT / "effective_rubric.json").read_text())
freeze = json.loads((ROOT / "final_output_freeze.json").read_text())
for relative, record in freeze["files"].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == record["sha256"]
assert hashlib.sha256((ROOT / "effective_rubric.json").read_bytes()).hexdigest() == "c7080374b467d2813189756d5c1606ec90df4d40017025a6ae5ed2de1d96eca8"

documents = {}
for run in ("run_a", "run_b"):
    initial = json.loads((ROOT / run / "task_first.json").read_text())
    final = json.loads((ROOT / run / "idea_portfolio.json").read_text())
    documents[run] = {
        "initial": {idea["id"]: idea for idea in initial["initial_ideas"]},
        "final": {idea["id"]: idea for idea in final["idea_cards"]},
        "selected": [entry["idea_id"] for entry in final["portfolio"]],
    }

STAGES = ("task_first", "all_cards", "top_five")


def assessment(run, ratings, citations=(), missing="", reuse="No nonzero match.", scope=""):
    result = dict(zip(STAGES, ratings))
    result["matched_idea_ids_by_stage"] = {stage: [] for stage in STAGES}
    result["evidence"] = []
    for phase, idea_id, field, quote in citations:
        idea = documents[run][phase][idea_id]
        field_value = idea[field]
        source_text = "\n".join(field_value) if isinstance(field_value, list) else field_value
        assert quote in source_text, (run, idea_id, field, quote)
        stages = ["task_first"] if phase == "initial" else ["all_cards"]
        if phase == "final" and idea_id in documents[run]["selected"]:
            stages.append("top_five")
        stages = [stage for stage in stages if result[stage] > 0]
        for stage in stages:
            result["matched_idea_ids_by_stage"][stage].append(idea_id)
        result["evidence"].append({
            "stages": stages,
            "idea_id": idea_id,
            "field": field,
            "quote": quote,
            "path": f"{run}/{'task_first' if phase == 'initial' else 'idea_portfolio'}.json",
        })
    for stage in STAGES:
        result["matched_idea_ids_by_stage"][stage] = sorted(set(result["matched_idea_ids_by_stage"][stage]))
        assert (result[stage] == 0) or result["matched_idea_ids_by_stage"][stage], (run, stage)
    result["matched_idea_ids"] = sorted({idea_id for ids in result["matched_idea_ids_by_stage"].values() for idea_id in ids})
    result["missing_defining_steps"] = ([{"stages": [stage for stage in STAGES if result[stage] < 2], "text": missing}] if missing else [])
    result["historical_reuse"] = reuse
    result["scope_note"] = scope
    return result


rows = {}


def add(criterion_id, a, b):
    rows[criterion_id] = {"run_a": a, "run_b": b}


add("R01",
    assessment("run_a", [1, 1, 0], [
        ("initial", "TF06", "target", "voxel占有確率、時間方向の対応、分裂に伴う領域の分岐"),
        ("final", "I10", "changed_mechanism", "新規画像モデルがvoxel占有確率と時間対応を共同出力する"),
    ], "No query-specific relative-cell maps, separate division-state target or count-then-child-set selection; I10 is outside top five.", "Task-first image proposal retained as I10; no historical query-occupancy mechanism was in the packet.", "TF06/I10 explicitly defer new image learning pending policy change."),
    assessment("run_b", [1, 1, 1], [
        ("initial", "TF05", "output", "Dense or locally queried transition distributions"),
        ("final", "I01", "hypothesis", "Separating daughter-set ranking from cardinality"),
    ], "Initial transition distributions lack explicit query-relative occupancy/state heads; final cardinality scoring lacks the defining learned query-image maps.", "Task-first TF05 predates detailed evidence. I01's cardinality/set reformulation responds explicitly to E02's existing set model and E08 selection failures.", "TF05 is conditional new-image learning; I01 is a fixed-feature downstream partial component."))

add("R02",
    assessment("run_a", [0, 0, 0], missing="No extraction/insertion of labeled real-cell temporal patches for faint-cell detector training; I08 is downstream observation augmentation."),
    assessment("run_b", [0, 0, 0], missing="No labeled real-cell patch synthesis and detector-training pass; graph/score corruption is a different operation."))

add("R03",
    assessment("run_a", [1, 1, 0], [
        ("initial", "TF02", "objective", "複数scaleのblob領域とframe間対応を作り、局所画像の一致・領域重複を計測する"),
        ("final", "I07", "changed_mechanism", "現行point predictorとcandidate setを候補生成に使わず"),
    ], "Generates new image-region proposals rather than reusing unused low-confidence detections; no explicit retention of a persistent unattachable independent track or free-endpoint-only attachment.", "Task-first independent-image proposal retained; E06 exposes a candidate ceiling but does not provide this recovery operation.", "Nonlearned new candidate generation is a separately scoped comparison, outside selected fixed-candidate top five."),
    assessment("run_b", [1, 1, 0], [
        ("initial", "TF04", "output", "Alternative cell-center trajectories and split-event proposals independently extracted from images"),
        ("final", "I09", "changed_mechanism", "Construct image-supported temporal regions, centers and split proposals without the existing predictor or candidate list."),
    ], "Independent raw-image trajectory generation omits recovery of existing discarded detections, explicit unanchored-track retention and compatible-free-endpoint attachment.", "Task-first trajectory proposal retained as I09; final motivation uses E06's fixed-node limitation, not a packet example of faint-track add-back.", "I09 requires a separately approved candidate comparison; it is outside top five and does not train a detector."))

add("R04",
    assessment("run_a", [1, 1, 1], [
        ("initial", "TF01", "objective", "未知部分には教師maskを使う"),
        ("final", "I01", "input_target_decode", "既知event対のmasked ranking lossでevent scoreを学び"),
    ], "Unknown masking is specified, but no evaluated-probability a versus conditional-correctness q decomposition or a*q/a*(1-q) candidate costs.", "Unknown-label masking is already required by the task policy and baseline; no independent discovery of the two-probability mechanism."),
    assessment("run_b", [1, 1, 1], [
        ("initial", "TF01", "teacher_assumptions", "unknown edges are masked"),
        ("final", "I01", "input_target_decode", "masked known-pair-versus-subset ranking trains a downstream scorer"),
    ], "No distinct probability of being scored and correctness conditional on scoring; masking/lower-bound targets alone are partial.", "Sparse-label masking is supplied by the task policy and prior evidence; the source probability decomposition is absent."))

add("R05",
    assessment("run_a", [2, 2, 2], [
        ("initial", "TF01", "target", "1母からの継続、2娘への分裂、対応なしを、node集合を共有する局所event集合として扱う"),
        ("initial", "TF01", "decode", "重なるeventのnode占有をそろえ、局所eventを共同選択して時間方向のgraphに戻す"),
        ("final", "I01", "changed_mechanism", "継続path・二娘分裂・保持eventを列挙し、共有nodeの競合を共同で解く"),
    ], reuse="TF01 supplies a task-first joint-event mechanism before E02/E08. Final I01 substantially reuses E02's mother-daughter set representation and E08's displaced-edge/singleton diagnostics, adding downstream three-frame relative selection; fidelity does not imply independent rediscovery of the whole source solution.", scope="Joint alternatives allow replacing current ordinary explanations under shared-node capacity; teacher completeness and error tradeoffs still need measurement."),
    assessment("run_b", [2, 2, 2], [
        ("initial", "TF01", "target", "Mutually compatible continuation, mother-to-two-daughters, boundary birth and termination events"),
        ("initial", "TF01", "decode", "one mother per daughter and compatible tracklet endpoints"),
        ("final", "I01", "changed_mechanism", "Replace independent-edge or flat-set decisions with scored, mutually compatible continuation and two-daughter events."),
    ], reuse="TF01 is task-first joint event selection. Final I01 explicitly derives separate cardinality and pair-vs-subset comparisons from E02/E08; existing historical set-selection information is reused, not independently recovered after seeing it.", scope="Whole-event competition replaces ordinary assignments subject to one-parent constraints; no observed accuracy gain is claimed."))

add("R06",
    assessment("run_a", [2, 2, 0], [
        ("initial", "TF06", "objective", "部分GEFFを中心近傍と既知対応の教師として使い、教師のないvoxelをnegativeにせず、画像一致の補助lossを併用する"),
        ("initial", "TF06", "decode", "対応分布の容量制約でcontinuation/division edgeを作る"),
        ("final", "I10", "input_target_decode", "raw画像から占有・対応分布を出し、疎い中心と既知edgeのmasked lossと画像一致で学び"),
    ], "The independent learned image-correspondence proposal is not in top five.", "Direct image/temporal teacher proposal is already present in task-first TF06, then retained as deferred I10; no historical learned-image-crop design was supplied.", "Precise for this broad independent-image-evidence criterion only. New image-model training is explicitly conditional; dense shape identifiability and cost are unresolved."),
    assessment("run_b", [2, 0, 0], [
        ("initial", "TF05", "objective", "Image reconstruction consistency with sparse known point-to-point correspondence loss"),
        ("initial", "TF05", "decode", "Jointly follow spatial probability mass and branch when two compatible daughter modes persist"),
    ], "TF05 is absent from final idea_cards and selected five; merely mentioning it in rejected/notes does not count as a final card.", "Task-first independent learned image correspondence; no later historical image-model mechanism was provided.", "TF05 acknowledges required image-learning policy change, missing supervision coverage and GPU cost. Full volumes versus crops do not invalidate this broad criterion."))

add("R07",
    assessment("run_a", [1, 1, 1], [
        ("initial", "TF03", "decode", "補正された局所scoreで既存候補を再選択"),
        ("final", "I02", "input_target_decode", "局所移動と親競合marginを入力し、既知別親だけのmasked lossで校正量を学び、固定候補を再選択する"),
    ], "Uses a current-graph motion reference and one downstream rescoring, but no provisional-solve division-cost revision and repeated constrained solve, or count-based objective relinearization.", "Task-first reference direction; final I02 reuses E03/E07 temporal evidence in a new role. Iterative topology-driven source mechanism is not demonstrated."),
    assessment("run_b", [1, 1, 1], [
        ("initial", "TF02", "objective", "robust context estimation, then masked conditional edge ranking"),
        ("final", "I02", "input_target_decode", "Context links estimate physical residuals; masked known-parent ranking trains downstream calibration"),
    ], "Once-used predicted-link reference does not specify division re-evaluation after a provisional solve, a repeat ILP, or expected-count relinearization.", "Task-first motion-reference proposal; final role change is motivated by E03/E07 failures, rather than a supplied iterative-selection method."))

add("R08",
    assessment("run_a", [0, 0, 0], missing="No high-confidence-first/lower-confidence-second ILP, first-pass-division cancellation, independent extra-cell retention and link-only third solve."),
    assessment("run_b", [0, 0, 0], missing="Pooling anchor/permissive events is not confidence-staged selection with cancellation of new second-pass divisions and final link re-solve."))

add("R09",
    assessment("run_a", [0, 0, 0], missing="No candidate-edge self-attention paired with independently contrastively trained image embeddings."),
    assessment("run_b", [0, 0, 0], missing="No edge-token attention or independent contrastive image representation; soft event fusion is a different operation."))

add("R10",
    assessment("run_a", [0, 0, 0], missing="Occupancy/temporal correspondence does not specify dense own-center and daughter-to-parent displacement votes or their integration into instances."),
    assessment("run_b", [1, 0, 0], [
        ("initial", "TF05", "target", "A spatial displacement distribution with continuation/division uncertainty"),
        ("initial", "TF05", "output", "Dense or locally queried transition distributions"),
    ], "Learned displacement distributions are related, but no own-center voxel votes, daughter-voxel parent-center consensus or instance recovery by vote integration. TF05 is absent from final cards.", "Task-first spatial-displacement proposal; no source vote-based mechanism in the historical packet.", "TF05 is explicitly deferred new-image training, not currently executable under the policy."))

add("R11",
    assessment("run_a", [1, 1, 1], [
        ("initial", "TF04", "output", "保持・削除・代替接続のscoreと信頼度"),
        ("final", "I03", "input_target_decode", "既知部分の保持・削除・親変更lossを学び、容量制約で修復edgeをdecodeする"),
    ], "Concrete learned edit scoring, but training targets reconstruct known motifs rather than signed official TP/FP/node-count edit utility; later official evaluation is not such a teacher.", "Task-first semantic repair model, later corruption/pretrain/refine design; packet does not supply the rank-17 metric-utility teacher."),
    assessment("run_b", [1, 1, 1], [
        ("initial", "TF03", "output", "Probabilities of retaining, deleting or exchanging complete continuation/division events"),
        ("final", "I03", "input_target_decode", "compatible graph edits through masked motif ranking"),
    ], "Learned edits have motif-ranking/reconstruction teachers, not signed utility from evaluated TP/FP and node costs or exact official delta.", "Task-first repair direction, later corrupted-conditioning training; no historical metric-labeled-edit mechanism in the packet."))

add("R12",
    assessment("run_a", [0, 0, 0], missing="No explicit daughter-track completion followed by final division recheck on the completed graph and competing ordinary edges."),
    assessment("run_b", [0, 0, 0], missing="Longer tracklet scoring and generic graph repair do not specify daughter completion before final division re-evaluation."))

add("R13",
    assessment("run_a", [0, 0, 0], missing="Corrupted-graph repair training does not feed completed predicted-track nodes into weighted detector self-training."),
    assessment("run_b", [0, 0, 0], missing="No final-track-derived pseudo-label generation and new detector-training round."))

add("R14",
    assessment("run_a", [0, 1, 0], [
        ("final", "I06", "changed_mechanism", "提出座標zero/refinedと特徴補間位置zero/refinedの2×2比較を同じnode identityで行う"),
    ], "Crossed coordinate/feature-route audit does not explicitly apply a final correction with graph links held fixed; no carried/restored division operation. I06 is outside top five.", "Explicitly motivated by baseline exp043 and E04/E05's existing coordinate correction and unresolved feature-routing confound; this is historical reuse plus causal diagnosis.", "Fixed upstream routes are compatible, but required feature recapture and graph-link constancy remain unspecified."),
    assessment("run_b", [0, 1, 0], [
        ("final", "I08", "changed_mechanism", "Cross zero/refined geometry with zero/refined feature sampling"),
    ], "Ablation of geometry/sampling routes does not implement final-graph coordinate correction with identities and links fixed, or division restoration; I08 is unselected.", "E04/E05 already specify coordinate correction and the unseparated routes; final I08 develops a diagnostic of those historical changes.", "No new raw-image learning, but feature recapture cost is missing and locked-ID diagnostics do not replace official matching."))

add("R15",
    assessment("run_a", [0, 0, 0], missing="No sparse local graph-attention network with image-learned velocity/uncertainty jointly trained with detection."),
    assessment("run_b", [0, 0, 0], missing="Robust sequence-motion uncertainty is not sparse graph attention or image-derived motion/uncertainty joint detect/link learning."))

add("R16",
    assessment("run_a", [0, 0, 0], missing="Complete division-event ranking is not explicit learned birth/no-parent and mother-division heads with calibrated start/division costs."),
    assessment("run_b", [1, 1, 1], [
        ("initial", "TF01", "target", "boundary birth and termination events"),
        ("final", "I01", "title", "Rank complete division and continuation events with separate cardinality scores"),
        ("final", "I01", "preserved_invariants", "unknown outdegree remains masked"),
    ], "Initial event proposal mentions births; final cardinality scoring covers a division component. Neither specifies separate learned no-parent probabilities, their sparse birth teacher, and probability-derived graph start/division costs.", "Task packet already mentions exp028 mother/no-match prediction. Final cardinality refinement explicitly reuses E02 set-selection evidence; full explicit birth/prior mechanism is not recovered.", "Partial downstream heads over frozen features are within policy; unknown births must not be inferred from missing annotation."))

criteria = []
for criterion in rubric["criteria"]:
    criteria.append({
        "id": criterion["id"],
        "source_ranks": criterion["source_rank"],
        "policy": criterion["policy"],
        "mechanism": criterion["mechanism"],
        "runs": rows[criterion["id"]],
    })

counts = {}
for run in documents:
    counts[run] = {}
    for stage in STAGES:
        counts[run][stage] = {}
        for policy in ("within_policy", "change_required", "all"):
            values = [row["runs"][run][stage] for row in criteria if policy == "all" or row["policy"] == policy]
            counts[run][stage][policy] = {
                "criteria": len(values),
                "precise_matches": values.count(2),
                "partial_matches": values.count(1),
                "absent": values.count(0),
            }

payload = {
    "schema_version": "1",
    "effective_rubric_sha256": "c7080374b467d2813189756d5c1606ec90df4d40017025a6ae5ed2de1d96eca8",
    "original_rubric_sha256": "493e45ca3bf3875c4b8acc70ff075dc1ce5f5543e795799db0993ec2ce3ba199",
    "rubric_source_audit": "judge/rubric_source_audit.json",
    "output_freeze": "final_output_freeze.json",
    "frozen_file_sha_verification": "PASS for all eight supplied run files",
    "stage_definition": {
        "task_first": "Maximum per criterion over exactly six frozen initial_ideas, before baseline_contract/evidence reading.",
        "all_cards": "Maximum per criterion over the ten final idea_cards only; pass notes, assumptions and rejected proposals do not add mechanisms.",
        "top_five": "Maximum per criterion over the five selected final cards; both runs selected I01–I05.",
        "rating": "0 absent/incompatible/label-only; 1 concrete related direction missing a defining step; 2 specified mechanism fidelity, never observed accuracy or winning-solution replication.",
    },
    "methodological_notes": [
        "Primary-source fidelity was audited and effective errata finalized before generator outputs were accessed. The original rubric remains preserved.",
        "Both runs had task-first hashes frozen before detailed historical evidence and final files frozen before scoring. Access logs report no forbidden source, Web, rubric, winning writeup or other-run access; the parent separately reports provenance and schema PASS.",
        "Parent interrupted extended drafting and resumed each same context to finalize exactly ten concise cards from existing own passes, without new evidence or target/rubric hints. This intervention is recorded in finalization_intervention.json; the workflow was not unattended.",
        "Each generator used sequential passes in one context, not independent fresh contexts per pass. Two runs and a selected, overlapping sixteen-criterion rubric cannot estimate discovery probabilities, model capability probabilities or leaderboard performance.",
        "Partial matches explicitly distinguish related operations: independent raw proposals from unused-detection add-back; semantic graph repair from metric-utility teachers; route ablation from final geometry correction; one-pass reference calibration from iterative topology reoptimization.",
        "No mechanisms were repaired or supplied by the evaluator. Shorter finalization omits some notes-only ideas; these cannot count as final-card recall.",
        "Historical mother-daughter set and candidate-competition evidence substantially informs final R05 and B's cardinality direction. Task-first and post-evidence stages therefore have different discovery interpretations.",
        "Task-first means before the baseline contract and numerical history, not before all historical method information: task_packet already contains the policy section naming exp028 and daughter-to-mother/no-match teaching.",
        "Mechanism counts are unweighted counts of precise and partial rubric matches, not an overall performance grade or proportion of source-solution value.",
    ],
    "scope_safety_notes": {
        "common": [
            "Selected I01–I05 explicitly preserve frozen detector, encoder, coordinate head and existing tracker. They define label-free hidden inputs, unknown-label masking, training-side tuning, anchor fallback, early diagnostic gates and both-embryo official full-graph evaluation.",
            "Public upstream model/head exposure remains, so proposed outer-embryo downstream tests are conditional comparisons rather than independent image-model CV. Sparse teachers cannot verify complete real negative labels or dense instance masks.",
            "Runtime, cache completeness, solver constraints, GPU quota and teacher counts are missing; plans to measure them are not evidence of 45-GPU-hour/12-hour feasibility or improved score.",
            "Event cardinality, no-parent/birth and synthetic motif completion require care: absent annotation cannot certify null/birth/no-division. Both acknowledge that risk; complete deployable birth teachers are not specified.",
            "Exact decomposition requires the complete variable-constraint incidence structure, tie behavior and worst-component measurements. Faster exact solving of the unchanged objective cannot itself improve prediction.",
            "Observed metric gains can reflect node-count adjustment and matching rather than better tracking. No proposed teacher/readout constitutes a measured leaderboard gain.",
        ],
        "run_a": [
            "TF06/I10 explicitly require new image-learning approval and remain outside the selected five. Their precise R06 match is broad independent image-correspondence fidelity, not a feasible replication of any source architecture.",
            "TF02/I07 raw-image candidate recovery is a separate comparison, not a silently changed fixed-candidate selected experiment. I08's image changes require consistent windowed feature re-extraction; this cost is acknowledged but unmeasured.",
        ],
        "run_b": [
            "TF05 explicitly requires new image learning and is removed from final cards, appearing only as a rejected immediate-action proposal. It receives task-first R06/R10 credit only.",
            "TF04/I09 independent image-component candidates require a separately approved candidate comparison and remain outside selected five. I10 deliberately excludes transformed appearance features to avoid assuming frozen encoder equivariance.",
        ],
    },
    "criteria": criteria,
    "counts": counts,
    "not_claimed": ["Observed accuracy improvement", "Expected public/private score", "Replication of complete winning solutions", "Discovery probability", "Unattended workflow success"],
}

(ROOT / "judge" / "scores.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(counts, ensure_ascii=False, indent=2))
