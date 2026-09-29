"""Record the saved-result analysis without changing the experiment decision."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXP = ROOT / "experiments/exp051_x138_ilp_score_clipping"
CONTROL = "expanded_optuna_cost"
CLIPPED = CONTROL + "_clipped"


def main():
    report = json.loads((HERE / "analysis.json").read_text())
    metrics_path = EXP / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    previous_status = metrics["status"]
    groups = report["groups"]
    inner_control = report["tuning"][f"0/{CONTROL}"]["selected_summary"]
    inner_clipped = report["tuning"][f"0/{CLIPPED}"]["selected_summary"]
    inner_score_delta = inner_clipped["score"] - inner_control["score"]
    inner_division_bonus_delta = 0.1 * (
        inner_clipped["division_jaccard"] - inner_control["division_jaccard"]
    )
    evidence = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "scope": report["scope"],
        "files": {
            name: {
                "path": str((HERE / name).relative_to(ROOT)),
                "sha256": hashlib.sha256((HERE / name).read_bytes()).hexdigest(),
            }
            for name in (
                "analyze.py",
                "record_findings.py",
                "analysis.json",
                "videos.csv",
                "stages.csv",
                "trials.csv",
            )
        },
        "input_summary_sha256": report["sources"],
        "official_score_reaggregation_tolerance": 1e-12,
        "final_node_count_matches_official": True,
        "groups": {
            name: {key: group[key] for key in ("arms", "clipped_minus_control", "edge_changes")}
            for name, group in groups.items()
        },
        "all_video_stage_counts": {
            name: {
                "graph": groups["all"]["graph_stages"][name],
                "fixed_id_diagnostic": groups["all"]["stages"][name],
            }
            for name in ("ilp", "relink", "division", "final")
        },
        "selected_tuning": {
            key: {name: value for name, value in tuned.items() if name != "trials"}
            for key, tuned in report["tuning"].items()
        },
        "inner_44b6_to_outer_6bba": {
            "inner_score_delta": inner_score_delta,
            "inner_division_bonus_delta": inner_division_bonus_delta,
            "fraction_of_inner_gain_from_division_bonus": (
                inner_division_bonus_delta / inner_score_delta
            ),
            "inner_annotated_divisions": (
                inner_control["division_tp"] + inner_control["division_fn"]
            ),
            "inner_control_correct_divisions": inner_control["division_tp"],
            "inner_clipped_correct_divisions": inner_clipped["division_tp"],
        },
        "interpretation_limits": [
            "Aggregate score decomposition is arithmetic, not an isolated causal experiment.",
            "Costs and cap were selected jointly; clipping alone was not isolated.",
            "Fixed original-candidate matching differs from official final-graph matching.",
            "Excluding timeout videos is post-hoc diagnosis, not a replacement official score.",
            "Optimal results for the two time-limited graphs are unknown.",
            "Both arms use CPU postprocessing; CPU/GPU equivalence is unverified.",
            "No new inference, tuning, adoption decision, or submission was performed.",
        ],
    }
    metrics["evidence"]["score_degradation_analysis"] = evidence
    assert metrics["status"] == previous_status
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    note = (
        "- 2026-09-29: ユーザーのスコア低下原因の考察依頼を受け、保存済み公式評価・"
        "調整4 receipt・400 stageを`studies/exp051_clipping_failure_20260929/analyze.py`で再集計。"
        "公式scoreの再集計を1e-12以内で照合し、最終stageの細胞点数も公式値と一致。"
        "全体差-0.006121は接続Jaccard差-0.003475と細胞数補正寄与差-0.002646に分かれ、"
        "公式TPは6減、FPは41増、予測細胞点は12,957増。ILPでの既知接続の差+105が"
        "relink後+33、最終+21へ縮み、上限側でILPが回収した既知分裂3件はrelink後0。"
        "6bba向け設定の内部score差+0.027376のうち+0.025は分裂1件の回収によるが、"
        "外側では両条件とも分裂TP0。6bbaで複数親が同点化した娘は6個のみ。"
        "時間切れ2動画を両条件から除いた18動画でも差-0.003136。"
        "費用と上限を別々に変えた比較ではないため因果を分離せず、後処理による変更と"
        "少数分裂事例から別胚への一般化の限界をresult.mdに記録。"
        "各集計・解析コードのSHAはmetricsのevidence.score_degradation_analysisを正とする。"
        "追加推論・再調整・Notebook再push・採否/完了判断・commit/pushは行っていない。\n"
    )
    notes_path = EXP / "SESSION_NOTES.md"
    notes = notes_path.read_text()
    if note not in notes:
        notes_path.write_text(notes.rstrip() + "\n" + note)
    print(
        json.dumps(
            {
                "status_preserved": previous_status,
                "analysis_sha256": evidence["files"]["analysis.json"]["sha256"],
                "inner_division_share": evidence["inner_44b6_to_outer_6bba"][
                    "fraction_of_inner_gain_from_division_bonus"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
