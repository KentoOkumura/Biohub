"""Freeze historical evidence separately from the withheld assessment criteria."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PACKET = BASE / "packet"
CUTOFF = "7b648a3"


def historical(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{CUTOFF}:{path}"], cwd=ROOT, text=True
    )


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def compact(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: compact(item)
            for key, item in value.items()
            if key not in {
                "artifacts", "artifact_sha256", "reproducibility", "per_movie",
                "per_video", "submission_validation", "reruns", "notes",
                "environment", "command", "commands", "kernels", "kernel",
                "benchmark", "folds", "per_dataset", "per_sample", "per_stem",
            }
            and not any(part in key.lower() for part in ("sha", "path", "receipt"))
        }
    if isinstance(value, list):
        if len(value) > 5:
            return {"omitted_list_entries": len(value), "reason": "Packet keeps aggregate evidence; long row/event lists remain in audit_sources."}
        return [compact(item) for item in value]
    return value


def main() -> None:
    PACKET.mkdir(parents=True, exist_ok=True)
    sources = []
    task_parts = [
        "# 発想評価用の入力\n\n"
        "情報時点: 2026-09-27 17:41:06 JST / Git 7b648a3。\n"
        "目的は当時の情報から反証可能な改善案を作ること。学習・提出は行わない。\n"
        "現在のskillを当時の情報へ適用する評価であり、当時のモデルやskillの再現ではない。\n"
        "事実として不明な入力・教師・費用は不明のまま扱う。\n"
    ]
    for path in ("docs/01_competition.md", "docs/02_metric.md", "docs/04_data.md"):
        content = historical(path)
        # Keep task and factual specifications, not infrastructure implementation notes.
        if path.endswith("01_competition.md"):
            content = content.split("## リポジトリ設定の確認結果")[0]
        elif path.endswith("02_metric.md"):
            content = content.split("## ローカル実装")[0]
        task_parts.append(f"\n## 入力資料: {path}\n\n{content}")
        sources.append({"path": path, "git_commit": CUTOFF, "source_sha256": sha(historical(path)), "selection": "task/factual sections"})
    direction = historical("backlog/KAGGLE_DIRECTION.md")
    policy = direction.split("## 今後の学習方針\n", 1)[1].split("## 承認済みの進め方", 1)[0]
    task_parts.append("\n## 当時の学習・計算方針\n\n" + policy)
    task_parts.append(
        "\n## 評価の作業範囲\n\n"
        "主な5案は上記方針で実行可能なものから選ぶ。"
        "方針変更を必要とする案も探索の候補として記載できるが、"
        "必要な変更を明記し、承認済みの実験や現在実行可能な案と混同しない。"
        "新しい画像モデルを固定特徴で置き換えた場合は別案とする。"
        "リンク先へ移動せず、渡した資料の本文だけを使う。\n"
    )
    sources.append({"path": "backlog/KAGGLE_DIRECTION.md", "git_commit": CUTOFF, "source_sha256": sha(direction), "selection": "今後の学習方針のみ; candidate index excluded"})
    (PACKET / "task_packet.md").write_text("\n".join(task_parts))
    evidence = {
        "evidence_cutoff": "2026-09-27T17:41:06+09:00",
        "git_commit": CUTOFF,
        "interpretation": [
            "各値は時点固定したmetricsからの抜粋。公開画像モデルの学習来歴を含む条件付き比較で、画像モデルまで独立したCVではない。",
            "primary validationは一方の胚で下流モデルを学習し反対胚で評価する2方向。ただし公開画像モデル由来の学習露出は残る。",
            "公開順位表のスコアは公式submission、trainの診断値とは区別する。Private scoreはまだ取得していない。",
            "固定公開モデルの候補・特徴を使用する。必要な生成物の保存範囲とGPU残量は、資料にない場合は未確認。",
            "exp050・051はこの時点では進行中。以後の結果を仮定しない。",
            "先行実験の失敗はその実装・教師・比較条件の証拠であり、別の使い方全体を否定するものではない。",
        ],
        "evidence": [],
    }
    experiments = [
        "exp003_official_metric_audit", "exp029_mother_daughter_set_selection",
        "exp041_past_feature_cross_attention", "exp043_x138_self_trained_head",
        "exp045_x138_coordinate_effect_audit", "exp047_x138_edge_candidates",
        "exp048_x138_primary_past_feature_attention", "exp049_x138_edge_selection_diagnostic",
    ]
    raw_dir = BASE / "audit_sources"
    raw_dir.mkdir(exist_ok=True)
    for number, exp in enumerate(experiments, 1):
        path = f"experiments/{exp}/metrics.json"
        raw = historical(path)
        parsed = json.loads(raw)
        (raw_dir / f"{exp}_metrics.json").write_text(raw)
        evidence["evidence"].append({"id": f"E{number:02}", "source": f"{CUTOFF}:{path}", "metrics": compact(parsed)})
        sources.append({"path": path, "git_commit": CUTOFF, "source_sha256": sha(raw), "selection": "structured numerical evidence; artifact hashes, prose notes, detailed benchmark/fold/per-sample tables and lists longer than five entries omitted"})
    baseline_path = "experiments/exp043_x138_self_trained_head/requirements.md"
    baseline = historical(baseline_path)
    (PACKET / "baseline_contract.md").write_text(baseline)
    sources.append({"path": baseline_path, "git_commit": CUTOFF, "source_sha256": sha(baseline), "selection": "complete approved baseline contract"})
    save_json(PACKET / "evidence.json", evidence)
    for source, destination in (
        (ROOT / ".agents/skills/kaggle-idea-forge/SKILL.md", PACKET / "idea_forge_skill.md"),
        (ROOT / ".agents/skills/kaggle-idea-forge/references/portfolio-schema.md", PACKET / "portfolio_schema.md"),
    ):
        shutil.copyfile(source, destination)
        sources.append({"path": str(source.relative_to(ROOT)), "git_commit": "current file", "source_sha256": sha(source.read_text()), "selection": "current evaluation method; not historical evidence"})
    file_hashes = {str(path.relative_to(BASE)): sha(path.read_text()) for path in sorted(PACKET.iterdir()) if path.is_file()}
    save_json(BASE / "input_manifest.json", {
        "cutoff_commit": subprocess.check_output(["git", "rev-parse", CUTOFF], cwd=ROOT, text=True).strip(),
        "cutoff_time_jst": "2026-09-27T17:41:06+09:00",
        "source_selection": "Specification, historical policy, baseline contract, numerical results. No prior idea reports, candidate files, winning writeups, later results, current strategy index or Web.",
        "sources": sources, "packet_sha256": file_hashes,
        "limitation": "Packet assembled by a parent already familiar with the solutions. Historical byte provenance is verified, but evidence selection bias remains possible.",
    })
    for run in ("run_a", "run_b"):
        (BASE / run).mkdir(exist_ok=True)
    print(json.dumps({"packet_files": file_hashes, "evidence_bytes": (PACKET / "evidence.json").stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
