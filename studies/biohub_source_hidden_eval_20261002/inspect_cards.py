"""Print compact, read-only views of the frozen generated cards."""

import json
from pathlib import Path

base = Path(__file__).resolve().parent
for run in ("run_a", "run_b"):
    data = json.loads((base / run / "idea_portfolio.json").read_text())
    print(run)
    for card in data["idea_cards"]:
        print(json.dumps({key: card.get(key) for key in (
            "id", "title", "input_target_decode", "exact_difference",
            "origin_pass", "evidence_ids",
        )}, ensure_ascii=False))
    print(json.dumps({key: value for key, value in data.items()
                      if "selection" in key or "top" in key}, ensure_ascii=False))
