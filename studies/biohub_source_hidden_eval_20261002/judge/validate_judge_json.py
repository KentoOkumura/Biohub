"""Validate only explicitly supplied judge JSON files; never discover run files."""

import json
import sys
from pathlib import Path


for argument in sys.argv[1:]:
    path = Path(argument)
    payload = json.loads(path.read_text())
    if "criteria" in payload:
        expected = [f"R{index:02d}" for index in range(1, 17)]
        assert [entry["id"] for entry in payload["criteria"]] == expected
    print(f"Valid JSON: {path}")
