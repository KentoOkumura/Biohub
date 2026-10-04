"""Shared utilities for Kaggle competition report generation."""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Rate limiting: seconds between API calls
API_DELAY = 3

# Skill root: 3 levels up from modules/comp-report/scripts/utils.py
SKILL_ROOT = Path(__file__).resolve().parents[3]


def get_api():
    """Initialize and authenticate the Kaggle API client."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def get_username() -> str:
    """Get the Kaggle username from env or kaggle.json."""
    username = os.getenv("KAGGLE_USERNAME")
    if username:
        return username
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        creds = json.loads(kaggle_json.read_text())
        return creds.get("username", "")
    return ""


def check_credentials() -> bool:
    """Verify Kaggle credentials are configured and API authenticates."""
    checker_path = SKILL_ROOT / "shared" / "check_all_credentials.py"
    spec = importlib.util.spec_from_file_location("kaggle_credentials", checker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load credential checker: {checker_path}")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    if not checker.check_all_credentials(requirement="python-api"):
        return False

    # Try to authenticate
    try:
        api = get_api()
        # Quick check: list competitions to verify auth works
        result = api.competitions_list(page=1)
        comps = unwrap_response(result, "competitions")
        username = get_username()
        print(f"OK: Kaggle API authenticated as '{username}'")
        print(f"  API returned {len(comps)} competition(s) in smoke test")
        return True
    except (Exception, SystemExit) as e:
        print(f"ERROR: Kaggle API authentication failed: {e}")
        return False


def unwrap_response(result, attr: str = "competitions") -> list:
    """Unwrap a Kaggle API response object to get the inner list.

    The newer kagglesdk returns response objects (e.g. ApiListCompetitionsResponse)
    with the actual data in a named attribute (e.g. .competitions). Older versions
    returned plain lists. This handles both.
    """
    if isinstance(result, list):
        return result
    if hasattr(result, attr):
        return getattr(result, attr) or []
    # Try common attributes
    for fallback in ["competitions", "files", "kernels", "results"]:
        if hasattr(result, fallback):
            return getattr(result, fallback) or []
    # Last resort: try to iterate
    try:
        return list(result)
    except TypeError:
        return []


def rate_limit():
    """Sleep for API_DELAY seconds to avoid throttling."""
    time.sleep(API_DELAY)


if __name__ == "__main__":
    ok = check_credentials()
    sys.exit(0 if ok else 1)
