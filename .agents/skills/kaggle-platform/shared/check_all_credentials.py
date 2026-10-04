#!/usr/bin/env python3
"""Unified Kaggle credential checker.

Checks configured credential sources used by the locked Kaggle clients:
  1. KAGGLE_API_TOKEN env var (token value or existing token-file path)
  2. ~/.kaggle/access_token, then access_token.txt
  3. KAGGLE_USERNAME + KAGGLE_KEY env vars (legacy)
  4. ~/.kaggle/kaggle.json (legacy)
  5. ~/.kaggle/credentials.json (OAuth login for Kaggle CLI and Python API)

Returns structured JSON output for easy parsing.
Never prints actual credential values — only masked status.

Usage:
    uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py
    uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --json
    uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --require api-token
    uv run python \
        .agents/skills/kaggle-platform/shared/check_all_credentials.py --require python-api
    uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --require cli
    uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --require kagglehub

The default ``--require any`` accepts any usable credential source. Use
``api-token`` for MCP and other Bearer-token-only clients, ``python-api`` for
Kaggle Python API operations, ``kagglehub`` for the separate kagglehub client,
and ``cli`` for Kaggle CLI operations. This is a local configuration check,
not authentication or a check of expiry, permissions, or network connectivity.

Exit codes:
    0 — A credential source satisfying the selected requirement was found
    1 — No configured credential source satisfies the selected requirement
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from kagglesdk.kaggle_env import get_access_token_from_env


def _ensure_mode_600(path: Path) -> None:
    """Auto-tighten file mode to 600 if anything else is set.

    Credential files must never be group- or world-readable. Previously this
    only warned and continued; now it self-heals because credentials in a
    world-readable file are an active leak, not a future risk.
    """
    mode = path.stat().st_mode & 0o777
    if mode != 0o600:
        try:
            path.chmod(0o600)
            print(f"[INFO] Tightened {path} permissions from {oct(mode)[-3:]} to 600")
        except OSError as e:
            print(f"[WARN] {path} permissions are {oct(mode)[-3:]}, could not chmod 600: {e}")


def _resolve_api_token() -> tuple[str, str | None]:
    """Use the same resolver as locked Kaggle CLI, Python API, and kagglehub.

    It handles token-file environment values and the .txt fallback. In
    particular, an explicitly selected empty file must not fall back to a
    different token or count the path itself as a credential.
    """
    token, source = get_access_token_from_env()
    if token:
        if source == "KAGGLE_API_TOKEN":
            configured = Path(os.environ["KAGGLE_API_TOKEN"])
            if configured.is_file():
                _ensure_mode_600(configured)
        elif source == "access_token":
            for filename in ("access_token", "access_token.txt"):
                configured = Path.home() / ".kaggle" / filename
                if configured.is_file():
                    _ensure_mode_600(configured)
    label = "env" if source == "KAGGLE_API_TOKEN" else source
    return token or "", label


def _read_kaggle_json() -> dict:
    """Read ~/.kaggle/kaggle.json if it exists and is valid."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        return {}
    try:
        creds = json.loads(kaggle_json.read_text())
        _ensure_mode_600(kaggle_json)
        if not isinstance(creds, dict):
            raise ValueError("Expected credential object")
        return creds
    except (OSError, ValueError):
        print(f"[WARN] {kaggle_json} exists but is malformed")
        return {}


def _oauth_credentials_path() -> Path:
    """Return the OAuth path shared by Kaggle CLI and Kaggle Python API."""
    return Path.home() / ".kaggle" / "credentials.json"


def _has_oauth_credentials() -> bool:
    """Check the local shape consumed by kagglesdk.KaggleCredentials.load.

    Never return or print the token. An expired access token can be refreshed
    by the client; this check does not contact Kaggle or validate revocation.
    """
    path = _oauth_credentials_path()
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return False
        refresh_token = data.get("refresh_token")
        if not isinstance(refresh_token, str) or not refresh_token.strip():
            return False
        expiration = data.get("access_token_expiration")
        if expiration:
            parsed = datetime.fromisoformat(expiration)
            if parsed.tzinfo is None:
                return False
        _ensure_mode_600(path)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _mask(value: str, prefix_len: int = 0) -> str:
    """Mask a credential value, showing only first prefix_len and last 4 chars."""
    if not value:
        return "****"
    if len(value) <= prefix_len + 4:
        return "****"
    return value[:prefix_len] + "*" * max(0, len(value) - prefix_len - 4) + value[-4:]


def check_all_credentials(output_json: bool = False, requirement: str = "any") -> bool:
    """Check credentials and enforce an optional client-specific requirement."""
    if requirement not in {"any", "api-token", "python-api", "cli", "kagglehub"}:
        raise ValueError(f"Unsupported credential requirement: {requirement}")
    results = {}
    found_any = False

    # KAGGLE_TOKEN is not a supported alias: its token type is ambiguous, and
    # treating it as a legacy KAGGLE_KEY can silently select the wrong auth
    # mechanism.
    if os.getenv("KAGGLE_TOKEN") and not os.getenv("KAGGLE_API_TOKEN"):
        print("[WARN] KAGGLE_TOKEN is not used; rename it to KAGGLE_API_TOKEN")

    # --- API Token (primary, recommended) ---
    # An explicit environment value must override a persistent local file so
    # CI, Colab, and managed secret stores can select their own credential.
    api_token, source = _resolve_api_token()

    if api_token:
        results["KAGGLE_API_TOKEN"] = {
            "status": "OK",
            "value": _mask(api_token, 5),
            "source": source,
            "type": "API token",
        }
        print(f"[OK] API Token: {_mask(api_token, 5)} (from {source})")
        found_any = True
    else:
        results["KAGGLE_API_TOKEN"] = {"status": "MISSING", "value": None, "source": None}
        print("[MISSING] API Token")
        print("          Generate at: https://www.kaggle.com/settings")
        print("          → API Tokens (Recommended) → Generate New Token")
        print("          Save as ~/.kaggle/access_token or set KAGGLE_API_TOKEN env var")

    # --- OAuth credentials from `kaggle auth login` ---
    oauth_path = _oauth_credentials_path()
    oauth_found = _has_oauth_credentials()
    if oauth_found:
        _ensure_mode_600(oauth_path)
        results["KAGGLE_OAUTH_CREDENTIALS"] = {
            "status": "OK",
            "value": None,
            "source": str(oauth_path),
        }
        print(f"[OK] OAuth credentials: {oauth_path} (from kaggle auth login)")
        found_any = True
    else:
        results["KAGGLE_OAUTH_CREDENTIALS"] = {
            "status": "MISSING",
            "value": None,
            "source": None,
        }
        print(
            "[INFO] OAuth credentials not found "
            "(optional; run `uv run kaggle auth login` for CLI / Python API auth)"
        )

    # --- Legacy credentials (optional) ---
    kaggle_json_data = _read_kaggle_json()

    # Kaggle API merges environment settings with its legacy config; kagglehub
    # accepts either a complete environment pair or a complete file pair.
    legacy_env = {
        "username": os.getenv("KAGGLE_USERNAME"),
        "key": os.getenv("KAGGLE_KEY"),
    }
    if requirement == "kagglehub" and not all(legacy_env.values()):
        legacy_env = {}

    # KAGGLE_USERNAME
    username = legacy_env.get("username") or kaggle_json_data.get("username")
    if username:
        source = "env" if legacy_env.get("username") else "kaggle.json"
        results["KAGGLE_USERNAME"] = {"status": "OK", "value": username, "source": source}
        print(f"[OK] KAGGLE_USERNAME: {username} (from {source})")
    else:
        results["KAGGLE_USERNAME"] = {"status": "MISSING", "value": None, "source": None}
        print("[INFO] KAGGLE_USERNAME not set (optional with API token)")

    # KAGGLE_KEY
    key = legacy_env.get("key") or kaggle_json_data.get("key")
    if key:
        source = "env" if legacy_env.get("key") else "kaggle.json"
        results["KAGGLE_KEY"] = {
            "status": "OK",
            "value": _mask(key),
            "source": source,
            "type": "Legacy API key",
        }
        print(f"[OK] KAGGLE_KEY: {_mask(key)} (Legacy API key, from {source})")
    else:
        results["KAGGLE_KEY"] = {"status": "MISSING", "value": None, "source": None}
        if not api_token:
            print("[MISSING] KAGGLE_KEY")
            print("          Legacy API key. Generate at: https://www.kaggle.com/settings")
            print("          → Legacy API Credentials → Create Legacy API Key")
        else:
            print("[INFO] KAGGLE_KEY not set (optional when API token is available)")

    if key and not username:
        print("[WARN] Legacy KAGGLE_KEY is unusable without KAGGLE_USERNAME")

    # --- Summary ---
    legacy_pair_found = bool(username and key)
    found_any = bool(api_token) or oauth_found or legacy_pair_found
    print()
    if requirement == "api-token":
        requirement_met = bool(api_token)
    elif requirement == "kagglehub":
        requirement_met = bool(api_token) or legacy_pair_found
    else:
        requirement_met = found_any

    if found_any:
        if api_token:
            print("API token found — you're ready to go!")
            print("(Supported by kaggle CLI >= 1.8.0, kagglehub >= 0.4.1, MCP Server)")
        elif legacy_pair_found:
            print(
                "Legacy username/key credentials found — supported by CLI and Python API clients."
            )
            print("MCP operations require an API token from Generate New Token.")
        elif oauth_found:
            print("OAuth credentials configured for Kaggle CLI and Kaggle Python API.")
            print("kagglehub needs an API token or a legacy username/key pair.")
    else:
        print("No Kaggle credentials found. To set up:")
        print()
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Under 'API Tokens (Recommended)', click 'Generate New Token'")
        print("  3. Keep the token local; do not paste it into chat or command arguments")
        print()
        print("     # Option 0: Interactive CLI OAuth login")
        print("     uv run kaggle auth login")
        print()
        print("     # Option A: Local hidden-input helper")
        print(
            "     uv run python .agents/skills/kaggle-platform/modules/registration/"
            "scripts/configure_token.py"
        )
        print()
        print(
            "  Full guide: .agents/skills/kaggle-platform/modules/registration/"
            "references/kaggle-setup.md"
        )

    if found_any and not requirement_met:
        print()
        print("Configured credentials do not satisfy this client's requirement.")
        if requirement == "api-token":
            print("This operation requires an API token from Generate New Token.")
            print("OAuth and legacy username/key credentials cannot authenticate MCP.")
        elif requirement == "kagglehub":
            print("This operation requires an API token or a legacy username/key pair.")
            print("kagglehub does not load the CLI / Python API OAuth credential file.")

    if output_json:
        print()
        print("--- JSON ---")
        print(json.dumps(results, indent=2))

    return requirement_met


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check configured Kaggle credentials")
    parser.add_argument("--json", action="store_true", help="Print structured JSON details")
    parser.add_argument(
        "--require",
        choices=("any", "api-token", "python-api", "cli", "kagglehub"),
        default="any",
        help="Require credentials compatible with a specific client type",
    )
    args = parser.parse_args()
    ok = check_all_credentials(output_json=args.json, requirement=args.require)
    sys.exit(0 if ok else 1)
