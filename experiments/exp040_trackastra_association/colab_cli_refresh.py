"""Refresh a named CLI session's proxy credentials for its existing assignment."""

from __future__ import annotations

import argparse

from colab_cli.common import state


def refresh(session_name: str) -> None:
    session = state.store.get(session_name)
    if session is None:
        raise RuntimeError("Colab session is absent from local state")
    matches = [
        item for item in state.client.list_assignments() if item.endpoint == session.endpoint
    ]
    if len(matches) != 1:
        raise RuntimeError("Colab assignment endpoint is unavailable or ambiguous")
    assignment = matches[0]
    if assignment.accelerator.value.lower() != "t4":
        raise RuntimeError("Colab assignment is no longer T4")
    session.token = assignment.runtime_proxy_info.token
    session.url = assignment.runtime_proxy_info.url
    state.store.add(session)
    print("Refreshed proxy credentials for the same active T4 assignment", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    refresh(parser.parse_args().session)


if __name__ == "__main__":
    main()
