"""State machine tracking via structured Linear comments."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("stokowski.tracking")

STATE_PATTERN = re.compile(r"<!-- stokowski:state ({.*?}) -->")
GATE_PATTERN = re.compile(r"<!-- stokowski:gate ({.*?}) -->")


def make_state_comment(state: str, run: int = 1, workflow: str | None = None) -> str:
    """Build a structured state-tracking comment.

    The workflow name rides along so a restart can recover which pipeline an
    in-flight issue was running. Without it, `_resolve_current_state` would
    re-route from labels — and a label edited mid-run would silently move the
    issue onto a different state machine.
    """
    payload: dict[str, Any] = {
        "state": state,
        "run": run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if workflow:
        payload["workflow"] = workflow

    machine = f"<!-- stokowski:state {json.dumps(payload)} -->"
    human = f"**[Stokowski]** Entering state: **{state}** (run {run})"
    if workflow:
        human += f" · workflow `{workflow}`"
    return f"{machine}\n\n{human}"


def make_gate_comment(
    state: str,
    status: str,
    prompt: str = "",
    rework_to: str | None = None,
    run: int = 1,
) -> str:
    """Build a structured gate-tracking comment."""
    payload: dict[str, Any] = {
        "state": state,
        "status": status,
        "run": run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if rework_to:
        payload["rework_to"] = rework_to

    machine = f"<!-- stokowski:gate {json.dumps(payload)} -->"

    if status == "waiting":
        human = f"**[Stokowski]** Awaiting human review: **{state}**"
        if prompt:
            human += f" — {prompt}"
    elif status == "approved":
        human = f"**[Stokowski]** Gate **{state}** approved."
    elif status == "rework":
        human = (
            f"**[Stokowski]** Rework requested at **{state}**. "
            f"Returning to: **{rework_to}**"
        )
        if run > 1:
            human += f" (run {run})"
    elif status == "escalated":
        human = (
            f"**[Stokowski]** Max rework exceeded at **{state}**. "
            f"Escalating for human intervention."
        )
    else:
        human = f"**[Stokowski]** Gate **{state}** status: {status}"

    return f"{machine}\n\n{human}"


def _oldest_first(comments: list[dict]) -> list[dict]:
    """Order comments oldest-first, regardless of how the caller supplied them."""
    return sorted(comments, key=lambda c: c.get("createdAt") or "")


def _tracking_entries(comments: list[dict]):
    """Yield every tracking entry, oldest first, with ``type`` set.

    The one place that parses tracking markers. A comment carrying both a state
    and a gate marker yields the state entry first, then the gate entry.
    """
    for comment in _oldest_first(comments):
        body = comment.get("body", "")
        for kind, pattern in (("state", STATE_PATTERN), ("gate", GATE_PATTERN)):
            match = pattern.search(body)
            if not match:
                continue
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            data["type"] = kind
            yield data


def parse_latest_tracking(comments: list[dict]) -> dict[str, Any] | None:
    """Find the latest state or gate tracking entry.

    Returns a dict with keys:
        - "type": "state" or "gate"
        - Plus all fields from the JSON payload

    Returns None if no tracking comments found.
    """
    # Sort rather than trust the caller (``_tracking_entries`` does). This
    # function decides which state an issue resumes in; reading it off an
    # unsorted list resolves a ticket that reached a gate back to its very
    # first stage, with no error anywhere.
    latest: dict[str, Any] | None = None
    for entry in _tracking_entries(comments):
        latest = entry
    return latest


def get_context_start_timestamp(
    comments: list[dict], state: str, run: int
) -> str | None:
    """Find where the comments relevant to a run of ``state`` begin.

    Normally this is the latest tracking comment. When the run was entered
    from a gate decision (``approved`` or ``rework``), it is the gate's
    ``waiting`` comment instead. A human writes their instructions while the
    issue waits at the gate and only then acts on it, so the orchestrator's
    decision and state-entry comments always come after those instructions.
    Starting the window at the latest tracking comment dropped them.
    """
    latest: str | None = None
    waiting: str | None = None
    entered_from_gate = False

    for entry in _tracking_entries(comments):
        timestamp = entry.get("timestamp")
        is_this_run = (
            entry["type"] == "state"
            and entry.get("state") == state
            and entry.get("run", 1) == run
        )
        if not is_this_run:
            entered_from_gate = (
                entry["type"] == "gate" and entry.get("status") in ("approved", "rework")
            )
        if entry["type"] == "gate" and entry.get("status") == "waiting":
            waiting = timestamp or waiting
        latest = timestamp or latest

    if entered_from_gate and waiting:
        return waiting
    return latest


def get_last_tracking_timestamp(comments: list[dict]) -> str | None:
    """Find the timestamp of the latest tracking comment."""
    latest_ts: str | None = None
    for entry in _tracking_entries(comments):
        latest_ts = entry.get("timestamp") or latest_ts
    return latest_ts


def get_comments_since(
    comments: list[dict], since_timestamp: str | None
) -> list[dict]:
    """Filter comments to only those after a given timestamp.

    Returns comments that are NOT stokowski tracking comments and
    were created after the given timestamp.
    """
    result = []
    since_dt = None
    if since_timestamp:
        try:
            since_dt = datetime.fromisoformat(
                since_timestamp.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            pass

    for comment in comments:
        body = comment.get("body", "")
        if "<!-- stokowski:" in body:
            continue

        if since_dt:
            created = comment.get("createdAt", "")
            if created:
                try:
                    created_dt = datetime.fromisoformat(
                        created.replace("Z", "+00:00")
                    )
                    if created_dt <= since_dt:
                        continue
                except (ValueError, AttributeError):
                    pass

        result.append(comment)

    return result
