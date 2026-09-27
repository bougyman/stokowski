"""Blocked runs, proposed follow-ups, and what a prompt can see.

EXT-68 (rubyists/linear-cli) surfaced four gaps at once:

- `merge` reported "blocked" because its PR had no approval, and the issue was
  moved to Done anyway: a successful run always followed "complete".
- The follow-ups a human approved at the glean gate were never created. The
  approval comment was written while the issue waited at the gate, and the
  prompt's comment window started after the orchestrator's own "approved" and
  "Entering state" comments, so the next stage never saw it.
- The stage that creates follow-ups had to re-derive them from prose.
- Prompts written as `{{ issue.identifier }}` rendered it as an empty string.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from stokowski import report as report_mod
from stokowski.models import Issue, RunAttempt
from stokowski.orchestrator import Orchestrator
from stokowski.prompt import assemble_prompt, build_template_context, render_template
from stokowski.tracking import (
    get_comments_since,
    get_context_start_timestamp,
    make_gate_comment,
    make_state_comment,
)

REPO = Path(__file__).resolve().parent.parent
ISSUE_ID = "issue-1"


class FakeLinear:
    def __init__(self):
        self.posted: list[str] = []
        self.moves: list[str] = []

    async def post_comment(self, _issue_id, body):
        self.posted.append(body)
        return True

    async def update_issue_state(self, _issue_id, state):
        self.moves.append(state)
        return True


@pytest.fixture
def orchestrator(tmp_path):
    shutil.copy(REPO / "workflow.example.yaml", tmp_path / "workflow.yaml")
    shutil.copytree(REPO / "workflows", tmp_path / "workflows")
    shutil.copytree(REPO / "prompts", tmp_path / "prompts")

    result = Orchestrator(tmp_path / "workflow.yaml")
    result._load_workflow()
    return result


def make_issue():
    return Issue(id=ISSUE_ID, identifier="ENG-1", title="Example", state="In Progress")


def make_attempt(state_name="merge", verdict=None):
    return RunAttempt(
        issue_id=ISSUE_ID,
        issue_identifier="ENG-1",
        state_name=state_name,
        state_run=1,
        status="succeeded",
        report_verdict=verdict,
    )


def declare_blocked(orchestrator, state="merge", target="merge-review"):
    orchestrator._states_for(make_issue())[state].transitions["blocked"] = target


# ── A blocked run follows its state's "blocked" transition ───────────────────


def test_blocked_report_follows_the_declared_blocked_transition(orchestrator):
    declare_blocked(orchestrator)
    attempt = make_attempt(verdict="blocked")

    assert orchestrator._exit_transition(make_issue(), attempt) == "blocked"


def test_blocked_report_without_a_declared_transition_still_completes(orchestrator):
    attempt = make_attempt(verdict="blocked")

    assert orchestrator._exit_transition(make_issue(), attempt) == "complete"


def test_any_other_verdict_completes(orchestrator):
    declare_blocked(orchestrator)

    for verdict in ("complete", "approve", None):
        attempt = make_attempt(verdict=verdict)
        assert orchestrator._exit_transition(make_issue(), attempt) == "complete"


def test_worker_exit_takes_the_blocked_transition(orchestrator):
    declare_blocked(orchestrator)
    issue = make_issue()
    attempt = make_attempt(verdict="blocked")
    orchestrator.running[ISSUE_ID] = attempt
    orchestrator._issue_current_state[ISSUE_ID] = "merge"
    orchestrator._issue_state_runs[ISSUE_ID] = 1
    orchestrator._safe_transition = AsyncMock()

    async def exit_and_settle():
        orchestrator._on_worker_exit(issue, attempt)
        await asyncio.sleep(0)

    asyncio.run(exit_and_settle())

    orchestrator._safe_transition.assert_awaited_once_with(issue, "blocked", attempt)


def test_blocked_transition_parks_the_issue_at_a_gate_not_done(orchestrator):
    declare_blocked(orchestrator)
    linear = FakeLinear()
    orchestrator._linear = linear
    orchestrator._issue_current_state[ISSUE_ID] = "merge"

    asyncio.run(orchestrator._transition(make_issue(), "blocked"))

    assert orchestrator._pending_gates[ISSUE_ID] == "merge-review"
    assert '"state": "merge-review", "status": "waiting"' in linear.posted[0]
    assert linear.moves == [orchestrator.cfg.linear_states.review]
    assert ISSUE_ID not in orchestrator.completed


# ── Follow-ups: rendered for the human, saved for the stage that creates them ─


def follow_up(fid, title="Fix it", description="Body\n\n- criterion"):
    return {"id": fid, "title": title, "description": description}


def test_incomplete_and_duplicate_follow_ups_are_dropped():
    report = {"follow_ups": [
        follow_up("G1"),
        {"id": "G2", "title": "No description"},
        follow_up("G1", title="Duplicate id"),
        "not an object",
    ]}

    follow_ups, dropped = report_mod.follow_ups_of(report)

    assert [f["id"] for f in follow_ups] == ["G1"]
    assert dropped == 3


def test_follow_ups_are_rendered_by_id_with_their_full_description():
    report = {"headline": "done", "follow_ups": [
        {**follow_up("G1", "Fail safe on EOF"), "priority": "High", "labels": ["bug"]},
        follow_up("G2", "Keep JSON stdout valid", "Line one\nLine two"),
    ]}

    body = report_mod.render(report, state="glean", run=1)

    assert "### Proposed follow-ups" in body
    assert "#### G1 — Fail safe on EOF" in body
    assert "priority `high` · labels `bug`" in body
    assert "#### G2 — Keep JSON stdout valid" in body
    assert "> Line one\n> Line two" in body


def test_follow_ups_are_saved_for_a_later_stage(tmp_path):
    report = {"follow_ups": [follow_up("G1"), {"id": "G2"}]}

    report_mod.save_follow_ups(tmp_path, report, state="glean", run=2)

    saved = json.loads((tmp_path / report_mod.FOLLOW_UPS_PATH).read_text())
    assert saved == {"state": "glean", "run": 2, "follow_ups": [follow_up("G1")]}


def test_a_report_without_follow_ups_keeps_the_saved_ones(tmp_path):
    report_mod.save_follow_ups(tmp_path, {"follow_ups": [follow_up("G1")]}, state="glean", run=1)

    report_mod.save_follow_ups(tmp_path, {"headline": "improvement"}, state="improvement", run=1)

    saved = json.loads((tmp_path / report_mod.FOLLOW_UPS_PATH).read_text())
    assert [f["id"] for f in saved["follow_ups"]] == ["G1"]


def test_an_empty_follow_up_list_replaces_the_saved_ones(tmp_path):
    report_mod.save_follow_ups(tmp_path, {"follow_ups": [follow_up("G1")]}, state="glean", run=1)

    report_mod.save_follow_ups(tmp_path, {"follow_ups": []}, state="glean", run=2)

    saved = json.loads((tmp_path / report_mod.FOLLOW_UPS_PATH).read_text())
    assert saved["follow_ups"] == []


# ── The prompt sees what a human wrote at the gate ───────────────────────────


def _c(body, created):
    return {"id": created, "body": body, "createdAt": created, "user": {"name": "Reporter"}}


def ext_68_history(next_state="improvement"):
    """glean → human approves follow-ups at the gate → the next stage starts."""
    return [
        _c(make_state_comment(state="glean", run=1), "2026-09-27T01:53:20Z"),
        _c(make_gate_comment(state="glean-review", status="waiting", run=1),
           "2026-09-27T01:59:43Z"),
        _c("Approve follow-ups: G1, G3", "2026-09-27T12:50:00Z"),
        _c(make_gate_comment(state="glean-review", status="approved", run=1),
           "2026-09-27T12:58:44Z"),
        _c(make_state_comment(state=next_state, run=1), "2026-09-27T12:58:45Z"),
    ]


def _with_embedded_times(comments):
    # make_* stamp the current time into the tracking JSON; pin it to createdAt
    # so the history reads in the order written above.
    for comment in comments:
        body = comment["body"]
        if "<!-- stokowski:" in body:
            head, rest = body.split(" ", 2)[:2], body.split(" ", 2)[2]
            payload, tail = rest.split(" -->", 1)
            data = json.loads(payload)
            data["timestamp"] = comment["createdAt"]
            comment["body"] = f"{head[0]} {head[1]} {json.dumps(data)} -->{tail}"
    return comments


def test_the_window_starts_at_the_gate_not_at_the_approval():
    comments = _with_embedded_times(ext_68_history())

    start = get_context_start_timestamp(comments, "improvement", 1)
    recent = get_comments_since(comments, start)

    assert start == "2026-09-27T01:59:43Z"
    assert [c["body"] for c in recent] == ["Approve follow-ups: G1, G3"]


def test_the_next_stage_prompt_includes_the_approval(orchestrator):
    # The example workflow has no improvement state; merge follows a gate too.
    comments = _with_embedded_times(ext_68_history(next_state="merge"))
    issue = make_issue()

    prompt = assemble_prompt(
        cfg=orchestrator.cfg,
        workflow_dir=str(orchestrator.workflow_path.parent),
        issue=issue,
        state_name="merge",
        state_cfg=orchestrator._states_for(issue)["merge"],
        run=1,
        comments=comments,
    )

    assert "Approve follow-ups: G1, G3" in prompt


def test_templates_can_use_the_issue_object():
    issue = Issue(id="i", identifier="EXT-68", title="Filter unassign", state="Todo",
                  url="https://linear.app/x/EXT-68")
    context = build_template_context(issue=issue, state_name="improvement")

    rendered = render_template(
        "{{ issue.identifier }}: {{ issue.title }} {{ issue.url }} / {{ issue_identifier }}",
        context,
    )

    assert rendered == "EXT-68: Filter unassign https://linear.app/x/EXT-68 / EXT-68"
