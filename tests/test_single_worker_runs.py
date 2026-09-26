"""One worker per issue, and run numbers that restart after each approval.

EXT-68 stalled because approving a gate started two workers for the same
state: `_transition` scheduled a retry dispatch and the tick's dispatch loop
started another. The retry replaced the first worker in `self.running`, so the
first worker's successful completion was discarded as superseded, and the
issue never reached its next gate.

The same ticket also carried the rework count of `research-review` (run 2)
into every later stage, although none of those stages had been sent back.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stokowski.models import Issue, RetryEntry, RunAttempt
from stokowski.orchestrator import Orchestrator
from stokowski.tracking import make_gate_comment

REPO = Path(__file__).resolve().parent.parent
ISSUE_ID = "issue-1"


class FakeLinear:
    """The subset of LinearClient the gate and retry paths call."""

    def __init__(self, by_state=None, comments=None, candidates=None):
        self.by_state = by_state or {}
        self.comments = comments or []
        self.candidates = candidates or []
        self.posted: list[str] = []
        self.moves: list[str] = []

    async def fetch_issues_by_states(self, _slug, states, assignee=None):
        return [i for s in states for i in self.by_state.get(s, [])]

    async def fetch_comments(self, _issue_id):
        return self.comments

    async def fetch_candidate_issues(self, _slug, _states, assignee=None):
        return self.candidates

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
    result._schedule_retry = MagicMock()
    return result


def make_issue(state="In Progress"):
    return Issue(id=ISSUE_ID, identifier="ENG-1", title="Example", state=state)


def make_attempt(state_name="implement", state_run=1):
    return RunAttempt(
        issue_id=ISSUE_ID,
        issue_identifier="ENG-1",
        state_name=state_name,
        state_run=state_run,
    )


# ── One worker per issue ─────────────────────────────────────────────────────


def test_dispatch_refuses_a_second_worker(orchestrator):
    running = make_attempt()
    orchestrator.running[ISSUE_ID] = running
    orchestrator._issue_current_state[ISSUE_ID] = "implement"

    # No event loop: a dispatch that got past the guard would fail on create_task.
    orchestrator._dispatch(make_issue())

    assert orchestrator.running[ISSUE_ID] is running
    assert ISSUE_ID not in orchestrator._tasks


def test_retry_does_not_replace_a_running_worker(orchestrator):
    running = make_attempt()
    orchestrator.running[ISSUE_ID] = running
    orchestrator._slot_held.add(ISSUE_ID)
    orchestrator.claimed.add(ISSUE_ID)
    orchestrator._linear = FakeLinear(candidates=[make_issue()])
    orchestrator.retry_attempts[ISSUE_ID] = RetryEntry(
        issue_id=ISSUE_ID, identifier="ENG-1", attempt=0, due_at_ms=0,
    )
    orchestrator._dispatch = MagicMock()

    asyncio.run(orchestrator._handle_retry(ISSUE_ID))

    orchestrator._dispatch.assert_not_called()
    assert orchestrator.running[ISSUE_ID] is running
    # The claim and slot still belong to the running worker.
    assert ISSUE_ID in orchestrator.claimed
    assert ISSUE_ID in orchestrator._slot_held


def test_approved_gate_leaves_one_dispatch_path(orchestrator):
    issue = make_issue(state=orchestrator.cfg.linear_states.gate_approved)
    orchestrator._linear = FakeLinear(
        by_state={orchestrator.cfg.linear_states.gate_approved: [issue]},
    )
    orchestrator._pending_gates[ISSUE_ID] = "research-review"
    orchestrator._issue_state_runs[ISSUE_ID] = 2

    asyncio.run(orchestrator._handle_gate_responses())

    orchestrator._schedule_retry.assert_called_once()
    # The retry is the dispatch; the tick's dispatch loop must skip the issue.
    assert not orchestrator._is_eligible(make_issue())


# ── Run numbers restart after approval ───────────────────────────────────────


def test_approval_starts_the_next_stage_at_run_one(orchestrator):
    issue = make_issue(state=orchestrator.cfg.linear_states.gate_approved)
    linear = FakeLinear(by_state={orchestrator.cfg.linear_states.gate_approved: [issue]})
    orchestrator._linear = linear
    orchestrator._pending_gates[ISSUE_ID] = "research-review"
    orchestrator._issue_state_runs[ISSUE_ID] = 2

    asyncio.run(orchestrator._handle_gate_responses())

    assert orchestrator._issue_current_state[ISSUE_ID] == "implement"
    assert orchestrator._issue_state_runs[ISSUE_ID] == 1
    gate_comment, state_comment = linear.posted
    # The approval is recorded against the run a human judged...
    assert '"status": "approved", "run": 2' in gate_comment
    # ...and the next stage starts its own count.
    assert '"state": "implement", "run": 1' in state_comment


def test_restart_after_approval_resumes_the_next_stage_at_run_one(orchestrator):
    orchestrator._linear = FakeLinear(comments=[{
        "id": "c1",
        "createdAt": "2026-09-26T02:21:06Z",
        "body": make_gate_comment(state="research-review", status="approved", run=2),
    }])

    state, run = asyncio.run(orchestrator._resolve_current_state(make_issue()))

    assert (state, run) == ("implement", 1)


def test_rework_after_a_reset_announces_the_revisited_run(orchestrator):
    # implement run 2 was announced during implementation-review's rework
    # cycle. After approval the count restarted, and merge-review now sends
    # the issue back to implement at run 2 again. That entry must be announced.
    rework_state = orchestrator.cfg.linear_states.rework
    linear = FakeLinear(by_state={rework_state: [make_issue(state=rework_state)]})
    orchestrator._linear = linear
    orchestrator._announced_states.add((ISSUE_ID, "implement", 2))
    orchestrator._pending_gates[ISSUE_ID] = "merge-review"
    orchestrator._issue_current_state[ISSUE_ID] = "merge-review"
    orchestrator._issue_state_runs[ISSUE_ID] = 1

    async def rework_then_announce():
        await orchestrator._handle_gate_responses()
        await orchestrator._announce_state(make_issue(), "implement", 2)

    asyncio.run(rework_then_announce())

    assert orchestrator._issue_current_state[ISSUE_ID] == "implement"
    assert orchestrator._issue_state_runs[ISSUE_ID] == 2
    assert any('"state": "implement", "run": 2' in body for body in linear.posted)
