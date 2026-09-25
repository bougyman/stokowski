"""Regression tests for stale state-machine worker completions."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from stokowski.models import Issue, RunAttempt
from stokowski.orchestrator import Orchestrator

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def orchestrator(tmp_path):
    shutil.copy(REPO / "workflow.example.yaml", tmp_path / "workflow.yaml")
    shutil.copytree(REPO / "workflows", tmp_path / "workflows")
    shutil.copytree(REPO / "prompts", tmp_path / "prompts")

    result = Orchestrator(tmp_path / "workflow.yaml")
    result._load_workflow()
    return result


def make_issue():
    return Issue(id="issue-1", identifier="ENG-1", title="Example", state="In Progress")


def make_attempt(state_name, state_run=2):
    return RunAttempt(
        issue_id="issue-1",
        issue_identifier="ENG-1",
        state_name=state_name,
        state_run=state_run,
        status="succeeded",
    )


def test_stale_completion_does_not_advance_the_newer_state(orchestrator):
    current = make_attempt("glean")
    stale = make_attempt("code-review")
    orchestrator.running[current.issue_id] = current
    orchestrator._issue_current_state[current.issue_id] = "glean"
    orchestrator._issue_state_runs[current.issue_id] = 2
    orchestrator._safe_transition = AsyncMock()

    orchestrator._on_worker_exit(make_issue(), stale)

    assert orchestrator.running[stale.issue_id] is current
    orchestrator._safe_transition.assert_not_awaited()


def test_duplicate_completions_advance_a_state_once(orchestrator):
    issue = make_issue()
    first = make_attempt("code-review")
    second = make_attempt("code-review")
    orchestrator._issue_current_state[issue.id] = "code-review"
    orchestrator._issue_state_runs[issue.id] = 2

    async def complete_both():
        started = asyncio.Event()
        release = asyncio.Event()
        transitions = []

        async def transition(*args):
            transitions.append(args)
            started.set()
            await release.wait()

        orchestrator._transition = transition
        first_task = asyncio.create_task(
            orchestrator._safe_transition(issue, "complete", first)
        )
        await started.wait()
        await orchestrator._safe_transition(issue, "complete", second)
        release.set()
        await first_task
        return transitions

    transitions = asyncio.run(complete_both())

    assert transitions == [(issue, "complete")]
