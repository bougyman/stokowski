import asyncio
import json
import stat
from pathlib import Path
from unittest.mock import AsyncMock, patch

from stokowski import continuity
from stokowski.models import Issue, RunAttempt
from stokowski.orchestrator import Orchestrator


def attempt(
    runner: str,
    session_id: str,
    *,
    state: str,
    result: str,
) -> RunAttempt:
    return RunAttempt(
        issue_id="issue-1",
        issue_identifier="EXT-1",
        runner_type=runner,
        state_name=state,
        status="succeeded",
        session_id=session_id,
        session_started=True,
        process_started=True,
        result_text=result,
    )


def test_same_runner_resumes_across_restart_without_replaying_own_handoff(tmp_path):
    continuity.complete(
        tmp_path,
        attempt("codex", "codex-thread-1", state="investigate", result="Plan ready"),
        {"headline": "Root cause found", "verdict": "complete"},
    )

    resumed = continuity.prepare(tmp_path, "codex", "inherit")

    assert resumed.session_id == "codex-thread-1"
    assert resumed.handoff == ""
    mode = stat.S_IMODE((tmp_path / continuity.CONTINUITY_PATH).stat().st_mode)
    assert mode == 0o600


def test_startup_event_can_persist_session_before_the_turn_completes(tmp_path):
    started = continuity.prepare(tmp_path, "codex", "inherit")
    continuity.save_session(
        tmp_path,
        "codex",
        "in-flight-thread",
        "gpt-5.6-sol",
        started.seen_sequence,
    )

    after_restart = continuity.prepare(tmp_path, "codex", "inherit")
    assert after_restart.session_id == "in-flight-thread"


def test_runner_round_trip_resumes_native_session_and_injects_only_new_work(tmp_path):
    continuity.complete(
        tmp_path,
        attempt("codex", "codex-thread-1", state="investigate", result="Plan ready"),
        {"headline": "Root cause found", "verdict": "complete"},
    )

    claude_start = continuity.prepare(tmp_path, "claude", "inherit")
    assert claude_start.session_id is None
    assert "Root cause found" in claude_start.handoff

    continuity.complete(
        tmp_path,
        attempt("claude", "claude-session-1", state="implement", result="Patch ready"),
        {"headline": "Implemented the patch", "verdict": "complete"},
    )

    codex_return = continuity.prepare(tmp_path, "codex", "inherit")
    assert codex_return.session_id == "codex-thread-1"
    assert "Implemented the patch" in codex_return.handoff
    assert "Root cause found" not in codex_return.handoff


def test_handoff_starts_a_new_native_session_but_keeps_portable_context(tmp_path):
    continuity.complete(
        tmp_path,
        attempt("claude", "claude-session-1", state="investigate", result="Found it"),
        {"headline": "Investigation complete"},
    )

    started = continuity.prepare(tmp_path, "claude", "handoff")

    assert started.session_id is None
    assert "Investigation complete" in started.handoff
    assert continuity.prepare(tmp_path, "claude", "inherit").session_id is None


def test_fresh_clears_native_session_and_omits_all_handoffs(tmp_path):
    continuity.complete(
        tmp_path,
        attempt("claude", "claude-session-1", state="investigate", result="Found it"),
        {"headline": "Investigation complete"},
    )

    started = continuity.prepare(tmp_path, "claude", "fresh")

    assert started.session_id is None
    assert started.handoff == ""
    assert continuity.prepare(tmp_path, "claude", "inherit").session_id is None


def test_failed_resume_is_discarded_so_retry_falls_back_to_handoff(tmp_path):
    continuity.complete(
        tmp_path,
        attempt("codex", "dead-thread", state="investigate", result="Prior result"),
        {"headline": "Prior outcome"},
    )
    started = continuity.prepare(tmp_path, "codex", "inherit")

    failed = RunAttempt(
        issue_id="issue-1",
        issue_identifier="EXT-1",
        runner_type="codex",
        state_name="implement",
        status="failed",
        error="thread not found",
        session_id=started.session_id,
        resumed_session_id=started.session_id,
        process_started=True,
        session_started=False,
    )
    continuity.complete(tmp_path, failed, None)

    retry = continuity.prepare(tmp_path, "codex", "inherit")
    assert retry.session_id is None
    assert "thread not found" in retry.handoff


def test_handoff_excludes_activity_thinking_and_unknown_report_fields(tmp_path):
    completed = attempt(
        "codex", "codex-thread-1", state="investigate", result="Public result"
    )
    completed.last_message = "private chain of thought"
    continuity.complete(
        tmp_path,
        completed,
        {"headline": "Public headline", "secret_thinking": "do not persist"},
    )

    raw = (tmp_path / continuity.CONTINUITY_PATH).read_text()
    assert "Public headline" in raw
    assert "private chain of thought" not in raw
    assert "secret_thinking" not in raw
    assert "do not persist" not in raw


def test_only_the_latest_handoffs_are_retained(tmp_path):
    for number in range(continuity.MAX_HANDOFFS + 3):
        continuity.complete(
            tmp_path,
            attempt(
                "codex",
                f"thread-{number}",
                state=f"state-{number}",
                result=f"result-{number}",
            ),
            None,
        )

    data = json.loads((tmp_path / continuity.CONTINUITY_PATH).read_text())
    assert len(data["handoffs"]) == continuity.MAX_HANDOFFS
    assert data["handoffs"][0]["state"] == "state-3"
    assert data["handoffs"][-1]["state"] == f"state-{continuity.MAX_HANDOFFS + 2}"


def test_large_reports_are_stored_as_bounded_valid_json(tmp_path):
    huge = "x" * 20_000
    continuity.complete(
        tmp_path,
        attempt("claude", "session-1", state="review", result="done"),
        {
            "headline": huge,
            "summary": huge,
            "key_points": [huge] * 20,
            "open_questions": [huge] * 20,
        },
    )

    raw = (tmp_path / continuity.CONTINUITY_PATH).read_text()
    data = json.loads(raw)
    stored_report = data["handoffs"][0]["report"]
    assert stored_report["truncated"] is True
    assert len(json.dumps(stored_report)) <= continuity.MAX_REPORT_CHARS


def test_orchestrator_resumes_session_persisted_by_startup_event_after_restart(
    tmp_path,
):
    prompt_path = tmp_path / "work.md"
    prompt_path.write_text("Do the work.")
    workflow_path = tmp_path / "workflow.yaml"
    workspace_root = tmp_path / "workspaces"
    workflow_path.write_text(
        f"""
tracker:
  project_slug: project-1
  api_key: lin_api_test
workspace:
  root: {workspace_root}
states:
  work:
    type: agent
    prompt: work.md
    runner: codex
    session: inherit
    transitions: {{complete: done}}
  done:
    type: terminal
    linear_state: terminal
"""
    )
    issue = Issue(
        id="issue-1",
        identifier="EXT-1",
        title="Durable resume",
        state="In Progress",
    )
    observed_sessions: list[str | None] = []

    async def fake_turn(**kwargs):
        current = kwargs["attempt"]
        observed_sessions.append(current.session_id)
        current.process_started = True
        current.session_id = "codex-thread-1"
        current.session_started = True
        kwargs["on_event"](issue.identifier, "thread.started", {})
        current.status = "succeeded"
        current.result_text = "Finished"
        return current

    async def run_once() -> None:
        orchestrator = Orchestrator(workflow_path)
        assert orchestrator._load_workflow() == []
        current = RunAttempt(
            issue_id=issue.id,
            issue_identifier=issue.identifier,
            attempt=1,
            state_name="work",
        )
        with (
            patch(
                "stokowski.orchestrator.run_turn",
                new=AsyncMock(side_effect=fake_turn),
            ),
            patch.object(
                orchestrator,
                "_render_prompt_async",
                new=AsyncMock(return_value="Prompt"),
            ),
            patch.object(
                orchestrator,
                "_publish_run_report",
                new=AsyncMock(),
            ),
            patch.object(orchestrator, "_on_worker_exit"),
        ):
            await orchestrator._run_worker(issue, current)

    asyncio.run(run_once())
    asyncio.run(run_once())

    assert observed_sessions == [None, "codex-thread-1"]
