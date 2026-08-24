import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from stokowski.config import (
    ClaudeConfig,
    HooksConfig,
    ProjectConfig,
    ServiceConfig,
    StateConfig,
    TrackerConfig,
    merge_state_config,
    parse_workflow_file,
    validate_config,
)
from stokowski.models import Issue, RunAttempt
from stokowski.runner import build_codex_args, run_codex_turn, run_turn


class CodexArgumentTests(unittest.TestCase):
    def test_builds_current_noninteractive_command(self):
        args = build_codex_args(
            model=None,
            prompt="Investigate the issue",
            workspace_path=Path("/tmp/example-workspace"),
        )

        self.assertEqual(
            args,
            [
                "codex",
                "--ask-for-approval",
                "never",
                "--sandbox",
                "workspace-write",
                "--cd",
                "/tmp/example-workspace",
                "exec",
                "Investigate the issue",
            ],
        )

    def test_adds_model_and_reasoning_overrides(self):
        args = build_codex_args(
            model="example-codex-model",
            prompt="Review the diff",
            workspace_path=Path("/tmp/example-workspace"),
            reasoning_effort="high",
        )

        self.assertEqual(
            args,
            [
                "codex",
                "--model",
                "example-codex-model",
                "--config",
                'model_reasoning_effort="high"',
                "--ask-for-approval",
                "never",
                "--sandbox",
                "workspace-write",
                "--cd",
                "/tmp/example-workspace",
                "exec",
                "Review the diff",
            ],
        )

    def test_rejects_unsupported_reasoning_effort(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Codex reasoning effort"):
            build_codex_args(
                model=None,
                prompt="Review the diff",
                workspace_path=Path("/tmp/example-workspace"),
                reasoning_effort="extreme",
            )


class CodexExecutionTests(unittest.IsolatedAsyncioTestCase):
    @patch("stokowski.runner.build_codex_args")
    async def test_stderr_progress_prevents_false_stall(self, build_args):
        script = (
            "import sys, time\n"
            "for _ in range(6):\n"
            "    print('working', file=sys.stderr, flush=True)\n"
            "    time.sleep(0.05)\n"
            "print('finished', flush=True)\n"
        )
        build_args.return_value = [sys.executable, "-c", script]

        with TemporaryDirectory() as directory:
            attempt = await run_codex_turn(
                model=None,
                hooks_cfg=HooksConfig(),
                prompt="Investigate",
                workspace_path=Path(directory),
                issue=Issue(
                    id="issue-1",
                    identifier="SYN-1",
                    title="Example",
                ),
                attempt=RunAttempt(
                    issue_id="issue-1",
                    issue_identifier="SYN-1",
                ),
                turn_timeout_ms=2_000,
                stall_timeout_ms=150,
            )

        self.assertEqual(attempt.status, "succeeded")
        self.assertEqual(attempt.last_message, "finished")


class CodexReasoningConfigTests(unittest.TestCase):
    def test_workflow_parses_per_state_model_and_reasoning(self):
        with TemporaryDirectory() as directory:
            workflow_path = Path(directory) / "workflow.yaml"
            workflow_path.write_text(
                """
tracker:
  project_slug: abc123
  api_key: lin_api_test
states:
  investigate:
    type: agent
    prompt: prompts/investigate.md
    runner: codex
    model: example-codex-model
    reasoning_effort: HIGH
    transitions: {complete: done}
  done:
    type: terminal
    linear_state: terminal
"""
            )

            workflow = parse_workflow_file(workflow_path)

        state = workflow.config.states["investigate"]
        self.assertEqual(state.model, "example-codex-model")
        self.assertEqual(state.reasoning_effort, "high")
        self.assertEqual(validate_config(workflow.config), [])

    def test_validation_rejects_unknown_reasoning_effort(self):
        errors = validate_config(
            self.service_config(
                StateConfig(
                    name="work",
                    prompt="prompts/work.md",
                    runner="codex",
                    reasoning_effort="extreme",
                    transitions={"complete": "done"},
                )
            )
        )

        self.assertIn(
            "project 'example' state 'work': unsupported reasoning_effort: "
            "'extreme'",
            errors,
        )

    def test_validation_rejects_reasoning_effort_for_claude(self):
        errors = validate_config(
            self.service_config(
                StateConfig(
                    name="work",
                    prompt="prompts/work.md",
                    runner="claude",
                    reasoning_effort="high",
                    transitions={"complete": "done"},
                )
            )
        )

        self.assertIn(
            "project 'example' state 'work': reasoning_effort requires runner: codex",
            errors,
        )

    def test_codex_state_does_not_inherit_root_claude_model(self):
        state = StateConfig(name="work", runner="codex")

        resolved, _hooks = merge_state_config(
            state,
            ClaudeConfig(model="claude-model"),
            HooksConfig(),
        )

        self.assertIsNone(resolved.model)

    @staticmethod
    def service_config(state: StateConfig) -> ServiceConfig:
        project = ProjectConfig(
            name="example",
            tracker=TrackerConfig(api_key="lin_api_test", project_slug="abc123"),
            states={
                "work": state,
                "done": StateConfig(
                    name="done",
                    type="terminal",
                    linear_state="terminal",
                ),
            },
        )
        return ServiceConfig(projects=[project])


class CodexDispatchTests(unittest.IsolatedAsyncioTestCase):
    @patch("stokowski.runner.run_codex_turn", new_callable=AsyncMock)
    async def test_dispatch_forwards_reasoning_effort(self, run_codex_turn):
        attempt = RunAttempt(issue_id="issue-1", issue_identifier="SYN-1")
        run_codex_turn.return_value = attempt

        result = await run_turn(
            runner_type="codex",
            claude_cfg=ClaudeConfig(model="example-codex-model"),
            hooks_cfg=HooksConfig(),
            prompt="Investigate",
            workspace_path=Path("/tmp/example-workspace"),
            issue=Issue(id="issue-1", identifier="SYN-1", title="Example"),
            attempt=attempt,
            reasoning_effort="high",
        )

        self.assertIs(result, attempt)
        self.assertEqual(
            run_codex_turn.await_args.kwargs["reasoning_effort"],
            "high",
        )
        self.assertEqual(
            run_codex_turn.await_args.kwargs["model"],
            "example-codex-model",
        )


if __name__ == "__main__":
    unittest.main()
