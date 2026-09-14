import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from stokowski.config import (
    ClaudeConfig,
    HooksConfig,
    ProjectConfig,
    RoutingConfig,
    ServiceConfig,
    StateConfig,
    TrackerConfig,
    WorkflowSpec,
    merge_state_config,
    parse_workflow_file,
    validate_config,
)
from stokowski.models import Issue, RunAttempt
from stokowski.runner import (
    build_claude_args,
    build_codex_args,
    run_codex_turn,
    run_turn,
)


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
                "exec",
                "--sandbox",
                "danger-full-access",
                "--ephemeral",
                "--json",
                "--cd",
                "/tmp/example-workspace",
                "--config",
                'approval_policy="never"',
                "Investigate the issue",
            ],
        )

    def test_adds_model_and_effort_overrides(self):
        args = build_codex_args(
            model="example-codex-model",
            prompt="Review the diff",
            workspace_path=Path("/tmp/example-workspace"),
            effort="max",
        )

        self.assertEqual(
            args,
            [
                "codex",
                "exec",
                "--sandbox",
                "danger-full-access",
                "--ephemeral",
                "--json",
                "--cd",
                "/tmp/example-workspace",
                "--config",
                'approval_policy="never"',
                "--model",
                "example-codex-model",
                "--config",
                'model_reasoning_effort="max"',
                "Review the diff",
            ],
        )

    def test_rejects_unsupported_effort(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Codex effort"):
            build_codex_args(
                model=None,
                prompt="Review the diff",
                workspace_path=Path("/tmp/example-workspace"),
                effort="extreme",
            )


class ClaudeEffortArgumentTests(unittest.TestCase):
    def test_adds_the_same_effort_value_to_claude(self):
        args = build_claude_args(
            ClaudeConfig(effort="max"),
            prompt="Review the diff",
            workspace_path=Path("/tmp/example-workspace"),
        )

        index = args.index("--effort")
        self.assertEqual(args[index : index + 2], ["--effort", "max"])

    def test_rejects_unsupported_effort(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Claude effort"):
            build_claude_args(
                ClaudeConfig(effort="extreme"),
                prompt="Review the diff",
                workspace_path=Path("/tmp/example-workspace"),
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


class UnifiedEffortConfigTests(unittest.TestCase):
    def test_workflow_parses_per_state_model_and_effort(self):
        with TemporaryDirectory() as directory:
            workflow_path = Path(directory) / "workflow.yaml"
            prompt_path = Path(directory) / "prompts" / "investigate.md"
            prompt_path.parent.mkdir()
            prompt_path.write_text("Investigate the issue.")
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
    effort: MAX
    transitions: {complete: done}
  done:
    type: terminal
    linear_state: terminal
"""
            )

            workflow = parse_workflow_file(workflow_path)

            state = workflow.config.states["investigate"]
            self.assertEqual(state.model, "example-codex-model")
            self.assertEqual(state.effort, "max")
            self.assertEqual(validate_config(workflow.config), [])

    def test_removed_reasoning_effort_key_is_rejected(self):
        with TemporaryDirectory() as directory:
            workflow_path = Path(directory) / "workflow.yaml"
            workflow_path.write_text(
                """
tracker:
  project_slug: abc123
  api_key: lin_api_test
states:
  work:
    prompt: work.md
    runner: codex
    reasoning_effort: high
"""
            )

            with self.assertRaisesRegex(
                ValueError,
                "'reasoning_effort' was removed; use 'effort'",
            ):
                parse_workflow_file(workflow_path)

    def test_validation_rejects_unknown_effort_for_codex(self):
        errors = validate_config(
            self.service_config(
                StateConfig(
                    name="work",
                    prompt="test_codex_runner.py",
                    runner="codex",
                    effort="extreme",
                    transitions={"complete": "done"},
                )
            )
        )

        self.assertIn(
            "project 'example' state 'work': unsupported effort: 'extreme' "
            "(valid: low, medium, high, xhigh, max)",
            errors,
        )

    def test_claude_accepts_the_same_effort_field_and_values(self):
        self.assertEqual(
            validate_config(
                self.service_config(
                    StateConfig(
                        name="work",
                        prompt="test_codex_runner.py",
                        runner="claude",
                        effort="max",
                        transitions={"complete": "done"},
                    )
                )
            ),
            [],
        )

    def test_validation_rejects_unknown_root_claude_effort(self):
        config = self.service_config(
            StateConfig(
                name="work",
                prompt="test_codex_runner.py",
                runner="claude",
                transitions={"complete": "done"},
            )
        )
        config.projects[0].claude.effort = "extreme"

        errors = validate_config(config)

        self.assertIn(
            "project 'example': unsupported claude.effort: 'extreme' "
            "(valid: low, medium, high, xhigh, max)",
            errors,
        )

    def test_codex_state_does_not_inherit_root_claude_defaults(self):
        state = StateConfig(name="work", runner="codex")

        resolved, _hooks = merge_state_config(
            state,
            ClaudeConfig(model="claude-model", effort="xhigh"),
            HooksConfig(),
        )

        self.assertIsNone(resolved.model)
        self.assertIsNone(resolved.effort)

    def test_codex_state_resolves_its_own_effort(self):
        resolved, _hooks = merge_state_config(
            StateConfig(name="work", runner="codex", effort="max"),
            ClaudeConfig(effort="low"),
            HooksConfig(),
        )

        self.assertEqual(resolved.effort, "max")

    def test_validation_rejects_unknown_effort_for_claude(self):
        errors = validate_config(
            self.service_config(
                StateConfig(
                    name="work",
                    prompt="test_codex_runner.py",
                    runner="claude",
                    effort="extreme",
                    transitions={"complete": "done"},
                )
            )
        )

        self.assertIn(
            "project 'example' state 'work': unsupported effort: 'extreme' "
            "(valid: low, medium, high, xhigh, max)",
            errors,
        )

    @staticmethod
    def service_config(state: StateConfig) -> ServiceConfig:
        states = {
            "work": state,
            "done": StateConfig(
                name="done",
                type="terminal",
                linear_state="terminal",
            ),
        }
        project = ProjectConfig(
            name="example",
            tracker=TrackerConfig(api_key="lin_api_test", project_slug="abc123"),
            states=states,
            workflows={"default": WorkflowSpec(name="default", states=states)},
            routing=RoutingConfig(default="default"),
            workflow_dir=Path(__file__).parent,
        )
        return ServiceConfig(projects=[project])


class CodexDispatchTests(unittest.IsolatedAsyncioTestCase):
    @patch("stokowski.runner.run_codex_turn", new_callable=AsyncMock)
    async def test_dispatch_forwards_effort(self, run_codex_turn):
        attempt = RunAttempt(issue_id="issue-1", issue_identifier="SYN-1")
        run_codex_turn.return_value = attempt

        result = await run_turn(
            runner_type="codex",
            claude_cfg=ClaudeConfig(model="example-codex-model", effort="max"),
            hooks_cfg=HooksConfig(),
            prompt="Investigate",
            workspace_path=Path("/tmp/example-workspace"),
            issue=Issue(id="issue-1", identifier="SYN-1", title="Example"),
            attempt=attempt,
        )

        self.assertIs(result, attempt)
        self.assertEqual(
            run_codex_turn.await_args.kwargs["effort"],
            "max",
        )
        self.assertEqual(
            run_codex_turn.await_args.kwargs["model"],
            "example-codex-model",
        )


if __name__ == "__main__":
    unittest.main()
