import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from stokowski.config import HooksConfig
from stokowski.models import Issue, RunAttempt
from stokowski.runner import build_codex_args, run_codex_turn


class CodexArgumentTests(unittest.TestCase):
    def test_builds_json_automation_command(self):
        args = build_codex_args(
            model="example-model",
            prompt="Investigate the issue",
            workspace_path=Path("/tmp/example-workspace"),
        )

        self.assertEqual(
            args,
            [
                "codex",
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--json",
                "--cd",
                "/tmp/example-workspace",
                "--model",
                "example-model",
                "Investigate the issue",
            ],
        )


class CodexExecutionTests(unittest.IsolatedAsyncioTestCase):
    def attempt(self):
        return RunAttempt(issue_id="issue-1", issue_identifier="SYN-1")

    def issue(self):
        return Issue(id="issue-1", identifier="SYN-1", title="Example")

    @patch("stokowski.runner.build_codex_args")
    async def test_closes_subprocess_stdin(self, build_args):
        script = (
            "import json\n"
            "print(json.dumps({'type': 'turn.completed', 'usage': {}}), flush=True)\n"
        )
        build_args.return_value = [sys.executable, "-c", script]
        create_subprocess_exec = asyncio.create_subprocess_exec
        spawn_kwargs = {}

        async def capture_spawn(*args, **kwargs):
            spawn_kwargs.update(kwargs)
            return await create_subprocess_exec(*args, **kwargs)

        with TemporaryDirectory() as directory:
            with patch(
                "stokowski.runner.asyncio.create_subprocess_exec",
                side_effect=capture_spawn,
            ):
                attempt = await run_codex_turn(
                    model=None,
                    hooks_cfg=HooksConfig(),
                    prompt="Investigate",
                    workspace_path=Path(directory),
                    issue=self.issue(),
                    attempt=self.attempt(),
                    turn_timeout_ms=2_000,
                    stall_timeout_ms=500,
                )

        self.assertEqual(attempt.status, "succeeded")
        self.assertIs(spawn_kwargs["stdin"], asyncio.subprocess.DEVNULL)

    @patch("stokowski.runner.build_codex_args")
    async def test_filters_environment_at_subprocess_boundary(self, build_args):
        create_subprocess_exec = asyncio.create_subprocess_exec
        spawn_kwargs = {}

        async def capture_spawn(*args, **kwargs):
            spawn_kwargs.update(kwargs)
            return await create_subprocess_exec(*args, **kwargs)

        requested_env = {
            "GH_TOKEN": "github-token",
            "PATH": os.environ.get("PATH", ""),
            "SSH_AUTH_SOCK": "/tmp/agent.sock",
            "LINEAR_API_KEY": "declared-secret",
            "STOKOWSKI_PROJECT": "example",
            "STOKOWSKI_ARTIFACTS": "/tmp/artifacts",
            "STOKOWSKI_ISSUE": "SYN-1",
            "STOKOWSKI_STATE": "implement",
            "AMBIENT_API_TOKEN": "must-not-cross-boundary",
            "UNRELATED": "drop",
        }

        with TemporaryDirectory() as directory:
            observed_path = Path(directory) / "observed-env.json"
            script = (
                "import json, os\n"
                f"open({str(observed_path)!r}, 'w').write(json.dumps(dict(os.environ)))\n"
                "print(json.dumps({'type': 'turn.completed', 'usage': {}}), flush=True)\n"
            )
            build_args.return_value = [sys.executable, "-c", script]

            with patch(
                "stokowski.runner.asyncio.create_subprocess_exec",
                side_effect=capture_spawn,
            ):
                attempt = await run_codex_turn(
                    model=None,
                    hooks_cfg=HooksConfig(),
                    prompt="Investigate",
                    workspace_path=Path(directory),
                    issue=self.issue(),
                    attempt=self.attempt(),
                    turn_timeout_ms=2_000,
                    stall_timeout_ms=500,
                    env=requested_env,
                )

            observed_env = json.loads(observed_path.read_text())

        expected_env = {
            key: requested_env[key]
            for key in (
                "GH_TOKEN",
                "PATH",
                "SSH_AUTH_SOCK",
                "LINEAR_API_KEY",
                "STOKOWSKI_PROJECT",
                "STOKOWSKI_ARTIFACTS",
                "STOKOWSKI_ISSUE",
                "STOKOWSKI_STATE",
            )
        }
        self.assertEqual(attempt.status, "succeeded")
        self.assertEqual(spawn_kwargs["env"], expected_env)
        self.assertEqual(
            {key: observed_env.get(key) for key in expected_env}, expected_env
        )
        self.assertNotIn("AMBIENT_API_TOKEN", observed_env)
        self.assertNotIn("UNRELATED", observed_env)

    @patch("stokowski.runner.build_codex_args")
    async def test_parses_and_logs_json_events(self, build_args):
        events = [
            {"type": "thread.started", "thread_id": "thread-1"},
            {"type": "turn.started"},
            {
                "type": "item.started",
                "item": {
                    "id": "item-1",
                    "type": "command_execution",
                    "command": "git status --short",
                    "status": "in_progress",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "item-2",
                    "type": "agent_message",
                    "text": "Investigation complete",
                },
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        ]
        script = (
            "import json\n"
            f"events = {events!r}\n"
            "for event in events:\n"
            "    print(json.dumps(event), flush=True)\n"
        )
        build_args.return_value = [sys.executable, "-c", script]
        forwarded_events = []

        with TemporaryDirectory() as directory:
            with self.assertLogs("stokowski.runner", level="INFO") as logs:
                attempt = await run_codex_turn(
                    model=None,
                    hooks_cfg=HooksConfig(),
                    prompt="Investigate",
                    workspace_path=Path(directory),
                    issue=self.issue(),
                    attempt=self.attempt(),
                    on_event=lambda *event: forwarded_events.append(event),
                    turn_timeout_ms=2_000,
                    stall_timeout_ms=500,
                )

        self.assertEqual(attempt.status, "succeeded")
        self.assertEqual(attempt.session_id, "thread-1")
        self.assertTrue(attempt.session_started)
        self.assertEqual(attempt.result_text, "Investigation complete")
        self.assertEqual(attempt.last_event, "turn.completed")
        self.assertEqual(
            attempt.last_message,
            "agent message completed: Investigation complete",
        )
        self.assertEqual(attempt.input_tokens, 10)
        self.assertEqual(attempt.output_tokens, 2)
        self.assertEqual(attempt.total_tokens, 12)
        self.assertEqual(len(forwarded_events), len(events))

        output = "\n".join(logs.output)
        self.assertIn("Codex started issue=SYN-1", output)
        self.assertIn("command started: git status --short", output)
        self.assertIn("turn completed tokens=12", output)

    @patch("stokowski.runner.build_codex_args")
    async def test_stderr_activity_is_drained_and_logged(self, build_args):
        script = (
            "import json, sys, time\n"
            "for _ in range(5):\n"
            "    print('waiting for service', file=sys.stderr, flush=True)\n"
            "    time.sleep(0.04)\n"
            "print(json.dumps({'type': 'turn.completed', 'usage': {}}), flush=True)\n"
        )
        build_args.return_value = [sys.executable, "-c", script]

        with TemporaryDirectory() as directory:
            with self.assertLogs("stokowski.runner", level="INFO") as logs:
                attempt = await run_codex_turn(
                    model=None,
                    hooks_cfg=HooksConfig(),
                    prompt="Investigate",
                    workspace_path=Path(directory),
                    issue=self.issue(),
                    attempt=self.attempt(),
                    turn_timeout_ms=2_000,
                    stall_timeout_ms=100,
                )

        self.assertEqual(attempt.status, "succeeded")
        self.assertIn("stderr: waiting for service", "\n".join(logs.output))

    @unittest.skipIf(os.name == "nt", "process groups are POSIX-specific")
    @patch("stokowski.runner.build_codex_args")
    async def test_stall_kills_the_entire_process_group(self, build_args):
        pid_events = []

        with TemporaryDirectory() as directory:
            child_pid_path = Path(directory) / "child.pid"
            script = (
                "import subprocess, sys, time\n"
                "child = subprocess.Popen("
                "[sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                f"open({str(child_pid_path)!r}, 'w').write(str(child.pid))\n"
                "time.sleep(30)\n"
            )
            build_args.return_value = [sys.executable, "-c", script]

            attempt = await run_codex_turn(
                model=None,
                hooks_cfg=HooksConfig(),
                prompt="Investigate",
                workspace_path=Path(directory),
                issue=self.issue(),
                attempt=self.attempt(),
                on_pid=lambda *event: pid_events.append(event),
                turn_timeout_ms=2_000,
                stall_timeout_ms=100,
            )
            child_pid = int(child_pid_path.read_text())

            for _ in range(20):
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.05)
            else:
                os.kill(child_pid, 9)
                self.fail("Codex descendant remained alive after stall handling")

        self.assertEqual(attempt.status, "stalled")
        self.assertTrue(attempt.error.startswith("No output for"))
        self.assertEqual(len(pid_events), 2)
        self.assertTrue(pid_events[0][1])
        self.assertFalse(pid_events[1][1])


if __name__ == "__main__":
    unittest.main()
