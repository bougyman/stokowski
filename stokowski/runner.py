"""Agent runner - launches Claude Code or Codex in headless mode."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import (
    CODEX_REASONING_EFFORTS,
    ClaudeConfig,
    HooksConfig,
    build_agent_env,
)
from .models import Issue, RunAttempt

logger = logging.getLogger("stokowski.runner")

# Callback type for events from the runner to the orchestrator
EventCallback = Callable[[str, str, dict[str, Any]], None]
# Callback for registering/unregistering child PIDs
PidCallback = Callable[[int, bool], None]  # (pid, is_register)


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill a subprocess and every descendant in its dedicated process group."""
    if proc.returncode is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except (AttributeError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _compact(value: Any, limit: int = 200) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _codex_item_summary(event_type: str, item: dict[str, Any]) -> str:
    phase = event_type.removeprefix("item.")
    item_type = str(item.get("type", "item"))

    if item_type == "command_execution":
        command = _compact(item.get("command"))
        return f"command {phase}: {command}" if command else f"command {phase}"

    if item_type == "mcp_tool_call":
        server = _compact(item.get("server"), 80)
        tool = _compact(item.get("tool"), 80)
        name = ".".join(part for part in (server, tool) if part)
        return f"MCP {phase}: {name}" if name else f"MCP call {phase}"

    if item_type == "web_search":
        query = _compact(item.get("query"))
        return f"web search {phase}: {query}" if query else f"web search {phase}"

    if item_type == "file_change":
        changes = item.get("changes", [])
        paths = [
            _compact(change.get("path"), 120)
            for change in changes
            if isinstance(change, dict) and change.get("path")
        ]
        detail = ", ".join(paths[:3])
        return f"file change {phase}: {detail}" if detail else f"file change {phase}"

    if item_type in {"agent_message", "reasoning"}:
        text = _compact(item.get("text"))
        label = item_type.replace("_", " ")
        return f"{label} {phase}: {text}" if text else f"{label} {phase}"

    return f"{item_type.replace('_', ' ')} {phase}"


def _process_codex_event(
    event: dict[str, Any],
    attempt: RunAttempt,
    identifier: str,
    on_event: EventCallback | None,
) -> None:
    """Update attempt state and log one Codex ``--json`` event."""
    event_type = str(event.get("type", "unknown"))
    attempt.last_event = event_type
    summary = event_type

    if event_type == "thread.started":
        # Codex runs are ephemeral, so this ID is useful activity telemetry but
        # must not be persisted as a resumable Claude session ID.
        summary = "thread started"
    elif event_type.startswith("item."):
        item = event.get("item", {})
        if isinstance(item, dict):
            summary = _codex_item_summary(event_type, item)
            attempt.last_message = summary[:200]
    elif event_type == "turn.started":
        summary = "turn started"
        attempt.last_message = summary
    elif event_type == "turn.completed":
        usage = event.get("usage", {})
        if isinstance(usage, dict):
            attempt.input_tokens = int(usage.get("input_tokens", 0) or 0)
            attempt.output_tokens = int(usage.get("output_tokens", 0) or 0)
            attempt.total_tokens = int(
                usage.get("total_tokens", 0)
                or attempt.input_tokens + attempt.output_tokens
            )
        summary = f"turn completed tokens={attempt.total_tokens}"
    elif event_type in {"turn.failed", "error"}:
        error = event.get("error", event.get("message", ""))
        if isinstance(error, dict):
            error = error.get("message", "")
        detail = _compact(error)
        summary = f"{event_type}: {detail}" if detail else event_type
        attempt.last_message = summary[:200]

    logger.info(
        f"Codex issue={identifier} {summary}",
        extra={"linked_to": identifier},
    )
    if on_event:
        on_event(identifier, event_type, event)


def build_claude_args(
    claude_cfg: ClaudeConfig,
    prompt: str,
    workspace_path: Path,
    session_id: str | None = None,
) -> list[str]:
    """Build the claude CLI argument list."""
    args = [claude_cfg.command]

    if session_id:
        # Continuation turn
        args.extend(["-p", prompt, "--resume", session_id])
    else:
        # First turn
        args.extend(["-p", prompt])

    args.extend(["--verbose", "--output-format", "stream-json"])

    # Permission mode
    if claude_cfg.permission_mode == "auto":
        args.append("--dangerously-skip-permissions")
    elif claude_cfg.permission_mode == "allowedTools" and claude_cfg.allowed_tools:
        args.extend(["--allowedTools", ",".join(claude_cfg.allowed_tools)])

    # Model override
    if claude_cfg.model:
        args.extend(["--model", claude_cfg.model])

    # System prompt - always include headless context, plus any user additions
    if not session_id:
        headless_context = (
            "You are running in headless/unattended mode via Stokowski orchestrator. "
            "Do NOT use interactive skills, slash commands, or the Skill tool. "
            "Do NOT invoke brainstorming, plan mode, or any interactive workflow. "
            "Work autonomously and directly on the task."
        )
        extra = claude_cfg.append_system_prompt or ""
        combined = f"{headless_context}\n{extra}".strip()
        args.extend(["--append-system-prompt", combined])

    return args


def build_codex_args(
    model: str | None,
    prompt: str,
    workspace_path: Path,
    reasoning_effort: str | None = None,
) -> list[str]:
    """Build a non-interactive Codex JSONL invocation."""
    if (
        reasoning_effort is not None
        and reasoning_effort not in CODEX_REASONING_EFFORTS
    ):
        raise ValueError(f"Unsupported Codex reasoning effort: {reasoning_effort!r}")

    args = [
        "codex",
        "exec",
        "--sandbox",
        "danger-full-access",
        "--ephemeral",
        "--json",
        "--cd",
        str(workspace_path),
        "--config",
        'approval_policy="never"',
    ]
    if model:
        args.extend(["--model", model])
    if reasoning_effort:
        args.extend(
            ["--config", f'model_reasoning_effort="{reasoning_effort}"']
        )
    args.append(prompt)
    return args


async def run_codex_turn(
    model: str | None,
    hooks_cfg: HooksConfig,
    prompt: str,
    workspace_path: Path,
    issue: Issue,
    attempt: RunAttempt,
    reasoning_effort: str | None = None,
    on_event: EventCallback | None = None,
    on_pid: PidCallback | None = None,
    turn_timeout_ms: int = 3_600_000,
    stall_timeout_ms: int = 300_000,
    env: dict[str, str] | None = None,
) -> RunAttempt:
    """Run a single Codex turn. Returns updated RunAttempt.

    Codex sessions are ephemeral here, so each state gets a fresh run. JSONL
    output keeps the activity monitor updated during long-running turns.
    """
    args = build_codex_args(model, prompt, workspace_path, reasoning_effort)

    # Run before_run hook
    if hooks_cfg.before_run:
        from .workspace import run_hook

        ok = await run_hook(
            hooks_cfg.before_run, workspace_path, hooks_cfg.timeout_ms, "before_run"
        )
        if not ok:
            attempt.status = "failed"
            attempt.error = "before_run hook failed"
            return attempt

    attempt.status = "streaming"
    attempt.started_at = attempt.started_at or datetime.now(timezone.utc)
    attempt.turn_count += 1
    attempt.last_event_at = datetime.now(timezone.utc)

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(workspace_path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=10 * 1024 * 1024,  # 10MB line buffer (default 64KB)
            env=build_agent_env(env),
        )
        if on_pid and proc.pid:
            on_pid(proc.pid, True)
        logger.info(
            f"Codex started issue={issue.identifier} "
            f"turn={attempt.turn_count} pid={proc.pid}",
            extra={"linked_to": issue.identifier},
        )
    except FileNotFoundError:
        attempt.status = "failed"
        attempt.error = "Codex command not found: codex"
        logger.error(attempt.error, extra={"linked_to": issue.identifier})
        return attempt

    loop = asyncio.get_running_loop()
    last_activity = loop.time()
    stall_timeout_s = stall_timeout_ms / 1000
    turn_timeout_s = turn_timeout_ms / 1000

    async def read_stdout():
        nonlocal last_activity
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            last_activity = loop.time()
            attempt.last_event_at = datetime.now(timezone.utc)
            line_str = line.decode(errors="replace").strip()
            if not line_str:
                continue

            try:
                event = json.loads(line_str)
            except json.JSONDecodeError:
                logger.warning(
                    f"Codex issue={issue.identifier} emitted non-JSON stdout: "
                    f"{line_str[:500]}",
                    extra={"linked_to": issue.identifier},
                )
                attempt.last_message = line_str[:200]
                continue

            if isinstance(event, dict):
                _process_codex_event(event, attempt, issue.identifier, on_event)

    async def read_stderr() -> str:
        nonlocal last_activity
        output_lines: list[str] = []
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            last_activity = loop.time()
            attempt.last_event_at = datetime.now(timezone.utc)
            line_str = line.decode(errors="replace").strip()
            if line_str:
                output_lines.append(line_str[-2000:])
                if len(output_lines) > 50:
                    del output_lines[0]
                logger.warning(
                    f"Codex issue={issue.identifier} stderr: {line_str[:500]}",
                    extra={"linked_to": issue.identifier},
                )
        return "\n".join(output_lines)

    async def stall_monitor():
        while proc.returncode is None:
            await asyncio.sleep(max(0.1, min(stall_timeout_s / 4, 30)))
            elapsed = loop.time() - last_activity
            if elapsed > stall_timeout_s:
                logger.warning(
                    f"Codex stall detected issue={issue.identifier} "
                    f"elapsed={elapsed:.0f}s",
                    extra={"linked_to": issue.identifier},
                )
                _kill_process_group(proc)
                attempt.status = "stalled"
                attempt.error = f"No output for {elapsed:.0f}s"
                return

    stdout_reader = asyncio.create_task(read_stdout())
    stderr_reader = asyncio.create_task(read_stderr())
    monitor = (
        asyncio.create_task(stall_monitor()) if stall_timeout_s > 0 else None
    )
    stderr_output = ""

    try:
        await asyncio.wait_for(proc.wait(), timeout=turn_timeout_s)
    except asyncio.TimeoutError:
        logger.warning(
            f"Codex turn timeout issue={issue.identifier}",
            extra={"linked_to": issue.identifier},
        )
        attempt.status = "timed_out"
        attempt.error = f"Turn exceeded {turn_timeout_s}s"
        _kill_process_group(proc)
    except Exception as e:
        logger.error(
            f"Codex runner error issue={issue.identifier}: {e}",
            extra={"linked_to": issue.identifier},
        )
        attempt.status = "failed"
        attempt.error = str(e)
        _kill_process_group(proc)
    finally:
        if proc.returncode is None:
            _kill_process_group(proc)
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                logger.error(
                    f"Codex process group did not exit issue={issue.identifier}",
                    extra={"linked_to": issue.identifier},
                )

        if monitor is not None:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass

        try:
            stream_results = await asyncio.wait_for(
                asyncio.gather(
                    stdout_reader,
                    stderr_reader,
                    return_exceptions=True,
                ),
                timeout=5,
            )
        except asyncio.TimeoutError:
            stdout_reader.cancel()
            stderr_reader.cancel()
            await asyncio.gather(
                stdout_reader,
                stderr_reader,
                return_exceptions=True,
            )
            logger.warning(
                f"Codex output streams did not close issue={issue.identifier}",
                extra={"linked_to": issue.identifier},
            )
        else:
            if isinstance(stream_results[1], str):
                stderr_output = stream_results[1][-500:]
            for stream_name, result in zip(
                ("stdout", "stderr"), stream_results
            ):
                if isinstance(result, Exception):
                    logger.error(
                        f"Codex {stream_name} reader failed "
                        f"issue={issue.identifier}: {result}",
                        extra={"linked_to": issue.identifier},
                    )
                    if attempt.status == "streaming":
                        attempt.status = "failed"
                        attempt.error = f"Codex {stream_name} reader failed: {result}"

        if on_pid and proc.pid:
            on_pid(proc.pid, False)

    # Determine final status from exit code if not already set
    if attempt.status == "streaming":
        if proc.returncode == 0:
            attempt.status = "succeeded"
        else:
            attempt.status = "failed"
            attempt.error = f"Codex exit code {proc.returncode}: {stderr_output}"

    # Run after_run hook
    if hooks_cfg.after_run:
        from .workspace import run_hook

        await run_hook(
            hooks_cfg.after_run, workspace_path, hooks_cfg.timeout_ms, "after_run"
        )

    logger.info(
        f"Codex turn complete issue={issue.identifier} "
        f"status={attempt.status}",
        extra={"linked_to": issue.identifier},
    )

    return attempt


async def run_agent_turn(
    claude_cfg: ClaudeConfig,
    hooks_cfg: HooksConfig,
    prompt: str,
    workspace_path: Path,
    issue: Issue,
    attempt: RunAttempt,
    on_event: EventCallback | None = None,
    on_pid: PidCallback | None = None,
    env: dict[str, str] | None = None,
) -> RunAttempt:
    """Run a single Claude Code turn. Returns updated RunAttempt."""
    args = build_claude_args(
        claude_cfg, prompt, workspace_path, attempt.session_id
    )

    logger.info(
        f"Launching claude issue={issue.identifier} "
        f"session={attempt.session_id or 'new'} "
        f"turn={attempt.turn_count + 1}",
        extra={"linked_to": issue.identifier},
    )

    # Run before_run hook
    if hooks_cfg.before_run:
        from .workspace import run_hook

        ok = await run_hook(
            hooks_cfg.before_run, workspace_path, hooks_cfg.timeout_ms, "before_run"
        )
        if not ok:
            attempt.status = "failed"
            attempt.error = "before_run hook failed"
            return attempt

    attempt.status = "streaming"
    attempt.started_at = attempt.started_at or datetime.now(timezone.utc)
    attempt.turn_count += 1
    attempt.last_event_at = datetime.now(timezone.utc)

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(workspace_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=10 * 1024 * 1024,  # 10MB line buffer (default 64KB)
            env=build_agent_env(env),
        )
        if on_pid and proc.pid:
            on_pid(proc.pid, True)
    except FileNotFoundError:
        attempt.status = "failed"
        attempt.error = f"Claude command not found: {claude_cfg.command}"
        logger.error(attempt.error, extra={"linked_to": issue.identifier})
        return attempt

    # Stream stdout (NDJSON events)
    loop = asyncio.get_running_loop()
    last_activity = loop.time()
    stall_timeout_s = claude_cfg.stall_timeout_ms / 1000
    turn_timeout_s = claude_cfg.turn_timeout_ms / 1000

    async def read_stream():
        nonlocal last_activity
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            last_activity = loop.time()
            attempt.last_event_at = datetime.now(timezone.utc)

            line_str = line.decode().strip()
            if not line_str:
                continue

            try:
                event = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            _process_event(event, attempt, on_event, issue.identifier)

    async def stall_monitor():
        while proc.returncode is None:
            await asyncio.sleep(min(stall_timeout_s / 4, 30))
            elapsed = loop.time() - last_activity
            if stall_timeout_s > 0 and elapsed > stall_timeout_s:
                logger.warning(
                    f"Stall detected issue={issue.identifier} "
                    f"elapsed={elapsed:.0f}s",
                    extra={"linked_to": issue.identifier},
                )
                proc.kill()
                attempt.status = "stalled"
                attempt.error = f"No output for {elapsed:.0f}s"
                return

    try:
        reader = asyncio.create_task(read_stream())
        monitor = asyncio.create_task(stall_monitor())

        # Overall turn timeout
        done, pending = await asyncio.wait(
            {reader, monitor},
            timeout=turn_timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            # Turn timeout
            logger.warning(f"Turn timeout issue={issue.identifier}", extra={"linked_to": issue.identifier})
            proc.kill()
            attempt.status = "timed_out"
            attempt.error = f"Turn exceeded {turn_timeout_s}s"
        else:
            # Wait for process to finish
            await asyncio.wait_for(proc.wait(), timeout=30)

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.error(f"Runner error issue={issue.identifier}: {e}", extra={"linked_to": issue.identifier})
        proc.kill()
        attempt.status = "failed"
        attempt.error = str(e)
        return attempt

    # Determine final status from exit code if not already set by stall/timeout
    if attempt.status == "streaming":
        if proc.returncode == 0:
            attempt.status = "succeeded"
        else:
            stderr_output = ""
            if proc.stderr:
                try:
                    stderr_bytes = await asyncio.wait_for(proc.stderr.read(), timeout=5)
                    stderr_output = stderr_bytes.decode()[:500]
                except (asyncio.TimeoutError, Exception):
                    pass
            attempt.status = "failed"
            attempt.error = f"Exit code {proc.returncode}: {stderr_output}"

    # Run after_run hook
    if hooks_cfg.after_run:
        from .workspace import run_hook

        await run_hook(
            hooks_cfg.after_run, workspace_path, hooks_cfg.timeout_ms, "after_run"
        )

    # Unregister PID
    if on_pid and proc.pid:
        on_pid(proc.pid, False)

    logger.info(
        f"Turn complete issue={issue.identifier} "
        f"status={attempt.status} "
        f"tokens={attempt.total_tokens}",
        extra={"linked_to": issue.identifier},
    )

    return attempt


def _process_event(
    event: dict,
    attempt: RunAttempt,
    on_event: EventCallback | None,
    identifier: str,
):
    """Process a single NDJSON event from Claude Code stream-json output."""
    event_type = event.get("type", "")
    attempt.last_event = event_type

    # Extract session_id from result events
    if event_type == "result":
        if "session_id" in event:
            attempt.session_id = event["session_id"]
        # Extract token usage
        usage = event.get("usage", {})
        if usage:
            attempt.input_tokens = usage.get("input_tokens", attempt.input_tokens)
            attempt.output_tokens = usage.get("output_tokens", attempt.output_tokens)
            attempt.total_tokens = (
                usage.get("total_tokens", 0)
                or attempt.input_tokens + attempt.output_tokens
            )
        # Extract result text for last_message
        result_text = event.get("result", "")
        if isinstance(result_text, str) and result_text:
            attempt.last_message = result_text[:200]

    elif event_type == "assistant":
        # Assistant message content
        msg = event.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            attempt.last_message = content[:200]
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    attempt.last_message = block.get("text", "")[:200]
                    break

    elif event_type == "tool_use":
        tool_name = event.get("name", event.get("tool", ""))
        attempt.last_message = f"Using tool: {tool_name}"

    # Forward to orchestrator callback
    if on_event:
        on_event(identifier, event_type, event)


async def run_turn(
    runner_type: str,
    claude_cfg: ClaudeConfig,
    hooks_cfg: HooksConfig,
    prompt: str,
    workspace_path: Path,
    issue: Issue,
    attempt: RunAttempt,
    reasoning_effort: str | None = None,
    on_event: EventCallback | None = None,
    on_pid: PidCallback | None = None,
    env: dict[str, str] | None = None,
) -> RunAttempt:
    """Route to the correct runner based on runner_type."""
    if runner_type == "codex":
        return await run_codex_turn(
            model=claude_cfg.model,
            reasoning_effort=reasoning_effort,
            hooks_cfg=hooks_cfg,
            prompt=prompt,
            workspace_path=workspace_path,
            issue=issue,
            attempt=attempt,
            on_event=on_event,
            on_pid=on_pid,
            turn_timeout_ms=claude_cfg.turn_timeout_ms,
            stall_timeout_ms=claude_cfg.stall_timeout_ms,
            env=env,
        )
    elif runner_type == "claude":
        return await run_agent_turn(
            claude_cfg=claude_cfg,
            hooks_cfg=hooks_cfg,
            prompt=prompt,
            workspace_path=workspace_path,
            issue=issue,
            attempt=attempt,
            on_event=on_event,
            on_pid=on_pid,
            env=env,
        )
    else:
        raise ValueError(f"Unknown runner type: {runner_type}")
