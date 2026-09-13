import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from stokowski.workspace import run_hook


class HookLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_logs_success_and_captured_output(self):
        with TemporaryDirectory() as directory:
            with self.assertLogs("stokowski.workspace", level="DEBUG") as logs:
                succeeded = await run_hook(
                    "printf 'fetched main'; printf 'diagnostic' >&2",
                    Path(directory),
                    1_000,
                    "before_run",
                )

        self.assertTrue(succeeded)
        output = "\n".join(logs.output)
        self.assertIn("hook=before_run complete rc=0", output)
        self.assertIn("hook=before_run stdout=fetched main", output)
        self.assertIn("hook=before_run stderr=diagnostic", output)

    async def test_logs_both_streams_on_failure(self):
        with TemporaryDirectory() as directory:
            with self.assertLogs("stokowski.workspace", level="ERROR") as logs:
                succeeded = await run_hook(
                    "printf 'context'; printf 'failure' >&2; exit 7",
                    Path(directory),
                    1_000,
                    "before_run",
                )

        self.assertFalse(succeeded)
        output = "\n".join(logs.output)
        self.assertIn("hook=before_run failed rc=7", output)
        self.assertIn("stdout=context", output)
        self.assertIn("stderr=failure", output)


if __name__ == "__main__":
    unittest.main()
