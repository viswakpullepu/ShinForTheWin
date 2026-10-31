"""One sandboxed bot: a child process (docker container or local subprocess) speaking JSON lines."""
import json
import os
import select
import subprocess
import sys
import time
import uuid
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_LINE_BYTES = 1 << 16


class BotFault(Exception):
    """kind: timeout | crash | error | invalid | illegal | budget. Crash and budget are fatal."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


class BotProcess:
    def __init__(self, submission_dir: str, backend: str = "docker", image: str = "cses-arcade-sandbox",
                 memory: str = "128m", cpus: str = "1", stderr_path: Optional[str] = None):
        self.submission_dir = os.path.abspath(submission_dir)
        self.backend = backend
        self.name = f"cses-bot-{uuid.uuid4().hex[:10]}"
        self._next_id = 0
        self._buf = b""
        stderr = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        if backend == "docker":
            cmd = ["docker", "run", "-i", "--rm", "--name", self.name, "--network=none",
                   f"--memory={memory}", f"--cpus={cpus}", "--pids-limit=64", "--read-only",
                   "--tmpfs", "/tmp:size=64m", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                   "-v", f"{self.submission_dir}:/submission:ro", image]
            env = None
        elif backend == "local":  # development only: no isolation or resource limits
            cmd = [sys.executable, os.path.join(ROOT, "sandbox", "runner.py")]
            env = dict(os.environ, SUBMISSION_DIR=self.submission_dir, PYTHONPATH=ROOT)
        else:
            raise ValueError(f"unknown backend {backend}")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, env=env)
        self._fd = self.proc.stdout.fileno()
        if stderr is not subprocess.DEVNULL:
            stderr.close()

    def _readline(self, deadline: float) -> bytes:
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BotFault("timeout")
            if sys.platform == "win32":
                chunk = self.proc.stdout.readline()
                if not chunk:
                    raise BotFault("crash", f"process exited with {self.proc.poll()}")
                self._buf += chunk
            else:
                ready, _, _ = select.select([self._fd], [], [], remaining)
                if not ready:
                    continue
                chunk = os.read(self._fd, 65536)
                if not chunk:
                    raise BotFault("crash", f"process exited with {self.proc.poll()}")
                self._buf += chunk
            if len(self._buf) > MAX_LINE_BYTES and b"\n" not in self._buf:
                raise BotFault("crash", "output line too long")
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def request(self, message: dict, timeout: float) -> dict:
        self._next_id += 1
        request_id = self._next_id
        payload = json.dumps(dict(message, id=request_id)) + "\n"
        deadline = time.monotonic() + timeout
        try:
            self.proc.stdin.write(payload.encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise BotFault("crash", "could not write to bot")
        while True:
            line = self._readline(deadline)
            try:
                reply = json.loads(line)
            except ValueError:
                raise BotFault("invalid", "reply is not JSON")
            if not isinstance(reply, dict):
                raise BotFault("invalid", "reply is not an object")
            if reply.get("id") != request_id:
                continue  # late answer to an earlier, timed-out request
            if "error" in reply:
                raise BotFault("error", str(reply["error"]))
            return reply

    def close(self) -> None:
        if self.backend == "docker":
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        try:
            self.proc.kill()
        except OSError:
            pass
        self.proc.wait()
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except OSError:
                pass
