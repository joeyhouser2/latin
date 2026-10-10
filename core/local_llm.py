"""Local LLM access through Ollama, including a private server pinned to one GPU.

Why Ollama rather than loading a model with transformers the way the stylizer
does: the models worth using for summarization here are 12-14B, which in fp16
is 24-28GB -- more than either card on this machine. Ollama serves 4-bit GGUF
builds of exactly those models (gemma4:12b is 7.6GB, qwen3:14b 9.3GB) and is
already installed with them pulled.

Why a *private* server rather than the Ollama service already running on 11434:
that service picks its own GPU (and will split a model across both cards if it
feels like it), and it may already be busy for another project. A summarize job
has to run on the card the translation job is *not* using, so it starts its own
``ollama serve`` on a free port, inheriting the job's ``CUDA_VISIBLE_DEVICES``
-- which the queue sets to one card's UUID -- and shuts it down afterwards. The
two servers share the same model files on disk; nothing is downloaded twice.

    with PrivateOllama(log_path="data/joblogs/ollama.log") as llm:
        out = llm.client.chat_json("gemma4:12b", system, user, schema)

``OllamaClient`` also works against the shared service directly, which is what
the web app does for the one cheap thing it needs at query time: embedding a
search query.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests

DEFAULT_URL = "http://" + os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").replace("http://", "")


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._caps: Dict[str, List[str]] = {}

    # -- discovery -----------------------------------------------------------

    def models(self, timeout: float = 5.0) -> List[Dict[str, Any]]:
        r = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
        r.raise_for_status()
        models = r.json().get("models", [])
        for m in models:
            self._caps[m["name"]] = m.get("capabilities") or []
        return models

    def has_model(self, name: str) -> bool:
        names = {m["name"] for m in self.models()}
        return name in names or f"{name}:latest" in names

    def capabilities(self, name: str) -> List[str]:
        if name not in self._caps:
            try:
                r = requests.post(f"{self.base_url}/api/show", json={"model": name},
                                  timeout=30)
                r.raise_for_status()
                self._caps[name] = r.json().get("capabilities") or []
            except requests.RequestException:
                self._caps[name] = []
        return self._caps[name]

    # -- generation ----------------------------------------------------------

    def chat_json(self, model: str, system: str, user: str, schema: Dict[str, Any],
                  *, num_ctx: int = 8192, temperature: float = 0.2,
                  num_predict: int = 700, retries: int = 1) -> Dict[str, Any]:
        """One chat turn constrained to ``schema``; returns the parsed object.

        Thinking is switched off for models that support it. A summary does not
        benefit from a thousand tokens of visible deliberation, and on a
        thinking model those tokens come out of the same ``num_predict`` budget
        -- the answer gets truncated and the JSON fails to parse.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": {"num_ctx": num_ctx, "temperature": temperature,
                        "num_predict": num_predict},
            "keep_alive": "15m",
        }
        # Only send `think` to models that know it: some Ollama versions reject
        # the field outright for a model without the capability.
        if "thinking" in self.capabilities(model):
            payload["think"] = False

        last_err: Optional[Exception] = None
        for _ in range(retries + 1):
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            if r.status_code != 200:
                last_err = OllamaError(f"{r.status_code}: {r.text[:300]}")
                continue
            content = r.json().get("message", {}).get("content", "")
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                last_err = OllamaError(f"model returned invalid JSON ({exc}): {content[:200]!r}")
                # A little warmer on the retry, so it does not reproduce the same
                # malformed output token for token.
                payload["options"]["temperature"] = min(0.7, temperature + 0.3)
        raise last_err or OllamaError("chat failed")

    def embed(self, model: str, texts: List[str], timeout: Optional[float] = None) -> List[List[float]]:
        if not texts:
            return []
        r = requests.post(f"{self.base_url}/api/embed",
                          json={"model": model, "input": texts, "keep_alive": "15m"},
                          timeout=timeout or self.timeout)
        if r.status_code != 200:
            raise OllamaError(f"embed {r.status_code}: {r.text[:300]}")
        return r.json().get("embeddings", [])

    def unload_all(self) -> None:
        """Evict every loaded model now, freeing its VRAM immediately."""
        try:
            loaded = requests.get(f"{self.base_url}/api/ps", timeout=10).json().get("models", [])
            for m in loaded:
                requests.post(f"{self.base_url}/api/generate",
                              json={"model": m["name"], "keep_alive": 0}, timeout=60)
        except requests.RequestException:
            pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _KillOnCloseJob:
    """A Windows Job Object that kills every process in it when we exit.

    ``ollama serve`` does the actual inference in a child ``llama-server.exe``,
    and on Windows killing a parent does not kill its children. Found the hard
    way: a terminated private server left its runner alive holding 8GB on a
    GPU, invisible to Ollama and to us, silently starving whatever the queue put
    on that card next. Processes assigned to this job -- and every process they
    spawn afterwards, which inherit membership -- are killed by the OS when the
    job handle closes, and the handle closes when this Python process dies *by
    any means*, including a hard kill that never runs a ``finally``.

    No-op off Windows (see ``PrivateOllama``'s process-group handling there).
    """

    def __init__(self) -> None:
        self.handle = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class _BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _BasicLimits),
                        ("IoInfo", _IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.OpenProcess.restype = wintypes.HANDLE
        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            return
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(handle, JobObjectExtendedLimitInformation,
                                           ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(handle)
            return
        self._k32, self.handle = k32, handle

    def add(self, pid: int) -> bool:
        if not self.handle:
            return False
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
        ph = self._k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not ph:
            return False
        try:
            return bool(self._k32.AssignProcessToJobObject(self.handle, ph))
        finally:
            self._k32.CloseHandle(ph)

    def close(self) -> None:
        """Kill everything in the job now (rather than waiting for our own exit)."""
        if self.handle:
            self._k32.CloseHandle(self.handle)
            self.handle = None


class PrivateOllama:
    """A throwaway ``ollama serve`` on a free port, for the life of a job.

    Inherits the current environment, so ``CUDA_VISIBLE_DEVICES`` (set by the
    job queue to one card's UUID) is what pins it -- *provided* Ollama's Vulkan
    backend is off. Ollama 0.34 enables Vulkan by default and enumerates every
    card through it regardless of CUDA_VISIBLE_DEVICES; in testing, a server
    "pinned" to the 4070 SUPER loaded its model onto the 4060 Ti over Vulkan.
    So the private server always runs with ``OLLAMA_VULKAN=false``.

    On exit it unloads its models, stops the server, and then closes a
    kill-on-close Job Object holding the whole process tree, so the llama
    runner cannot outlive it.
    """

    def __init__(self, log_path: Optional[str] = None, startup_timeout: float = 90.0):
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.proc: Optional[subprocess.Popen] = None
        self.client: Optional[OllamaClient] = None
        self._log = None
        self._job = _KillOnCloseJob()

    def __enter__(self) -> "PrivateOllama":
        exe = shutil.which("ollama")
        if not exe and os.name == "nt":
            # The per-user installer doesn't always reach the PATH of an
            # already-running parent process (app, IDE, terminal).
            cand = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                                "Programs", "Ollama", "ollama.exe")
            if os.path.isfile(cand):
                exe = cand
        if not exe:
            raise OllamaError("ollama is not installed or not on PATH")
        port = _free_port()
        env = dict(os.environ,
                   OLLAMA_HOST=f"127.0.0.1:{port}",
                   # One request at a time and at most two models resident (the
                   # chat model plus the small embedding model): predictable VRAM.
                   OLLAMA_NUM_PARALLEL="1",
                   OLLAMA_MAX_LOADED_MODELS="2",
                   # Vulkan ignores CUDA_VISIBLE_DEVICES -- see the class docstring.
                   OLLAMA_VULKAN="false")
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace") \
            if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen([exe, "serve"], env=env, stdout=self._log,
                                     stderr=subprocess.STDOUT,
                                     start_new_session=(os.name != "nt"))
        # Safe to assign after the fact: the server only spawns its llama runner
        # when the first model loads, which is after our first request.
        self._job.add(self.proc.pid)
        self.client = OllamaClient(f"http://127.0.0.1:{port}")
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise OllamaError(f"ollama serve exited early (code {self.proc.returncode}); "
                                  f"see {self.log_path}")
            try:
                requests.get(f"{self.client.base_url}/api/version", timeout=2).raise_for_status()
                return self
            except requests.RequestException:
                time.sleep(0.5)
        self.__exit__(None, None, None)
        raise OllamaError("ollama serve did not come up in time")

    def gpu_lines(self) -> List[str]:
        """What Ollama says it found -- the proof of which card it is using."""
        if not self.log_path or not os.path.isfile(self.log_path):
            return []
        with open(self.log_path, encoding="utf-8", errors="replace") as fh:
            return [l.strip() for l in fh if "inference compute" in l or "offloaded" in l]

    def __exit__(self, *exc) -> None:
        if self.client:
            self.client.unload_all()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        # Whatever is left of the tree -- the llama runner, above all -- dies here.
        self._job.close()
        if os.name != "nt" and self.proc is not None:
            import signal as _signal
            try:
                os.killpg(self.proc.pid, _signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
