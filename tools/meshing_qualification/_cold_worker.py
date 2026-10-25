# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bootstrap of one prepared cold worker; ``_runtime.prepare_cold_worker`` launches it.

``python -m tools.meshing_qualification._cold_worker MODULE LAUNCHER_PID`` records
its start on ``CLOCK_MONOTONIC`` before any heavy import, dies with its launcher,
reserves the launcher's stdout for its one result (stray prints go to stderr), and
then imports ``MODULE`` and serves exactly one control packet through
``_runtime.serve_cold_worker``.
"""

from __future__ import annotations

import time


_STARTED = time.clock_gettime(time.CLOCK_MONOTONIC)

import ctypes
import os
import signal
import sys


_PR_SET_PDEATHSIG = 1


def _bind_to_launcher(launcher: int, /) -> None:
    """Receive SIGKILL when the launcher dies, even if it never ends this session."""
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, f"prctl(PR_SET_PDEATHSIG) failed: {os.strerror(error)}")
    elif sys.platform == "darwin":
        import select
        from threading import Thread

        queue = select.kqueue()
        queue.control(
            [select.kevent(
                launcher, filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                fflags=select.KQ_NOTE_EXIT,
            )],
            0, 0,
        )

        def terminate_with_launcher() -> None:
            try:
                queue.control(None, 1, None)
            finally:
                queue.close()
            os.kill(os.getpid(), signal.SIGKILL)

        Thread(
            target=terminate_with_launcher, name="cold-launcher-lifetime", daemon=True,
        ).start()
    # The launcher may have died before the binding existed.
    if os.getppid() != launcher:
        raise RuntimeError(
            f"cold_worker.parent must be the launcher {launcher}; got {os.getppid()}"
        )


def main(argv: list[str], /) -> None:
    if len(argv) != 2:
        raise ValueError(f"cold_worker.argv must be MODULE LAUNCHER_PID; got {argv!r}")
    module, launcher = argv[0], int(argv[1])
    _bind_to_launcher(launcher)
    result = os.dup(1)
    os.dup2(2, 1)
    from tools.meshing_qualification._runtime import serve_cold_worker

    serve_cold_worker(module, started=_STARTED, result_descriptor=result)


if __name__ == "__main__":
    main(sys.argv[1:])
