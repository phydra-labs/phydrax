# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Cold-worker programs exercising the ``prepare_cold_worker`` lifecycle."""

from __future__ import annotations

import os
import time
from pathlib import Path
from time import perf_counter

from tools.meshing_qualification._runtime import ColdControlPacket, ColdWorkerReport


IMPORTED_MARKER_VARIABLE = "PHYDRAX_TEST_COLD_WORKER_IMPORTED"
"""When set, the imported worker writes its PID to this path before its packet wait."""

if (marker := os.environ.get(IMPORTED_MARKER_VARIABLE)) is not None:
    staged = Path(f"{marker}.staged")
    staged.write_text(str(os.getpid()), encoding="utf-8")
    staged.replace(marker)


def cold_worker(packet: ColdControlPacket) -> ColdWorkerReport:
    """Read the archive, then echo, fail, sleep or self-report by ``controls["mode"]``."""
    started = perf_counter()
    archive = packet.archive_path.read_text(encoding="utf-8")
    read_seconds = perf_counter() - started
    # Stray output must reach stderr, never the reserved result channel.
    print("cold_worker_programs diagnostic")
    match packet.controls["mode"]:
        case "echo":
            return ColdWorkerReport(
                record={
                    "archive": archive,
                    "content_id": packet.content_id,
                    "controls": packet.controls,
                    "remaining_seconds": packet.deadline.remaining(),
                },
                read_seconds=read_seconds,
            )
        case "fail":
            raise ValueError("cold_worker_programs.mode must refuse; got fail")
        case "sleep":
            while True:
                time.sleep(1.0)
        case "process_id":
            return ColdWorkerReport(
                record={"process_id": os.getpid()}, read_seconds=read_seconds
            )
        case mode:
            raise ValueError(f"cold_worker_programs.mode must be declared; got {mode!r}")
