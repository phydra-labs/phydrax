#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Safe rank environments and local multi-process launch without shell execution."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankEnvironment:
    coordinator_address: str
    num_processes: int
    process_id: int
    local_device_ids: tuple[int, ...] | None = None
    coordinator_bind_address: str | None = None

    def __post_init__(self) -> None:
        if not self.coordinator_address.strip():
            raise ValueError("coordinator_address must be non-empty")
        if self.num_processes <= 1:
            raise ValueError("rank environments require num_processes > 1")
        if not 0 <= self.process_id < self.num_processes:
            raise ValueError("process_id must be in [0, num_processes)")
        if self.local_device_ids is not None:
            if not self.local_device_ids or min(self.local_device_ids) < 0:
                raise ValueError("local_device_ids must be non-empty and non-negative")
            if len(set(self.local_device_ids)) != len(self.local_device_ids):
                raise ValueError("local_device_ids must be unique")

    def apply(self, environment: Mapping[str, str] | None = None) -> dict[str, str]:
        values = dict(os.environ if environment is None else environment)
        values.update(
            {
                "PHYDRAX_COORDINATOR_ADDRESS": self.coordinator_address,
                "PHYDRAX_NUM_PROCESSES": str(self.num_processes),
                "PHYDRAX_PROCESS_ID": str(self.process_id),
            }
        )
        if self.local_device_ids is not None:
            values["PHYDRAX_LOCAL_DEVICE_IDS"] = ",".join(
                str(value) for value in self.local_device_ids
            )
        if self.coordinator_bind_address is not None:
            values["PHYDRAX_COORDINATOR_BIND_ADDRESS"] = self.coordinator_bind_address
        return values


@dataclass(frozen=True, slots=True)
class LocalProcessLaunchPlan:
    """One homogeneous local process set with disjoint device assignments."""

    num_processes: int
    coordinator_address: str
    devices_per_process: int | None = None

    def __post_init__(self) -> None:
        if self.num_processes <= 1:
            raise ValueError("local process launch requires num_processes > 1")
        if not self.coordinator_address.startswith(("127.0.0.1:", "[::1]:")):
            raise ValueError("local coordinator_address must bind to loopback")
        if self.devices_per_process is not None and self.devices_per_process <= 0:
            raise ValueError("devices_per_process must be positive")

    def rank(self, process_id: int) -> RankEnvironment:
        local_devices = None
        if self.devices_per_process is not None:
            start = int(process_id) * self.devices_per_process
            local_devices = tuple(range(start, start + self.devices_per_process))
        return RankEnvironment(
            self.coordinator_address,
            self.num_processes,
            int(process_id),
            local_device_ids=local_devices,
            coordinator_bind_address=(
                self.coordinator_address if int(process_id) == 0 else None
            ),
        )


@dataclass(frozen=True, slots=True)
class LocalProcessResult:
    process_id: int
    returncode: int
    stdout: bytes
    stderr: bytes


class LocalProcessLaunchError(RuntimeError):
    def __init__(self, results: Sequence[LocalProcessResult], /) -> None:
        self.results = tuple(results)
        failed = tuple(result for result in self.results if result.returncode)
        summary = "; ".join(
            (
                f"rank {result.process_id} exited {result.returncode} "
                f"(stderr_bytes={len(result.stderr)})"
            )
            for result in failed
        )
        super().__init__(f"local distributed launch failed: {summary}")


def launch_local_processes(
    argv: Sequence[str],
    plan: LocalProcessLaunchPlan,
    /,
    *,
    environment: Mapping[str, str] | None = None,
    cwd: str | None = None,
) -> tuple[LocalProcessResult, ...]:
    """Run one argument vector per rank and return canonical rank-ordered output."""

    arguments = tuple(str(value) for value in argv)
    if not arguments or any(not value or "\x00" in value for value in arguments):
        raise ValueError("argv must contain non-empty NUL-free arguments")
    processes = tuple(
        subprocess.Popen(
            arguments,
            cwd=cwd,
            env=plan.rank(process_id).apply(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        for process_id in range(plan.num_processes)
    )

    def collect(item: tuple[int, subprocess.Popen[bytes]]) -> LocalProcessResult:
        process_id, process = item
        stdout, stderr = process.communicate()
        return LocalProcessResult(process_id, process.returncode, stdout, stderr)

    with ThreadPoolExecutor(
        max_workers=plan.num_processes,
        thread_name_prefix="phydrax-local-rank",
    ) as executor:
        results = tuple(executor.map(collect, enumerate(processes)))
    if any(result.returncode for result in results):
        raise LocalProcessLaunchError(results)
    return results


__all__ = (
    "LocalProcessLaunchError",
    "LocalProcessLaunchPlan",
    "LocalProcessResult",
    "RankEnvironment",
    "launch_local_processes",
)
