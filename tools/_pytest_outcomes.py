#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Pytest sessions reduced to one consumer-visible outcome per collected node.

Qualification runners execute pytest selections in-process, or in a fresh
interpreter when the startup environment must differ (forced host devices).
Both routes use the same plugin, so a node outcome means the same thing either
way: the first failed setup/call/teardown phase fails the node, otherwise a
skipped phase skips it, and only a passed call phase passes it.

The fresh-interpreter route runs this module::

    python -m tools._pytest_outcomes --root <rootdir> --workers 4 -- <pytest arguments>

and prints the observed run as JSON on stdout; pytest progress goes to stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODE_OUTCOMES = ("passed", "failed", "skipped")
_MESSAGE_LIMIT = 240
_ABORTED = (
    pytest.ExitCode.INTERRUPTED,
    pytest.ExitCode.INTERNAL_ERROR,
    pytest.ExitCode.USAGE_ERROR,
)


@dataclass(frozen=True)
class NodeOutcome:
    """Consumer-visible outcome and host duration of one pytest node."""

    nodeid: str
    outcome: str
    message: str
    duration_seconds: float


@dataclass(frozen=True)
class PytestRun:
    """Every observed node of one session, plus whether collection failed."""

    nodes: tuple[NodeOutcome, ...]
    collection_failed: bool

    def to_record(self) -> dict[str, object]:
        return {
            "nodes": [
                {
                    "nodeid": node.nodeid,
                    "outcome": node.outcome,
                    "message": node.message,
                    "duration_seconds": node.duration_seconds,
                }
                for node in self.nodes
            ],
            "collection_failed": self.collection_failed,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> PytestRun:
        nodes = record["nodes"]
        collection_failed = record["collection_failed"]
        if not isinstance(nodes, list) or not isinstance(collection_failed, bool):
            raise ValueError("A pytest run record needs a node list and a flag.")
        outcomes = []
        for node in nodes:
            if not isinstance(node, dict) or node.get("outcome") not in NODE_OUTCOMES:
                raise ValueError(f"Invalid pytest node record {node!r}.")
            outcomes.append(
                NodeOutcome(
                    str(node["nodeid"]),
                    str(node["outcome"]),
                    str(node["message"]),
                    float(node["duration_seconds"]),
                )
            )
        return cls(tuple(outcomes), collection_failed)


@dataclass(frozen=True)
class PytestCollection:
    """Node IDs a collection produced, the collectors that failed, and its exit code.

    An aborted collection (interrupted, internal error, usage error such as a
    conftest import failure) may have stopped before reaching any node, so an
    absent node is not evidence that it does not exist.
    """

    nodeids: tuple[str, ...]
    failed_collectors: tuple[str, ...]
    exit_code: int

    @property
    def aborted(self) -> bool:
        """Whether pytest stopped before collecting every requested path."""
        return self.exit_code in _ABORTED


def _short_message(report: pytest.TestReport | pytest.CollectReport, /) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple):
        # Skip reports carry a (path, line, reason) location triple.
        text = str(longrepr[2])
    else:
        crash = getattr(longrepr, "reprcrash", None)
        if crash is not None:
            text = crash.message
        else:
            lines = [line for line in report.longreprtext.splitlines() if line.strip()]
            text = lines[-1] if lines else ""
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line[:_MESSAGE_LIMIT]


def _node_outcome(nodeid: str, phases: Mapping[str, pytest.TestReport], /) -> NodeOutcome:
    ordered = [phases[when] for when in ("setup", "call", "teardown") if when in phases]
    duration = sum(report.duration for report in ordered)
    failed = [report for report in ordered if report.failed]
    if failed:
        return NodeOutcome(nodeid, "failed", _short_message(failed[0]), duration)
    skipped = [report for report in ordered if report.skipped]
    if skipped:
        return NodeOutcome(nodeid, "skipped", _short_message(skipped[0]), duration)
    if "call" in phases and phases["call"].passed:
        return NodeOutcome(nodeid, "passed", "", duration)
    return NodeOutcome(nodeid, "failed", "scenario produced no call phase", duration)


class _NodeCollector:
    """Pytest plugin reducing setup/call/teardown reports to one outcome per node.

    Distributed workers forward their reports to the controller's hooks, so the
    same plugin observes serial and parallel runs.
    """

    def __init__(self) -> None:
        self.phases: dict[str, dict[str, pytest.TestReport]] = {}
        self.collected: tuple[str, ...] = ()
        self.failed_collectors: list[str] = []

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # ty: ignore[invalid-assignment]
        self.phases.setdefault(report.nodeid, {})[report.when] = report

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.failed_collectors.append(report.nodeid)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.collected = tuple(item.nodeid for item in session.items)

    def outcomes(self) -> tuple[NodeOutcome, ...]:
        return tuple(
            _node_outcome(nodeid, phases) for nodeid, phases in self.phases.items()
        )


def _session(
    arguments: Sequence[str], root: Path, collector: _NodeCollector, /
) -> pytest.ExitCode | int:
    options = [
        *arguments,
        f"--rootdir={root}",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    # Pytest progress goes to stderr so a printed report remains the only stdout.
    with contextlib.redirect_stdout(sys.stderr):
        return pytest.main(options, plugins=[collector])


def run_pytest(arguments: Sequence[str], /, *, root: Path, workers: int) -> PytestRun:
    """Run one pytest selection in-process and reduce it to node outcomes."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    collector = _NodeCollector()
    options = [*arguments, *(("-n", str(workers)) if workers > 1 else ())]
    exit_code = _session(options, root, collector)
    return PytestRun(
        nodes=collector.outcomes(),
        collection_failed=bool(collector.failed_collectors) or exit_code in _ABORTED,
    )


def collect_pytest_nodes(paths: Sequence[str], /, *, root: Path) -> PytestCollection:
    """Collect the node IDs of test paths in-process without running them."""
    collector = _NodeCollector()
    exit_code = _session(
        [*paths, "--collect-only", "-q", "--continue-on-collection-errors"],
        root,
        collector,
    )
    return PytestCollection(
        nodeids=collector.collected,
        failed_collectors=tuple(sorted(set(collector.failed_collectors))),
        exit_code=int(exit_code),
    )


def _stop_process_group(process: subprocess.Popen[str], /) -> None:
    # pytest-xdist workers share the child's session; stopping the group leaves
    # no orphaned worker behind a missed deadline or an interrupted parent.
    # Darwin reports EPERM instead of ESRCH when only an unreaped zombie
    # remains in the group.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


def run_pytest_subprocess(
    arguments: Sequence[str],
    /,
    *,
    root: Path,
    workers: int,
    environment: Mapping[str, str],
    timeout: float | None = None,
) -> PytestRun:
    """Run one pytest selection in a fresh interpreter with extra startup variables.

    A crashed or unparseable child is reported as a failed collection with no
    observed nodes, never as passing evidence. With ``timeout`` (seconds), a
    child still running at the deadline is killed with its workers and likewise
    reported as a failed collection with no observed nodes.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    if timeout is not None and not (math.isfinite(timeout) and timeout > 0.0):
        raise ValueError("timeout must be positive and finite.")
    python_path = os.pathsep.join(
        entry for entry in (str(_PROJECT_ROOT), os.environ.get("PYTHONPATH", "")) if entry
    )
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tools._pytest_outcomes",
            "--root",
            str(root),
            "--workers",
            str(workers),
            "--",
            *arguments,
        ],
        cwd=root,
        env={**os.environ, **environment, "PYTHONPATH": python_path},
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # A missed deadline is an explicit inconclusive outcome, never evidence.
            stdout = None
        finally:
            _stop_process_group(process)
    if stdout is None:
        return PytestRun(nodes=(), collection_failed=True)
    # The record is the child's last stdout line; anything before it is not evidence.
    lines = stdout.strip().splitlines()
    try:
        return PytestRun.from_record(json.loads(lines[-1]))
    except (IndexError, ValueError, KeyError, TypeError):
        return PytestRun(nodes=(), collection_failed=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=_PROJECT_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    arguments = list(options.arguments)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    run = run_pytest(arguments, root=options.root, workers=options.workers)
    print(json.dumps(run.to_record(), sort_keys=True))


if __name__ == "__main__":
    main()
