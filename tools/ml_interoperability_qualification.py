#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Qualify the machine-learning interoperability gates G1-G24.

Every gate is witnessed by the pytest scenarios named ``test_g<N>_...`` in
``tests/integration/test_ml_interoperability_qualification.py``. The runner
executes the selected gate scenarios in-process, then binds each gate's outcome
into one causal record chain: support tuple, quantitative criterion
(zero unqualified scenarios), campaign start, raw observation, campaign
observation, and scientific qualification evidence. Records use a logical clock
and content addresses only, so an unchanged build, environment, and outcome set
reproduce a byte-identical report.

Usage, from the repository root::

    PYTHONPATH=. python tools/ml_interoperability_qualification.py \\
        --workers 8 --output report.json

``--gate G<N>`` (repeatable) restricts the run to a gate subset; unselected
gates are reported as missing and the report outcome is then ``inconclusive``.
The report is printed when ``--output`` is omitted. The exit status is nonzero
when any selected gate failed or was inconclusive.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    QualificationCriterion,
    QualificationEvidence,
    SupportTuple,
    validate_qualification_causality,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = "tests/integration/test_ml_interoperability_qualification.py"
CAPABILITY = "ml-interoperability.qualification-gate"
APPROVAL_ID = "ml-interoperability-final-plan"
REVIEWER_ID = "ml-interoperability-runner"
ENVIRONMENT_PACKAGES = ("equinox", "jax", "jaxlib", "numpy", "optax")

# Logical clock: the report carries causal order, never wall time.
CRITERION_TICK = 1
STARTED_TICK = 2
OBSERVED_TICK = 3
ISSUED_TICK = 4
EXPIRES_TICK = 5

_SCENARIO_NAME = re.compile(r"test_g([1-9][0-9]*)_")
_SCENARIO_OUTCOMES = ("passed", "failed", "skipped")
_MESSAGE_LIMIT = 240


@dataclass(frozen=True)
class QualificationGate:
    gate_id: str
    title: str
    assertion: str


GATES = (
    QualificationGate(
        "G1",
        "Statistical closure → PDE",
        "A fitted closure keeps its ports and binds frozen into a PDE; an explicit "
        "parameter subspace trains a coefficient subset; a mismatched port mapping "
        "is rejected before execution.",
    ),
    QualificationGate(
        "G2",
        "Learned constitutive → implicit mechanics",
        "Native Newton owns acceptance; the implicit parameter gradient matches "
        "finite differences; an invalid tangent poisons the derivative; a law "
        "without classical C1 regularity is refused.",
    ),
    QualificationGate(
        "G3",
        "Learned preconditioner → Krylov",
        "Trained through a fixed-work `AlgorithmicWorkObjective`, the "
        "preconditioner reduces Krylov work without changing answers; outside its "
        "support the solve cannot report success.",
    ),
    QualificationGate(
        "G4",
        "Neural dynamics → control",
        "The same bound model drives continuous and fixed-step discrete dynamics; "
        "a neural feedback policy's rollout gradient matches finite differences "
        "and trains; MPC sensitivity through the learned linearization matches "
        "finite differences.",
    ),
    QualificationGate(
        "G5",
        "Neural operator → native corrector",
        "Proposal and corrected state stay separately inspectable (including as "
        "field views); the native corrector re-evaluates its own residual; "
        "corrected answers do not depend on proposal quality.",
    ),
    QualificationGate(
        "G6",
        "External host refusal",
        "Host-only models run eagerly; jit, vmap, grad, jvp and vjp are refused "
        "before invocation; no derivative-free method is chosen silently.",
    ),
    QualificationGate(
        "G7",
        "Stateful functional model",
        "A rejected update commits nothing; an accepted update commits parameters "
        "and model state together and checkpoints restore both; evaluation does "
        "not mutate state.",
    ),
    QualificationGate(
        "G8",
        "Plan-embedded component",
        "A closure inside prepared finite-volume dynamics trains (directly and "
        "through a rollout objective); fixed data beside it does not.",
    ),
    QualificationGate(
        "G9",
        "Representation swap",
        "One face closure serves Cartesian, mapped and unstructured finite-volume "
        "owners and finite-element facet traces read through a field view; "
        "mismatched ports are rejected.",
    ),
    QualificationGate(
        "G10",
        "Silent-zero accelerator",
        "An ACCELERATOR under a SOLUTION_MAP-only objective is refused with "
        "`ValueError` before tracing; a fixed-work objective gives it a nonzero "
        "gradient.",
    ),
    QualificationGate(
        "G11",
        "Regularity mismatch",
        "A ReLU→linear PINN Laplacian is rejected at planning; ReLU→tanh is "
        "admitted only under an explicit almost-everywhere policy.",
    ),
    QualificationGate(
        "G12",
        "In-situ rollback",
        "A physical rejection restores parameters, model state, optimizer state, "
        "targets and cursors bitwise.",
    ),
    QualificationGate(
        "G13",
        "Staged external adjoint",
        "The staged VJP matches the all-JAX reference and finite differences; a "
        "replay mismatch is refused before the provider runs.",
    ),
    QualificationGate(
        "G14",
        "Ensemble lanes",
        "Members keep FIXED normalizers on the member lane; serial equals vmap; "
        "lane training moves parameters only.",
    ),
    QualificationGate(
        "G15",
        "Precision floor",
        "An undeclared float32 residual component is refused; a declared floor "
        "derives the achievable tolerance; accelerators do not raise the floor.",
    ),
    QualificationGate(
        "G16",
        "Frozen randomness",
        "Resampled randomness is refused on certified root maps; a frozen "
        "realization reproduces primal and derivative.",
    ),
    QualificationGate(
        "G17",
        "Axis identity",
        "Equal extents never create axis identity; raw outputs are validated only "
        "against their declaration; equal-extent ports do not share identity.",
    ),
    QualificationGate(
        "G18",
        "Closure geometry",
        "ALE closures receive exact unit normals, face measures and grid-normal "
        "velocities; Cartesian owners keep parity with the baseline.",
    ),
    QualificationGate(
        "G19",
        "Differentiable MPC",
        "The closed-loop gradient through active bounds matches finite differences "
        "and improves performance; one weak window refuses the complete "
        "derivative.",
    ),
    QualificationGate(
        "G20",
        "Mixed authority",
        "Each authority group trains only from a compatible contribution: rollout "
        "signal for DISCRETIZATION, fixed-work signal for ACCELERATOR.",
    ),
    QualificationGate(
        "G21",
        "Replay derivative",
        "DEM and reactive replay mismatches invalidate cotangents while preserving "
        "primal and replay evidence.",
    ),
    QualificationGate(
        "G22",
        "Identity stability",
        "A dynamic weight update changes the numeric revision, not the executable "
        "identity; revisions are verified on load; statically held weights change "
        "the executable signature.",
    ),
    QualificationGate(
        "G23",
        "Field validity",
        "C0 facet gradients are refused without a trace side; explicit trace sides "
        "define facet gradients; degenerate second derivatives are refused.",
    ),
    QualificationGate(
        "G24",
        "Coupled training policy",
        "`CoupledTrainingPolicy` decides whether a valid physical step survives a "
        "training rejection.",
    ),
)
_GATE_IDS = tuple(gate.gate_id for gate in GATES)


@dataclass(frozen=True)
class ScenarioOutcome:
    """Consumer-visible outcome of one gate-suite pytest node."""

    nodeid: str
    outcome: str
    message: str


@dataclass(frozen=True)
class ScenarioRun:
    """Every observed gate-suite node, plus whether suite collection failed."""

    scenarios: tuple[ScenarioOutcome, ...]
    collection_failed: bool


@dataclass(frozen=True)
class _Campaign:
    campaign_spec_id: str
    replay_id: str
    build_id: str
    environment_id: str
    backend: str
    precision: str


def scenario_gate(nodeid: str, /) -> str:
    """Return the gate ID owning one gate-suite pytest node ID."""
    path, separator, name = nodeid.partition("::")
    if path != SUITE_PATH or not separator:
        raise ValueError(f"{nodeid!r} is not a node of the gate suite {SUITE_PATH}.")
    match = _SCENARIO_NAME.match(name)
    if match is None:
        raise ValueError(f"{nodeid!r} does not name a test_g<N>_ gate scenario.")
    gate_id = f"G{match.group(1)}"
    if gate_id not in _GATE_IDS:
        raise ValueError(f"{nodeid!r} names gate {gate_id}, outside G1..G24.")
    return gate_id


def selected_gates(gate_ids: Sequence[str] | None, /) -> tuple[str, ...]:
    """Return a canonical G1..G24-ordered selection; ``None`` selects every gate."""
    if gate_ids is None:
        return _GATE_IDS
    unknown = sorted(set(gate_ids) - set(_GATE_IDS))
    if unknown:
        raise ValueError(f"Unknown qualification gates: {', '.join(unknown)}.")
    if len(set(gate_ids)) != len(gate_ids):
        raise ValueError("Qualification gates must be selected at most once.")
    if not gate_ids:
        raise ValueError("At least one qualification gate must be selected.")
    return tuple(gate_id for gate_id in _GATE_IDS if gate_id in gate_ids)


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


class _ScenarioCollector:
    """Pytest plugin reducing setup/call/teardown reports to one outcome per node.

    Distributed workers forward their reports to the controller's hooks, so the
    same plugin observes serial and parallel runs.
    """

    def __init__(self) -> None:
        self.phases: dict[str, dict[str, pytest.TestReport]] = {}
        self.collection_failed = False

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.phases.setdefault(report.nodeid, {})[report.when] = report

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collection_failed = True

    def outcomes(self) -> tuple[ScenarioOutcome, ...]:
        return tuple(
            _node_outcome(nodeid, phases) for nodeid, phases in self.phases.items()
        )


def _node_outcome(
    nodeid: str, phases: Mapping[str, pytest.TestReport], /
) -> ScenarioOutcome:
    ordered = [phases[when] for when in ("setup", "call", "teardown") if when in phases]
    failed = [report for report in ordered if report.failed]
    if failed:
        return ScenarioOutcome(nodeid, "failed", _short_message(failed[0]))
    skipped = [report for report in ordered if report.skipped]
    if skipped:
        return ScenarioOutcome(nodeid, "skipped", _short_message(skipped[0]))
    if "call" in phases and phases["call"].passed:
        return ScenarioOutcome(nodeid, "passed", "")
    return ScenarioOutcome(nodeid, "failed", "scenario produced no call phase")


def run_gate_scenarios(gate_ids: Sequence[str], /, *, workers: int) -> ScenarioRun:
    """Execute the selected gate scenarios in-process and collect their outcomes."""
    gates = selected_gates(gate_ids)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    collector = _ScenarioCollector()
    keyword = " or ".join(f"test_g{gate_id[1:]}_" for gate_id in gates)
    arguments = [
        str(_PROJECT_ROOT / SUITE_PATH),
        f"--rootdir={_PROJECT_ROOT}",
        "-q",
        "-p",
        "no:cacheprovider",
        "-k",
        keyword,
    ]
    if workers > 1:
        arguments += ["-n", str(workers)]
    # Pytest progress goes to stderr so a printed report remains the only stdout.
    with contextlib.redirect_stdout(sys.stderr):
        exit_code = pytest.main(arguments, plugins=[collector])
    aborted = exit_code in (
        pytest.ExitCode.INTERRUPTED,
        pytest.ExitCode.INTERNAL_ERROR,
        pytest.ExitCode.USAGE_ERROR,
    )
    return ScenarioRun(
        scenarios=collector.outcomes(),
        collection_failed=collector.collection_failed or aborted,
    )


def source_build_id(root: Path = _PROJECT_ROOT, /) -> str:
    """Content-address the package sources, the gate suite, and this runner."""
    paths = [*sorted((root / "phydrax").rglob("*.py")), root / SUITE_PATH]
    paths.append(root / "tools" / "ml_interoperability_qualification.py")
    return canonical_fingerprint(
        sorted(
            (
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in paths
            ),
            key=lambda entry: entry["path"],
        )
    )


def runtime_environment() -> dict[str, object]:
    """Return the interpreter, platform, and numerical-stack identity."""
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name) for name in ENVIRONMENT_PACKAGES
        },
    }


def _classify(
    scenarios: Sequence[ScenarioOutcome], collection_failed: bool, /
) -> tuple[str, str, int]:
    unqualified = sum(scenario.outcome != "passed" for scenario in scenarios)
    if collection_failed:
        return "inconclusive", "suite-collection-failed", unqualified
    if not scenarios:
        return "failed", "no-gate-scenario", unqualified
    if unqualified:
        return "failed", "unqualified-scenarios", unqualified
    return "passed", "all-gate-scenarios-passed", unqualified


def _gate_record(
    gate: QualificationGate,
    scenarios: Sequence[ScenarioOutcome],
    collection_failed: bool,
    campaign: _Campaign,
    /,
) -> dict[str, object]:
    support = SupportTuple(
        CAPABILITY,
        {
            "gate": gate.gate_id,
            "scenario-suite": SUITE_PATH,
            "backend": campaign.backend,
            "precision": campaign.precision,
        },
    )
    criterion = QualificationCriterion(
        support_tuple_id=support.support_tuple_id,
        metric="unqualified-scenarios",
        unit="count",
        comparison="equal",
        target=0,
        aggregation="sum",
        uncertainty="deterministic",
        applicability=gate.gate_id,
        approval_id=APPROVAL_ID,
        issued_at=CRITERION_TICK,
    )
    start = CampaignStartRecord(
        campaign_spec_id=campaign.campaign_spec_id,
        criterion_id=criterion.criterion_id,
        resolved_run_spec_id=campaign.replay_id,
        support_tuple_id=support.support_tuple_id,
        started_at=STARTED_TICK,
    )
    outcome, reason, unqualified = _classify(scenarios, collection_failed)
    raw = {
        "kind": "ml-interoperability-gate-observation",
        "gate": gate.gate_id,
        "scenarios": [
            {
                "nodeid": scenario.nodeid,
                "outcome": scenario.outcome,
                "message": scenario.message,
            }
            for scenario in sorted(scenarios, key=lambda scenario: scenario.nodeid)
        ],
        "unqualified_scenarios": unqualified,
    }
    raw_artifact_id = canonical_fingerprint(raw)
    observation = CampaignObservationRecord(
        start_record_id=start.start_record_id,
        campaign_spec_id=campaign.campaign_spec_id,
        criterion_id=criterion.criterion_id,
        resolved_run_spec_id=campaign.replay_id,
        support_tuple_id=support.support_tuple_id,
        raw_artifact_ids=(raw_artifact_id,),
        observed_at=OBSERVED_TICK,
    )
    evidence = QualificationEvidence(
        "scientific",
        outcome,
        (support.support_tuple_id,),
        build_id=campaign.build_id,
        environment_id=campaign.environment_id,
        backend=campaign.backend,
        topology="in-process-pytest",
        precision=campaign.precision,
        reduction="deterministic:sum",
        replay_id=campaign.replay_id,
        criteria_ids=(criterion.criterion_id,),
        raw_artifact_ids=(raw_artifact_id,),
        campaign_start_record_ids=(start.start_record_id,),
        campaign_observation_record_ids=(observation.observation_record_id,),
        reviewer_id=REVIEWER_ID,
        issued_at=ISSUED_TICK,
        expires_at=EXPIRES_TICK,
        reason=reason,
        requalification_triggers=(
            "build-change",
            "environment-change",
            "gate-suite-change",
        ),
    )
    validate_qualification_causality(criterion, start, observation, evidence)
    return {
        "gate": gate.gate_id,
        "title": gate.title,
        "assertion": gate.assertion,
        "support_tuple": support.to_record(),
        "criterion": criterion.to_record(),
        "campaign_start": start.to_record(),
        "raw_output": {**raw, "raw_artifact_id": raw_artifact_id},
        "campaign_observation": observation.to_record(),
        "evidence": evidence.to_record(),
    }


def qualification_report(
    run: ScenarioRun,
    gate_ids: Sequence[str] | None,
    /,
    *,
    build_id: str,
    environment: Mapping[str, object],
    backend: str,
    precision: str,
) -> dict[str, object]:
    """Bind observed scenario outcomes into one causal record chain per gate."""
    gates = selected_gates(gate_ids)
    by_gate: dict[str, list[ScenarioOutcome]] = {gate_id: [] for gate_id in gates}
    nodeids = [scenario.nodeid for scenario in run.scenarios]
    if len(set(nodeids)) != len(nodeids):
        raise ValueError("Each gate scenario must be observed exactly once.")
    for scenario in run.scenarios:
        if scenario.outcome not in _SCENARIO_OUTCOMES:
            raise ValueError(f"Unsupported scenario outcome {scenario.outcome!r}.")
        gate_id = scenario_gate(scenario.nodeid)
        if gate_id not in by_gate:
            raise ValueError(
                f"Scenario {scenario.nodeid!r} belongs to unselected {gate_id}."
            )
        by_gate[gate_id].append(scenario)
    environment_record = dict(environment)
    environment_id = canonical_fingerprint(environment_record)
    campaign = _Campaign(
        campaign_spec_id=canonical_fingerprint(
            {
                "kind": "ml-interoperability-qualification-campaign",
                "gates": list(gates),
                "suite": SUITE_PATH,
            }
        ),
        replay_id=canonical_fingerprint({"suite": SUITE_PATH, "gates": list(gates)}),
        build_id=build_id,
        environment_id=environment_id,
        backend=backend,
        precision=precision,
    )
    records = [
        _gate_record(gate, by_gate[gate.gate_id], run.collection_failed, campaign)
        for gate in GATES
        if gate.gate_id in by_gate
    ]
    outcomes = {record["gate"]: record["evidence"]["outcome"] for record in records}
    passed = [gate_id for gate_id in gates if outcomes[gate_id] == "passed"]
    failed = [gate_id for gate_id in gates if outcomes[gate_id] == "failed"]
    inconclusive = [gate_id for gate_id in gates if outcomes[gate_id] == "inconclusive"]
    missing = [gate_id for gate_id in _GATE_IDS if gate_id not in by_gate]
    if failed:
        outcome = "failed"
    elif missing or inconclusive:
        outcome = "inconclusive"
    else:
        outcome = "passed"
    return {
        "kind": "ml-interoperability-qualification-report",
        "clock": "logical",
        "build_id": build_id,
        "environment": {**environment_record, "environment_id": environment_id},
        "replay_id": campaign.replay_id,
        "gates": records,
        "passed_gates": passed,
        "failed_gates": failed,
        "inconclusive_gates": inconclusive,
        "missing_gates": missing,
        "outcome": outcome,
    }


def serialize_report(report: Mapping[str, object], /) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _write_atomic(path: Path, payload: str, /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
    # Temporary files are owner-only; the report is an ordinary shared artifact.
    os.chmod(handle.name, 0o644)
    os.replace(handle.name, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--gate", action="append", choices=_GATE_IDS, dest="gates", metavar="G<N>"
    )
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    gates = selected_gates(arguments.gates)
    run = run_gate_scenarios(gates, workers=arguments.workers)
    report = qualification_report(
        run,
        gates,
        build_id=source_build_id(),
        environment=runtime_environment(),
        backend=jax.default_backend(),
        precision="float64" if jax.config.jax_enable_x64 else "float32",
    )
    payload = serialize_report(report)
    if arguments.output is None:
        print(payload, end="")
    else:
        _write_atomic(arguments.output, payload)
    if report["failed_gates"] or report["inconclusive_gates"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
