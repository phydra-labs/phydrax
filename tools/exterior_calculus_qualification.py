#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run the bounded exterior-calculus IR and physics qualification scenarios.

A skipped, missing, or failed test never qualifies a scenario. This artifact is
observed numerical/semantic evidence, not a release authorization.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from benchmarks._io import write_json_atomic
from tools._pytest_outcomes import NodeOutcome, PytestRun, run_pytest_subprocess


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = _PROJECT_ROOT / "benchmarks/exterior_calculus_qualification.json"
CAPABILITY = "platform.exterior-calculus"


@dataclass(frozen=True, slots=True)
class Scenario:
    """One bounded consumer contract backed by actual pytest nodes."""

    scenario_id: str
    node_ids: tuple[str, ...]


SCENARIOS = (
    Scenario(
        "form-degree-and-twist",
        ("tests/unit/equations/test_exterior_ir.py::test_form_degree_and_twist_typing",),
    ),
    Scenario(
        "explicit-proxy-boundaries",
        (
            "tests/unit/equations/test_exterior_ir.py::test_form_proxy_crossing_is_refused",
        ),
    ),
    Scenario(
        "form-serialization",
        ("tests/unit/equations/test_exterior_ir.py::test_form_serialization_roundtrip",),
    ),
    Scenario(
        "smooth-polynomial-lowering",
        (
            "tests/unit/equations/test_exterior_ir.py::test_smooth_exterior_lowering_matches_polynomial",
        ),
    ),
    Scenario(
        "native-operator-lowering",
        (
            "tests/unit/equations/test_exterior_compile.py::test_exterior_lowering_matches_native_operators",
        ),
    ),
    Scenario(
        "prepared-dynamic-reuse",
        (
            "tests/unit/equations/test_exterior_compile.py::test_exterior_prepared_dynamic_reuse",
        ),
    ),
    Scenario(
        "relative-boundary-complex",
        (
            "tests/unit/equations/test_exterior_compile.py::test_exterior_relative_boundary_matches_restricted_complex",
        ),
    ),
    Scenario(
        "exterior-products-and-cartan",
        (
            "tests/unit/equations/test_exterior_compile.py::test_exterior_products_interior_and_lie_have_analytic_chain_values",
        ),
    ),
    Scenario(
        "matrix-fiber-wedge",
        (
            "tests/unit/equations/test_exterior_compile.py::test_exterior_matrix_wedge_preserves_noncommutative_fibers",
        ),
    ),
    Scenario(
        "boundary-trace-and-stokes",
        (
            "tests/unit/equations/test_exterior_compile.py::test_exterior_trace_lowers_boundary_derivative_and_stokes",
        ),
    ),
    Scenario(
        "scientific-binding-refusals",
        (
            "tests/unit/equations/test_exterior_compile.py::test_exterior_binding_refuses_equal_shape_wrong_identity_and_missing_geometry",
        ),
    ),
    Scenario(
        "hodge-laplace-convergence",
        ("tests/unit/solver/test_hodge_laplace.py::test_hodge_plan_convergence",),
    ),
    Scenario(
        "maxwell-cavity-spectrum",
        ("tests/unit/solver/test_hodge_laplace.py::test_cavity_spectrum",),
    ),
    Scenario(
        "learned-hodge-conservation",
        ("tests/unit/solver/test_hodge_laplace.py::test_learned_hodge_conservation",),
    ),
)


def _matches(reference: str, observed: str, /) -> bool:
    return observed == reference or observed.startswith(reference + "[")


def _scenario_record(scenario: Scenario, observation: PytestRun, /) -> dict[str, object]:
    outcomes = tuple(
        node
        for node in observation.nodes
        if any(_matches(reference, node.nodeid) for reference in scenario.node_ids)
    )
    missing = tuple(
        reference
        for reference in scenario.node_ids
        if not any(_matches(reference, node.nodeid) for node in outcomes)
    )
    return {
        "scenario_id": scenario.scenario_id,
        "references": list(scenario.node_ids),
        "missing_references": list(missing),
        "nodes": [_node_record(node) for node in outcomes],
        "passed": not observation.collection_failed
        and not missing
        and all(node.outcome == "passed" for node in outcomes),
    }


def _node_record(node: NodeOutcome, /) -> dict[str, object]:
    return {
        "nodeid": node.nodeid,
        "outcome": node.outcome,
        "message": node.message,
        "duration_seconds": node.duration_seconds,
    }


def run(
    scenario_ids: tuple[str, ...] = (),
    /,
    *,
    workers: int = 1,
) -> dict[str, object]:
    """Execute selected nodes once in an x64-enabled fresh interpreter."""
    available = {scenario.scenario_id for scenario in SCENARIOS}
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Scenario selections must be unique.")
    unknown = set(scenario_ids) - available
    if unknown:
        raise ValueError(f"Unknown exterior-calculus scenarios: {sorted(unknown)}.")
    selected = tuple(
        scenario
        for scenario in SCENARIOS
        if not scenario_ids or scenario.scenario_id in scenario_ids
    )
    references = sorted({node for scenario in selected for node in scenario.node_ids})
    observation = run_pytest_subprocess(
        [f"{_PROJECT_ROOT.as_posix()}/{reference}" for reference in references],
        root=_PROJECT_ROOT,
        workers=workers,
        environment={"JAX_ENABLE_X64": "1"},
    )
    node_ids = tuple(node.nodeid for node in observation.nodes)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Each qualification node must be observed exactly once.")
    cases = [_scenario_record(scenario, observation) for scenario in selected]
    return {
        "capability": CAPABILITY,
        "owner": "phydrax.exterior",
        "documentation": "docs/guides_exterior_calculus.md",
        "release_authorized": False,
        "collection_failed": observation.collection_failed,
        "workers": workers,
        "cases": cases,
        "passed": not observation.collection_failed
        and all(bool(case["passed"]) for case in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", action="append", choices=tuple(s.scenario_id for s in SCENARIOS)
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    report = run(tuple(arguments.scenario or ()), workers=arguments.workers)
    write_json_atomic(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
