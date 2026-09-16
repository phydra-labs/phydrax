#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run owner qualification and benchmark components without masking failures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_QUALIFICATION_COMPONENTS = (
    "tools/cm_periodic_qualification.py",
    "tools/cm_periodic_electronic_qualification.py",
    "tools/lattice_dynamics_qualification.py",
    "tools/cm_quantum_lattice_qualification.py",
    "tools/green_embedding_smoke.py",
    "tools/cm_transport_qualification.py",
    "tools/cm_magnetism_superconductivity.py",
    "tools/cm_spectroscopy_smoke.py",
    "tools/magnetic_resonance_smoke.py",
    "tools/cm_soft_matter_smoke.py",
    "tools/cm_frontier_field_smoke.py",
    "tools/cm_semiconductor_detector_qualification.py",
)
_BENCHMARK_COMPONENTS = (
    "benchmarks/cm_periodic_operators.py",
    "benchmarks/cm_periodic_electronic.py",
    "benchmarks/cm_lattice_dynamics.py",
    "benchmarks/cm_quantum_lattice.py",
    "benchmarks/cm_quantum_thermal_response.py",
    "benchmarks/cm_green_embedding.py",
    "benchmarks/cm_transport.py",
    "benchmarks/cm_magnetism_superconductivity.py",
    "benchmarks/cm_spectroscopy.py",
    "benchmarks/cm_magnetic_resonance.py",
    "benchmarks/cm_soft_matter.py",
    "benchmarks/cm_frontier_field.py",
    "benchmarks/cm_semiconductor_detector.py",
)


def _components(group: str, /) -> tuple[str, ...]:
    if group == "qualification":
        return _QUALIFICATION_COMPONENTS
    if group == "benchmark":
        return _BENCHMARK_COMPONENTS
    if group == "all":
        return _QUALIFICATION_COMPONENTS + _BENCHMARK_COMPONENTS
    raise ValueError("group must be 'all', 'qualification', or 'benchmark'.")


def run(group: str = "all", /) -> dict[str, object]:
    """Run every selected owner command and retain each exact return code."""

    results = []
    for relative in _components(group):
        completed = subprocess.run(
            (sys.executable, str(_ROOT / relative)),
            cwd=_ROOT,
            check=False,
        )
        results.append(
            {
                "component": relative,
                "returncode": completed.returncode,
                "passed": completed.returncode == 0,
            }
        )
    return {
        "kind": "condensed-matter-production-closure-orchestration",
        "group": group,
        "passed": all(result["passed"] for result in results),
        "components": results,
        "release_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=("all", "qualification", "benchmark"),
        default="all",
    )
    arguments = parser.parse_args()
    result = run(arguments.group)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
