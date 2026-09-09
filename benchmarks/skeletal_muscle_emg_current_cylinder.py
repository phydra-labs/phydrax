#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx

from benchmarks._runtime import capture_environment
from tools.skeletal_muscle_emg_current_cylinder_qualification import (
    make_conductor,
    make_source,
)


def case(fibers: int, nodes: int, angular: int, axial: int) -> dict[str, object]:
    start = time.perf_counter()
    _, source, snapshot, accepted = make_source(nodes, fibers)
    accepted.transmembrane_current_A.block_until_ready()
    source_prepare_ms = 1000 * (time.perf_counter() - start)
    start = time.perf_counter()
    conductor = make_conductor(source, angular=angular, axial=axial)
    conductor.contact_lead_field_ohm.block_until_ready()
    conductor_prepare_ms = 1000 * (time.perf_counter() - start)
    prior_source, prior_conductor = source.initialize(), conductor.initialize()

    @eqx.filter_jit
    def observe(source_state, conductor_state, fiber_state):
        candidate = source.propose(
            source_state,
            fiber_state,
            fiber_prepared_id=source.fiber_prepared_id,
            geometry_id=source.plan.geometry_id,
        )
        committed = candidate.commit(source_state, fiber_state)
        observation = conductor.propose(conductor_state, committed)
        return observation.commit(
            conductor_state, committed
        ), observation.evidence.successful

    start = time.perf_counter()
    first, ok = observe(prior_source, prior_conductor, snapshot)
    first.lead_voltage_V.block_until_ready()
    compile_and_first_ms = 1000 * (time.perf_counter() - start)
    repetitions = 20
    start = time.perf_counter()
    for _ in range(repetitions):
        result, ok = observe(prior_source, prior_conductor, snapshot)
        result.lead_voltage_V.block_until_ready()
    runtime_ms = 1000 * (time.perf_counter() - start) / repetitions
    return {
        "fiber_count": fibers,
        "nodes_per_fiber": nodes,
        "contact_count": 2,
        "lead_count": 1,
        "angular_modes": angular,
        "positive_longitudinal_modes": axial,
        "source_prepare_ms": source_prepare_ms,
        "conductor_prepare_ms": conductor_prepare_ms,
        "compile_and_first_ms": compile_and_first_ms,
        "observation_transaction_ms": runtime_ms,
        "lead_field_storage_bytes": conductor.contact_lead_field_ohm.nbytes,
        "radial_transfer_storage_bytes": conductor.radial_transfer_ohm_m.nbytes,
        "dtype": str(conductor.contact_lead_field_ohm.dtype),
        "successful": bool(ok),
        "claim_scope": "manufactured fixed straight fibers; preparation separate from observation, no anatomy claim",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/skeletal_muscle_emg_current_cylinder.json"),
    )
    args = parser.parse_args()
    result = case(1, 17, 2, 3) if args.smoke else case(128, 65, 8, 16)
    payload = {
        "environment": capture_environment().to_dict(),
        "case": result,
        "all_successful": result["successful"],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
