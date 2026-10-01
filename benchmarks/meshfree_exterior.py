# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded conservative graph metric preparation, refresh, action and adjoint scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._runtime import logical_array_bytes, measure_synchronized
from benchmarks.meshfree_scaling import (
    add_config_arguments,
    apply_baseline,
    config_from_arguments,
    execution_evidence,
    make_record,
    measured_phase,
    MeshfreeConfig,
    unavailable_phases,
)


def measure_capacity(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    from examples.meshfree_conservative_diffusion import exterior_plan
    from phydrax.discretization.meshfree import MeshfreeEdgeRelationPlan

    reservation = config.check_capacity(capacity)
    pair_budget = capacity * config.neighbors * 4
    symbolic_budget = config.working_set_bytes // 64
    if pair_budget * 64 > config.working_set_bytes:
        raise ValueError("Requested exterior edge capacity exceeds working-set budget.")
    plan = exterior_plan(
        size=capacity,
        dimension=config.dimension,
        seed=seed,
        maximum_pairs=pair_budget,
        maximum_symbolic_entries=symbolic_budget,
    )
    edges, neighbor_seconds = measure_synchronized(
        lambda: MeshfreeEdgeRelationPlan(
            plan.points,
            plan.radius,
            pair_budget,
            active=plan.active,
            target_chunk_size=config.chunk_rows,
        ).prepare()
    )
    prepared, assembly_seconds = measure_synchronized(
        lambda: plan.prepare(edge_relation=edges)
    )
    if not bool(np.asarray(prepared.metric_result.accepted)):
        raise AssertionError(
            "Native exterior moment provider refused the benchmark metric."
        )
    diffusion, diffusion_seconds = measure_synchronized(lambda: prepared.diffusion())
    if not bool(np.asarray(diffusion.admitted)):
        raise AssertionError("Native exterior diffusion admission failed.")
    values = jnp.asarray(np.cos(np.sum(np.asarray(prepared.points), axis=1)))
    execution = execution_evidence(diffusion.mv, values, config)
    result = np.asarray(execution.pop("result"))
    pairs = np.asarray(prepared.pairs)
    conductance = np.asarray(diffusion.conductances)
    difference = np.asarray(values)[pairs[:, 1]] - np.asarray(values)[pairs[:, 0]]
    amount_rate = np.zeros(prepared.points.shape[0])
    np.add.at(amount_rate, pairs[:, 0], conductance * difference)
    np.add.at(amount_rate, pairs[:, 1], -conductance * difference)
    reference = amount_rate / np.asarray(prepared.node_volumes)
    action_error = float(np.max(np.abs(result - reference)))
    conservation = float(abs(np.dot(np.asarray(prepared.node_volumes), result)))
    tolerance = 5e-3 if config.precision == "float32" else 1e-7
    if action_error > tolerance or conservation > tolerance:
        raise AssertionError(
            "Conservative native action disagrees with independent host edge ledger."
        )
    refreshed, refresh_seconds = measure_synchronized(
        lambda: prepared.metric_system.solve(prior=prepared.metric_system.prior * 1.01)
    )
    if not bool(np.asarray(refreshed.accepted)):
        raise AssertionError("Native numeric metric refresh was refused.")
    retained = logical_array_bytes((plan, prepared, diffusion, refreshed))
    if retained > config.working_set_bytes:
        raise ValueError("Retained exterior arrays exceed working-set budget.")
    phases = unavailable_phases()
    phases.update(execution.pop("phases"))
    phases["neighbor"] = measured_phase(neighbor_seconds)
    phases["assembly"] = {
        **measured_phase(assembly_seconds),
        "scope": "moment-symbolics,metric-solve,cochain-binding",
    }
    phases["numeric-refresh"] = measured_phase(refresh_seconds)
    phases["stencil"] = {
        "status": "unavailable",
        "reason": "Conservative edge moments do not build strong-form row stencils",
    }
    return {
        "capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "phases": phases,
        **execution,
        "diffusion_binding_seconds": diffusion_seconds,
        "retained_bytes": retained,
        "reserved_working_set_bytes": reservation,
        "edge_capacity": pair_budget,
        "active_edges": pairs.shape[0],
        "active_points": prepared.points.shape[0],
        "inactive_capacity": capacity - prepared.points.shape[0],
        "rank": prepared.metric_result.rank,
        "redundant_constraints": prepared.metric_result.redundant_constraints,
        "moment_residual": float(
            np.max(np.abs(np.asarray(prepared.metric_result.moment_residual)))
        ),
        "negative_weights": int(np.asarray(prepared.metric_result.negative_count)),
        "metric_status": int(np.asarray(prepared.metric_result.status)),
        "conservation_defect": conservation,
        "action_error": action_error,
        "domain": "explicit-union-of-Cartesian-nodal-control-volumes-with-incomplete-axial-neighborhood-Dirichlet-boundary",
        "oracle": "independent-host-NumPy edge flux accumulation and amount ledger",
    }


def run(config: MeshfreeConfig = MeshfreeConfig(), /) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", config.precision == "float64")
    import examples.meshfree_conservative_diffusion as consumer

    return make_record(
        config,
        [
            measure_capacity(size, seed, config)
            for size in config.sizes
            for seed in config.seeds
        ],
        Path(__file__),
        consumers=(Path(consumer.__file__),),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    args = parser.parse_args()
    record = run(config_from_arguments(args))
    apply_baseline(record, args.baseline)
    if args.output is not None:
        write_json_atomic(args.output, record)
    else:
        print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
