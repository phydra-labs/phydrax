# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded conservative graph metric preparation, refresh, action and adjoint scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._runtime import logical_array_bytes
from benchmarks.meshfree_scaling import (
    add_config_arguments,
    admitted_rows,
    apply_baseline,
    config_from_arguments,
    configure_precision,
    declare_reservation,
    execution_evidence,
    make_record,
    MeshfreeConfig,
    PhaseRecorder,
)


def measure_capacity(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    from examples.meshfree_conservative_diffusion import exterior_plan
    from phydrax.discretization.meshfree import MeshfreeEdgeRelationPlan

    reservation = config.check_capacity(capacity)
    pair_budget = capacity * config.neighbors * 4
    symbolic_budget = config.working_set_bytes // 64
    declare_reservation(pair_budget * 64, config, scope="exterior edge capacity")
    plan = exterior_plan(
        size=capacity,
        dimension=config.dimension,
        seed=seed,
        maximum_pairs=pair_budget,
        maximum_symbolic_entries=symbolic_budget,
    )
    recorder = PhaseRecorder()
    edges = recorder.run(
        "search",
        lambda: MeshfreeEdgeRelationPlan(
            plan.points,
            plan.radius,
            pair_budget,
            active=plan.active,
            target_chunk_size=config.chunk_rows,
        ).prepare(),
    )
    prepared = recorder.run(
        "rank-certificate",
        lambda: plan.prepare(edge_relation=edges),
        scope="moment-symbolics, rank certificate, exact metric solve, cochain binding",
    )
    for phase in ("local-fit", "conic", "ordering-fill"):
        recorder.unavailable(
            phase,
            "MeshfreeExteriorCalculusPlan.prepare fuses moment, rank and metric phases",
        )
    result = prepared.metric_result
    if not bool(np.asarray(result.accepted)):
        raise AssertionError(
            "Native exterior moment provider refused the benchmark metric."
        )
    diffusion = recorder.run(
        "assembly", lambda: prepared.diffusion(), scope="diffusion binding"
    )
    if not bool(np.asarray(diffusion.admitted)):
        raise AssertionError("Native exterior diffusion admission failed.")
    values = jnp.asarray(np.cos(np.sum(np.asarray(prepared.points), axis=1)))
    execution = execution_evidence(diffusion.mv, values, config, recorder)
    action = np.asarray(execution.pop("result"))
    pairs = np.asarray(prepared.pairs)
    conductance = np.asarray(diffusion.conductances)
    difference = np.asarray(values)[pairs[:, 1]] - np.asarray(values)[pairs[:, 0]]
    amount_rate = np.zeros(prepared.points.shape[0])
    np.add.at(amount_rate, pairs[:, 0], conductance * difference)
    np.add.at(amount_rate, pairs[:, 1], -conductance * difference)
    reference = amount_rate / np.asarray(prepared.node_volumes)
    action_error = float(np.max(np.abs(action - reference)))
    conservation = float(abs(np.dot(np.asarray(prepared.node_volumes), action)))
    tolerance = 5e-3 if config.precision == "float32" else 1e-7
    if action_error > tolerance or conservation > tolerance:
        raise AssertionError(
            "Conservative native action disagrees with independent host edge ledger."
        )
    refreshed = recorder.run(
        "numeric-refresh",
        lambda: prepared.metric_system.solve(prior=prepared.metric_system.prior * 1.01),
        scope="metric numeric refresh with a changed prior",
    )
    if not bool(np.asarray(refreshed.accepted)):
        raise AssertionError("Native numeric metric refresh was refused.")
    retained = logical_array_bytes((plan, prepared, diffusion, refreshed))
    declare_reservation(retained, config, scope="retained exterior arrays")
    return {
        "capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "status": "measured",
        "phases": recorder.record(),
        **execution,
        "retained_bytes": retained,
        "reserved_working_set_bytes": reservation,
        "edge_capacity": pair_budget,
        "active_edges": pairs.shape[0],
        "active_points": prepared.points.shape[0],
        "inactive_capacity": capacity - prepared.points.shape[0],
        "rank": int(np.asarray(result.rank)),
        "rank_maximal": bool(np.asarray(result.rank_maximal)),
        "exact": bool(np.asarray(result.exact)),
        "moment_residual": float(np.max(np.abs(np.asarray(result.moment_residual)))),
        "negative_weights": int(np.asarray(result.negative_count)),
        "metric_status": int(np.asarray(result.status)),
        "provider_status": int(np.asarray(result.provider_status)),
        "derivative_available": bool(np.asarray(result.derivative_available)),
        "derivative_contract": result.derivative_contract,
        "minimum_norm_iterations": None
        if result.linear_result is None
        else int(np.asarray(result.linear_result.diagnostics.iterations)),
        "conservation_defect": conservation,
        "action_error": action_error,
        "domain": "explicit-union-of-Cartesian-nodal-control-volumes-with-incomplete-axial-neighborhood-Dirichlet-boundary",
        "oracle": "independent-host-NumPy edge flux accumulation and amount ledger",
    }


def run(config: MeshfreeConfig = MeshfreeConfig(), /) -> dict[str, Any]:
    configure_precision(config, supported=("float64",))
    import examples.meshfree_conservative_diffusion as consumer

    return make_record(
        config,
        admitted_rows(measure_capacity, config),
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
