#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Decision benchmark for bounded fermionic frontier-field candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from _runtime import (
    capture_environment,
    logical_array_bytes,
    measure_host,
    measure_repeated,
)

from phydrax.applications import (
    diagrammatic_field as df,
    functional_rg as frg,
    nonequilibrium_field as nef,
    sign_problem as sp,
)
from phydrax.operators.quantum._two_particle_green import (
    FermionicTwoParticleChannelConvention,
    MatsubaraTwoParticleGreenFunction,
)


def _patch_case(side: int, radial_count: int, frequency_count: int, repeats: int):
    axis = 2.0 * np.pi * np.arange(side) / side
    momenta = np.asarray([(x, y) for x in axis for y in axis])
    angles = np.linspace(0.0, 2.0 * np.pi, momenta.shape[0], endpoint=False)
    velocities = np.stack(
        (1.25 + 0.2 * np.cos(angles), 0.75 + 0.2 * np.sin(angles)), axis=-1
    )
    half = radial_count // 2
    positive = np.linspace(0.1, 1.0, half)
    radial = np.concatenate((-positive[::-1], positive))
    plan = frg.FermiSurfacePatchRGPlan(
        momenta,
        np.diag((2.0 * np.pi, 2.0 * np.pi)),
        velocities,
        np.ones(momenta.shape[0]),
        radial,
        np.ones(radial.size) / radial.size,
        beta=6.0,
        matsubara_count=frequency_count,
        maximum_work_elements=40_000_000,
    )
    prepared, prepare_seconds = measure_host(plan.prepare)
    vertex = prepared.vertex(jnp.full((momenta.shape[0],) * 3, 0.1))
    result, execution = measure_repeated(
        lambda: prepared.evaluate(vertex, 0.5), warmup=1, repeats=repeats
    )
    return {
        "patches": momenta.shape[0],
        "radial_nodes": radial.size,
        "positive_matsubara_labels": frequency_count,
        "prepare_seconds": prepare_seconds,
        "steady": execution.to_seconds_dict(),
        "logical_bytes": logical_array_bytes((prepared, result)),
        "routing_residual": float(result.evidence.momentum_routing_residual),
        "crossing_residual": float(
            jnp.maximum(
                result.evidence.beta_incoming_crossing_residual,
                result.evidence.beta_outgoing_crossing_residual,
            )
        ),
        "admissible": bool(result.evidence.admissible),
        "compile": "not-applicable-host-planned-candidate",
    }


def _two_particle_vertex(label_count: int):
    start = -(label_count // 2)
    labels = jnp.arange(start, start + label_count, dtype=jnp.int32)
    convention = FermionicTwoParticleChannelConvention("particle-hole-direct")
    route = convention.route(
        labels[:, None, None], labels[None, :, None], labels[None, None, :]
    )
    values = 0.01 * (route[..., 0] - route[..., 2]) * (route[..., 1] - route[..., 3])
    return MatsubaraTwoParticleGreenFunction(
        convention,
        5.0,
        labels,
        labels,
        labels,
        values[..., None, None, None, None].astype(complex),
        fermion_label_minimum=start,
        fermion_label_count=label_count,
    )


def _parquet_case(label_count: int, repeats: int):
    vertex = _two_particle_vertex(label_count)
    plan = df.LatticeParquetPlan(maximum_iterations=64, tolerance=1e-10)
    prepared, prepare_seconds = measure_host(
        lambda: plan.prepare(vertex, jnp.zeros((3,) + vertex.values.shape[:3]))
    )
    result, execution = measure_repeated(prepared.iterate, warmup=0, repeats=repeats)
    return {
        "labels_per_axis": label_count,
        "vertex_elements": int(vertex.values.size),
        "prepare_seconds": prepare_seconds,
        "steady": execution.to_seconds_dict(),
        "logical_bytes": logical_array_bytes((prepared, result)),
        "iterations": int(result.iterations),
        "fixed_point_residual": float(result.evidence.fixed_point_residual),
        "parquet_identity_residual": float(result.evidence.parquet_identity_residual),
        "successful": bool(result.evidence.successful),
    }


def _fermion_diagram(order: int):
    field = df.FieldSpec(f"benchmark-psi-{order}", statistics="fermion")
    propagator = df.PropagatorSpec(field, mass=1.0)
    rule = df.VertexRule(
        f"benchmark-bilinear-{order}",
        (field, field),
        1.0,
        perturbative_order=order,
    )
    return df.DiagramGraph(
        (df.VertexInsertion("v", rule),),
        (df.PropagatorLine("loop", propagator, "v", "v", df.MomentumRoute([0.0])),),
    )


def _monte_carlo_case(steps: int, repeats: int):
    prepared = df.LowOrderFermionDiagramMonteCarloPlan(
        steps=steps, maximum_order=3, maximum_diagrams=3
    ).prepare(
        tuple(_fermion_diagram(order) for order in (1, 2, 3)),
        jnp.asarray((1.0 + 0.0j, -0.4 + 0.1j, 0.2 - 0.05j)),
        jnp.asarray(((0.5, 0.4, 0.1), (0.4, 0.2, 0.4), (0.1, 0.4, 0.5))),
    )
    result, execution = measure_repeated(
        lambda: prepared.run(jr.key(29)), warmup=1, repeats=repeats
    )
    study = sp.evaluate_controlled_sign_study(
        sp.ControlledSignStudyPlan(
            minimum_draws=min(steps, 16),
            minimum_effective_sample_size=2.0,
        ),
        result.chain.phases[None, :],
        result.chain.orders[None, :, None],
        study_id=f"benchmark-diagram-chain-{steps}",
    )
    return {
        "steps": steps,
        "orders": 3,
        "steady": execution.to_seconds_dict(),
        "logical_bytes": logical_array_bytes((prepared, result, study)),
        "acceptance_rate": float(result.evidence.acceptance_rate),
        "average_sign": float(result.evidence.average_sign),
        "order_effective_sample_size": float(result.evidence.order_effective_sample_size),
        "sign_effective_sample_size": float(study.effective_sample_size),
        "phase_covariance": np.asarray(study.phase_covariance).tolist(),
        "abstained": bool(study.abstained),
    }


def _second_born_case(time_count: int, repeats: int):
    grid = nef.ClosedTimePathPlan(jnp.linspace(0.0, 0.4, time_count)).prepare()
    propagators = jnp.broadcast_to(jnp.eye(1, dtype=complex), (time_count, 1, 1))
    free = nef.fermionic_keldysh_from_propagators(
        grid, propagators, jnp.asarray([[0.5]]), source_id=f"benchmark-free-{time_count}"
    )
    prepared = nef.FermionicSecondBornPlan(
        grid,
        jnp.zeros((1, 1)),
        jnp.asarray([0.02]),
        maximum_iterations=64,
        tolerance=1e-7,
        conservation_tolerance=1e-3,
    ).prepare(free)
    result, execution = measure_repeated(prepared.solve, warmup=0, repeats=repeats)
    return {
        "time_nodes": time_count,
        "modes": 1,
        "steady": execution.to_seconds_dict(),
        "logical_bytes": logical_array_bytes((prepared, result)),
        "iterations": int(result.evidence.iterations),
        "fixed_point_residual": float(result.evidence.fixed_point_residual),
        "schwinger_dyson_residual": float(result.evidence.schwinger_dyson_residual),
        "particle_number_drift": float(result.evidence.particle_number_drift),
        "relative_energy_drift": float(result.evidence.relative_energy_drift),
        "successful": bool(result.evidence.successful),
    }


def _json_default(value):
    if isinstance(value, (jax.Array, np.ndarray)):
        host = np.asarray(jax.device_get(value))
        return host.item() if host.shape == () else host.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "patch_frg": [
            _patch_case(2, radial, frequency, arguments.repeats)
            for radial, frequency in ((4, 2), (8, 4))
        ],
        "lattice_parquet": [
            _parquet_case(labels, arguments.repeats) for labels in (3, 5)
        ],
        "diagram_monte_carlo": [
            _monte_carlo_case(steps, arguments.repeats) for steps in (256, 1024)
        ],
        "fermionic_second_born": [
            _second_born_case(times, arguments.repeats) for times in (4, 8)
        ],
    }
    encoded = json.dumps(payload, indent=2, default=_json_default)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
