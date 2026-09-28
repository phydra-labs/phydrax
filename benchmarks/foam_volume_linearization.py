"""Foam constraint-basis preparation and volume JVP/VJP scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

from phydrax.applications.foams import (
    FoamDynamicsPlan,
    FoamDynamicsState,
    FoamMaterialPlan,
    PreparedFoamDynamics,
    RegionPressureAirPlan,
)
from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceSeed,
    PreparedMultiRegionSurface,
)


def _disconnected_tetra_bubbles(count: int, /) -> MultiRegionSurfaceSeed:
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    labels: list[tuple[int, int]] = []
    for region in range(count):
        offset = len(points)
        shift = 3.0 * region
        points.extend(
            (
                (shift, 0.0, 0.0),
                (shift + 1.0, 0.0, 0.0),
                (shift, 1.0, 0.0),
                (shift, 0.0, 1.0),
            )
        )
        faces.extend(
            (
                (offset + 1, offset + 2, offset + 3),
                (offset, offset + 3, offset + 2),
                (offset, offset + 1, offset + 3),
                (offset, offset + 2, offset + 1),
            )
        )
        labels.extend(((region, count),) * 4)
    return MultiRegionSurfaceSeed(
        np.asarray(points),
        np.asarray(faces),
        np.asarray(labels),
        tuple(f"bubble-{index}" for index in range(count)) + ("ambient",),
        ("finite",) * count + ("boundary",),
        source=f"foam-volume-linearization-{count}",
    )


def _prepared(
    bubble_count: int, headroom: float, /
) -> tuple[PreparedFoamDynamics, FoamDynamicsState]:
    seed = _disconnected_tetra_bubbles(bubble_count)
    topology = seed.topology(
        seed.capacity_plan(
            resource_id=(f"foam-volume-linearization-{bubble_count}-{headroom:.17g}"),
            headroom=headroom,
        )
    )
    surface_state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, surface_state)
    finite_slots = jnp.asarray(topology.finite_region_indices, dtype=jnp.int32)
    target = surface.region_volumes(surface_state.positions)[finite_slots]
    state = FoamDynamicsState(surface_state)
    prepared = PreparedFoamDynamics(
        FoamDynamicsPlan(time_step=1.0e-4, friction=5.0),
        surface,
        FoamMaterialPlan.soap_film(topology.region_ids, 0.025),
        RegionPressureAirPlan.incompressible(target),
        state,
    )
    return prepared, state


def _measure(
    bubble_count: int, headroom: float, warmup: int, repeats: int, /
) -> dict[str, Any]:
    prepared, state = _prepared(bubble_count, headroom)
    positions = state.surface.positions
    direction = jnp.linspace(-0.5, 0.5, positions.size, dtype=positions.dtype).reshape(
        positions.shape
    )
    weights = jnp.ones((prepared.finite_slots.size,), dtype=positions.dtype)

    def evaluate(
        values: jax.Array, tangent: jax.Array, cotangent: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        operator = prepared.volume_operator(values)
        return (
            operator.linearization.primal,
            operator.mv(tangent),
            operator.transpose_mv(cotangent),
        )

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(positions, direction, weights),
        lambda lowered: lowered.compile(),
    )
    _, execution = measure_repeated(
        lambda: compiled(positions, direction, weights),
        warmup=warmup,
        repeats=repeats,
    )
    compiler = compiler_evidence(
        compiled.cost_analysis(), compiled.memory_analysis(), source="xla"
    )
    operator = prepared.volume_operator(positions)
    itemsize = positions.dtype.itemsize
    constraint_count = prepared.constrained_slots.size
    basis = prepared.constraint_basis.evidence
    return {
        "bubble_count": bubble_count,
        "headroom": headroom,
        "vertex_capacity": prepared.surface.topology.vertex_capacity,
        "face_capacity": prepared.surface.topology.face_capacity,
        "region_capacity": prepared.surface.topology.region_capacity,
        "constraint_count": constraint_count,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "warm_execution": execution.to_seconds_dict(),
        "compiler": {
            "flops": compiler.flops,
            "argument_bytes": compiler.argument_bytes,
            "temporary_bytes": compiler.temporary_bytes,
            "output_bytes": compiler.output_bytes,
            "generated_code_bytes": compiler.generated_code_bytes,
        },
        "logical_surface_bytes": logical_array_bytes(prepared.surface),
        "logical_prepared_dynamics_bytes": logical_array_bytes(prepared),
        "constraint_basis": {
            "status": basis.status.name,
            "partition_count": basis.partition_count,
            "closed_partition_count": basis.closed_partition_count,
            "numerical_rank": basis.numerical_rank,
            "rank_check_actions": basis.rank_check_actions,
            "preparation_bytes": basis.preparation_bytes,
            "logical_retained_bytes": basis.logical_retained_bytes,
        },
        "logical_volume_operator_bytes": logical_array_bytes(operator),
        "logical_constraint_gram_bytes": constraint_count**2 * itemsize,
        "dense_region_vertex_jacobian_bytes_avoided": (
            constraint_count * positions.size * itemsize
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headrooms", nargs="+", type=float, default=(1.0, 4.0, 16.0))
    parser.add_argument("--bubble-counts", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        any(count < 1 for count in arguments.bubble_counts)
        or any(headroom < 1.0 for headroom in arguments.headrooms)
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError(
            "bubble counts >= 1, headrooms >= 1, warmup >= 0, and repeats >= 1 "
            "are required."
        )
    payload = {
        "benchmark": "foam-volume-linearization",
        "environment": capture_environment().to_dict(),
        "rows": [
            _measure(
                bubble_count,
                headroom,
                arguments.warmup,
                arguments.repeats,
            )
            for bubble_count in arguments.bubble_counts
            for headroom in arguments.headrooms
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
