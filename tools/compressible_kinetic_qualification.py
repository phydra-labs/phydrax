#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from phydrax.discretization.discrete_velocity import (
    entropic_d3q343_plan,
    FilteredD3Q33Plan,
    FullRangeQuasiEquilibriumPlan,
    guided_d3q39_plan,
    IntegerLatticeTransportPlan,
    KineticVelocityPartitionPlan,
)


def qualify(shape: tuple[int, int, int]) -> dict[str, object]:
    density = jnp.ones(shape)
    velocity = jnp.zeros(shape + (3,))
    temperature = jnp.ones(shape)
    guided = guided_d3q39_plan()
    guided_state = guided.initialize(density, velocity, temperature)
    guided_collision = guided.collide(guided_state, 1.0)
    quasi_collision, quasi_evidence = FullRangeQuasiEquilibriumPlan(
        guided, prandtl_number=0.72
    ).collide(guided_state, 1.0)
    streamed, stream_evidence = IntegerLatticeTransportPlan(guided.rule, shape).stream(
        guided_state
    )
    del streamed

    reference = entropic_d3q343_plan()
    reference_state = reference.initialize(
        jnp.ones((1,)), jnp.zeros((1, 3)), jnp.ones((1,))
    )
    reference_collision = reference.collide(reference_state, 0.5)
    partition = KineticVelocityPartitionPlan(reference.rule, 7)
    _, partition_evidence = partition.assemble(
        partition.partition(reference_state.population("particle"))
    )

    filtered = FilteredD3Q33Plan()
    filtered_state = filtered.initialize(density, velocity, temperature)
    filtered_collision, filtered_evidence = filtered.collide(filtered_state, 1.0)

    metrics = {
        "guided_successful": bool(jnp.all(guided_collision.successful)),
        "guided_mass_defect": float(
            jnp.max(jnp.abs(guided_collision.conservation.mass_defect))
        ),
        "guided_energy_defect": float(
            jnp.max(jnp.abs(guided_collision.conservation.energy_defect))
        ),
        "quasi_successful": bool(jnp.all(quasi_collision.successful)),
        "quasi_prandtl": float(quasi_evidence.requested_prandtl.reshape(-1)[0]),
        "stream_successful": bool(stream_evidence.successful),
        "reference_successful": bool(jnp.all(reference_collision.successful)),
        "partition_successful": bool(partition_evidence.successful),
        "filtered_successful": bool(jnp.all(filtered_collision.successful)),
        "filtered_conservation_defect": float(
            jnp.max(filtered_evidence.conserved_moment_defect)
        ),
    }
    return {
        "kind": "compressible-kinetic-qualification",
        "grid_shape": list(shape),
        "backend": jax.default_backend(),
        "metrics": metrics,
        "successful": all(
            bool(value) for name, value in metrics.items() if name.endswith("successful")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size < 7:
        raise ValueError("size must exceed the D3Q39 reach on both sides.")
    report = qualify((args.size, args.size, args.size))
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
