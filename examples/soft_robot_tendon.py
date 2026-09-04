"""Run one contact-free, tendon-driven spatial reduced-rod transaction."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.solid_mechanics import (
    FrictionlessElasticTendonPlan,
    prepare_frictionless_elastic_tendon,
    prepare_reduced_rod,
    prepare_reduced_rod_dynamics,
    prepare_reduced_rod_plant,
    prepare_rod,
    prepare_tendon_driven_rod_plant,
    ReducedRodPlan,
    ReducedRodSemiImplicitVelocityEuler,
    RodMaterialStation,
    RodPlan,
    RodStrainBasisPlan,
    TendonRoutePlan,
)
from phydrax.dynamics import PlantStepContext


def build_plant():
    """Prepare the exact fixed-base/contact-free two-tendon profile."""
    dtype = jnp.float32
    rest_positions = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.4, 0.0, 0.0),
            (0.8, 0.0, 0.0),
            (1.2, 0.0, 0.0),
        ),
        dtype=dtype,
    )
    segment_count = rest_positions.shape[0] - 1
    rod = prepare_rod(
        RodPlan(
            jnp.stack(
                (
                    jnp.arange(segment_count, dtype=jnp.int32),
                    jnp.arange(1, segment_count + 1, dtype=jnp.int32),
                ),
                axis=-1,
            ),
            rest_positions,
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (segment_count, 3, 3)),
            jnp.ones((segment_count + 1,), dtype=dtype),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((0.02, 0.02, 0.01), dtype=dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((120.0, 45.0, 45.0), dtype=dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((5.0, 5.0, 3.0), dtype=dtype)),
                (segment_count - 1, 3, 3),
            ),
        )
    )
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        component_scales=jnp.asarray((0.08, 0.08, 0.08, 0.2, 0.2, 0.2), dtype=dtype),
        quadrature_order=3,
    )
    reduced = prepare_reduced_rod(rod, ReducedRodPlan(basis))
    dynamics = prepare_reduced_rod_dynamics(reduced)
    base_plant = prepare_reduced_rod_plant(
        dynamics,
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=0.01,
            energy_balance_tolerance=1.0,
        ),
    )

    tendons = []
    for index, offset_y in enumerate((0.025, -0.025)):
        route = TendonRoutePlan(
            (
                RodMaterialStation(
                    0, 0.0, jnp.asarray((0.0, offset_y, 0.0), dtype=dtype)
                ),
                RodMaterialStation(
                    segment_count - 1,
                    1.0,
                    jnp.asarray((0.0, offset_y, 0.0), dtype=dtype),
                ),
            ),
            label=f"tendon-{index}",
        )
        plan = FrictionlessElasticTendonPlan(
            route,
            30.0,
            free_length_bounds=(0.9, 1.4),
            payout_rate_bounds=(-0.05, 0.05),
            tendon_length_bounds=(1.0, 1.4),
            maximum_tension=20.0,
            power_tolerance=1.0e-5,
            label=f"tendon-{index}",
        )
        tendons.append(prepare_frictionless_elastic_tendon(plan, reduced))

    return prepare_tendon_driven_rod_plant(
        base_plant,
        tuple(tendons),
        (jnp.asarray(1.18, dtype=dtype), jnp.asarray(1.18, dtype=dtype)),
    )


def main() -> None:
    plant = build_plant()
    parameters = plant.bind_parameters()
    reset = plant.reset(jax.random.key(90210), parameters)
    if not bool(np.asarray(reset.successful)):
        raise RuntimeError(f"tendon plant reset failed with status {reset.status}")

    source = reset.accepted_state
    command = plant.command(
        (
            jnp.asarray(-0.01, dtype=jnp.float32),
            jnp.asarray(0.01, dtype=jnp.float32),
        )
    )
    context = PlantStepContext(
        source.time,
        source.time + jnp.asarray(0.002, dtype=source.time.dtype),
        source.step_index,
    )
    step = plant.step(context, source, command, parameters)
    if not bool(np.asarray(step.successful)):
        raise RuntimeError(f"tendon plant step failed with status {step.status}")

    checkpoint = plant.checkpoint(source)
    digest = plant.state_digest(step.accepted_state)
    replay = plant.replay(
        checkpoint,
        (context,),
        (command,),
        parameters,
        expected_digests=(digest,),
    )
    if not bool(np.asarray(replay.successful)) or not replay.matched:
        raise RuntimeError("accepted tendon transaction did not replay exactly")

    evidence = step.evidence
    print(
        json.dumps(
            {
                "accepted_step_index": int(np.asarray(step.accepted_state.step_index)),
                "actuation": {
                    "balanced": bool(np.asarray(evidence.tendon_ledger.balanced)),
                    "rod_work": float(np.asarray(evidence.tendon_ledger.total_rod_work)),
                    "spool_work": float(
                        np.asarray(evidence.tendon_ledger.total_spool_work)
                    ),
                    "total_energy_residual": float(
                        np.asarray(evidence.tendon_ledger.total_energy_residual)
                    ),
                },
                "plant_id": plant.plant_id,
                "replay_matched": replay.matched,
                "status": int(np.asarray(step.status)),
                "successful": bool(np.asarray(step.successful)),
                "tendon_ids": plant.tendon_ids,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
