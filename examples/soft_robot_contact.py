"""Run the qualified spatial capsule–plane reduced-rod contact profile."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.contact import (
    prepare_reduced_rod_contact_participant,
    RodCapsuleGeometryPlan,
    RodContactCCDPlan,
    RodContactSearchPlan,
)
from phydrax.applications.solid_mechanics import (
    prepare_reduced_rod,
    prepare_reduced_rod_contact_plant,
    prepare_reduced_rod_dynamics,
    prepare_rod,
    ReducedRodPlan,
    ReducedRodSemiImplicitVelocityEuler,
    ReducedRodState,
    RodPlan,
    RodStrainBasisPlan,
)
from phydrax.discretization import (
    CollisionFeatureKind,
    CollisionFeaturePolicy,
    PlaneContactGeometry,
)
from phydrax.dynamics import PlantStepContext


def build_plant():
    """Prepare one fixed-base circular-capsule/plane/self-contact plant."""
    dtype = jnp.float32
    segment_count = 5
    rest_positions = jnp.asarray(
        tuple((float(index), 0.0, 0.55) for index in range(segment_count + 1)),
        dtype=dtype,
    )
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
                jnp.diag(jnp.asarray((0.2, 0.2, 0.1), dtype=dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((30.0, 12.0, 12.0), dtype=dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((2.0, 2.0, 2.0), dtype=dtype)),
                (segment_count - 1, 3, 3),
            ),
        )
    )
    basis = RodStrainBasisPlan.shifted_legendre(
        0, dimension=3, component_scales=jnp.ones((6,), dtype=dtype)
    )
    reduced = prepare_reduced_rod(rod, ReducedRodPlan(basis))
    dynamics = prepare_reduced_rod_dynamics(reduced)
    capsule = RodCapsuleGeometryPlan(
        jnp.full((segment_count,), 0.1, dtype=dtype),
        participant_id=3,
        body_id=5,
        material_id=7,
        patch_id=11,
    ).prepare(rod)
    participant = prepare_reduced_rod_contact_participant(reduced, capsule)
    plane_features = CollisionFeaturePolicy(
        jnp.asarray((10_000,), dtype=jnp.int64),
        jnp.asarray((int(CollisionFeatureKind.ANALYTIC),), dtype=jnp.int32),
        participant_ids=101,
        body_ids=103,
        material_ids=107,
        patch_ids=109,
        static_mask=True,
        provenance_id="soft-robot-contact-example:plane",
    )
    plane = PlaneContactGeometry(
        jnp.asarray((0.0, 0.0, 1.0), dtype=dtype),
        0.0,
        feature_policy=plane_features,
    )
    search = RodContactSearchPlan(
        capacity=24,
        plane_capacity=segment_count,
        activation_distance=0.04,
        route="dense",
    ).prepare(capsule, planes=(plane,))
    source = reduced.initialize_state()
    initial = ReducedRodState(
        source.coefficients,
        jnp.asarray((0.0, 0.0, -12.0, 0.0, 0.0, 0.0), dtype=dtype),
    )
    return prepare_reduced_rod_contact_plant(
        dynamics,
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=0.2, energy_balance_tolerance=1.0e3
        ),
        participant,
        search,
        RodContactCCDPlan(),
        dynamic_friction=0.0,
        static_friction=0.0,
        initial_reduced_state=initial,
        gap_tolerance=2.0e-5,
        energy_tolerance=5.0e-4,
        conservation_tolerance=5.0e-5,
    )


def main() -> None:
    plant = build_plant()
    parameters = plant.bind_parameters()
    reset = plant.reset(jax.random.key(314159), parameters)
    if not bool(np.asarray(reset.successful)):
        raise RuntimeError(f"contact plant reset failed with status {reset.status}")
    source = reset.accepted_state
    context = PlantStepContext(
        source.time,
        source.time + jnp.asarray(0.02, dtype=source.time.dtype),
        source.step_index,
    )
    step = plant.step(context, source, None, parameters)
    if not bool(np.asarray(step.successful)):
        raise RuntimeError(f"contact plant step failed with status {step.status}")
    evidence = step.evidence
    if not bool(np.asarray(evidence.full_interval_covered)):
        raise RuntimeError("contact transaction did not certify the requested interval")
    print(
        json.dumps(
            {
                "capability_id": plant.capability_id,
                "ccd": {
                    "full_step_safe": bool(
                        np.asarray(evidence.swept_ccd.evidence.full_step_safe)
                    ),
                    "impact_detected": bool(
                        np.asarray(evidence.swept_ccd.evidence.impact_detected)
                    ),
                },
                "contact_history": {
                    "active_count": int(
                        np.asarray(
                            jnp.sum(step.accepted_state.payload.contact_state.active)
                        )
                    ),
                    "occupied_count": int(
                        np.asarray(
                            jnp.sum(step.accepted_state.payload.contact_state.occupied)
                        )
                    ),
                },
                "energy": {
                    "friction_dissipation": float(
                        np.asarray(evidence.energy.friction_dissipation)
                    ),
                    "friction_dissipative": bool(
                        np.asarray(evidence.energy.friction_dissipative)
                    ),
                },
                "final_minimum_gap": float(np.asarray(evidence.final_minimum_gap)),
                "full_interval_covered": bool(np.asarray(evidence.full_interval_covered)),
                "status": int(np.asarray(step.status)),
                "successful": bool(np.asarray(step.successful)),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
