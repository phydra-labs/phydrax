#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""A real committed fiber step observed in an explicitly idealized cylinder."""

import json

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.electromyography import (
    Farina2004CylindricalConductorPlan,
    PereiraBotelho2019FiberCurrentPlan,
)
from phydrax.applications.skeletal_muscle.fibers import (
    PrescribedFiberStimulusSchedule,
    SkeletalFiberBundlePlan,
)


def main():
    nodes = 7
    mask = jnp.zeros((1, 1, nodes), dtype=bool).at[0, 0, nodes // 2].set(True)
    stimulus = PrescribedFiberStimulusSchedule([0.0], [0.5], [150.0], mask)
    fiber = SkeletalFiberBundlePlan(
        ("fiber-0",), nodes, [12.0], [0.05], stimulus
    ).prepare()
    positions = jnp.zeros((1, nodes, 3)).at[..., 0].set(0.015)
    positions = positions.at[..., 2].set(jnp.linspace(-0.006, 0.006, nodes))
    current = PereiraBotelho2019FiberCurrentPlan(
        ("fiber-0",),
        positions,
        [25e-6],
        geometry_source_id="explicit-idealized-example-not-anatomy",
        geometry_license="CC0-1.0 manufactured numerical input",
    ).prepare(fiber)
    cylinder = Farina2004CylindricalConductorPlan(
        [0.03, 0.035, 0.04],
        [0.1, 0.5, 0.05, 1.0],
        [[0.0, 0.0], [0.0, 0.005]],
        [[0.002, 0.003], [0.002, 0.003]],
        [[1.0, -1.0]],
        ("contact-0", "contact-1"),
        ("bipolar",),
        axial_period_m=0.1,
        longitudinal_modes=4,
        angular_modes=3,
        coordinate_frame_id="example-cylinder-origin-axis-z",
        material_source_id="Farina-2004-conductivities-only-not-source-geometry",
        electrode_source_id="explicit-idealized-rectangular-apertures",
    ).prepare(current, coordinate_frame_id="example-cylinder-origin-axis-z")
    initial = fiber.initialize()
    fiber_candidate = fiber.candidate(initial, 0.02)
    accepted_fiber = fiber_candidate.commit()
    current_prior = current.initialize()
    current_candidate = current.propose(
        current_prior,
        accepted_fiber,
        fiber_prepared_id=fiber.prepared_id,
        geometry_id=current.plan.geometry_id,
    )
    accepted_current = current_candidate.commit(current_prior, accepted_fiber)
    observation_prior = cylinder.initialize()
    observation_candidate = cylinder.propose(observation_prior, accepted_current)
    observation = observation_candidate.commit(observation_prior, accepted_current)
    successful = (
        fiber_candidate.evidence.successful
        & current_candidate.evidence.successful
        & observation_candidate.evidence.successful
    )
    print(
        json.dumps(
            {
                "successful": bool(successful),
                "accepted_time_ms": float(observation.time_ms),
                "net_fiber_current_A": float(
                    jnp.sum(accepted_current.transmembrane_current_A)
                ),
                "lead_voltage_V": observation.lead_voltage_V.tolist(),
                "claim_scope": "idealized static cylinder; no anatomical or intramuscular fidelity",
            },
            indent=2,
        )
    )
    if not bool(successful):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
