#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""One explicit MPM step with canonical traction and a viscous absorber.

Run: python -m examples.material_point_boundary_loads
The printed ledgers are measured step results, not a wave-reflection qualification.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.solid_mechanics import (
    MechanicalLoadState,
    ReferenceDeadTraction,
)
from phydrax.discretization.mpm import (
    LysmerKuhlemeyerDashpot,
    PreparedMPMBoundaryLoad,
)


def simulate() -> dict[str, object]:
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float64)
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(9, periodic=False, endpoint=True),) * 2,
        axis_names=("x", "y"),
    ).prepare(bounds)
    x, y = jnp.meshgrid(
        jnp.linspace(0.3125, 0.6875, 4, dtype=jnp.float64),
        jnp.linspace(0.3125, 0.6875, 4, dtype=jnp.float64),
        indexing="ij",
    )
    position = jnp.stack((x, y), axis=-1).reshape(-1, 2)
    count = position.shape[0]
    volume = jnp.full((count,), 0.25 / count, dtype=jnp.float64)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count, dtype=jnp.int32), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        boundary="reject",
    ).prepare(particles)

    # Four explicit boundary quadrature points, each carrying physical area 1/8.
    boundary_position = jnp.stack(
        (
            jnp.linspace(0.3125, 0.6875, 4, dtype=jnp.float64),
            jnp.full((4,), 0.6875, dtype=jnp.float64),
        ),
        axis=-1,
    )
    area_vectors = jnp.tile(jnp.asarray([[0.0, 0.125]], dtype=jnp.float64), (4, 1))
    traction = PreparedMPMBoundaryLoad(
        splat,
        boundary_position,
        area_vectors,
        ReferenceDeadTraction(jnp.asarray([0.0, 0.1], dtype=jnp.float64)),
    )

    # Trapezoidal nodal areas integrate a boundary of length 1/2 (unit depth).
    nodal_area_vectors = np.zeros((9, 9, 2), dtype=np.float64)
    nodal_area_vectors[2:7, 6, 1] = np.asarray(
        [0.0625, 0.125, 0.125, 0.125, 0.0625], dtype=np.float64
    )
    dashpot = LysmerKuhlemeyerDashpot(nodal_area_vectors, 1.0, 2.0, 1.0)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "canonical-boundary-loads",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
            boundary_loads=(traction,),
            dashpots=(dashpot,),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.25, 0.25], [0.75, 0.75]], dtype=jnp.float64),
            periodic=(False, False),
            support_margin=0.1875,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(1.0, 2.0),
        mechanical_load_states=(MechanicalLoadState(),),
    )
    velocity = jnp.tile(jnp.asarray([[0.0, 0.01]], dtype=jnp.float64), (count, 1))
    state = compiled.initialize_state(position, velocity, volume, arguments)
    detail = compiled.dynamics.step_detailed(
        state, jnp.asarray(1e-4, dtype=jnp.float64), arguments
    )
    if not bool(detail.successful):
        raise RuntimeError(
            "The boundary-loaded step was rejected without committing state."
        )
    load_result = detail.diagnostics.boundary_loads[0]
    dashpot_result = detail.diagnostics.dashpots[0]
    return {
        "successful": bool(detail.successful),
        "accepted_steps": int(detail.accepted_state.accepted_step),
        "time": float(detail.accepted_state.time),
        "traction": {
            "successful": bool(load_result.successful),
            "total_impulse": np.sum(
                np.asarray(load_result.impulse).reshape(-1, 2), axis=0
            ).tolist(),
            "work": float(load_result.work),
            "dissipation": float(load_result.dissipation),
        },
        "dashpot": {
            "successful": bool(dashpot_result.successful),
            "total_impulse": np.sum(
                np.asarray(dashpot_result.impulse).reshape(-1, 2), axis=0
            ).tolist(),
            "work": float(dashpot_result.work),
            "dissipation": float(dashpot_result.dissipation),
        },
    }


def main() -> None:
    print(json.dumps(simulate(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
