#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Explicit near-incompressible elastic block with selectable volumetric treatment.

Run: python -m examples.material_point_volumetric_stabilization --mode f_bar
This is a dynamic consumer example, not an equilibrium locking qualification.
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax.discretization.mpm import (
    MPMVolumetricStabilizationKind,
    MPMVolumetricStabilizationPlan,
)
from phydrax.typing import parse


def simulate(kind: MPMVolumetricStabilizationKind, steps: int) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be positive.")
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float64)
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(33, periodic=False, endpoint=True),) * 2,
        axis_names=("x", "y"),
    ).prepare(bounds)
    x, y = jnp.meshgrid(
        jnp.linspace(0.35, 0.65, 12, dtype=jnp.float64),
        jnp.linspace(0.35, 0.65, 12, dtype=jnp.float64),
        indexing="ij",
    )
    position = jnp.stack((x, y), axis=-1).reshape(-1, 2)
    count = position.shape[0]
    volume = jnp.full((count,), 0.09 / count, dtype=jnp.float64)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count, dtype=jnp.int32), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        boundary="reject",
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "near-incompressible-block",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(
            volumetric_stabilization=MPMVolumetricStabilizationPlan(kind)
        ),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.1, 0.1], [0.9, 0.9]], dtype=jnp.float64),
            periodic=(False, False),
            support_margin=0.07,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 1000.0)
    )
    velocity = 0.02 * jnp.stack((position[:, 0] - 0.5, 0.5 - position[:, 1]), axis=-1)
    state = compiled.initialize_state(position, velocity, volume, arguments)
    step_size = jnp.asarray(1e-5, dtype=jnp.float64)
    advance = jax.jit(
        lambda value: compiled.dynamics.step_detailed(value, step_size, arguments)
    )
    detail = advance(state)
    if not bool(detail.successful):
        raise RuntimeError(
            "The explicit step was rejected; the committed state is unchanged."
        )
    state = detail.accepted_state
    for _ in range(1, steps):
        detail = advance(state)
        if not bool(detail.successful):
            raise RuntimeError(
                "The explicit step was rejected; the committed state is unchanged."
            )
        state = detail.accepted_state
    return {
        "mode": kind,
        "accepted_steps": int(state.accepted_step),
        "time": float(state.time),
        "finite": bool(jnp.all(jnp.isfinite(state.particles.deformation_gradient))),
        "relative_mass_defect": float(detail.diagnostics.transfer.relative_mass_defect),
        "relative_momentum_defect": float(
            detail.diagnostics.transfer.relative_momentum_defect
        ),
        "maximum_assumed_deformation_change": float(
            jnp.max(
                jnp.abs(
                    state.particles.deformation_gradient
                    - jnp.eye(2, dtype=jnp.float64)[None, :, :]
                )
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="f_bar")
    parser.add_argument("--steps", type=int, default=20)
    arguments = parser.parse_args()
    kind = parse(arguments.mode, MPMVolumetricStabilizationKind, "mode")
    print(
        json.dumps(
            simulate(kind, arguments.steps), indent=2, sort_keys=True, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
