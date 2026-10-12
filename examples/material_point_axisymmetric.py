#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Relax an initially unstressed, full-ring thick cylinder under internal pressure.

Run with ``python -m examples.material_point_axisymmetric``. Pressure is applied as
an equivalent force on the innermost particle layer; no boundary-traction owner
is required. The axial grid velocity is prescribed to zero (plane strain), and
physical body drag dissipates the transient. This is an actual explicit MPM
relaxation from F=I, not a prestressed analytic-force cancellation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

import phydrax as phx


class CylinderRelaxationResult(NamedTuple):
    final_state: phx.discretization.MPMRuntimeState
    initial_position: Array
    accepted: Array
    kinetic_energy: Array
    mean_radial_displacement: Array


def _pressure_and_drag(
    time: Array,
    position: Array,
    velocity: Array,
    loads: Array,
) -> Array:
    del time, position
    return loads[:, :2] - loads[:, 2, None] * velocity


def relax_cylinder(
    *,
    radial_cells: int = 24,
    steps: int = 4000,
    step_size: float = 0.003,
) -> CylinderRelaxationResult:
    """Return the accepted relaxation trajectory diagnostics and final particle state.

    Radii are 1 and 2; length is 0.25; density=1, mu=1, K=2, pressure=1e-3.
    The small pressure keeps finite-strain corrections below the spatial error.
    Two radial particles per cell integrate full-ring volumes by midpoint rule.
    """
    if radial_cells < 8 or radial_cells % 4 != 0:
        raise ValueError("radial_cells must be a multiple of four and at least eight.")
    if steps < 1 or step_size <= 0.0:
        raise ValueError("steps and step_size must be positive.")
    inner, outer, length, pressure = 1.0, 2.0, 0.25, 1e-3
    spacing = (outer - inner) / radial_cells
    dr, dz = spacing / 2.0, length / 4.0
    radial, axial = jnp.meshgrid(
        inner + (jnp.arange(2 * radial_cells, dtype=jnp.float64) + 0.5) * dr,
        (jnp.arange(4, dtype=jnp.float64) + 0.5) * dz,
        indexing="ij",
    )
    position = jnp.stack((radial, axial), axis=-1).reshape(-1, 2)
    volume = 2.0 * jnp.pi * position[:, 0] * dr * dz
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                9 * radial_cells // 4 + 1, periodic=False, endpoint=True
            ),
            phx.discretization.UniformAxisSpec(4, periodic=True, endpoint=False),
        ),
        axis_names=("r", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0), (2.25, length)), dtype=jnp.float64))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0], dtype=jnp.int32),
        volume,
        ambient_dimension=2,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        boundary="reject",
    ).prepare(particles)
    mask = jnp.zeros(grid.vertices().shape + (2,), dtype=jnp.bool_).at[..., 1].set(True)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "axisymmetric-pressurized-cylinder",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3),
            external_acceleration=_pressure_and_drag,
            external_acceleration_id="ring-pressure-and-linear-body-drag",
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(
            geometry=phx.discretization.MPMGeometryPlan(
                "axisymmetric", spatial_dimension=2
            ),
        ),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(((0.875, 0.0), (2.0625, length)), dtype=jnp.float64),
            periodic=(False, True),
            support_margin=(1.5 * spacing, 0.0),
        ),
        boundary=phx.discretization.PrescribedGridVelocityPlan(mask),
    )
    # Each loaded particle represents one axial strip of the complete inner ring.
    # Sum(force_r)=2*pi*inner*length*pressure, with no kernel-side geometric factor.
    layer = position[:, 0] < inner + dr
    force = jnp.where(layer, 2.0 * jnp.pi * inner * dz * pressure, 0.0)
    loads = jnp.stack(
        (force / volume, jnp.zeros_like(force), jnp.full_like(force, 2.0)), axis=-1
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(1.0, 2.0),
        loads,
    )
    initial = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )

    def advance(
        state: phx.discretization.MPMRuntimeState,
        unused: None,
    ) -> tuple[phx.discretization.MPMRuntimeState, tuple[Array, Array, Array]]:
        del unused
        detail = compiled.dynamics.step_detailed(state, step_size, arguments)
        accepted = detail.accepted_state
        kinetic = 0.5 * jnp.sum(volume * jnp.sum(accepted.particles.velocity**2, axis=-1))
        displacement = jnp.sum(
            volume * (accepted.particles.position[:, 0] - position[:, 0])
        ) / jnp.sum(volume)
        return accepted, (detail.successful, kinetic, displacement)

    final, (accepted, kinetic, displacement) = jax.jit(
        lambda state: jax.lax.scan(advance, state, None, length=steps)
    )(initial)
    return CylinderRelaxationResult(final, position, accepted, kinetic, displacement)


def run() -> dict[str, float | bool | int]:
    result = relax_cylinder()
    if not bool(jnp.all(result.accepted)):
        raise RuntimeError("Axisymmetric cylinder relaxation rejected a step.")
    r = result.initial_position[:, 0]
    # Independent linear-elastic plane-strain Lamé displacement:
    # sigma_r=A-B/r^2, sigma_theta=A+B/r^2, u=A*r/[2(mu+lambda)]+B/[2*mu*r].
    mu, lame_lambda, pressure = 1.0, 4.0 / 3.0, 1e-3
    a, b = pressure / 3.0, 4.0 * pressure / 3.0
    reference = a * r / (2.0 * (mu + lame_lambda)) + b / (2.0 * mu * r)
    displacement = result.final_state.particles.position[:, 0] - r
    return {
        "successful": True,
        "accepted_steps": int(result.final_state.accepted_step),
        "ring_volume": float(jnp.sum(result.final_state.particles.reference_volume)),
        "relative_displacement_error": float(
            jnp.linalg.norm(displacement - reference) / jnp.linalg.norm(reference)
        ),
        "final_kinetic_energy": float(result.kinetic_energy[-1]),
        "peak_kinetic_energy": float(jnp.max(result.kinetic_energy)),
    }


if __name__ == "__main__":
    print(run())
