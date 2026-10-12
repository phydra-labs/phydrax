#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""One saturated double-point MPM/FLIP attempt with a physical pressure reservoir.

Run: python -m examples.double_point_mpm
Printed evidence belongs to this step, not a consolidation qualification.
"""

from __future__ import annotations

import json
from math import prod

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def simulate() -> dict[str, object]:
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float64)
    fluid_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4),) * 2,
        axis_names=("x", "y"),
    ).prepare(bounds)
    finite_volume = phx.discretization.FiniteVolumePlan(fluid_grid).prepare()
    mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    # The MPM domain has a complete one-cell halo for the linear assignment.
    solid_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(7, periodic=False, endpoint=True),) * 2,
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[-0.25, -0.25], [1.25, 1.25]], dtype=jnp.float64))
    coordinates = jnp.linspace(0.125, 0.875, 4, dtype=jnp.float64)
    x, y = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    position = jnp.stack((x, y), axis=-1).reshape((-1, 2))
    count = position.shape[0]
    reference_volume = jnp.full((count,), 1.0 / 16.0, dtype=jnp.float64)
    porosity, solid_density, fluid_density = 0.4, 2650.0, 1000.0
    solid_particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count, dtype=jnp.int32),
        solid_density * (1.0 - porosity) * reference_volume,
        ambient_dimension=2,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        solid_grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(1),
        boundary="reject",
    ).prepare(solid_particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "double-point-effective-skeleton",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        solid_particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            bounds, periodic=(False, False), support_margin=0.25
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
            2.0e4, 1.0e5
        )
    )
    solid_state = compiled.initialize_state(
        position, jnp.zeros_like(position), reference_volume, arguments
    )
    fluid_particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count, dtype=jnp.int32),
        fluid_density * porosity * reference_volume,
        ambient_dimension=2,
    ).prepare()
    fluid_transfer = phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(
        fluid_particles
    )
    solid_to_mac = phx.applications.solid_mechanics.MPMToMACFaceTransferPlan(
        compiled.dynamics,
        mac,
        maximum_routes=4
        * sum(prod(layout.shape) for layout in finite_volume.face_layouts),
    ).prepare()
    # Side/bottom normal velocities are stationary; the top is a reservoir.
    nodal_x, nodal_y = np.meshgrid(*splat.layout.coordinates_by_axis, indexing="ij")
    solid_free = np.stack(((nodal_x > 0.0) & (nodal_x < 1.0), nodal_y > 0.0), axis=-1)
    fluid_x_free = np.ones(finite_volume.face_layouts[0].shape, dtype=np.bool_)
    fluid_x_free[[0, -1], :] = False
    fluid_y_free = np.ones(finite_volume.face_layouts[1].shape, dtype=np.bool_)
    fluid_y_free[:, 0] = False
    pressure_mask = np.zeros(finite_volume.cell_shape, dtype=np.bool_)
    pressure_mask[:, -1] = True
    plan = phx.solver.DoublePointMPMPlan(
        compiled.dynamics,
        fluid_transfer,
        solid_to_mac,
        phx.equations.InterphaseDragLawPlan("stokes"),
        solid_density=solid_density,
        fluid_density=fluid_density,
        fluid_viscosity=1.0e-3,
        grain_diameter=1.0e-3,
        pressure_mask=pressure_mask,
        pressure_values=0.0,
        solid_free=solid_free,
        fluid_free=(fluid_x_free, fluid_y_free),
        tolerance=1.0e-8,
        maximum_iterations=256,
    )
    # A supplied excess-pressure profile is zero on the physical reservoir row.
    initial_pressure = 1000.0 * (0.875 - y) / 0.875
    fluid_velocity = jnp.stack(
        (jnp.zeros_like(position[:, 0]), jnp.full((count,), 1.0e-5, dtype=jnp.float64)),
        axis=-1,
    )
    fluid_state = phx.discretization.flip.FLIPRuntimeState(
        particles=phx.discretization.flip.FLIPParticleState(position, fluid_velocity),
        pressure=initial_pressure,
        time=jnp.asarray(0.0, dtype=jnp.float64),
        accepted_step=jnp.asarray(0, dtype=jnp.int32),
        status=jnp.asarray(0, dtype=jnp.int32),
        geometry_epoch=jnp.asarray(-1, dtype=jnp.int32),
        geometry_id="",
    )
    state = plan.initialize(solid_state, fluid_state)
    result = phx.solver.advance_double_point_mpm_step(
        plan, state, jnp.asarray(1.0e-5, dtype=jnp.float64), arguments=arguments
    )
    return {
        "successful": bool(result.successful),
        "status": int(result.status),
        "solid_accepted_steps": int(result.accepted_state.solid.accepted_step),
        "fluid_accepted_steps": int(result.accepted_state.fluid.accepted_step),
        "solid_time_s": float(result.accepted_state.solid.time),
        "fluid_time_s": float(result.accepted_state.fluid.time),
        "porosity_min": float(jnp.min(result.porosity)),
        "porosity_max": float(jnp.max(result.porosity)),
        "transfer_successful": bool(solid_to_mac.evidence.successful),
        "transfer_required_routes": solid_to_mac.evidence.required_routes,
        "solid_transfer_momentum_defect": np.asarray(
            result.solid_momentum_defect
        ).tolist(),
        "fluid_transfer_momentum_defect": np.asarray(
            result.fluid_momentum_defect
        ).tolist(),
        "drag_law_successful": all(bool(value.successful.all()) for value in result.drag),
        "drag_reynolds_max": max(
            float(jnp.max(value.reynolds_number)) for value in result.drag
        ),
        "drag_residual_norm": float(result.drag_residual_norm),
        "drag_momentum_defect": np.asarray(result.drag_momentum_defect).tolist(),
        "effective_stress_max_pa": float(jnp.max(jnp.abs(result.effective_stress))),
        "total_stress_max_pa": float(jnp.max(jnp.abs(result.total_stress))),
        "projection_successful": bool(result.projection.successful),
        "projection_linear_status": int(result.projection.linear_status),
        "projection_iterations": int(result.projection.iterations),
        "projection_residual_norm": float(result.projection.residual_norm),
        "unconstrained_divergence_norm": float(result.projection.divergence_norm),
        "divergence_identity_defect": float(
            jnp.max(jnp.abs(result.divergence_identity_defect))
        ),
        "accepted_pressure_pa": np.asarray(result.accepted_state.fluid.pressure).tolist(),
    }


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    evidence = simulate()
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
    if not evidence["successful"]:
        raise RuntimeError(
            "Double-point attempt rejected; both phases retained their committed state."
        )


if __name__ == "__main__":
    main()
