#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp

import phydrax as phx


grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformCellAxisSpec(16),
        phx.discretization.UniformCellAxisSpec(16),
    ),
    axis_names=("x", "y"),
).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
boundaries = phx.discretization.MACBoundaryPlan(mac).prepare()
projection = phx.solver.MACFreeSurfaceProjectionPlan(
    mac, boundaries=boundaries, tolerance=1.0e-7
)
ghost = phx.solver.MACGhostFluidProjectionPlan(projection)

x = jnp.linspace(0.3, 0.7, 6)
y = jnp.linspace(0.3, 0.7, 6)
xx, yy = jnp.meshgrid(x, y, indexing="ij")
position = jnp.stack((xx.reshape((-1,)), yy.reshape((-1,))), axis=-1)
particle_support = phx.discretization.ParticleSetPlan(
    jnp.arange(position.shape[0]),
    jnp.full((position.shape[0],), 1.0 / position.shape[0]),
    ambient_dimension=2,
).prepare()
population = phx.discretization.ParticlePopulationPlan(particle_support).initialize()
particles = phx.discretization.flip.FLIPParticleState(position, jnp.zeros_like(position))

interface = phx.discretization.flip.ParticleLevelSetPlan(
    grid, 0.075, narrow_band_cells=4
).evaluate(position, population.active)
capillary = phx.discretization.finite_volume.MACGhostFluidCapillaryPlan(
    0.07, interface_width=0.08
).evaluate(interface)
zero_velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
projected = ghost.project(
    zero_velocity,
    interface,
    1.0e-3,
    pressure_jump=capillary.pressure_jump,
)


# Stationary cut-cell cylinder. Viscous measures accept only qualified sharp
# geometry, so the exact circle distance is enclosed rather than ramped.
def solid_sdf(points: Any, time: Any, args: Any) -> Any:
    del time, args
    return jnp.sqrt(jnp.sum((points - jnp.asarray([0.15, 0.5])) ** 2, axis=-1)) - 0.08


solid = phx.discretization.MACExactSDFMeasurePlan(
    mac,
    solid_sdf,
    # ty: ignore[invalid-argument-type]
    phx.geometry.ExactSDFEnclosureCertificate(
        phx.geometry.exact_signed_distance_certificate(smooth=False)
    ),
    source_id="stationary-cylinder",
    subdivisions=16,
).prepare(0.0)
measures = phx.discretization.finite_volume.MACFreeSurfaceViscousMeasurePlan(
    mac, 1.0
).evaluate(interface, 0.1, solid=solid)
momentum = phx.discretization.MACMomentumPlan(mac, boundaries=boundaries).prepare()
viscous = phx.solver.MACVariationalViscosityPlan(momentum, tolerance=1.0e-7).solve(
    projected.velocity,
    measures.face_density,
    measures.cell_viscosity,
    1.0e-3,
    boundaries.homogeneous_stage(),
)
if not bool(
    interface.successful & projected.successful & solid.accepted & viscous.successful
):
    raise RuntimeError(
        "Advanced FLIP interface, projection, geometry, or viscosity failed"
    )

print(
    {
        "interface_successful": True,
        "ghost_projection_successful": True,
        "surface_energy": float(capillary.surface_energy),
        "cut_geometry_successful": True,
        "viscous_successful": True,
        "viscous_dissipation": float(viscous.dissipation),
    }
)
