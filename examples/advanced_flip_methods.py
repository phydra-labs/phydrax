#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

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


# Deterministic moving-solid/cut-cell geometry.
def solid_sdf(points, time, args):
    del time, args
    return jnp.sqrt(jnp.sum((points - jnp.asarray([0.15, 0.5])) ** 2, axis=-1)) - 0.08


def wall_velocity(points, time, args):
    del time, args
    return jnp.zeros_like(points)


solid_plan = phx.discretization.finite_volume.MACDiffuseSDFGeometryPlan(
    mac,
    solid_sdf,
    wall_velocity,
    field_id="stationary-cylinder",
    interface_width=0.04,
)
solid = solid_plan.evaluate(0.0)
measures = phx.discretization.finite_volume.MACFreeSurfaceViscousMeasurePlan(
    mac, 1.0
).evaluate(interface, 0.1, solid=solid)
viscous = phx.solver.MACVariationalViscosityPlan(mac, tolerance=1.0e-7).solve(
    projected.velocity, measures, 1.0e-3
)

print(
    {
        "interface_successful": bool(interface.successful),
        "ghost_projection_successful": bool(projected.successful),
        "surface_energy": float(capillary.surface_energy),
        "cut_geometry_successful": bool(solid.successful),
        "viscous_successful": bool(viscous.successful),
        "viscous_dissipation": float(viscous.dissipation),
    }
)
