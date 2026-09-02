#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformCellAxisSpec(48, periodic=True),
        phx.discretization.UniformCellAxisSpec(48, periodic=True),
    ),
    axis_names=("x", "y"),
).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
tracer_space = grid.field_space(
    "dye",
    entity_layout=finite_volume.cell_layout,
    dtype=mac.pressure_space.dtype,
    representation="point_value",
)
transport = phx.discretization.MACPassiveTracerMacCormackPlan(
    mac,
    tracer_space,
).prepare()

centers = finite_volume.cell_centers
tracer = jnp.exp(-160.0 * jnp.sum((centers - 0.35) ** 2, axis=-1))
velocity = (
    jnp.full(finite_volume.face_layouts[0].shape, 0.25),
    jnp.full(finite_volume.face_layouts[1].shape, -0.10),
)
result = transport.advance(tracer, velocity, jnp.asarray(0.01))

print(
    {
        "successful": bool(result.success),
        "bounded": bool(result.donor_bounded),
        "limiter_cells": int(result.limiter_active_count),
        "integral_defect": float(result.integral_defect),
        "maximum_displacement_cells": float(result.maximum_displacement_cell_widths),
    }
)
