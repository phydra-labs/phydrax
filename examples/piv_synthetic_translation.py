#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


first = jr.normal(jr.key(12), (48, 48))
second = jnp.zeros_like(first).at[2:, :-1].set(first[:-2, 1:])
geometry = phx.velocimetry.imaging.ImageGeometry2D(first.shape)
plan = phx.velocimetry.piv.PIVPlan(
    (phx.velocimetry.piv.PIVPassPlan(16, 8, 4),),
    correlation_mode="extended",
    minimum_valid_fraction=0.5,
    minimum_peak_ratio=0.0,
    minimum_correlation=-1.0,
    minimum_neighbors=0,
    replacement_iterations=0,
    chunk_size=4,
)
result = phx.velocimetry.piv.piv(
    first,
    second,
    plan,
    geometry=geometry,
    delta_t=0.01,
)
valid_displacement = result.raw.displacement_rc[result.raw.valid]
median_displacement = jnp.median(valid_displacement, axis=0)
calibration = phx.velocimetry.piv.AffinePixelMap2D(
    jnp.asarray([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]]),
    spatial_unit="mm",
)
physical = phx.velocimetry.piv.convert_to_physical(
    result.replaced,
    calibration,
    delta_t=0.01,
    time_unit="s",
)

print("median image displacement (row, column)", median_displacement)
print("valid vector fraction", float(jnp.mean(result.raw.valid)))
print("median physical velocity (x, y)", jnp.median(physical.velocity_xy, axis=(0, 1)))
print(
    "successful",
    bool(jnp.allclose(median_displacement, jnp.asarray([2.0, -1.0]), atol=0.2)),
)
