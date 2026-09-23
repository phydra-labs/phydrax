#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


first = jr.normal(jr.key(12), (48, 48))
second = jnp.zeros_like(first).at[2:, :-1].set(first[:-2, 1:])
geometry = phx.imaging.ImagePlaneSupport(first.shape)
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
coordinate_contract = phx.SpatialCoordinateContract(
    phx.units.MILLIMETER,
    coordinate_system="image-calibration",
    reference_frame="synthetic-piv",
)
calibration = phx.velocimetry.piv.AffinePixelMap2D(
    jnp.asarray([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]]),
    coordinate_contract,
)
physical = phx.velocimetry.piv.convert_to_physical(
    result.replaced,
    calibration,
    delta_t=0.01,
    time_unit=phx.units.SECOND,
)

valid_fraction = jnp.mean(result.raw.valid)
status_ok = jnp.all(
    jnp.where(
        result.raw.valid,
        result.status.correlated & result.status.peak_fitted & result.status.validated,
        True,
    )
)
evidence_ok = jnp.array_equal(
    result.validation_evidence.valid,
    result.validated.valid,
)
physical_ok = jnp.array_equal(physical.valid, result.replaced.valid) & jnp.all(
    jnp.where(
        physical.valid[..., None],
        jnp.isfinite(physical.velocity_xy),
        True,
    )
)
expected_ok = jnp.allclose(
    median_displacement,
    jnp.asarray([2.0, -1.0]),
    atol=0.2,
)
qualified = status_ok & evidence_ok & physical_ok & (valid_fraction >= 0.5) & expected_ok
if not bool(qualified):
    raise RuntimeError("Synthetic PIV status, evidence, or displacement check failed")

print("median image displacement (row, column)", median_displacement)
print("valid vector fraction", float(valid_fraction))
print(
    "median physical velocity (x, y)",
    jnp.median(physical.velocity_xy[physical.valid], axis=0),
)
print("successful", True)
