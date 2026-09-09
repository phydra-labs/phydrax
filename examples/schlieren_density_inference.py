#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiate a path-integrated Schlieren prediction against detector data."""

import json

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


spatial = phx.SpatialCoordinateContract(
    phx.units.METER,
    coordinate_system="cartesian-world",
    reference_frame="test-section",
)
image = phx.imaging.ImagePlaneSupport((2, 2), detector_frame_id="camera")
rays = phx.measurement.RaySampleSupport(
    np.zeros((4, 3)),
    np.tile((0.0, 0.0, 1.0), (4, 1)),
    tuple(f"pixel-{index}" for index in range(4)),
    spatial,
)
density_gradient_unit = phx.units.derived_unit(
    "kg/m4", ((phx.units.KILOGRAM, 1), (phx.units.METER, -4))
)
coefficient_unit = phx.units.derived_unit(
    "m3/kg", ((phx.units.METER, 3), (phx.units.KILOGRAM, -1))
)
gradient_quantity = phx.measurement.QuantitySpec(
    "schlieren",
    "density-gradient",
    "density-gradient",
    density_gradient_unit,
    "physical.density-gradient",
)
deflection_quantity = phx.measurement.QuantitySpec(
    "schlieren",
    "angular-deflection",
    "angular-deflection",
    phx.units.RADIAN,
    "sensor.angular-deflection",
)
plan = phx.imaging.SchlierenDeflectionPlan(
    rays,
    image,
    np.full((4, 4), 0.25),
    np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
    gradient_quantity,
    deflection_quantity,
    relation=phx.imaging.GladstoneDaleRelation(0.2, coefficient_unit, "synthetic-medium"),
    gradient_frame_id="test-section",
    small_angle_limit=0.5,
)
target = jnp.full((2, 2), 0.3)


def loss(scale):
    gradients = jnp.zeros((4, 4, 3)).at[..., 0].set(scale)
    predicted = plan.evaluate(gradients)
    return jnp.mean((predicted.deflection_rc[..., 0] - target) ** 2)


value, derivative = jax.value_and_grad(loss)(jnp.asarray(1.0))
if not (jnp.isfinite(value) & jnp.isfinite(derivative)):
    raise RuntimeError("Schlieren measurement-space loss is not differentiable.")
print(
    json.dumps(
        {
            "loss": float(value),
            "gradient": float(derivative),
            "target_deflection_rad": 0.3,
        },
        indent=2,
    )
)
