#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import RealTrigonometricTransform


class IsotropicRadialTransformPlan(StrictModule, NonTrainableState):
    """Interior-node 3-D isotropic Fourier transform with fixed DST-I convention."""

    count: int = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    length_unit_id: str = eqx.field(static=True)
    dtype_name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        count: int,
        maximum_radius: float,
        /,
        *,
        length_unit_id: str = "dimensionless",
        dtype: Any = jnp.float64,
    ):
        size = int(count)
        radius = float(maximum_radius)
        unit = str(length_unit_id).strip()
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        if (
            size <= 0
            or not math.isfinite(radius)
            or radius <= 0.0
            or not unit
            or not jnp.issubdtype(dtype_, jnp.floating)
        ):
            raise ValueError("Isotropic radial-transform configuration is invalid.")
        self.count = size
        self.maximum_radius = radius
        self.length_unit_id = unit
        self.dtype_name = dtype_.str
        self.plan_id = canonical_fingerprint(
            {
                "kind": "isotropic-radial-transform-plan",
                "count": size,
                "maximum_radius": radius,
                "length_unit_id": unit,
                "dtype": dtype_.str,
                "radial_nodes": "r_j=(j+1)R/(N+1)",
                "wave_nodes": "k_n=(n+1)pi/R",
                "quadrature": "interior-dst-i",
            }
        )

    def prepare(self, /) -> "PreparedIsotropicRadialTransform":
        return PreparedIsotropicRadialTransform(self)


class PreparedIsotropicRadialTransform(StrictModule, NonTrainableState):
    plan: IsotropicRadialTransformPlan
    radii: Array
    wave_numbers: Array
    radial_spacing: Array
    wave_number_spacing: Array
    forward_scale: Array
    inverse_scale: Array
    transform: RealTrigonometricTransform
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: IsotropicRadialTransformPlan, /):
        if not isinstance(plan, IsotropicRadialTransformPlan):
            raise TypeError("plan must be IsotropicRadialTransformPlan.")
        dtype = np.dtype(plan.dtype_name)
        spacing = plan.maximum_radius / (plan.count + 1)
        wave_spacing = math.pi / plan.maximum_radius
        radii = spacing * np.arange(1, plan.count + 1, dtype=np.float64)
        wave = wave_spacing * np.arange(1, plan.count + 1, dtype=np.float64)
        normalization = math.sqrt((plan.count + 1) / 2.0)
        forward = 4.0 * math.pi * spacing * normalization / wave
        inverse = wave_spacing * normalization / (2.0 * math.pi**2 * radii)
        self.plan = plan
        self.radii = jnp.asarray(radii, dtype=dtype)
        self.wave_numbers = jnp.asarray(wave, dtype=dtype)
        self.radial_spacing = jnp.asarray(spacing, dtype=dtype)
        self.wave_number_spacing = jnp.asarray(wave_spacing, dtype=dtype)
        self.forward_scale = jnp.asarray(forward, dtype=dtype)
        self.inverse_scale = jnp.asarray(inverse, dtype=dtype)
        self.transform = RealTrigonometricTransform("dst", 1, plan.count, dtype=dtype)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-isotropic-radial-transform",
                "plan": plan.plan_id,
                "base_transform": self.transform.transform_id,
            }
        )

    def _map_transform(self, values: Array, /) -> Array:
        leading = values.shape[:-1]
        flattened = values.reshape((-1, self.plan.count))
        transformed = jax.vmap(self.transform.analyze)(flattened)
        return transformed.reshape(leading + (self.plan.count,))

    def forward(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values, dtype=self.radii.dtype)
        if value.shape[-1:] != (self.plan.count,):
            raise ValueError("Radial values must end with the prepared radial count.")
        transformed = self._map_transform(value * self.radii)
        return transformed * self.forward_scale

    def inverse(self, coefficients: ArrayLike, /) -> Array:
        value = jnp.asarray(coefficients, dtype=self.wave_numbers.dtype)
        if value.shape[-1:] != (self.plan.count,):
            raise ValueError("Wave-space values must end with the prepared wave count.")
        transformed = self._map_transform(value * self.wave_numbers)
        return transformed * self.inverse_scale


class RadialTransformEvidence(StrictModule):
    round_trip_relative_error: Array
    forward_reference_relative_error: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


def radial_transform_evidence(
    transform: PreparedIsotropicRadialTransform,
    values: ArrayLike,
    /,
    *,
    reference_forward: ArrayLike | None = None,
    tolerance: float = 1.0e-8,
) -> RadialTransformEvidence:
    if not isinstance(transform, PreparedIsotropicRadialTransform):
        raise TypeError("transform must be PreparedIsotropicRadialTransform.")
    value = jnp.asarray(values, dtype=transform.radii.dtype)
    forward = transform.forward(value)
    restored = transform.inverse(forward)
    tiny = jnp.finfo(value.dtype).tiny
    round_trip = jnp.linalg.norm(restored - value) / jnp.maximum(
        jnp.linalg.norm(value), tiny
    )
    if reference_forward is None:
        reference_error = jnp.asarray(0.0, dtype=value.dtype)
        reference_valid = jnp.asarray(True)
    else:
        reference = jnp.asarray(reference_forward, dtype=forward.dtype)
        if reference.shape != forward.shape:
            raise ValueError("reference_forward must match the transform output shape.")
        reference_error = jnp.linalg.norm(forward - reference) / jnp.maximum(
            jnp.linalg.norm(reference), tiny
        )
        reference_valid = jnp.all(jnp.isfinite(reference))
    threshold = jnp.asarray(float(tolerance), dtype=value.dtype)
    successful = (
        jnp.all(jnp.isfinite(value))
        & jnp.all(jnp.isfinite(forward))
        & jnp.all(jnp.isfinite(restored))
        & reference_valid
        & (round_trip <= threshold)
        & (reference_error <= threshold)
    )
    return RadialTransformEvidence(
        round_trip, reference_error, successful, transform.prepared_id
    )


__all__ = [
    "IsotropicRadialTransformPlan",
    "PreparedIsotropicRadialTransform",
    "RadialTransformEvidence",
    "radial_transform_evidence",
]
