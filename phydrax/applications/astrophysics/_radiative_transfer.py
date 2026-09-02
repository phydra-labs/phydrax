#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._observation_status import AstrophysicsObservationStatus


class RayTransferResult(StrictModule):
    intensity: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class RayTransferPlan(StrictModule, NonTrainableState):
    """Absorption-emission transfer along independent prescribed rays."""

    segment_lengths: Array
    ray_count: int = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    squeeze_ray_axis: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, segment_lengths: ArrayLike, /, *, ray_id: str):
        lengths = np.asarray(segment_lengths, dtype=float)
        squeeze = lengths.ndim == 1
        if squeeze:
            lengths = lengths[None, :]
        identifier = str(ray_id)
        if (
            lengths.ndim != 2
            or lengths.shape[1] == 0
            or np.any(~np.isfinite(lengths))
            or np.any(lengths < 0.0)
            or not identifier
        ):
            raise ValueError(
                "segment_lengths must be a finite nonnegative ray-by-sample array."
            )
        self.segment_lengths = jnp.asarray(lengths)
        self.ray_count, self.sample_count = lengths.shape
        self.squeeze_ray_axis = squeeze
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ray-transfer",
                "ray_id": identifier,
                "lengths": array_tree_fingerprint(lengths),
            }
        )

    def evaluate(
        self,
        emissivity: ArrayLike,
        extinction: ArrayLike,
        incident: ArrayLike = 0.0,
        /,
    ) -> RayTransferResult:
        source = jnp.asarray(emissivity)
        opacity = jnp.asarray(extinction, dtype=source.dtype)
        if self.squeeze_ray_axis:
            source = source[None, :]
            opacity = opacity[None, :]
        expected = (self.ray_count, self.sample_count)
        if source.shape != expected or opacity.shape != expected:
            raise ValueError("Ray emissivity and extinction must match segment lengths.")
        incident_value = jnp.broadcast_to(
            jnp.asarray(incident, dtype=source.dtype), (self.ray_count,)
        )

        def one_ray(lengths, emission, absorption, initial):
            def step(intensity, sample):
                ds, emissivity_value, extinction_value = sample
                optical_depth = extinction_value * ds
                small = jnp.abs(optical_depth) < 1.0e-7
                phi = jnp.where(
                    small,
                    1.0 - 0.5 * optical_depth + optical_depth**2 / 6.0,
                    -jnp.expm1(-optical_depth) / optical_depth,
                )
                result = intensity * jnp.exp(-optical_depth) + emissivity_value * ds * phi
                return result, None

            final, _ = jax.lax.scan(
                step,
                initial,
                (lengths, emission, absorption),
            )
            return final

        intensity = jax.vmap(one_ray)(
            self.segment_lengths,
            source,
            opacity,
            incident_value,
        )
        valid = (
            jnp.all(jnp.isfinite(source), axis=-1)
            & jnp.all(source >= 0.0, axis=-1)
            & jnp.all(jnp.isfinite(opacity), axis=-1)
            & jnp.all(opacity >= 0.0, axis=-1)
            & jnp.isfinite(incident_value)
            & (incident_value >= 0.0)
        )
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.NONPHYSICAL_MODEL),
        ).astype(jnp.int32)
        output = jnp.where(valid, intensity, 0.0)
        if self.squeeze_ray_axis:
            output = output[0]
            valid = valid[0]
            status = status[0]
        return RayTransferResult(output, valid, status, self.plan_id)


class RadiativeTransferResult(StrictModule):
    emergent: Array
    iterations: Array
    residual: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class PolarizedRadiativeTransferPlan(StrictModule, NonTrainableState):
    segment_lengths: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, segment_lengths, /, *, plan_id="polarized-radiative-transfer"):
        lengths = np.asarray(segment_lengths, dtype=float)
        identifier = str(plan_id)
        if (
            lengths.ndim != 1
            or lengths.size == 0
            or np.any(~np.isfinite(lengths))
            or np.any(lengths < 0.0)
            or not identifier
        ):
            raise ValueError("Polarized segment lengths must be finite and nonnegative.")
        self.segment_lengths = jnp.asarray(lengths)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarized-radiative-transfer",
                "plan_id": identifier,
                "lengths": array_tree_fingerprint(lengths),
            }
        )

    def evaluate(
        self, emission: ArrayLike, propagation_matrix: ArrayLike, incident: ArrayLike, /
    ) -> RadiativeTransferResult:
        emission_value = jnp.asarray(emission)
        matrix = jnp.asarray(propagation_matrix, dtype=emission_value.dtype)
        incident_value = jnp.asarray(incident, dtype=emission_value.dtype)
        if emission_value.shape != (self.segment_lengths.size, 4) or matrix.shape != (
            self.segment_lengths.size,
            4,
            4,
        ):
            raise ValueError("Polarized transfer arrays have incompatible shapes.")
        if incident_value.shape != (4,):
            raise ValueError("Polarized incident Stokes vector must have shape (4,).")

        def step(stokes, values):
            ds, source, operator = values
            augmented = jnp.zeros((5, 5), dtype=stokes.dtype)
            augmented = augmented.at[:4, :4].set(-operator)
            augmented = augmented.at[:4, 4].set(source)
            propagated = jsp.linalg.expm(augmented * ds) @ jnp.concatenate(
                (stokes, jnp.ones((1,), dtype=stokes.dtype))
            )
            result = propagated[:4]
            return result, result

        emergent, history = jax.lax.scan(
            step,
            incident_value,
            (self.segment_lengths, emission_value, matrix),
        )
        valid = (
            jnp.all(jnp.isfinite(emission_value))
            & jnp.all(jnp.isfinite(matrix))
            & jnp.all(jnp.isfinite(incident_value))
            & jnp.all(jnp.isfinite(history))
        )
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        return RadiativeTransferResult(
            emergent,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=emergent.dtype),
            valid,
            status,
            self.plan_id,
        )


__all__ = [
    "PolarizedRadiativeTransferPlan",
    "RadiativeTransferResult",
    "RayTransferPlan",
    "RayTransferResult",
]
