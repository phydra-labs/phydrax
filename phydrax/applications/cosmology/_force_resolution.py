#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PeriodicForceQualificationResult(StrictModule):
    reference_acceleration: Array
    candidate_acceleration: Array
    absolute_error: Array
    relative_error: Array
    maximum_absolute_error: Array
    rms_absolute_error: Array
    percentile_99_relative_error: Array
    reference_net_force: Array
    candidate_net_force: Array
    tolerance_met: Array
    finite: Array
    successful: Array


class PeriodicImageForcePlan(StrictModule, NonTrainableState):
    """Small-N softened periodic image-shell force qualification oracle."""

    box_size: tuple[float, ...] = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    image_shells: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    image_offsets: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        box_size: tuple[float, ...],
        gravitational_constant: float,
        /,
        *,
        softening: float,
        image_shells: int = 1,
        absolute_tolerance: float = 1.0e-6,
        relative_tolerance: float = 1.0e-3,
    ):
        lengths = tuple(float(value) for value in box_size)
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        shells = int(image_shells)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not lengths
            or any(not np.isfinite(value) or value <= 0.0 for value in lengths)
            or not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or shells < 0
            or not np.isfinite(absolute)
            or absolute <= 0.0
            or not np.isfinite(relative)
            or relative <= 0.0
        ):
            raise ValueError("Periodic force qualification configuration is invalid.")
        integer_offsets = np.asarray(
            tuple(product(range(-shells, shells + 1), repeat=len(lengths))),
            dtype=float,
        )
        self.box_size = lengths
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.image_shells = shells
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.image_offsets = jnp.asarray(integer_offsets * np.asarray(lengths)[None, :])
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-softened-image-force-qualification",
                "box_size": list(lengths),
                "gravitational_constant": gravity,
                "softening": epsilon,
                "image_shells": shells,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
            }
        )

    def acceleration(self, positions: ArrayLike, masses: ArrayLike, /) -> Array:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        if (
            position.ndim != 2
            or position.shape[1] != len(self.box_size)
            or mass.shape != (position.shape[0],)
        ):
            raise ValueError("Periodic force positions/masses have incompatible shapes.")
        position = eqx.error_if(
            position,
            jnp.any(~jnp.isfinite(position))
            | jnp.any(~jnp.isfinite(mass))
            | jnp.any(mass <= 0.0),
            "Periodic force inputs must be finite with positive masses.",
        )
        target = position[:, None, None, :]
        source = position[None, :, None, :] + self.image_offsets[None, None, :, :]
        displacement = source - target
        squared = jnp.sum(displacement**2, axis=-1) + self.softening**2
        zero_offset = jnp.all(self.image_offsets == 0.0, axis=-1)
        self_pair = (
            jnp.eye(position.shape[0], dtype=bool)[:, :, None]
            & zero_offset[None, None, :]
        )
        inverse_cube = jnp.where(self_pair, 0.0, squared ** (-1.5))
        contribution = (
            self.gravitational_constant
            * mass[None, :, None, None]
            * displacement
            * inverse_cube[..., None]
        )
        return jnp.sum(contribution, axis=(1, 2))

    def qualify(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        candidate_acceleration: ArrayLike,
        /,
    ) -> PeriodicForceQualificationResult:
        reference = self.acceleration(positions, masses)
        candidate = jnp.asarray(candidate_acceleration, dtype=reference.dtype)
        mass = jnp.asarray(masses, dtype=reference.dtype)
        if candidate.shape != reference.shape:
            raise ValueError("Candidate acceleration must match reference shape.")
        difference = candidate - reference
        absolute = jnp.sqrt(jnp.sum(difference**2, axis=-1))
        reference_norm = jnp.sqrt(jnp.sum(reference**2, axis=-1))
        relative = absolute / jnp.maximum(reference_norm, self.absolute_tolerance)
        maximum_absolute = jnp.max(absolute)
        rms_absolute = jnp.sqrt(jnp.mean(absolute**2))
        sorted_relative = jnp.sort(relative)
        percentile_index = jnp.minimum(
            jnp.asarray(jnp.ceil(0.99 * sorted_relative.size) - 1, dtype=jnp.int32),
            sorted_relative.size - 1,
        )
        percentile = sorted_relative[percentile_index]
        tolerance_met = (maximum_absolute <= self.absolute_tolerance) | (
            percentile <= self.relative_tolerance
        )
        finite = jnp.all(jnp.isfinite(reference)) & jnp.all(jnp.isfinite(candidate))
        return PeriodicForceQualificationResult(
            reference,
            candidate,
            absolute,
            relative,
            maximum_absolute,
            rms_absolute,
            percentile,
            jnp.sum(mass[:, None] * reference, axis=0),
            jnp.sum(mass[:, None] * candidate, axis=0),
            tolerance_met,
            finite,
            tolerance_met & finite,
        )


__all__ = ["PeriodicForceQualificationResult", "PeriodicImageForcePlan"]
