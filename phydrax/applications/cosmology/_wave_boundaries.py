#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Boundary contracts for periodic and explicitly isolated wave domains."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PeriodicWaveBoundaryDescriptor(StrictModule, NonTrainableState):
    """Exact periodic identification with a mean-zero peculiar-potential gauge."""

    potential_gauge: str = eqx.field(static=True)
    descriptor_id: str = eqx.field(static=True)

    def __init__(self):
        self.potential_gauge = "volume-weighted-mean-zero"
        self.descriptor_id = canonical_fingerprint(
            {
                "kind": "periodic-wave-boundary",
                "potential_gauge": self.potential_gauge,
            }
        )


class IsolatedPotentialGauge(StrictModule, NonTrainableState):
    """Finite-domain multipole potential reference for an isolated solve."""

    expansion_center: tuple[float, ...] = eqx.field(static=True)
    multipole_order: int = eqx.field(static=True)
    reference_radius: float = eqx.field(static=True)
    reference_potential: float = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)

    def __init__(
        self,
        expansion_center: Sequence[float],
        reference_radius: float,
        /,
        *,
        multipole_order: int = 2,
        reference_potential: float = 0.0,
    ):
        center = tuple(float(value) for value in expansion_center)
        radius = float(reference_radius)
        order = int(multipole_order)
        potential = float(reference_potential)
        if (
            not center
            or len(center) > 3
            or not all(isfinite(value) for value in center)
            or not isfinite(radius)
            or radius <= 0.0
            or order < 0
            or not isfinite(potential)
        ):
            raise ValueError("Isolated finite-domain potential gauge is invalid.")
        self.expansion_center = center
        self.multipole_order = order
        self.reference_radius = radius
        self.reference_potential = potential
        self.gauge_id = canonical_fingerprint(
            {
                "kind": "isolated-finite-domain-multipole-gauge",
                "center": center,
                "multipole_order": order,
                "reference_radius": radius,
                "reference_potential": potential,
            }
        )


class AbsorbingWaveResult(StrictModule):
    psi: Array
    candidate_psi: Array
    attenuation: Array
    initial_probability: Array
    candidate_probability: Array
    probability_loss_fraction: Array
    finite: Array
    successful: Array
    policy_id: str = eqx.field(static=True)


class AbsorbingWaveBoundaryPolicy(StrictModule, NonTrainableState):
    """Explicit nonunitary sponge applied only inside a named boundary width."""

    width: float = eqx.field(static=True)
    strength: float = eqx.field(static=True)
    polynomial_order: int = eqx.field(static=True)
    maximum_probability_loss_fraction: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        width: float,
        strength: float,
        /,
        *,
        polynomial_order: int = 2,
        maximum_probability_loss_fraction: float = 0.05,
    ):
        width_ = float(width)
        strength_ = float(strength)
        order = int(polynomial_order)
        loss = float(maximum_probability_loss_fraction)
        if (
            not isfinite(width_)
            or width_ <= 0.0
            or not isfinite(strength_)
            or strength_ < 0.0
            or order < 1
            or not isfinite(loss)
            or not 0.0 <= loss < 1.0
        ):
            raise ValueError("Absorbing-wave boundary policy is invalid.")
        self.width = width_
        self.strength = strength_
        self.polynomial_order = order
        self.maximum_probability_loss_fraction = loss
        self.policy_id = canonical_fingerprint(
            {
                "kind": "absorbing-wave-boundary-policy",
                "width": width_,
                "strength": strength_,
                "polynomial_order": order,
                "maximum_probability_loss_fraction": loss,
            }
        )

    def attenuation(
        self,
        coordinates: ArrayLike,
        bounds: ArrayLike,
        action_factor: ArrayLike,
        /,
    ) -> Array:
        """Return the real sponge multiplier; this is never a periodic wrap."""
        points = jnp.asarray(coordinates)
        if not jnp.issubdtype(points.dtype, jnp.floating):
            raise TypeError("Absorbing coordinates must have a real floating dtype.")
        limits = jnp.asarray(bounds, dtype=points.dtype)
        action = jnp.asarray(action_factor, dtype=points.dtype)
        if points.ndim < 1:
            raise ValueError("Absorbing coordinates require a final spatial axis.")
        dimension = points.shape[-1]
        if limits.shape != (2, dimension) or action.shape != ():
            raise ValueError("Absorbing bounds or action factor have invalid shape.")
        invalid = (
            ~jnp.all(jnp.isfinite(points))
            | ~jnp.all(jnp.isfinite(limits))
            | ~jnp.isfinite(action)
            | jnp.any(limits[1] <= limits[0])
        )
        lower_distance = points - limits[0]
        upper_distance = limits[1] - points
        distance = jnp.minimum(lower_distance, upper_distance)
        depth = jnp.maximum(1.0 - distance / self.width, 0.0)
        rate = self.strength * jnp.sum(depth**self.polynomial_order, axis=-1)
        attenuation = jnp.exp(-jnp.abs(action) * rate)
        return eqx.error_if(
            attenuation,
            invalid,
            "Absorbing coordinates, bounds, and action factor must be finite, "
            "with strictly increasing bounds.",
        )

    def apply(
        self,
        psi: ArrayLike,
        coordinates: ArrayLike,
        bounds: ArrayLike,
        action_factor: ArrayLike,
        probability_weights: ArrayLike,
        /,
    ) -> AbsorbingWaveResult:
        values = jnp.asarray(psi)
        if not jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Absorbing wave action requires complex psi.")
        weights = jnp.asarray(probability_weights, dtype=values.real.dtype)
        if weights.shape != values.shape:
            raise ValueError("Absorbing probability weights must match psi shape.")
        attenuation = self.attenuation(coordinates, bounds, action_factor)
        if attenuation.shape != values.shape:
            raise ValueError("Absorbing coordinates must match psi leading shape.")
        candidate = values * attenuation
        initial_probability = jnp.sum(weights * jnp.abs(values) ** 2)
        candidate_probability = jnp.sum(weights * jnp.abs(candidate) ** 2)
        loss = (initial_probability - candidate_probability) / initial_probability
        finite = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(weights))
            & jnp.all(weights > 0.0)
            & jnp.isfinite(initial_probability)
            & jnp.isfinite(candidate_probability)
            & jnp.isfinite(loss)
            & (initial_probability > 0.0)
        )
        successful = (
            finite & (loss >= 0.0) & (loss <= self.maximum_probability_loss_fraction)
        )
        return AbsorbingWaveResult(
            jnp.where(successful, candidate, values),
            candidate,
            attenuation,
            initial_probability,
            candidate_probability,
            loss,
            finite,
            successful,
            self.policy_id,
        )


class IsolatedWaveBoundaryDescriptor(StrictModule, NonTrainableState):
    """Nonperiodic wave boundary with explicit gravity gauge and wave policy."""

    gauge: IsolatedPotentialGauge
    absorbing: AbsorbingWaveBoundaryPolicy | None
    descriptor_id: str = eqx.field(static=True)

    def __init__(
        self,
        gauge: IsolatedPotentialGauge,
        /,
        *,
        absorbing: AbsorbingWaveBoundaryPolicy | None = None,
    ):
        if not isinstance(gauge, IsolatedPotentialGauge):
            raise TypeError("Isolated wave boundaries require IsolatedPotentialGauge.")
        if absorbing is not None and not isinstance(
            absorbing, AbsorbingWaveBoundaryPolicy
        ):
            raise TypeError("absorbing must be AbsorbingWaveBoundaryPolicy or None.")
        self.gauge = gauge
        self.absorbing = absorbing
        self.descriptor_id = canonical_fingerprint(
            {
                "kind": "isolated-wave-boundary",
                "potential_gauge": gauge.gauge_id,
                "absorbing": None if absorbing is None else absorbing.policy_id,
                "periodic_alias": False,
            }
        )


WaveBoundaryDescriptor = PeriodicWaveBoundaryDescriptor | IsolatedWaveBoundaryDescriptor


__all__ = [
    "AbsorbingWaveBoundaryPolicy",
    "AbsorbingWaveResult",
    "IsolatedPotentialGauge",
    "IsolatedWaveBoundaryDescriptor",
    "PeriodicWaveBoundaryDescriptor",
    "WaveBoundaryDescriptor",
]
