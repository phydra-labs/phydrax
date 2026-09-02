#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._strict import StrictModule


class PrivacyEvidence(StrictModule):
    sensitivity: Array
    noise_standard_deviation: Array
    projection_residual: Array
    finite: Array
    released: Array
    key_fingerprint: Array
    mechanism: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        sensitivity: ArrayLike,
        noise_standard_deviation: ArrayLike,
        projection_residual: ArrayLike,
        finite: ArrayLike,
        released: ArrayLike,
        key_fingerprint: ArrayLike,
        mechanism: str,
    ):
        self.sensitivity = jnp.asarray(sensitivity)
        self.noise_standard_deviation = jnp.asarray(noise_standard_deviation)
        self.projection_residual = jnp.asarray(projection_residual)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.released = jnp.asarray(released, dtype=bool)
        self.key_fingerprint = jnp.asarray(key_fingerprint, dtype=jnp.uint32)
        self.mechanism = str(mechanism)


class PrivateRelease(StrictModule):
    value: Array
    evidence: PrivacyEvidence

    def __init__(self, value: ArrayLike, evidence: PrivacyEvidence, /):
        self.value = jnp.asarray(value)
        self.evidence = evidence


class RiemannianOutputGaussianMechanism(StrictModule):
    """Ambient Gaussian mechanism followed by deterministic manifold projection.

    Privacy is inherited solely by post-processing. No intrinsic exponential-map
    Gaussian or sampled sensitivity claim is made.
    """

    projection: Callable[[Array], Array]
    sensitivity: float = eqx.field(static=True)
    noise_multiplier: float = eqx.field(static=True)
    projection_tolerance: float = eqx.field(static=True)
    sensitivity_certified: bool = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)

    def __init__(
        self,
        projection: Callable[[Array], Array],
        /,
        *,
        sensitivity: float,
        noise_multiplier: float,
        sensitivity_certified: bool,
        projection_tolerance: float = 1e-6,
        mechanism_id: str = "riemannian-output-gaussian",
    ):
        if not callable(projection):
            raise TypeError("projection must be callable.")
        if float(sensitivity) <= 0.0 or float(noise_multiplier) <= 0.0:
            raise ValueError("sensitivity and noise_multiplier must be positive.")
        if float(projection_tolerance) <= 0.0:
            raise ValueError("projection_tolerance must be positive.")
        if not bool(sensitivity_certified):
            raise ValueError(
                "Output Gaussian preparation requires certified sensitivity."
            )
        self.projection = projection
        self.sensitivity = float(sensitivity)
        self.noise_multiplier = float(noise_multiplier)
        self.projection_tolerance = float(projection_tolerance)
        self.sensitivity_certified = True
        self.mechanism_id = str(mechanism_id)

    @property
    def noise_standard_deviation(self) -> float:
        return self.sensitivity * self.noise_multiplier

    def release(self, value: ArrayLike, key: Array, /) -> PrivateRelease:
        ambient = jnp.asarray(value)
        key_fingerprint = jnp.bitwise_xor.reduce(jax.random.key_data(key))
        if not jnp.issubdtype(ambient.dtype, jnp.inexact):
            raise TypeError("Private release value must have an inexact dtype.")
        noise = jax.random.normal(key, ambient.shape, dtype=ambient.real.dtype)
        if jnp.issubdtype(ambient.dtype, jnp.complexfloating):
            key, imaginary_key = jax.random.split(key)
            noise = (
                noise
                + 1j
                * jax.random.normal(
                    imaginary_key, ambient.shape, dtype=ambient.real.dtype
                )
            ) / jnp.sqrt(jnp.asarray(2.0, dtype=ambient.real.dtype))
        noisy = (
            ambient
            + jnp.asarray(self.noise_standard_deviation, dtype=ambient.real.dtype) * noise
        )
        projected = jnp.asarray(self.projection(noisy), dtype=ambient.dtype)
        if projected.shape != ambient.shape:
            raise ValueError("Projection must preserve the ambient release shape.")
        reprojection = jnp.asarray(self.projection(projected), dtype=ambient.dtype)
        residual = jnp.max(jnp.abs(reprojection - projected))
        finite = jnp.all(jnp.isfinite(projected)) & jnp.isfinite(residual)
        released = finite & (residual <= self.projection_tolerance)
        safe = jnp.where(released, projected, jnp.full_like(projected, jnp.nan))
        evidence = PrivacyEvidence(
            sensitivity=self.sensitivity,
            noise_standard_deviation=self.noise_standard_deviation,
            projection_residual=residual,
            finite=finite,
            released=released,
            key_fingerprint=key_fingerprint,
            mechanism=self.mechanism_id,
        )
        return PrivateRelease(safe, evidence)


class TangentNoiseFrame(StrictModule):
    """Certified fixed-capacity tangent Gaussian sampler."""

    sample_function: Callable[[PyTree, Array], PyTree]
    noise_dimension: int = eqx.field(static=True)
    maximum_isotropy_residual: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_function: Callable[[PyTree, Array], PyTree],
        /,
        *,
        noise_dimension: int,
        maximum_isotropy_residual: float,
        tolerance: float = 1e-6,
        frame_id: str,
    ):
        if not callable(sample_function):
            raise TypeError("sample_function must be callable.")
        if int(noise_dimension) < 1 or float(tolerance) <= 0.0:
            raise ValueError("noise_dimension and tolerance must be positive.")
        if not jnp.isfinite(maximum_isotropy_residual) or float(
            maximum_isotropy_residual
        ) > float(tolerance):
            raise ValueError("Tangent frame isotropy is not certified within tolerance.")
        self.sample_function = sample_function
        self.noise_dimension = int(noise_dimension)
        self.maximum_isotropy_residual = float(maximum_isotropy_residual)
        self.tolerance = float(tolerance)
        self.frame_id = str(frame_id)

    def sample(self, parameters: PyTree, key: Array, /) -> PyTree:
        return self.sample_function(parameters, key)


class RDPLedger(StrictModule):
    """Immutable finite-alpha RDP composition ledger."""

    orders: Array
    epsilon: Array
    steps: Array
    sampler: Literal["full_batch", "poisson"] = eqx.field(static=True)
    sampling_probability: float = eqx.field(static=True)

    def __init__(
        self,
        orders: Sequence[float],
        /,
        *,
        sampler: Literal["full_batch", "poisson"] = "full_batch",
        sampling_probability: float = 1.0,
    ):
        orders_ = jnp.asarray(tuple(float(value) for value in orders))
        if orders_.ndim != 1 or orders_.size == 0 or bool(jnp.any(orders_ <= 1.0)):
            raise ValueError(
                "RDP orders must be a nonempty finite grid strictly above one."
            )
        if sampler not in ("full_batch", "poisson"):
            raise ValueError("Only full_batch and poisson sampling are supported.")
        probability = float(sampling_probability)
        if sampler == "full_batch" and probability != 1.0:
            raise ValueError("full_batch sampling_probability must equal one.")
        if sampler == "poisson" and not (0.0 < probability <= 1.0):
            raise ValueError("Poisson sampling_probability must lie in (0, 1].")
        self.orders = orders_
        self.epsilon = jnp.zeros_like(orders_)
        self.steps = jnp.asarray(0, dtype=jnp.int32)
        self.sampler = sampler
        self.sampling_probability = probability

    def compose_gaussian(
        self, sensitivity: ArrayLike, noise_standard_deviation: ArrayLike, /
    ) -> RDPLedger:
        sensitivity_ = jnp.asarray(sensitivity, dtype=self.orders.dtype)
        sigma = jnp.asarray(noise_standard_deviation, dtype=self.orders.dtype)
        base = self.orders * sensitivity_**2 / (2.0 * sigma**2)
        # Poisson mode deliberately uses the full-batch bound: valid and conservative,
        # with no unsupported amplification claim.
        increment = base
        result = object.__new__(RDPLedger)
        object.__setattr__(result, "orders", self.orders)
        object.__setattr__(result, "epsilon", self.epsilon + increment)
        object.__setattr__(result, "steps", self.steps + 1)
        object.__setattr__(result, "sampler", self.sampler)
        object.__setattr__(
            result,
            "sampling_probability",
            self.sampling_probability,
        )
        return result

    def epsilon_at_delta(self, delta: float, /) -> Array:
        if not (0.0 < float(delta) < 1.0):
            raise ValueError("delta must lie in (0, 1).")
        candidates = self.epsilon + jnp.log(1.0 / float(delta)) / (self.orders - 1.0)
        return jnp.min(candidates)


__all__ = [
    "PrivacyEvidence",
    "PrivateRelease",
    "RDPLedger",
    "RiemannianOutputGaussianMechanism",
    "TangentNoiseFrame",
]
