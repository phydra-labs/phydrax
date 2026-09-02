#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._symmetric_coordinates import smat, svec, symmetric_packed_dimension
from ..linalg import FactorizationPolicy, inverse, OperatorProperties
from ._contracts import (
    _AbstractAnalyticExponentialFamily,
    _mean_domain_result,
    _natural_domain_result,
    ExponentialFamilyDomainResult,
    ExponentialFamilySignature,
    NaturalCoordinates,
    StatisticBatch,
)


def _error_if(value: Array, predicate: Array, message: str, /) -> Array:
    if isinstance(predicate, jax_core.Tracer):
        return eqx.error_if(value, predicate, message)
    if bool(predicate):
        raise eqx.EquinoxRuntimeError(message)
    return value


def _positive_definite_inverse(matrix: Array, /) -> Array:
    result = inverse(
        matrix,
        FactorizationPolicy("cholesky"),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "asserted",
                "positive_definite": "asserted",
            },
        ),
    )
    return jnp.where(
        result.successful[..., None, None],
        result.value,
        jnp.nan,
    )


class MultivariateNormalFamily(_AbstractAnalyticExponentialFamily):
    """Multivariate Normal laws in orthonormal linear-quadratic coordinates."""

    event_size: int = eqx.field(static=True)
    packed_size: int = eqx.field(static=True)
    _signature: ExponentialFamilySignature = eqx.field(static=True)

    def __init__(self, event_size: int):
        dimension = int(event_size)
        if dimension <= 0:
            raise ValueError("event_size must be positive.")
        packed = symmetric_packed_dimension(dimension)
        self.event_size = dimension
        self.packed_size = packed
        self._signature = ExponentialFamilySignature(
            "multivariate-normal",
            dimension + packed,
            (dimension,),
            "lebesgue",
            f"real-vector-{dimension}",
            f"linear-svec-quadratic-{dimension}",
        )

    @property
    def signature(self) -> ExponentialFamilySignature:
        return self._signature

    def natural_from_location_covariance(
        self, location: ArrayLike, covariance: ArrayLike, /
    ) -> NaturalCoordinates:
        """Return natural coordinates from a location and dense covariance."""
        location_array = jnp.asarray(location)
        covariance_array = jnp.asarray(covariance)
        if jnp.issubdtype(location_array.dtype, jnp.complexfloating) or jnp.issubdtype(
            covariance_array.dtype, jnp.complexfloating
        ):
            raise TypeError("Multivariate Normal parameters must be real-valued.")
        if location_array.ndim == 0 or int(location_array.shape[-1]) != self.event_size:
            raise ValueError(
                f"location must end in event_size={self.event_size}; got {location_array.shape}."
            )
        if covariance_array.ndim < 2 or covariance_array.shape[-2:] != (
            self.event_size,
            self.event_size,
        ):
            raise ValueError(
                "covariance must end in square event axes "
                f"({self.event_size}, {self.event_size}); got {covariance_array.shape}."
            )
        dtype = jnp.result_type(location_array, covariance_array, 0.0)
        location_array = location_array.astype(dtype)
        covariance_array = covariance_array.astype(dtype)
        location_array = _error_if(
            location_array,
            jnp.any(~jnp.isfinite(location_array)),
            "location must contain only finite values.",
        )
        covariance_array = _error_if(
            covariance_array,
            jnp.any(~jnp.isfinite(covariance_array)),
            "covariance must contain only finite values.",
        )
        symmetry_scale = jnp.max(jnp.abs(covariance_array), axis=(-2, -1))
        symmetry_tolerance = 64.0 * jnp.finfo(dtype).eps * symmetry_scale[..., None, None]
        covariance_array = _error_if(
            covariance_array,
            jnp.any(
                jnp.abs(covariance_array - jnp.swapaxes(covariance_array, -1, -2))
                > symmetry_tolerance
            ),
            "covariance must be symmetric.",
        )
        batch_shape = jnp.broadcast_shapes(
            location_array.shape[:-1], covariance_array.shape[:-2]
        )
        location_array = jnp.broadcast_to(
            location_array, batch_shape + (self.event_size,)
        )
        covariance_array = jnp.broadcast_to(
            covariance_array,
            batch_shape + (self.event_size, self.event_size),
        )
        precision = _positive_definite_inverse(covariance_array)
        linear = ein.contract("...ij,...j->...i", precision, location_array)
        return self.natural(jnp.concatenate((linear, svec(-0.5 * precision)), axis=-1))

    def law_from_location_covariance(self, location: ArrayLike, covariance: ArrayLike, /):
        """Return a multivariate Normal law from conventional parameters."""
        return self.law(self.natural_from_location_covariance(location, covariance))

    def location_covariance_from_natural(
        self, natural: NaturalCoordinates, /
    ) -> tuple[Array, Array]:
        domain = self.natural_domain(natural)
        location, covariance = self._location_covariance(natural.values)
        return (
            jnp.where(domain.valid[..., None], location, jnp.nan),
            jnp.where(domain.valid[..., None, None], covariance, jnp.nan),
        )

    def _split(self, values: Array, /) -> tuple[Array, Array]:
        return values[..., : self.event_size], values[..., self.event_size :]

    def _precision(self, natural_values: Array, /) -> Array:
        _, quadratic = self._split(natural_values)
        return -2.0 * smat(quadratic, matrix_dimension=self.event_size)

    def _location_covariance(self, natural_values: Array, /) -> tuple[Array, Array]:
        linear, _ = self._split(natural_values)
        precision = self._precision(natural_values)
        covariance = _positive_definite_inverse(precision)
        location = ein.contract("...ij,...j->...i", covariance, linear)
        return location, covariance

    def _natural_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        precision = self._precision(values)
        eigenvalues = jnp.linalg.eigvalsh(precision)
        scale = jnp.max(jnp.abs(eigenvalues), axis=-1)
        tolerance = 64.0 * jnp.finfo(values.dtype).eps * scale
        minimum = eigenvalues[..., 0]
        return _natural_domain_result(
            self.signature,
            values,
            interior=minimum > tolerance,
            boundary=(minimum >= -tolerance) & (minimum <= tolerance),
        )

    def _mean_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        location, second_packed = self._split(values)
        second = smat(second_packed, matrix_dimension=self.event_size)
        covariance = second - ein.contract("...i,...j->...ij", location, location)
        covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
        eigenvalues = jnp.linalg.eigvalsh(covariance)
        scale = jnp.max(jnp.abs(eigenvalues), axis=-1)
        tolerance = 64.0 * jnp.finfo(values.dtype).eps * scale
        minimum = eigenvalues[..., 0]
        return _mean_domain_result(
            self.signature,
            values,
            interior=minimum > tolerance,
            boundary=(minimum >= -tolerance) & (minimum <= tolerance),
        )

    def _sufficient_statistics(self, value: ArrayLike, /) -> StatisticBatch:
        raw = jnp.asarray(value)
        if jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise TypeError("Multivariate Normal observations must be real-valued.")
        if raw.ndim == 0 or int(raw.shape[-1]) != self.event_size:
            raise ValueError(
                "Multivariate Normal observations must end in event dimension "
                f"{self.event_size}; got {raw.shape}."
            )
        observation = raw.astype(jnp.result_type(raw, 0.0))
        valid = jnp.all(jnp.isfinite(observation), axis=-1)
        safe = jnp.where(valid[..., None], observation, 0.0)
        outer = ein.contract("...i,...j->...ij", safe, safe)
        return StatisticBatch(
            jnp.concatenate((safe, svec(outer)), axis=-1),
            valid,
            self.signature,
        )

    def _log_base_density(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        if values.ndim == 0 or int(values.shape[-1]) != self.event_size:
            raise ValueError(
                "Multivariate Normal observations have an incompatible event shape."
            )
        return jnp.zeros(values.shape[:-1], dtype=jnp.result_type(values, 0.0))

    def _log_normalizer(self, natural_values: Array, /) -> Array:
        linear, _ = self._split(natural_values)
        precision = self._precision(natural_values)
        location = jnp.linalg.solve(precision, linear[..., None])[..., 0]
        _, log_determinant = jnp.linalg.slogdet(precision)
        quadratic = jnp.sum(linear * location, axis=-1)
        return (
            0.5 * quadratic
            - 0.5 * log_determinant
            + 0.5 * self.event_size * jnp.log(2.0 * jnp.pi)
        )

    def _mean_values(self, natural_values: Array, /) -> Array:
        location, covariance = self._location_covariance(natural_values)
        second = covariance + ein.contract("...i,...j->...ij", location, location)
        return jnp.concatenate((location, svec(second)), axis=-1)

    def _natural_from_mean_values(self, mean_values: Array, /) -> Array:
        location, second_packed = self._split(mean_values)
        second = smat(second_packed, matrix_dimension=self.event_size)
        covariance = second - ein.contract("...i,...j->...ij", location, location)
        precision = _positive_definite_inverse(covariance)
        linear = ein.contract("...ij,...j->...i", precision, location)
        return jnp.concatenate((linear, svec(-0.5 * precision)), axis=-1)

    def _sample(
        self,
        key,
        natural_values: Array,
        sample_shape: tuple[int, ...],
        /,
    ) -> Array:
        location, covariance = self._location_covariance(natural_values)
        factor = jnp.linalg.cholesky(covariance)
        noise = jr.normal(
            key,
            shape=sample_shape + location.shape,
            dtype=natural_values.dtype,
        )
        transformed = ein.contract("...ij,...j->...i", factor, noise)
        return location + transformed


__all__ = ["MultivariateNormalFamily"]
