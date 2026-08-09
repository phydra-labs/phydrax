#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._probability import AbstractProbabilityLaw
from .._strict import StrictModule
from ..domain._measure import MeasureKind


ExponentialFamilyStatus: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7]
EXPONENTIAL_FAMILY_SUCCESS: ExponentialFamilyStatus = 0
EXPONENTIAL_FAMILY_NONFINITE: ExponentialFamilyStatus = 1
EXPONENTIAL_FAMILY_INVALID_EVENT: ExponentialFamilyStatus = 2
EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN: ExponentialFamilyStatus = 3
EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN: ExponentialFamilyStatus = 4
EXPONENTIAL_FAMILY_MEAN_BOUNDARY: ExponentialFamilyStatus = 5
EXPONENTIAL_FAMILY_INSUFFICIENT_WEIGHT: ExponentialFamilyStatus = 6
EXPONENTIAL_FAMILY_NONCONVERGED: ExponentialFamilyStatus = 7


def exponential_family_status_name(status: int, /) -> str:
    """Return the stable name of one exponential-family status code."""
    names = (
        "success",
        "nonfinite",
        "invalid_event",
        "outside_natural_domain",
        "outside_mean_domain",
        "mean_boundary",
        "insufficient_weight",
        "nonconverged",
    )
    code = int(status)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown exponential-family status {code}.")
    return names[code]


class ExponentialFamilySignature(StrictModule):
    """Static identity of one intrinsic exponential-family coordinate system."""

    family_id: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    density_measure_kind: MeasureKind = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    coordinate_chart_id: str = eqx.field(static=True)

    def __init__(
        self,
        family_id: str,
        dimension: int,
        event_shape: tuple[int, ...],
        density_measure_kind: MeasureKind,
        support_id: str,
        coordinate_chart_id: str,
    ):
        dimensions = int(dimension)
        events = tuple(int(size) for size in event_shape)
        if not family_id or not support_id or not coordinate_chart_id:
            raise ValueError("Exponential-family signature IDs must be non-empty.")
        if dimensions <= 0:
            raise ValueError("Exponential-family coordinate dimension must be positive.")
        if any(size < 0 for size in events):
            raise ValueError("Exponential-family event dimensions must be non-negative.")
        if density_measure_kind not in (
            "lebesgue",
            "hausdorff",
            "probability",
            "counting",
            "dirac",
            "trajectory",
            "riemannian",
            "external",
        ):
            raise ValueError(f"Unknown density measure kind {density_measure_kind!r}.")
        self.family_id = str(family_id)
        self.dimension = dimensions
        self.event_shape = events
        self.density_measure_kind = density_measure_kind
        self.support_id = str(support_id)
        self.coordinate_chart_id = str(coordinate_chart_id)

    @property
    def key(self) -> tuple[str, int, tuple[int, ...], str, str, str]:
        return (
            self.family_id,
            self.dimension,
            self.event_shape,
            self.density_measure_kind,
            self.support_id,
            self.coordinate_chart_id,
        )


def _coordinate_array(values: ArrayLike, signature: ExponentialFamilySignature) -> Array:
    array = jnp.asarray(values)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        if array.weak_type:
            array = array.astype(float)
        else:
            raise TypeError(
                "Exponential-family coordinates must be real floating arrays."
            )
    if array.ndim == 0 or int(array.shape[-1]) != signature.dimension:
        raise ValueError(
            "Exponential-family coordinates must end in intrinsic dimension "
            f"{signature.dimension}; got {array.shape}."
        )
    return array


def _require_signature(
    actual: ExponentialFamilySignature,
    expected: ExponentialFamilySignature,
) -> None:
    if actual.key != expected.key:
        raise ValueError(
            "Exponential-family coordinate signature does not match the family."
        )


class NaturalCoordinates(StrictModule):
    """Finite-dimensional natural coordinates tagged by family identity."""

    values: Array
    signature: ExponentialFamilySignature = eqx.field(static=True)

    def __init__(self, values: ArrayLike, signature: ExponentialFamilySignature):
        if not isinstance(signature, ExponentialFamilySignature):
            raise TypeError("signature must be an ExponentialFamilySignature.")
        self.values = _coordinate_array(values, signature)
        self.signature = signature

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape[:-1])


class MeanCoordinates(StrictModule):
    """Expected sufficient-statistic coordinates tagged by family identity."""

    values: Array
    signature: ExponentialFamilySignature = eqx.field(static=True)

    def __init__(self, values: ArrayLike, signature: ExponentialFamilySignature):
        if not isinstance(signature, ExponentialFamilySignature):
            raise TypeError("signature must be an ExponentialFamilySignature.")
        self.values = _coordinate_array(values, signature)
        self.signature = signature

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape[:-1])


class StatisticBatch(StrictModule):
    """Sufficient statistics and support validity for one observation batch."""

    values: Array
    valid: Array
    signature: ExponentialFamilySignature = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        valid: ArrayLike,
        signature: ExponentialFamilySignature,
    ):
        if not isinstance(signature, ExponentialFamilySignature):
            raise TypeError("signature must be an ExponentialFamilySignature.")
        statistics = _coordinate_array(values, signature)
        validity = jnp.broadcast_to(jnp.asarray(valid, dtype=bool), statistics.shape[:-1])
        self.values = statistics
        self.valid = validity
        self.signature = signature


class ExponentialFamilyDomainResult(StrictModule):
    """Interior, boundary, and validity classification of family coordinates."""

    interior: Array
    boundary: Array
    valid: Array
    status: Array
    signature: ExponentialFamilySignature = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        interior: ArrayLike,
        boundary: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        signature: ExponentialFamilySignature,
        domain_id: str,
    ):
        interior_array = jnp.asarray(interior, dtype=bool)
        shape = interior_array.shape
        if not domain_id:
            raise ValueError("domain_id must be non-empty.")
        self.interior = interior_array
        self.boundary = jnp.broadcast_to(jnp.asarray(boundary, dtype=bool), shape)
        self.valid = jnp.broadcast_to(jnp.asarray(valid, dtype=bool), shape)
        self.status = jnp.broadcast_to(jnp.asarray(status, dtype=jnp.int32), shape)
        self.signature = signature
        self.domain_id = str(domain_id)


class ExponentialFamilyConversionResult(StrictModule):
    """Audited conversion from mean coordinates to natural coordinates."""

    mean: MeanCoordinates
    natural: NaturalCoordinates
    valid: Array
    status: Array
    residual: Array
    iterations: Array
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        mean: MeanCoordinates,
        natural: NaturalCoordinates,
        valid: ArrayLike,
        status: ArrayLike,
        residual: ArrayLike,
        iterations: ArrayLike,
        method_id: str,
    ):
        _require_signature(natural.signature, mean.signature)
        if natural.batch_shape != mean.batch_shape:
            raise ValueError("Mean and natural coordinate batch shapes must match.")
        if not method_id:
            raise ValueError("method_id must be non-empty.")
        shape = mean.batch_shape
        self.mean = mean
        self.natural = natural
        self.valid = jnp.broadcast_to(jnp.asarray(valid, dtype=bool), shape)
        self.status = jnp.broadcast_to(jnp.asarray(status, dtype=jnp.int32), shape)
        self.residual = jnp.broadcast_to(jnp.asarray(residual), shape)
        self.iterations = jnp.broadcast_to(
            jnp.asarray(iterations, dtype=jnp.int32), shape
        )
        self.method_id = str(method_id)


def _natural_domain_result(
    signature: ExponentialFamilySignature,
    values: Array,
    /,
    *,
    interior: ArrayLike,
    boundary: ArrayLike,
) -> ExponentialFamilyDomainResult:
    finite = jnp.all(jnp.isfinite(values), axis=-1)
    interior_array = finite & jnp.asarray(interior, dtype=bool)
    boundary_array = finite & jnp.asarray(boundary, dtype=bool)
    status = jnp.where(
        ~finite,
        EXPONENTIAL_FAMILY_NONFINITE,
        jnp.where(
            interior_array,
            EXPONENTIAL_FAMILY_SUCCESS,
            EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN,
        ),
    )
    return ExponentialFamilyDomainResult(
        interior=interior_array,
        boundary=boundary_array,
        valid=interior_array,
        status=status,
        signature=signature,
        domain_id="natural",
    )


def _mean_domain_result(
    signature: ExponentialFamilySignature,
    values: Array,
    /,
    *,
    interior: ArrayLike,
    boundary: ArrayLike,
) -> ExponentialFamilyDomainResult:
    finite = jnp.all(jnp.isfinite(values), axis=-1)
    interior_array = finite & jnp.asarray(interior, dtype=bool)
    boundary_array = finite & jnp.asarray(boundary, dtype=bool)
    status = jnp.where(
        ~finite,
        EXPONENTIAL_FAMILY_NONFINITE,
        jnp.where(
            interior_array,
            EXPONENTIAL_FAMILY_SUCCESS,
            jnp.where(
                boundary_array,
                EXPONENTIAL_FAMILY_MEAN_BOUNDARY,
                EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN,
            ),
        ),
    )
    return ExponentialFamilyDomainResult(
        interior=interior_array,
        boundary=boundary_array,
        valid=interior_array,
        status=status,
        signature=signature,
        domain_id="mean",
    )


class AbstractExponentialFamily(StrictModule):
    """Regular exponential family in one explicit intrinsic coordinate chart."""

    @property
    @abstractmethod
    def signature(self) -> ExponentialFamilySignature:
        raise NotImplementedError

    @abstractmethod
    def _natural_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        raise NotImplementedError

    @abstractmethod
    def _mean_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        raise NotImplementedError

    @abstractmethod
    def _sufficient_statistics(self, value: ArrayLike, /) -> StatisticBatch:
        raise NotImplementedError

    @abstractmethod
    def _log_base_density(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _log_normalizer(self, natural_values: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _mean_values(self, natural_values: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _natural_from_mean_values(self, mean_values: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _sample(
        self,
        key,
        natural_values: Array,
        sample_shape: tuple[int, ...],
        /,
    ) -> Array:
        raise NotImplementedError

    def natural(self, values: ArrayLike, /) -> NaturalCoordinates:
        return NaturalCoordinates(values, self.signature)

    def mean(self, values: ArrayLike, /) -> MeanCoordinates:
        return MeanCoordinates(values, self.signature)

    def natural_domain(
        self, natural: NaturalCoordinates, /
    ) -> ExponentialFamilyDomainResult:
        _require_signature(natural.signature, self.signature)
        return self._natural_domain(natural.values)

    def mean_domain(self, mean: MeanCoordinates, /) -> ExponentialFamilyDomainResult:
        _require_signature(mean.signature, self.signature)
        return self._mean_domain(mean.values)

    def sufficient_statistics(self, value: ArrayLike, /) -> StatisticBatch:
        statistics = self._sufficient_statistics(value)
        _require_signature(statistics.signature, self.signature)
        return statistics

    def log_base_density(self, value: ArrayLike, /) -> Array:
        return self._log_base_density(value)

    def log_normalizer(self, natural: NaturalCoordinates, /) -> Array:
        _require_signature(natural.signature, self.signature)
        return self._log_normalizer(natural.values)

    def mean_from_natural(self, natural: NaturalCoordinates, /) -> MeanCoordinates:
        _require_signature(natural.signature, self.signature)
        return MeanCoordinates(self._mean_values(natural.values), self.signature)

    @abstractmethod
    def _natural_from_mean_result(
        self,
        mean: MeanCoordinates,
        domain: ExponentialFamilyDomainResult,
        /,
    ) -> ExponentialFamilyConversionResult:
        candidate_values = self._natural_from_mean_values(mean.values)
        candidate_values = jnp.where(
            domain.interior[..., None], candidate_values, jnp.nan
        )
        natural = NaturalCoordinates(candidate_values, self.signature)
        reconstructed = self._mean_values(candidate_values)
        residual = jnp.linalg.norm(reconstructed - mean.values, axis=-1)
        candidate_finite = jnp.all(jnp.isfinite(candidate_values), axis=-1)
        residual_finite = jnp.isfinite(residual)
        valid = domain.valid & candidate_finite & residual_finite
        status = jnp.where(
            domain.valid & ~valid,
            EXPONENTIAL_FAMILY_NONFINITE,
            domain.status,
        )
        return ExponentialFamilyConversionResult(
            mean=mean,
            natural=natural,
            valid=valid,
            status=status,
            residual=jnp.where(valid, residual, jnp.inf),
            iterations=jnp.zeros(mean.batch_shape, dtype=jnp.int32),
            method_id=f"{self.signature.family_id}-analytic",
        )

    def natural_from_mean(
        self, mean: MeanCoordinates, /
    ) -> ExponentialFamilyConversionResult:
        _require_signature(mean.signature, self.signature)
        domain = self.mean_domain(mean)
        return self._natural_from_mean_result(mean, domain)

    def log_prob(
        self,
        natural: NaturalCoordinates,
        value: ArrayLike,
        /,
    ) -> Array:
        domain = self.natural_domain(natural)
        statistics = self.sufficient_statistics(value)
        pairing = jnp.sum(natural.values * statistics.values, axis=-1)
        result = pairing - self.log_normalizer(natural) + self.log_base_density(value)
        return jnp.where(domain.valid & statistics.valid, result, -jnp.inf)

    def canonical_loss(
        self,
        natural: NaturalCoordinates,
        value: ArrayLike,
        /,
    ) -> Array:
        domain = self.natural_domain(natural)
        statistics = self.sufficient_statistics(value)
        pairing = jnp.sum(natural.values * statistics.values, axis=-1)
        result = self.log_normalizer(natural) - pairing
        return jnp.where(domain.valid & statistics.valid, result, jnp.inf)

    def canonical_score(
        self,
        natural: NaturalCoordinates,
        value: ArrayLike,
        /,
    ) -> Array:
        statistics = self.sufficient_statistics(value)
        mean = self.mean_from_natural(natural)
        score = mean.values - statistics.values
        return jnp.where(statistics.valid[..., None], score, jnp.nan)

    def fisher_action(
        self,
        natural: NaturalCoordinates,
        direction: ArrayLike,
        /,
    ) -> Array:
        _require_signature(natural.signature, self.signature)
        vector = jnp.asarray(direction)
        if vector.shape != natural.values.shape:
            raise ValueError("Fisher direction must match natural-coordinate shape.")
        _, action = jax.jvp(
            self._mean_values,
            (natural.values,),
            (vector,),
        )
        return action

    def kl_divergence(
        self,
        left: NaturalCoordinates,
        right: NaturalCoordinates,
        /,
    ) -> Array:
        _require_signature(left.signature, self.signature)
        _require_signature(right.signature, self.signature)
        mean = self._mean_values(left.values)
        return (
            self._log_normalizer(right.values)
            - self._log_normalizer(left.values)
            - jnp.sum((right.values - left.values) * mean, axis=-1)
        )

    def law(self, natural: NaturalCoordinates | ArrayLike, /) -> "ExponentialFamilyLaw":
        coordinates = (
            natural
            if isinstance(natural, NaturalCoordinates)
            else NaturalCoordinates(natural, self.signature)
        )
        return ExponentialFamilyLaw(self, coordinates)

    def sample(
        self,
        key,
        natural: NaturalCoordinates,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        domain = self.natural_domain(natural)
        checked = eqx.error_if(
            natural.values,
            jnp.any(~domain.valid),
            "Cannot sample an invalid exponential-family law.",
        )
        return self._sample(key, checked, tuple(sample_shape))


class _AbstractAnalyticExponentialFamily(AbstractExponentialFamily):
    def _natural_from_mean_result(
        self,
        mean: MeanCoordinates,
        domain: ExponentialFamilyDomainResult,
        /,
    ) -> ExponentialFamilyConversionResult:
        return super()._natural_from_mean_result(mean, domain)


class ExponentialFamilyLaw(AbstractProbabilityLaw):
    """Normalized exponential-family law with audited natural coordinates."""

    family: AbstractExponentialFamily
    natural: NaturalCoordinates

    def __init__(
        self,
        family: AbstractExponentialFamily,
        natural: NaturalCoordinates,
    ):
        if not isinstance(family, AbstractExponentialFamily):
            raise TypeError("family must implement AbstractExponentialFamily.")
        _require_signature(natural.signature, family.signature)
        self.family = family
        self.natural = natural

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.family.signature.event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.natural.batch_shape

    @property
    def density_measure_kind(self) -> MeasureKind:
        return self.family.signature.density_measure_kind

    @property
    def valid(self) -> Array:
        return self.family.natural_domain(self.natural).valid

    @property
    def status(self) -> Array:
        return self.family.natural_domain(self.natural).status

    @property
    def mean_coordinates(self) -> MeanCoordinates:
        return self.family.mean_from_natural(self.natural)

    def contains(self, value: ArrayLike, /) -> Array:
        return self.family.sufficient_statistics(value).valid

    def log_prob(self, value: ArrayLike, /) -> Array:
        return self.family.log_prob(self.natural, value)

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.family.sample(key, self.natural, tuple(sample_shape))

    def kl_divergence(self, other: "ExponentialFamilyLaw", /) -> Array:
        if not isinstance(other, ExponentialFamilyLaw):
            raise TypeError("other must be an ExponentialFamilyLaw.")
        _require_signature(other.family.signature, self.family.signature)
        return self.family.kl_divergence(self.natural, other.natural)

    def fisher_action(self, direction: ArrayLike, /) -> Array:
        return self.family.fisher_action(self.natural, direction)


__all__ = [
    "AbstractExponentialFamily",
    "EXPONENTIAL_FAMILY_INSUFFICIENT_WEIGHT",
    "EXPONENTIAL_FAMILY_INVALID_EVENT",
    "EXPONENTIAL_FAMILY_MEAN_BOUNDARY",
    "EXPONENTIAL_FAMILY_NONFINITE",
    "EXPONENTIAL_FAMILY_NONCONVERGED",
    "EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN",
    "EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN",
    "EXPONENTIAL_FAMILY_SUCCESS",
    "ExponentialFamilyConversionResult",
    "ExponentialFamilyDomainResult",
    "ExponentialFamilyLaw",
    "ExponentialFamilySignature",
    "ExponentialFamilyStatus",
    "MeanCoordinates",
    "NaturalCoordinates",
    "StatisticBatch",
    "exponential_family_status_name",
]
