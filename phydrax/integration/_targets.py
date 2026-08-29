#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast, TypeAlias

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.domain import ComponentSum, DomainComponent, ProbabilityDomain

from .._frozendict import frozendict
from .._strict import StrictModule
from ._rules import ReferenceRule


def _axes(
    value: int | str | tuple[int, ...] | tuple[str, ...],
    /,
) -> tuple[int, ...] | tuple[str, ...]:
    axes = value if isinstance(value, tuple) else (value,)
    if not axes:
        raise ValueError("sample_axes must contain at least one axis.")
    if all(isinstance(axis, int) for axis in axes):
        return tuple(int(axis) for axis in axes)
    if all(isinstance(axis, str) and axis for axis in axes):
        return tuple(str(axis) for axis in axes)
    raise TypeError("sample_axes must contain only integers or only non-empty names.")


def _target_mass(value: Array | None, /) -> Array | None:
    if value is None:
        return None
    mass = jnp.asarray(value, dtype=float)
    if bool(jnp.any(~jnp.isfinite(mass) | (mass <= 0.0))):
        raise ValueError("target_mass must be finite and strictly positive.")
    return mass


def _aligned_field(
    value: Array | cx.Field,
    reference: cx.Field,
    /,
    *,
    dtype: Any,
    name: str,
) -> cx.Field:
    if isinstance(value, cx.Field):
        if set(value.named_dims) - set(reference.named_dims):
            raise ValueError(f"{name} dimensions must be present in the weights.")
        field = value
    else:
        field = cx.Field(
            jnp.broadcast_to(jnp.asarray(value, dtype=dtype), reference.shape),
            dims=reference.dims,
        )
    return cx.Field(
        jnp.asarray(field.broadcast_like(reference).data, dtype=dtype),
        dims=reference.dims,
    )


def _sample_identifiers(
    value: Array | None,
    count: int,
    name: str,
    /,
) -> Array | None:
    if value is None:
        return None
    identifiers = jnp.asarray(value, dtype=jnp.int32).reshape((-1,))
    if identifiers.shape != (count,):
        raise ValueError(f"{name} must have one entry per sampled unit.")
    return identifiers


class ComponentTarget(StrictModule):
    """A physical/counting-measure domain component integration target."""

    component: DomainComponent | ComponentSum
    axes: str | tuple[str, ...] | None = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)

    def __init__(
        self,
        component: DomainComponent | ComponentSum,
        /,
        *,
        axes: str | tuple[str, ...] | None = None,
        normalized: bool = False,
    ):
        if not isinstance(component, (DomainComponent, ComponentSum)):
            raise TypeError("component must be a DomainComponent or ComponentSum.")
        self.component = component
        self.axes = axes
        self.normalized = bool(normalized)


class ProbabilityTarget(StrictModule):
    """Expectation under a normalized ``ProbabilityDomain``."""

    probability: ProbabilityDomain
    target_id: str = eqx.field(static=True)
    normalized: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        probability: ProbabilityDomain,
        /,
        *,
        target_id: str | None = None,
    ):
        if not isinstance(probability, ProbabilityDomain):
            raise TypeError("probability must be a ProbabilityDomain.")
        identifier = probability.label if target_id is None else str(target_id)
        if not identifier:
            raise ValueError("target_id must be non-empty.")
        self.probability = probability
        self.normalized = True
        self.target_id = identifier

class DensityTarget(StrictModule):
    """A density relative to another target's base measure."""

    base: Any
    log_density: Any
    normalized: bool = eqx.field(static=True)

    def __init__(
        self,
        base: Any,
        log_density: Callable | Array,
        /,
        *,
        normalized: bool,
    ):
        self.base = base
        self.log_density = log_density
        self.normalized = bool(normalized)


class DiscreteMeasureTarget(StrictModule):
    """An externally supplied deterministic discrete measure."""

    points: Any
    weights: cx.Field | frozendict[str, cx.Field]
    mask: cx.Field | None
    target_mass: Array | None
    axes: tuple[str, ...] = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        points: Any,
        weights: cx.Field | Mapping[str, cx.Field],
        /,
        *,
        axes: str | tuple[str, ...],
        mask: cx.Field | None = None,
        normalized: bool = False,
        target_mass: Array | None = None,
        provenance: str = "external-discrete",
    ):
        reduced_axes = (axes,) if isinstance(axes, str) else tuple(axes)
        if not reduced_axes or any(not axis for axis in reduced_axes):
            raise ValueError("axes must contain at least one non-empty name.")
        if len(set(reduced_axes)) != len(reduced_axes):
            raise ValueError("axes must be unique.")
        if isinstance(weights, cx.Field):
            missing = tuple(
                axis for axis in reduced_axes if axis not in weights.named_dims
            )
            if missing:
                raise ValueError(f"Discrete weights are missing axes {missing!r}.")
            if any(dim is None for dim in weights.dims):
                raise ValueError("Discrete weight fields must name every dimension.")
            resolved_weights: cx.Field | frozendict[str, cx.Field] = weights
        elif isinstance(weights, Mapping):
            resolved = frozendict(weights)
            if tuple(resolved) != reduced_axes:
                raise ValueError(
                    "Separable discrete weight keys must exactly match axes in order."
                )
            for axis, weight in resolved.items():
                if not isinstance(weight, cx.Field) or weight.dims != (axis,):
                    raise ValueError(
                        f"weights[{axis!r}] must be a one-dimensional field on that axis."
                    )
            resolved_weights = resolved
        else:
            raise TypeError("weights must be a coordax.Field or a mapping of fields.")
        if mask is not None and not isinstance(mask, cx.Field):
            raise TypeError("mask must be a coordax.Field or None.")
        if mask is not None:
            reference_dims = (
                set(resolved_weights.named_dims)
                if isinstance(resolved_weights, cx.Field)
                else set(reduced_axes)
            )
            if set(mask.named_dims) - reference_dims:
                raise ValueError("mask dimensions must be present in the weights.")
        provenance_ = str(provenance)
        if not provenance_:
            raise ValueError("provenance must be non-empty.")
        self.points = points
        self.weights = resolved_weights
        self.mask = mask
        self.target_mass = _target_mass(target_mass)
        self.axes = reduced_axes
        self.normalized = bool(normalized)
        self.provenance = provenance_


class WeightedSampleTarget(StrictModule):
    """An externally supplied masked log-weighted empirical measure."""

    samples: Any
    log_weights: Array | cx.Field
    mask: Array | cx.Field | None
    target_mass: Array | None
    ancestry: Array | cx.Field | None
    support_valid: Array | None
    stratum_ids: Array | None
    pair_ids: Array | None
    replicate_ids: Array | None
    normalized: bool = eqx.field(static=True)
    independent: bool = eqx.field(static=True)
    sample_axes: tuple[int, ...] | tuple[str, ...] = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        samples: Any,
        log_weights: Array | cx.Field,
        /,
        *,
        normalized: bool = True,
        target_mass: Array | None = None,
        independent: bool = False,
        ancestry: Array | cx.Field | None = None,
        support_valid: Array | None = None,
        stratum_ids: Array | None = None,
        pair_ids: Array | None = None,
        replicate_ids: Array | None = None,
        mask: Array | cx.Field | None = None,
        sample_axes: int | str | tuple[int, ...] | tuple[str, ...] = 0,
        provenance: str = "external-weighted-samples",
    ):
        axes = _axes(sample_axes)
        if isinstance(log_weights, cx.Field):
            if not all(isinstance(axis, str) for axis in axes):
                raise TypeError("Named log-weight fields require named sample_axes.")
            named_axes = cast(tuple[str, ...], axes)
            missing = tuple(
                axis for axis in named_axes if axis not in log_weights.named_dims
            )
            if missing:
                raise ValueError(f"log_weights is missing sample axes {missing!r}.")
            weights: Array | cx.Field = log_weights
            if any(dim is None for dim in log_weights.dims):
                raise ValueError("Named log-weight fields must name every dimension.")
            mask_ = (
                None
                if mask is None
                else _aligned_field(
                    mask,
                    log_weights,
                    dtype=bool,
                    name="mask",
                )
            )
            ancestry_ = (
                None
                if ancestry is None
                else _aligned_field(
                    ancestry,
                    log_weights,
                    dtype=jnp.int32,
                    name="ancestry",
                )
            )
            sample_count = 1
            for axis in named_axes:
                sample_count *= int(log_weights.named_shape[axis])
            resolved_axes: tuple[int, ...] | tuple[str, ...] = named_axes
        else:
            weights = jnp.asarray(log_weights, dtype=float)
            if weights.ndim < 1:
                raise ValueError("Weighted samples require at least one weight axis.")
            if not all(isinstance(axis, int) for axis in axes):
                raise TypeError("Raw log-weight arrays require integer sample_axes.")
            integer_axes = cast(tuple[int, ...], axes)
            resolved = tuple(
                axis + weights.ndim if axis < 0 else axis for axis in integer_axes
            )
            if any(axis < 0 or axis >= weights.ndim for axis in resolved):
                raise ValueError("sample_axes contains an out-of-range axis.")
            if len(set(resolved)) != len(resolved):
                raise ValueError("sample_axes must not contain duplicates.")
            resolved_axes = resolved
            mask_ = (
                None
                if mask is None
                else jnp.broadcast_to(jnp.asarray(mask, dtype=bool), weights.shape)
            )
            ancestry_ = (
                None
                if ancestry is None
                else jnp.broadcast_to(
                    jnp.asarray(ancestry, dtype=jnp.int32), weights.shape
                )
            )
            sample_count = 1
            for axis in resolved:
                sample_count *= int(weights.shape[axis])
        provenance_ = str(provenance)
        if not provenance_:
            raise ValueError("provenance must be non-empty.")
        self.samples = samples
        self.log_weights = weights
        self.mask = mask_
        self.target_mass = _target_mass(target_mass)
        self.ancestry = ancestry_
        self.support_valid = (
            None if support_valid is None else jnp.asarray(support_valid, dtype=bool)
        )
        self.stratum_ids = _sample_identifiers(stratum_ids, sample_count, "stratum_ids")
        self.pair_ids = _sample_identifiers(pair_ids, sample_count, "pair_ids")
        self.replicate_ids = _sample_identifiers(
            replicate_ids, sample_count, "replicate_ids"
        )
        self.normalized = bool(normalized)
        self.independent = bool(independent)
        self.sample_axes = resolved_axes
        self.provenance = provenance_


class MappedTarget(StrictModule):
    """Reference-cell integration through a supplied map and Jacobian."""

    reference_rule: ReferenceRule
    mapping: Any
    jacobian: Any
    mask: Any
    target_mass: Array | None

    def __init__(
        self,
        reference_rule: ReferenceRule,
        mapping: Callable,
        jacobian: Callable,
        /,
        *,
        mask: Callable | Array | None = None,
        target_mass: Array | None = None,
    ):
        self.reference_rule = reference_rule
        self.mapping = mapping
        self.jacobian = jacobian
        self.mask = mask
        self.target_mass = target_mass


MultilevelSampler: TypeAlias = Callable[[int, Array, Any], Any]


class MultilevelTarget(StrictModule):
    """A random-access coupled hierarchy with prefix-stable global sample indices.

    The sampler receives explicit global indices. Its returned values must depend on
    those indices, the hierarchy level, and the root key—not on request batching or
    call order. Multilevel execution validates the returned indices and fine/coarse
    pair identities before admitting every batch.
    """

    hierarchy: Any
    sampler: MultilevelSampler
    sampler_id: str = eqx.field(static=True)
    normalized: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        hierarchy: Any,
        sampler: MultilevelSampler,
        /,
        *,
        sampler_id: str,
    ):
        from ..stochastic._hierarchy import StochasticCouplingPlan

        if not isinstance(hierarchy, StochasticCouplingPlan):
            raise TypeError("hierarchy must be a StochasticCouplingPlan.")
        if not callable(sampler):
            raise TypeError("sampler must be callable.")
        identifier = str(sampler_id)
        if not identifier:
            raise ValueError("sampler_id must be non-empty.")
        self.hierarchy = hierarchy
        self.sampler = sampler
        self.sampler_id = identifier
        self.normalized = True


def multilevel(
    hierarchy: Any,
    sampler: MultilevelSampler,
    /,
    *,
    sampler_id: str,
) -> MultilevelTarget:
    """Construct a coupled multilevel expectation target."""
    return MultilevelTarget(hierarchy, sampler, sampler_id=sampler_id)


IntegrationTarget: TypeAlias = (
    ComponentTarget
    | ProbabilityTarget
    | DensityTarget
    | DiscreteMeasureTarget
    | MappedTarget
    | WeightedSampleTarget
    | MultilevelTarget
)


def over(
    component: DomainComponent | ComponentSum,
    /,
    *,
    axes: str | tuple[str, ...] | None = None,
) -> ComponentTarget:
    """Construct an unnormalized component target."""
    return ComponentTarget(component, axes=axes, normalized=False)


def mean_over(
    component: DomainComponent | ComponentSum,
    /,
    *,
    axes: str | tuple[str, ...] | None = None,
) -> ComponentTarget:
    """Construct a normalized physical/counting-measure target."""
    return ComponentTarget(component, axes=axes, normalized=True)


def expectation(
    probability: ProbabilityDomain,
    /,
    *,
    target_id: str | None = None,
) -> ProbabilityTarget:
    """Construct an expectation target with an optional stable identity."""
    return ProbabilityTarget(probability, target_id=target_id)


def density(
    base: IntegrationTarget,
    log_density: Callable | Array,
    /,
) -> DensityTarget:
    """Construct an unnormalized density integral target."""
    return DensityTarget(base, log_density, normalized=False)


def normalized_density(
    base: IntegrationTarget,
    log_density: Callable | Array,
    /,
) -> DensityTarget:
    """Construct a self-normalized density expectation target."""
    return DensityTarget(base, log_density, normalized=True)


def discrete(
    points: Any,
    weights: cx.Field | Mapping[str, cx.Field],
    /,
    *,
    axes: str | tuple[str, ...],
    mask: cx.Field | None = None,
    normalized: bool = False,
    target_mass: Array | None = None,
    provenance: str = "external-discrete",
) -> DiscreteMeasureTarget:
    """Construct an externally weighted deterministic discrete measure."""
    return DiscreteMeasureTarget(
        points,
        weights,
        axes=axes,
        mask=mask,
        normalized=normalized,
        target_mass=target_mass,
        provenance=provenance,
    )


def weighted(
    samples: Any,
    log_weights: Array | cx.Field,
    /,
    *,
    normalized: bool = True,
    target_mass: Array | None = None,
    independent: bool = False,
    ancestry: Array | cx.Field | None = None,
    support_valid: Array | None = None,
    stratum_ids: Array | None = None,
    pair_ids: Array | None = None,
    replicate_ids: Array | None = None,
    mask: Array | cx.Field | None = None,
    sample_axes: int | str | tuple[int, ...] | tuple[str, ...] = 0,
    provenance: str = "external-weighted-samples",
) -> WeightedSampleTarget:
    """Construct an explicit masked raw-log-weighted empirical measure."""
    return WeightedSampleTarget(
        samples,
        log_weights,
        normalized=normalized,
        target_mass=target_mass,
        independent=independent,
        ancestry=ancestry,
        support_valid=support_valid,
        stratum_ids=stratum_ids,
        pair_ids=pair_ids,
        replicate_ids=replicate_ids,
        mask=mask,
        sample_axes=sample_axes,
        provenance=provenance,
    )


def mapped(
    reference_rule: ReferenceRule,
    mapping: Callable,
    jacobian: Callable,
    /,
    *,
    mask: Callable | Array | None = None,
    target_mass: Array | None = None,
) -> MappedTarget:
    """Construct a supplied reference-to-physical mapping target."""
    return MappedTarget(
        reference_rule,
        mapping,
        jacobian,
        mask=mask,
        target_mass=target_mass,
    )


__all__ = [
    "ComponentTarget",
    "DensityTarget",
    "DiscreteMeasureTarget",
    "IntegrationTarget",
    "MappedTarget",
    "MultilevelSampler",
    "MultilevelTarget",
    "ProbabilityTarget",
    "WeightedSampleTarget",
    "density",
    "discrete",
    "expectation",
    "mapped",
    "multilevel",
    "mean_over",
    "normalized_density",
    "over",
    "weighted",
]
