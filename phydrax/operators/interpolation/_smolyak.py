#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, NamedTuple

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ..._numerics import (
    axis_level,
    barycentric_basis,
    smolyak_axis_data,
    smolyak_terms,
    SmolyakAxisRule,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...domain._domain import RelabeledDomain
from ...domain._function import _drop_derivative_hook_metadata, DomainFunction
from ...domain._probability import ProbabilityDomain
from ...domain._scalar import _AbstractScalarDomain
from ...domain._structure import PointsBatch, ProductStructure
from ._plans import SmolyakInterpolationPlan, SmolyakInterpolationRule


class _TermTopology(NamedTuple):
    coefficient: int
    axes: tuple[int, ...]
    signature: tuple[int, ...]
    nodes: tuple[np.ndarray, ...]
    barycentric_weights: tuple[np.ndarray, ...]
    gather_indices: np.ndarray


class SmolyakInterpolationBlock(StrictModule, NonTrainableState):
    """Shape-homogeneous tensor terms evaluated in one vectorized block."""

    axes: Array
    nodes: tuple[Array, ...]
    barycentric_weights: tuple[Array, ...]
    values: Array
    coefficients: Array
    signature: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        axes: Array,
        nodes: tuple[Array, ...],
        barycentric_weights: tuple[Array, ...],
        values: Array,
        coefficients: Array,
        signature: tuple[int, ...],
    ):
        self.axes = jnp.asarray(axes, dtype=jnp.int32)
        self.nodes = tuple(jnp.asarray(value, dtype=float) for value in nodes)
        self.barycentric_weights = tuple(
            jnp.asarray(value, dtype=float) for value in barycentric_weights
        )
        self.values = jnp.asarray(values)
        self.coefficients = jnp.asarray(coefficients)
        self.signature = tuple(int(value) for value in signature)

    def evaluate(self, reference: Array, /) -> Array:
        term_values = self.values
        for position, (nodes, weights) in enumerate(
            zip(self.nodes, self.barycentric_weights, strict=True)
        ):
            query = reference[self.axes[:, position]]
            basis = jax.vmap(barycentric_basis)(query, nodes, weights)
            term_values = jax.vmap(
                lambda basis_row, values: jnp.tensordot(
                    basis_row,
                    values,
                    axes=((0,), (0,)),
                )
            )(basis, term_values)
        coefficient_shape = (int(self.coefficients.shape[0]),) + (1,) * (
            term_values.ndim - 1
        )
        weighted = term_values * jnp.reshape(self.coefficients, coefficient_shape)
        return jnp.sum(weighted, axis=0)


def _unwrap(factor: Any, /) -> Any:
    return factor.base if isinstance(factor, RelabeledDomain) else factor


def _resolve_axis_rules(
    factors: tuple[_AbstractScalarDomain, ...],
    requested: tuple[SmolyakInterpolationRule, ...],
    /,
) -> tuple[SmolyakAxisRule, ...]:
    resolved: list[SmolyakAxisRule] = []
    for axis, (factor, rule) in enumerate(zip(factors, requested, strict=True)):
        if rule == "auto":
            if isinstance(factor, ProbabilityDomain):
                if not factor.supports_reference_transform:
                    raise ValueError(
                        f"Probability interpolation axis {axis} has no canonical "
                        "reference transform."
                    )
                rule_ = (
                    "gauss-hermite"
                    if factor.reference_measure == "standard-normal"
                    else "leja"
                )
            else:
                rule_ = "leja"
        else:
            rule_ = rule
        if isinstance(factor, ProbabilityDomain):
            if not factor.supports_reference_transform:
                raise ValueError(
                    f"Probability interpolation axis {axis} has no canonical "
                    "reference transform."
                )
            expected = "standard-normal" if rule_ == "gauss-hermite" else "uniform"
            if factor.reference_measure != expected:
                raise ValueError(
                    f"Interpolation rule {rule_!r} on probability axis {axis} "
                    f"requires reference measure {expected!r}."
                )
        elif rule_ == "gauss-hermite":
            raise TypeError(
                f"Gauss--Hermite interpolation axis {axis} requires a probability factor."
            )
        resolved.append(rule_)
    return tuple(resolved)


def _from_reference(factor: _AbstractScalarDomain, rule: SmolyakAxisRule, value: Any, /):
    reference = jnp.asarray(value, dtype=float)
    if isinstance(factor, ProbabilityDomain):
        return factor.from_reference(reference)
    if rule == "gauss-hermite":
        raise TypeError("Gauss--Hermite interpolation requires a probability factor.")
    lower = factor.fixed("start")
    upper = factor.fixed("end")
    return 0.5 * (upper - lower) * reference + 0.5 * (upper + lower)


def _to_reference(factor: _AbstractScalarDomain, rule: SmolyakAxisRule, value: Any, /):
    physical = jnp.asarray(value, dtype=float)
    if isinstance(factor, ProbabilityDomain):
        return factor.to_reference(physical)
    if rule == "gauss-hermite":
        raise TypeError("Gauss--Hermite interpolation requires a probability factor.")
    lower = factor.fixed("start")
    upper = factor.fixed("end")
    return (2.0 * physical - lower - upper) / (upper - lower)


def _build_topology(
    dimension: int,
    level: int,
    anisotropy: tuple[float, ...],
    rules: tuple[SmolyakAxisRule, ...],
    /,
) -> tuple[np.ndarray, tuple[_TermTopology, ...]]:
    point_indices: dict[tuple[tuple[str, int, int], ...], int] = {}
    points: list[tuple[float, ...]] = []
    topologies: list[_TermTopology] = []
    for term in smolyak_terms(dimension, level, anisotropy):
        axis_data = tuple(
            smolyak_axis_data(rule, axis_level(term.index, axis))
            for axis, rule in enumerate(rules)
        )
        shape = tuple(int(data.nodes.shape[0]) for data in axis_data)
        local_indices: list[int] = []
        ranges = tuple(range(count) for count in shape)
        for position in itertools.product(*ranges):
            identifier = tuple(
                axis_data[axis].node_ids[node] for axis, node in enumerate(position)
            )
            if identifier not in point_indices:
                point_indices[identifier] = len(points)
                points.append(
                    tuple(
                        float(axis_data[axis].nodes[node])
                        for axis, node in enumerate(position)
                    )
                )
            local_indices.append(point_indices[identifier])
        active_axes_original = tuple(axis for axis in range(dimension) if shape[axis] > 1)
        active_axes = tuple(
            sorted(active_axes_original, key=lambda axis: (-shape[axis], axis))
        )
        signature = tuple(shape[axis] for axis in active_axes)
        gather = np.asarray(local_indices, dtype=np.int32).reshape(shape)
        inactive_axes = tuple(axis for axis in range(dimension) if shape[axis] == 1)
        if inactive_axes:
            gather = np.squeeze(gather, axis=inactive_axes)
        if len(active_axes) > 1 and active_axes != active_axes_original:
            permutation = tuple(active_axes_original.index(axis) for axis in active_axes)
            gather = np.transpose(gather, axes=permutation)
        topologies.append(
            _TermTopology(
                term.coefficient,
                active_axes,
                signature,
                tuple(axis_data[axis].nodes for axis in active_axes),
                tuple(axis_data[axis].barycentric_weights for axis in active_axes),
                np.asarray(gather, dtype=np.int32),
            )
        )
    return np.asarray(points, dtype=float).reshape((-1, dimension)), tuple(topologies)


def _build_blocks(
    topologies: tuple[_TermTopology, ...],
    values: Array,
    /,
) -> tuple[SmolyakInterpolationBlock, ...]:
    groups: dict[tuple[int, ...], list[_TermTopology]] = defaultdict(list)
    for topology in topologies:
        groups[topology.signature].append(topology)
    blocks: list[SmolyakInterpolationBlock] = []
    for signature in sorted(groups, key=lambda value: (len(value), value)):
        terms = groups[signature]
        rank = len(signature)
        axes = np.asarray(tuple(term.axes for term in terms), dtype=np.int32).reshape(
            (len(terms), rank)
        )
        nodes = tuple(
            jnp.stack(
                tuple(jnp.asarray(term.nodes[position]) for term in terms),
                axis=0,
            )
            for position in range(rank)
        )
        barycentric_weights = tuple(
            jnp.stack(
                tuple(jnp.asarray(term.barycentric_weights[position]) for term in terms),
                axis=0,
            )
            for position in range(rank)
        )
        term_values = jnp.stack(
            tuple(values[jnp.asarray(term.gather_indices)] for term in terms),
            axis=0,
        )
        coefficients = jnp.asarray(
            tuple(term.coefficient for term in terms),
            dtype=values.dtype,
        )
        blocks.append(
            SmolyakInterpolationBlock(
                axes=axes,
                nodes=nodes,
                barycentric_weights=barycentric_weights,
                values=term_values,
                coefficients=coefficients,
                signature=signature,
            )
        )
    return tuple(blocks)


class SmolyakInterpolant(StrictModule, NonTrainableState):
    """Immutable, differentiable Smolyak interpolant in physical coordinates."""

    blocks: tuple[SmolyakInterpolationBlock, ...]
    factors: tuple[_AbstractScalarDomain, ...]
    axis_labels: tuple[str, ...] = eqx.field(static=True)
    axis_rules: tuple[SmolyakAxisRule, ...] = eqx.field(static=True)
    anisotropy: tuple[float, ...] = eqx.field(static=True)
    level: int = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    num_terms: int = eqx.field(static=True)
    num_evaluations: int = eqx.field(static=True)
    num_unique_nodes: int = eqx.field(static=True)
    maximum_active_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        blocks: tuple[SmolyakInterpolationBlock, ...],
        factors: tuple[_AbstractScalarDomain, ...],
        axis_labels: tuple[str, ...],
        axis_rules: tuple[SmolyakAxisRule, ...],
        anisotropy: tuple[float, ...],
        level: int,
        output_shape: tuple[int, ...],
        num_terms: int,
        num_evaluations: int,
        maximum_active_dimension: int,
    ):
        self.blocks = blocks
        self.factors = factors
        self.axis_labels = axis_labels
        self.axis_rules = axis_rules
        self.anisotropy = anisotropy
        self.level = int(level)
        self.output_shape = output_shape
        self.num_terms = int(num_terms)
        self.num_evaluations = int(num_evaluations)
        self.num_unique_nodes = int(num_evaluations)
        self.maximum_active_dimension = int(maximum_active_dimension)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def dtype(self):
        return self.blocks[0].values.dtype

    def __call__(
        self,
        *coordinates: Any,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del key, iter_
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"SmolyakInterpolant received unsupported keywords: {names}.")
        if len(coordinates) != len(self.factors):
            raise ValueError(
                f"Expected {len(self.factors)} interpolation coordinates, "
                f"got {len(coordinates)}."
            )
        reference = jnp.stack(
            tuple(
                jnp.asarray(_to_reference(factor, rule, coordinate)).reshape(())
                for factor, rule, coordinate in zip(
                    self.factors, self.axis_rules, coordinates, strict=True
                )
            )
        )
        result = self.blocks[0].evaluate(reference)
        for block in self.blocks[1:]:
            result = result + block.evaluate(reference)
        return result


def _dependency_domain(function: DomainFunction, /):
    factors = tuple(function.domain.factor(label) for label in function.deps)
    domain = factors[0]
    for factor in factors[1:]:
        domain = domain.join(factor)
    return domain


def interpolate_smolyak(
    function: DomainFunction,
    plan: SmolyakInterpolationPlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> DomainFunction:
    """Fit a reusable Smolyak interpolant and return it as a `DomainFunction`."""
    if not isinstance(function, DomainFunction):
        raise TypeError("interpolate_smolyak requires a DomainFunction.")
    dependencies = tuple(function.deps)
    if len(dependencies) != plan.dimension:
        raise ValueError(
            f"SmolyakInterpolationPlan dimension={plan.dimension} but function has "
            f"{len(dependencies)} dependencies."
        )
    raw_factors = tuple(function.domain.factor(label) for label in dependencies)
    factors = tuple(_unwrap(factor) for factor in raw_factors)
    if any(not isinstance(factor, _AbstractScalarDomain) for factor in factors):
        raise TypeError("Smolyak interpolation requires scalar dependency factors.")
    scalar_factors = tuple(factors)
    rules = _resolve_axis_rules(scalar_factors, plan.axis_rules)
    canonical_points, topologies = _build_topology(
        plan.dimension,
        plan.level,
        plan.anisotropy,
        rules,
    )
    physical_columns = tuple(
        _from_reference(
            factor,
            rule,
            jnp.asarray(canonical_points[:, axis], dtype=float),
        )
        for axis, (factor, rule) in enumerate(zip(scalar_factors, rules, strict=True))
    )
    dependency_domain = _dependency_domain(function)
    structure = ProductStructure((dependencies,)).canonicalize(dependency_domain.labels)
    sample_axis = structure.axis_for(dependencies[0])
    if sample_axis is None:
        raise RuntimeError("Smolyak interpolation structure has no sample axis.")
    points = PointsBatch(
        frozendict(
            {
                label: cx.Field(column, dims=(sample_axis,))
                for label, column in zip(dependencies, physical_columns, strict=True)
            }
        ),
        structure,
    )
    fitting_function = DomainFunction(
        domain=dependency_domain,
        deps=dependencies,
        func=function.func,
        metadata={},
    )
    evaluated = fitting_function(points, key=key)
    values = jnp.asarray(evaluated.data)
    num_evaluations = int(canonical_points.shape[0])
    if values.ndim < 1 or int(values.shape[0]) != num_evaluations:
        raise ValueError(
            "Smolyak source evaluation must return one leading value per node."
        )
    if bool(jnp.any(~jnp.isfinite(values))):
        raise ValueError("Smolyak source evaluation produced non-finite values.")
    blocks = _build_blocks(topologies, values)
    interpolant = SmolyakInterpolant(
        blocks=blocks,
        factors=scalar_factors,
        axis_labels=dependencies,
        axis_rules=rules,
        anisotropy=plan.anisotropy,
        level=plan.level,
        output_shape=tuple(int(value) for value in values.shape[1:]),
        num_terms=len(topologies),
        num_evaluations=num_evaluations,
        maximum_active_dimension=max(len(topology.axes) for topology in topologies),
    )
    metadata: Mapping[str, Any] = _drop_derivative_hook_metadata(function.metadata)
    return DomainFunction(
        domain=function.domain,
        deps=dependencies,
        func=interpolant,
        metadata=metadata,
    )


__all__ = [
    "SmolyakInterpolant",
    "interpolate_smolyak",
]
