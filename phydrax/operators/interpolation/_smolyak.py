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

from phydrax.domain import (
    AbstractScalarDomain,
    DomainFunction,
    PointBatch,
    ProbabilityDomain,
    SampleLayout,
)

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ..._interpolation import barycentric_basis
from ..._numerics import (
    axis_level,
    smolyak_axis_data,
    smolyak_terms,
    smolyak_terms_for_index_set,
    SmolyakAxisRule,
    SmolyakFrontier,
    SmolyakIndexSet,
    SmolyakRefinementEpoch,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._plans import (
    AdaptiveSmolyakInterpolationPlan,
    SmolyakInterpolationPlan,
    SmolyakInterpolationRule,
)


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
    return factor


def _resolve_axis_rules(
    factors: tuple[AbstractScalarDomain, ...],
    requested: tuple[SmolyakInterpolationRule, ...],
    /,
) -> tuple[SmolyakAxisRule, ...]:
    resolved: list[SmolyakAxisRule] = []
    for axis, (factor, rule) in enumerate(zip(factors, requested, strict=True)):
        if rule == "auto":
            if isinstance(factor, ProbabilityDomain):
                transport = factor.reference_transport
                rule_ = (
                    "gauss-hermite"
                    if transport.reference_measure == "standard-normal"
                    else "leja"
                )
            else:
                rule_ = "leja"
        else:
            rule_ = rule
        if isinstance(factor, ProbabilityDomain):
            transport = factor.reference_transport
            expected = "standard-normal" if rule_ == "gauss-hermite" else "uniform"
            if transport.reference_measure != expected:
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


def _from_reference(factor: AbstractScalarDomain, rule: SmolyakAxisRule, value: Any, /):
    reference = jnp.asarray(value, dtype=float)
    if isinstance(factor, ProbabilityDomain):
        return factor.reference_transport.from_reference(reference)
    if rule == "gauss-hermite":
        raise TypeError("Gauss--Hermite interpolation requires a probability factor.")
    lower = factor.fixed("start")
    upper = factor.fixed("end")
    return 0.5 * (upper - lower) * reference + 0.5 * (upper + lower)


def _to_reference(factor: AbstractScalarDomain, rule: SmolyakAxisRule, value: Any, /):
    physical = jnp.asarray(value, dtype=float)
    if isinstance(factor, ProbabilityDomain):
        return factor.reference_transport.to_reference(physical)
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
    *,
    index_set: SmolyakIndexSet | None = None,
) -> tuple[np.ndarray, tuple[_TermTopology, ...]]:
    point_indices: dict[tuple[tuple[str, int, int], ...], int] = {}
    points: list[tuple[float, ...]] = []
    topologies: list[_TermTopology] = []
    terms = (
        smolyak_terms(dimension, level, anisotropy)
        if index_set is None
        else smolyak_terms_for_index_set(index_set)
    )
    for term in terms:
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
    factors: tuple[AbstractScalarDomain, ...]
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
        factors: tuple[AbstractScalarDomain, ...],
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
    if any(not isinstance(factor, AbstractScalarDomain) for factor in factors):
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
    structure = SampleLayout((dependencies,)).canonicalize(dependency_domain.labels)
    sample_axis = structure.axis_for(dependencies[0])
    if sample_axis is None:
        raise RuntimeError("Smolyak interpolation structure has no sample axis.")
    points = PointBatch(
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
    metadata: Mapping[str, Any] = function.metadata
    return DomainFunction(
        domain=function.domain,
        deps=dependencies,
        func=interpolant,
        metadata=metadata,
    )


class AdaptiveSmolyakInterpolationDiagnostics(StrictModule, NonTrainableState):
    status: str = eqx.field(static=True)
    frontier_indicator: float = eqx.field(static=True)
    accepted_indices: int = eqx.field(static=True)
    num_unique_nodes: int = eqx.field(static=True)
    num_rounds: int = eqx.field(static=True)


class AdaptiveSmolyakInterpolationResult(StrictModule, NonTrainableState):
    function: DomainFunction
    epochs: tuple[SmolyakRefinementEpoch, ...]
    diagnostics: AdaptiveSmolyakInterpolationDiagnostics


def _interpolate_index_set(
    function: DomainFunction,
    plan: AdaptiveSmolyakInterpolationPlan,
    index_set: SmolyakIndexSet,
    /,
    *,
    key: Key[Array, ""],
) -> tuple[DomainFunction, np.ndarray]:
    dependencies = tuple(function.deps)
    raw_factors = tuple(function.domain.factor(label) for label in dependencies)
    factors = tuple(_unwrap(factor) for factor in raw_factors)
    if any(not isinstance(factor, AbstractScalarDomain) for factor in factors):
        raise TypeError("Adaptive Smolyak interpolation requires scalar factors.")
    scalar_factors = tuple(factors)
    rules = _resolve_axis_rules(scalar_factors, plan.axis_rules)
    canonical_points, topologies = _build_topology(
        plan.dimension,
        plan.initial_level,
        plan.anisotropy,
        rules,
        index_set=index_set,
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
    structure = SampleLayout((dependencies,)).canonicalize(dependency_domain.labels)
    sample_axis = structure.axis_for(dependencies[0])
    if sample_axis is None:
        raise RuntimeError("Adaptive Smolyak structure has no sample axis.")
    points = PointBatch(
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
    values = jnp.asarray(fitting_function(points, key=key).data)
    if values.ndim < 1 or int(values.shape[0]) != int(canonical_points.shape[0]):
        raise ValueError(
            "Adaptive Smolyak source evaluation must preserve the node axis."
        )
    if bool(jnp.any(~jnp.isfinite(values))):
        raise ValueError("Adaptive Smolyak source evaluation produced non-finite values.")
    blocks = _build_blocks(topologies, values)
    interpolant = SmolyakInterpolant(
        blocks=blocks,
        factors=scalar_factors,
        axis_labels=dependencies,
        axis_rules=rules,
        anisotropy=plan.anisotropy,
        level=max(sum(index) for index in index_set.indices) + 1,
        output_shape=tuple(int(size) for size in values.shape[1:]),
        num_terms=len(topologies),
        num_evaluations=int(canonical_points.shape[0]),
        maximum_active_dimension=max(len(topology.axes) for topology in topologies),
    )
    return (
        DomainFunction(
            domain=function.domain,
            deps=dependencies,
            func=interpolant,
            metadata=function.metadata,
        ),
        canonical_points,
    )


def _adaptive_axis_rules(
    function: DomainFunction,
    plan: AdaptiveSmolyakInterpolationPlan,
    /,
) -> tuple[SmolyakAxisRule, ...]:
    raw_factors = tuple(function.domain.factor(label) for label in function.deps)
    factors = tuple(_unwrap(factor) for factor in raw_factors)
    if any(not isinstance(factor, AbstractScalarDomain) for factor in factors):
        raise TypeError("Adaptive Smolyak interpolation requires scalar factors.")
    return _resolve_axis_rules(tuple(factors), plan.axis_rules)


def _axis_node_count(rule: SmolyakAxisRule, level: int, /) -> int:
    if rule == "clenshaw-curtis":
        return 1 if level == 0 else 2**level + 1
    if rule in ("leja", "gauss-hermite"):
        return level + 1
    raise ValueError(f"Unsupported interpolation axis rule {rule!r}.")


def _index_set_node_count(
    index_set: SmolyakIndexSet,
    rules: tuple[SmolyakAxisRule, ...],
    /,
    *,
    limit: int,
) -> int:
    """Count unique canonical nodes without constructing an interpolant."""
    identifiers: set[tuple[tuple[str, int, int], ...]] = set()
    for term in smolyak_terms_for_index_set(index_set):
        tensor_nodes = 1
        for axis, rule in enumerate(rules):
            tensor_nodes *= _axis_node_count(rule, axis_level(term.index, axis))
            if tensor_nodes > limit:
                return limit + 1
        axis_data = tuple(
            smolyak_axis_data(rule, axis_level(term.index, axis))
            for axis, rule in enumerate(rules)
        )
        node_ranges = tuple(range(data.nodes.shape[0]) for data in axis_data)
        for position in itertools.product(*node_ranges):
            identifiers.add(
                tuple(
                    axis_data[axis].node_ids[point] for axis, point in enumerate(position)
                )
            )
            if len(identifiers) > limit:
                return len(identifiers)
    return len(identifiers)


def _adaptive_indicator(values: Array, norm: str, /) -> float:
    magnitude = np.abs(np.asarray(values))
    if norm == "max":
        return float(np.max(magnitude, initial=0.0))
    return float(np.sqrt(np.mean(magnitude**2)))


def interpolate_adaptive_smolyak(
    function: DomainFunction,
    plan: AdaptiveSmolyakInterpolationPlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> AdaptiveSmolyakInterpolationResult:
    """Prepare a dimension-adaptive immutable Smolyak interpolant."""
    if not isinstance(function, DomainFunction):
        raise TypeError("interpolate_adaptive_smolyak requires a DomainFunction.")
    if len(function.deps) != plan.dimension:
        raise ValueError(
            "AdaptiveSmolyakInterpolationPlan dimension must match dependencies."
        )
    index_set = SmolyakIndexSet.weighted_total_degree(
        plan.dimension,
        plan.initial_level,
        anisotropy=plan.anisotropy,
    )
    if len(index_set.indices) > plan.max_indices:
        raise ValueError("Initial adaptive interpolation index set exceeds max_indices.")
    rules = _adaptive_axis_rules(function, plan)
    initial_nodes = _index_set_node_count(index_set, rules, limit=plan.max_nodes)
    if initial_nodes > plan.max_nodes:
        raise ValueError("Initial adaptive interpolation grid exceeds max_nodes.")
    current, current_points = _interpolate_index_set(function, plan, index_set, key=key)
    epochs: list[SmolyakRefinementEpoch] = []
    frontier_indicator = float("inf")
    status = "maximum-rounds"
    for _ in range(plan.max_rounds):
        frontier = index_set.frontier(plan.anisotropy)
        if not frontier.candidates:
            status = "stagnated"
            epochs.append(
                SmolyakRefinementEpoch(
                    index_set=index_set,
                    frontier=frontier,
                    selected=None,
                    indicators=(),
                    new_work=(),
                    status=status,
                )
            )
            break
        if len(index_set.indices) >= plan.max_indices:
            status = "maximum-indices"
            epochs.append(
                SmolyakRefinementEpoch(
                    index_set=index_set,
                    frontier=frontier,
                    selected=None,
                    indicators=(),
                    new_work=(),
                    status=status,
                )
            )
            break
        candidates = []
        indicators = []
        work_values = []
        eligible_candidates: list[tuple[int, ...]] = []
        eligible_costs: list[float] = []
        node_limited = False
        for candidate, cost in zip(
            frontier.candidates,
            frontier.anisotropic_costs,
            strict=True,
        ):
            proposed = index_set.add(candidate)
            proposed_nodes = _index_set_node_count(
                proposed,
                rules,
                limit=plan.max_nodes,
            )
            work = max(
                1,
                proposed_nodes - int(current_points.shape[0]),
            )
            if proposed_nodes > plan.max_nodes:
                node_limited = True
                continue
            eligible_candidates.append(candidate)
            eligible_costs.append(cost)
            proposed_function, proposed_points = _interpolate_index_set(
                function, plan, proposed, key=key
            )
            factors = proposed_function.func.factors
            proposed_rules = proposed_function.func.axis_rules
            physical = tuple(
                _from_reference(
                    factor,
                    rule,
                    jnp.asarray(proposed_points[:, axis], dtype=float),
                )
                for axis, (factor, rule) in enumerate(
                    zip(factors, proposed_rules, strict=True)
                )
            )
            differences = jnp.stack(
                tuple(
                    proposed_function.func(*(column[row] for column in physical))
                    - current.func(*(column[row] for column in physical))
                    for row in range(int(proposed_points.shape[0]))
                )
            )
            indicator = _adaptive_indicator(differences, plan.indicator_norm)
            indicators.append(indicator)
            work_values.append(work)
            candidates.append(
                (
                    indicator / work,
                    candidate,
                    proposed,
                    proposed_function,
                    proposed_points,
                )
            )
        eligible_frontier = SmolyakFrontier(
            candidates=tuple(eligible_candidates),
            anisotropic_costs=tuple(eligible_costs),
        )
        frontier_indicator = float("inf") if node_limited else float(sum(indicators))
        if not candidates:
            status = "maximum-nodes"
            epochs.append(
                SmolyakRefinementEpoch(
                    index_set=index_set,
                    frontier=frontier,
                    selected=None,
                    indicators=(),
                    new_work=(),
                    status=status,
                )
            )
            break
        magnitude = _adaptive_indicator(
            jnp.stack(
                tuple(
                    current.func(
                        *(
                            _from_reference(
                                factor,
                                rule,
                                jnp.asarray(0.0),
                            )
                            for factor, rule in zip(
                                current.func.factors,
                                current.func.axis_rules,
                                strict=True,
                            )
                        )
                    )
                    for _ in range(1)
                )
            ),
            plan.indicator_norm,
        )
        if (
            frontier_indicator
            <= plan.absolute_tolerance + plan.relative_tolerance * magnitude
        ):
            status = "converged"
            epochs.append(
                SmolyakRefinementEpoch(
                    index_set=index_set,
                    frontier=eligible_frontier,
                    selected=None,
                    indicators=tuple(indicators),
                    new_work=tuple(work_values),
                    status=status,
                )
            )
            break
        candidates.sort(key=lambda value: (-value[0], value[1]))
        _, selected, proposed, proposed_function, proposed_points = candidates[0]
        epochs.append(
            SmolyakRefinementEpoch(
                index_set=index_set,
                frontier=eligible_frontier,
                selected=selected,
                indicators=tuple(indicators),
                new_work=tuple(work_values),
                status="accepted",
            )
        )
        index_set = proposed
        current = proposed_function
        current_points = proposed_points
    diagnostics = AdaptiveSmolyakInterpolationDiagnostics(
        status=status,
        frontier_indicator=frontier_indicator,
        accepted_indices=len(index_set.indices),
        num_unique_nodes=int(current_points.shape[0]),
        num_rounds=len(epochs),
    )
    return AdaptiveSmolyakInterpolationResult(
        function=current,
        epochs=tuple(epochs),
        diagnostics=diagnostics,
    )


__all__ = [
    "AdaptiveSmolyakInterpolationDiagnostics",
    "AdaptiveSmolyakInterpolationResult",
    "SmolyakInterpolant",
    "interpolate_adaptive_smolyak",
    "interpolate_smolyak",
]
