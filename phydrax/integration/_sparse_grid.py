#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from typing import Any

import coordax as cx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key

from phydrax.domain import (
    AbstractGeometry,
    AbstractScalarDomain,
    ComponentSum,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
    PointBatch,
    ProbabilityDomain,
    SampleLayout,
)

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._numerics import (
    axis_level,
    normalize_axis_rules,
    smolyak_axis_data,
    smolyak_terms,
    smolyak_terms_for_index_set,
    SmolyakAxisRule,
    SmolyakFrontier,
    SmolyakIndexSet,
    SmolyakRefinementEpoch,
)
from .._strict import StrictModule
from ._batches import PointIntegrationBatch
from ._estimates import (
    IntegrationEstimate,
    IntegrationProvenance,
    SparseGridDiagnostics,
)
from ._fixed import integrate_fixed_component, integrate_fixed_density
from ._lowering import _component_base_mass
from ._plans import AdaptiveSparseGridPlan, SparseGridPlan
from ._precision import IntegrationPrecisionPolicy
from ._status import IntegrationStatus
from ._targets import ComponentTarget, DensityTarget


class SparseGridRealization(StrictModule):
    """A Smolyak batch and its immediately coarser comparison batch."""

    batch: PointIntegrationBatch
    previous: PointIntegrationBatch | None
    level: int
    num_unique_nodes: int
    previous_num_unique_nodes: int
    num_terms: int
    axis_rules: tuple[SmolyakAxisRule, ...]

    def __init__(
        self,
        batch: PointIntegrationBatch,
        previous: PointIntegrationBatch | None,
        /,
        *,
        level: int,
        num_terms: int,
        axis_rules: tuple[SmolyakAxisRule, ...],
    ):
        self.batch = batch
        self.previous = previous
        self.level = int(level)
        self.num_unique_nodes = int(batch.weights.data.size)
        self.previous_num_unique_nodes = (
            0 if previous is None else int(previous.weights.data.size)
        )
        self.num_terms = int(num_terms)
        self.axis_rules = axis_rules


class AdaptiveSparseGridDiagnostics(StrictModule):
    status: Array
    frontier_indicator: Array
    accepted_indices: int
    num_unique_nodes: int
    num_rounds: int


class AdaptiveSparseGridResult(StrictModule):
    """Prepared immutable realization and eager topology-transition evidence."""

    realization: SparseGridRealization
    estimate: IntegrationEstimate
    epochs: tuple[SmolyakRefinementEpoch, ...]
    diagnostics: AdaptiveSparseGridDiagnostics


def _unwrap(factor: Any, /) -> Any:
    return factor


def _smolyak_rule_from_terms(
    dimension: int,
    rules: tuple[SmolyakAxisRule, ...],
    terms,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    table: dict[
        tuple[tuple[str, int, int], ...],
        tuple[tuple[float, ...], float],
    ] = {}
    for term in terms:
        one_dimensional = tuple(
            smolyak_axis_data(rule, axis_level(term.index, axis))
            for axis, rule in enumerate(rules)
        )
        if any(data.quadrature_weights is None for data in one_dimensional):
            raise ValueError(
                "Every sparse-grid integration rule needs quadrature weights."
            )
        node_ranges = tuple(range(data.nodes.shape[0]) for data in one_dimensional)
        for position in itertools.product(*node_ranges):
            identifier = tuple(
                one_dimensional[axis].node_ids[point]
                for axis, point in enumerate(position)
            )
            node = tuple(
                float(one_dimensional[axis].nodes[point])
                for axis, point in enumerate(position)
            )
            weight = float(term.coefficient)
            for axis, point in enumerate(position):
                quadrature_weights = one_dimensional[axis].quadrature_weights
                if quadrature_weights is None:
                    raise RuntimeError("Smolyak quadrature weights disappeared.")
                weight *= float(quadrature_weights[point])
            if identifier in table:
                existing_node, existing_weight = table[identifier]
                table[identifier] = (existing_node, existing_weight + weight)
            else:
                table[identifier] = (node, weight)
    ordered = tuple(sorted(table))
    nodes = np.asarray(tuple(table[key][0] for key in ordered), dtype=float).reshape(
        (-1, dimension)
    )
    weights = np.asarray(tuple(table[key][1] for key in ordered), dtype=float)
    weight_scale = max(1.0, float(np.max(np.abs(weights), initial=0.0)))
    active = np.abs(weights) > 64.0 * np.finfo(float).eps * weight_scale
    return nodes[active], weights[active]


def _smolyak_rule(
    dimension: int,
    level: int,
    anisotropy: tuple[float, ...] | None,
    /,
    axis_rules: SmolyakAxisRule | tuple[SmolyakAxisRule, ...] = "clenshaw-curtis",
) -> tuple[np.ndarray, np.ndarray]:
    dimension_ = int(dimension)
    rules = normalize_axis_rules(
        dimension_,
        axis_rules,
        default="clenshaw-curtis",
        allowed=("clenshaw-curtis", "gauss-hermite"),
    )
    return _smolyak_rule_from_terms(
        dimension_,
        rules,
        smolyak_terms(dimension_, int(level), anisotropy),
    )


def _fixed_field(factor: Any, selector: Any, /) -> cx.Field:
    factor = _unwrap(factor)
    if isinstance(factor, AbstractScalarDomain):
        if isinstance(selector, FixedStart):
            value = factor.fixed("start")
        elif isinstance(selector, FixedEnd):
            value = factor.fixed("end")
        elif isinstance(selector, Fixed):
            value = selector.value
        else:
            raise TypeError("Expected a fixed scalar selector.")
        return cx.Field(jnp.asarray(value, dtype=float).reshape(()), dims=())
    if isinstance(factor, AbstractGeometry) and isinstance(selector, Fixed):
        return cx.Field(
            jnp.asarray(selector.value, dtype=float).reshape((factor.spatial_dim,)),
            dims=(None,),
        )
    raise TypeError("Unsupported fixed sparse-grid factor.")


def _materialize_level(
    target: ComponentTarget,
    plan: SparseGridPlan | AdaptiveSparseGridPlan,
    level: int,
    /,
    *,
    index_set: SmolyakIndexSet | None = None,
) -> PointIntegrationBatch:
    component = target.component
    if isinstance(component, ComponentSum):
        raise TypeError("Sparse-grid component unions must be integrated term by term.")
    fixed_labels = frozenset(
        label
        for label in component.domain.labels
        if isinstance(component.spec.selection_for(label), (FixedStart, FixedEnd, Fixed))
    )
    varying = tuple(
        label for label in component.domain.labels if label not in fixed_labels
    )
    if target.axes is not None:
        requested = (target.axes,) if isinstance(target.axes, str) else target.axes
        if len(requested) != len(varying) or frozenset(requested) != frozenset(varying):
            raise ValueError(
                "Sparse-grid targets use one coupled axis; axes must select every "
                f"non-fixed label {varying!r}."
            )
    unsupported = tuple(
        label
        for label in varying
        if not isinstance(component.spec.selection_for(label), Interior)
    )
    if unsupported:
        raise TypeError(
            "Sparse grids support only Interior() or fixed component selectors; "
            f"unsupported labels: {unsupported!r}."
        )
    if len(varying) != plan.dimension:
        raise ValueError(
            f"SparseGridPlan dimension={plan.dimension} but target has "
            f"{len(varying)} non-fixed factors."
        )
    factors = tuple(_unwrap(component.domain.factor(label)) for label in varying)
    if any(not isinstance(factor, AbstractScalarDomain) for factor in factors):
        raise TypeError("Sparse grids currently support scalar and probability factors.")
    if index_set is None:
        canonical_nodes, canonical_weights = _smolyak_rule(
            plan.dimension,
            level,
            plan.anisotropy,
            axis_rules=plan.axis_rules,
        )
    else:
        canonical_nodes, canonical_weights = _smolyak_rule_from_terms(
            plan.dimension,
            plan.axis_rules,
            smolyak_terms_for_index_set(index_set),
        )
    scale = jnp.asarray(1.0, dtype=float)
    mapped_columns: list[Array] = []
    for axis, (label, factor, rule) in enumerate(
        zip(varying, factors, plan.axis_rules, strict=True)
    ):
        coordinate = jnp.asarray(canonical_nodes[:, axis], dtype=float)
        if rule == "gauss-hermite":
            if not isinstance(factor, ProbabilityDomain):
                raise TypeError(
                    f"Gauss--Hermite axis {label!r} requires a probability factor."
                )
            transport = factor.reference_transport
            if transport.reference_measure != "standard-normal":
                raise ValueError(
                    f"Gauss--Hermite axis {label!r} requires a standard-normal "
                    "reference transport."
                )
            mapped = transport.from_reference(coordinate)
        elif isinstance(factor, ProbabilityDomain):
            transport = factor.reference_transport
            if transport.reference_measure != "uniform":
                raise ValueError(
                    f"Clenshaw--Curtis axis {label!r} requires bounded probability "
                    "support with a uniform reference transport."
                )
            mapped = transport.from_reference(coordinate)
            scale = scale * 0.5
        else:
            lower = factor.fixed("start")
            upper = factor.fixed("end")
            mapped = 0.5 * (upper - lower) * coordinate + 0.5 * (upper + lower)
            scale = scale * 0.5 * (upper - lower)
        mapped_columns.append(jnp.asarray(mapped))
    structure = SampleLayout((varying,)).canonicalize(
        component.domain.labels, fixed_labels=fixed_labels
    )
    axis_name = structure.axis_for(varying[0])
    if axis_name is None:
        raise RuntimeError("Sparse-grid structure has no sample axis.")
    points: dict[str, cx.Field] = {}
    varying_index = {label: index for index, label in enumerate(varying)}
    for label in component.domain.labels:
        if label in fixed_labels:
            points[label] = _fixed_field(
                component.domain.factor(label), component.spec.selection_for(label)
            )
        else:
            points[label] = cx.Field(
                mapped_columns[varying_index[label]], dims=(axis_name,)
            )
    point_batch = PointBatch(frozendict(points), structure)
    weights = cx.Field(
        scale * jnp.asarray(canonical_weights, dtype=float), dims=(axis_name,)
    )
    return PointIntegrationBatch(
        point_batch,
        weights,
        axes=(axis_name,),
        target_mass=_component_base_mass(component),
        provenance=(
            f"smolyak:level-{level}:rules-{'+'.join(plan.axis_rules)}"
            if index_set is None
            else f"smolyak:indices-{len(index_set.indices)}:"
            f"rules-{'+'.join(plan.axis_rules)}"
        ),
    )


def materialize_sparse_grid(
    target: ComponentTarget | DensityTarget,
    plan: SparseGridPlan,
    /,
) -> SparseGridRealization:
    """Materialize a Smolyak rule and its immediately coarser comparison."""
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ComponentTarget):
        raise TypeError("Sparse grids require a component-based target.")
    batch = _materialize_level(base, plan, plan.level)
    previous = None if plan.level == 1 else _materialize_level(base, plan, plan.level - 1)
    num_terms = len(smolyak_terms(plan.dimension, plan.level, plan.anisotropy))
    return SparseGridRealization(
        batch,
        previous,
        level=plan.level,
        num_terms=num_terms,
        axis_rules=plan.axis_rules,
    )


def integrate_sparse_grid(
    integrand: Any,
    target: ComponentTarget | DensityTarget,
    realization: SparseGridRealization,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
    precision: IntegrationPrecisionPolicy | None = None,
) -> IntegrationEstimate:
    """Reduce a Smolyak batch and report its deterministic level difference."""
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy.")
    if isinstance(target, DensityTarget):
        current = integrate_fixed_density(
            integrand,
            target,
            realization.batch,
            key=key,
            kwargs=kwargs,
            precision=precision_,
        )
        previous = (
            None
            if realization.previous is None
            else integrate_fixed_density(
                integrand,
                target,
                realization.previous,
                key=key,
                kwargs=kwargs,
                precision=precision_,
            )
        )
    else:
        current = integrate_fixed_component(
            integrand,
            target,
            realization.batch,
            key=key,
            kwargs=kwargs,
            precision=precision_,
        )
        previous = (
            None
            if realization.previous is None
            else integrate_fixed_component(
                integrand,
                target,
                realization.previous,
                key=key,
                kwargs=kwargs,
                precision=precision_,
            )
        )
    converged_status = int(IntegrationStatus.CONVERGED)
    current_status = jnp.where(
        current.status != converged_status,
        current.status,
        jnp.where(
            jnp.all(jnp.isfinite(current.value.data)),
            current.status,
            int(IntegrationStatus.NONFINITE_INTEGRAND),
        ),
    )
    if previous is None:
        level_difference = None
        error = None
        status = current_status
    else:
        level_difference = precision_.accumulation(
            current.value.data - previous.value.data
        )
        level_magnitude = jnp.abs(level_difference)
        difference_is_finite = jnp.all(jnp.isfinite(level_magnitude))
        error = precision_.decision(
            jnp.where(
                difference_is_finite,
                jnp.max(level_magnitude),
                jnp.asarray(jnp.inf, dtype=level_magnitude.dtype),
            )
        )
        previous_status = jnp.where(
            previous.status != converged_status,
            previous.status,
            jnp.where(
                jnp.all(jnp.isfinite(previous.value.data)),
                previous.status,
                int(IntegrationStatus.NONFINITE_INTEGRAND),
            ),
        )
        combined_status = jnp.where(
            current_status != converged_status,
            current_status,
            previous_status,
        )
        status = jnp.where(
            (combined_status == converged_status) & (~difference_is_finite),
            int(IntegrationStatus.NONFINITE_INTEGRAND),
            combined_status,
        )
    num_evaluations = current.num_evaluations + (
        0 if previous is None else previous.num_evaluations
    )
    diagnostics = SparseGridDiagnostics(
        status=status,
        num_evaluations=num_evaluations,
        level_difference=level_difference,
        level=realization.level,
        num_unique_nodes=realization.num_unique_nodes,
        previous_num_unique_nodes=realization.previous_num_unique_nodes,
        num_terms=realization.num_terms,
        axis_rules=realization.axis_rules,
    )
    return IntegrationEstimate(
        current.value,
        status=status,
        num_evaluations=num_evaluations,
        error_estimate=error,
        error_kind="sparse-grid-level-difference" if error is not None else None,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "sparse-grid", "component", realization.batch.provenance
        ),
    )


def _indicator(value: Array, norm: str, /) -> float:
    magnitude = np.abs(np.asarray(value))
    if norm == "max":
        return float(np.max(magnitude, initial=0.0))
    return float(np.sqrt(np.mean(magnitude**2)))


def _axis_node_count(rule: SmolyakAxisRule, level: int, /) -> int:
    if rule == "clenshaw-curtis":
        return 1 if level == 0 else 2**level + 1
    if rule == "gauss-hermite":
        return level + 1
    raise ValueError(f"Unsupported integration axis rule {rule!r}.")


def _index_set_node_count(
    dimension: int,
    rules: tuple[SmolyakAxisRule, ...],
    index_set: SmolyakIndexSet,
    /,
    *,
    limit: int,
) -> int:
    """Count active quadrature nodes without constructing a point batch."""
    weights: dict[tuple[tuple[str, int, int], ...], float] = {}
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
            identifier = tuple(
                axis_data[axis].node_ids[point] for axis, point in enumerate(position)
            )
            weight = float(term.coefficient)
            for axis, point in enumerate(position):
                quadrature_weights = axis_data[axis].quadrature_weights
                if quadrature_weights is None:
                    raise RuntimeError("Smolyak quadrature weights disappeared.")
                weight *= float(quadrature_weights[point])
            weights[identifier] = weights.get(identifier, 0.0) + weight
            if len(weights) > limit:
                return limit + 1
    weight_scale = max(
        1.0,
        max((abs(weight) for weight in weights.values()), default=0.0),
    )
    threshold = 64.0 * np.finfo(float).eps * weight_scale
    return sum(abs(weight) > threshold for weight in weights.values())


def prepare_adaptive_sparse_grid(
    integrand: Any,
    target: ComponentTarget | DensityTarget,
    plan: AdaptiveSparseGridPlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
    precision: IntegrationPrecisionPolicy | None = None,
) -> AdaptiveSparseGridResult:
    """Prepare a dimension-adaptive lower set and freeze its final realization."""
    if not isinstance(plan, AdaptiveSparseGridPlan):
        raise TypeError("plan must be an AdaptiveSparseGridPlan.")
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ComponentTarget):
        raise TypeError("Adaptive sparse grids require a component-based target.")
    index_set = SmolyakIndexSet.weighted_total_degree(
        plan.dimension,
        plan.initial_level,
        anisotropy=plan.anisotropy,
    )
    if len(index_set.indices) > plan.max_indices:
        raise ValueError("Initial sparse index set exceeds max_indices.")
    initial_nodes = _index_set_node_count(
        plan.dimension,
        plan.axis_rules,
        index_set,
        limit=plan.max_nodes,
    )
    if initial_nodes > plan.max_nodes:
        raise ValueError("Initial sparse grid exceeds max_nodes.")
    batch = _materialize_level(
        base,
        plan,
        plan.initial_level,
        index_set=index_set,
    )
    realization = SparseGridRealization(
        batch,
        None,
        level=plan.initial_level,
        num_terms=len(smolyak_terms_for_index_set(index_set)),
        axis_rules=plan.axis_rules,
    )
    current = integrate_sparse_grid(
        integrand,
        target,
        realization,
        key=key,
        kwargs=kwargs,
        precision=precision_,
    )
    epochs: list[SmolyakRefinementEpoch] = []
    previous_batch = None
    frontier_indicator = float("inf")
    terminal_status = IntegrationStatus.MAXIMUM_ROUNDS_REACHED
    for _ in range(plan.max_rounds):
        frontier = index_set.frontier(plan.anisotropy)
        if not frontier.candidates:
            terminal_status = IntegrationStatus.REFINEMENT_STAGNATION
            epochs.append(
                SmolyakRefinementEpoch(
                    index_set=index_set,
                    frontier=frontier,
                    selected=None,
                    indicators=(),
                    new_work=(),
                    status="stagnated",
                )
            )
            break
        if len(index_set.indices) >= plan.max_indices:
            terminal_status = IntegrationStatus.MAXIMUM_INDICES_REACHED
            epochs.append(
                SmolyakRefinementEpoch(
                    index_set=index_set,
                    frontier=frontier,
                    selected=None,
                    indicators=(),
                    new_work=(),
                    status="maximum-indices",
                )
            )
            break
        candidates: list[
            tuple[
                float,
                tuple[int, ...],
                SmolyakIndexSet,
                PointIntegrationBatch,
                IntegrationEstimate,
                int,
            ]
        ] = []
        indicators: list[float] = []
        work_values: list[int] = []
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
                plan.dimension,
                plan.axis_rules,
                proposed,
                limit=plan.max_nodes,
            )
            work = max(1, proposed_nodes - int(batch.weights.data.size))
            if proposed_nodes > plan.max_nodes:
                node_limited = True
                continue
            eligible_candidates.append(candidate)
            eligible_costs.append(cost)
            proposed_batch = _materialize_level(
                base,
                plan,
                plan.initial_level,
                index_set=proposed,
            )
            proposed_realization = SparseGridRealization(
                proposed_batch,
                None,
                level=max(sum(index) for index in proposed.indices) + 1,
                num_terms=len(smolyak_terms_for_index_set(proposed)),
                axis_rules=plan.axis_rules,
            )
            proposed_estimate = integrate_sparse_grid(
                integrand,
                target,
                proposed_realization,
                key=key,
                kwargs=kwargs,
                precision=precision_,
            )
            indicator = _indicator(
                proposed_estimate.value.data - current.value.data,
                plan.indicator_norm,
            )
            indicators.append(indicator)
            work_values.append(work)
            candidates.append(
                (
                    indicator / work,
                    candidate,
                    proposed,
                    proposed_batch,
                    proposed_estimate,
                    work,
                )
            )
        eligible_frontier = SmolyakFrontier(
            candidates=tuple(eligible_candidates),
            anisotropic_costs=tuple(eligible_costs),
        )
        frontier_indicator = float("inf") if node_limited else float(sum(indicators))
        if not candidates:
            terminal_status = IntegrationStatus.MAXIMUM_NODES_REACHED
            epochs.append(
                SmolyakRefinementEpoch(
                    index_set=index_set,
                    frontier=frontier,
                    selected=None,
                    indicators=(),
                    new_work=(),
                    status="maximum-nodes",
                )
            )
            break
        magnitude = _indicator(current.value.data, plan.indicator_norm)
        absolute = 0.0 if plan.absolute_tolerance is None else plan.absolute_tolerance
        relative = 0.0 if plan.relative_tolerance is None else plan.relative_tolerance
        if frontier_indicator <= absolute + relative * magnitude:
            terminal_status = IntegrationStatus.CONVERGED
            epochs.append(
                SmolyakRefinementEpoch(
                    index_set=index_set,
                    frontier=eligible_frontier,
                    selected=None,
                    indicators=tuple(indicators),
                    new_work=tuple(work_values),
                    status="converged",
                )
            )
            break
        candidates.sort(key=lambda value: (-value[0], value[1]))
        _, selected, proposed, proposed_batch, proposed_estimate, _ = candidates[0]
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
        previous_batch = batch
        index_set = proposed
        batch = proposed_batch
        current = proposed_estimate
    frozen = SparseGridRealization(
        batch,
        previous_batch,
        level=max(sum(index) for index in index_set.indices) + 1,
        num_terms=len(smolyak_terms_for_index_set(index_set)),
        axis_rules=plan.axis_rules,
    )
    diagnostics = AdaptiveSparseGridDiagnostics(
        status=jnp.asarray(int(terminal_status), dtype=jnp.int32),
        frontier_indicator=jnp.asarray(frontier_indicator, dtype=float),
        accepted_indices=len(index_set.indices),
        num_unique_nodes=int(batch.weights.data.size),
        num_rounds=len(epochs),
    )
    return AdaptiveSparseGridResult(
        realization=frozen,
        estimate=current,
        epochs=tuple(epochs),
        diagnostics=diagnostics,
    )


__all__ = [
    "AdaptiveSparseGridDiagnostics",
    "AdaptiveSparseGridResult",
    "SparseGridRealization",
    "integrate_sparse_grid",
    "materialize_sparse_grid",
    "prepare_adaptive_sparse_grid",
]
