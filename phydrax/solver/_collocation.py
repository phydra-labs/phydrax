#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import cast, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..domain._probability import ProbabilityDomain
from ..integration._sparse_grid import _smolyak_rule
from ..operators.interpolation._plans import SmolyakInterpolationRule
from ..operators.interpolation._smolyak import (
    _build_blocks,
    _build_topology,
    _resolve_axis_rules,
    SmolyakInterpolant,
)


CollocationAxisRule: TypeAlias = Literal["auto", "clenshaw-curtis", "gauss-hermite"]

COLLOCATION_SUCCESS = 0
COLLOCATION_SOLVER_FAILURE = 1
COLLOCATION_NONFINITE = 2


def _name(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _rules(
    factors: tuple[ProbabilityDomain, ...],
    requested: CollocationAxisRule | Sequence[CollocationAxisRule] | None,
    /,
) -> tuple[Literal["clenshaw-curtis", "gauss-hermite"], ...]:
    if requested is None or requested == "auto":
        values: tuple[str, ...] = ("auto",) * len(factors)
    elif isinstance(requested, str):
        values = (requested,) * len(factors)
    else:
        values = tuple(str(rule) for rule in requested)
    if len(values) != len(factors):
        raise ValueError("axis_rules must contain one rule per uncertain input.")
    resolved: list[Literal["clenshaw-curtis", "gauss-hermite"]] = []
    for factor, rule in zip(factors, values, strict=True):
        if not factor.supports_reference_transform:
            raise ValueError(
                f"Probability factor {factor.label!r} has no canonical reference transform."
            )
        selected = (
            "gauss-hermite"
            if rule == "auto" and factor.reference_measure == "standard-normal"
            else "clenshaw-curtis"
            if rule == "auto"
            else rule
        )
        if selected not in ("clenshaw-curtis", "gauss-hermite"):
            raise ValueError(
                "Collocation axis rules must be 'auto', 'clenshaw-curtis', or "
                "'gauss-hermite'."
            )
        expected = "standard-normal" if selected == "gauss-hermite" else "uniform"
        if factor.reference_measure != expected:
            raise ValueError(
                f"Rule {selected!r} for {factor.label!r} requires reference measure "
                f"{expected!r}."
            )
        resolved.append(selected)
    return tuple(resolved)


class StochasticCollocationPlan(StrictModule):
    """Smolyak plan over explicit input-uncertainty axes only.

    Process realizations are deliberately absent: stochastic-process sampling remains a
    separate axis owned by the conditional solver or an outer coupled estimator.
    """

    factors: tuple[ProbabilityDomain, ...]
    level: int = eqx.field(static=True)
    anisotropy: tuple[float, ...] = eqx.field(static=True)
    axis_rules: tuple[Literal["clenshaw-curtis", "gauss-hermite"], ...] = eqx.field(
        static=True
    )
    include_previous: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    input_axis_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        factors: Sequence[ProbabilityDomain],
        level: int,
        /,
        *,
        anisotropy: Sequence[float] | None = None,
        axis_rules: (CollocationAxisRule | Sequence[CollocationAxisRule] | None) = "auto",
        include_previous: bool = True,
        plan_id: str = "stochastic-collocation",
    ):
        probability_factors = tuple(factors)
        if not probability_factors or any(
            not isinstance(factor, ProbabilityDomain) for factor in probability_factors
        ):
            raise TypeError("factors must contain one or more ProbabilityDomain objects.")
        labels = tuple(factor.label for factor in probability_factors)
        if len(set(labels)) != len(labels):
            raise ValueError("Stochastic collocation factor labels must be unique.")
        resolved_level = int(level)
        if resolved_level < 1:
            raise ValueError("level must be positive.")
        if anisotropy is None:
            resolved_anisotropy = (1.0,) * len(probability_factors)
        else:
            resolved_anisotropy = tuple(float(value) for value in anisotropy)
            if len(resolved_anisotropy) != len(probability_factors):
                raise ValueError(
                    "anisotropy must contain one weight per uncertain input."
                )
            if any(
                not np.isfinite(value) or value <= 0.0 for value in resolved_anisotropy
            ):
                raise ValueError("anisotropy weights must be finite and positive.")
        self.factors = probability_factors
        self.level = resolved_level
        self.anisotropy = resolved_anisotropy
        self.axis_rules = _rules(probability_factors, axis_rules)
        self.include_previous = bool(include_previous)
        self.plan_id = _name(plan_id, owner="plan_id")
        self.input_axis_labels = labels

    @property
    def dimension(self) -> int:
        return len(self.factors)


class StochasticCollocationNode(StrictModule):
    """One stable, named uncertain-input node passed to a conditional solver."""

    reference_coordinates: Array
    physical_coordinates: Array
    parameters: frozendict[str, Array]
    node_id: str = eqx.field(static=True)
    index: int = eqx.field(static=True)


class StochasticCollocationDesign(StrictModule):
    """Materialized union of fine and coarse interpolation/quadrature nodes."""

    plan: StochasticCollocationPlan
    nodes: tuple[StochasticCollocationNode, ...]
    current_indices: Array
    previous_indices: Array | None
    current_quadrature_indices: Array
    current_quadrature_weights: Array
    previous_quadrature_indices: Array | None
    previous_quadrature_weights: Array | None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


class StochasticCollocationNodeEvaluation(StrictModule):
    """Explicit conditional-solver value and status for one collocation node."""

    value: Array
    valid: Array
    status: Array
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        /,
        *,
        valid: ArrayLike = True,
        status: ArrayLike = COLLOCATION_SUCCESS,
        provenance: str = "conditional-solver",
    ):
        valid_value = jnp.asarray(valid, dtype=bool)
        status_value = jnp.asarray(status, dtype=jnp.int32)
        if valid_value.shape != () or status_value.shape != ():
            raise ValueError("valid and status must be scalar values.")
        self.value = jnp.asarray(value)
        self.valid = valid_value
        self.status = status_value
        self.provenance = _name(provenance, owner="provenance")


class StochasticCollocationDiagnostics(StrictModule):
    """Node failures, quadrature normalization, and level-difference diagnostics."""

    node_status: Array
    node_valid: Array
    current_weight_sum: Array
    previous_weight_sum: Array | None
    mean_level_difference_norm: Array | None
    variance_level_difference_norm: Array | None
    num_nodes: int = eqx.field(static=True)
    num_current_nodes: int = eqx.field(static=True)
    num_previous_nodes: int = eqx.field(static=True)
    num_failed_nodes: int = eqx.field(static=True)
    input_axis_labels: tuple[str, ...] = eqx.field(static=True)
    axis_rules: tuple[str, ...] = eqx.field(static=True)


class StochasticCollocationResult(StrictModule):
    """Conditional solves, fitted surrogate, moments, and coarse-level comparison."""

    design: StochasticCollocationDesign
    evaluations: tuple[StochasticCollocationNodeEvaluation, ...]
    values: Array
    valid: Array
    status: Array
    interpolant: SmolyakInterpolant | None
    previous_interpolant: SmolyakInterpolant | None
    mean: Array
    variance: Array
    second_moment: Array
    previous_mean: Array | None
    previous_variance: Array | None
    mean_level_difference: Array | None
    variance_level_difference: Array | None
    diagnostics: StochasticCollocationDiagnostics

    @property
    def successful(self) -> Array:
        return (
            jnp.all(self.valid[self.design.current_indices])
            & jnp.all(jnp.isfinite(self.mean))
            & jnp.all(jnp.isfinite(self.variance))
            & (self.interpolant is not None)
        )


def _reference_identity(row: np.ndarray, /) -> tuple[str, ...]:
    return tuple(np.float64(0.0 if value == 0.0 else value).hex() for value in row)


def _node_id(plan: StochasticCollocationPlan, identity: tuple[str, ...], /) -> str:
    digest = hashlib.sha256(f"phydrax-collocation\0{plan.plan_id}\0".encode())
    for label, rule, coordinate in zip(
        plan.input_axis_labels, plan.axis_rules, identity, strict=True
    ):
        digest.update(f"{label}\0{rule}\0{coordinate}\0".encode())
    return digest.hexdigest()


def _physical_coordinates(
    plan: StochasticCollocationPlan,
    reference: np.ndarray,
    /,
) -> Array:
    return jnp.stack(
        tuple(
            factor.from_reference(jnp.asarray(reference[axis], dtype=float)).reshape(())
            for axis, factor in enumerate(plan.factors)
        )
    )


def _quadrature(
    plan: StochasticCollocationPlan,
    level: int,
    index_by_identity: Mapping[tuple[str, ...], int],
    /,
) -> tuple[Array, Array]:
    nodes, weights = _smolyak_rule(
        plan.dimension,
        level,
        plan.anisotropy,
        axis_rules=plan.axis_rules,
    )
    scale = 1.0
    for factor, rule in zip(plan.factors, plan.axis_rules, strict=True):
        if rule == "clenshaw-curtis":
            if factor.reference_measure != "uniform":
                raise ValueError(
                    "Clenshaw--Curtis collocation requires uniform reference axes."
                )
            scale *= 0.5
    indices = tuple(index_by_identity[_reference_identity(row)] for row in nodes)
    return jnp.asarray(indices, dtype=jnp.int32), scale * jnp.asarray(weights)


def materialize_stochastic_collocation(
    plan: StochasticCollocationPlan,
    /,
) -> StochasticCollocationDesign:
    """Materialize stable input nodes while sharing fine/coarse duplicates exactly."""
    if not isinstance(plan, StochasticCollocationPlan):
        raise TypeError("plan must be a StochasticCollocationPlan.")
    current_reference, _ = _build_topology(
        plan.dimension,
        plan.level,
        plan.anisotropy,
        plan.axis_rules,
    )
    previous_level = plan.level - 1 if plan.include_previous and plan.level > 1 else None
    previous_reference = (
        None
        if previous_level is None
        else _build_topology(
            plan.dimension,
            previous_level,
            plan.anisotropy,
            plan.axis_rules,
        )[0]
    )
    identities: list[tuple[str, ...]] = []
    references: list[np.ndarray] = []
    index_by_identity: dict[tuple[str, ...], int] = {}
    sources = (
        (current_reference,)
        if previous_reference is None
        else (current_reference, previous_reference)
    )
    for source in sources:
        for row in source:
            identity = _reference_identity(row)
            if identity not in index_by_identity:
                index_by_identity[identity] = len(identities)
                identities.append(identity)
                references.append(np.asarray(row, dtype=float))
    nodes = tuple(
        StochasticCollocationNode(
            reference_coordinates=jnp.asarray(reference),
            physical_coordinates=(physical := _physical_coordinates(plan, reference)),
            parameters=frozendict(
                {
                    label: physical[axis]
                    for axis, label in enumerate(plan.input_axis_labels)
                }
            ),
            node_id=_node_id(plan, identity),
            index=index,
        )
        for index, (identity, reference) in enumerate(
            zip(identities, references, strict=True)
        )
    )
    current_indices = jnp.asarray(
        tuple(index_by_identity[_reference_identity(row)] for row in current_reference),
        dtype=jnp.int32,
    )
    previous_indices = (
        None
        if previous_reference is None
        else jnp.asarray(
            tuple(
                index_by_identity[_reference_identity(row)] for row in previous_reference
            ),
            dtype=jnp.int32,
        )
    )
    current_quadrature_indices, current_weights = _quadrature(
        plan, plan.level, index_by_identity
    )
    if previous_level is None:
        previous_quadrature_indices = None
        previous_weights = None
    else:
        previous_quadrature_indices, previous_weights = _quadrature(
            plan, previous_level, index_by_identity
        )
    return StochasticCollocationDesign(
        plan=plan,
        nodes=nodes,
        current_indices=current_indices,
        previous_indices=previous_indices,
        current_quadrature_indices=current_quadrature_indices,
        current_quadrature_weights=current_weights,
        previous_quadrature_indices=previous_quadrature_indices,
        previous_quadrature_weights=previous_weights,
    )


def _interpolant(
    design: StochasticCollocationDesign,
    level: int,
    indices: Array,
    values: Array,
    valid: Array,
    /,
) -> SmolyakInterpolant | None:
    selected_values = values[indices]
    selected_valid = valid[indices]
    if not bool(jnp.all(selected_valid)) or bool(jnp.any(~jnp.isfinite(selected_values))):
        return None
    _, topologies = _build_topology(
        design.plan.dimension,
        level,
        design.plan.anisotropy,
        design.plan.axis_rules,
    )
    blocks = _build_blocks(topologies, selected_values)
    interpolation_rules = cast(
        tuple[SmolyakInterpolationRule, ...], design.plan.axis_rules
    )
    resolved_rules = _resolve_axis_rules(design.plan.factors, interpolation_rules)
    return SmolyakInterpolant(
        blocks=blocks,
        factors=design.plan.factors,
        axis_labels=design.plan.input_axis_labels,
        axis_rules=resolved_rules,
        anisotropy=design.plan.anisotropy,
        level=level,
        output_shape=tuple(int(size) for size in values.shape[1:]),
        num_terms=len(topologies),
        num_evaluations=int(indices.shape[0]),
        maximum_active_dimension=max(len(topology.axes) for topology in topologies),
    )


def _moments(
    values: Array,
    valid: Array,
    indices: Array,
    weights: Array,
    /,
) -> tuple[Array, Array, Array]:
    selected = values[indices]
    selected_valid = valid[indices]
    output_shape = values.shape[1:]
    if not bool(jnp.all(selected_valid)) or bool(jnp.any(~jnp.isfinite(selected))):
        dtype = jnp.result_type(values.dtype, float)
        invalid = jnp.full(output_shape, jnp.nan, dtype=dtype)
        return invalid, jnp.real(invalid), jnp.real(invalid)
    weight_shape = weights.shape + (1,) * len(output_shape)
    expanded = weights.reshape(weight_shape)
    mean = jnp.sum(expanded * selected, axis=0)
    second = jnp.sum(expanded * jnp.abs(selected) ** 2, axis=0)
    variance = jnp.maximum(jnp.real(second - jnp.abs(mean) ** 2), 0.0)
    return mean, variance, jnp.real(second)


def assemble_stochastic_collocation(
    design: StochasticCollocationDesign,
    evaluations: Sequence[StochasticCollocationNodeEvaluation],
    /,
) -> StochasticCollocationResult:
    """Assemble externally or locally evaluated nodes into one audited result."""
    if not isinstance(design, StochasticCollocationDesign):
        raise TypeError("design must be a StochasticCollocationDesign.")
    node_evaluations = tuple(evaluations)
    if len(node_evaluations) != design.num_nodes or any(
        not isinstance(item, StochasticCollocationNodeEvaluation)
        for item in node_evaluations
    ):
        raise ValueError("evaluations must contain one node evaluation per design node.")
    output_shape = node_evaluations[0].value.shape
    if any(item.value.shape != output_shape for item in node_evaluations):
        raise ValueError("Every collocation node value must have the same output shape.")
    values = jnp.stack(tuple(item.value for item in node_evaluations))
    declared_valid = jnp.stack(tuple(item.valid for item in node_evaluations))
    declared_status = jnp.stack(tuple(item.status for item in node_evaluations))
    finite = jnp.all(
        jnp.isfinite(values).reshape(
            (design.num_nodes, prod(output_shape) if output_shape else 1)
        ),
        axis=-1,
    )
    valid = declared_valid & (declared_status == COLLOCATION_SUCCESS) & finite
    status = jnp.where(
        ~finite,
        COLLOCATION_NONFINITE,
        jnp.where(
            declared_valid & (declared_status == COLLOCATION_SUCCESS),
            COLLOCATION_SUCCESS,
            jnp.where(
                declared_status == COLLOCATION_SUCCESS,
                COLLOCATION_SOLVER_FAILURE,
                declared_status,
            ),
        ),
    ).astype(jnp.int32)
    current_interpolant = _interpolant(
        design,
        design.plan.level,
        design.current_indices,
        values,
        valid,
    )
    mean, variance, second = _moments(
        values,
        valid,
        design.current_quadrature_indices,
        design.current_quadrature_weights,
    )
    if (
        design.previous_indices is None
        or design.previous_quadrature_indices is None
        or design.previous_quadrature_weights is None
    ):
        previous_interpolant = None
        previous_mean = None
        previous_variance = None
        mean_difference = None
        variance_difference = None
        previous_weight_sum = None
        num_previous = 0
    else:
        previous_interpolant = _interpolant(
            design,
            design.plan.level - 1,
            design.previous_indices,
            values,
            valid,
        )
        previous_mean, previous_variance, _ = _moments(
            values,
            valid,
            design.previous_quadrature_indices,
            design.previous_quadrature_weights,
        )
        mean_difference = mean - previous_mean
        variance_difference = variance - previous_variance
        previous_weight_sum = jnp.sum(design.previous_quadrature_weights)
        num_previous = int(design.previous_indices.shape[0])
    mean_difference_norm = (
        None if mean_difference is None else jnp.linalg.norm(jnp.ravel(mean_difference))
    )
    variance_difference_norm = (
        None
        if variance_difference is None
        else jnp.linalg.norm(jnp.ravel(variance_difference))
    )
    diagnostics = StochasticCollocationDiagnostics(
        node_status=status,
        node_valid=valid,
        current_weight_sum=jnp.sum(design.current_quadrature_weights),
        previous_weight_sum=previous_weight_sum,
        mean_level_difference_norm=mean_difference_norm,
        variance_level_difference_norm=variance_difference_norm,
        num_nodes=design.num_nodes,
        num_current_nodes=int(design.current_indices.shape[0]),
        num_previous_nodes=num_previous,
        num_failed_nodes=int(jnp.sum(~valid)),
        input_axis_labels=design.plan.input_axis_labels,
        axis_rules=design.plan.axis_rules,
    )
    return StochasticCollocationResult(
        design=design,
        evaluations=node_evaluations,
        values=values,
        valid=valid,
        status=status,
        interpolant=current_interpolant,
        previous_interpolant=previous_interpolant,
        mean=mean,
        variance=variance,
        second_moment=second,
        previous_mean=previous_mean,
        previous_variance=previous_variance,
        mean_level_difference=mean_difference,
        variance_level_difference=variance_difference,
        diagnostics=diagnostics,
    )


def evaluate_stochastic_collocation(
    design: StochasticCollocationDesign,
    node_solver: Callable[
        [StochasticCollocationNode], StochasticCollocationNodeEvaluation
    ],
    /,
) -> StochasticCollocationResult:
    """Evaluate every unique node exactly once and assemble the collocation result."""
    if not isinstance(design, StochasticCollocationDesign):
        raise TypeError("design must be a StochasticCollocationDesign.")
    if not callable(node_solver):
        raise TypeError("node_solver must be callable.")
    evaluations = tuple(node_solver(node) for node in design.nodes)
    return assemble_stochastic_collocation(design, evaluations)


def run_stochastic_collocation(
    plan: StochasticCollocationPlan,
    node_solver: Callable[
        [StochasticCollocationNode], StochasticCollocationNodeEvaluation
    ],
    /,
) -> StochasticCollocationResult:
    """Materialize, solve, fit, and integrate one stochastic collocation plan."""
    return evaluate_stochastic_collocation(
        materialize_stochastic_collocation(plan), node_solver
    )


__all__ = [
    "assemble_stochastic_collocation",
    "COLLOCATION_NONFINITE",
    "COLLOCATION_SOLVER_FAILURE",
    "COLLOCATION_SUCCESS",
    "CollocationAxisRule",
    "evaluate_stochastic_collocation",
    "materialize_stochastic_collocation",
    "run_stochastic_collocation",
    "StochasticCollocationDesign",
    "StochasticCollocationDiagnostics",
    "StochasticCollocationNode",
    "StochasticCollocationNodeEvaluation",
    "StochasticCollocationPlan",
    "StochasticCollocationResult",
]
