#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import ceil, prod
from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._method import (
    AbstractLinearCombinatorialMethod,
    CombinatorialPlan,
    make_combinatorial_plan,
)
from ._problem import AbstractCombinatorialSpace, LinearCombinatorialProblem
from ._types import (
    CombinatorialCertificate,
    CombinatorialCertification,
    CombinatorialFeasibility,
    CombinatorialMethodCapabilities,
    CombinatorialProvenance,
    CombinatorialResult,
    CombinatorialStatus,
)


class ExplicitDecision(StrictModule):
    """Canonical index and payload selected from an explicit decision catalog."""

    index: Array
    value: PyTree[Array]


def _catalog(
    name: str,
    tree: PyTree[Any],
    /,
    *,
    real_features: bool,
) -> tuple[PyTree[Array], int, Any, tuple[tuple[int, ...], ...], tuple[str, ...]]:
    path_leaves, tree_definition = jax.tree_util.tree_flatten_with_path(tree)
    if not path_leaves:
        raise ValueError(f"{name} must contain at least one array leaf.")
    arrays: list[Array] = []
    payload_shapes: list[tuple[int, ...]] = []
    dtypes: list[str] = []
    count: int | None = None
    for path, raw in path_leaves:
        if isinstance(raw, (str, bytes)):
            raise TypeError(f"{name} leaves must be numerical or boolean arrays.")
        array = jnp.asarray(raw)
        if array.ndim == 0:
            raise ValueError(f"{name} leaves require a leading candidate axis.")
        if np.dtype(array.dtype).kind not in "biufc":
            raise TypeError(f"{name} leaves must be numerical or boolean arrays.")
        if real_features and jnp.issubdtype(array.dtype, jnp.complexfloating):
            raise TypeError("objective feature leaves must be real-valued.")
        leading = int(array.shape[0])
        if leading == 0:
            raise ValueError(f"{name} candidate dimensions must be nonempty.")
        if count is None:
            count = leading
        elif leading != count:
            label = jax.tree_util.keystr(path) or "<root>"
            raise ValueError(
                f"every {name} leaf must share one candidate count; {label} differs."
            )
        if (
            real_features
            and jnp.issubdtype(array.dtype, jnp.inexact)
            and not bool(jnp.all(jnp.isfinite(array)))
        ):
            raise ValueError("objective feature catalogs must be finite.")
        arrays.append(array)
        payload_shapes.append(tuple(int(size) for size in array.shape[1:]))
        dtypes.append(str(array.dtype))
    assert count is not None
    return (
        tree_definition.unflatten(arrays),
        count,
        tree_definition,
        tuple(payload_shapes),
        tuple(dtypes),
    )


class ExplicitDecisionSpace(AbstractCombinatorialSpace):
    """Explicit finite feasible set with an independent linear feature catalog."""

    decisions: PyTree[Array]
    features: PyTree[Array]
    valid: Array
    candidate_count: int = eqx.field(static=True)
    decision_tree_definition: Any = eqx.field(static=True)
    feature_tree_definition: Any = eqx.field(static=True)
    decision_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    feature_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    decision_dtypes: tuple[str, ...] = eqx.field(static=True)
    feature_dtypes: tuple[str, ...] = eqx.field(static=True)
    _structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        decisions: PyTree[Any],
        features: PyTree[Any],
        /,
        *,
        valid: Any | None = None,
    ):
        (
            decision_values,
            decision_count,
            decision_tree,
            decision_shapes,
            decision_dtypes,
        ) = _catalog("decisions", decisions, real_features=False)
        (
            feature_values,
            feature_count,
            feature_tree,
            feature_shapes,
            feature_dtypes,
        ) = _catalog("features", features, real_features=True)
        if feature_count != decision_count:
            raise ValueError("decisions and features must share one candidate count.")
        if valid is None:
            validity = jnp.ones((decision_count,), dtype=bool)
        else:
            validity = jnp.asarray(valid, dtype=bool)
            if validity.shape != (decision_count,):
                raise ValueError(
                    f"valid must have shape {(decision_count,)}; got {validity.shape}."
                )
        self.decisions = decision_values
        self.features = feature_values
        self.valid = validity
        self.candidate_count = decision_count
        self.decision_tree_definition = decision_tree
        self.feature_tree_definition = feature_tree
        self.decision_shapes = decision_shapes
        self.feature_shapes = feature_shapes
        self.decision_dtypes = decision_dtypes
        self.feature_dtypes = feature_dtypes
        self._structure_id = canonical_fingerprint(
            {
                "kind": "explicit-decision-space",
                "decisions": array_tree_fingerprint(decision_values),
                "features": array_tree_fingerprint(feature_values),
                "valid": array_tree_fingerprint(validity),
            }
        )

    @property
    def structure_id(self) -> str:
        return self._structure_id

    def decision_spec(self, /) -> ExplicitDecision:
        leaves = tuple(
            jax.ShapeDtypeStruct(shape, np.dtype(dtype))
            for shape, dtype in zip(
                self.decision_shapes,
                self.decision_dtypes,
                strict=True,
            )
        )
        return ExplicitDecision(
            jax.ShapeDtypeStruct((), jnp.int32),
            self.decision_tree_definition.unflatten(leaves),
        )

    def feature_spec(self, /) -> PyTree[jax.ShapeDtypeStruct]:
        leaves = tuple(
            jax.ShapeDtypeStruct(shape, np.dtype(dtype))
            for shape, dtype in zip(
                self.feature_shapes,
                self.feature_dtypes,
                strict=True,
            )
        )
        return self.feature_tree_definition.unflatten(leaves)

    def _safe_index(self, index: Array, /) -> tuple[Array, Array]:
        in_range = (index >= 0) & (index < self.candidate_count)
        return jnp.clip(index, 0, self.candidate_count - 1), in_range

    def canonicalize(self, decision: ExplicitDecision, /) -> ExplicitDecision:
        if not isinstance(decision, ExplicitDecision):
            raise TypeError("explicit decisions must be ExplicitDecision values.")
        index = jnp.asarray(decision.index, dtype=jnp.int32)
        safe, in_range = self._safe_index(index)
        values = jax.tree_util.tree_map(lambda catalog: catalog[safe], self.decisions)
        values = jax.tree_util.tree_map(
            lambda value: jnp.where(
                in_range.reshape(in_range.shape + (1,) * (value.ndim - in_range.ndim)),
                value,
                jnp.zeros_like(value),
            ),
            values,
        )
        return ExplicitDecision(jnp.where(in_range, index, -1), values)

    def encode(self, decision: ExplicitDecision, /) -> PyTree[Array]:
        canonical = self.canonicalize(decision)
        safe, in_range = self._safe_index(canonical.index)
        values = jax.tree_util.tree_map(lambda catalog: catalog[safe], self.features)
        return jax.tree_util.tree_map(
            lambda value: jnp.where(
                in_range.reshape(in_range.shape + (1,) * (value.ndim - in_range.ndim)),
                value,
                jnp.zeros_like(value),
            ),
            values,
        )

    def audit(self, decision: ExplicitDecision, /) -> CombinatorialFeasibility:
        canonical = self.canonicalize(decision)
        safe, in_range = self._safe_index(canonical.index)
        catalog_valid = self.valid[safe]
        value_residual = jnp.zeros_like(canonical.index, dtype=float)
        provided_leaves, provided_tree = jax.tree_util.tree_flatten(decision.value)
        canonical_leaves, canonical_tree = jax.tree_util.tree_flatten(canonical.value)
        if provided_tree != canonical_tree:
            raise ValueError(
                "explicit decision payload has incompatible PyTree structure."
            )
        for provided, expected in zip(provided_leaves, canonical_leaves, strict=True):
            value = jnp.asarray(provided)
            if value.shape != expected.shape:
                raise ValueError("explicit decision payload has incompatible shape.")
            mismatch = value != expected
            payload_rank = value.ndim - canonical.index.ndim
            if payload_rank:
                mismatch = jnp.any(mismatch, axis=tuple(range(-payload_rank, 0)))
            value_residual = value_residual + mismatch.astype(value_residual.dtype)
        feasible = in_range & catalog_valid & (value_residual == 0)
        return CombinatorialFeasibility(
            feasible,
            (~in_range).astype(float) + (~catalog_valid).astype(float) + value_residual,
        )


class ExhaustiveLinearOracle(AbstractLinearCombinatorialMethod):
    """Exact batched enumeration over an ExplicitDecisionSpace."""

    batch_size: int = eqx.field(static=True)
    maximum_candidates: int = eqx.field(static=True)

    def __init__(self, batch_size: int = 256, *, maximum_candidates: int = 1_000_000):
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be a positive integer.")
        if isinstance(maximum_candidates, bool) or not isinstance(
            maximum_candidates, Integral
        ):
            raise TypeError("maximum_candidates must be a positive integer.")
        if int(batch_size) <= 0 or int(maximum_candidates) <= 0:
            raise ValueError("enumeration resource limits must be positive.")
        self.batch_size = int(batch_size)
        self.maximum_candidates = int(maximum_candidates)

    @property
    def method_id(self) -> str:
        return "native-exhaustive-linear"

    @property
    def capabilities(self) -> CombinatorialMethodCapabilities:
        return CombinatorialMethodCapabilities(
            exact=True,
            jax_native=True,
            jit=True,
            batched=True,
            signed_costs=True,
            deterministic_ties=True,
            optimality_certificate=True,
            surrogate_pullback=True,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (
            ("batch_size", str(self.batch_size)),
            ("maximum_candidates", str(self.maximum_candidates)),
        )

    def plan(
        self,
        problem: LinearCombinatorialProblem,
        certification: CombinatorialCertification,
        /,
    ) -> CombinatorialPlan:
        if not isinstance(problem.space, ExplicitDecisionSpace):
            raise TypeError("ExhaustiveLinearOracle requires ExplicitDecisionSpace.")
        count = problem.space.candidate_count
        if count > self.maximum_candidates:
            raise ValueError(
                f"explicit decision count {count} exceeds maximum_candidates "
                f"{self.maximum_candidates}."
            )
        feature_size = sum(prod(shape) for shape in problem.space.feature_shapes)
        return make_combinatorial_plan(
            problem,
            self,
            certification,
            work_estimate=problem.batch_size * count * feature_size,
            workspace_elements=problem.batch_size * self.batch_size,
            certificate_kind="complete-enumeration",
        )

    def solve(
        self,
        problem: LinearCombinatorialProblem,
        plan: CombinatorialPlan,
        /,
    ) -> CombinatorialResult:
        space = problem.space
        if not isinstance(space, ExplicitDecisionSpace):
            raise TypeError("ExhaustiveLinearOracle requires ExplicitDecisionSpace.")
        batch_shape = problem.batch_shape
        flat_batch = problem.batch_size
        count = space.candidate_count
        width = min(self.batch_size, count)
        num_batches = ceil(count / width)
        costs = jax.tree_util.tree_map(
            lambda value: value.reshape((flat_batch,) + value.shape[len(batch_shape) :]),
            problem.costs,
        )
        finite_cost = jnp.ones((flat_batch,), dtype=bool)
        for value in jax.tree_util.tree_leaves(costs):
            feature_rank = value.ndim - 1
            finite = jnp.isfinite(value)
            if feature_rank:
                finite = jnp.all(finite, axis=tuple(range(1, value.ndim)))
            finite_cost = finite_cost & finite

        initial = (
            jnp.full((flat_batch,), jnp.inf, dtype=np.dtype(problem.cost_dtype)),
            jnp.full((flat_batch,), jnp.inf, dtype=np.dtype(problem.cost_dtype)),
            jnp.full((flat_batch,), -1, dtype=jnp.int32),
            jnp.zeros((flat_batch,), dtype=bool),
            jnp.zeros((flat_batch,), dtype=bool),
        )

        feature_catalogs = jax.tree_util.tree_leaves(space.features)
        cost_leaves = jax.tree_util.tree_leaves(costs)

        def body(batch_index, state):
            best, second, best_index, has_best, has_second = state
            raw_indices = batch_index * width + jnp.arange(width, dtype=jnp.int32)
            in_range = raw_indices < count
            safe_indices = jnp.minimum(raw_indices, count - 1)
            candidate_valid = in_range & space.valid[safe_indices]
            score = jnp.zeros((flat_batch, width), dtype=best.dtype)
            for cost, catalog in zip(cost_leaves, feature_catalogs, strict=True):
                feature = catalog[safe_indices].astype(cost.dtype)
                feature_rank = feature.ndim - 1
                product_ = cost[:, None] * feature[None, ...]
                if feature_rank:
                    product_ = jnp.sum(
                        product_,
                        axis=tuple(range(-feature_rank, 0)),
                    )
                score = score + product_
            valid = finite_cost[:, None] & candidate_valid[None, :] & jnp.isfinite(score)
            combined_values = jnp.concatenate(
                (best[:, None], second[:, None], score),
                axis=-1,
            )
            combined_valid = jnp.concatenate(
                (has_best[:, None], has_second[:, None], valid),
                axis=-1,
            )
            safe_values = jnp.where(combined_valid, combined_values, jnp.inf)
            order = jnp.argsort(safe_values, axis=-1, stable=True)
            first_slot = order[:, 0]
            second_slot = order[:, 1]
            next_best = jnp.take_along_axis(
                combined_values, first_slot[:, None], axis=-1
            )[:, 0]
            next_second = jnp.take_along_axis(
                combined_values, second_slot[:, None], axis=-1
            )[:, 0]
            next_has_best = jnp.any(combined_valid, axis=-1)
            next_has_second = jnp.sum(combined_valid, axis=-1) > 1
            combined_indices = jnp.concatenate(
                (
                    best_index[:, None],
                    jnp.full((flat_batch, 1), -1, dtype=jnp.int32),
                    jnp.broadcast_to(raw_indices, (flat_batch, width)),
                ),
                axis=-1,
            )
            next_index = jnp.take_along_axis(
                combined_indices, first_slot[:, None], axis=-1
            )[:, 0]
            return (
                jnp.where(next_has_best, next_best, jnp.inf),
                jnp.where(next_has_second, next_second, jnp.inf),
                jnp.where(next_has_best, next_index, -1),
                next_has_best,
                next_has_second,
            )

        best, second, flat_index, has_best, has_second = jax.lax.fori_loop(
            0,
            num_batches,
            body,
            initial,
        )
        safe_index = jnp.clip(flat_index, 0, count - 1)
        decision_values = jax.tree_util.tree_map(
            lambda catalog: catalog[safe_index], space.decisions
        )
        decision_values = jax.tree_util.tree_map(
            lambda value: jnp.where(
                has_best.reshape((flat_batch,) + (1,) * (value.ndim - 1)),
                value,
                jnp.zeros_like(value),
            ).reshape(batch_shape + value.shape[1:]),
            decision_values,
        )
        decision = ExplicitDecision(
            jnp.where(has_best, flat_index, -1).reshape(batch_shape),
            decision_values,
        )
        features = jax.tree_util.tree_map(
            lambda catalog: catalog[safe_index], space.features
        )
        features = jax.tree_util.tree_map(
            lambda value: jnp.where(
                has_best.reshape((flat_batch,) + (1,) * (value.ndim - 1)),
                value,
                jnp.zeros_like(value),
            ).reshape(batch_shape + value.shape[1:]),
            features,
        )
        features = jax.tree_util.tree_map(
            lambda value: value.astype(np.dtype(problem.cost_dtype)), features
        )
        objective = problem.objective(features)
        best_shaped = best.reshape(batch_shape)
        finite_shaped = finite_cost.reshape(batch_shape)
        feasible = space.audit(decision)
        tolerance = plan.certification.threshold(objective, best_shaped)
        objective_residual = jnp.abs(objective - best_shaped)
        objective_consistent = has_best.reshape(batch_shape) & (
            objective_residual <= tolerance
        )
        optimality = finite_shaped & feasible.feasible & objective_consistent
        status = jnp.where(
            ~finite_shaped,
            int(CombinatorialStatus.NONFINITE_INPUT),
            jnp.where(
                ~has_best.reshape(batch_shape),
                int(CombinatorialStatus.INFEASIBLE),
                jnp.where(
                    optimality,
                    int(CombinatorialStatus.OPTIMAL),
                    int(CombinatorialStatus.CERTIFICATION_FAILED),
                ),
            ),
        ).astype(jnp.int32)
        valid = status == int(CombinatorialStatus.OPTIMAL)
        zero = jnp.zeros(batch_shape, dtype=objective.dtype)
        gap_available = has_best.reshape(batch_shape)
        tie_available = has_second.reshape(batch_shape)
        tie_margin = jnp.where(
            tie_available,
            (second - best).reshape(batch_shape),
            jnp.nan,
        )
        certificate = CombinatorialCertificate(
            finite=finite_shaped,
            feasible=feasible.feasible,
            objective_consistent=objective_consistent,
            optimality_proven=optimality,
            primal_residual=feasible.residual.astype(objective.dtype),
            dual_residual=zero,
            absolute_gap=zero,
            relative_gap=zero,
            tie_margin=tie_margin,
            dual_available=jnp.zeros(batch_shape, dtype=bool),
            gap_available=gap_available,
            tie_available=tie_available,
        )
        provenance = CombinatorialProvenance(
            problem_id=problem.problem_id,
            structure_id=problem.structure_id,
            method_id=self.method_id,
            plan_id=plan.plan_id,
            implementation="phydrax-native-jax",
            tie_policy="lowest-candidate-index",
            certificate_kind=plan.certificate_kind,
            exact=True,
            signed_costs=True,
            configuration=plan.configuration,
        )
        decision = ExplicitDecision(
            jnp.where(valid, decision.index, -1),
            jax.tree_util.tree_map(
                lambda value: jnp.where(
                    valid.reshape(batch_shape + (1,) * (value.ndim - len(batch_shape))),
                    value,
                    jnp.zeros_like(value),
                ),
                decision.value,
            ),
        )
        features = jax.tree_util.tree_map(
            lambda value: jnp.where(
                valid.reshape(batch_shape + (1,) * (value.ndim - len(batch_shape))),
                value,
                jnp.zeros_like(value),
            ),
            features,
        )
        return CombinatorialResult(
            decision=decision,
            features=features,
            objective_value=jnp.where(valid, objective, jnp.nan),
            status=status,
            valid=valid,
            certificate=certificate,
            iterations=jnp.full(batch_shape, num_batches, dtype=jnp.int32),
            work=jnp.full(batch_shape, count, dtype=jnp.int64),
            provenance=provenance,
            batch_shape=batch_shape,
        )


__all__ = [
    "ExhaustiveLinearOracle",
    "ExplicitDecision",
    "ExplicitDecisionSpace",
]
