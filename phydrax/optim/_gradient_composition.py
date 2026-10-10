#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import (
    tree_allfinite,
    tree_inner,
    tree_norm,
    tree_scale,
    validate_inexact_tree,
    validate_real_inexact_tree,
)
from ..linalg._dense_pseudoinverse import (
    apply_connected_pseudoinverse,
    apply_pseudoinverse,
    factor_pseudoinverse,
)
from ..linalg._policies import RankPolicy
from ..typing import parse


ConflictFreeFailureMode: TypeAlias = Literal["status", "error"]


class ConflictFreeGradientStatus(IntEnum):
    SUCCESS = 0
    STATIONARY = 1
    INFEASIBLE = 2
    NONFINITE = 3


class ConflictFreeGradientPolicy(StrictModule, NonTrainableState):
    """Numerical policy for inverse-gradient multi-objective composition."""

    rank_policy: RankPolicy
    minimum_norm: float = eqx.field(static=True)
    projection_tolerance: float = eqx.field(static=True)
    failure: ConflictFreeFailureMode = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        rank_policy: RankPolicy | None = None,
        minimum_norm: float = 1e-12,
        projection_tolerance: float = 1e-10,
        failure: ConflictFreeFailureMode = "status",
    ) -> None:
        rank = RankPolicy() if rank_policy is None else rank_policy
        if not isinstance(rank, RankPolicy):
            raise TypeError("rank_policy must be a RankPolicy.")
        norm = float(minimum_norm)
        tolerance = float(projection_tolerance)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("minimum_norm must be finite and strictly positive.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("projection_tolerance must be finite and nonnegative.")
        failure = parse(failure, ConflictFreeFailureMode, "failure")
        self.rank_policy = rank
        self.minimum_norm = norm
        self.projection_tolerance = tolerance
        self.failure = failure
        self.policy_id = canonical_fingerprint(
            {
                "kind": "conflict-free-gradient-policy",
                "rank": {
                    "relative_cutoff": rank.relative_cutoff,
                    "absolute_cutoff": rank.absolute_cutoff,
                    "require_full_rank": rank.require_full_rank,
                },
                "minimum_norm": norm,
                "projection_tolerance": tolerance,
                "failure": failure,
            }
        )


class ConflictFreeGradientResult(StrictModule):
    """Composed direction and complete local multi-objective evidence."""

    direction: PyTree[Array]
    norms: Array
    cosine_matrix: Array
    projections: Array
    active: Array
    stationary: Array
    conflicts: Array
    rank: Array
    rank_cutoff: Array
    condition_estimate: Array
    direction_norm: Array
    successful: Array
    status: Array
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        direction: PyTree[Array],
        /,
        *,
        norms: Array,
        cosine_matrix: Array,
        projections: Array,
        active: Array,
        stationary: Array,
        conflicts: Array,
        rank: Array,
        rank_cutoff: Array,
        condition_estimate: Array,
        direction_norm: Array,
        successful: Array,
        status: Array,
        policy_id: str,
    ) -> None:
        self.direction = direction
        self.norms = jnp.asarray(norms)
        self.cosine_matrix = jnp.asarray(cosine_matrix)
        self.projections = jnp.asarray(projections)
        self.active = jnp.asarray(active, dtype=jnp.bool_)
        self.stationary = jnp.asarray(stationary, dtype=jnp.bool_)
        self.conflicts = jnp.asarray(conflicts, dtype=jnp.bool_)
        self.rank = jnp.asarray(rank)
        self.rank_cutoff = jnp.asarray(rank_cutoff)
        self.condition_estimate = jnp.asarray(condition_estimate)
        self.direction_norm = jnp.asarray(direction_norm)
        self.successful = jnp.asarray(successful, dtype=jnp.bool_)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.policy_id = str(policy_id)


def _tree_add(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x + y, left, right)


def conflict_free_gradient(
    gradients: Sequence[PyTree[Any]],
    /,
    *,
    active: Any | None = None,
    policy: ConflictFreeGradientPolicy | None = None,
) -> ConflictFreeGradientResult:
    """Compose objective gradients without materializing parameter coordinates."""

    values = tuple(
        validate_inexact_tree(value, name=f"objective gradient {index}")
        for index, value in enumerate(gradients)
    )
    if not values:
        raise ValueError("At least one objective gradient is required.")
    structure = jax.tree.structure(values[0])
    if any(jax.tree.structure(value) != structure for value in values[1:]):
        raise ValueError("Objective gradients must have congruent PyTree structures.")
    shapes = tuple(
        tuple(leaf.shape for leaf in jax.tree.leaves(value)) for value in values
    )
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError("Objective gradient leaves must have congruent shapes.")
    resolved = ConflictFreeGradientPolicy() if policy is None else policy
    if not isinstance(resolved, ConflictFreeGradientPolicy):
        raise TypeError("policy must be a ConflictFreeGradientPolicy.")
    count = len(values)
    active_mask = (
        jnp.ones((count,), dtype=jnp.bool_)
        if active is None
        else jnp.asarray(active, dtype=jnp.bool_)
    )
    if active_mask.shape != (count,):
        raise ValueError("active must contain one Boolean per objective gradient.")

    norms = jnp.stack(tuple(tree_norm(value) for value in values))
    finite = jnp.stack(tuple(tree_allfinite(value) for value in values))
    stationary = active_mask & finite & (norms <= resolved.minimum_norm)
    effective = active_mask & finite & ~stationary
    safe_norms = jnp.where(effective, norms, jnp.ones_like(norms))
    normalized = tuple(
        tree_scale(effective[index] / safe_norms[index], value)
        for index, value in enumerate(values)
    )
    gram = jnp.stack(
        tuple(
            jnp.stack(tuple(tree_inner(left, right) for right in normalized))
            for left in normalized
        )
    )
    factors = factor_pseudoinverse(
        gram,
        resolved.rank_policy,
        hermitian=True,
    )
    coefficients = apply_pseudoinverse(
        factors,
        effective.astype(gram.dtype),
    )
    candidate = tree_scale(0.0, values[0])
    for coefficient, direction in zip(coefficients, normalized, strict=True):
        candidate = _tree_add(candidate, tree_scale(coefficient, direction))
    candidate_norm = tree_norm(candidate)
    candidate_finite = tree_allfinite(candidate)
    usable_candidate = (
        candidate_finite & factors.finite & (candidate_norm > resolved.minimum_norm)
    )
    unit_candidate = tree_scale(
        jnp.where(usable_candidate, 1.0 / candidate_norm, 0.0),
        candidate,
    )
    unit_projections = jnp.stack(
        tuple(tree_inner(value, unit_candidate) for value in values)
    )
    scale = jnp.sum(jnp.where(effective, unit_projections, 0.0))
    positive_scale = jnp.maximum(scale, 0.0)
    direction = tree_scale(positive_scale, unit_candidate)
    projections = jnp.stack(tuple(tree_inner(value, direction) for value in values))
    conflicts = effective & (projections < -resolved.projection_tolerance)
    all_input_finite = jnp.all(~active_mask | finite)
    active_count = jnp.sum(effective.astype(jnp.int32))
    stationary_only = active_count == 0
    feasible = usable_candidate & ~jnp.any(conflicts) & (positive_scale > 0.0)
    successful = all_input_finite & (stationary_only | feasible)
    status = jnp.where(
        ~all_input_finite,
        int(ConflictFreeGradientStatus.NONFINITE),
        jnp.where(
            stationary_only,
            int(ConflictFreeGradientStatus.STATIONARY),
            jnp.where(
                feasible,
                int(ConflictFreeGradientStatus.SUCCESS),
                int(ConflictFreeGradientStatus.INFEASIBLE),
            ),
        ),
    )
    if resolved.failure == "error":
        direction = jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                ~successful,
                "Objective gradients do not admit a finite conflict-free direction.",
            ),
            direction,
        )
    return ConflictFreeGradientResult(
        direction,
        norms=norms,
        cosine_matrix=gram,
        projections=projections,
        active=active_mask,
        stationary=stationary,
        conflicts=conflicts,
        rank=factors.rank,
        rank_cutoff=factors.rank_cutoff,
        condition_estimate=factors.condition_estimate,
        direction_norm=tree_norm(direction),
        successful=successful,
        status=status,
        policy_id=resolved.policy_id,
    )


class ActiveGradientBankResult(StrictModule):
    """Local whole-bank direction evidence, never a feasibility certificate."""

    direction: PyTree[Array]
    valid: Array
    resource_refused: Array
    work_units: Array
    rank: Array
    equality_rank: Array
    projections: Array
    equality_projections: Array
    bucket: Array
    equality_bucket: Array

    def __init__(
        self,
        *,
        direction: PyTree[Array],
        valid: Array,
        resource_refused: Array,
        work_units: Array,
        rank: Array,
        equality_rank: Array,
        projections: Array,
        equality_projections: Array,
        bucket: Array,
        equality_bucket: Array,
    ) -> None:
        self.direction = direction
        self.valid = valid
        self.resource_refused = resource_refused
        self.work_units = work_units
        self.rank = rank
        self.equality_rank = equality_rank
        self.projections = projections
        self.equality_projections = equality_projections
        self.bucket = bucket
        self.equality_bucket = equality_bucket


def active_gradient_bank_work(
    parameter_size: int, active_bucket: int, equality_bucket: int, bank_capacity: int, /
) -> int:
    """Conservative row-algebra units for one complete executed bank path."""
    a, k, d = active_bucket, equality_bucket, parameter_size
    return (
        18 * bank_capacity
        + 2 * a
        + 8 * (a * a + k * k + a * k) * d
        + 64 * (a**3 + k**3)
        + 16 * (a + k) * d
        + 16 * (a * a + k * k)
    )


def compose_active_gradient_bank(
    gradients: PyTree[Any],
    raw_improvement_targets: Array,
    inequality_active: Array,
    equality_active: Array,
    original_norm: Array,
    remaining_work: Array,
    /,
    *,
    rank_policy: RankPolicy,
    minimum_norm: float = 1.0e-14,
    projection_tolerance: float = 1.0e-10,
) -> ActiveGradientBankResult:
    """Compose ALL signed improvement rows in a bounded parameter bank.

    Signed inequalities require strictly positive projections. Equalities
    retain the caller's numerical linear-solve threshold; scientific zero
    tolerance remains with actual source residual/publication admission.
    Raw targets prescribe ORIGINAL row-dot-direction progress, not unit-row
    margins. A compatible rank solve preserves their ratios; final seed-norm
    rescaling preserves achieved ratios, not absolute target magnitudes.
    Zero/rank failure only refuses this direction, never proves stationarity
    or scientific infeasibility. Every complete bucket is admitted before
    its gathers, Gram products or canonical economy rank factorization.
    """
    values = validate_real_inexact_tree(gradients, name="active gradient bank")
    leaves = jax.tree.leaves(values)
    if not leaves or any(leaf.ndim < 1 for leaf in leaves):
        raise ValueError("A gradient bank requires nonempty leading-row arrays.")
    capacity = leaves[0].shape[0]
    if capacity not in (4, 8, 16, 32):
        raise ValueError("Gradient bank capacity must be 4, 8, 16, or 32.")
    if any(leaf.ndim < 1 or leaf.shape[0] != capacity for leaf in leaves):
        raise ValueError("Every gradient leaf must have the same leading bank capacity.")
    if inequality_active.shape != (capacity,) or equality_active.shape != (capacity,):
        raise ValueError("Bank activity masks must match its leading capacity.")
    if raw_improvement_targets.shape != (capacity,):
        raise ValueError("Raw improvement targets must match the leading bank capacity.")
    if raw_improvement_targets.dtype.kind != "f":
        raise TypeError("Raw improvement targets must be real floating arrays.")
    if inequality_active.dtype.kind != "b" or equality_active.dtype.kind != "b":
        raise TypeError("Bank activity masks must be Boolean arrays.")
    if (
        original_norm.ndim != 0
        or remaining_work.ndim != 0
        or remaining_work.dtype.kind != "i"
    ):
        raise TypeError("Original norm and signed model-work allowance must be scalars.")
    if not isinstance(rank_policy, RankPolicy):
        raise TypeError("rank_policy must be a RankPolicy.")
    if (
        not np.isfinite(minimum_norm)
        or not np.isfinite(projection_tolerance)
        or minimum_norm < 0.0
        or projection_tolerance < 0.0
    ):
        raise ValueError("Numerical norm and projection thresholds must be nonnegative.")
    parameter_size = sum(leaf.size // capacity for leaf in leaves)
    dtype = leaves[0].dtype
    scan_work = 12 * capacity

    def refused(
        work: Array, resource: Array, a: int = 0, k: int = 0
    ) -> ActiveGradientBankResult:
        return ActiveGradientBankResult(
            direction=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape[1:], dtype=leaf.dtype), values
            ),
            valid=jnp.asarray(False),
            resource_refused=resource,
            work_units=work,
            rank=jnp.asarray(0, dtype=jnp.int32),
            equality_rank=jnp.asarray(0, dtype=jnp.int32),
            projections=jnp.zeros((capacity,), dtype=dtype),
            equality_projections=jnp.zeros((capacity,), dtype=dtype),
            bucket=jnp.asarray(a, dtype=jnp.int32),
            equality_bucket=jnp.asarray(k, dtype=jnp.int32),
        )

    def gram(bank: PyTree[Array], rows: int) -> Array:
        result = jnp.zeros((rows, rows), dtype=dtype)
        for leaf in jax.tree.leaves(bank):
            coordinates = leaf.reshape((rows, -1))
            result = result + coordinates @ jnp.conj(coordinates.T)
        return result

    def execute(
        a: int,
        count_a: Array,
        projected: PyTree[Array],
        equality_rank: Array,
        equality_finite: Array,
        equality_bucket: Array,
        prior_work: Array,
    ) -> ActiveGradientBankResult:
        # Equality projection is complete before this independent A dispatch.
        post_work = (
            4 * a * parameter_size
            + 4 * capacity * parameter_size
            + 4 * parameter_size
            + 8 * a * a
            + 8 * capacity
        )
        preparation_work = (
            active_gradient_bank_work(parameter_size, a, 0, capacity)
            - 64 * a**3
            - post_work
            - scan_work
        )

        def admitted(_: None) -> ActiveGradientBankResult:
            active_rows = jnp.nonzero(inequality_active, size=a, fill_value=0)[0]
            active_valid = jnp.arange(a) < count_a
            bank = jax.tree.map(
                lambda leaf: jnp.where(
                    active_valid.reshape((a,) + (1,) * (leaf.ndim - 1)),
                    leaf[active_rows],
                    0.0,
                ),
                projected,
            )
            norms_squared = jnp.zeros((a,), dtype=dtype)
            for leaf in jax.tree.leaves(bank):
                norms_squared = norms_squared + jnp.sum(
                    jnp.real(leaf.reshape((a, -1)) * jnp.conj(leaf.reshape((a, -1)))),
                    axis=-1,
                )
            norms = jnp.sqrt(norms_squared)
            rows_usable = active_valid & jnp.isfinite(norms) & (norms > minimum_norm)
            denominators = jnp.where(rows_usable, norms, 1.0)
            normalized = jax.tree.map(
                lambda leaf: leaf / denominators.reshape((a,) + (1,) * (leaf.ndim - 1)),
                bank,
            )
            normalized_gram = gram(normalized, a)
            compact_targets = jnp.where(
                active_valid, raw_improvement_targets[active_rows], 0.0
            )
            factors = apply_connected_pseudoinverse(
                normalized_gram,
                compact_targets / denominators,
                rank_policy,
                active_valid,
                remaining_work - prior_work - preparation_work - post_work,
            )
            factor_spent = prior_work + preparation_work + factors.work_units

            def completed(_: None) -> ActiveGradientBankResult:
                candidate = jax.tree.map(
                    lambda leaf: jnp.tensordot(factors.value, leaf, axes=(0, 0)),
                    normalized,
                )
                norm = tree_norm(candidate)
                finite = tree_allfinite(candidate) & factors.finite & equality_finite
                usable = (
                    finite
                    & jnp.isfinite(original_norm)
                    & (original_norm > 0.0)
                    & (norm > minimum_norm)
                )
                direction = tree_scale(
                    jnp.where(
                        usable, original_norm / jnp.where(norm > 0.0, norm, 1.0), 0.0
                    ),
                    candidate,
                )
                projections = jnp.zeros((capacity,), dtype=dtype)
                for leaf, direction_leaf in zip(
                    jax.tree.leaves(values), jax.tree.leaves(direction), strict=True
                ):
                    projections = projections + jnp.real(
                        leaf.reshape((capacity, -1))
                        @ jnp.conj(direction_leaf.reshape((-1,)))
                    )
                valid = (
                    usable
                    & jnp.all((~active_valid) | rows_usable)
                    & jnp.all(
                        (~inequality_active)
                        | (jnp.isfinite(projections) & (projections > 0.0))
                    )
                    & jnp.all(
                        (~equality_active)
                        | (
                            jnp.isfinite(projections)
                            & (jnp.abs(projections) <= projection_tolerance)
                        )
                    )
                )
                return ActiveGradientBankResult(
                    direction=direction,
                    valid=valid,
                    resource_refused=jnp.asarray(False),
                    work_units=factor_spent + post_work,
                    rank=factors.rank,
                    equality_rank=equality_rank,
                    projections=projections,
                    equality_projections=jnp.where(equality_active, projections, 0.0),
                    bucket=jnp.asarray(a, dtype=jnp.int32),
                    equality_bucket=equality_bucket,
                )

            return jax.lax.cond(
                factors.finite & (~factors.resource_refused),
                completed,
                lambda _: refused(factor_spent, factors.resource_refused, a),
                None,
            )

        return jax.lax.cond(
            remaining_work >= prior_work + preparation_work,
            admitted,
            lambda _: refused(prior_work, jnp.asarray(True), a),
            None,
        )

    def project_equalities(
        k: int, count_k: Array
    ) -> tuple[PyTree[Array], Array, Array, Array, Array, Array]:
        if k == 0:
            return (
                values,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(True),
                jnp.asarray(False),
                jnp.asarray(scan_work, dtype=jnp.int64),
                jnp.asarray(0, dtype=jnp.int32),
            )
        # Full-B cross/project work is larger than A-column projection and
        # is explicitly charged, not hidden by the compilation cutback.
        equality_work = (
            6 * capacity
            + 8 * (k * k + k * capacity) * parameter_size
            + 16 * (k + capacity) * parameter_size
            + 64 * k**3
            + 16 * k * k
        )

        def admitted(_: None) -> tuple[PyTree[Array], Array, Array, Array, Array, Array]:
            rows = jnp.nonzero(equality_active, size=k, fill_value=0)[0]
            valid = jnp.arange(k) < count_k
            eq = jax.tree.map(
                lambda leaf: jnp.where(
                    valid.reshape((k,) + (1,) * (leaf.ndim - 1)), leaf[rows], 0.0
                ),
                values,
            )
            included = inequality_active | equality_active
            whole = jax.tree.map(
                lambda leaf: jnp.where(
                    included.reshape((capacity,) + (1,) * (leaf.ndim - 1)), leaf, 0.0
                ),
                values,
            )
            factors = factor_pseudoinverse(gram(eq, k), rank_policy, hermitian=True)
            cross = jnp.zeros((k, capacity), dtype=dtype)
            for eq_leaf, leaf in zip(
                jax.tree.leaves(eq), jax.tree.leaves(whole), strict=True
            ):
                cross = cross + eq_leaf.reshape((k, -1)) @ jnp.conj(
                    leaf.reshape((capacity, -1)).T
                )
            coefficients = apply_pseudoinverse(factors, cross)
            projected = jax.tree.map(
                lambda leaf, eq_leaf: (
                    leaf
                    - (jnp.conj(coefficients.T) @ eq_leaf.reshape((k, -1))).reshape(
                        leaf.shape
                    )
                ),
                whole,
                eq,
            )
            return (
                projected,
                factors.rank,
                factors.finite,
                jnp.asarray(False),
                jnp.asarray(scan_work + equality_work, dtype=jnp.int64),
                jnp.asarray(k, dtype=jnp.int32),
            )

        return jax.lax.cond(
            remaining_work >= scan_work + equality_work,
            admitted,
            lambda _: (
                values,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(scan_work, dtype=jnp.int64),
                jnp.asarray(k, dtype=jnp.int32),
            ),
            None,
        )

    def scanned(_: None) -> ActiveGradientBankResult:
        count_a = jnp.sum(inequality_active)
        count_k = jnp.sum(equality_active)
        targets_valid = jnp.all(
            (~inequality_active)
            | (jnp.isfinite(raw_improvement_targets) & (raw_improvement_targets > 0.0))
        ) & jnp.all(
            (~equality_active)
            | (jnp.isfinite(raw_improvement_targets) & (raw_improvement_targets == 0.0))
        )
        a_index = jnp.where(
            count_a <= 4, 0, jnp.where(count_a <= 8, 1, jnp.where(count_a <= 16, 2, 3))
        )
        k_index = jnp.where(
            count_k == 0,
            0,
            jnp.where(
                count_k <= 4,
                1,
                jnp.where(count_k <= 8, 2, jnp.where(count_k <= 16, 3, 4)),
            ),
        )

        def eligible(_: None) -> ActiveGradientBankResult:
            projected, eq_rank, eq_finite, eq_resource, spent, eq_bucket = jax.lax.switch(
                k_index,
                tuple(
                    (lambda _, k=k: project_equalities(k, count_k))
                    for k in (0, 4, 8, 16, 32)
                ),
                None,
            )
            # Independent A switch: connected factor graphs occur once/A,
            # rather than once for every A/equality combination.
            return jax.lax.cond(
                eq_finite & (~eq_resource),
                lambda _: jax.lax.switch(
                    a_index,
                    tuple(
                        (
                            lambda _, a=a: execute(
                                a,
                                count_a,
                                projected,
                                eq_rank,
                                eq_finite,
                                eq_bucket,
                                spent,
                            )
                        )
                        for a in (4, 8, 16, 32)
                    ),
                    None,
                ),
                lambda _: refused(spent, eq_resource),
                None,
            )

        return jax.lax.cond(
            (count_a > 0) & targets_valid,
            eligible,
            lambda _: refused(
                jnp.asarray(scan_work, dtype=jnp.int64), jnp.asarray(False)
            ),
            None,
        )

    return jax.lax.cond(
        remaining_work >= scan_work,
        scanned,
        lambda _: refused(jnp.asarray(0, dtype=jnp.int64), jnp.asarray(True)),
        None,
    )


__all__ = [
    "ConflictFreeFailureMode",
    "ConflictFreeGradientPolicy",
    "ConflictFreeGradientResult",
    "ConflictFreeGradientStatus",
    "conflict_free_gradient",
    "ActiveGradientBankResult",
    "active_gradient_bank_work",
    "compose_active_gradient_bank",
]
