#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._spaces import ArraySpace
from ._subspaces import LinearSubspace


class AlgebraDerivationStatus(IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    AMBIGUOUS_RANK = 2
    RESIDUAL_FAILURE = 3


class AlgebraDerivationPolicy(StrictModule, NonTrainableState):
    absolute_cutoff: float = eqx.field(static=True)
    relative_cutoff: float = eqx.field(static=True)
    minimum_singular_gap: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_cutoff: float = 1e-12,
        relative_cutoff: float = 1e-10,
        minimum_singular_gap: float = 1e4,
        residual_tolerance: float = 1e-10,
        dtype: Any = np.float64,
    ):
        values = tuple(
            float(value)
            for value in (
                absolute_cutoff,
                relative_cutoff,
                minimum_singular_gap,
                residual_tolerance,
            )
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Algebra derivation policy values must be finite and nonnegative."
            )
        if values[2] < 1.0:
            raise ValueError("minimum_singular_gap must be at least one.")
        dtype_ = np.dtype(dtype)
        if not np.issubdtype(dtype_, np.floating):
            raise TypeError("Algebra derivation preparation requires floating dtype.")
        (
            self.absolute_cutoff,
            self.relative_cutoff,
            self.minimum_singular_gap,
            self.residual_tolerance,
        ) = values
        self.dtype = dtype_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "algebra-derivation-policy-v1",
                "absolute_cutoff": values[0].hex(),
                "relative_cutoff": values[1].hex(),
                "minimum_singular_gap": values[2].hex(),
                "residual_tolerance": values[3].hex(),
                "dtype": dtype_.str,
            }
        )


class AlgebraDerivationPlan(StrictModule, NonTrainableState):
    constraint: Any
    policy: AlgebraDerivationPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        constraint: Any,
        /,
        *,
        policy: AlgebraDerivationPolicy | None = None,
    ):
        from ..metrix.algebra import AlgebraDerivationConstraint

        if not isinstance(constraint, AlgebraDerivationConstraint):
            raise TypeError("constraint must be an AlgebraDerivationConstraint.")
        policy_ = AlgebraDerivationPolicy() if policy is None else policy
        if not isinstance(policy_, AlgebraDerivationPolicy):
            raise TypeError("policy must be AlgebraDerivationPolicy or None.")
        self.constraint = constraint
        self.policy = policy_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "algebra-derivation-plan-v1",
                "constraint": constraint.constraint_id,
                "policy": policy_.policy_id,
            }
        )


class PreparedAlgebraDerivations(StrictModule):
    plan: AlgebraDerivationPlan
    subspace: LinearSubspace
    singular_values: Array
    cutoff: Array
    singular_gap: Array
    maximum_leibniz_residual: Array
    maximum_unit_fixing_residual: Array
    maximum_commutator_closure_residual: Array
    converged: Array
    status: Array
    constraint_matrix_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    basis_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: AlgebraDerivationPlan,
        subspace: LinearSubspace,
        singular_values: Array,
        cutoff: Array,
        singular_gap: Array,
        maximum_leibniz_residual: Array,
        maximum_unit_fixing_residual: Array,
        maximum_commutator_closure_residual: Array,
        converged: Array,
        status: Array,
        /,
        *,
        constraint_matrix_bytes: int,
        workspace_bytes: int,
        basis_bytes: int,
    ):
        self.plan = plan
        self.subspace = subspace
        self.singular_values = jnp.asarray(singular_values)
        self.cutoff = jnp.asarray(cutoff)
        self.singular_gap = jnp.asarray(singular_gap)
        self.maximum_leibniz_residual = jnp.asarray(maximum_leibniz_residual)
        self.maximum_unit_fixing_residual = jnp.asarray(maximum_unit_fixing_residual)
        self.maximum_commutator_closure_residual = jnp.asarray(
            maximum_commutator_closure_residual
        )
        self.converged = jnp.asarray(converged, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.constraint_matrix_bytes = int(constraint_matrix_bytes)
        self.workspace_bytes = int(workspace_bytes)
        self.basis_bytes = int(basis_bytes)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-algebra-derivations-v1",
                "plan": plan.plan_id,
                "subspace": subspace.subspace_id,
                "basis": array_tree_fingerprint(subspace.basis),
                "constraint_matrix_bytes": int(constraint_matrix_bytes),
                "workspace_bytes": int(workspace_bytes),
                "basis_bytes": int(basis_bytes),
            }
        )

    @property
    def dimension(self) -> Array:
        return self.subspace.dimension

    def project(self, matrix: Array, /) -> Array:
        return self.subspace.project(matrix)


def plan_algebra_derivations(
    algebra: Any,
    /,
    *,
    budget: Any | None = None,
    policy: AlgebraDerivationPolicy | None = None,
) -> AlgebraDerivationPlan:
    from ..metrix.algebra import AlgebraDerivationConstraint

    return AlgebraDerivationPlan(
        AlgebraDerivationConstraint(algebra, budget=budget),
        policy=policy,
    )


def _maximum_or_zero(value: Array, /) -> Array:
    if value.size == 0:
        return jnp.asarray(0.0, dtype=value.dtype)
    return jnp.max(jnp.abs(value))


def prepare_algebra_derivations(
    plan: AlgebraDerivationPlan,
    /,
) -> PreparedAlgebraDerivations:
    """Prepare a numerically certified derivation nullspace from exact constraints."""
    if not isinstance(plan, AlgebraDerivationPlan):
        raise TypeError("plan must be an AlgebraDerivationPlan.")
    constraint = plan.constraint
    policy = plan.policy
    dimension = constraint.algebra.coordinate_dimension
    itemsize = policy.dtype.itemsize
    matrix_bytes = constraint.equation_count * constraint.variable_count * itemsize
    workspace_bytes = (
        constraint.equation_count * constraint.variable_count
        + constraint.variable_count**2
        + constraint.variable_count
    ) * itemsize
    maximum_basis_bytes = constraint.variable_count**2 * itemsize
    constraint.budget.admit_materialization(
        matrix_bytes,
        workspace_bytes,
        maximum_basis_bytes,
    )
    matrix = constraint.materialize(policy.dtype)
    _, singular_values, right_adjoint = jnp.linalg.svd(matrix, full_matrices=False)
    maximum = jnp.max(singular_values)
    cutoff = jnp.maximum(
        jnp.asarray(policy.absolute_cutoff, dtype=singular_values.dtype),
        jnp.asarray(policy.relative_cutoff, dtype=singular_values.dtype) * maximum,
    )
    retained = singular_values > cutoff
    rank = int(jax.device_get(jnp.sum(retained, dtype=jnp.int32)))
    nullity = constraint.variable_count - rank
    right_vectors = jnp.swapaxes(jnp.conj(right_adjoint), -1, -2)
    basis = right_vectors[:, rank:]
    space = ArraySpace((dimension, dimension), dtype=policy.dtype)
    subspace = LinearSubspace(
        space,
        basis,
        dimension=nullity,
        orthonormal=True,
        subspace_id=canonical_fingerprint(
            {
                "kind": "algebra-derivation-subspace-v1",
                "plan": plan.plan_id,
                "capacity": nullity,
            }
        ),
    )

    leibniz_residual = _maximum_or_zero(matrix @ basis)
    unit = jnp.asarray(
        [
            numerator / denominator
            for numerator, denominator in constraint.algebra.unit.entries
        ],
        dtype=policy.dtype,
    )
    matrices = jnp.swapaxes(basis, 0, 1).reshape((nullity, dimension, dimension))
    unit_residual = _maximum_or_zero(ein.contract("nij,j->ni", matrices, unit))
    if nullity:
        commutators = ein.contract("aij,bjk->abik", matrices, matrices) - ein.contract(
            "bij,ajk->abik",
            matrices,
            matrices,
        )
        flattened = commutators.reshape((nullity, nullity, constraint.variable_count))
        projector = basis @ jnp.swapaxes(jnp.conj(basis), -1, -2)
        projected = ein.contract("...v,vw->...w", flattened, projector)
        closure_residual = _maximum_or_zero(flattened - projected)
    else:
        closure_residual = jnp.asarray(0.0, dtype=policy.dtype)

    tiny = jnp.finfo(singular_values.dtype).tiny
    retained_margin = (
        singular_values[rank - 1] / jnp.maximum(cutoff, tiny)
        if rank
        else jnp.asarray(jnp.inf, dtype=policy.dtype)
    )
    discarded_margin = (
        cutoff / jnp.maximum(singular_values[rank], tiny)
        if rank < constraint.variable_count
        else jnp.asarray(jnp.inf, dtype=policy.dtype)
    )
    singular_gap = jnp.minimum(retained_margin, discarded_margin)
    finite = (
        jnp.all(jnp.isfinite(singular_values))
        & jnp.isfinite(leibniz_residual)
        & jnp.isfinite(unit_residual)
        & jnp.isfinite(closure_residual)
    )
    gap_resolved = singular_gap >= policy.minimum_singular_gap
    residual_ok = (
        (leibniz_residual <= policy.residual_tolerance)
        & (unit_residual <= policy.residual_tolerance)
        & (closure_residual <= policy.residual_tolerance)
    )
    converged = finite & gap_resolved & residual_ok
    status = jnp.where(
        ~finite,
        int(AlgebraDerivationStatus.NONFINITE),
        jnp.where(
            ~gap_resolved,
            int(AlgebraDerivationStatus.AMBIGUOUS_RANK),
            jnp.where(
                ~residual_ok,
                int(AlgebraDerivationStatus.RESIDUAL_FAILURE),
                int(AlgebraDerivationStatus.SUCCESS),
            ),
        ),
    )
    basis_bytes = constraint.variable_count * nullity * itemsize
    return PreparedAlgebraDerivations(
        plan,
        subspace,
        singular_values,
        cutoff,
        singular_gap,
        leibniz_residual,
        unit_residual,
        closure_residual,
        converged,
        status,
        constraint_matrix_bytes=matrix_bytes,
        workspace_bytes=workspace_bytes,
        basis_bytes=basis_bytes,
    )


__all__ = [
    "AlgebraDerivationPlan",
    "AlgebraDerivationPolicy",
    "AlgebraDerivationStatus",
    "PreparedAlgebraDerivations",
    "plan_algebra_derivations",
    "prepare_algebra_derivations",
]
