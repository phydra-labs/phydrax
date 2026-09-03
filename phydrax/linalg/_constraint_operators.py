#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._constraints import ConstraintMap
from ._costs import _array_tree_storage_bytes
from ._factorizations import (
    FactorizationPolicy,
    factorize,
    PreparedFactorization,
    refresh_factorization,
)
from ._materialization import MaterializationPolicy, materialize
from ._operators import AbstractLinearOperator, DenseLinearOperator
from ._policies import RankPolicy, SolveResourcePolicy
from ._spaces import _coordinate_dtype, AbstractVectorSpace, ArraySpace, RHSLayout
from ._subspaces import LinearSubspace


ConstraintOperatorKind: TypeAlias = Literal["dense", "structured", "matrix-free"]
ConstraintFactorizationKind: TypeAlias = Literal["auto", "svd", "qr"]


class ConstraintOperatorEvidence(StrictModule, NonTrainableState):
    """Numerical rank, algebraic residual, and resource evidence for constraints."""

    singular_values: Array
    generalized_right_inverse_residual_norm: Array
    strict_right_inverse_residual_norm: Array
    nullspace_residual_norm: Array
    minimum_norm_residual_norm: Array
    compatibility_tolerance: Array
    residual_tolerance: Array
    numeric_version: Array
    rank: int = eqx.field(static=True)
    nullity: int = eqx.field(static=True)
    source_dimension: int = eqx.field(static=True)
    target_dimension: int = eqx.field(static=True)
    full_row_rank: bool = eqx.field(static=True)
    full_column_rank: bool = eqx.field(static=True)
    operator_kind: ConstraintOperatorKind = eqx.field(static=True)
    factorization_kind: Literal["svd", "qr"] = eqx.field(static=True)
    operator_matrix_bytes: int = eqx.field(static=True)
    factorization_bytes: int = eqx.field(static=True)
    right_inverse_bytes: int = eqx.field(static=True)
    nullspace_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    setup_matvec_count: int = eqx.field(static=True)
    factorization_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        singular_values: ArrayLike,
        generalized_right_inverse_residual_norm: ArrayLike,
        strict_right_inverse_residual_norm: ArrayLike,
        nullspace_residual_norm: ArrayLike,
        minimum_norm_residual_norm: ArrayLike,
        compatibility_tolerance: ArrayLike,
        residual_tolerance: ArrayLike,
        numeric_version: ArrayLike,
        rank: int,
        nullity: int,
        source_dimension: int,
        target_dimension: int,
        full_row_rank: bool,
        full_column_rank: bool,
        operator_kind: ConstraintOperatorKind,
        factorization_kind: Literal["svd", "qr"],
        operator_matrix_bytes: int,
        factorization_bytes: int,
        right_inverse_bytes: int,
        nullspace_bytes: int,
        preparation_workspace_bytes: int,
        setup_matvec_count: int,
        factorization_id: str,
        plan_id: str,
    ):
        singular_values_ = jnp.asarray(singular_values)
        scalars = tuple(
            jnp.asarray(value)
            for value in (
                generalized_right_inverse_residual_norm,
                strict_right_inverse_residual_norm,
                nullspace_residual_norm,
                minimum_norm_residual_norm,
                compatibility_tolerance,
                residual_tolerance,
                numeric_version,
            )
        )
        if singular_values_.ndim != 1 or any(value.shape != () for value in scalars):
            raise ValueError("Constraint evidence arrays have invalid shapes.")
        if not jnp.issubdtype(singular_values_.dtype, jnp.floating):
            raise TypeError("Constraint singular values must be real floating values.")
        if not bool(jnp.all(jnp.isfinite(singular_values_))) or any(
            not bool(jnp.isfinite(value)) for value in scalars
        ):
            raise ValueError("Constraint evidence arrays must be finite.")
        rank_ = int(rank)
        nullity_ = int(nullity)
        source_ = int(source_dimension)
        target_ = int(target_dimension)
        if source_ < 1 or target_ < 1:
            raise ValueError("Constraint dimensions must be positive.")
        if rank_ < 0 or rank_ > min(source_, target_):
            raise ValueError("Constraint rank is outside its dimensional bounds.")
        if nullity_ != source_ - rank_:
            raise ValueError("Constraint nullity must equal source dimension minus rank.")
        if operator_kind not in ("dense", "structured", "matrix-free"):
            raise ValueError("Unknown constraint operator kind.")
        if factorization_kind not in ("svd", "qr"):
            raise ValueError("Unknown constraint factorization kind.")
        resources = tuple(
            int(value)
            for value in (
                operator_matrix_bytes,
                factorization_bytes,
                right_inverse_bytes,
                nullspace_bytes,
                preparation_workspace_bytes,
                setup_matvec_count,
            )
        )
        if any(value < 0 for value in resources):
            raise ValueError("Constraint resource evidence must be non-negative.")
        factorization_id_ = str(factorization_id)
        plan_id_ = str(plan_id)
        if not factorization_id_ or not plan_id_:
            raise ValueError("Constraint evidence identifiers must be non-empty.")
        self.singular_values = singular_values_
        (
            self.generalized_right_inverse_residual_norm,
            self.strict_right_inverse_residual_norm,
            self.nullspace_residual_norm,
            self.minimum_norm_residual_norm,
            self.compatibility_tolerance,
            self.residual_tolerance,
            self.numeric_version,
        ) = scalars
        self.rank = rank_
        self.nullity = nullity_
        self.source_dimension = source_
        self.target_dimension = target_
        self.full_row_rank = bool(full_row_rank)
        self.full_column_rank = bool(full_column_rank)
        self.operator_kind = operator_kind
        self.factorization_kind = factorization_kind
        (
            self.operator_matrix_bytes,
            self.factorization_bytes,
            self.right_inverse_bytes,
            self.nullspace_bytes,
            self.preparation_workspace_bytes,
            self.setup_matvec_count,
        ) = resources
        self.factorization_id = factorization_id_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "constraint-operator-evidence-v1",
                "plan": plan_id_,
                "factorization": factorization_id_,
                "numeric_version": int(np.asarray(scalars[6])),
                "rank": rank_,
                "nullity": nullity_,
                "source_dimension": source_,
                "target_dimension": target_,
                "operator_kind": operator_kind,
                "factorization_kind": factorization_kind,
                "singular_values": array_tree_fingerprint(singular_values_),
                "generalized_residual": array_tree_fingerprint(scalars[0]),
                "strict_residual": array_tree_fingerprint(scalars[1]),
                "nullspace_residual": array_tree_fingerprint(scalars[2]),
                "minimum_norm_residual": array_tree_fingerprint(scalars[3]),
                "resources": list(resources),
            }
        )

    @property
    def retained_storage_bytes(self) -> int:
        return (
            self.operator_matrix_bytes
            + self.factorization_bytes
            + self.right_inverse_bytes
            + self.nullspace_bytes
        )


class ConstraintOperatorPlan(StrictModule, NonTrainableState):
    """Immutable preparation policy for one finite-dimensional constraint map."""

    operator: AbstractLinearOperator
    rank: RankPolicy
    resources: SolveResourcePolicy
    materialization: MaterializationPolicy
    require_full_row_rank: bool = eqx.field(static=True)
    factorization_kind: ConstraintFactorizationKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        require_full_row_rank: bool = True,
        rank: RankPolicy | None = None,
        resources: SolveResourcePolicy | None = None,
        materialization: MaterializationPolicy | None = None,
        factorization_kind: ConstraintFactorizationKind = "auto",
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape:
            raise ValueError(
                "Constraint operator preparation requires an unbatched operator."
            )
        if operator.source.size < 1 or operator.target.size < 1:
            raise ValueError("Constraint operator spaces must be non-empty.")
        if not isinstance(require_full_row_rank, bool):
            raise TypeError("require_full_row_rank must be boolean.")
        if require_full_row_rank and operator.target.size > operator.source.size:
            raise ValueError(
                "Full-row-rank constraints require target dimension no larger than source dimension."
            )
        rank_ = RankPolicy() if rank is None else rank
        resources_ = SolveResourcePolicy() if resources is None else resources
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        if not isinstance(rank_, RankPolicy):
            raise TypeError("rank must be a RankPolicy or None.")
        if not isinstance(resources_, SolveResourcePolicy):
            raise TypeError("resources must be a SolveResourcePolicy or None.")
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy or None.")
        if factorization_kind not in ("auto", "svd", "qr"):
            raise ValueError("factorization_kind must be 'auto', 'svd', or 'qr'.")
        if factorization_kind == "qr" and (
            not require_full_row_rank or operator.source.size != operator.target.size
        ):
            raise ValueError(
                "QR constraint preparation is restricted to strict square constraints; "
                "use SVD for rectangular or generalized constraints."
            )
        self.operator = operator
        self.rank = rank_
        self.resources = resources_
        self.materialization = materialization_
        self.require_full_row_rank = require_full_row_rank
        self.factorization_kind = factorization_kind
        self.plan_id = canonical_fingerprint(
            {
                "kind": "constraint-operator-plan-v1",
                "operator": operator.operator_id,
                "source": operator.source.space_id,
                "target": operator.target.space_id,
                "require_full_row_rank": require_full_row_rank,
                "factorization_kind": factorization_kind,
                "rank": {
                    "relative_cutoff": rank_.relative_cutoff,
                    "absolute_cutoff": rank_.absolute_cutoff,
                },
                "resources": {
                    "factorization_bytes": resources_.factorization_bytes,
                    "workspace_bytes": resources_.workspace_bytes,
                },
                "materialization": {
                    "max_entries": materialization_.max_entries,
                    "max_bytes": materialization_.max_bytes,
                },
            }
        )

    def prepare(self, /) -> "PreparedConstraintOperator":
        return _prepare_plan(self)


class PreparedConstraintOperator(StrictModule, NonTrainableState):
    """Reusable generalized inverse, compatibility projector, and nullspace chart."""

    plan: ConstraintOperatorPlan
    operator: AbstractLinearOperator
    factorization: PreparedFactorization
    right_inverse_operator: AbstractLinearOperator
    nullspace: LinearSubspace
    nullspace_operator: AbstractLinearOperator
    evidence: ConstraintOperatorEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConstraintOperatorPlan,
        operator: AbstractLinearOperator,
        factorization: PreparedFactorization,
        right_inverse_operator: AbstractLinearOperator,
        nullspace: LinearSubspace,
        nullspace_operator: AbstractLinearOperator,
        evidence: ConstraintOperatorEvidence,
        /,
    ):
        if not isinstance(plan, ConstraintOperatorPlan):
            raise TypeError("plan must be a ConstraintOperatorPlan.")
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if not isinstance(factorization, PreparedFactorization):
            raise TypeError("factorization must be a PreparedFactorization.")
        if not isinstance(right_inverse_operator, AbstractLinearOperator):
            raise TypeError("right_inverse_operator must be an AbstractLinearOperator.")
        if not isinstance(nullspace, LinearSubspace):
            raise TypeError("nullspace must be a LinearSubspace.")
        if not isinstance(nullspace_operator, AbstractLinearOperator):
            raise TypeError("nullspace_operator must be an AbstractLinearOperator.")
        if not isinstance(evidence, ConstraintOperatorEvidence):
            raise TypeError("evidence must be ConstraintOperatorEvidence.")
        if not operator.source.compatible(
            plan.operator.source
        ) or not operator.target.compatible(plan.operator.target):
            raise ValueError("Prepared constraint spaces must match their plan.")
        if not right_inverse_operator.source.compatible(operator.target) or not (
            right_inverse_operator.target.compatible(operator.source)
        ):
            raise ValueError(
                "The prepared right inverse must map target to source space."
            )
        if not nullspace.space.compatible(operator.source):
            raise ValueError("The prepared nullspace must belong to the operator source.")
        if not nullspace_operator.target.compatible(operator.source):
            raise ValueError("The nullspace operator must map into the operator source.")
        if nullspace_operator.source.size != evidence.nullity:
            raise ValueError(
                "The nullspace operator source dimension must equal nullity."
            )
        if evidence.factorization_id != factorization.factorization_id:
            raise ValueError("Constraint evidence and factorization identifiers differ.")
        self.plan = plan
        self.operator = operator
        self.factorization = factorization
        self.right_inverse_operator = right_inverse_operator
        self.nullspace = nullspace
        self.nullspace_operator = nullspace_operator
        self.evidence = evidence
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-constraint-operator-v1",
                "plan": plan.plan_id,
                "operator": operator.operator_id,
                "factorization": factorization.factorization_id,
                "evidence": evidence.evidence_id,
            }
        )

    @property
    def source_space(self) -> AbstractVectorSpace:
        return self.operator.source

    @property
    def target_space(self) -> AbstractVectorSpace:
        return self.operator.target

    @property
    def rank(self) -> int:
        return self.evidence.rank

    @property
    def nullity(self) -> int:
        return self.evidence.nullity

    @property
    def right_inverse(self) -> Array:
        if not isinstance(self.right_inverse_operator, DenseLinearOperator):
            raise TypeError(
                "Prepared right inverse is not represented in dense coordinates."
            )
        return self.right_inverse_operator.matrix

    @property
    def nullspace_basis(self) -> Array:
        return self.nullspace.basis[:, : self.nullity]

    def apply(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.operator.mv(self.source_space.validate(vector))

    def generalized_right_inverse(self, target: PyTree[Any], /) -> PyTree[Array]:
        """Apply the pairing-aware Moore-Penrose generalized right inverse."""
        return self.right_inverse_operator.mv(self.target_space.validate(target))

    def strict_right_inverse(self, target: PyTree[Any], /) -> PyTree[Array]:
        """Apply a true right inverse, rejecting nonsurjective constraint maps."""
        if not self.evidence.full_row_rank:
            raise ValueError(
                "Strict right inverse requires a full-row-rank constraint operator."
            )
        return self.generalized_right_inverse(target)

    def right_inverse_transpose(self, source_covector: PyTree[Any], /) -> PyTree[Array]:
        return self.right_inverse_operator.transpose_mv(source_covector)

    def right_inverse_adjoint(self, source_vector: PyTree[Any], /) -> PyTree[Array]:
        return self.right_inverse_operator.adjoint_mv(source_vector)

    def compatibility_residual(self, target: PyTree[Any], /) -> PyTree[Array]:
        value = self.target_space.validate(target)
        projected = self.apply(self.generalized_right_inverse(value))
        return self.target_space.validate(
            jax.tree.map(lambda left, right: left - right, value, projected)
        )

    def compatibility_defect(self, target: PyTree[Any], /) -> Array:
        residual = self.compatibility_residual(target)
        return jnp.sqrt(
            jnp.maximum(jnp.real(self.target_space.inner(residual, residual)), 0.0)
        )

    def is_compatible(
        self,
        target: PyTree[Any],
        /,
        *,
        absolute_tolerance: ArrayLike | None = None,
        relative_tolerance: ArrayLike | None = None,
    ) -> Array:
        value = self.target_space.validate(target)
        absolute = (
            self.evidence.compatibility_tolerance
            if absolute_tolerance is None
            else _nonnegative_scalar(absolute_tolerance, "absolute_tolerance")
        )
        relative = (
            self.evidence.compatibility_tolerance
            if relative_tolerance is None
            else _nonnegative_scalar(relative_tolerance, "relative_tolerance")
        )
        norm = jnp.sqrt(jnp.maximum(jnp.real(self.target_space.inner(value, value)), 0.0))
        return self.compatibility_defect(value) <= absolute + relative * norm

    def minimum_norm_lift(
        self,
        target: PyTree[Any],
        /,
        *,
        check_compatibility: bool = True,
    ) -> PyTree[Array]:
        """Return the source-pairing minimum-norm compatible lift."""
        if not isinstance(check_compatibility, bool):
            raise TypeError("check_compatibility must be boolean.")
        value = self.target_space.validate(target)
        lift = self.generalized_right_inverse(value)
        if not check_compatibility:
            return lift
        compatible = self.is_compatible(value)
        return jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                ~compatible,
                "Constraint target is incompatible with the operator range.",
            ),
            lift,
        )

    def nullspace_action(self, coordinates: ArrayLike, /) -> PyTree[Array]:
        return self.nullspace_operator.mv(coordinates)

    def nullspace_transpose(self, source_covector: PyTree[Any], /) -> Array:
        return self.nullspace_operator.transpose_mv(source_covector)

    def nullspace_adjoint(self, source_vector: PyTree[Any], /) -> Array:
        return self.nullspace_operator.adjoint_mv(source_vector)

    def project_nullspace(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.nullspace.project(self.source_space.validate(vector))

    def constraint_map(self, /, *, constraint_id: str | None = None) -> ConstraintMap:
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "prepared-nullspace-constraint-map",
                    "prepared": self.prepared_id,
                }
            )
            if constraint_id is None
            else constraint_id
        )
        return ConstraintMap(
            self.source_space,
            self.nullspace_operator.source,
            self.nullspace_operator,
            constraint_id=identifier,
        )


def prepare_constraint_operator(
    plan_or_operator: ConstraintOperatorPlan | AbstractLinearOperator,
    /,
    *,
    require_full_row_rank: bool = True,
    rank: RankPolicy | None = None,
    resources: SolveResourcePolicy | None = None,
    materialization: MaterializationPolicy | None = None,
    factorization_kind: ConstraintFactorizationKind = "auto",
) -> PreparedConstraintOperator:
    """Prepare one reusable finite-dimensional constraint operator."""
    if isinstance(plan_or_operator, ConstraintOperatorPlan):
        if (
            require_full_row_rank is not True
            or rank is not None
            or resources is not None
            or materialization is not None
            or factorization_kind != "auto"
        ):
            raise ValueError("Preparation policy keywords must be omitted with a plan.")
        plan = plan_or_operator
    elif isinstance(plan_or_operator, AbstractLinearOperator):
        plan = ConstraintOperatorPlan(
            plan_or_operator,
            require_full_row_rank=require_full_row_rank,
            rank=rank,
            resources=resources,
            materialization=materialization,
            factorization_kind=factorization_kind,
        )
    else:
        raise TypeError("Expected a ConstraintOperatorPlan or AbstractLinearOperator.")
    return _prepare_plan(plan)


def refresh_constraint_operator(
    prepared: PreparedConstraintOperator,
    operator: AbstractLinearOperator,
    /,
) -> PreparedConstraintOperator:
    """Refresh numerical values while preserving spaces, policy, and numerical rank."""
    if not isinstance(prepared, PreparedConstraintOperator):
        raise TypeError("prepared must be a PreparedConstraintOperator.")
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if operator.batch_shape:
        raise ValueError("Constraint operator refresh requires an unbatched operator.")
    if not operator.source.compatible(
        prepared.source_space
    ) or not operator.target.compatible(prepared.target_space):
        raise ValueError("Constraint refresh must preserve source and target spaces.")
    if operator.operator_id != prepared.operator.operator_id:
        raise ValueError(
            "Constraint refresh must preserve the structural operator identity."
        )
    dense_operator, matrix, operator_kind, setup_matvec_count = (
        _dense_constraint_operator(
            operator,
            prepared.plan.materialization,
        )
    )
    factorization = refresh_factorization(prepared.factorization, dense_operator)
    refreshed = _bind_prepared_constraint(
        prepared.plan,
        operator,
        dense_operator,
        matrix,
        operator_kind,
        setup_matvec_count,
        factorization,
    )
    if refreshed.rank != prepared.rank:
        raise ValueError(
            "Constraint numeric refresh changed numerical rank; create a new symbolic plan."
        )
    return refreshed


def _prepare_plan(plan: ConstraintOperatorPlan, /) -> PreparedConstraintOperator:
    dense_operator, matrix, operator_kind, setup_matvec_count = (
        _dense_constraint_operator(
            plan.operator,
            plan.materialization,
        )
    )
    selected_kind: Literal["svd", "qr"] = (
        "svd" if plan.factorization_kind == "auto" else plan.factorization_kind
    )
    factorization = factorize(
        dense_operator,
        FactorizationPolicy(
            selected_kind,
            rank=RankPolicy(
                relative_cutoff=plan.rank.relative_cutoff,
                absolute_cutoff=plan.rank.absolute_cutoff,
                require_full_rank=plan.require_full_row_rank,
            ),
            materialization=plan.materialization,
            resources=plan.resources,
        ),
    )
    return _bind_prepared_constraint(
        plan,
        plan.operator,
        dense_operator,
        matrix,
        operator_kind,
        setup_matvec_count,
        factorization,
    )


def _bind_prepared_constraint(
    plan: ConstraintOperatorPlan,
    operator: AbstractLinearOperator,
    dense_operator: DenseLinearOperator,
    matrix: Array,
    operator_kind: ConstraintOperatorKind,
    setup_matvec_count: int,
    factorization: PreparedFactorization,
    /,
) -> PreparedConstraintOperator:
    del dense_operator
    rows = operator.target.size
    columns = operator.source.size
    rank = int(np.asarray(factorization.rank()))
    if rank < 0:
        raise RuntimeError("Constraint factorization could not determine numerical rank.")
    full_row_rank = rank == rows
    if plan.require_full_row_rank and not full_row_rank:
        raise ValueError(
            f"Constraint operator has numerical rank {rank}, but full row rank {rows} is required."
        )
    identity = jnp.eye(rows, dtype=_coordinate_dtype(operator.target))
    target_block = _unflatten_coordinate_block(operator.target, identity)
    solve_result = factorization.solve(target_block, rhs_layout=RHSLayout((rows,)))
    right_matrix = _flatten_vector_block(operator.source, solve_result.value, rows)
    if right_matrix.shape != (columns, rows):
        raise RuntimeError(
            "Constraint right-inverse block solve returned an invalid shape."
        )
    if not bool(jnp.all(jnp.isfinite(right_matrix))):
        raise RuntimeError(
            "Constraint right-inverse block solve returned nonfinite data."
        )
    right_operator = DenseLinearOperator(
        right_matrix,
        source=operator.target,
        target=operator.source,
        operator_id=canonical_fingerprint(
            {
                "kind": "constraint-generalized-right-inverse",
                "plan": plan.plan_id,
                "numeric_version": int(
                    np.asarray(factorization.prepared_solve.numeric_version)
                ),
            }
        ),
    )
    nullity = columns - rank
    if factorization.capabilities.nullspaces:
        nullspace = factorization.right_nullspace()
        reported_nullity = int(np.asarray(nullspace.dimension))
        if reported_nullity != nullity:
            raise RuntimeError("Constraint factorization reported inconsistent nullity.")
        nullspace_basis = nullspace.basis[:, :nullity]
    else:
        if nullity != 0:
            raise RuntimeError(
                "Selected factorization cannot expose a nontrivial nullspace."
            )
        nullspace_basis = jnp.zeros(
            (columns, 0), dtype=_coordinate_dtype(operator.source)
        )
        nullspace = LinearSubspace(
            operator.source,
            nullspace_basis,
            dimension=0,
            orthonormal=True,
        )
    reduced_space = ArraySpace(
        (nullity,),
        dtype=_coordinate_dtype(operator.source),
        space_id=canonical_fingerprint(
            {
                "kind": "constraint-nullspace-coordinate-space",
                "plan": plan.plan_id,
                "nullity": nullity,
            }
        ),
    )
    nullspace_operator = DenseLinearOperator(
        nullspace_basis,
        source=reduced_space,
        target=operator.source,
        operator_id=canonical_fingerprint(
            {
                "kind": "constraint-nullspace-operator",
                "plan": plan.plan_id,
                "numeric_version": int(
                    np.asarray(factorization.prepared_solve.numeric_version)
                ),
                "nullity": nullity,
            }
        ),
    )
    target_projector = oe.contract("ij,jk->ik", matrix, right_matrix)
    generalized_residual = oe.contract("ij,jk->ik", target_projector, matrix) - matrix
    strict_residual = target_projector - jnp.eye(rows, dtype=matrix.dtype)
    nullspace_residual = oe.contract("ij,jk->ik", matrix, nullspace_basis)
    minimum_norm_residual = _minimum_norm_residual(
        operator.source, nullspace_basis, right_matrix
    )
    epsilon = jnp.finfo(matrix.real.dtype).eps
    dimension_scale = max(rows, columns, 1)
    matrix_norm = jnp.linalg.norm(matrix)
    right_norm = jnp.linalg.norm(right_matrix)
    residual_tolerance = (
        1024.0
        * epsilon
        * dimension_scale
        * jnp.maximum(matrix_norm * jnp.maximum(right_norm, 1.0), 1.0)
    )
    compatibility_tolerance = (
        1024.0
        * epsilon
        * dimension_scale
        * jnp.maximum(jnp.linalg.norm(target_projector), 1.0)
    )
    generalized_norm = jnp.linalg.norm(generalized_residual)
    strict_norm = jnp.linalg.norm(strict_residual)
    nullspace_norm = jnp.linalg.norm(nullspace_residual)
    minimum_norm = jnp.linalg.norm(minimum_norm_residual)
    if not bool(generalized_norm <= residual_tolerance):
        raise RuntimeError(
            "Generalized constraint right inverse failed its residual check."
        )
    if full_row_rank and not bool(strict_norm <= residual_tolerance):
        raise RuntimeError("Strict constraint right inverse failed its residual check.")
    if not bool(nullspace_norm <= residual_tolerance):
        raise RuntimeError("Constraint nullspace failed its residual check.")
    if not bool(minimum_norm <= residual_tolerance):
        raise RuntimeError("Constraint lift failed its pairing minimum-norm check.")
    singular_values = (
        factorization.singular_values()
        if factorization.capabilities.singular_values
        else jnp.empty((0,), dtype=matrix.real.dtype)
    )
    evidence = ConstraintOperatorEvidence(
        singular_values=singular_values,
        generalized_right_inverse_residual_norm=generalized_norm,
        strict_right_inverse_residual_norm=strict_norm,
        nullspace_residual_norm=nullspace_norm,
        minimum_norm_residual_norm=minimum_norm,
        compatibility_tolerance=compatibility_tolerance,
        residual_tolerance=residual_tolerance,
        numeric_version=factorization.prepared_solve.numeric_version,
        rank=rank,
        nullity=nullity,
        source_dimension=columns,
        target_dimension=rows,
        full_row_rank=full_row_rank,
        full_column_rank=rank == columns,
        operator_kind=operator_kind,
        factorization_kind=factorization.policy.kind,
        operator_matrix_bytes=int(matrix.nbytes),
        factorization_bytes=_array_tree_storage_bytes(factorization.prepared_solve.state),
        right_inverse_bytes=int(right_matrix.nbytes),
        nullspace_bytes=int(nullspace_basis.nbytes),
        preparation_workspace_bytes=max(
            int(matrix.nbytes),
            int(right_matrix.nbytes),
            int(nullspace.basis.nbytes),
        ),
        setup_matvec_count=setup_matvec_count,
        factorization_id=factorization.factorization_id,
        plan_id=plan.plan_id,
    )
    return PreparedConstraintOperator(
        plan,
        operator,
        factorization,
        right_operator,
        nullspace,
        nullspace_operator,
        evidence,
    )


def _dense_constraint_operator(
    operator: AbstractLinearOperator,
    policy: MaterializationPolicy,
    /,
) -> tuple[DenseLinearOperator, Array, ConstraintOperatorKind, int]:
    entries = operator.source.size * operator.target.size
    coordinate_dtype = _coordinate_dtype(operator.target)
    required_bytes = entries * coordinate_dtype.itemsize
    if entries > policy.max_entries or required_bytes > policy.max_bytes:
        raise ValueError(
            "Constraint operator dense preparation exceeds its materialization policy."
        )
    if isinstance(operator, DenseLinearOperator):
        matrix = operator.matrix
        operator_kind: ConstraintOperatorKind = "dense"
        setup_matvec_count = 0
    elif operator.capabilities.materialize:
        matrix = materialize(operator, policy)
        operator_kind = "structured"
        setup_matvec_count = 0
    else:
        identity = jnp.eye(operator.source.size, dtype=_coordinate_dtype(operator.source))
        matrix = jnp.asarray(operator.mv_block(identity))
        expected = (operator.target.size, operator.source.size)
        if matrix.shape != expected or matrix.dtype != coordinate_dtype:
            raise ValueError(
                "Matrix-free constraint block action must return canonical coordinates "
                f"with shape/dtype {expected} and {coordinate_dtype}."
            )
        operator_kind = "matrix-free"
        setup_matvec_count = operator.source.size
    if matrix.ndim != 2:
        raise ValueError("Constraint operator preparation requires one unbatched matrix.")
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError("Constraint operator contains nonfinite entries.")
    dense = DenseLinearOperator(
        matrix,
        source=operator.source,
        target=operator.target,
        properties=operator.properties,
        operator_id=operator.operator_id,
    )
    return dense, matrix, operator_kind, setup_matvec_count


def _unflatten_coordinate_block(
    space: AbstractVectorSpace, matrix: Array, /
) -> PyTree[Array]:
    return jax.vmap(space.unflatten, in_axes=1, out_axes=-1)(matrix)


def _flatten_vector_block(
    space: AbstractVectorSpace,
    values: PyTree[Any],
    block_size: int,
    /,
) -> Array:
    coordinates = jax.vmap(space.flatten, in_axes=-1, out_axes=1)(values)
    if coordinates.shape != (space.size, block_size):
        raise ValueError("Vector block does not match its declared coordinate space.")
    return coordinates


def _minimum_norm_residual(
    space: AbstractVectorSpace,
    nullspace_basis: Array,
    right_inverse: Array,
    /,
) -> Array:
    if nullspace_basis.shape[1] == 0:
        return jnp.zeros((0, right_inverse.shape[1]), dtype=right_inverse.dtype)

    def pair(column: Array, lift: Array) -> Array:
        return space.inner(space.unflatten(column), space.unflatten(lift))

    return jax.vmap(
        lambda column: jax.vmap(lambda lift: pair(column, lift), in_axes=1)(
            right_inverse
        ),
        in_axes=1,
    )(nullspace_basis)


def _nonnegative_scalar(value: ArrayLike, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"{name} must be scalar.")
    if not jnp.issubdtype(scalar.dtype, jnp.floating):
        raise TypeError(f"{name} must be real floating.")
    return eqx.error_if(
        scalar,
        ~jnp.isfinite(scalar) | (scalar < 0.0),
        f"{name} must be finite and non-negative.",
    )


__all__ = [
    "ConstraintFactorizationKind",
    "ConstraintOperatorEvidence",
    "ConstraintOperatorKind",
    "ConstraintOperatorPlan",
    "PreparedConstraintOperator",
    "prepare_constraint_operator",
    "refresh_constraint_operator",
]
