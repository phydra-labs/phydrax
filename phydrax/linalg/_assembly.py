#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, cast, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._materialization import MaterializationPolicy, materialize
from ._operators import (
    _assemble_operator_diagonal,
    AbstractLinearOperator,
    AdjointLinearOperator,
    ComposedLinearOperator,
    DenseLinearOperator,
    DiagonalLinearOperator,
    IdentityLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    TransposeLinearOperator,
)
from ._properties import LinearCapabilityError, OperatorProperties
from ._spaces import (
    _coordinate_dtype,
    _coordinate_pairing_weights,
    _has_diagonal_pairing,
    AbstractVectorSpace,
)
from ._sparse_contract import AbstractSparseLinearOperator
from ._structured_operators import (
    BandedLinearOperator,
    BlockDiagonalLinearOperator,
    KroneckerLinearOperator,
    KroneckerSumLinearOperator,
    LocalBlockDiagonalLinearOperator,
    PermutationLinearOperator,
    TriangularLinearOperator,
    TridiagonalLinearOperator,
)


def assemble_diagonal(
    operator: AbstractLinearOperator,
    /,
    *,
    materialization: MaterializationPolicy | None = None,
) -> Array:
    """Assemble an exact canonical-coordinate diagonal without implicit densification."""
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not operator.source.compatible(operator.target):
        raise ValueError("Diagonal assembly requires an endomorphism.")
    if materialization is not None and not isinstance(
        materialization, MaterializationPolicy
    ):
        raise TypeError("materialization must be a MaterializationPolicy or None.")

    if operator.capabilities.diagonal_assembly:
        expected = operator.batch_shape + (operator.source.size,)
        abstract = jax.eval_shape(lambda: _assemble_operator_diagonal(operator))
        if not isinstance(abstract, jax.ShapeDtypeStruct):
            raise TypeError("Diagonal assembly must return one array.")
        if abstract.shape != expected:
            raise ValueError(
                f"Diagonal assembly must have shape {expected}; got {abstract.shape}."
            )
        diagonal = jnp.asarray(_assemble_operator_diagonal(operator))
        if diagonal.shape != expected or diagonal.dtype != abstract.dtype:
            raise ValueError(
                "Diagonal assembly changed shape or dtype between abstract "
                "evaluation and execution."
            )
        return diagonal

    if materialization is None:
        raise LinearCapabilityError(
            "Operator does not support exact diagonal assembly; an explicit "
            "materialization policy is required for dense fallback."
        )
    matrix = materialize(operator, materialization)
    return jnp.diagonal(matrix, axis1=-2, axis2=-1)


def assemble_uniform_blocks(
    operator: AbstractLinearOperator,
    block_size: int,
    /,
    *,
    policy: SparseAssemblyPolicy | None = None,
) -> Array:
    """Assemble exact equal-sized canonical diagonal blocks without densification."""
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("Uniform block assembly requires an unbatched endomorphism.")
    size = int(block_size)
    if size < 1:
        raise ValueError("block_size must be positive.")
    dimension = operator.source.size
    if dimension % size:
        raise ValueError("block_size must divide the operator dimension.")
    policy_ = SparseAssemblyPolicy() if policy is None else policy
    if not isinstance(policy_, SparseAssemblyPolicy):
        raise TypeError("policy must be a SparseAssemblyPolicy or None.")
    num_blocks = dimension // size

    if isinstance(operator, DenseLinearOperator):
        grouped = jnp.arange(dimension, dtype=jnp.int32).reshape((num_blocks, size))
        return operator.matrix[
            grouped[:, :, None],
            grouped[:, None, :],
        ]
    if (
        isinstance(operator, LocalBlockDiagonalLinearOperator)
        and operator.num_blocks == num_blocks
        and operator.input_block_size == size
        and operator.output_block_size == size
    ):
        return operator.blocks

    sparse = assemble_sparse(operator, policy_)
    storage = sparse.sparse_storage()
    rows = jnp.repeat(
        jnp.arange(dimension, dtype=storage.indptr.dtype),
        jnp.diff(storage.indptr),
        total_repeat_length=storage.values.size,
    )
    columns = storage.indices
    same_block = rows // size == columns // size
    block_indices = rows // size
    local_rows = rows % size
    local_columns = columns % size
    contributions = jnp.where(
        same_block,
        storage.values,
        jnp.zeros((), dtype=storage.values.dtype),
    )
    blocks = jnp.zeros(
        (num_blocks, size, size),
        dtype=storage.values.dtype,
    )
    return blocks.at[block_indices, local_rows, local_columns].add(contributions)


SparseAssemblyKind: TypeAlias = Literal[
    "sparse",
    "identity",
    "diagonal",
    "permutation",
    "triangular",
    "tridiagonal",
    "banded",
    "local-block",
    "materialized",
    "scale",
    "sum",
    "composition",
    "transpose",
    "adjoint",
    "block-diagonal",
    "kronecker",
    "kronecker-sum",
]


class SparseAssemblyPolicy(StrictModule):
    """Explicit output and symbolic-workspace limits for exact sparse assembly."""

    max_nnz: int = eqx.field(static=True)
    max_bytes: int = eqx.field(static=True)
    max_contributions: int = eqx.field(static=True)
    max_workspace_bytes: int = eqx.field(static=True)
    materialization: MaterializationPolicy | None

    def __init__(
        self,
        *,
        max_nnz: int = 1_000_000,
        max_bytes: int = 128 * 1024 * 1024,
        max_contributions: int = 4_000_000,
        max_workspace_bytes: int = 256 * 1024 * 1024,
        materialization: MaterializationPolicy | None = None,
    ):
        limits = {
            "max_nnz": int(max_nnz),
            "max_bytes": int(max_bytes),
            "max_contributions": int(max_contributions),
            "max_workspace_bytes": int(max_workspace_bytes),
        }
        if any(value < 1 for value in limits.values()):
            raise ValueError("Sparse assembly limits must be positive.")
        if materialization is not None and not isinstance(
            materialization,
            MaterializationPolicy,
        ):
            raise TypeError("materialization must be a MaterializationPolicy or None.")
        self.max_nnz = limits["max_nnz"]
        self.max_bytes = limits["max_bytes"]
        self.max_contributions = limits["max_contributions"]
        self.max_workspace_bytes = limits["max_workspace_bytes"]
        self.materialization = materialization


class SparseAssemblyCostEstimate(StrictModule):
    """Static output and workspace estimate for one sparse assembly plan."""

    result_nnz: int = eqx.field(static=True)
    maximum_intermediate_nnz: int = eqx.field(static=True)
    maximum_contributions: int = eqx.field(static=True)
    output_bytes: int = eqx.field(static=True)
    recipe_bytes: int = eqx.field(static=True)
    symbolic_workspace_bytes: int = eqx.field(static=True)
    numeric_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        result_nnz: int,
        maximum_intermediate_nnz: int,
        maximum_contributions: int,
        output_bytes: int,
        recipe_bytes: int,
        symbolic_workspace_bytes: int,
        numeric_workspace_bytes: int,
    ):
        values = tuple(
            int(value)
            for value in (
                result_nnz,
                maximum_intermediate_nnz,
                maximum_contributions,
                output_bytes,
                recipe_bytes,
                symbolic_workspace_bytes,
                numeric_workspace_bytes,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("Sparse assembly cost estimates must be non-negative.")
        (
            self.result_nnz,
            self.maximum_intermediate_nnz,
            self.maximum_contributions,
            self.output_bytes,
            self.recipe_bytes,
            self.symbolic_workspace_bytes,
            self.numeric_workspace_bytes,
        ) = values


class _SparseAssemblyRecipe(StrictModule):
    rows: Array
    columns: Array
    children: tuple["_SparseAssemblyRecipe", ...]
    input_indices: tuple[Array, ...]
    output_indices: tuple[Array, ...]
    kind: SparseAssemblyKind = eqx.field(static=True)
    operator_type: type = eqx.field(static=True)
    shape: tuple[int, int] = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    target_space_id: str = eqx.field(static=True)
    payload: tuple[Any, ...] = eqx.field(static=True)
    contribution_count: int = eqx.field(static=True)
    symbolic_workspace_bytes: int = eqx.field(static=True)
    numeric_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        kind: SparseAssemblyKind,
        operator: AbstractLinearOperator,
        rows: np.ndarray,
        columns: np.ndarray,
        children: tuple["_SparseAssemblyRecipe", ...] = (),
        input_indices: tuple[np.ndarray, ...] = (),
        output_indices: tuple[np.ndarray, ...] = (),
        payload: tuple[Any, ...] = (),
        contribution_count: int,
        symbolic_workspace_bytes: int,
        numeric_workspace_bytes: int,
    ):
        self.rows = jnp.asarray(rows, dtype=jnp.int32)
        self.columns = jnp.asarray(columns, dtype=jnp.int32)
        self.children = children
        self.input_indices = tuple(
            jnp.asarray(indices, dtype=jnp.int32) for indices in input_indices
        )
        self.output_indices = tuple(
            jnp.asarray(indices, dtype=jnp.int32) for indices in output_indices
        )
        self.kind = kind
        self.operator_type = type(operator)
        self.shape = (operator.target.size, operator.source.size)
        self.source_space_id = operator.source.space_id
        self.target_space_id = operator.target.space_id
        self.payload = payload
        self.contribution_count = int(contribution_count)
        self.symbolic_workspace_bytes = int(symbolic_workspace_bytes)
        self.numeric_workspace_bytes = int(numeric_workspace_bytes)


class SparseAssemblyPlan(StrictModule):
    """Immutable symbolic sparse pattern and numerical assembly recipe."""

    policy: SparseAssemblyPolicy
    source: AbstractVectorSpace
    target: AbstractVectorSpace
    properties: OperatorProperties
    cost: SparseAssemblyCostEstimate
    _recipe: _SparseAssemblyRecipe
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        policy: SparseAssemblyPolicy,
        recipe: _SparseAssemblyRecipe,
        cost: SparseAssemblyCostEstimate,
        /,
        *,
        plan_id: str,
    ):
        self.policy = policy
        self.source = operator.source
        self.target = operator.target
        self.properties = operator.properties
        self.cost = cost
        self._recipe = recipe
        self.plan_id = str(plan_id)

    @property
    def shape(self) -> tuple[int, int]:
        return self._recipe.shape

    @property
    def nnz(self) -> int:
        return int(self._recipe.rows.size)

    @property
    def row_indices(self) -> Array:
        return self._recipe.rows

    @property
    def column_indices(self) -> Array:
        return self._recipe.columns

    @property
    def uses_materialization(self) -> bool:
        return any(node.kind == "materialized" for node in _walk_recipes(self._recipe))


class PreparedSparseAssembly(StrictModule):
    """One evaluated sparse operator retaining its reusable symbolic plan."""

    plan: SparseAssemblyPlan
    operator: AbstractSparseLinearOperator
    numeric_version: Array

    def __init__(
        self,
        plan: SparseAssemblyPlan,
        operator: AbstractSparseLinearOperator,
        /,
        *,
        numeric_version: Any,
    ):
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.ndim != 0:
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version,
            version < 0,
            "numeric_version must be non-negative.",
        )
        self.plan = plan
        self.operator = operator
        self.numeric_version = version


def plan_sparse_assembly(
    operator: AbstractLinearOperator,
    policy: SparseAssemblyPolicy | None = None,
    /,
) -> SparseAssemblyPlan:
    """Plan exact canonical sparse assembly without evaluating numerical values."""
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if operator.batch_shape:
        raise ValueError("Sparse assembly currently requires an unbatched operator.")
    policy_ = SparseAssemblyPolicy() if policy is None else policy
    if not isinstance(policy_, SparseAssemblyPolicy):
        raise TypeError("policy must be a SparseAssemblyPolicy or None.")

    recipe = _plan_sparse_recipe(operator, policy_)
    recipe_bytes = _array_storage_bytes(recipe)
    maximum_intermediate_nnz = max(int(node.rows.size) for node in _walk_recipes(recipe))
    maximum_contributions = max(node.contribution_count for node in _walk_recipes(recipe))
    symbolic_workspace_bytes = max(
        node.symbolic_workspace_bytes for node in _walk_recipes(recipe)
    )
    numeric_workspace_bytes = max(
        node.numeric_workspace_bytes for node in _walk_recipes(recipe)
    )
    if recipe_bytes > policy_.max_workspace_bytes:
        raise LinearCapabilityError(
            f"Sparse assembly recipe requires {recipe_bytes} bytes, exceeding "
            f"the workspace limit {policy_.max_workspace_bytes}."
        )
    output_bytes = _sparse_output_bytes(
        recipe.shape,
        int(recipe.rows.size),
        _coordinate_dtype(operator.target).itemsize,
    )
    cost = SparseAssemblyCostEstimate(
        result_nnz=int(recipe.rows.size),
        maximum_intermediate_nnz=maximum_intermediate_nnz,
        maximum_contributions=maximum_contributions,
        output_bytes=output_bytes,
        recipe_bytes=recipe_bytes,
        symbolic_workspace_bytes=symbolic_workspace_bytes,
        numeric_workspace_bytes=numeric_workspace_bytes,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "sparse-assembly-plan",
            "source": operator.source.space_id,
            "target": operator.target.space_id,
            "recipe": _recipe_signature(recipe),
            "arrays": array_tree_fingerprint(recipe),
        }
    )
    return SparseAssemblyPlan(
        operator,
        policy_,
        recipe,
        cost,
        plan_id=plan_id,
    )


def prepare_sparse_assembly(
    plan: SparseAssemblyPlan,
    operator: AbstractLinearOperator,
    /,
) -> PreparedSparseAssembly:
    """Bind current operator coefficients to one symbolic sparse assembly plan."""
    if not isinstance(plan, SparseAssemblyPlan):
        raise TypeError("plan must be a SparseAssemblyPlan.")
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    return _prepare_sparse_assembly(plan, operator, numeric_version=0)


def refresh_sparse_assembly(
    prepared: PreparedSparseAssembly,
    operator: AbstractLinearOperator,
    /,
) -> PreparedSparseAssembly:
    """Refresh numerical values and reject every symbolic structure change."""
    if not isinstance(prepared, PreparedSparseAssembly):
        raise TypeError("prepared must be a PreparedSparseAssembly.")
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not prepared.plan.source.compatible(operator.source) or not (
        prepared.plan.target.compatible(operator.target)
    ):
        raise ValueError("Sparse assembly refresh changed source or target space.")
    return _prepare_sparse_assembly(
        prepared.plan,
        operator,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def assemble_sparse(
    operator: AbstractLinearOperator,
    policy: SparseAssemblyPolicy | None = None,
    /,
) -> AbstractSparseLinearOperator:
    """Plan and evaluate one exact canonical sparse operator."""
    plan = plan_sparse_assembly(operator, policy)
    return prepare_sparse_assembly(plan, operator).operator


def _prepare_sparse_assembly(
    plan: SparseAssemblyPlan,
    operator: AbstractLinearOperator,
    /,
    *,
    numeric_version: Any,
) -> PreparedSparseAssembly:
    from ..sparse import EdgeRelation, SparseCoordinateOperator

    values = _evaluate_sparse_recipe(
        plan._recipe,
        operator,
        plan.policy,
    )
    if values.shape != (plan.nnz,):
        raise ValueError(
            f"Sparse assembly recipe returned shape {values.shape}; "
            f"expected {(plan.nnz,)}."
        )
    relation = EdgeRelation(
        plan.column_indices,
        plan.row_indices,
        source_size=plan.source.size,
        target_size=plan.target.size,
    )
    assembled = SparseCoordinateOperator(
        relation,
        values,
        source=plan.source,
        target=plan.target,
        properties=plan.properties,
        operator_id=f"{plan.plan_id}:operator",
    )
    return PreparedSparseAssembly(
        plan,
        assembled,
        numeric_version=numeric_version,
    )


def _plan_sparse_recipe(
    operator: AbstractLinearOperator,
    policy: SparseAssemblyPolicy,
    /,
) -> _SparseAssemblyRecipe:
    if operator.batch_shape:
        raise LinearCapabilityError(
            "Sparse assembly recipes do not support operator batches."
        )
    if isinstance(operator, AbstractSparseLinearOperator):
        storage = operator.sparse_storage()
        if not storage.canonical or not storage.sorted_indices:
            raise LinearCapabilityError(
                "Sparse assembly requires canonical, row-sorted sparse leaves."
            )
        rows = np.repeat(
            np.arange(storage.shape[0], dtype=np.int64),
            np.diff(np.asarray(storage.indptr, dtype=np.int64)),
        )
        columns = np.asarray(storage.indices, dtype=np.int64)
        canonical_rows, canonical_columns, mapping = _canonical_pattern(
            rows,
            columns,
            storage.shape,
            policy,
        )
        if canonical_rows.size != rows.size or not np.array_equal(
            mapping, np.arange(rows.size)
        ):
            raise LinearCapabilityError(
                "Sparse leaf storage is not canonical despite its declared contract."
            )
        return _make_sparse_recipe(
            "sparse",
            operator,
            canonical_rows,
            canonical_columns,
            policy,
            contribution_count=rows.size,
        )
    if isinstance(operator, IdentityLinearOperator):
        indices = np.arange(operator.source.size, dtype=np.int64)
        return _make_sparse_recipe(
            "identity",
            operator,
            indices,
            indices,
            policy,
            contribution_count=indices.size,
        )
    if isinstance(operator, DiagonalLinearOperator):
        indices = np.arange(operator.source.size, dtype=np.int64)
        return _make_sparse_recipe(
            "diagonal",
            operator,
            indices,
            indices,
            policy,
            contribution_count=indices.size,
        )
    if isinstance(operator, PermutationLinearOperator):
        rows = np.arange(operator.source.size, dtype=np.int64)
        columns = np.asarray(operator.permutation, dtype=np.int64)
        rows, columns, _ = _canonical_pattern(
            rows,
            columns,
            (operator.target.size, operator.source.size),
            policy,
        )
        return _make_sparse_recipe(
            "permutation",
            operator,
            rows,
            columns,
            policy,
            contribution_count=rows.size,
        )
    if isinstance(operator, TriangularLinearOperator):
        size = operator.source.size
        rows, columns = np.tril_indices(size) if operator.lower else np.triu_indices(size)
        rows, columns, _ = _canonical_pattern(
            rows,
            columns,
            (size, size),
            policy,
        )
        return _make_sparse_recipe(
            "triangular",
            operator,
            rows,
            columns,
            policy,
            payload=(operator.lower, operator.unit_diagonal),
            contribution_count=rows.size,
        )
    if isinstance(operator, TridiagonalLinearOperator):
        size = operator.source.size
        diagonal = np.arange(size, dtype=np.int64)
        rows = np.concatenate((diagonal, np.arange(1, size), np.arange(size - 1)))
        columns = np.concatenate((diagonal, np.arange(size - 1), np.arange(1, size)))
        rows, columns, _ = _canonical_pattern(
            rows,
            columns,
            (size, size),
            policy,
        )
        return _make_sparse_recipe(
            "tridiagonal",
            operator,
            rows,
            columns,
            policy,
            contribution_count=rows.size,
        )
    if isinstance(operator, BandedLinearOperator):
        row_parts = []
        column_parts = []
        size = operator.source.size
        for offset in range(
            -operator.upper_bandwidth,
            operator.lower_bandwidth + 1,
        ):
            column_start = max(0, -offset)
            column_stop = min(size, size - offset)
            columns = np.arange(column_start, column_stop, dtype=np.int64)
            row_parts.append(columns + offset)
            column_parts.append(columns)
        rows = _concatenate_indices(row_parts)
        columns = _concatenate_indices(column_parts)
        rows, columns, _ = _canonical_pattern(
            rows,
            columns,
            (size, size),
            policy,
        )
        return _make_sparse_recipe(
            "banded",
            operator,
            rows,
            columns,
            policy,
            payload=(
                operator.lower_bandwidth,
                operator.upper_bandwidth,
            ),
            contribution_count=rows.size,
        )
    if isinstance(operator, LocalBlockDiagonalLinearOperator):
        local_rows = np.repeat(
            np.arange(operator.output_block_size, dtype=np.int64),
            operator.input_block_size,
        )
        local_columns = np.tile(
            np.arange(operator.input_block_size, dtype=np.int64),
            operator.output_block_size,
        )
        rows = np.concatenate(
            [
                local_rows + block * operator.output_block_size
                for block in range(operator.num_blocks)
            ]
        )
        columns = np.concatenate(
            [
                local_columns + block * operator.input_block_size
                for block in range(operator.num_blocks)
            ]
        )
        return _make_sparse_recipe(
            "local-block",
            operator,
            rows,
            columns,
            policy,
            payload=(
                operator.num_blocks,
                operator.input_block_size,
                operator.output_block_size,
            ),
            contribution_count=rows.size,
        )
    if isinstance(operator, ScaledLinearOperator):
        child = _plan_sparse_recipe(operator.operator, policy)
        return _make_sparse_recipe(
            "scale",
            operator,
            np.asarray(child.rows),
            np.asarray(child.columns),
            policy,
            children=(child,),
            contribution_count=int(child.rows.size),
        )
    if isinstance(operator, SumLinearOperator):
        children = (
            _plan_sparse_recipe(operator.left, policy),
            _plan_sparse_recipe(operator.right, policy),
        )
        rows = np.concatenate(
            tuple(np.asarray(child.rows, dtype=np.int64) for child in children)
        )
        columns = np.concatenate(
            tuple(np.asarray(child.columns, dtype=np.int64) for child in children)
        )
        result_rows, result_columns, mapping = _canonical_pattern(
            rows,
            columns,
            (operator.target.size, operator.source.size),
            policy,
        )
        split = int(children[0].rows.size)
        return _make_sparse_recipe(
            "sum",
            operator,
            result_rows,
            result_columns,
            policy,
            children=children,
            output_indices=(mapping[:split], mapping[split:]),
            contribution_count=rows.size,
        )
    if isinstance(operator, ComposedLinearOperator):
        return _plan_sparse_composition(operator, policy)
    if isinstance(operator, TransposeLinearOperator):
        child = _plan_sparse_recipe(operator.operator, policy)
        rows, columns, mapping = _canonical_pattern(
            np.asarray(child.columns),
            np.asarray(child.rows),
            (operator.target.size, operator.source.size),
            policy,
        )
        return _make_sparse_recipe(
            "transpose",
            operator,
            rows,
            columns,
            policy,
            children=(child,),
            output_indices=(mapping,),
            contribution_count=int(child.rows.size),
        )
    if isinstance(operator, AdjointLinearOperator):
        if not (
            _has_diagonal_pairing(operator.operator.source)
            and _has_diagonal_pairing(operator.operator.target)
        ):
            return _materialized_sparse_recipe(operator, policy)
        child = _plan_sparse_recipe(operator.operator, policy)
        rows, columns, mapping = _canonical_pattern(
            np.asarray(child.columns),
            np.asarray(child.rows),
            (operator.target.size, operator.source.size),
            policy,
        )
        return _make_sparse_recipe(
            "adjoint",
            operator,
            rows,
            columns,
            policy,
            children=(child,),
            output_indices=(mapping,),
            contribution_count=int(child.rows.size),
        )
    if isinstance(operator, BlockDiagonalLinearOperator):
        return _plan_sparse_block_diagonal(operator, policy)
    if isinstance(operator, KroneckerLinearOperator):
        return _plan_sparse_kronecker(operator, policy)
    if isinstance(operator, KroneckerSumLinearOperator):
        return _plan_sparse_kronecker_sum(operator, policy)
    return _materialized_sparse_recipe(operator, policy)


def _materialized_sparse_recipe(
    operator: AbstractLinearOperator,
    policy: SparseAssemblyPolicy,
    /,
) -> _SparseAssemblyRecipe:
    materialization = policy.materialization
    if materialization is None:
        raise LinearCapabilityError(
            f"{type(operator).__name__} has no exact sparse assembly recipe; "
            "an explicit materialization policy is required for a structurally "
            "dense fallback."
        )
    if not operator.capabilities.materialize:
        raise LinearCapabilityError(
            f"{type(operator).__name__} declares no materialization capability."
        )
    rows_count, columns_count = operator.target.size, operator.source.size
    entries = rows_count * columns_count
    required_bytes = entries * _coordinate_dtype(operator.target).itemsize
    if entries > materialization.max_entries:
        raise LinearCapabilityError(
            f"Dense sparse-assembly fallback requires {entries} entries, exceeding "
            f"the materialization limit {materialization.max_entries}."
        )
    if required_bytes > materialization.max_bytes:
        raise LinearCapabilityError(
            f"Dense sparse-assembly fallback requires {required_bytes} bytes, "
            f"exceeding the materialization limit {materialization.max_bytes}."
        )
    _check_contribution_budget(policy, entries, arrays=3)
    rows = np.repeat(np.arange(rows_count, dtype=np.int64), columns_count)
    columns = np.tile(np.arange(columns_count, dtype=np.int64), rows_count)
    return _make_sparse_recipe(
        "materialized",
        operator,
        rows,
        columns,
        policy,
        contribution_count=entries,
    )


def _make_sparse_recipe(
    kind: SparseAssemblyKind,
    operator: AbstractLinearOperator,
    rows: np.ndarray,
    columns: np.ndarray,
    policy: SparseAssemblyPolicy,
    /,
    *,
    children: tuple[_SparseAssemblyRecipe, ...] = (),
    input_indices: tuple[np.ndarray, ...] = (),
    output_indices: tuple[np.ndarray, ...] = (),
    payload: tuple[Any, ...] = (),
    contribution_count: int,
) -> _SparseAssemblyRecipe:
    rows_ = np.asarray(rows, dtype=np.int64).reshape((-1,))
    columns_ = np.asarray(columns, dtype=np.int64).reshape((-1,))
    if rows_.shape != columns_.shape:
        raise ValueError("Sparse assembly row and column patterns must match.")
    shape = (operator.target.size, operator.source.size)
    if rows_.size and (
        np.any(rows_ < 0)
        or np.any(rows_ >= shape[0])
        or np.any(columns_ < 0)
        or np.any(columns_ >= shape[1])
    ):
        raise ValueError("Sparse assembly produced an out-of-bounds route.")
    largest = max(shape, default=0)
    if largest >= np.iinfo(np.int32).max or rows_.size >= np.iinfo(np.int32).max:
        raise LinearCapabilityError(
            "Sparse assembly currently requires 32-bit route indices."
        )
    nnz = int(rows_.size)
    contributions = int(contribution_count)
    if nnz > policy.max_nnz:
        raise LinearCapabilityError(
            f"Sparse assembly requires {nnz} nonzeros, exceeding the "
            f"limit {policy.max_nnz}."
        )
    if contributions > policy.max_contributions:
        raise LinearCapabilityError(
            f"Sparse assembly requires {contributions} symbolic contributions, "
            f"exceeding the limit {policy.max_contributions}."
        )
    output_bytes = _sparse_output_bytes(
        shape,
        nnz,
        _coordinate_dtype(operator.target).itemsize,
    )
    if output_bytes > policy.max_bytes:
        raise LinearCapabilityError(
            f"Sparse assembly output requires {output_bytes} bytes, exceeding "
            f"the limit {policy.max_bytes}."
        )
    mapping_entries = sum(
        np.asarray(indices).size for indices in (*input_indices, *output_indices)
    )
    recipe_bytes = _array_storage_bytes(children) + 4 * (2 * nnz + mapping_entries)
    if recipe_bytes > policy.max_workspace_bytes:
        raise LinearCapabilityError(
            f"Sparse assembly recipe requires {recipe_bytes} bytes, exceeding "
            f"the workspace limit {policy.max_workspace_bytes}."
        )
    symbolic_workspace_bytes = 8 * (6 * contributions + 2 * nnz) + 4 * mapping_entries
    numeric_workspace_bytes = (contributions + nnz) * _coordinate_dtype(
        operator.target
    ).itemsize
    required_workspace = max(
        symbolic_workspace_bytes,
        numeric_workspace_bytes,
    )
    if required_workspace > policy.max_workspace_bytes:
        raise LinearCapabilityError(
            f"Sparse assembly requires {required_workspace} workspace bytes, "
            f"exceeding the limit {policy.max_workspace_bytes}."
        )
    return _SparseAssemblyRecipe(
        kind=kind,
        operator=operator,
        rows=rows_,
        columns=columns_,
        children=children,
        input_indices=input_indices,
        output_indices=output_indices,
        payload=payload,
        contribution_count=contributions,
        symbolic_workspace_bytes=symbolic_workspace_bytes,
        numeric_workspace_bytes=numeric_workspace_bytes,
    )


def _canonical_pattern(
    rows: np.ndarray,
    columns: np.ndarray,
    shape: tuple[int, int],
    policy: SparseAssemblyPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_ = np.asarray(rows, dtype=np.int64).reshape((-1,))
    columns_ = np.asarray(columns, dtype=np.int64).reshape((-1,))
    if rows_.shape != columns_.shape:
        raise ValueError("Sparse contribution rows and columns must match.")
    _check_contribution_budget(policy, int(rows_.size), arrays=8)
    if rows_.size == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, empty
    if (
        np.any(rows_ < 0)
        or np.any(rows_ >= shape[0])
        or np.any(columns_ < 0)
        or np.any(columns_ >= shape[1])
    ):
        raise ValueError("Sparse contributions contain an out-of-bounds route.")
    order = np.lexsort((columns_, rows_))
    sorted_rows = rows_[order]
    sorted_columns = columns_[order]
    starts = np.concatenate(
        (
            np.asarray([True]),
            (sorted_rows[1:] != sorted_rows[:-1])
            | (sorted_columns[1:] != sorted_columns[:-1]),
        )
    )
    groups = np.cumsum(starts, dtype=np.int64) - 1
    mapping = np.empty(rows_.size, dtype=np.int64)
    mapping[order] = groups
    return sorted_rows[starts], sorted_columns[starts], mapping


def _check_contribution_budget(
    policy: SparseAssemblyPolicy,
    contributions: int,
    /,
    *,
    arrays: int,
) -> None:
    count = int(contributions)
    if count > policy.max_contributions:
        raise LinearCapabilityError(
            f"Sparse assembly requires {count} symbolic contributions, "
            f"exceeding the limit {policy.max_contributions}."
        )
    required = count * int(arrays) * np.dtype(np.int64).itemsize
    if required > policy.max_workspace_bytes:
        raise LinearCapabilityError(
            f"Sparse symbolic construction requires {required} workspace bytes, "
            f"exceeding the limit {policy.max_workspace_bytes}."
        )


def _concatenate_indices(parts: list[np.ndarray], /) -> np.ndarray:
    if not parts:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(parts).astype(np.int64, copy=False)


def _plan_sparse_composition(
    operator: ComposedLinearOperator,
    policy: SparseAssemblyPolicy,
    /,
) -> _SparseAssemblyRecipe:
    left = _plan_sparse_recipe(operator.left, policy)
    right = _plan_sparse_recipe(operator.right, policy)
    left_columns = np.asarray(left.columns, dtype=np.int64)
    right_rows = np.asarray(right.rows, dtype=np.int64)
    inner_size = operator.left.source.size
    left_counts = np.bincount(left_columns, minlength=inner_size)
    right_counts = np.bincount(right_rows, minlength=inner_size)
    contributions = int(np.dot(left_counts, right_counts))
    _check_contribution_budget(policy, contributions, arrays=10)

    left_order = np.argsort(left_columns, kind="stable")
    right_order = np.argsort(right_rows, kind="stable")
    left_offsets = np.concatenate(([0], np.cumsum(left_counts)))
    right_offsets = np.concatenate(([0], np.cumsum(right_counts)))
    left_parts = []
    right_parts = []
    for inner in range(inner_size):
        left_indices = left_order[left_offsets[inner] : left_offsets[inner + 1]]
        right_indices = right_order[right_offsets[inner] : right_offsets[inner + 1]]
        if left_indices.size and right_indices.size:
            left_parts.append(np.repeat(left_indices, right_indices.size))
            right_parts.append(np.tile(right_indices, left_indices.size))
    left_indices = _concatenate_indices(left_parts)
    right_indices = _concatenate_indices(right_parts)
    rows = np.asarray(left.rows, dtype=np.int64)[left_indices]
    columns = np.asarray(right.columns, dtype=np.int64)[right_indices]
    result_rows, result_columns, mapping = _canonical_pattern(
        rows,
        columns,
        (operator.target.size, operator.source.size),
        policy,
    )
    return _make_sparse_recipe(
        "composition",
        operator,
        result_rows,
        result_columns,
        policy,
        children=(left, right),
        input_indices=(left_indices, right_indices),
        output_indices=(mapping,),
        contribution_count=contributions,
    )


def _plan_sparse_block_diagonal(
    operator: BlockDiagonalLinearOperator,
    policy: SparseAssemblyPolicy,
    /,
) -> _SparseAssemblyRecipe:
    children = tuple(_plan_sparse_recipe(block, policy) for block in operator.blocks)
    row_parts = []
    column_parts = []
    row_offset = 0
    column_offset = 0
    for block, child in zip(operator.blocks, children, strict=True):
        row_parts.append(np.asarray(child.rows, dtype=np.int64) + row_offset)
        column_parts.append(np.asarray(child.columns, dtype=np.int64) + column_offset)
        row_offset += block.target.size
        column_offset += block.source.size
    rows = _concatenate_indices(row_parts)
    columns = _concatenate_indices(column_parts)
    result_rows, result_columns, mapping = _canonical_pattern(
        rows,
        columns,
        (operator.target.size, operator.source.size),
        policy,
    )
    output_indices = []
    offset = 0
    for child in children:
        stop = offset + int(child.rows.size)
        output_indices.append(mapping[offset:stop])
        offset = stop
    return _make_sparse_recipe(
        "block-diagonal",
        operator,
        result_rows,
        result_columns,
        policy,
        children=children,
        output_indices=tuple(output_indices),
        contribution_count=rows.size,
    )


def _plan_sparse_kronecker(
    operator: KroneckerLinearOperator,
    policy: SparseAssemblyPolicy,
    /,
) -> _SparseAssemblyRecipe:
    children = tuple(_plan_sparse_recipe(factor, policy) for factor in operator.factors)
    counts = tuple(int(child.rows.size) for child in children)
    contributions = prod(counts)
    _check_contribution_budget(
        policy,
        contributions,
        arrays=6 + 2 * len(children),
    )
    if contributions:
        input_indices = tuple(
            grid.reshape((-1,))
            for grid in np.meshgrid(
                *(np.arange(count, dtype=np.int64) for count in counts),
                indexing="ij",
            )
        )
        rows = np.zeros((contributions,), dtype=np.int64)
        columns = np.zeros((contributions,), dtype=np.int64)
        for child, indices in zip(children, input_indices, strict=True):
            rows = rows * child.shape[0] + np.asarray(child.rows, dtype=np.int64)[indices]
            columns = (
                columns * child.shape[1]
                + np.asarray(child.columns, dtype=np.int64)[indices]
            )
    else:
        input_indices = tuple(np.zeros((0,), dtype=np.int64) for _ in children)
        rows = np.zeros((0,), dtype=np.int64)
        columns = np.zeros((0,), dtype=np.int64)
    result_rows, result_columns, mapping = _canonical_pattern(
        rows,
        columns,
        (operator.target.size, operator.source.size),
        policy,
    )
    return _make_sparse_recipe(
        "kronecker",
        operator,
        result_rows,
        result_columns,
        policy,
        children=children,
        input_indices=input_indices,
        output_indices=(mapping,),
        contribution_count=contributions,
    )


def _plan_sparse_kronecker_sum(
    operator: KroneckerSumLinearOperator,
    policy: SparseAssemblyPolicy,
    /,
) -> _SparseAssemblyRecipe:
    children = tuple(_plan_sparse_recipe(factor, policy) for factor in operator.factors)
    sizes = tuple(factor.source.size for factor in operator.factors)
    term_counts = tuple(
        int(child.rows.size) * prod(sizes[:axis] + sizes[axis + 1 :])
        for axis, child in enumerate(children)
    )
    contributions = sum(term_counts)
    _check_contribution_budget(
        policy,
        contributions,
        arrays=8 + 2 * len(children),
    )
    strides = tuple(prod(sizes[axis + 1 :]) for axis in range(len(sizes)))
    row_parts = []
    column_parts = []
    input_indices = []
    for axis, (child, term_count) in enumerate(zip(children, term_counts, strict=True)):
        child_nnz = int(child.rows.size)
        if not term_count:
            row_parts.append(np.zeros((0,), dtype=np.int64))
            column_parts.append(np.zeros((0,), dtype=np.int64))
            input_indices.append(np.zeros((0,), dtype=np.int64))
            continue
        other_axes = tuple(index for index in range(len(sizes)) if index != axis)
        other_sizes = tuple(sizes[index] for index in other_axes)
        if other_sizes:
            other_coordinates = np.indices(
                other_sizes,
                dtype=np.int64,
            ).reshape((len(other_sizes), -1))
        else:
            other_coordinates = np.zeros((0, 1), dtype=np.int64)
        factor_indices = np.repeat(
            np.arange(child_nnz, dtype=np.int64),
            other_coordinates.shape[1],
        )
        rows = np.asarray(child.rows, dtype=np.int64)[factor_indices] * strides[axis]
        columns = (
            np.asarray(child.columns, dtype=np.int64)[factor_indices] * strides[axis]
        )
        for other_position, other_axis in enumerate(other_axes):
            coordinates = np.tile(
                other_coordinates[other_position],
                child_nnz,
            )
            rows = rows + coordinates * strides[other_axis]
            columns = columns + coordinates * strides[other_axis]
        row_parts.append(rows)
        column_parts.append(columns)
        input_indices.append(factor_indices)
    rows = _concatenate_indices(row_parts)
    columns = _concatenate_indices(column_parts)
    result_rows, result_columns, mapping = _canonical_pattern(
        rows,
        columns,
        (operator.target.size, operator.source.size),
        policy,
    )
    output_indices = []
    offset = 0
    for term_count in term_counts:
        stop = offset + term_count
        output_indices.append(mapping[offset:stop])
        offset = stop
    return _make_sparse_recipe(
        "kronecker-sum",
        operator,
        result_rows,
        result_columns,
        policy,
        children=children,
        input_indices=tuple(input_indices),
        output_indices=tuple(output_indices),
        contribution_count=contributions,
    )


def _sparse_output_bytes(
    shape: tuple[int, int],
    nnz: int,
    itemsize: int,
    /,
) -> int:
    index_itemsize = np.dtype(np.int32).itemsize
    return int(nnz * (itemsize + index_itemsize) + (shape[0] + 1) * index_itemsize)


def _array_storage_bytes(value: Any, /) -> int:
    arrays = {id(leaf): leaf for leaf in jax.tree.leaves(value) if eqx.is_array(leaf)}
    return sum(int(array.size * array.dtype.itemsize) for array in arrays.values())


def _walk_recipes(
    recipe: _SparseAssemblyRecipe,
    /,
) -> tuple[_SparseAssemblyRecipe, ...]:
    return (
        recipe,
        *(descendant for child in recipe.children for descendant in _walk_recipes(child)),
    )


def _recipe_signature(recipe: _SparseAssemblyRecipe, /) -> dict[str, Any]:
    return {
        "kind": recipe.kind,
        "operator_type": (
            f"{recipe.operator_type.__module__}.{recipe.operator_type.__qualname__}"
        ),
        "shape": list(recipe.shape),
        "source": recipe.source_space_id,
        "target": recipe.target_space_id,
        "payload": list(recipe.payload),
        "children": [_recipe_signature(child) for child in recipe.children],
    }


def _evaluate_sparse_recipe(
    recipe: _SparseAssemblyRecipe,
    operator: AbstractLinearOperator,
    policy: SparseAssemblyPolicy,
    /,
) -> Array:
    _validate_recipe_operator(recipe, operator)
    kind = recipe.kind
    rows = recipe.rows
    columns = recipe.columns

    if kind == "sparse":
        if not isinstance(operator, AbstractSparseLinearOperator):
            raise ValueError("Sparse assembly refresh changed sparse leaf structure.")
        storage = operator.sparse_storage()
        if not storage.canonical or not storage.sorted_indices:
            raise ValueError("Sparse assembly refresh produced noncanonical storage.")
        storage_rows = np.repeat(
            np.arange(storage.shape[0], dtype=np.int64),
            np.diff(np.asarray(storage.indptr, dtype=np.int64)),
        )
        _validate_numeric_pattern(
            recipe,
            storage_rows,
            np.asarray(storage.indices, dtype=np.int64),
        )
        values = jnp.asarray(storage.values)
        if values.shape != rows.shape:
            raise ValueError("Sparse assembly refresh changed sparse leaf capacity.")
        return values
    if kind == "identity":
        return jnp.ones(
            rows.shape,
            dtype=_coordinate_dtype(operator.target),
        )
    if kind == "diagonal":
        if not isinstance(operator, DiagonalLinearOperator):
            raise ValueError("Sparse assembly refresh changed diagonal structure.")
        return jnp.asarray(operator.diagonal).reshape((-1,))
    if kind == "permutation":
        if not isinstance(operator, PermutationLinearOperator):
            raise ValueError("Sparse assembly refresh changed permutation structure.")
        current_rows = np.arange(operator.source.size, dtype=np.int64)
        current_columns = np.asarray(operator.permutation, dtype=np.int64)
        _validate_numeric_pattern(recipe, current_rows, current_columns)
        return jnp.ones(
            rows.shape,
            dtype=_coordinate_dtype(operator.target),
        )
    if kind == "triangular":
        if not isinstance(operator, TriangularLinearOperator) or recipe.payload != (
            operator.lower,
            operator.unit_diagonal,
        ):
            raise ValueError("Sparse assembly refresh changed triangular structure.")
        return operator.matrix[rows, columns]
    if kind == "tridiagonal":
        if not isinstance(operator, TridiagonalLinearOperator):
            raise ValueError("Sparse assembly refresh changed tridiagonal structure.")
        offsets = rows - columns
        lower_indices = jnp.clip(columns, 0, max(operator.lower.size - 1, 0))
        upper_indices = jnp.clip(rows, 0, max(operator.upper.size - 1, 0))
        lower = (
            jnp.zeros(rows.shape, dtype=operator.diagonal.dtype)
            if operator.lower.size == 0
            else operator.lower[lower_indices]
        )
        upper = (
            jnp.zeros(rows.shape, dtype=operator.diagonal.dtype)
            if operator.upper.size == 0
            else operator.upper[upper_indices]
        )
        return jnp.where(
            offsets == 0,
            operator.diagonal[rows],
            jnp.where(offsets == 1, lower, upper),
        )
    if kind == "banded":
        if not isinstance(operator, BandedLinearOperator) or recipe.payload != (
            operator.lower_bandwidth,
            operator.upper_bandwidth,
        ):
            raise ValueError("Sparse assembly refresh changed banded structure.")
        band_indices = operator.upper_bandwidth + rows - columns
        return operator.bands[band_indices, columns]
    if kind == "local-block":
        if not isinstance(operator, LocalBlockDiagonalLinearOperator) or (
            recipe.payload
            != (
                operator.num_blocks,
                operator.input_block_size,
                operator.output_block_size,
            )
        ):
            raise ValueError("Sparse assembly refresh changed local-block structure.")
        block_indices = rows // operator.output_block_size
        local_rows = rows % operator.output_block_size
        local_columns = columns % operator.input_block_size
        return operator.blocks[block_indices, local_rows, local_columns]
    if kind == "materialized":
        materialization = policy.materialization
        if materialization is None:
            raise ValueError(
                "Sparse assembly plan lost its explicit materialization policy."
            )
        matrix = materialize(operator, materialization)
        return matrix[rows, columns]
    if kind == "scale":
        scaled = cast(ScaledLinearOperator, operator)
        child_values = _evaluate_sparse_recipe(
            recipe.children[0],
            scaled.operator,
            policy,
        )
        return scaled.scalar * child_values
    if kind == "sum":
        summed = cast(SumLinearOperator, operator)
        child_operators = (summed.left, summed.right)
        return _sum_recipe_contributions(
            recipe,
            child_operators,
            policy,
        )
    if kind == "composition":
        composed = cast(ComposedLinearOperator, operator)
        left_values = _evaluate_sparse_recipe(
            recipe.children[0],
            composed.left,
            policy,
        )
        right_values = _evaluate_sparse_recipe(
            recipe.children[1],
            composed.right,
            policy,
        )
        contributions = (
            left_values[recipe.input_indices[0]] * right_values[recipe.input_indices[1]]
        )
        return _scatter_recipe_values(
            contributions,
            recipe.output_indices[0],
            int(rows.size),
        )
    if kind == "transpose":
        transposed = cast(TransposeLinearOperator, operator)
        child_values = _evaluate_sparse_recipe(
            recipe.children[0],
            transposed.operator,
            policy,
        )
        return _scatter_recipe_values(
            child_values,
            recipe.output_indices[0],
            int(rows.size),
        )
    if kind == "adjoint":
        adjointed = cast(AdjointLinearOperator, operator)
        child = recipe.children[0]
        original = adjointed.operator
        child_values = _evaluate_sparse_recipe(child, original, policy)
        target_weights = _coordinate_pairing_weights(original.target)
        source_weights = _coordinate_pairing_weights(original.source)
        contributions = (
            jnp.conj(child_values)
            * target_weights[child.rows]
            / source_weights[child.columns]
        )
        return _scatter_recipe_values(
            contributions,
            recipe.output_indices[0],
            int(rows.size),
        )
    if kind == "block-diagonal":
        blocked = cast(BlockDiagonalLinearOperator, operator)
        return _sum_recipe_contributions(
            recipe,
            blocked.blocks,
            policy,
        )
    if kind == "kronecker":
        kronecker = cast(KroneckerLinearOperator, operator)
        child_values = tuple(
            _evaluate_sparse_recipe(child, factor, policy)
            for child, factor in zip(
                recipe.children,
                kronecker.factors,
                strict=True,
            )
        )
        dtype = jnp.result_type(
            *(value.dtype for value in child_values),
            _coordinate_dtype(kronecker.target),
        )
        contributions = jnp.ones(
            (recipe.contribution_count,),
            dtype=dtype,
        )
        for values, indices in zip(
            child_values,
            recipe.input_indices,
            strict=True,
        ):
            contributions = contributions * values[indices]
        return _scatter_recipe_values(
            contributions,
            recipe.output_indices[0],
            int(rows.size),
        )
    if kind == "kronecker-sum":
        kronecker_sum = cast(KroneckerSumLinearOperator, operator)
        child_values = tuple(
            _evaluate_sparse_recipe(child, factor, policy)
            for child, factor in zip(
                recipe.children,
                kronecker_sum.factors,
                strict=True,
            )
        )
        dtype = jnp.result_type(
            *(value.dtype for value in child_values),
            _coordinate_dtype(kronecker_sum.target),
        )
        result = jnp.zeros((rows.size,), dtype=dtype)
        for values, input_indices, output_indices in zip(
            child_values,
            recipe.input_indices,
            recipe.output_indices,
            strict=True,
        ):
            result = result.at[output_indices].add(values[input_indices])
        return result
    raise TypeError(f"Unknown sparse assembly recipe kind {kind!r}.")


def _sum_recipe_contributions(
    recipe: _SparseAssemblyRecipe,
    operators: tuple[AbstractLinearOperator, ...],
    policy: SparseAssemblyPolicy,
    /,
) -> Array:
    if len(recipe.children) != len(operators):
        raise ValueError("Sparse assembly refresh changed composite arity.")
    child_values = tuple(
        _evaluate_sparse_recipe(child, child_operator, policy)
        for child, child_operator in zip(
            recipe.children,
            operators,
            strict=True,
        )
    )
    dtype = jnp.result_type(*(value.dtype for value in child_values))
    result = jnp.zeros((recipe.rows.size,), dtype=dtype)
    for values, output_indices in zip(
        child_values,
        recipe.output_indices,
        strict=True,
    ):
        result = result.at[output_indices].add(values)
    return result


def _scatter_recipe_values(
    values: Array,
    output_indices: Array,
    size: int,
    /,
) -> Array:
    return jnp.zeros((size,), dtype=values.dtype).at[output_indices].add(values)


def _validate_recipe_operator(
    recipe: _SparseAssemblyRecipe,
    operator: AbstractLinearOperator,
    /,
) -> None:
    if type(operator) is not recipe.operator_type:
        raise ValueError(
            "Sparse assembly refresh changed operator structure: expected "
            f"{recipe.operator_type.__name__}, got {type(operator).__name__}."
        )
    if operator.batch_shape:
        raise ValueError("Sparse assembly refresh introduced an operator batch.")
    if (operator.target.size, operator.source.size) != recipe.shape:
        raise ValueError("Sparse assembly refresh changed operator shape.")
    if (
        operator.source.space_id != recipe.source_space_id
        or operator.target.space_id != recipe.target_space_id
    ):
        raise ValueError("Sparse assembly refresh changed an internal vector space.")


def _validate_numeric_pattern(
    recipe: _SparseAssemblyRecipe,
    rows: np.ndarray,
    columns: np.ndarray,
    /,
) -> None:
    if not np.array_equal(np.asarray(recipe.rows), np.asarray(rows)) or not (
        np.array_equal(np.asarray(recipe.columns), np.asarray(columns))
    ):
        raise ValueError("Sparse assembly refresh changed the symbolic pattern.")


__all__ = [
    "PreparedSparseAssembly",
    "SparseAssemblyCostEstimate",
    "SparseAssemblyPlan",
    "SparseAssemblyPolicy",
    "assemble_diagonal",
    "assemble_uniform_blocks",
    "assemble_sparse",
    "plan_sparse_assembly",
    "prepare_sparse_assembly",
    "refresh_sparse_assembly",
]
