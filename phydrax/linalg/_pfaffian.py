#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


SkewMode = Literal["require", "project"]


class PfaffianStatus(IntEnum):
    """Per-matrix outcome of a Pfaffian evaluation."""

    SUCCESS = 0
    STRUCTURAL_ZERO = 1
    SINGULAR = 2
    NONFINITE_INPUT = 3
    ANTISYMMETRY_VIOLATION = 4
    NONFINITE_VALUE = 5


class PfaffianPolicy(StrictModule):
    """Skew validation, numerical-rank, and hard resource policy."""

    skew_mode: SkewMode = eqx.field(static=True)
    antisymmetry_tolerance: float = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    verify_determinant: bool = eqx.field(static=True)
    max_dimension: int = eqx.field(static=True)
    max_batch_size: int = eqx.field(static=True)
    max_storage_bytes: int = eqx.field(static=True)
    max_workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        skew_mode: SkewMode = "require",
        antisymmetry_tolerance: float = 1.0e-10,
        pivot_tolerance: float = 1.0e-14,
        verify_determinant: bool = False,
        max_dimension: int = 8192,
        max_batch_size: int = 65536,
        max_storage_bytes: int = 512 * 1024 * 1024,
        max_workspace_bytes: int = 512 * 1024 * 1024,
    ):
        if skew_mode not in ("require", "project"):
            raise ValueError("skew_mode must be 'require' or 'project'.")
        antisymmetry = float(antisymmetry_tolerance)
        pivot = float(pivot_tolerance)
        if any(
            not math.isfinite(value) or value < 0.0 for value in (antisymmetry, pivot)
        ):
            raise ValueError("Pfaffian tolerances must be finite and non-negative.")
        if not isinstance(verify_determinant, bool):
            raise TypeError("verify_determinant must be bool.")
        limits = tuple(
            int(value)
            for value in (
                max_dimension,
                max_batch_size,
                max_storage_bytes,
                max_workspace_bytes,
            )
        )
        if limits[0] < 0 or any(value < 1 for value in limits[1:]):
            raise ValueError(
                "max_dimension must be non-negative and Pfaffian resource limits "
                "must be positive."
            )
        self.skew_mode = skew_mode
        self.antisymmetry_tolerance = antisymmetry
        self.pivot_tolerance = pivot
        self.verify_determinant = verify_determinant
        (
            self.max_dimension,
            self.max_batch_size,
            self.max_storage_bytes,
            self.max_workspace_bytes,
        ) = limits
        self.policy_id = canonical_fingerprint(
            {
                "kind": "pfaffian-policy",
                "skew_mode": skew_mode,
                "antisymmetry_tolerance": antisymmetry.hex(),
                "pivot_tolerance": pivot.hex(),
                "verify_determinant": verify_determinant,
                "max_dimension": limits[0],
                "max_batch_size": limits[1],
                "max_storage_bytes": limits[2],
                "max_workspace_bytes": limits[3],
            }
        )


class PfaffianPlan(StrictModule):
    """Static shape, dtype, work, and storage admission for skew elimination."""

    policy: PfaffianPolicy
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    pair_count: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    elimination_steps: int = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        policy: PfaffianPolicy,
        batch_shape: tuple[int, ...],
        dimension: int,
        dtype: Any,
        storage_bytes: int,
        workspace_bytes: int,
    ):
        dtype_ = np.dtype(dtype)
        batch = tuple(int(size) for size in batch_shape)
        dimension_ = int(dimension)
        self.policy = policy
        self.batch_shape = batch
        self.dimension = dimension_
        self.pair_count = dimension_ // 2
        self.batch_size = math.prod(batch)
        self.dtype = dtype_
        self.elimination_steps = dimension_ // 2
        self.storage_bytes = int(storage_bytes)
        self.workspace_bytes = int(workspace_bytes)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pfaffian-plan",
                "policy": policy.policy_id,
                "batch_shape": batch,
                "dimension": dimension_,
                "dtype": dtype_.str,
                "algorithm": "pivoted-skew-ldlt-lax",
                "elimination_steps": self.elimination_steps,
                "storage_bytes": self.storage_bytes,
                "workspace_bytes": self.workspace_bytes,
            }
        )


class PreparedPfaffian(StrictModule):
    """Reusable numerical skew-LDLᵀ factors and validation evidence."""

    plan: PfaffianPlan
    matrix: Array
    factor_scale: Array
    lower: Array
    pivots: Array
    permutation: Array
    swap_sign: Array
    singular: Array
    minimum_pivot: Array
    input_finite: Array
    antisymmetry_residual: Array
    antisymmetric: Array
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan: PfaffianPlan,
        matrix: Array,
        factor_scale: Array,
        lower: Array,
        pivots: Array,
        permutation: Array,
        swap_sign: Array,
        singular: Array,
        minimum_pivot: Array,
        input_finite: Array,
        antisymmetry_residual: Array,
        antisymmetric: Array,
        numeric_version: Any,
    ):
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("numeric_version must be scalar.")
        self.plan = plan
        self.matrix = jnp.asarray(matrix)
        self.factor_scale = jnp.asarray(factor_scale)
        self.lower = jnp.asarray(lower)
        self.pivots = jnp.asarray(pivots)
        self.permutation = jnp.asarray(permutation, dtype=jnp.int32)
        self.swap_sign = jnp.asarray(swap_sign)
        self.singular = jnp.asarray(singular, dtype=bool)
        self.minimum_pivot = jnp.asarray(minimum_pivot)
        self.input_finite = jnp.asarray(input_finite, dtype=bool)
        self.antisymmetry_residual = jnp.asarray(antisymmetry_residual)
        self.antisymmetric = jnp.asarray(antisymmetric, dtype=bool)
        self.numeric_version = version
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-pfaffian",
                "plan": plan.plan_id,
                "state": "numeric",
            }
        )


class PfaffianResult(StrictModule):
    """Signed-log Pfaffian value with skew, singularity, and factor evidence."""

    value: Array
    value_finite: Array
    sign: Array
    log_abs: Array
    singular: Array
    antisymmetry_residual: Array
    determinant_identity_residual: Array
    determinant_identity_verified: Array
    pivot_magnitudes: Array
    minimum_pivot: Array
    input_finite: Array
    antisymmetric: Array
    log_derivative_valid: Array
    value_derivative_valid: Array
    status: Array
    numeric_version: Array
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        value: Array,
        value_finite: Array,
        sign: Array,
        log_abs: Array,
        singular: Array,
        antisymmetry_residual: Array,
        determinant_identity_residual: Array,
        determinant_identity_verified: Array,
        pivot_magnitudes: Array,
        minimum_pivot: Array,
        input_finite: Array,
        antisymmetric: Array,
        log_derivative_valid: Array,
        value_derivative_valid: Array,
        status: Array,
        numeric_version: Array,
        plan_id: str,
        prepared_id: str,
    ):
        self.value = jnp.asarray(value)
        self.value_finite = jnp.asarray(value_finite, dtype=bool)
        self.sign = jnp.asarray(sign)
        self.log_abs = jnp.asarray(log_abs)
        self.singular = jnp.asarray(singular, dtype=bool)
        self.antisymmetry_residual = jnp.asarray(antisymmetry_residual)
        self.determinant_identity_residual = jnp.asarray(determinant_identity_residual)
        self.determinant_identity_verified = jnp.asarray(
            determinant_identity_verified,
            dtype=bool,
        )
        self.pivot_magnitudes = jnp.asarray(pivot_magnitudes)
        self.minimum_pivot = jnp.asarray(minimum_pivot)
        self.input_finite = jnp.asarray(input_finite, dtype=bool)
        self.antisymmetric = jnp.asarray(antisymmetric, dtype=bool)
        self.log_derivative_valid = jnp.asarray(log_derivative_valid, dtype=bool)
        self.value_derivative_valid = jnp.asarray(value_derivative_valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.numeric_version = jnp.asarray(numeric_version, dtype=jnp.int32)
        self.plan_id = plan_id
        self.prepared_id = prepared_id

    @property
    def successful(self) -> Array:
        return (self.status <= int(PfaffianStatus.SINGULAR)) | (
            self.status == int(PfaffianStatus.NONFINITE_VALUE)
        )


def plan_pfaffian(
    matrix: ArrayLike,
    policy: PfaffianPolicy | None = None,
    /,
) -> PfaffianPlan:
    """Admit a statically shaped dense Pfaffian factorization."""
    value = _as_matrix(matrix)
    policy_ = PfaffianPolicy() if policy is None else policy
    if not isinstance(policy_, PfaffianPolicy):
        raise TypeError("policy must be a PfaffianPolicy or None.")
    dimension = value.shape[-1]
    batch_shape = value.shape[:-2]
    batch_size = math.prod(batch_shape)
    pairs = dimension // 2
    itemsize = np.dtype(value.dtype).itemsize
    real_itemsize = np.dtype(jnp.empty((), dtype=value.dtype).real.dtype).itemsize
    storage_bytes = batch_size * (
        (2 * dimension * dimension + pairs) * itemsize
        + dimension * np.dtype(np.int32).itemsize
        + (4 * real_itemsize + itemsize)
    )
    workspace_matrices = 5 if policy_.verify_determinant else 3
    workspace_bytes = (
        batch_size
        * itemsize
        * (workspace_matrices * dimension * dimension + 3 * dimension + 1)
    )
    failures = []
    if dimension > policy_.max_dimension:
        failures.append("dimension exceeds max_dimension")
    if batch_size > policy_.max_batch_size:
        failures.append("batch size exceeds max_batch_size")
    if storage_bytes > policy_.max_storage_bytes:
        failures.append("persistent factors exceed max_storage_bytes")
    if workspace_bytes > policy_.max_workspace_bytes:
        failures.append("factorization work exceeds max_workspace_bytes")
    if failures:
        raise ValueError("Pfaffian resource rejection: " + "; ".join(failures) + ".")
    return PfaffianPlan(
        policy=policy_,
        batch_shape=batch_shape,
        dimension=dimension,
        dtype=value.dtype,
        storage_bytes=storage_bytes,
        workspace_bytes=workspace_bytes,
    )


def prepare_pfaffian(
    matrix: ArrayLike,
    policy: PfaffianPolicy | PfaffianPlan | None = None,
    /,
    *,
    numeric_version: Any = 0,
) -> PreparedPfaffian:
    """Validate and factor one fixed-shape batch of skew matrices."""
    value = _as_matrix(matrix)
    plan = policy if isinstance(policy, PfaffianPlan) else plan_pfaffian(value, policy)
    if not isinstance(plan, PfaffianPlan):
        raise TypeError("policy must be a PfaffianPolicy, PfaffianPlan, or None.")
    _validate_plan_matrix(plan, value)
    return _prepare_numeric(value, plan, numeric_version=numeric_version)


def refresh_pfaffian(
    prepared: PreparedPfaffian,
    matrix: ArrayLike,
    /,
) -> PreparedPfaffian:
    """Re-factor changed values under the same admitted shape and policy."""
    if not isinstance(prepared, PreparedPfaffian):
        raise TypeError("prepared must be a PreparedPfaffian.")
    value = _as_matrix(matrix)
    _validate_plan_matrix(prepared.plan, value)
    return _prepare_numeric(
        value,
        prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def evaluate_pfaffian(
    matrix_or_prepared: ArrayLike | PreparedPfaffian,
    policy: PfaffianPolicy | PfaffianPlan | None = None,
    /,
) -> PfaffianResult:
    """Evaluate a Pfaffian from new input or reusable prepared factors."""
    if isinstance(matrix_or_prepared, PreparedPfaffian):
        if policy is not None:
            raise ValueError("policy must be omitted when evaluating prepared state.")
        prepared = matrix_or_prepared
    else:
        prepared = prepare_pfaffian(matrix_or_prepared, policy)

    raw_value, raw_sign, raw_log_abs = _prepared_pfaffian_outputs(
        prepared.matrix,
        prepared.lower,
        prepared.factor_scale,
        prepared.pivots,
        prepared.permutation,
        prepared.swap_sign,
        prepared.singular,
    )
    policy_ = prepared.plan.policy
    valid_skew = prepared.antisymmetric | (policy_.skew_mode == "project")
    valid_input = prepared.input_finite & valid_skew
    nan_value = jnp.asarray(jnp.nan, dtype=raw_value.dtype)
    value = jnp.where(valid_input, raw_value, nan_value)
    sign = jnp.where(valid_input, raw_sign, nan_value)
    nan_log = jnp.asarray(jnp.nan, dtype=raw_log_abs.dtype)
    log_abs = jnp.where(valid_input, raw_log_abs, nan_log)

    dimension = prepared.plan.dimension
    status = jnp.full(
        prepared.plan.batch_shape,
        int(PfaffianStatus.SUCCESS),
        dtype=jnp.int32,
    )
    if dimension % 2:
        status = jnp.full_like(status, int(PfaffianStatus.STRUCTURAL_ZERO))
    status = jnp.where(
        prepared.singular & (dimension % 2 == 0),
        int(PfaffianStatus.SINGULAR),
        status,
    )
    status = jnp.where(
        valid_input & ~prepared.singular & ~jnp.isfinite(raw_value),
        int(PfaffianStatus.NONFINITE_VALUE),
        status,
    )
    status = jnp.where(
        prepared.input_finite & ~valid_skew,
        int(PfaffianStatus.ANTISYMMETRY_VIOLATION),
        status,
    )
    status = jnp.where(
        ~prepared.input_finite,
        int(PfaffianStatus.NONFINITE_INPUT),
        status,
    )
    if policy_.verify_determinant:
        determinant_residual = _determinant_identity_residual(
            prepared.matrix,
            raw_sign,
            raw_log_abs,
        )
        determinant_residual = jnp.where(
            valid_input,
            determinant_residual,
            nan_log,
        )
        determinant_verified = valid_input & jnp.isfinite(determinant_residual)
    else:
        determinant_residual = jnp.full(
            prepared.plan.batch_shape,
            jnp.nan,
            dtype=raw_log_abs.dtype,
        )
        determinant_verified = jnp.zeros(
            prepared.plan.batch_shape,
            dtype=bool,
        )
    minimum_pivot = jnp.where(valid_input, prepared.minimum_pivot, nan_log)
    log_derivative_valid = valid_input & ~prepared.singular & jnp.isfinite(raw_log_abs)
    value_derivative_valid = valid_input & (
        ~prepared.singular | (dimension <= 4) | (dimension % 2 == 1)
    )
    return PfaffianResult(
        value=value,
        value_finite=valid_input & jnp.isfinite(raw_value),
        sign=sign,
        log_abs=log_abs,
        singular=prepared.singular,
        antisymmetry_residual=prepared.antisymmetry_residual,
        determinant_identity_residual=determinant_residual,
        determinant_identity_verified=determinant_verified,
        pivot_magnitudes=jnp.abs(prepared.pivots) * prepared.factor_scale[..., None],
        minimum_pivot=minimum_pivot,
        input_finite=prepared.input_finite,
        antisymmetric=prepared.antisymmetric,
        log_derivative_valid=log_derivative_valid,
        value_derivative_valid=value_derivative_valid,
        status=status,
        numeric_version=prepared.numeric_version,
        plan_id=prepared.plan.plan_id,
        prepared_id=prepared.prepared_id,
    )


def _as_matrix(matrix: ArrayLike, /) -> Array:
    value = jnp.asarray(matrix)
    if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
        raise ValueError("Pfaffian input must have shape (..., n, n).")
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        raise TypeError("Pfaffian input must have a floating or complex dtype.")
    return value


def _validate_plan_matrix(plan: PfaffianPlan, matrix: Array, /) -> None:
    expected = (*plan.batch_shape, plan.dimension, plan.dimension)
    if matrix.shape != expected:
        raise ValueError(
            f"Pfaffian plan expects shape {expected}, received {matrix.shape}."
        )
    if np.dtype(matrix.dtype) != plan.dtype:
        raise TypeError(
            f"Pfaffian plan expects dtype {plan.dtype}, received {matrix.dtype}."
        )


def _prepare_numeric(
    matrix: Array,
    plan: PfaffianPlan,
    /,
    *,
    numeric_version: Any,
) -> PreparedPfaffian:
    dimension = plan.dimension
    batch_shape = plan.batch_shape
    if dimension == 0:
        input_finite = jnp.ones(batch_shape, dtype=bool)
        matrix_scale = jnp.ones(batch_shape, dtype=matrix.real.dtype)
        factor_scale = matrix_scale
        residual = jnp.zeros(batch_shape, dtype=matrix.real.dtype)
    else:
        input_finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
        absolute_scale = jnp.max(jnp.abs(matrix), axis=(-2, -1))
        matrix_scale = jnp.where(
            input_finite,
            jnp.maximum(jnp.asarray(1, dtype=absolute_scale.dtype), absolute_scale),
            jnp.asarray(1, dtype=absolute_scale.dtype),
        )
        factor_scale = jnp.where(
            input_finite & (absolute_scale > 0.0),
            absolute_scale,
            jnp.asarray(1, dtype=absolute_scale.dtype),
        )
        factor_scale = jax.lax.stop_gradient(factor_scale)
        residual_raw = (
            jnp.max(
                jnp.abs(matrix + jnp.swapaxes(matrix, -1, -2)),
                axis=(-2, -1),
            )
            / matrix_scale
        )
        residual = jnp.where(input_finite, residual_raw, jnp.inf)
    roundoff_tolerance = 16.0 * max(dimension, 1) * jnp.finfo(matrix.real.dtype).eps
    antisymmetry_tolerance = jnp.maximum(
        jnp.asarray(
            plan.policy.antisymmetry_tolerance,
            dtype=residual.dtype,
        ),
        jnp.asarray(roundoff_tolerance, dtype=residual.dtype),
    )
    antisymmetric = input_finite & (residual <= antisymmetry_tolerance)
    effective = 0.5 * matrix - 0.5 * jnp.swapaxes(matrix, -1, -2)
    effective = jnp.where(input_finite[..., None, None], effective, 0)
    normalized = effective / factor_scale[..., None, None]
    threshold = (
        jnp.asarray(plan.policy.pivot_tolerance, dtype=matrix_scale.dtype)
        * matrix_scale
        / factor_scale
    )
    if dimension == 0:
        lower = normalized
        pivots = jnp.empty((*batch_shape, 0), dtype=matrix.dtype)
        permutation = jnp.empty((*batch_shape, 0), dtype=jnp.int32)
        swap_sign = jnp.ones(batch_shape, dtype=matrix.dtype)
        factor_singular = jnp.zeros(batch_shape, dtype=bool)
        minimum_pivot = jnp.full(
            batch_shape,
            jnp.inf,
            dtype=matrix.real.dtype,
        )
    else:
        (
            lower,
            pivots,
            permutation,
            swap_sign,
            factor_singular,
            minimum_pivot,
        ) = _factor_batch(normalized, threshold)
    singular = factor_singular | (dimension % 2 == 1)
    return PreparedPfaffian(
        plan=plan,
        matrix=effective,
        factor_scale=factor_scale,
        lower=lower,
        pivots=pivots,
        permutation=permutation,
        swap_sign=swap_sign,
        singular=singular,
        minimum_pivot=minimum_pivot * factor_scale,
        input_finite=input_finite,
        antisymmetry_residual=residual,
        antisymmetric=antisymmetric,
        numeric_version=numeric_version,
    )


def _factor_batch(
    matrix: Array,
    threshold: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    batch_shape = matrix.shape[:-2]
    dimension = matrix.shape[-1]
    batch_size = math.prod(batch_shape)
    flat_matrix = matrix.reshape((batch_size, dimension, dimension))
    flat_threshold = threshold.reshape((batch_size,))
    lower, pivots, permutation, swap_sign, singular, minimum_pivot = jax.vmap(
        _factor_one
    )(flat_matrix, flat_threshold)
    return (
        lower.reshape((*batch_shape, dimension, dimension)),
        pivots.reshape((*batch_shape, dimension // 2)),
        permutation.reshape((*batch_shape, dimension)),
        swap_sign.reshape(batch_shape),
        singular.reshape(batch_shape),
        minimum_pivot.reshape(batch_shape),
    )


def _factor_one(
    matrix: Array,
    threshold: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    dimension = matrix.shape[0]
    pair_count = dimension // 2
    lower = jnp.eye(dimension, dtype=matrix.dtype)
    pivots = jnp.zeros((pair_count,), dtype=matrix.dtype)
    permutation = jnp.arange(dimension, dtype=jnp.int32)
    swap_sign = jnp.asarray(1, dtype=matrix.dtype)
    singular = jnp.asarray(False)
    minimum_pivot = jnp.asarray(jnp.inf, dtype=matrix.real.dtype)
    coordinates = jnp.arange(dimension)

    def body(pair, state):
        work, factors, block_pivots, order, parity, failed, minimum = state
        first_index = 2 * pair
        second_index = first_index + 1
        candidates = jnp.where(
            coordinates > first_index,
            jnp.abs(work[first_index]),
            -jnp.inf,
        )
        pivot_index = jnp.argmax(candidates).astype(jnp.int32)
        work = _symmetric_swap(work, second_index, pivot_index)
        factors = _swap_factor_prefix(
            factors,
            second_index,
            pivot_index,
            first_index,
        )
        order = _swap_vector(order, second_index, pivot_index)
        parity = jnp.where(pivot_index == second_index, parity, -parity)
        pivot = work[first_index, second_index]
        pivot_magnitude = jnp.abs(pivot)
        pivot_good = jnp.isfinite(pivot_magnitude) & (pivot_magnitude > threshold)
        active = ~failed & pivot_good
        recorded_pivot = jnp.where(failed, jnp.zeros_like(pivot), pivot)
        block_pivots = block_pivots.at[pair].set(recorded_pivot)
        minimum = jnp.minimum(
            minimum,
            jnp.where(failed, jnp.asarray(0, dtype=minimum.dtype), pivot_magnitude),
        )
        safe_pivot = jnp.where(active, pivot, jnp.ones_like(pivot))
        first = work[first_index]
        second = work[second_index]
        trailing = coordinates > second_index
        first_column = jnp.where(trailing & active, -second / safe_pivot, 0)
        second_column = jnp.where(trailing & active, first / safe_pivot, 0)
        factors = factors.at[:, first_index].set(
            jnp.where(trailing, first_column, factors[:, first_index])
        )
        factors = factors.at[:, second_index].set(
            jnp.where(trailing, second_column, factors[:, second_index])
        )
        correction = (
            ein.contract("i,j->ij", second, first)
            - ein.contract("i,j->ij", first, second)
        ) / safe_pivot
        trailing_block = trailing[:, None] & trailing[None, :]
        work = jnp.where(trailing_block & active, work + correction, work)
        return (
            work,
            factors,
            block_pivots,
            order,
            parity,
            failed | ~pivot_good,
            minimum,
        )

    initial = (
        matrix,
        lower,
        pivots,
        permutation,
        swap_sign,
        singular,
        minimum_pivot,
    )
    _, lower, pivots, permutation, swap_sign, singular, minimum_pivot = jax.lax.fori_loop(
        0, pair_count, body, initial
    )
    return lower, pivots, permutation, swap_sign, singular, minimum_pivot


def _swap_vector(value: Array, first: Array | int, second: Array | int, /) -> Array:
    first_value = jax.lax.dynamic_index_in_dim(value, first, keepdims=False)
    second_value = jax.lax.dynamic_index_in_dim(value, second, keepdims=False)
    value = jax.lax.dynamic_update_index_in_dim(value, second_value, first, axis=0)
    return jax.lax.dynamic_update_index_in_dim(value, first_value, second, axis=0)


def _symmetric_swap(
    matrix: Array,
    first: Array | int,
    second: Array | int,
    /,
) -> Array:
    first_row = jax.lax.dynamic_index_in_dim(matrix, first, axis=0, keepdims=False)
    second_row = jax.lax.dynamic_index_in_dim(matrix, second, axis=0, keepdims=False)
    matrix = jax.lax.dynamic_update_index_in_dim(matrix, second_row, first, axis=0)
    matrix = jax.lax.dynamic_update_index_in_dim(matrix, first_row, second, axis=0)
    first_column = jax.lax.dynamic_index_in_dim(matrix, first, axis=1, keepdims=False)
    second_column = jax.lax.dynamic_index_in_dim(matrix, second, axis=1, keepdims=False)
    matrix = jax.lax.dynamic_update_index_in_dim(matrix, second_column, first, axis=1)
    return jax.lax.dynamic_update_index_in_dim(matrix, first_column, second, axis=1)


def _swap_factor_prefix(
    factors: Array,
    first: Array | int,
    second: Array | int,
    prefix_end: Array | int,
    /,
) -> Array:
    first_row = jax.lax.dynamic_index_in_dim(factors, first, axis=0, keepdims=False)
    second_row = jax.lax.dynamic_index_in_dim(factors, second, axis=0, keepdims=False)
    prefix = jnp.arange(factors.shape[1]) < prefix_end
    new_first = jnp.where(prefix, second_row, first_row)
    new_second = jnp.where(prefix, first_row, second_row)
    factors = jax.lax.dynamic_update_index_in_dim(factors, new_first, first, axis=0)
    return jax.lax.dynamic_update_index_in_dim(factors, new_second, second, axis=0)


def _signed_log(
    pivots: Array,
    swap_sign: Array,
    singular: Array,
    factor_scale: Array,
    /,
) -> tuple[Array, Array]:
    magnitudes = jnp.abs(pivots)
    safe_magnitudes = jnp.where(magnitudes > 0, magnitudes, 1)
    units = jnp.where(magnitudes > 0, pivots / safe_magnitudes, 1)
    sign = swap_sign * jnp.prod(units, axis=-1)
    log_abs = jnp.sum(jnp.log(safe_magnitudes), axis=-1)
    log_abs = log_abs + pivots.shape[-1] * jnp.log(factor_scale)
    sign = jnp.where(singular, jnp.zeros_like(sign), sign)
    log_abs = jnp.where(singular, -jnp.inf, log_abs)
    return sign, log_abs


@jax.custom_jvp
def _prepared_pfaffian_outputs(
    matrix: Array,
    lower: Array,
    factor_scale: Array,
    pivots: Array,
    permutation: Array,
    swap_sign: Array,
    singular: Array,
) -> tuple[Array, Array, Array]:
    del matrix, lower, permutation
    sign, log_abs = _signed_log(pivots, swap_sign, singular, factor_scale)
    value = jnp.where(singular, jnp.zeros_like(sign), sign * jnp.exp(log_abs))
    return value, sign, log_abs


@_prepared_pfaffian_outputs.defjvp
def _prepared_pfaffian_outputs_jvp(primals, tangents):
    matrix, lower, factor_scale, pivots, permutation, swap_sign, singular = primals
    matrix_tangent, _, _, _, _, _, _ = tangents
    sign, log_abs = _signed_log(pivots, swap_sign, singular, factor_scale)
    value = jnp.where(singular, jnp.zeros_like(sign), sign * jnp.exp(log_abs))
    dimension = matrix.shape[-1]
    if dimension == 0:
        value_tangent = jnp.zeros_like(value)
        sign_tangent = jnp.zeros_like(sign)
        log_tangent = jnp.zeros_like(log_abs)
    elif dimension % 2:
        value_tangent = jnp.zeros_like(value)
        sign_tangent = jnp.zeros_like(sign)
        log_tangent = jnp.full_like(log_abs, jnp.nan)
    else:
        scaled_tangent = matrix_tangent / factor_scale[..., None, None]
        half_trace = 0.5 * _trace_inverse_action(
            lower,
            pivots,
            permutation,
            scaled_tangent,
        )
        regular_value_tangent = value * half_trace
        singular_value_tangent = _small_singular_pfaffian_tangent(
            matrix,
            matrix_tangent,
        )
        valid = ~singular
        value_tangent = jnp.where(
            valid,
            regular_value_tangent,
            singular_value_tangent,
        )
        sign_tangent = sign * (half_trace - jnp.real(half_trace))
        log_tangent = jnp.real(half_trace)
        sign_tangent = jnp.where(valid, sign_tangent, jnp.nan)
        log_tangent = jnp.where(valid, log_tangent, jnp.nan)
    return (value, sign, log_abs), (
        value_tangent,
        sign_tangent,
        log_tangent,
    )


def _trace_inverse_action(
    lower: Array,
    pivots: Array,
    permutation: Array,
    tangent: Array,
    /,
) -> Array:
    """Apply the prepared solve to all tangent columns, then contract its trace."""
    inverse_action = _solve_factor(lower, pivots, permutation, tangent)
    return ein.contract("...ii->...", inverse_action)


def _small_singular_pfaffian_tangent(matrix: Array, tangent: Array, /) -> Array:
    dimension = matrix.shape[-1]
    skew_tangent = 0.5 * tangent - 0.5 * jnp.swapaxes(tangent, -1, -2)
    if dimension == 2:
        return skew_tangent[..., 0, 1]
    if dimension == 4:
        return (
            skew_tangent[..., 0, 1] * matrix[..., 2, 3]
            + matrix[..., 0, 1] * skew_tangent[..., 2, 3]
            - skew_tangent[..., 0, 2] * matrix[..., 1, 3]
            - matrix[..., 0, 2] * skew_tangent[..., 1, 3]
            + skew_tangent[..., 0, 3] * matrix[..., 1, 2]
            + matrix[..., 0, 3] * skew_tangent[..., 1, 2]
        )
    return jnp.full(matrix.shape[:-2], jnp.nan, dtype=matrix.dtype)


def _solve_factor(
    lower: Array,
    pivots: Array,
    permutation: Array,
    right_hand_side: Array,
    /,
) -> Array:
    matrix_rhs = right_hand_side.ndim == lower.ndim
    if matrix_rhs:
        indices = jnp.broadcast_to(
            permutation[..., :, None],
            right_hand_side.shape,
        )
        permuted = jnp.take_along_axis(right_hand_side, indices, axis=-2)
    else:
        permuted = jnp.take_along_axis(
            right_hand_side,
            permutation,
            axis=-1,
        )
    forward = jsp.linalg.solve_triangular(
        lower,
        permuted,
        lower=True,
        unit_diagonal=True,
    )
    safe_pivots = jnp.where(jnp.abs(pivots) > 0, pivots, 1)
    diagonal = jnp.empty_like(forward)
    if matrix_rhs:
        scaled_pivots = safe_pivots[..., :, None]
        diagonal = diagonal.at[..., 0::2, :].set(-forward[..., 1::2, :] / scaled_pivots)
        diagonal = diagonal.at[..., 1::2, :].set(forward[..., 0::2, :] / scaled_pivots)
    else:
        diagonal = diagonal.at[..., 0::2].set(-forward[..., 1::2] / safe_pivots)
        diagonal = diagonal.at[..., 1::2].set(forward[..., 0::2] / safe_pivots)
    backward = jsp.linalg.solve_triangular(
        jnp.swapaxes(lower, -1, -2),
        diagonal,
        lower=False,
        unit_diagonal=True,
    )
    inverse_permutation = jnp.argsort(permutation, axis=-1)
    if matrix_rhs:
        indices = jnp.broadcast_to(
            inverse_permutation[..., :, None],
            backward.shape,
        )
        return jnp.take_along_axis(backward, indices, axis=-2)
    return jnp.take_along_axis(backward, inverse_permutation, axis=-1)


def _determinant_identity_residual(
    matrix: Array,
    pfaffian_sign: Array,
    pfaffian_log_abs: Array,
    /,
) -> Array:
    if matrix.shape[-1] == 0:
        return jnp.zeros(matrix.shape[:-2], dtype=matrix.real.dtype)
    determinant_sign, determinant_log_abs = jnp.linalg.slogdet(matrix)
    finite_log = jnp.isfinite(determinant_log_abs)
    scale_log = jnp.maximum(
        jnp.asarray(0, dtype=determinant_log_abs.dtype),
        jnp.where(finite_log, determinant_log_abs, 0),
    )
    pfaffian_scaled = jnp.where(
        pfaffian_sign == 0,
        jnp.zeros_like(pfaffian_sign),
        pfaffian_sign * pfaffian_sign * jnp.exp(2 * pfaffian_log_abs - scale_log),
    )
    determinant_scaled = jnp.where(
        determinant_sign == 0,
        jnp.zeros_like(determinant_sign),
        determinant_sign * jnp.exp(determinant_log_abs - scale_log),
    )
    return jnp.abs(pfaffian_scaled - determinant_scaled)


__all__ = [
    "PfaffianPlan",
    "PfaffianPolicy",
    "PfaffianResult",
    "PfaffianStatus",
    "PreparedPfaffian",
    "SkewMode",
    "evaluate_pfaffian",
    "plan_pfaffian",
    "prepare_pfaffian",
    "refresh_pfaffian",
]
