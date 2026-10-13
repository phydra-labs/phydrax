#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from .._strict import StrictModule
from ._operators import AbstractLinearOperator
from ._results import LinearSolveStatus


@final
class InverseResidualCertificate(StrictModule):
    """All-direction, numerical inverse-residual and condition evidence."""

    status: Array
    residual_upper_bound: Array
    operator_norm_upper_bound: Array
    inverse_norm_upper_bound: Array
    condition_upper_bound: Array
    rank: Array
    valid: Array
    finite: Array
    operator_id: str = eqx.field(static=True)
    evidence: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)


def inverse_residual_certificate(
    operator: AbstractLinearOperator,
    inverse_action: Callable[[Array], tuple[Array, Array]],
    /,
    *,
    block_size: int = 8,
    condition_limit: float = 1e12,
    accumulation_dtype: DTypeLike | None = None,
) -> InverseResidualCertificate:
    """Stream columns of A, B, I-AB without retaining an inverse or identity.

    ``inverse_action`` accepts canonical ``(n, block_size)`` coordinate columns
    and returns approximate inverse columns and scalar or per-column validity.
    All columns are checked, including directions absent from the consumer RHS.
    The inverse action must preserve the supplied column order. Native
    ``solve_many`` composes directly with this contract through its diagnostics.
    The Neumann condition bound is n*||A||inf*||B||inf/(1-||I-AB||inf).
    Blocks use native fused actions and a scan with no archived outputs; working
    storage is O(n*block_size). Padding in the last block is never certified.
    """
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not callable(inverse_action):
        raise TypeError("inverse_action must be callable.")
    n = operator.source.size
    if n < 1 or operator.target.size != n or operator.batch_shape:
        raise ValueError(
            "Inverse residual certification requires an unbatched nonempty square operator."
        )
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    if not isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("condition_limit must be finite and exceed one.")
    source_dtypes = [spec.dtype for spec in jax.tree.leaves(operator.source.structure())]
    target_dtypes = [spec.dtype for spec in jax.tree.leaves(operator.target.structure())]
    dtype = jnp.result_type(*source_dtypes, *target_dtypes)
    action_real_dtype = jnp.empty((), dtype=dtype).real.dtype
    accumulation = jnp.dtype(dtype if accumulation_dtype is None else accumulation_dtype)
    if not jnp.issubdtype(accumulation, jnp.inexact):
        raise TypeError("accumulation_dtype must be an inexact dtype.")
    real_dtype = jnp.empty((), dtype=accumulation).real.dtype
    width = min(block_size, n)
    zero = jnp.zeros((n,), dtype=real_dtype)
    block_count = (n + width - 1) // width

    def block(
        carry: tuple[Array, Array, Array, Array], index: Array
    ) -> tuple[tuple[Array, Array, Array, Array], None]:
        a_rows, b_rows, error_rows, successful = carry
        columns = index * width + jnp.arange(width, dtype=jnp.int32)
        mask = columns < n
        basis = (jnp.arange(n, dtype=jnp.int32)[:, None] == columns[None, :]).astype(
            dtype
        )
        inverse_columns, valid = inverse_action(basis)
        inverse_columns = jnp.asarray(inverse_columns)
        valid = jnp.asarray(valid, dtype=jnp.bool_)
        if inverse_columns.shape != (n, width):
            raise ValueError("Inverse action must preserve the canonical block shape.")
        if valid.shape not in ((), (width,)):
            raise ValueError("Inverse action validity must be scalar or per column.")
        inverse_columns = jnp.where(mask[None, :], inverse_columns, 0.0)
        a_columns = operator.mv_block(basis)
        residual = basis - operator.mv_block(inverse_columns)
        return (
            a_rows + jnp.sum(jnp.abs(a_columns), axis=1, dtype=real_dtype),
            b_rows + jnp.sum(jnp.abs(inverse_columns), axis=1, dtype=real_dtype),
            error_rows + jnp.sum(jnp.abs(residual), axis=1, dtype=real_dtype),
            successful & jnp.all(~mask | valid),
        ), None

    (a_rows, b_rows, error_rows, successful), _ = jax.lax.scan(
        block,
        (zero, zero, zero, jnp.asarray(True)),
        jnp.arange(block_count, dtype=jnp.int32),
    )
    # Account conservatively for row-sum and residual-product rounding. This is
    # numerical verification, not an interval-arithmetic theorem for arbitrary
    # user-defined actions; failed or uncertain numerical evidence refuses.
    eps = jnp.asarray(jnp.finfo(action_real_dtype).eps, dtype=real_dtype)
    gamma = (n + 2) * eps / jnp.maximum(1.0 - (n + 2) * eps, eps)
    a_norm = jnp.max(a_rows) * (1.0 + gamma)
    b_norm = jnp.max(b_rows) * (1.0 + gamma)
    error = jnp.max(error_rows) * (1.0 + gamma) + gamma * a_norm * b_norm
    condition = n * a_norm * b_norm / jnp.maximum(1.0 - error, jnp.finfo(real_dtype).tiny)
    finite = (
        jnp.isfinite(error)
        & jnp.isfinite(condition)
        & jnp.isfinite(a_norm)
        & jnp.isfinite(b_norm)
    )
    valid = successful & finite & (error < 1.0) & (condition < condition_limit)
    status = jnp.where(
        ~finite,
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        jnp.where(
            ~successful | (error >= 1.0),
            int(LinearSolveStatus.RESIDUAL_TOO_LARGE),
            jnp.where(
                condition >= condition_limit,
                int(LinearSolveStatus.CONDITION_LIMIT_REACHED),
                int(LinearSolveStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    return InverseResidualCertificate(
        status,
        error,
        a_norm,
        b_norm,
        jnp.where(valid, condition, jnp.inf),
        jnp.where(valid, n, -1).astype(jnp.int32),
        valid,
        finite,
        operator_id=operator.operator_id,
        evidence="verified",
        scope="numerical",
        block_size=width,
        dimension=n,
    )


__all__ = ["InverseResidualCertificate", "inverse_residual_certificate"]
