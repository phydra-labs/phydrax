#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._strict import StrictModule
from ._dense_inverse import dense_inverse


class MatrixContinuedFractionStatus(IntEnum):
    """Per-shift status for one matrix-valued continued fraction."""

    SUCCESS = 0
    SINGULAR = 1
    NONFINITE = 2


class MatrixContinuedFractionDiagnostics(StrictModule):
    """Per-shift inversion evidence for a matrix continued fraction."""

    finite: Array
    singular_level: Array
    nonfinite_level: Array
    maximum_relative_inverse_residual: Array
    coefficients_finite: Array
    block_count: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)


class MatrixContinuedFractionProvenance(StrictModule):
    """Algorithm and ordering convention for a matrix continued fraction."""

    method: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    coupling: str = eqx.field(static=True)
    termination: str = eqx.field(static=True)


class MatrixContinuedFractionResult(StrictModule):
    """Leading resolvent block of a finite or explicitly terminated block chain."""

    value: Array
    shifts: Array
    terminal_self_energy: Array
    status: Array
    diagnostics: MatrixContinuedFractionDiagnostics
    provenance: MatrixContinuedFractionProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(MatrixContinuedFractionStatus.SUCCESS)

    @property
    def all_successful(self) -> Array:
        return jnp.all(self.successful)


def matrix_continued_fraction(
    diagonal_blocks: ArrayLike,
    upper_couplings: ArrayLike | None,
    shifts: ArrayLike,
    /,
    *,
    lower_couplings: ArrayLike | None = None,
    terminal_self_energy: ArrayLike | None = None,
) -> MatrixContinuedFractionResult:
    """Evaluate a block-Jacobi resolvent by a matrix continued fraction.

    For diagonal blocks ``D_k`` and ordered couplings ``U_k`` and ``L_k``, the
    backward recurrence is ``G_k = (z I - D_k - U_k G_(k+1) L_k)^-1``.  Omitting
    ``lower_couplings`` sets ``L_k = U_k^H``.  ``terminal_self_energy`` is the
    complete Schur-complement correction subtracted at the deepest retained block.
    """
    (
        diagonal,
        upper,
        lower,
        shift_values,
        terminal,
        coupling_convention,
    ) = _validated_inputs(
        diagonal_blocks,
        upper_couplings,
        shifts,
        lower_couplings=lower_couplings,
        terminal_self_energy=terminal_self_energy,
    )
    block_count, block_size, _ = diagonal.shape
    identity = jnp.eye(block_size, dtype=diagonal.dtype)
    padded_upper = jnp.concatenate((upper, identity[None, ...]), axis=0)
    padded_lower = jnp.concatenate((lower, identity[None, ...]), axis=0)

    coefficients_finite = (
        jnp.all(jnp.isfinite(diagonal))
        & jnp.all(jnp.isfinite(upper))
        & jnp.all(jnp.isfinite(lower))
    )
    lane_input_finite = (
        jnp.isfinite(shift_values)
        & jnp.all(jnp.isfinite(terminal), axis=(-2, -1))
        & coefficients_finite
    )
    safe_shifts = jnp.where(lane_input_finite, shift_values, 0)
    safe_terminal = jnp.where(
        lane_input_finite[..., None, None],
        terminal,
        jnp.zeros_like(terminal),
    )
    safe_diagonal = jnp.where(
        coefficients_finite,
        diagonal,
        jnp.zeros_like(diagonal),
    )
    safe_upper = jnp.where(
        coefficients_finite,
        padded_upper,
        jnp.zeros_like(padded_upper),
    )
    safe_lower = jnp.where(
        coefficients_finite,
        padded_lower,
        jnp.zeros_like(padded_lower),
    )

    (
        raw_value,
        singular_level,
        nonfinite_level,
        arithmetic_finite,
        maximum_inverse_residual,
    ) = _evaluate_matrix_fraction(
        safe_diagonal,
        safe_upper,
        safe_lower,
        safe_shifts,
        safe_terminal,
        lane_input_finite,
    )
    finite = (
        lane_input_finite
        & arithmetic_finite
        & jnp.all(jnp.isfinite(raw_value), axis=(-2, -1))
    )
    value = jnp.where(
        finite[..., None, None],
        raw_value,
        jnp.full(raw_value.shape, jnp.nan, dtype=raw_value.dtype),
    )
    status = jnp.full(
        shift_values.shape,
        int(MatrixContinuedFractionStatus.SUCCESS),
        dtype=jnp.int32,
    )
    status = jnp.where(
        ~finite | (nonfinite_level >= 0),
        int(MatrixContinuedFractionStatus.NONFINITE),
        status,
    )
    status = jnp.where(
        singular_level >= 0,
        int(MatrixContinuedFractionStatus.SINGULAR),
        status,
    )
    residual = jnp.where(
        finite,
        maximum_inverse_residual,
        jnp.asarray(jnp.nan, dtype=maximum_inverse_residual.dtype),
    )
    diagnostics = MatrixContinuedFractionDiagnostics(
        finite=finite,
        singular_level=singular_level,
        nonfinite_level=nonfinite_level,
        maximum_relative_inverse_residual=residual,
        coefficients_finite=coefficients_finite,
        block_count=block_count,
        block_size=block_size,
    )
    provenance = MatrixContinuedFractionProvenance(
        method="backward-block-schur-complement",
        convention="shift-identity-minus-block-tridiagonal",
        coupling=coupling_convention,
        termination=(
            "finite-zero-self-energy"
            if terminal_self_energy is None
            else "explicit-terminal-self-energy"
        ),
    )
    return MatrixContinuedFractionResult(
        value=value,
        shifts=shift_values,
        terminal_self_energy=terminal,
        status=status,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _evaluate_matrix_fraction(
    diagonal: Array,
    padded_upper: Array,
    padded_lower: Array,
    shifts: Array,
    terminal_self_energy: Array,
    input_finite: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    block_count, block_size, _ = diagonal.shape
    identity = jnp.eye(block_size, dtype=diagonal.dtype)
    identity_batch = jnp.broadcast_to(identity, terminal_self_energy.shape)
    real_dtype = diagonal.real.dtype
    initial_level = jnp.full(shifts.shape, -1, dtype=jnp.int32)
    state = (
        terminal_self_energy,
        input_finite,
        initial_level,
        initial_level,
        jnp.zeros(shifts.shape, dtype=real_dtype),
    )
    tiny = jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype)
    identity_norm = jnp.asarray(jnp.sqrt(block_size), dtype=real_dtype)

    def step(offset, current):
        next_value, healthy, singular_level, nonfinite_level, max_residual = current
        index = block_count - 1 - offset
        correction = contract(
            "ij,...jk,kl->...il",
            padded_upper[index],
            next_value,
            padded_lower[index],
        )
        denominator = shifts[..., None, None] * identity - diagonal[index] - correction
        denominator_finite = jnp.all(jnp.isfinite(denominator), axis=(-2, -1))
        safe_for_factorization = jnp.where(
            denominator_finite[..., None, None],
            denominator,
            identity_batch,
        )
        determinant_sign, log_abs_determinant = jnp.linalg.slogdet(safe_for_factorization)
        determinant_finite = jnp.isfinite(determinant_sign) & jnp.isfinite(
            log_abs_determinant
        )
        singular_here = healthy & denominator_finite & (determinant_sign == 0)
        valid_factorization = (
            healthy & denominator_finite & (determinant_sign != 0) & determinant_finite
        )
        safe_denominator = jnp.where(
            valid_factorization[..., None, None],
            denominator,
            identity_batch,
        )
        candidate = dense_inverse(safe_denominator)
        candidate_finite = jnp.all(jnp.isfinite(candidate), axis=(-2, -1))
        valid_candidate = valid_factorization & candidate_finite
        nonfinite_here = healthy & (
            ~denominator_finite
            | ((determinant_sign != 0) & ~determinant_finite)
            | (valid_factorization & ~candidate_finite)
        )
        safe_candidate = jnp.where(
            candidate_finite[..., None, None],
            candidate,
            jnp.zeros_like(candidate),
        )
        product = contract("...ij,...jk->...ik", safe_denominator, safe_candidate)
        inverse_residual = jnp.linalg.norm(
            product - identity,
            axis=(-2, -1),
        )
        residual_scale = (
            jnp.linalg.norm(safe_denominator, axis=(-2, -1))
            * jnp.linalg.norm(safe_candidate, axis=(-2, -1))
            + identity_norm
        )
        relative_residual = inverse_residual / jnp.maximum(residual_scale, tiny)
        relative_residual = jnp.where(valid_candidate, relative_residual, 0)
        next_residual = jnp.maximum(max_residual, relative_residual)
        next_singular_level = jnp.where(
            singular_here,
            jnp.asarray(index, dtype=jnp.int32),
            singular_level,
        )
        next_nonfinite_level = jnp.where(
            nonfinite_here,
            jnp.asarray(index, dtype=jnp.int32),
            nonfinite_level,
        )
        value = jnp.where(
            valid_candidate[..., None, None],
            safe_candidate,
            jnp.zeros_like(safe_candidate),
        )
        return (
            value,
            healthy & valid_candidate,
            next_singular_level,
            next_nonfinite_level,
            next_residual,
        )

    value, healthy, singular_level, nonfinite_level, max_residual = jax.lax.fori_loop(
        0,
        block_count,
        step,
        state,
    )
    return value, singular_level, nonfinite_level, healthy, max_residual


def _validated_inputs(
    diagonal_blocks: ArrayLike,
    upper_couplings: ArrayLike | None,
    shifts: ArrayLike,
    /,
    *,
    lower_couplings: ArrayLike | None,
    terminal_self_energy: ArrayLike | None,
) -> tuple[Array, Array, Array, Array, Array, str]:
    diagonal = jnp.asarray(diagonal_blocks)
    if (
        diagonal.ndim != 3
        or diagonal.shape[0] < 1
        or diagonal.shape[1] < 1
        or diagonal.shape[1] != diagonal.shape[2]
    ):
        raise ValueError(
            "diagonal_blocks must have shape (block_count, block_size, block_size)."
        )
    _validate_numeric(diagonal, "diagonal_blocks")
    block_count, block_size, _ = diagonal.shape
    coupling_shape = (block_count - 1, block_size, block_size)
    if upper_couplings is None:
        if block_count != 1:
            raise ValueError("upper_couplings may be None only for one block.")
        upper = jnp.empty(coupling_shape, dtype=diagonal.dtype)
    else:
        upper = jnp.asarray(upper_couplings)
        if upper.shape != coupling_shape:
            raise ValueError(f"upper_couplings must have shape {coupling_shape}.")
        _validate_numeric(upper, "upper_couplings")

    if lower_couplings is None:
        lower = jnp.conj(jnp.swapaxes(upper, -1, -2))
        coupling_convention = "none" if block_count == 1 else "adjoint-paired-upper-lower"
    else:
        lower = jnp.asarray(lower_couplings)
        if lower.shape != coupling_shape:
            raise ValueError(f"lower_couplings must have shape {coupling_shape}.")
        _validate_numeric(lower, "lower_couplings")
        coupling_convention = "explicit-upper-lower"

    shift_values = jnp.asarray(shifts)
    if shift_values.ndim not in (0, 1) or (
        shift_values.ndim == 1 and shift_values.size < 1
    ):
        raise ValueError("shifts must be a scalar or one nonempty rank-one array.")
    _validate_numeric(shift_values, "shifts")
    terminal_shape = shift_values.shape + (block_size, block_size)
    if terminal_self_energy is None:
        terminal = jnp.zeros(terminal_shape, dtype=shift_values.dtype)
    else:
        terminal = jnp.asarray(terminal_self_energy)
        if terminal.shape == (block_size, block_size):
            terminal = jnp.broadcast_to(terminal, terminal_shape)
        elif terminal.shape != terminal_shape:
            raise ValueError(
                "terminal_self_energy must have shape (block_size, block_size) "
                "or shifts.shape + (block_size, block_size)."
            )
        _validate_numeric(terminal, "terminal_self_energy")

    dtype = jnp.result_type(
        diagonal.dtype,
        upper.dtype,
        lower.dtype,
        shift_values.dtype,
        terminal.dtype,
    )
    return (
        diagonal.astype(dtype),
        upper.astype(dtype),
        lower.astype(dtype),
        shift_values.astype(dtype),
        terminal.astype(dtype),
        coupling_convention,
    )


def _validate_numeric(value: Array, name: str, /) -> None:
    if not jnp.issubdtype(value.dtype, jnp.number) or jnp.issubdtype(
        value.dtype, jnp.bool_
    ):
        raise TypeError(f"{name} must contain real or complex numbers.")


__all__ = [
    "MatrixContinuedFractionDiagnostics",
    "MatrixContinuedFractionProvenance",
    "MatrixContinuedFractionResult",
    "MatrixContinuedFractionStatus",
    "matrix_continued_fraction",
]
