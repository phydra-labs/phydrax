#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from phydrax._trainable import combine_trainable
from phydrax.domain import DomainFunction
from phydrax.integration import IntegrationRealization
from phydrax.operators.differential._runtime import derivative_runtime_context
from phydrax.terms import ResidualPenalty

from ..optim._kfac._blocks import (
    AffineFactorObservation,
    BlockCurvatureObservation,
    estimate_kron_factors_from_chunks,
)
from ._functional_objective import _PreparedObjective
from ._kfac_derivative_requests import trace_derivative_requests
from ._kfac_layout import ParameterLayout


@dataclass(frozen=True, slots=True)
class FrozenResidualTerm:
    """One materialized ResidualPenalty reused for curvature, gradient, and search."""

    term: ResidualPenalty
    realization: IntegrationRealization
    scale: float = 1.0


def materialize_frozen_terms(
    prepared: _PreparedObjective,
    /,
) -> tuple[FrozenResidualTerm, ...]:
    """Lower one prepared objective to KFAC residual-root realizations."""
    frozen: list[FrozenResidualTerm] = []
    for prepared_term in prepared.terms:
        term = prepared_term.term
        if not isinstance(term, ResidualPenalty):
            raise TypeError(
                "KFAC supports ResidualPenalty training terms only; "
                f"got {type(term).__name__}."
            )
        if prepared_term.payload_kind != "realization" or not isinstance(
            prepared_term.payload,
            IntegrationRealization,
        ):
            raise TypeError(
                "KFAC residual terms require a prepared IntegrationRealization."
            )
        realization = prepared_term.payload
        frozen.append(
            FrozenResidualTerm(
                term=term,
                realization=realization,
                scale=float(prepared.selection.scale),
            )
        )
    return tuple(frozen)


def validate_derivative_coverage(
    terms: Sequence[Any],
    functions: dict[str, DomainFunction] | Any,
    /,
) -> None:
    """Reject training terms and derivatives outside KFAC's support boundary."""

    for term in terms:
        if not isinstance(term, ResidualPenalty):
            raise TypeError(
                "KFAC supports ResidualPenalty training terms only; "
                f"got {type(term).__name__}."
            )
        trace_derivative_requests(term.condition.residual, functions)


def _scaled_residual_data(data, /, *, scale: float) -> Array:
    pieces: list[Array] = []
    for residual, coefficient in zip(
        data.residuals,
        data.coefficients,
        strict=True,
    ):
        root = cx.Field(
            jnp.sqrt(float(scale) * jnp.asarray(coefficient.data)),
            dims=coefficient.dims,
        )
        scaled = root * residual
        values = jnp.asarray(scaled.data)
        pieces.append(jnp.real(values).reshape((-1,)))
        if jnp.iscomplexobj(values):
            pieces.append(jnp.imag(values).reshape((-1,)))
    if not pieces:
        return jnp.zeros((0,), dtype=float)
    return jnp.concatenate(tuple(pieces), axis=0)


def frozen_term_residual_vector(
    params: PyTree[Any],
    non_trainable: PyTree[Any],
    solver,
    term: FrozenResidualTerm,
    /,
    *,
    iter_: Array | int | None,
) -> Array:
    """Evaluate square-root-weighted residual roots for one frozen term."""

    functions = combine_trainable(params, non_trainable)
    enforced = (
        functions if solver.enforcement is None else solver.enforcement.apply(functions)
    )
    with derivative_runtime_context():
        data = term.term._quadratic_residual_data(
            enforced,
            realization=term.realization,
            iter_=iter_,
        )
    return _scaled_residual_data(data, scale=term.scale)


def _block_jacobian_chunks(
    flat_params: Array,
    unravel,
    non_trainable: PyTree[Any],
    solver,
    term: FrozenResidualTerm,
    indices: tuple[int, ...],
    /,
    *,
    chunk_size: int,
    iter_: Array | int | None,
) -> Iterator[Array]:
    """Differentiate bounded residual chunks with respect to one parameter block."""

    index_array = jnp.asarray(indices, dtype=jnp.int32)
    block_params = jnp.take(flat_params, index_array)

    def residual_from_block(values):
        candidate = flat_params.at[index_array].set(values)
        return frozen_term_residual_vector(
            unravel(candidate),
            non_trainable,
            solver,
            term,
            iter_=iter_,
        )

    residual_size = int(residual_from_block(block_params).size)
    if residual_size == 0:
        raise ValueError("KFAC requires every active term to yield residual roots.")
    padded_size = ((residual_size + int(chunk_size) - 1) // int(chunk_size)) * int(
        chunk_size
    )

    def residual_chunk(values, start):
        residual = residual_from_block(values)
        padded = jnp.pad(residual, (0, padded_size - residual_size))
        return jax.lax.dynamic_slice_in_dim(
            padded,
            start,
            int(chunk_size),
        )

    chunk_jacobian = jax.jacrev(residual_chunk, argnums=0)
    for start in range(0, residual_size, int(chunk_size)):
        valid_size = min(int(chunk_size), residual_size - start)
        chunk = chunk_jacobian(
            block_params,
            jnp.asarray(start, dtype=jnp.int32),
        )
        yield chunk[:valid_size]


def term_block_curvature_observations(
    params: PyTree[Any],
    non_trainable: PyTree[Any],
    solver,
    terms: tuple[FrozenResidualTerm, ...],
    layout: ParameterLayout,
    /,
    *,
    approximation: Literal["expand", "reduce"],
    chunk_size: int,
    iter_: Array | int | None,
) -> tuple[Array, tuple[BlockCurvatureObservation, ...]]:
    """Extract block-local type-II GGN observations without a global Jacobian."""

    if int(chunk_size) <= 0:
        raise ValueError("factor_chunk_size must be positive.")
    if approximation not in ("expand", "reduce"):
        raise ValueError("approximation must be either 'expand' or 'reduce'.")
    flat_params, unravel = ravel_pytree(params)
    observations: list[BlockCurvatureObservation] = []
    for term in terms:
        affine: list[AffineFactorObservation] = []
        for block in layout.affine_blocks:
            chunks = _block_jacobian_chunks(
                flat_params,
                unravel,
                non_trainable,
                solver,
                term,
                block.indices,
                chunk_size=chunk_size,
                iter_=iter_,
            )
            activation, sensitivity = estimate_kron_factors_from_chunks(
                chunks,
                block,
                approximation=approximation,
            )
            affine.append(AffineFactorObservation(activation, sensitivity))

        uncovered_spec = layout.uncovered_block
        if uncovered_spec is None:
            uncovered = None
        else:
            chunks = _block_jacobian_chunks(
                flat_params,
                unravel,
                non_trainable,
                solver,
                term,
                uncovered_spec.indices,
                chunk_size=chunk_size,
                iter_=iter_,
            )
            if uncovered_spec.approximation == "exact":
                uncovered = sum(
                    (chunk.T @ chunk for chunk in chunks),
                    jnp.zeros(
                        (uncovered_spec.parameter_count,) * 2,
                        dtype=flat_params.dtype,
                    ),
                )
                uncovered = 0.5 * (uncovered + uncovered.T)
            else:
                uncovered = sum(
                    (jnp.sum(jnp.square(chunk), axis=0) for chunk in chunks),
                    jnp.zeros(
                        (uncovered_spec.parameter_count,),
                        dtype=flat_params.dtype,
                    ),
                )
        observations.append(BlockCurvatureObservation(tuple(affine), uncovered))
    return flat_params, tuple(observations)


def term_residual_jacobians(
    params: PyTree[Any],
    non_trainable: PyTree[Any],
    solver,
    terms: tuple[FrozenResidualTerm, ...],
    /,
    *,
    iter_: Array | int | None,
) -> tuple[Array, tuple[Array, ...], Any]:
    """Differentiate each residual term separately through its complete ansatz graph."""

    flat_params, unravel = ravel_pytree(params)
    jacobians: list[Array] = []
    for term in terms:

        def residual_from_flat(flat):
            return frozen_term_residual_vector(
                unravel(flat),
                non_trainable,
                solver,
                term,
                iter_=iter_,
            )

        jacobian = jax.jacrev(residual_from_flat)(flat_params)
        jacobians.append(jnp.asarray(jacobian).reshape((-1, int(flat_params.size))))
    return flat_params, tuple(jacobians), unravel


def frozen_loss(
    params: PyTree[Any],
    non_trainable: PyTree[Any],
    solver,
    terms: tuple[FrozenResidualTerm, ...],
    /,
    *,
    iter_: Array | int | None,
) -> Array:
    """Evaluate the total loss while reusing every materialized term."""

    functions = combine_trainable(params, non_trainable)
    enforced = (
        functions if solver.enforcement is None else solver.enforcement.apply(functions)
    )
    total = jnp.asarray(0.0, dtype=float)
    with derivative_runtime_context():
        for term in terms:
            data = term.term._quadratic_residual_data(
                enforced,
                realization=term.realization,
                iter_=iter_,
            )
            total = total + float(term.scale) * jnp.asarray(data.loss).reshape(())
    return total


def frozen_loss_and_flat_gradient(
    params: PyTree[Any],
    non_trainable: PyTree[Any],
    solver,
    terms: tuple[FrozenResidualTerm, ...],
    /,
    *,
    iter_: Array | int | None,
) -> tuple[Array, Array, Any]:
    """Return frozen-batch loss and gradient in the shared flat parameter order."""

    flat_params, unravel = ravel_pytree(params)

    def loss_from_flat(flat):
        return frozen_loss(
            unravel(flat),
            non_trainable,
            solver,
            terms,
            iter_=iter_,
        )

    loss, gradient = jax.value_and_grad(loss_from_flat)(flat_params)
    return loss, gradient, unravel


__all__ = [
    "FrozenResidualTerm",
    "frozen_loss",
    "frozen_loss_and_flat_gradient",
    "frozen_term_residual_vector",
    "materialize_frozen_terms",
    "term_block_curvature_observations",
    "term_residual_jacobians",
    "validate_derivative_coverage",
]
