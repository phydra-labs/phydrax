#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from phydrax.domain import DomainFunction
from phydrax.terms import ResidualPenalty

from ..operators.differential._requests import trace_derivative_requests
from ..optim._kfac._blocks import (
    AffineFactorObservation,
    BlockCurvatureObservation,
    estimate_kron_factors_from_chunks,
)
from ._functional_residual import (
    prepared_term_residual_vector,
    PreparedFunctionalResidual,
    PreparedResidualTerm,
)
from ._kfac_layout import ParameterLayout


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
        requests = trace_derivative_requests(term.condition.residual, functions)
        for request in requests:
            if request.order > 2:
                raise ValueError(
                    "KFAC supports residual derivatives through order two; "
                    f"field {request.field!r} requested order {request.order}."
                )



def _block_jacobian_chunks(
    flat_params: Array,
    unravel,
    non_trainable: PyTree[Any],
    solver,
    term: PreparedResidualTerm,
    indices: tuple[int, ...],
    /,
    *,
    chunk_size: int,
    iter_: Array | int | None,
    functional_residual: PreparedFunctionalResidual | None = None,
) -> Iterator[Array]:
    """Differentiate bounded residual chunks with respect to one parameter block."""

    index_array = jnp.asarray(indices, dtype=jnp.int32)
    block_params = jnp.take(flat_params, index_array)

    def residual_from_block(values):
        candidate = unravel(flat_params.at[index_array].set(values))
        if functional_residual is not None:
            blocks = functional_residual.blocks_for(candidate, term)
            pieces = tuple(block.values for block in blocks)
            return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)
        return prepared_term_residual_vector(
            candidate,
            non_trainable,
            solver.enforcement,
            term,
            iteration=iter_,
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
    terms: tuple[PreparedResidualTerm, ...],
    layout: ParameterLayout,
    /,
    *,
    approximation: Literal["expand", "reduce"],
    chunk_size: int,
    iter_: Array | int | None,
    functional_residual: PreparedFunctionalResidual | None = None,
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
                functional_residual=functional_residual,
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
                functional_residual=functional_residual,
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




__all__ = [
    "term_block_curvature_observations",
    "validate_derivative_coverage",
]
