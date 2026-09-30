#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Dynamic metric Gram spaces with native prepared Riesz solves."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike

from .._validation import canonical_identifier
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DiagonalPairing,
    DifferentiationPolicy,
    DualSpace,
    FailurePolicy,
    JacobiPreconditionerBuilder,
    LinearDerivativeSolvePolicy,
    LinearSolvePolicy,
    LinearSystem,
    OperatorPairing,
    OperatorProperties,
    PCG,
    PreconditioningPolicy,
    prepare,
    TolerancePolicy,
)
from ..sparse import EdgeRelation, SparseCoordinateOperator
from ..sparse._linear import _SparseStoragePlan


def gram_solve_policy(size: int, /, *, tolerance: float = 1e-10) -> LinearSolvePolicy:
    """Prepare native PCG/Jacobi with mathematical value and right-side derivatives."""
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("gram_tolerance must be finite, positive, and below one.")
    steps = max(1, 4 * size)
    return LinearSolvePolicy(
        PCG(),
        tolerance=TolerancePolicy(relative=tolerance, absolute=0.0, max_steps=steps),
        preconditioning=PreconditioningPolicy(JacobiPreconditionerBuilder()),
        differentiation=DifferentiationPolicy("mathematical"),
        derivative_solve=LinearDerivativeSolvePolicy(
            relative_tolerance=tolerance, absolute_tolerance=0.0, maximum_steps=steps
        ),
        failure=FailurePolicy("status"),
    )


def sparse_gram_space(
    targets: ArrayLike,
    sources: ArrayLike,
    values: ArrayLike,
    /,
    *,
    size: int,
    dtype: DTypeLike,
    space_id: str,
    gram_tolerance: float = 1e-10,
    policy: LinearSolvePolicy | None = None,
    storage_plan: _SparseStoragePlan | None = None,
) -> tuple[ArraySpace, AbstractLinearOperator]:
    """Bind a symmetric SPD metric's static routes and dynamic real values.

    Repeated routes accumulate. Symmetry and definiteness are the metric owner's
    admission contract. The numerical binding never moves values to the host.
    Prepared inverse status remains available through ``space.pairing``; an
    unsuccessful inverse Riesz action raises rather than returning a false result.
    """
    identifier = canonical_identifier(space_id, "space_id")
    coefficients = jnp.asarray(values)
    if coefficients.ndim != 1 or jnp.issubdtype(coefficients.dtype, jnp.complexfloating):
        raise ValueError("Gram route values must be one real vector.")
    coefficients = coefficients.astype(jnp.float64)
    relation = EdgeRelation(sources, targets, source_size=size, target_size=size)
    if storage_plan is None:
        with jax.ensure_compile_time_eval():
            admitted_relation = EdgeRelation(
                sources, targets, source_size=size, target_size=size
            )
            storage_plan = _SparseStoragePlan(admitted_relation)
    if coefficients.shape != relation.source_indices.shape:
        raise ValueError("Gram values must match the static route vector.")
    selected = (
        gram_solve_policy(size, tolerance=gram_tolerance) if policy is None else policy
    )
    if not isinstance(selected, LinearSolvePolicy):
        raise TypeError("policy must be a LinearSolvePolicy or None.")
    coefficients = eqx.error_if(
        coefficients, jnp.any(~jnp.isfinite(coefficients)), "Gram values must be finite."
    )
    coordinates = ArraySpace((size,), dtype=dtype, space_id=f"{identifier}:coordinates")
    gram = SparseCoordinateOperator(
        relation,
        coefficients,
        source=coordinates,
        target=coordinates,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id=f"{identifier}:gram",
        accumulation_dtype=dtype,
        storage_plan=storage_plan,
    )
    inverse = prepare(
        LinearSystem(gram, problem_id=f"{identifier}:gram-system"), selected
    )
    space = ArraySpace(
        (size,),
        dtype=dtype,
        pairing=OperatorPairing(
            gram, prepared_inverse=inverse, pairing_id=f"{identifier}:pairing"
        ),
        space_id=identifier,
    )
    mass = SparseCoordinateOperator(
        relation,
        coefficients,
        source=space,
        target=DualSpace(space),
        operator_id=f"{identifier}:mass",
        accumulation_dtype=dtype,
        storage_plan=storage_plan,
    )
    return space, mass


def diagonal_gram_space(
    measures: ArrayLike, /, *, dtype: DTypeLike, space_id: str
) -> tuple[ArraySpace, AbstractLinearOperator]:
    """Bind dynamic positive metric weights without host numerical admission."""
    identifier = canonical_identifier(space_id, "space_id")
    weights = jnp.asarray(measures)
    if weights.ndim != 1 or jnp.issubdtype(weights.dtype, jnp.complexfloating):
        raise ValueError("Gram diagonal must be one real vector.")
    weights = weights.astype(jnp.float64)
    weights = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights) | (weights <= 0.0)),
        "Gram diagonal must be finite and positive.",
    )
    size = weights.shape[0]
    cells = np.arange(size, dtype=np.int32)
    with jax.ensure_compile_time_eval():
        relation = EdgeRelation(cells, cells, source_size=size, target_size=size)
        storage_plan = _SparseStoragePlan(relation)
    space = ArraySpace(
        (size,),
        dtype=dtype,
        pairing=DiagonalPairing(weights, pairing_id=f"{identifier}:measure"),
        space_id=identifier,
    )
    mass = SparseCoordinateOperator(
        relation,
        weights,
        source=space,
        target=DualSpace(space),
        operator_id=f"{identifier}:mass",
        accumulation_dtype=dtype,
        storage_plan=storage_plan,
    )
    return space, mass


__all__ = ["diagonal_gram_space", "sparse_gram_space"]
