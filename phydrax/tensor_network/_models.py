#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._core import MatrixProductOperator
from ._environments import mpo_hermiticity_residual, mpo_norm
from ._mpo import add_mpo, product_mpo, scale_mpo
from ._precision import TensorNetworkPrecisionPolicy


class FiniteLocalTerm(StrictModule):
    """A contiguous finite-chain operator term with an explicit coefficient."""

    operators: tuple[Array, ...]
    coefficient: Array
    start: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        start: int,
        operators: Sequence[ArrayLike],
        /,
        *,
        coefficient: ArrayLike = 1.0,
    ):
        start_ = int(start)
        values = tuple(jnp.asarray(value) for value in operators)
        coefficient_ = jnp.asarray(coefficient)
        if start_ < 0:
            raise ValueError("Local-term start must be nonnegative.")
        if not values or any(
            value.ndim != 2 or value.shape[0] != value.shape[1] for value in values
        ):
            raise ValueError("Local-term operators must be nonempty square matrices.")
        if coefficient_.ndim != 0:
            raise ValueError("Local-term coefficient must be scalar.")
        self.operators = values
        self.coefficient = coefficient_
        self.start = start_
        self.term_id = canonical_fingerprint(
            {
                "kind": "finite-local-term",
                "start": start_,
                "shapes": tuple(
                    tuple(int(size) for size in value.shape) for value in values
                ),
                "dtypes": tuple(str(value.dtype) for value in values),
            }
        )


class FiniteMPOBuildEvidence(StrictModule):
    hermiticity_residual: Array
    operator_scale: Array
    hermitian: Array
    term_count: int = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    builder_id: str = eqx.field(static=True)


class FiniteMPOBuildResult(StrictModule):
    operator: MatrixProductOperator
    evidence: FiniteMPOBuildEvidence


def _validated_dimension(site_count: int, local_dimension: int, /) -> tuple[int, int]:
    sites = int(site_count)
    dimension = int(local_dimension)
    if sites < 1 or dimension < 1:
        raise ValueError("site_count and local_dimension must be positive.")
    return sites, dimension


def build_local_term_mpo(
    site_count: int,
    local_dimension: int,
    terms: Sequence[FiniteLocalTerm],
    /,
    *,
    hermiticity_tolerance: float = 1e-10,
    precision: TensorNetworkPrecisionPolicy | None = None,
) -> FiniteMPOBuildResult:
    """Build the exact sum of bounded contiguous terms as an open-boundary MPO."""
    sites, dimension = _validated_dimension(site_count, local_dimension)
    tolerance = float(hermiticity_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("hermiticity_tolerance must be finite and nonnegative.")
    values = tuple(terms)
    if not values or any(not isinstance(term, FiniteLocalTerm) for term in values):
        raise ValueError("terms must contain at least one FiniteLocalTerm.")
    identity = jnp.eye(
        dimension, dtype=jnp.result_type(*(term.operators[0] for term in values))
    )
    summands = []
    for term in values:
        if term.start + len(term.operators) > sites:
            raise ValueError("A local term extends beyond the finite chain.")
        if any(operator.shape != (dimension, dimension) for operator in term.operators):
            raise ValueError("Local-term dimensions must match local_dimension.")
        local = [identity for _ in range(sites)]
        for offset, operator in enumerate(term.operators):
            local[term.start + offset] = operator
        local[term.start] = term.coefficient * local[term.start]
        summands.append(product_mpo(jnp.stack(local), precision=precision))
    operator = summands[0]
    for summand in summands[1:]:
        operator = add_mpo(operator, summand)
    residual = mpo_hermiticity_residual(operator)
    scale = mpo_norm(operator)
    evidence = FiniteMPOBuildEvidence(
        residual,
        scale,
        jnp.isfinite(residual) & (residual <= tolerance),
        len(values),
        max((1,) + operator.bond_dimensions),
        canonical_fingerprint(
            {
                "kind": "finite-local-term-mpo",
                "sites": sites,
                "dimension": dimension,
                "terms": tuple(term.term_id for term in values),
            }
        ),
    )
    return FiniteMPOBuildResult(operator, evidence)


def build_string_mpo(
    site_count: int,
    local_dimension: int,
    start: int,
    operators: Sequence[ArrayLike],
    /,
    *,
    coefficient: ArrayLike = 1.0,
    hermiticity_tolerance: float = 1e-10,
    precision: TensorNetworkPrecisionPolicy | None = None,
) -> FiniteMPOBuildResult:
    """Build one bounded operator string, padded by identities."""
    return build_local_term_mpo(
        site_count,
        local_dimension,
        (FiniteLocalTerm(start, operators, coefficient=coefficient),),
        hermiticity_tolerance=hermiticity_tolerance,
        precision=precision,
    )


class FixedStructureMPOCoefficients(StrictModule):
    """A finite coefficient table over a fixed ordered MPO basis."""

    basis_operators: tuple[MatrixProductOperator, ...]
    coefficients: Array
    step_count: int = eqx.field(static=True)
    basis_count: int = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis_operators: Sequence[MatrixProductOperator],
        coefficients: ArrayLike,
        /,
    ):
        basis = tuple(basis_operators)
        values = jnp.asarray(coefficients)
        if not basis or any(
            not isinstance(operator, MatrixProductOperator) for operator in basis
        ):
            raise ValueError("basis_operators must be nonempty MPO values.")
        reference = basis[0]
        for operator in basis[1:]:
            if (
                operator.output_dimensions != reference.output_dimensions
                or operator.input_dimensions != reference.input_dimensions
                or operator.precision.policy_id != reference.precision.policy_id
            ):
                raise ValueError(
                    "Fixed-structure MPO basis dimensions and precision must match."
                )
        if values.ndim != 2 or values.shape[1] != len(basis) or values.shape[0] < 1:
            raise ValueError("coefficients require shape (steps + 1, basis_count).")
        self.basis_operators = basis
        self.coefficients = values
        self.step_count = int(values.shape[0] - 1)
        self.basis_count = len(basis)
        self.structure_id = canonical_fingerprint(
            {
                "kind": "fixed-structure-mpo-coefficients",
                "basis": tuple(operator.structure_id for operator in basis),
                "steps": int(values.shape[0] - 1),
                "dtype": str(values.dtype),
            }
        )

    def operator_at(self, step: int, /) -> MatrixProductOperator:
        step_ = int(step)
        if not 0 <= step_ <= self.step_count:
            raise ValueError("Coefficient step is outside the fixed schedule.")
        operator = scale_mpo(self.basis_operators[0], self.coefficients[step_, 0])
        for index in range(1, self.basis_count):
            operator = add_mpo(
                operator,
                scale_mpo(self.basis_operators[index], self.coefficients[step_, index]),
            )
        return operator


__all__ = [
    "FiniteLocalTerm",
    "FiniteMPOBuildEvidence",
    "FiniteMPOBuildResult",
    "FixedStructureMPOCoefficients",
    "build_local_term_mpo",
    "build_string_mpo",
]
