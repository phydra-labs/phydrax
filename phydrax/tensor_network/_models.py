#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._core import MatrixProductOperator
from ._environments import mpo_hermiticity_residual, mpo_norm
from ._mpo import add_mpo, scale_mpo
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
                "operators": array_tree_fingerprint(values),
                "coefficient": array_tree_fingerprint(coefficient_),
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


class PrefixQuadraticMPOEvidence(StrictModule):
    """Exact compact-MPO evidence for a weighted prefix-square operator."""

    hermiticity_residual: Array
    operator_scale: Array
    hermitian: Array
    site_count: int = eqx.field(static=True)
    active_prefix_count: int = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    builder_id: str = eqx.field(static=True)


class PrefixQuadraticMPOResult(StrictModule):
    operator: MatrixProductOperator
    evidence: PrefixQuadraticMPOEvidence


def _validated_dimensions(local_dimensions: Sequence[int], /) -> tuple[int, ...]:
    dimensions = tuple(local_dimensions)
    if not dimensions or any(dimension < 1 for dimension in dimensions):
        raise ValueError("local_dimensions must contain positive entries.")
    return dimensions


def build_local_term_mpo(
    local_dimensions: Sequence[int],
    terms: Sequence[FiniteLocalTerm],
    /,
    *,
    hermiticity_tolerance: float = 1e-10,
    precision: TensorNetworkPrecisionPolicy | None = None,
) -> FiniteMPOBuildResult:
    """Build the exact sum of bounded contiguous terms on a heterogeneous chain."""
    dimensions = _validated_dimensions(local_dimensions)
    sites = len(dimensions)
    tolerance = float(hermiticity_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("hermiticity_tolerance must be finite and nonnegative.")
    values = tuple(terms)
    if not values or any(not isinstance(term, FiniteLocalTerm) for term in values):
        raise ValueError("terms must contain at least one FiniteLocalTerm.")
    dtype = jnp.result_type(
        *(operator for term in values for operator in term.operators),
        *(term.coefficient for term in values),
    )
    identities = tuple(jnp.eye(dimension, dtype=dtype) for dimension in dimensions)
    summands: list[MatrixProductOperator] = []
    for term in values:
        if term.start + len(term.operators) > sites:
            raise ValueError("A local term extends beyond the finite chain.")
        for offset, operator in enumerate(term.operators):
            dimension = dimensions[term.start + offset]
            if operator.shape != (dimension, dimension):
                raise ValueError(
                    "Local-term dimensions must match their heterogeneous sites."
                )
        local = list(identities)
        for offset, operator in enumerate(term.operators):
            local[term.start + offset] = operator.astype(dtype)
        local[term.start] = term.coefficient.astype(dtype) * local[term.start]
        summands.append(
            MatrixProductOperator(
                tuple(matrix[None, :, :, None] for matrix in local),
                precision=precision,
            )
        )
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
                "local_dimensions": list(dimensions),
                "terms": tuple(term.term_id for term in values),
            }
        ),
    )
    return FiniteMPOBuildResult(operator, evidence)


def build_string_mpo(
    local_dimensions: Sequence[int],
    start: int,
    operators: Sequence[ArrayLike],
    /,
    *,
    coefficient: ArrayLike = 1.0,
    hermiticity_tolerance: float = 1e-10,
    precision: TensorNetworkPrecisionPolicy | None = None,
) -> FiniteMPOBuildResult:
    """Build one bounded operator string, padded by heterogeneous identities."""
    return build_local_term_mpo(
        local_dimensions,
        (FiniteLocalTerm(start, operators, coefficient=coefficient),),
        hermiticity_tolerance=hermiticity_tolerance,
        precision=precision,
    )


def build_prefix_quadratic_mpo(
    generators: Sequence[ArrayLike],
    prefix_offsets: ArrayLike,
    /,
    *,
    prefix_weights: ArrayLike | None = None,
    hermiticity_tolerance: float = 1e-10,
    precision: TensorNetworkPrecisionPolicy | None = None,
) -> PrefixQuadraticMPOResult:
    """Build ``sum_n w_n (offset_n + sum_{j<=n} Q_j)^2`` exactly."""
    operators = tuple(jnp.asarray(value) for value in generators)
    if not operators or any(
        value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 1
        for value in operators
    ):
        raise ValueError("generators must contain non-empty square matrices.")
    site_count = len(operators)
    offsets = jnp.asarray(prefix_offsets)
    weights = (
        jnp.ones((site_count,), dtype=jnp.real(offsets).dtype)
        if prefix_weights is None
        else jnp.asarray(prefix_weights)
    )
    if offsets.shape != (site_count,) or weights.shape != (site_count,):
        raise ValueError("prefix offsets and weights must provide one value per site.")
    if jnp.iscomplexobj(offsets) or jnp.iscomplexobj(weights):
        raise TypeError("Prefix offsets and weights must be real-valued.")
    if bool(jnp.any(~jnp.isfinite(offsets))) or bool(
        jnp.any(~jnp.isfinite(weights) | (weights < 0.0))
    ):
        raise ValueError("Prefix offsets and non-negative weights must be finite.")
    tolerance = float(hermiticity_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("hermiticity_tolerance must be finite and non-negative.")
    for operator in operators:
        residual = jnp.max(jnp.abs(operator - jnp.conj(operator.T)))
        if not bool(jnp.all(jnp.isfinite(operator))) or bool(residual > tolerance):
            raise ValueError("Every prefix generator must be finite and Hermitian.")
    dtype = jnp.result_type(*operators, offsets, weights, 1j)
    operators = tuple(value.astype(dtype) for value in operators)
    offsets = offsets.astype(jnp.real(jnp.zeros((), dtype=dtype)).dtype)
    weights = weights.astype(offsets.dtype)
    suffix_weight = jnp.flip(jnp.cumsum(jnp.flip(weights)))
    suffix_offset = 2.0 * jnp.flip(jnp.cumsum(jnp.flip(weights * offsets)))
    constant = jnp.sum(weights * offsets**2)
    tensors = []
    for site, operator in enumerate(operators):
        dimension = operator.shape[0]
        identity = jnp.eye(dimension, dtype=dtype)
        local = (
            suffix_weight[site] * (operator @ operator) + suffix_offset[site] * operator
        )
        if site == 0:
            local = local + constant * identity
        if site_count == 1:
            tensors.append(local[None, :, :, None])
            continue
        tensor = jnp.zeros((3, dimension, dimension, 3), dtype=dtype)
        tensor = tensor.at[0, :, :, 0].set(identity)
        tensor = tensor.at[0, :, :, 1].set(operator)
        tensor = tensor.at[0, :, :, 2].set(local)
        tensor = tensor.at[1, :, :, 1].set(identity)
        tensor = tensor.at[1, :, :, 2].set(2.0 * suffix_weight[site] * operator)
        tensor = tensor.at[2, :, :, 2].set(identity)
        if site == 0:
            tensors.append(tensor[0:1])
        elif site == site_count - 1:
            tensors.append(tensor[..., 2:3])
        else:
            tensors.append(tensor)
    operator = MatrixProductOperator(tuple(tensors), precision=precision)
    residual = mpo_hermiticity_residual(operator)
    scale = mpo_norm(operator)
    builder_id = canonical_fingerprint(
        {
            "kind": "prefix-quadratic-mpo",
            "generators": tuple(array_tree_fingerprint(value) for value in operators),
            "offsets": array_tree_fingerprint(offsets),
            "weights": array_tree_fingerprint(weights),
            "precision": operator.precision.policy_id,
        }
    )
    evidence = PrefixQuadraticMPOEvidence(
        hermiticity_residual=residual,
        operator_scale=scale,
        hermitian=jnp.isfinite(residual) & (residual <= tolerance),
        site_count=site_count,
        active_prefix_count=int(jnp.count_nonzero(weights)),
        maximum_bond_dimension=max((1,) + operator.bond_dimensions),
        builder_id=builder_id,
    )
    return PrefixQuadraticMPOResult(operator=operator, evidence=evidence)


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
        self.step_count = values.shape[0] - 1
        self.basis_count = len(basis)
        self.structure_id = canonical_fingerprint(
            {
                "kind": "fixed-structure-mpo-coefficients",
                "basis": tuple(operator.structure_id for operator in basis),
                "steps": values.shape[0] - 1,
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
    "PrefixQuadraticMPOEvidence",
    "PrefixQuadraticMPOResult",
    "build_local_term_mpo",
    "build_prefix_quadratic_mpo",
    "build_string_mpo",
]
