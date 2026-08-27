#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from operator import index
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import AbstractFiniteRealAlgebraSpec


class AlgebraSymmetryBudget(StrictModule, NonTrainableState):
    maximum_constraint_equations: int = eqx.field(static=True)
    maximum_constraint_nonzeros: int = eqx.field(static=True)
    maximum_materialized_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_basis_bytes: int = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_constraint_equations: int = 32_768,
        maximum_constraint_nonzeros: int = 1_000_000,
        maximum_materialized_bytes: int = 64 * 1024**2,
        maximum_workspace_bytes: int = 128 * 1024**2,
        maximum_basis_bytes: int = 16 * 1024**2,
    ):
        names = (
            "maximum_constraint_equations",
            "maximum_constraint_nonzeros",
            "maximum_materialized_bytes",
            "maximum_workspace_bytes",
            "maximum_basis_bytes",
        )
        raw = (
            maximum_constraint_equations,
            maximum_constraint_nonzeros,
            maximum_materialized_bytes,
            maximum_workspace_bytes,
            maximum_basis_bytes,
        )
        if any(isinstance(value, bool) for value in raw):
            raise TypeError("Algebra symmetry resource limits must be integers.")
        values = tuple(index(value) for value in raw)
        if any(value <= 0 for value in values):
            raise ValueError("Algebra symmetry resource limits must be positive.")
        for name, value in zip(names, values, strict=True):
            setattr(self, name, value)
        self.budget_id = canonical_fingerprint(
            {
                "kind": "algebra-symmetry-budget-v1",
                **dict(zip(names, values, strict=True)),
            }
        )

    def admit_constraint(self, equations: int, nonzeros: int, /) -> None:
        equation_count = index(equations)
        nonzero_count = index(nonzeros)
        if equation_count < 0 or equation_count > self.maximum_constraint_equations:
            raise ValueError("Algebra derivation equation budget exceeded.")
        if nonzero_count < 0 or nonzero_count > self.maximum_constraint_nonzeros:
            raise ValueError("Algebra derivation nonzero budget exceeded.")

    def admit_materialization(
        self,
        matrix_bytes: int,
        workspace_bytes: int,
        basis_bytes: int,
        /,
    ) -> None:
        matrix = index(matrix_bytes)
        workspace = index(workspace_bytes)
        basis = index(basis_bytes)
        if matrix < 0 or matrix > self.maximum_materialized_bytes:
            raise ValueError("Algebra derivation materialization budget exceeded.")
        if workspace < 0 or workspace > self.maximum_workspace_bytes:
            raise ValueError("Algebra derivation workspace budget exceeded.")
        if basis < 0 or basis > self.maximum_basis_bytes:
            raise ValueError("Algebra derivation basis budget exceeded.")


class AlgebraSymmetryResourceEvidence(StrictModule, NonTrainableState):
    equation_count: int = eqx.field(static=True)
    variable_count: int = eqx.field(static=True)
    nonzero_count: int = eqx.field(static=True)
    sparse_bytes: int = eqx.field(static=True)
    materialized_bytes: int = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        equation_count: int,
        variable_count: int,
        nonzero_count: int,
        sparse_bytes: int,
        materialized_bytes: int,
        budget: AlgebraSymmetryBudget,
    ):
        if not isinstance(budget, AlgebraSymmetryBudget):
            raise TypeError("budget must be an AlgebraSymmetryBudget.")
        values = tuple(
            index(value)
            for value in (
                equation_count,
                variable_count,
                nonzero_count,
                sparse_bytes,
                materialized_bytes,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("Algebra symmetry resource evidence must be nonnegative.")
        budget.admit_constraint(values[0], values[2])
        if values[4] > budget.maximum_materialized_bytes:
            raise ValueError("Algebra derivation materialization budget exceeded.")
        (
            self.equation_count,
            self.variable_count,
            self.nonzero_count,
            self.sparse_bytes,
            self.materialized_bytes,
        ) = values
        self.budget_id = budget.budget_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "algebra-symmetry-resources-v1",
                "equations": values[0],
                "variables": values[1],
                "nonzeros": values[2],
                "sparse_bytes": values[3],
                "materialized_bytes": values[4],
                "budget": budget.budget_id,
            }
        )


class AlgebraDerivationConstraint(StrictModule, NonTrainableState):
    """Exact sparse Leibniz constraints for real algebra derivation matrices."""

    algebra: AbstractFiniteRealAlgebraSpec
    budget: AlgebraSymmetryBudget
    row_indices: tuple[int, ...] = eqx.field(static=True)
    column_indices: tuple[int, ...] = eqx.field(static=True)
    coefficient_numerators: tuple[int, ...] = eqx.field(static=True)
    coefficient_denominators: tuple[int, ...] = eqx.field(static=True)
    equation_count: int = eqx.field(static=True)
    variable_count: int = eqx.field(static=True)
    resources: AlgebraSymmetryResourceEvidence
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: AbstractFiniteRealAlgebraSpec,
        /,
        *,
        budget: AlgebraSymmetryBudget | None = None,
    ):
        if not isinstance(algebra, AbstractFiniteRealAlgebraSpec):
            raise TypeError("algebra must implement AbstractFiniteRealAlgebraSpec.")
        budget_ = AlgebraSymmetryBudget() if budget is None else budget
        if not isinstance(budget_, AlgebraSymmetryBudget):
            raise TypeError("budget must be AlgebraSymmetryBudget or None.")
        dimension = algebra.coordinate_dimension
        equations = dimension**3
        variables = dimension**2
        coefficients: dict[tuple[int, int], Fraction] = {}

        def add(row: int, column: int, value: Fraction) -> None:
            key = (row, column)
            coefficients[key] = coefficients.get(key, Fraction(0)) + value

        for left, right, output, numerator, denominator in algebra.structure.terms:
            coefficient = Fraction(numerator, denominator)
            for result in range(dimension):
                first_row = (left * dimension + right) * dimension + result
                first_column = result * dimension + output
                add(first_row, first_column, coefficient)

                second_row = (result * dimension + right) * dimension + output
                second_column = left * dimension + result
                add(second_row, second_column, -coefficient)

                third_row = (left * dimension + result) * dimension + output
                third_column = right * dimension + result
                add(third_row, third_column, -coefficient)

        normalized = tuple(
            (row, column, value)
            for (row, column), value in sorted(coefficients.items())
            if value
        )
        budget_.admit_constraint(equations, len(normalized))
        sparse_bytes = len(normalized) * 4 * 8
        materialized_bytes = equations * variables * 8
        resources = AlgebraSymmetryResourceEvidence(
            equation_count=equations,
            variable_count=variables,
            nonzero_count=len(normalized),
            sparse_bytes=sparse_bytes,
            materialized_bytes=materialized_bytes,
            budget=budget_,
        )
        self.algebra = algebra
        self.budget = budget_
        self.row_indices = tuple(row for row, _, _ in normalized)
        self.column_indices = tuple(column for _, column, _ in normalized)
        self.coefficient_numerators = tuple(value.numerator for _, _, value in normalized)
        self.coefficient_denominators = tuple(
            value.denominator for _, _, value in normalized
        )
        self.equation_count = equations
        self.variable_count = variables
        self.resources = resources
        self.constraint_id = canonical_fingerprint(
            {
                "kind": "algebra-derivation-constraint-v1",
                "algebra": algebra.algebra_id,
                "rows": self.row_indices,
                "columns": self.column_indices,
                "numerators": self.coefficient_numerators,
                "denominators": self.coefficient_denominators,
                "resources": resources.evidence_id,
            }
        )

    @property
    def matrix_shape(self) -> tuple[int, int]:
        dimension = self.algebra.coordinate_dimension
        return dimension, dimension

    def materialize(self, dtype: Any = np.float64, /) -> Array:
        dtype_ = np.dtype(dtype)
        if not np.issubdtype(dtype_, np.floating):
            raise TypeError("Algebra derivation constraints require floating dtype.")
        itemsize = dtype_.itemsize
        matrix_bytes = self.equation_count * self.variable_count * itemsize
        self.budget.admit_materialization(matrix_bytes, 0, 0)
        rows = jnp.asarray(self.row_indices, dtype=jnp.int32)
        columns = jnp.asarray(self.column_indices, dtype=jnp.int32)
        coefficients = jnp.asarray(
            self.coefficient_numerators, dtype=dtype_
        ) / jnp.asarray(
            self.coefficient_denominators,
            dtype=dtype_,
        )
        return (
            jnp.zeros(
                (self.equation_count, self.variable_count),
                dtype=dtype_,
            )
            .at[rows, columns]
            .add(coefficients)
        )


__all__ = [
    "AlgebraDerivationConstraint",
    "AlgebraSymmetryBudget",
    "AlgebraSymmetryResourceEvidence",
]
