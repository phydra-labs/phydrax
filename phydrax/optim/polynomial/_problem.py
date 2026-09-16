#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...algebraic._system import SparsePolynomialSystem


def _validate_real_system(system: SparsePolynomialSystem, name: str, /) -> None:
    if not isinstance(system, SparsePolynomialSystem):
        raise TypeError(f"{name} must be a SparsePolynomialSystem.")
    if jnp.issubdtype(system.coefficients.dtype, jnp.complexfloating):
        raise TypeError(f"{name} coefficients must be real-valued for optimization.")


def _equation_terms(
    system: SparsePolynomialSystem, equation_index: int, /
) -> tuple[np.ndarray, Array]:
    equation_indices = np.asarray(system.support.equation_indices)
    selection = np.flatnonzero(equation_indices == equation_index)
    exponents = np.asarray(system.support.exponents)[selection]
    coefficients = system.coefficients[jnp.asarray(selection, dtype=jnp.int32)]
    return exponents, coefficients


def _equation_degrees(system: SparsePolynomialSystem, /) -> tuple[int, ...]:
    degrees: list[int] = []
    for equation in range(system.support.equation_count):
        exponents, _ = _equation_terms(system, equation)
        degrees.append(
            int(np.max(np.sum(exponents, axis=1))) if exponents.shape[0] else 0
        )
    return tuple(degrees)


class PolynomialOptimizationProblem(StrictModule):
    """Scalar polynomial minimization with ``h(x) = 0`` and ``g(x) >= 0``."""

    objective: SparsePolynomialSystem
    equalities: SparsePolynomialSystem | None
    inequalities: SparsePolynomialSystem | None
    variable_labels: tuple[str, ...] = eqx.field(static=True)
    variable_count: int = eqx.field(static=True)
    equality_count: int = eqx.field(static=True)
    inequality_count: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        objective: SparsePolynomialSystem,
        /,
        *,
        equalities: SparsePolynomialSystem | None = None,
        inequalities: SparsePolynomialSystem | None = None,
    ):
        _validate_real_system(objective, "objective")
        if objective.support.equation_count != 1:
            raise ValueError("objective must contain exactly one polynomial equation.")
        if equalities is not None:
            _validate_real_system(equalities, "equalities")
        if inequalities is not None:
            _validate_real_system(inequalities, "inequalities")
        labels = tuple(objective.support.variable_labels)
        for name, system in (
            ("equalities", equalities),
            ("inequalities", inequalities),
        ):
            if system is not None and tuple(system.support.variable_labels) != labels:
                raise ValueError(
                    f"{name} must use the objective's ordered variable labels."
                )
        equality_count = 0 if equalities is None else equalities.support.equation_count
        inequality_count = (
            0 if inequalities is None else inequalities.support.equation_count
        )
        structure = canonical_fingerprint(
            {
                "kind": "polynomial-optimization-structure",
                "objective_support": objective.support.support_id,
                "equality_support": (
                    None if equalities is None else equalities.support.support_id
                ),
                "inequality_support": (
                    None if inequalities is None else inequalities.support.support_id
                ),
                "inequality_convention": "nonnegative",
            }
        )
        self.objective = objective
        self.equalities = equalities
        self.inequalities = inequalities
        self.variable_labels = labels
        self.variable_count = objective.support.variable_count
        self.equality_count = equality_count
        self.inequality_count = inequality_count
        self.structure_id = structure
        self.problem_id = canonical_fingerprint(
            {
                "kind": "polynomial-optimization-problem",
                "structure": structure,
                "objective": objective.system_id,
                "equalities": None if equalities is None else equalities.system_id,
                "inequalities": (
                    None if inequalities is None else inequalities.system_id
                ),
            }
        )

    def objective_value(self, point: Any, /) -> Array:
        return self.objective.evaluate(point)[..., 0]

    def equality_values(self, point: Any, /) -> Array:
        values = jnp.asarray(point)
        if self.equalities is None:
            return jnp.empty(values.shape[:-1] + (0,), dtype=values.dtype)
        return self.equalities.evaluate(values)

    def inequality_values(self, point: Any, /) -> Array:
        values = jnp.asarray(point)
        if self.inequalities is None:
            return jnp.empty(values.shape[:-1] + (0,), dtype=values.dtype)
        return self.inequalities.evaluate(values)


__all__ = ["PolynomialOptimizationProblem"]
