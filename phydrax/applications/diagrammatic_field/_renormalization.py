#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from itertools import product
from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator


class CountertermContract(Protocol):
    """Structural contract for finite local counterterm contributions."""

    counterterm_id: str
    perturbative_order: int

    def evaluate(self, variables: ArrayLike, /) -> Array: ...


class RenormalizationSchemeContract(Protocol):
    """Structural contract fixing scale, subtraction point, and polynomial degree."""

    scheme_id: str
    scale: Array
    subtraction_point: Array
    subtraction_degree: int


class MomentumSubtractionScheme(StrictModule, NonTrainableState):
    """Finite momentum-subtraction conditions at one Euclidean point."""

    name: str = eqx.field(static=True)
    scale: Array
    subtraction_point: Array
    subtraction_degree: int = eqx.field(static=True)
    scheme_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        scale: float,
        subtraction_point: ArrayLike,
        subtraction_degree: int,
        /,
    ):
        name_ = str(name).strip()
        scale_ = float(scale)
        point = np.asarray(subtraction_point, dtype=np.float64)
        degree = int(subtraction_degree)
        if not name_:
            raise ValueError("Renormalization scheme name must be non-empty.")
        if not np.isfinite(scale_) or scale_ <= 0.0:
            raise ValueError("Renormalization scale must be finite and positive.")
        if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
            raise ValueError("subtraction_point must be a finite non-empty vector.")
        if degree < 0:
            raise ValueError("subtraction_degree must be non-negative.")
        self.name = name_
        self.scale = jnp.asarray(scale_)
        self.subtraction_point = jnp.asarray(point)
        self.subtraction_degree = degree
        self.scheme_id = canonical_fingerprint(
            {
                "kind": "momentum-subtraction-scheme",
                "name": name_,
                "scale": scale_,
                "point": array_tree_fingerprint(point),
                "degree": degree,
            }
        )


class PolynomialCounterterm(StrictModule, NonTrainableState):
    """Sparse local polynomial counterterm in declared invariant variables."""

    exponents: Array
    coefficients: Array
    variable_count: int = eqx.field(static=True)
    term_count: int = eqx.field(static=True)
    perturbative_order: int = eqx.field(static=True)
    counterterm_id: str = eqx.field(static=True)

    def __init__(
        self,
        exponents: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        perturbative_order: int,
        maximum_terms: int = 4_096,
    ):
        powers = np.asarray(exponents)
        values = np.asarray(coefficients, dtype=np.complex128)
        order, maximum = int(perturbative_order), int(maximum_terms)
        if powers.ndim != 2 or powers.shape[0] == 0:
            raise ValueError("exponents must be a non-empty rank-two array.")
        if powers.shape[0] > maximum or maximum <= 0:
            raise ValueError("Counterterm terms exceed maximum_terms.")
        if values.shape != (powers.shape[0],):
            raise ValueError("coefficients must provide one value per exponent row.")
        if not np.issubdtype(powers.dtype, np.integer) or np.any(powers < 0):
            raise ValueError("Counterterm exponents must be non-negative integers.")
        if not np.all(np.isfinite(values)) or order <= 0:
            raise ValueError(
                "Counterterm coefficients must be finite and order positive."
            )
        self.exponents = jnp.asarray(powers, dtype=jnp.int32)
        self.coefficients = jnp.asarray(values)
        self.variable_count = powers.shape[1]
        self.term_count = powers.shape[0]
        self.perturbative_order = order
        self.counterterm_id = canonical_fingerprint(
            {
                "kind": "polynomial-counterterm",
                "exponents": array_tree_fingerprint(powers),
                "coefficients": array_tree_fingerprint(values),
                "order": order,
            }
        )

    def evaluate(self, variables: ArrayLike, /) -> Array:
        values = jnp.asarray(variables)
        if values.shape[-1:] != (self.variable_count,):
            raise ValueError("Counterterm variables have the wrong trailing dimension.")
        monomials = jnp.prod(
            values[..., None, :] ** self.exponents,
            axis=-1,
        )
        return jnp.sum(monomials * self.coefficients, axis=-1)


class BPHZSubtractionEvidence(StrictModule, NonTrainableState):
    subtraction_conditions: Array
    maximum_condition_residual: Array
    finite: Array
    successful: Array
    status: Array


class BPHZSubtractionResult(StrictModule, NonTrainableState):
    coefficients: Array
    removed_coefficients: Array
    evidence: BPHZSubtractionEvidence
    scheme_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def evaluate(self, variables: ArrayLike, /) -> Array:
        values = jnp.asarray(variables)
        variable_count = self.coefficients.ndim
        if values.shape[-1:] != (variable_count,):
            raise ValueError("Polynomial variables have the wrong trailing dimension.")
        result = jnp.zeros(values.shape[:-1], dtype=self.coefficients.dtype)
        for index in np.ndindex(self.coefficients.shape):
            result = result + self.coefficients[index] * jnp.prod(
                values ** jnp.asarray(index)
            )
        return result


class BPHZSubtractionPlan(StrictModule, NonTrainableState):
    """Immutable polynomial shape and momentum-subtraction resource policy."""

    scheme: MomentumSubtractionScheme
    polynomial_shape: tuple[int, ...] = eqx.field(static=True)
    maximum_terms: int = eqx.field(static=True)
    maximum_matrix_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scheme: MomentumSubtractionScheme,
        polynomial_shape: tuple[int, ...],
        /,
        *,
        maximum_terms: int = 4_096,
        maximum_matrix_elements: int = 1_000_000,
    ):
        if not isinstance(scheme, MomentumSubtractionScheme):
            raise TypeError("scheme must be a MomentumSubtractionScheme.")
        shape = tuple(polynomial_shape)
        maximum = int(maximum_terms)
        maximum_matrix = int(maximum_matrix_elements)
        if len(shape) != scheme.subtraction_point.size or any(
            value <= 0 for value in shape
        ):
            raise ValueError("Polynomial shape must have one positive axis per variable.")
        term_count = math.prod(shape)
        if maximum <= 0 or term_count > maximum:
            raise ValueError("Polynomial basis exceeds maximum_terms before allocation.")
        if maximum_matrix <= 0 or term_count * term_count > maximum_matrix:
            raise ValueError(
                "BPHZ subtraction map exceeds maximum_matrix_elements before allocation."
            )
        self.scheme = scheme
        self.polynomial_shape = shape
        self.maximum_terms = maximum
        self.maximum_matrix_elements = maximum_matrix
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bphz-subtraction-plan",
                "scheme": scheme.scheme_id,
                "shape": shape,
                "maximum_terms": maximum,
                "maximum_matrix_elements": maximum_matrix,
            }
        )

    def prepare(self, /) -> "PreparedBPHZSubtraction":
        shape = self.polynomial_shape
        indices = tuple(np.ndindex(shape))
        lookup = {index: position for position, index in enumerate(indices)}
        point = np.asarray(self.scheme.subtraction_point)
        taylor = np.zeros((len(indices), len(indices)), dtype=np.float64)
        condition_indices = tuple(
            index for index in indices if sum(index) <= self.scheme.subtraction_degree
        )
        conditions = np.zeros((len(condition_indices), len(indices)), dtype=np.float64)

        for column, alpha in enumerate(indices):
            beta_ranges = tuple(range(power + 1) for power in alpha)
            for beta in product(*beta_ranges):
                if sum(beta) > self.scheme.subtraction_degree:
                    continue
                coefficient = 1.0
                for axis, (a, b) in enumerate(zip(alpha, beta, strict=True)):
                    coefficient *= math.comb(a, b) * point[axis] ** (a - b)
                gamma_ranges = tuple(range(power + 1) for power in beta)
                for gamma in product(*gamma_ranges):
                    expanded = coefficient
                    for axis, (b, g) in enumerate(zip(beta, gamma, strict=True)):
                        expanded *= math.comb(b, g) * (-point[axis]) ** (b - g)
                    taylor[lookup[gamma], column] += expanded
            for row, beta in enumerate(condition_indices):
                if all(a >= b for a, b in zip(alpha, beta, strict=True)):
                    derivative = 1.0
                    for axis, (a, b) in enumerate(zip(alpha, beta, strict=True)):
                        derivative *= (
                            math.factorial(a)
                            / math.factorial(a - b)
                            * point[axis] ** (a - b)
                        )
                    conditions[row, column] = derivative

        subtraction = np.eye(len(indices)) - taylor
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-bphz-subtraction",
                "plan": self.plan_id,
                "subtraction": array_tree_fingerprint(subtraction),
            }
        )
        return PreparedBPHZSubtraction(
            self,
            DenseLinearOperator(
                jnp.asarray(subtraction, dtype=jnp.complex128),
                operator_id=f"{prepared_id}:subtraction",
            ),
            jnp.asarray(conditions, dtype=jnp.complex128),
            prepared_id,
        )


class PreparedBPHZSubtraction(StrictModule, NonTrainableState):
    """Prepared native linear subtraction and renormalization conditions."""

    plan: BPHZSubtractionPlan
    subtraction_operator: DenseLinearOperator
    condition_matrix: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: BPHZSubtractionPlan,
        subtraction_operator: DenseLinearOperator,
        condition_matrix: Array,
        prepared_id: str,
        /,
    ):
        self.plan = plan
        self.subtraction_operator = subtraction_operator
        self.condition_matrix = condition_matrix
        self.prepared_id = str(prepared_id)

    def subtract(self, coefficients: ArrayLike, /) -> BPHZSubtractionResult:
        original = jnp.asarray(coefficients)
        if original.shape != self.plan.polynomial_shape:
            raise ValueError("coefficients do not match the prepared polynomial shape.")
        if not jnp.issubdtype(original.dtype, jnp.inexact):
            original = original.astype("float64")
        flattened = original.reshape((-1,)).astype(jnp.complex128)
        subtracted = self.subtraction_operator(flattened)
        removed = flattened - subtracted
        conditions = self.condition_matrix @ subtracted
        maximum = jnp.max(jnp.abs(conditions))
        finite = jnp.all(jnp.isfinite(subtracted)) & jnp.all(jnp.isfinite(conditions))
        tolerance = (
            64.0
            * jnp.finfo(subtracted.real.dtype).eps
            * jnp.maximum(1.0, jnp.max(jnp.abs(flattened)))
        )
        successful = finite & (maximum <= tolerance)
        evidence = BPHZSubtractionEvidence(
            conditions,
            maximum,
            finite,
            successful,
            jnp.where(successful, 0, 1).astype(jnp.int32),
        )
        return BPHZSubtractionResult(
            subtracted.reshape(self.plan.polynomial_shape),
            removed.reshape(self.plan.polynomial_shape),
            evidence,
            self.plan.scheme.scheme_id,
            self.prepared_id,
        )


def renormalized_local_amplitude(
    bare_amplitude: ArrayLike,
    counterterms: tuple[CountertermContract, ...],
    variables: ArrayLike,
    /,
) -> Array:
    """Add explicitly supplied local counterterms without hidden scheme defaults."""
    amplitude = jnp.asarray(bare_amplitude)
    for counterterm in counterterms:
        amplitude = amplitude + counterterm.evaluate(variables)
    return amplitude


__all__ = [
    "BPHZSubtractionEvidence",
    "BPHZSubtractionPlan",
    "BPHZSubtractionResult",
    "CountertermContract",
    "MomentumSubtractionScheme",
    "PolynomialCounterterm",
    "PreparedBPHZSubtraction",
    "RenormalizationSchemeContract",
    "renormalized_local_amplitude",
]
