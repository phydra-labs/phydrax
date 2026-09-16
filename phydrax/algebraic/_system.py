#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._grading import PolynomialVariableGroup


_INT32_MAX = int(np.iinfo(np.int32).max)


def _labels(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    labels = tuple(str(value).strip() for value in values)
    if not labels or any(not value for value in labels):
        raise ValueError(f"{name} must contain non-empty labels.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} must be unique.")
    return labels


def _host_nonnegative_integer_array(value: Any, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers.")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative.")
    if array.size and int(np.max(array)) > _INT32_MAX:
        raise ValueError(f"{name} exceeds the supported compiled integer range.")
    return np.asarray(array, dtype=np.int32)


def _canonical_order(
    equations: np.ndarray,
    exponents: np.ndarray,
    /,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(int(equations.shape[0])),
            key=lambda term: (
                int(equations[term]),
                *(int(value) for value in exponents[term]),
            ),
        )
    )


def _validate_groups(
    groups: Sequence[PolynomialVariableGroup],
    variable_count: int,
    equation_count: int,
    equations: np.ndarray,
    exponents: np.ndarray,
    /,
) -> tuple[PolynomialVariableGroup, ...]:
    groups_ = tuple(groups)
    if any(not isinstance(group, PolynomialVariableGroup) for group in groups_):
        raise TypeError("groups must contain PolynomialVariableGroup values.")
    labels = tuple(group.label for group in groups_)
    if len(set(labels)) != len(labels):
        raise ValueError("Polynomial variable-group labels must be unique.")
    assigned: set[int] = set()
    for group in groups_:
        if group.variable_indices[-1] >= variable_count:
            raise ValueError(
                "A polynomial variable group references an unknown variable."
            )
        overlap = assigned.intersection(group.variable_indices)
        if overlap:
            raise ValueError("Polynomial variable groups must be disjoint.")
        assigned.update(group.variable_indices)
        if group.geometry != "projective":
            continue
        for equation in range(equation_count):
            rows = exponents[equations == equation]
            degrees = {
                sum(int(row[index]) for index in group.variable_indices) for row in rows
            }
            if len(degrees) > 1:
                raise ValueError(
                    f"Equation {equation} is not homogeneous in projective group "
                    f"{group.label!r}."
                )
    return groups_


class SparsePolynomialSupport(StrictModule, NonTrainableState):
    """Canonical unpadded COO support for a labelled polynomial system."""

    equation_indices: Array
    exponents: Array
    variable_labels: tuple[str, ...] = eqx.field(static=True)
    equation_labels: tuple[str, ...] = eqx.field(static=True)
    groups: tuple[PolynomialVariableGroup, ...]
    variable_count: int = eqx.field(static=True)
    equation_count: int = eqx.field(static=True)
    term_count: int = eqx.field(static=True)
    canonical_term_permutation: tuple[int, ...] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        variable_labels: Sequence[str],
        equation_labels: Sequence[str],
        equation_indices: Any,
        exponents: Any,
        /,
        *,
        groups: Sequence[PolynomialVariableGroup] = (),
    ):
        variables = _labels(variable_labels, "variable_labels")
        equations_labels = _labels(equation_labels, "equation_labels")
        equation_array = _host_nonnegative_integer_array(
            equation_indices, "equation_indices"
        )
        exponent_array = _host_nonnegative_integer_array(exponents, "exponents")
        if equation_array.ndim != 1:
            raise ValueError("equation_indices must have shape (term_count,).")
        if exponent_array.ndim != 2:
            raise ValueError("exponents must have shape (term_count, variable_count).")
        if exponent_array.shape != (equation_array.shape[0], len(variables)):
            raise ValueError(
                "exponents must have one row per term and one column per variable."
            )
        if equation_array.size and int(np.max(equation_array)) >= len(equations_labels):
            raise ValueError("equation_indices references an unknown equation.")
        if any(
            not np.any(equation_array == index) for index in range(len(equations_labels))
        ):
            raise ValueError(
                "Every equation must have at least one declared support term."
            )
        order = _canonical_order(equation_array, exponent_array)
        canonical_equations = equation_array[np.asarray(order, dtype=np.intp)]
        canonical_exponents = exponent_array[np.asarray(order, dtype=np.intp)]
        term_keys = tuple(
            (int(equation), *(int(value) for value in exponent))
            for equation, exponent in zip(
                canonical_equations, canonical_exponents, strict=True
            )
        )
        if len(set(term_keys)) != len(term_keys):
            raise ValueError(
                "Sparse polynomial support cannot contain duplicate equation/exponent "
                "terms."
            )
        groups_ = _validate_groups(
            groups,
            len(variables),
            len(equations_labels),
            canonical_equations,
            canonical_exponents,
        )
        self.variable_labels = variables
        self.equation_labels = equations_labels
        self.equation_indices = jnp.asarray(canonical_equations, dtype=jnp.int32)
        self.exponents = jnp.asarray(canonical_exponents, dtype=jnp.int32)
        self.groups = groups_
        self.variable_count = len(variables)
        self.equation_count = len(equations_labels)
        self.term_count = len(order)
        self.canonical_term_permutation = order
        self.support_id = canonical_fingerprint(
            {
                "kind": "sparse-polynomial-support-v1",
                "variable_labels": list(variables),
                "equation_labels": list(equations_labels),
                "equation_indices": canonical_equations,
                "exponents": canonical_exponents,
                "groups": [
                    {
                        "label": group.label,
                        "variable_indices": list(group.variable_indices),
                        "geometry": group.geometry,
                    }
                    for group in groups_
                ],
            }
        )


class SparsePolynomialSystem(StrictModule):
    """Fixed-support real or complex polynomial coefficients and compiled evaluation."""

    support: SparsePolynomialSupport
    coefficients: Array
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: SparsePolynomialSupport,
        coefficients: ArrayLike,
        /,
    ):
        if not isinstance(support, SparsePolynomialSupport):
            raise TypeError("support must be SparsePolynomialSupport.")
        coefficients_ = jnp.asarray(coefficients)
        if coefficients_.shape != (support.term_count,):
            raise ValueError(
                f"coefficients must have shape {(support.term_count,)}; "
                f"got {coefficients_.shape}."
            )
        if not jnp.issubdtype(coefficients_.dtype, jnp.inexact):
            coefficients_ = coefficients_.astype(float)
        self.support = support
        self.coefficients = coefficients_
        self.system_id = canonical_fingerprint(
            {
                "kind": "sparse-polynomial-system-v1",
                "support": support.support_id,
                "coefficients": array_tree_fingerprint(coefficients_),
            }
        )

    @classmethod
    def from_coo(
        cls,
        variable_labels: Sequence[str],
        equation_labels: Sequence[str],
        equation_indices: Any,
        exponents: Any,
        coefficients: ArrayLike,
        /,
        *,
        groups: Sequence[PolynomialVariableGroup] = (),
    ) -> SparsePolynomialSystem:
        """Jointly canonicalize COO support and its correspondingly ordered values."""
        raw_support = SparsePolynomialSupport(
            variable_labels,
            equation_labels,
            equation_indices,
            exponents,
            groups=groups,
        )
        coefficients_ = jnp.asarray(coefficients)
        if coefficients_.shape != (raw_support.term_count,):
            raise ValueError(
                f"coefficients must have shape {(raw_support.term_count,)}; "
                f"got {coefficients_.shape}."
            )
        order = jnp.asarray(raw_support.canonical_term_permutation, dtype=jnp.int32)
        canonical_coefficients = coefficients_[order]
        canonical_support = SparsePolynomialSupport(
            raw_support.variable_labels,
            raw_support.equation_labels,
            np.asarray(raw_support.equation_indices),
            np.asarray(raw_support.exponents),
            groups=raw_support.groups,
        )
        return cls(canonical_support, canonical_coefficients)

    def evaluate(self, points: ArrayLike, /) -> Array:
        """Evaluate at points shaped ``(..., variable_count)``."""
        points_ = jnp.asarray(points)
        if points_.ndim < 1 or points_.shape[-1] != self.support.variable_count:
            raise ValueError(
                "Polynomial points must have trailing shape "
                f"({self.support.variable_count},); got {points_.shape}."
            )
        if not jnp.issubdtype(points_.dtype, jnp.inexact):
            points_ = points_.astype(float)
        dtype = jnp.result_type(points_.dtype, self.coefficients.dtype)
        bases = points_.astype(dtype)[..., None, :]
        factors = jnp.power(bases, self.support.exponents)
        terms = jnp.prod(factors, axis=-1) * self.coefficients.astype(dtype)
        values = jnp.zeros(
            points_.shape[:-1] + (self.support.equation_count,), dtype=dtype
        )
        return values.at[..., self.support.equation_indices].add(terms)

    def jacobian(self, points: ArrayLike, /) -> Array:
        """Evaluate the analytic Jacobian without division by point coordinates."""
        points_ = jnp.asarray(points)
        if points_.ndim < 1 or points_.shape[-1] != self.support.variable_count:
            raise ValueError(
                "Polynomial points must have trailing shape "
                f"({self.support.variable_count},); got {points_.shape}."
            )
        if not jnp.issubdtype(points_.dtype, jnp.inexact):
            points_ = points_.astype(float)
        dtype = jnp.result_type(points_.dtype, self.coefficients.dtype)
        bases = points_.astype(dtype)[..., None, :]
        exponents = self.support.exponents
        factors = jnp.power(bases, exponents)
        ones = jnp.ones(factors.shape[:-1] + (1,), dtype=dtype)
        prefix = jnp.concatenate((ones, jnp.cumprod(factors, axis=-1)[..., :-1]), axis=-1)
        suffix = jnp.concatenate(
            (
                jnp.cumprod(factors[..., ::-1], axis=-1)[..., :-1][..., ::-1],
                ones,
            ),
            axis=-1,
        )
        decremented = jnp.maximum(exponents - 1, 0)
        own_derivative = jnp.where(
            exponents > 0,
            exponents.astype(dtype) * jnp.power(bases, decremented),
            jnp.zeros((), dtype=dtype),
        )
        term_jacobians = (
            prefix * suffix * own_derivative * self.coefficients.astype(dtype)[..., None]
        )
        jacobian = jnp.zeros(
            points_.shape[:-1]
            + (self.support.equation_count, self.support.variable_count),
            dtype=dtype,
        )
        return jacobian.at[..., self.support.equation_indices, :].add(term_jacobians)

    def with_coefficients(self, coefficients: ArrayLike, /) -> SparsePolynomialSystem:
        """Retain the exact support, including explicitly zero coefficient slots."""
        return SparsePolynomialSystem(self.support, coefficients)


class PolynomialScaling(StrictModule, NonTrainableState):
    """Positive diagonal variable and equation scales for one support shape."""

    variable_scale: Array
    equation_scale: Array
    variable_count: int = eqx.field(static=True)
    equation_count: int = eqx.field(static=True)
    scaling_id: str = eqx.field(static=True)

    def __init__(
        self,
        variable_scale: ArrayLike,
        equation_scale: ArrayLike,
        /,
    ):
        variable = jnp.asarray(variable_scale)
        equation = jnp.asarray(equation_scale)
        if variable.ndim != 1 or equation.ndim != 1:
            raise ValueError("Polynomial scales must be one-dimensional arrays.")
        if not jnp.issubdtype(variable.dtype, jnp.floating) or not jnp.issubdtype(
            equation.dtype, jnp.floating
        ):
            raise TypeError("Polynomial scales must use real floating dtypes.")
        variable_host = np.asarray(variable)
        equation_host = np.asarray(equation)
        if not np.all(np.isfinite(variable_host) & (variable_host > 0.0)) or not np.all(
            np.isfinite(equation_host) & (equation_host > 0.0)
        ):
            raise ValueError("Polynomial scales must be finite and strictly positive.")
        self.variable_scale = variable
        self.equation_scale = equation
        self.variable_count = int(variable.shape[0])
        self.equation_count = int(equation.shape[0])
        self.scaling_id = canonical_fingerprint(
            {
                "kind": "polynomial-scaling-v1",
                "variable_scale": array_tree_fingerprint(variable),
                "equation_scale": array_tree_fingerprint(equation),
            }
        )

    def _validate_points(self, points: ArrayLike, /) -> Array:
        points_ = jnp.asarray(points)
        if points_.ndim < 1 or points_.shape[-1] != self.variable_count:
            raise ValueError(
                f"Points must have trailing shape {(self.variable_count,)}; "
                f"got {points_.shape}."
            )
        return points_

    def _validate_residuals(self, residuals: ArrayLike, /) -> Array:
        residuals_ = jnp.asarray(residuals)
        if residuals_.ndim < 1 or residuals_.shape[-1] != self.equation_count:
            raise ValueError(
                f"Residuals must have trailing shape {(self.equation_count,)}; "
                f"got {residuals_.shape}."
            )
        return residuals_

    def to_scaled_points(self, physical_points: ArrayLike, /) -> Array:
        return self._validate_points(physical_points) / self.variable_scale

    def to_physical_points(self, scaled_points: ArrayLike, /) -> Array:
        return self._validate_points(scaled_points) * self.variable_scale

    def to_scaled_residuals(self, physical_residuals: ArrayLike, /) -> Array:
        return self._validate_residuals(physical_residuals) / self.equation_scale

    def to_physical_residuals(self, scaled_residuals: ArrayLike, /) -> Array:
        return self._validate_residuals(scaled_residuals) * self.equation_scale

    def scale_system(self, system: SparsePolynomialSystem, /) -> SparsePolynomialSystem:
        """Return ``g(y)=equation_scale⁻¹ f(variable_scale*y)``."""
        self._validate_system(system)
        term_scale = jnp.prod(
            jnp.power(self.variable_scale[None, :], system.support.exponents), axis=-1
        )
        coefficients = (
            system.coefficients
            * term_scale
            / self.equation_scale[system.support.equation_indices]
        )
        return SparsePolynomialSystem(system.support, coefficients)

    def unscale_system(
        self,
        system: SparsePolynomialSystem,
        /,
    ) -> SparsePolynomialSystem:
        """Invert :meth:`scale_system` while preserving every support slot."""
        self._validate_system(system)
        term_scale = jnp.prod(
            jnp.power(self.variable_scale[None, :], system.support.exponents), axis=-1
        )
        coefficients = (
            system.coefficients
            * self.equation_scale[system.support.equation_indices]
            / term_scale
        )
        return SparsePolynomialSystem(system.support, coefficients)

    def _validate_system(self, system: SparsePolynomialSystem, /) -> None:
        if not isinstance(system, SparsePolynomialSystem):
            raise TypeError("system must be SparsePolynomialSystem.")
        if (
            system.support.variable_count != self.variable_count
            or system.support.equation_count != self.equation_count
        ):
            raise ValueError("Polynomial scaling and system dimensions must match.")


__all__ = [
    "PolynomialScaling",
    "SparsePolynomialSupport",
    "SparsePolynomialSystem",
]
