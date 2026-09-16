#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Declared linear actions and verified subspaces for sparse polynomials.

This module deliberately stops at finite tables, supplied reductive Lie-algebra
relations, and one declared scaling action.  It is not a general Lie-group
framework.  Numerical residuals establish only the represented relations and
subspaces described by their evidence objects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..algebraic._system import PolynomialScaling, SparsePolynomialSupport
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    ComposedLinearOperator,
    DenseLinearOperator,
    DiagonalPairing,
    eigen as eigen_api,
    FailurePolicy,
    OperatorProperties,
    svd as svd_api,
)
from ._linear_representation import (
    AbstractLinearRepresentation,
    CallableLinearRepresentation,
    LinearAssemblyEvidence,
    LinearConditionAssembly,
    LinearRepresentationCertificate,
)


PolynomialActionStatus: TypeAlias = Literal["verified", "ambiguous"]
PolynomialActionKind: TypeAlias = Literal["finite", "declared-reductive", "scaling"]
PolynomialSubspaceKind: TypeAlias = Literal[
    "invariant", "equivariant", "casimir-spectral-block"
]


def _identifier(value: Any, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be nonempty.")
    return identifier


def _tolerances(
    verification_tolerance: float,
    rejection_tolerance: float,
    /,
) -> tuple[float, float]:
    verification = float(verification_tolerance)
    rejection = float(rejection_tolerance)
    if (
        not math.isfinite(verification)
        or not math.isfinite(rejection)
        or verification < 0.0
        or rejection < verification
    ):
        raise ValueError(
            "Action tolerances must be finite and satisfy "
            "0 <= verification_tolerance <= rejection_tolerance."
        )
    return verification, rejection


def _numeric_array(value: Any, name: str, /, *, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _common_inexact(*values: np.ndarray) -> tuple[np.ndarray, ...]:
    complex_valued = any(
        np.issubdtype(value.dtype, np.complexfloating) for value in values
    )
    dtype = np.complex128 if complex_valued else np.float64
    return tuple(np.asarray(value, dtype=dtype) for value in values)


def _maximum_absolute(value: np.ndarray, /) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def _relation_status(
    residuals: np.ndarray,
    verification_tolerance: float,
    rejection_tolerance: float,
    /,
) -> PolynomialActionStatus:
    maximum = _maximum_absolute(residuals)
    if maximum > rejection_tolerance:
        raise ValueError(
            "Supplied action matrices do not represent the declared relations: "
            f"maximum residual {maximum} exceeds {rejection_tolerance}."
        )
    return "verified" if maximum <= verification_tolerance else "ambiguous"


def _is_exact_input(*values: np.ndarray) -> bool:
    return all(
        np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.bool_)
        for value in values
    )


class PolynomialActionEvidence(StrictModule, NonTrainableState):
    """Relation residuals for one support-local action declaration.

    ``reductivity == "declared"`` records a caller declaration; it is not a
    numerical proof that the supplied Lie algebra or its integrating group is
    reductive.
    """

    kind: PolynomialActionKind = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    relation_residuals: Array
    verification_tolerance: float = eqx.field(static=True)
    rejection_tolerance: float = eqx.field(static=True)
    status: PolynomialActionStatus = eqx.field(static=True)
    exact_relations: bool = eqx.field(static=True)
    complex_valued: bool = eqx.field(static=True)
    reductivity: Literal["not-applicable", "declared"] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: PolynomialActionKind,
        support_id: str,
        relation_residuals: ArrayLike,
        /,
        *,
        verification_tolerance: float,
        rejection_tolerance: float,
        exact_relations: bool,
        complex_valued: bool,
        reductivity: Literal["not-applicable", "declared"] = "not-applicable",
    ):
        if kind not in ("finite", "declared-reductive", "scaling"):
            raise ValueError("Unknown polynomial action kind.")
        if reductivity not in ("not-applicable", "declared"):
            raise ValueError("Unknown reductivity evidence scope.")
        verification, rejection = _tolerances(verification_tolerance, rejection_tolerance)
        residuals_host = np.asarray(relation_residuals, dtype=float).reshape((-1,))
        if not np.all(np.isfinite(residuals_host)):
            raise ValueError("Action relation residuals must be finite.")
        status = _relation_status(residuals_host, verification, rejection)
        support_id_ = _identifier(support_id, "support_id")
        residuals = jnp.asarray(residuals_host)
        self.kind = kind
        self.support_id = support_id_
        self.relation_residuals = residuals
        self.verification_tolerance = verification
        self.rejection_tolerance = rejection
        self.status = status
        self.exact_relations = bool(exact_relations)
        self.complex_valued = bool(complex_valued)
        self.reductivity = reductivity
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "polynomial-action-evidence-v1",
                "action_kind": kind,
                "support": support_id_,
                "residuals": array_tree_fingerprint(residuals),
                "verification_tolerance": verification,
                "rejection_tolerance": rejection,
                "status": status,
                "exact_relations": bool(exact_relations),
                "complex_valued": bool(complex_valued),
                "reductivity": reductivity,
            }
        )

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    @property
    def maximum_relation_residual(self) -> Array:
        return jnp.max(self.relation_residuals, initial=0.0)


class FinitePolynomialAction(StrictModule, NonTrainableState):
    """A complete finite-group table and its input/output linear actions."""

    support: SparsePolynomialSupport
    element_labels: tuple[str, ...] = eqx.field(static=True)
    multiplication_table: Array
    identity_index: int = eqx.field(static=True)
    variable_actions: Array
    equation_actions: Array
    evidence: PolynomialActionEvidence
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: SparsePolynomialSupport,
        variable_actions: ArrayLike,
        multiplication_table: ArrayLike,
        /,
        *,
        equation_actions: ArrayLike | None = None,
        identity_index: int = 0,
        element_labels: Sequence[str] = (),
        verification_tolerance: float = 1e-10,
        rejection_tolerance: float = 1e-7,
    ):
        if not isinstance(support, SparsePolynomialSupport):
            raise TypeError("support must be a SparsePolynomialSupport.")
        variable_input = _numeric_array(variable_actions, "variable_actions", ndim=3)
        count = int(variable_input.shape[0])
        if variable_input.shape[1:] != (support.variable_count, support.variable_count):
            raise ValueError("variable_actions have incompatible variable dimensions.")
        if count < 1:
            raise ValueError("A finite action requires at least the identity element.")
        table = np.asarray(multiplication_table)
        if table.shape != (count, count) or not np.issubdtype(table.dtype, np.integer):
            raise ValueError(
                "multiplication_table must be an integer square table matching the actions."
            )
        table = np.asarray(table, dtype=np.int64)
        if np.any((table < 0) | (table >= count)):
            raise ValueError(
                "multiplication_table contains an out-of-range element index."
            )
        identity = int(identity_index)
        if identity < 0 or identity >= count:
            raise ValueError("identity_index is outside the finite action.")
        indices = np.arange(count)
        if not np.array_equal(table[identity], indices) or not np.array_equal(
            table[:, identity], indices
        ):
            raise ValueError("multiplication_table does not have the declared identity.")
        for left in range(count):
            if not any(
                table[left, candidate] == identity and table[candidate, left] == identity
                for candidate in range(count)
            ):
                raise ValueError(
                    "multiplication_table contains an element without an inverse."
                )
            for middle in range(count):
                for right in range(count):
                    if (
                        table[table[left, middle], right]
                        != table[left, table[middle, right]]
                    ):
                        raise ValueError("multiplication_table is not associative.")
        if equation_actions is None:
            equation_input = np.broadcast_to(
                np.eye(support.equation_count, dtype=variable_input.dtype),
                (count, support.equation_count, support.equation_count),
            ).copy()
        else:
            equation_input = _numeric_array(equation_actions, "equation_actions", ndim=3)
            if equation_input.shape != (
                count,
                support.equation_count,
                support.equation_count,
            ):
                raise ValueError(
                    "equation_actions have incompatible equation dimensions."
                )
        exact_inputs = _is_exact_input(variable_input, equation_input)
        variable, equation = _common_inexact(variable_input, equation_input)
        residuals: list[float] = [
            _maximum_absolute(variable[identity] - np.eye(support.variable_count)),
            _maximum_absolute(equation[identity] - np.eye(support.equation_count)),
        ]
        for left in range(count):
            for right in range(count):
                product = int(table[left, right])
                residuals.extend(
                    (
                        _maximum_absolute(
                            variable[left] @ variable[right] - variable[product]
                        ),
                        _maximum_absolute(
                            equation[left] @ equation[right] - equation[product]
                        ),
                    )
                )
        labels = (
            tuple(f"g{index}" for index in range(count))
            if not element_labels
            else tuple(_identifier(label, "element label") for label in element_labels)
        )
        if len(labels) != count or len(set(labels)) != count:
            raise ValueError("element_labels must uniquely name every finite element.")
        evidence = PolynomialActionEvidence(
            "finite",
            support.support_id,
            np.asarray(residuals),
            verification_tolerance=verification_tolerance,
            rejection_tolerance=rejection_tolerance,
            exact_relations=exact_inputs and not any(residuals),
            complex_valued=np.iscomplexobj(variable) or np.iscomplexobj(equation),
        )
        variable_array = jnp.asarray(variable)
        equation_array = jnp.asarray(equation)
        table_array = jnp.asarray(table, dtype=jnp.int32)
        self.support = support
        self.element_labels = labels
        self.multiplication_table = table_array
        self.identity_index = identity
        self.variable_actions = variable_array
        self.equation_actions = equation_array
        self.evidence = evidence
        self.action_id = canonical_fingerprint(
            {
                "kind": "finite-polynomial-action-v1",
                "support": support.support_id,
                "labels": labels,
                "table": array_tree_fingerprint(table_array),
                "identity": identity,
                "variable_actions": array_tree_fingerprint(variable_array),
                "equation_actions": array_tree_fingerprint(equation_array),
                "evidence": evidence.evidence_id,
            }
        )


class DeclaredReductivePolynomialAction(StrictModule, NonTrainableState):
    """Supplied Lie-algebra generators with verified bracket relations.

    The matrices are checked against ``structure_constants``.  The word
    ``reductive`` remains an explicit declaration and is recorded as such in
    the evidence; these numerical checks do not prove reductivity.
    """

    support: SparsePolynomialSupport
    generator_labels: tuple[str, ...] = eqx.field(static=True)
    variable_generators: Array
    equation_generators: Array
    structure_constants: Array
    evidence: PolynomialActionEvidence
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: SparsePolynomialSupport,
        variable_generators: ArrayLike,
        structure_constants: ArrayLike,
        /,
        *,
        equation_generators: ArrayLike | None = None,
        generator_labels: Sequence[str] = (),
        verification_tolerance: float = 1e-10,
        rejection_tolerance: float = 1e-7,
    ):
        if not isinstance(support, SparsePolynomialSupport):
            raise TypeError("support must be a SparsePolynomialSupport.")
        variable_input = _numeric_array(
            variable_generators, "variable_generators", ndim=3
        )
        count = int(variable_input.shape[0])
        if count < 1 or variable_input.shape[1:] != (
            support.variable_count,
            support.variable_count,
        ):
            raise ValueError("variable_generators have incompatible dimensions.")
        structure_input = _numeric_array(
            structure_constants, "structure_constants", ndim=3
        )
        if structure_input.shape != (count, count, count):
            raise ValueError("structure_constants must have shape (G, G, G).")
        if equation_generators is None:
            equation_input = np.zeros(
                (count, support.equation_count, support.equation_count),
                dtype=variable_input.dtype,
            )
        else:
            equation_input = _numeric_array(
                equation_generators, "equation_generators", ndim=3
            )
            if equation_input.shape != (
                count,
                support.equation_count,
                support.equation_count,
            ):
                raise ValueError(
                    "equation_generators have incompatible equation dimensions."
                )
        exact_inputs = _is_exact_input(variable_input, equation_input, structure_input)
        variable, equation, structure = _common_inexact(
            variable_input, equation_input, structure_input
        )
        residuals: list[float] = [
            _maximum_absolute(structure + np.swapaxes(structure, 0, 1))
        ]
        jacobi = np.zeros((count, count, count, count), dtype=structure.dtype)
        for left in range(count):
            for middle in range(count):
                for right in range(count):
                    for contracted in range(count):
                        jacobi[left, middle, right] += (
                            structure[middle, right, contracted]
                            * structure[left, contracted]
                            + structure[right, left, contracted]
                            * structure[middle, contracted]
                            + structure[left, middle, contracted]
                            * structure[right, contracted]
                        )
        residuals.append(_maximum_absolute(jacobi))
        for left in range(count):
            for right in range(count):
                variable_bracket = (
                    variable[left] @ variable[right] - variable[right] @ variable[left]
                )
                equation_bracket = (
                    equation[left] @ equation[right] - equation[right] @ equation[left]
                )
                variable_expected = np.zeros_like(variable_bracket)
                equation_expected = np.zeros_like(equation_bracket)
                for output in range(count):
                    variable_expected += structure[left, right, output] * variable[output]
                    equation_expected += structure[left, right, output] * equation[output]
                residuals.extend(
                    (
                        _maximum_absolute(variable_bracket - variable_expected),
                        _maximum_absolute(equation_bracket - equation_expected),
                    )
                )
        labels = (
            tuple(f"X{index}" for index in range(count))
            if not generator_labels
            else tuple(
                _identifier(label, "generator label") for label in generator_labels
            )
        )
        if len(labels) != count or len(set(labels)) != count:
            raise ValueError("generator_labels must uniquely name every generator.")
        evidence = PolynomialActionEvidence(
            "declared-reductive",
            support.support_id,
            np.asarray(residuals),
            verification_tolerance=verification_tolerance,
            rejection_tolerance=rejection_tolerance,
            exact_relations=exact_inputs and not any(residuals),
            complex_valued=(
                np.iscomplexobj(variable)
                or np.iscomplexobj(equation)
                or np.iscomplexobj(structure)
            ),
            reductivity="declared",
        )
        variable_array = jnp.asarray(variable)
        equation_array = jnp.asarray(equation)
        structure_array = jnp.asarray(structure)
        self.support = support
        self.generator_labels = labels
        self.variable_generators = variable_array
        self.equation_generators = equation_array
        self.structure_constants = structure_array
        self.evidence = evidence
        self.action_id = canonical_fingerprint(
            {
                "kind": "declared-reductive-polynomial-action-v1",
                "support": support.support_id,
                "labels": labels,
                "variable_generators": array_tree_fingerprint(variable_array),
                "equation_generators": array_tree_fingerprint(equation_array),
                "structure_constants": array_tree_fingerprint(structure_array),
                "evidence": evidence.evidence_id,
            }
        )


class PolynomialScalingAction(StrictModule, NonTrainableState):
    """One declared scaling equation ``p(D x) = E p(x)`` on fixed support."""

    support: SparsePolynomialSupport
    scaling: PolynomialScaling
    variable_action: Array
    equation_action: Array
    evidence: PolynomialActionEvidence
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: SparsePolynomialSupport,
        scaling: PolynomialScaling,
        /,
    ):
        if not isinstance(support, SparsePolynomialSupport):
            raise TypeError("support must be a SparsePolynomialSupport.")
        if not isinstance(scaling, PolynomialScaling):
            raise TypeError("scaling must be a PolynomialScaling.")
        variable_scale = np.asarray(scaling.variable_scale)
        equation_scale = np.asarray(scaling.equation_scale)
        if variable_scale.shape != (support.variable_count,) or equation_scale.shape != (
            support.equation_count,
        ):
            raise ValueError("PolynomialScaling dimensions do not match the support.")
        variable, equation = _common_inexact(
            np.diag(variable_scale), np.diag(equation_scale)
        )
        evidence = PolynomialActionEvidence(
            "scaling",
            support.support_id,
            np.zeros((1,), dtype=float),
            verification_tolerance=0.0,
            rejection_tolerance=0.0,
            exact_relations=False,
            complex_valued=False,
        )
        variable_array = jnp.asarray(variable)
        equation_array = jnp.asarray(equation)
        self.support = support
        self.scaling = scaling
        self.variable_action = variable_array
        self.equation_action = equation_array
        self.evidence = evidence
        self.action_id = canonical_fingerprint(
            {
                "kind": "polynomial-scaling-action-v1",
                "support": support.support_id,
                "variable": array_tree_fingerprint(variable_array),
                "equation": array_tree_fingerprint(equation_array),
            }
        )


PolynomialAction: TypeAlias = (
    FinitePolynomialAction | DeclaredReductivePolynomialAction | PolynomialScalingAction
)


def weighted_polynomial_action(
    support: SparsePolynomialSupport,
    variable_weights: ArrayLike,
    /,
    *,
    equation_weights: ArrayLike | None = None,
    label: str = "weight",
    verification_tolerance: float = 1e-10,
    rejection_tolerance: float = 1e-7,
) -> DeclaredReductivePolynomialAction:
    """Build the one-generator abelian weight action on polynomial support."""
    if not isinstance(support, SparsePolynomialSupport):
        raise TypeError("support must be a SparsePolynomialSupport.")

    variable = _numeric_array(variable_weights, "variable_weights", ndim=1)
    if variable.shape != (support.variable_count,):
        raise ValueError("variable_weights do not match support.variable_count.")
    equation = (
        np.zeros((support.equation_count,), dtype=variable.dtype)
        if equation_weights is None
        else _numeric_array(equation_weights, "equation_weights", ndim=1)
    )
    if equation.shape != (support.equation_count,):
        raise ValueError("equation_weights do not match support.equation_count.")
    return DeclaredReductivePolynomialAction(
        support,
        np.diag(variable)[None, ...],
        np.zeros((1, 1, 1), dtype=int),
        equation_generators=np.diag(equation)[None, ...],
        generator_labels=(_identifier(label, "label"),),
        verification_tolerance=verification_tolerance,
        rejection_tolerance=rejection_tolerance,
    )


def scaling_polynomial_action(
    support: SparsePolynomialSupport,
    scaling: PolynomialScaling,
    /,
) -> PolynomialScalingAction:
    """Build a support-local action from a core ``PolynomialScaling``."""

    return PolynomialScalingAction(support, scaling)


def _monomial_metric(exponent: Sequence[int], /) -> float:
    degree = sum(int(value) for value in exponent)
    if degree == 0:
        return 1.0
    numerator = math.prod(math.factorial(int(value)) for value in exponent)
    return float(numerator / math.factorial(degree))


def _expanded_monomial(
    exponent: tuple[int, ...],
    action: np.ndarray,
    /,
) -> dict[tuple[int, ...], complex | float]:
    variable_count = len(exponent)
    zero = (0,) * variable_count
    polynomial: dict[tuple[int, ...], complex | float] = {zero: 1.0}
    for source, power in enumerate(exponent):
        for _ in range(power):
            updated: dict[tuple[int, ...], complex | float] = {}
            for current, coefficient in polynomial.items():
                for target in range(variable_count):
                    factor = action[source, target]
                    if factor == 0:
                        continue
                    term = list(current)
                    term[target] += 1
                    key = tuple(term)
                    updated[key] = updated.get(key, 0.0) + coefficient * factor
            polynomial = updated
    return polynomial


def _finite_columns(
    support: SparsePolynomialSupport,
    variable: np.ndarray,
    equation: np.ndarray,
    /,
) -> list[dict[tuple[int, tuple[int, ...]], complex | float]]:
    equation_indices = np.asarray(support.equation_indices, dtype=int)
    exponents = np.asarray(support.exponents, dtype=int)
    columns: list[dict[tuple[int, tuple[int, ...]], complex | float]] = []
    for source_equation, exponent_array in zip(equation_indices, exponents, strict=True):
        exponent = tuple(int(value) for value in exponent_array)
        column: dict[tuple[int, tuple[int, ...]], complex | float] = {}
        for transformed_exponent, coefficient in _expanded_monomial(
            exponent, variable
        ).items():
            key = int(source_equation), transformed_exponent
            column[key] = column.get(key, 0.0) + coefficient
        for target_equation in range(support.equation_count):
            coefficient = equation[target_equation, source_equation]
            if coefficient == 0:
                continue
            key = target_equation, exponent
            column[key] = column.get(key, 0.0) - coefficient
        columns.append(column)
    return columns


def _infinitesimal_columns(
    support: SparsePolynomialSupport,
    variable: np.ndarray,
    equation: np.ndarray,
    /,
) -> list[dict[tuple[int, tuple[int, ...]], complex | float]]:
    equation_indices = np.asarray(support.equation_indices, dtype=int)
    exponents = np.asarray(support.exponents, dtype=int)
    columns: list[dict[tuple[int, tuple[int, ...]], complex | float]] = []
    for source_equation, exponent_array in zip(equation_indices, exponents, strict=True):
        exponent = tuple(int(value) for value in exponent_array)
        column: dict[tuple[int, tuple[int, ...]], complex | float] = {}
        for source_variable, power in enumerate(exponent):
            if power == 0:
                continue
            for target_variable in range(support.variable_count):
                coefficient = power * variable[source_variable, target_variable]
                if coefficient == 0:
                    continue
                transformed = list(exponent)
                transformed[source_variable] -= 1
                transformed[target_variable] += 1
                key = int(source_equation), tuple(transformed)
                column[key] = column.get(key, 0.0) + coefficient
        for target_equation in range(support.equation_count):
            coefficient = equation[target_equation, source_equation]
            if coefficient == 0:
                continue
            key = target_equation, exponent
            column[key] = column.get(key, 0.0) - coefficient
        columns.append(column)
    return columns


class PolynomialActionConstraints(StrictModule, NonTrainableState):
    """Ambient coefficient residuals for all supplied action generators/elements."""

    action_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    matrix: Array
    ambient_equation_indices: Array
    ambient_exponents: Array
    source_metric_weights: Array
    ambient_metric_weights: Array
    action_count: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    constraints_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        action_id: str,
        support_id: str,
        matrix: ArrayLike,
        ambient_equation_indices: ArrayLike,
        ambient_exponents: ArrayLike,
        source_metric_weights: ArrayLike,
        ambient_metric_weights: ArrayLike,
        action_count: int,
        exact: bool,
    ):
        matrix_ = jnp.asarray(matrix)
        equations = jnp.asarray(ambient_equation_indices, dtype=jnp.int32)
        exponents = jnp.asarray(ambient_exponents, dtype=jnp.int32)
        source_weights = jnp.asarray(source_metric_weights)
        ambient_weights = jnp.asarray(ambient_metric_weights)
        count = int(action_count)
        ambient_count = int(equations.shape[0])
        if matrix_.ndim != 2 or matrix_.shape[0] != count * ambient_count:
            raise ValueError(
                "Constraint matrix rows do not match the ambient action layout."
            )
        if exponents.ndim != 2 or exponents.shape[0] != ambient_count:
            raise ValueError("Ambient exponents do not match ambient equations.")
        if source_weights.shape != (matrix_.shape[1],) or ambient_weights.shape != (
            ambient_count,
        ):
            raise ValueError("Polynomial metric weights do not match the constraints.")
        self.action_id = _identifier(action_id, "action_id")
        self.support_id = _identifier(support_id, "support_id")
        self.matrix = matrix_
        self.ambient_equation_indices = equations
        self.ambient_exponents = exponents
        self.source_metric_weights = source_weights
        self.ambient_metric_weights = ambient_weights
        self.action_count = count
        self.exact = bool(exact)
        self.constraints_id = canonical_fingerprint(
            {
                "kind": "polynomial-action-constraints-v1",
                "action": self.action_id,
                "support": self.support_id,
                "matrix": array_tree_fingerprint(matrix_),
                "ambient_equations": array_tree_fingerprint(equations),
                "ambient_exponents": array_tree_fingerprint(exponents),
                "source_metric": array_tree_fingerprint(source_weights),
                "ambient_metric": array_tree_fingerprint(ambient_weights),
                "action_count": count,
                "exact": bool(exact),
            }
        )


def polynomial_action_constraints(
    action: PolynomialAction,
    /,
) -> PolynomialActionConstraints:
    """Assemble ``p(Ax)-B p(x)`` or its infinitesimal counterpart."""

    if not isinstance(
        action,
        (
            FinitePolynomialAction,
            DeclaredReductivePolynomialAction,
            PolynomialScalingAction,
        ),
    ):
        raise TypeError("action must be a declared polynomial action.")
    support = action.support
    if isinstance(action, FinitePolynomialAction):
        variables = np.asarray(action.variable_actions)
        equations = np.asarray(action.equation_actions)
        columns_by_action = [
            _finite_columns(support, variable, equation)
            for variable, equation in zip(variables, equations, strict=True)
        ]
    elif isinstance(action, DeclaredReductivePolynomialAction):
        variables = np.asarray(action.variable_generators)
        equations = np.asarray(action.equation_generators)
        columns_by_action = [
            _infinitesimal_columns(support, variable, equation)
            for variable, equation in zip(variables, equations, strict=True)
        ]
    else:
        variables = np.asarray(action.variable_action)[None, ...]
        equations = np.asarray(action.equation_action)[None, ...]
        columns_by_action = [_finite_columns(support, variables[0], equations[0])]
    support_equations = np.asarray(support.equation_indices, dtype=int)
    support_exponents = np.asarray(support.exponents, dtype=int)
    ambient = {
        (int(equation), tuple(int(value) for value in exponent))
        for equation, exponent in zip(support_equations, support_exponents, strict=True)
    }
    for action_columns in columns_by_action:
        for column in action_columns:
            ambient.update(column)
    ambient_terms = tuple(sorted(ambient))
    row_index = {term: index for index, term in enumerate(ambient_terms)}
    dtype = np.result_type(variables.dtype, equations.dtype, np.float64)
    blocks: list[np.ndarray] = []
    for action_columns in columns_by_action:
        block = np.zeros((len(ambient_terms), support.term_count), dtype=dtype)
        for column_index, column in enumerate(action_columns):
            for term, coefficient in column.items():
                block[row_index[term], column_index] += coefficient
        blocks.append(block)
    matrix = np.concatenate(blocks, axis=0)
    ambient_equations = np.asarray([term[0] for term in ambient_terms], dtype=np.int32)
    ambient_exponents = np.asarray([term[1] for term in ambient_terms], dtype=np.int32)
    source_weights = np.asarray(
        [
            _monomial_metric(tuple(int(value) for value in exponent))
            for exponent in support_exponents
        ],
        dtype=float,
    )
    ambient_weights = np.asarray(
        [_monomial_metric(term[1]) for term in ambient_terms], dtype=float
    )
    return PolynomialActionConstraints(
        action_id=action.action_id,
        support_id=support.support_id,
        matrix=jnp.asarray(matrix),
        ambient_equation_indices=ambient_equations,
        ambient_exponents=ambient_exponents,
        source_metric_weights=jnp.asarray(source_weights),
        ambient_metric_weights=jnp.asarray(ambient_weights),
        action_count=len(columns_by_action),
        exact=action.evidence.exact_relations,
    )


class PolynomialSubspaceEvidence(StrictModule, NonTrainableState):
    """Numerical residual and claim scope for one polynomial coefficient block."""

    action_evidence_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    kind: PolynomialSubspaceKind = eqx.field(static=True)
    label: str = eqx.field(static=True)
    status: PolynomialActionStatus = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    numerical_rank: int = eqx.field(static=True)
    maximum_selected_residual: Array
    minimum_excluded_residual: Array
    verification_tolerance: float = eqx.field(static=True)
    rejection_tolerance: float = eqx.field(static=True)
    method: str = eqx.field(static=True)
    method_evidence_id: str = eqx.field(static=True)
    candidate_isotypic: bool = eqx.field(static=True)
    irreducibility_proven: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        action_evidence_id: str,
        source_id: str,
        kind: PolynomialSubspaceKind,
        label: str,
        status: PolynomialActionStatus,
        dimension: int,
        numerical_rank: int,
        maximum_selected_residual: ArrayLike,
        minimum_excluded_residual: ArrayLike,
        verification_tolerance: float,
        rejection_tolerance: float,
        method: str,
        method_evidence_id: str,
        candidate_isotypic: bool = False,
    ):
        if kind not in ("invariant", "equivariant", "casimir-spectral-block"):
            raise ValueError("Unknown polynomial subspace kind.")
        if status not in ("verified", "ambiguous"):
            raise ValueError("Unknown polynomial subspace status.")
        verification, rejection = _tolerances(verification_tolerance, rejection_tolerance)
        dimension_ = int(dimension)
        rank_ = int(numerical_rank)
        if dimension_ < 0 or rank_ < 0:
            raise ValueError("Subspace dimension and numerical rank must be nonnegative.")
        maximum = jnp.asarray(maximum_selected_residual)
        minimum = jnp.asarray(minimum_excluded_residual)
        if maximum.shape != () or minimum.shape != ():
            raise ValueError("Subspace residual evidence must be scalar.")
        self.action_evidence_id = _identifier(action_evidence_id, "action_evidence_id")
        self.source_id = _identifier(source_id, "source_id")
        self.kind = kind
        self.label = _identifier(label, "label")
        self.status = status
        self.dimension = dimension_
        self.numerical_rank = rank_
        self.maximum_selected_residual = maximum
        self.minimum_excluded_residual = minimum
        self.verification_tolerance = verification
        self.rejection_tolerance = rejection
        self.method = _identifier(method, "method")
        self.method_evidence_id = _identifier(method_evidence_id, "method_evidence_id")
        self.candidate_isotypic = bool(candidate_isotypic)
        # Neither a numerical Casimir cluster nor a character-weight kernel proves
        # irreducibility without an independent representation-theoretic proof.
        self.irreducibility_proven = False
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "polynomial-subspace-evidence-v1",
                "action_evidence": self.action_evidence_id,
                "source": self.source_id,
                "subspace_kind": kind,
                "label": self.label,
                "status": status,
                "dimension": dimension_,
                "numerical_rank": rank_,
                "maximum_selected_residual": array_tree_fingerprint(maximum),
                "minimum_excluded_residual": array_tree_fingerprint(minimum),
                "verification_tolerance": verification,
                "rejection_tolerance": rejection,
                "method": self.method,
                "method_evidence": self.method_evidence_id,
                "candidate_isotypic": bool(candidate_isotypic),
                "irreducibility_proven": False,
            }
        )

    @property
    def verified(self) -> bool:
        return self.status == "verified"


class PolynomialSubspaceBasis(StrictModule, NonTrainableState):
    """A fixed polynomial basis orthonormal in the symmetric-tensor metric."""

    support: SparsePolynomialSupport
    basis: Array
    metric_weights: Array
    evidence: PolynomialSubspaceEvidence
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: SparsePolynomialSupport,
        basis: ArrayLike,
        metric_weights: ArrayLike,
        evidence: PolynomialSubspaceEvidence,
        /,
    ):
        if not isinstance(support, SparsePolynomialSupport):
            raise TypeError("support must be a SparsePolynomialSupport.")
        if not isinstance(evidence, PolynomialSubspaceEvidence):
            raise TypeError("evidence must be PolynomialSubspaceEvidence.")
        basis_ = jnp.asarray(basis)
        weights = jnp.asarray(metric_weights)
        if basis_.ndim != 2 or basis_.shape[0] != support.term_count:
            raise ValueError("Polynomial basis must have shape (term_count, dimension).")
        if weights.shape != (support.term_count,):
            raise ValueError("Polynomial metric weights must match term_count.")
        if basis_.shape[1] != evidence.dimension:
            raise ValueError("Polynomial basis dimension differs from its evidence.")
        self.support = support
        self.basis = basis_
        self.metric_weights = weights
        self.evidence = evidence
        self.basis_id = canonical_fingerprint(
            {
                "kind": "polynomial-subspace-basis-v1",
                "support": support.support_id,
                "basis": array_tree_fingerprint(basis_),
                "metric": array_tree_fingerprint(weights),
                "evidence": evidence.evidence_id,
            }
        )

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])

    def coordinates(self, coefficients: ArrayLike, /) -> Array:
        value = jnp.asarray(coefficients)
        if value.shape != (self.support.term_count,):
            raise ValueError("Polynomial coefficients do not match the support.")
        if value.dtype != self.basis.dtype:
            raise TypeError("Polynomial coefficients and basis must share one dtype.")
        return jnp.conj(self.basis.T) @ (self.metric_weights * value)

    def coefficients(self, coordinates: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinates)
        if value.shape != (self.dimension,):
            raise ValueError("Polynomial subspace coordinates have the wrong shape.")
        if value.dtype != self.basis.dtype:
            raise TypeError("Polynomial coordinates and basis must share one dtype.")
        return self.basis @ value

    def project(self, coefficients: ArrayLike, /) -> Array:
        return self.coefficients(self.coordinates(coefficients))


def _subspace_tolerances(
    action: PolynomialAction,
    verification_tolerance: float | None,
    rejection_tolerance: float | None,
    /,
) -> tuple[float, float]:
    verification = (
        max(action.evidence.verification_tolerance, 64.0 * np.finfo(float).eps)
        if verification_tolerance is None
        else float(verification_tolerance)
    )
    rejection = (
        max(action.evidence.rejection_tolerance, 32.0 * verification)
        if rejection_tolerance is None
        else float(rejection_tolerance)
    )
    return _tolerances(verification, rejection)


def _extract_subspace(
    action: PolynomialAction,
    kind: Literal["invariant", "equivariant"],
    /,
    *,
    verification_tolerance: float | None,
    rejection_tolerance: float | None,
    label: str,
) -> PolynomialSubspaceBasis:
    constraints = polynomial_action_constraints(action)
    verification, rejection = _subspace_tolerances(
        action, verification_tolerance, rejection_tolerance
    )
    matrix = constraints.matrix
    term_count = action.support.term_count
    row_count = int(matrix.shape[0])
    if row_count < term_count:
        matrix = jnp.pad(matrix, ((0, term_count - row_count), (0, 0)))
    padded_rows = int(matrix.shape[0])
    row_weights = jnp.tile(constraints.ambient_metric_weights, constraints.action_count)
    if padded_rows > row_count:
        row_weights = jnp.pad(
            row_weights,
            (0, padded_rows - row_count),
            constant_values=1.0,
        )
    source = ArraySpace(
        (term_count,),
        dtype=matrix.dtype,
        pairing=DiagonalPairing(
            constraints.source_metric_weights.astype(matrix.real.dtype)
        ),
    )
    target = ArraySpace(
        (padded_rows,),
        dtype=matrix.dtype,
        pairing=DiagonalPairing(row_weights.astype(matrix.real.dtype)),
    )
    operator = DenseLinearOperator(
        matrix,
        source=source,
        target=target,
        operator_id=canonical_fingerprint(
            {
                "kind": "polynomial-action-constraint-operator-v1",
                "constraints": constraints.constraints_id,
                "padded_rows": padded_rows,
            }
        ),
    )
    decomposition = svd_api.svd(
        svd_api.SVDProblem(
            operator,
            problem_id=f"polynomial-action/{constraints.constraints_id}",
        ),
        policy=svd_api.SVDSolvePolicy(
            count=term_count,
            which="smallest",
            failure=FailurePolicy("status"),
        ),
    )
    singular_host = np.asarray(decomposition.singular_values)
    verified_mask = singular_host <= verification
    ambiguous_mask = (singular_host > verification) & (singular_host <= rejection)
    dimension = int(np.count_nonzero(verified_mask))
    basis = jnp.asarray(decomposition.right_vectors)[:, :dimension]
    maximum_selected = (
        jnp.max(decomposition.singular_values[:dimension], initial=0.0)
        if dimension
        else jnp.asarray(0.0, dtype=decomposition.singular_values.dtype)
    )
    minimum_excluded = (
        decomposition.singular_values[dimension]
        if dimension < term_count
        else jnp.asarray(jnp.inf, dtype=decomposition.singular_values.dtype)
    )
    successful = bool(np.asarray(decomposition.successful))
    status: PolynomialActionStatus = (
        "verified"
        if action.evidence.verified and successful and not np.any(ambiguous_mask)
        else "ambiguous"
    )
    evidence = PolynomialSubspaceEvidence(
        action_evidence_id=action.evidence.evidence_id,
        source_id=constraints.constraints_id,
        kind=kind,
        label=label,
        status=status,
        dimension=dimension,
        numerical_rank=term_count - dimension,
        maximum_selected_residual=maximum_selected,
        minimum_excluded_residual=minimum_excluded,
        verification_tolerance=verification,
        rejection_tolerance=rejection,
        method="phydrax-pairing-aware-dense-svd",
        method_evidence_id=decomposition.provenance.plan_id,
    )
    return PolynomialSubspaceBasis(
        action.support,
        basis,
        constraints.source_metric_weights.astype(basis.real.dtype),
        evidence,
    )


def extract_equivariant_subspace(
    action: PolynomialAction,
    /,
    *,
    verification_tolerance: float | None = None,
    rejection_tolerance: float | None = None,
    label: str = "equivariant",
) -> PolynomialSubspaceBasis:
    """Extract coefficients satisfying the declared input/output action."""

    return _extract_subspace(
        action,
        "equivariant",
        verification_tolerance=verification_tolerance,
        rejection_tolerance=rejection_tolerance,
        label=_identifier(label, "label"),
    )


def extract_invariant_subspace(
    action: FinitePolynomialAction | DeclaredReductivePolynomialAction,
    /,
    *,
    verification_tolerance: float | None = None,
    rejection_tolerance: float | None = None,
    label: str = "invariant",
) -> PolynomialSubspaceBasis:
    """Extract a subspace only when the declared equation action is trivial."""

    if isinstance(action, FinitePolynomialAction):
        expected = np.broadcast_to(
            np.eye(action.support.equation_count),
            np.asarray(action.equation_actions).shape,
        )
        residual = _maximum_absolute(np.asarray(action.equation_actions) - expected)
    elif isinstance(action, DeclaredReductivePolynomialAction):
        residual = _maximum_absolute(np.asarray(action.equation_generators))
    else:
        raise TypeError("Invariant extraction requires a finite or reductive action.")
    if residual > action.evidence.verification_tolerance:
        raise ValueError(
            "Invariant extraction requires a verified trivial equation action; "
            "use extract_equivariant_subspace for a nontrivial output action."
        )
    return _extract_subspace(
        action,
        "invariant",
        verification_tolerance=verification_tolerance,
        rejection_tolerance=rejection_tolerance,
        label=_identifier(label, "label"),
    )


def _coefficient_generator_matrices(
    action: DeclaredReductivePolynomialAction,
    /,
) -> tuple[Array, Array]:
    constraints = polynomial_action_constraints(action)
    support_terms = tuple(
        sorted(
            (
                int(equation),
                tuple(int(value) for value in exponent),
            )
            for equation, exponent in zip(
                np.asarray(action.support.equation_indices),
                np.asarray(action.support.exponents),
                strict=True,
            )
        )
    )
    ambient_terms = tuple(
        (
            int(equation),
            tuple(int(value) for value in exponent),
        )
        for equation, exponent in zip(
            np.asarray(constraints.ambient_equation_indices),
            np.asarray(constraints.ambient_exponents),
            strict=True,
        )
    )
    if ambient_terms != support_terms:
        raise ValueError(
            "Casimir blocks require support closed under every supplied generator."
        )
    matrices = constraints.matrix.reshape(
        (constraints.action_count, action.support.term_count, action.support.term_count)
    )
    return matrices, constraints.source_metric_weights


def casimir_isotypic_blocks(
    action: DeclaredReductivePolynomialAction,
    /,
    *,
    generator_metric: ArrayLike | None = None,
    verification_tolerance: float | None = None,
    rejection_tolerance: float | None = None,
    labels: Sequence[str] = (),
) -> tuple[PolynomialSubspaceBasis, ...]:
    """Return verified central-Casimir spectral blocks.

    Blocks are evidence for Casimir spectral/isotypic candidates only.  They do
    not claim that a block is irreducible, nor that equal Casimir eigenvalues
    distinguish all isotypic components.
    """

    if not isinstance(action, DeclaredReductivePolynomialAction):
        raise TypeError("Casimir blocks require a DeclaredReductivePolynomialAction.")
    verification, rejection = _subspace_tolerances(
        action, verification_tolerance, rejection_tolerance
    )
    generators, metric_weights = _coefficient_generator_matrices(action)
    generators_host = np.asarray(generators)
    count = int(generators_host.shape[0])
    generator_metric_host = (
        np.eye(count, dtype=generators_host.dtype)
        if generator_metric is None
        else _numeric_array(generator_metric, "generator_metric", ndim=2)
    )
    if generator_metric_host.shape != (count, count):
        raise ValueError(
            "generator_metric must have shape (generator_count, generator_count)."
        )
    if (
        _maximum_absolute(generator_metric_host - np.conj(generator_metric_host.T))
        > verification
    ):
        raise ValueError(
            "generator_metric must be Hermitian within verification tolerance."
        )
    casimir = np.zeros(
        (action.support.term_count, action.support.term_count),
        dtype=np.result_type(generators_host.dtype, generator_metric_host.dtype),
    )
    for left in range(count):
        for right in range(count):
            casimir -= (
                generator_metric_host[left, right]
                * generators_host[left]
                @ generators_host[right]
            )
    weights_host = np.asarray(metric_weights)
    weighted_adjoint = np.conj(casimir.T) * weights_host[None, :] / weights_host[:, None]
    self_adjoint_residual = _maximum_absolute(casimir - weighted_adjoint)
    if self_adjoint_residual > verification:
        raise ValueError(
            "The supplied quadratic generator metric does not produce a verified "
            "self-adjoint Casimir on this polynomial metric."
        )
    centrality_residual = max(
        (
            _maximum_absolute(casimir @ generator - generator @ casimir)
            for generator in generators_host
        ),
        default=0.0,
    )
    if centrality_residual > rejection:
        raise ValueError(
            "The supplied quadratic generator metric is not central on this support."
        )
    dtype = jnp.asarray(casimir).dtype
    space = ArraySpace(
        (action.support.term_count,),
        dtype=dtype,
        pairing=DiagonalPairing(
            jnp.asarray(weights_host, dtype=jnp.asarray(casimir).real.dtype)
        ),
    )
    operator = DenseLinearOperator(
        jnp.asarray(casimir),
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "verified"},
        ),
        operator_id=canonical_fingerprint(
            {
                "kind": "polynomial-casimir-v1",
                "action": action.action_id,
                "generator_metric": array_tree_fingerprint(generator_metric_host),
            }
        ),
    )
    spectrum = eigen_api.self_adjoint_spectrum(
        eigen_api.Eigenproblem(
            operator,
            problem_id=f"polynomial-casimir/{action.action_id}",
        ),
        policy=eigen_api.SelfAdjointSpectrumPolicy(
            relative_tolerance=verification,
            absolute_tolerance=verification,
            failure=FailurePolicy("status"),
        ),
    )
    values_host = np.asarray(spectrum.eigenvalues)
    scale = max(_maximum_absolute(values_host), 1.0)
    verified_gap = verification * scale
    rejected_gap = rejection * scale
    clusters: list[tuple[int, int, bool]] = []
    start = 0
    ambiguous_boundary = False
    for index in range(1, values_host.size):
        gap = float(abs(values_host[index] - values_host[index - 1]))
        if gap <= verified_gap:
            continue
        boundary_ambiguous = gap <= rejected_gap
        clusters.append((start, index, boundary_ambiguous))
        ambiguous_boundary = ambiguous_boundary or boundary_ambiguous
        start = index
    clusters.append((start, values_host.size, False))
    labels_ = (
        tuple(f"casimir-{index}" for index in range(len(clusters)))
        if not labels
        else tuple(_identifier(label, "block label") for label in labels)
    )
    if len(labels_) != len(clusters) or len(set(labels_)) != len(clusters):
        raise ValueError("labels must uniquely name every extracted Casimir block.")
    spectrum_successful = bool(np.asarray(spectrum.successful))
    blocks: list[PolynomialSubspaceBasis] = []
    weights = jnp.asarray(metric_weights, dtype=spectrum.eigenvectors.real.dtype)
    for block_index, (first, stop, boundary_ambiguous) in enumerate(clusters):
        basis = spectrum.eigenvectors[:, first:stop]
        action_residuals: list[Array] = []
        for generator in generators:
            image = generator @ basis
            coordinates = jnp.conj(basis.T) @ (weights[:, None] * image)
            remainder = image - basis @ coordinates
            action_residuals.append(
                jnp.sqrt(jnp.sum(weights[:, None] * jnp.abs(remainder) ** 2))
            )
        maximum_residual = (
            jnp.max(jnp.stack(action_residuals), initial=0.0)
            if action_residuals
            else jnp.asarray(0.0, dtype=spectrum.eigenvalues.dtype)
        )
        maximum_host = float(np.asarray(maximum_residual))
        status: PolynomialActionStatus = (
            "verified"
            if (
                action.evidence.verified
                and spectrum_successful
                and centrality_residual <= verification
                and maximum_host <= verification
                and not ambiguous_boundary
                and not boundary_ambiguous
            )
            else "ambiguous"
        )
        left_gap = abs(values_host[first] - values_host[first - 1]) if first else np.inf
        right_gap = (
            abs(values_host[stop] - values_host[stop - 1])
            if stop < values_host.size
            else np.inf
        )
        evidence = PolynomialSubspaceEvidence(
            action_evidence_id=action.evidence.evidence_id,
            source_id=operator.operator_id,
            kind="casimir-spectral-block",
            label=labels_[block_index],
            status=status,
            dimension=stop - first,
            numerical_rank=stop - first,
            maximum_selected_residual=maximum_residual,
            minimum_excluded_residual=jnp.asarray(
                min(left_gap, right_gap), dtype=spectrum.eigenvalues.dtype
            ),
            verification_tolerance=verification,
            rejection_tolerance=rejection,
            method="phydrax-self-adjoint-spectrum",
            method_evidence_id=spectrum.provenance.plan_id,
            candidate_isotypic=True,
        )
        blocks.append(PolynomialSubspaceBasis(action.support, basis, weights, evidence))
    return tuple(blocks)


def _require_verified_basis(basis: PolynomialSubspaceBasis, /) -> None:
    if not isinstance(basis, PolynomialSubspaceBasis):
        raise TypeError("basis must be a PolynomialSubspaceBasis.")
    if not basis.evidence.verified:
        raise ValueError("Only a verified polynomial basis may be lowered.")
    if basis.dimension < 1:
        raise ValueError("A zero-dimensional polynomial basis cannot be lowered.")


def lower_polynomial_enforcement_operator(
    operator: AbstractLinearOperator,
    basis: PolynomialSubspaceBasis,
    /,
) -> AbstractLinearOperator:
    """Restrict an existing coefficient enforcement map to a verified basis."""

    _require_verified_basis(basis)
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not isinstance(operator.source, ArraySpace) or operator.source.shape != (
        basis.support.term_count,
    ):
        raise ValueError(
            "Polynomial lowering requires a flat enforcement-operator source "
            "matching term_count."
        )
    if operator.source.dtype != np.dtype(basis.basis.dtype):
        raise TypeError("Enforcement operator source and polynomial basis dtypes differ.")
    reduced_space = ArraySpace((basis.dimension,), dtype=basis.basis.dtype)
    embedding = DenseLinearOperator(
        basis.basis,
        source=reduced_space,
        target=operator.source,
        operator_id=canonical_fingerprint(
            {
                "kind": "polynomial-subspace-embedding-v1",
                "basis": basis.basis_id,
                "target": operator.source.space_id,
            }
        ),
    )
    return ComposedLinearOperator(operator, embedding)


def lower_polynomial_linear_representation(
    representation: AbstractLinearRepresentation,
    basis: PolynomialSubspaceBasis,
    /,
) -> CallableLinearRepresentation:
    """Restrict an existing callable/enforcement representation to ``basis``."""

    _require_verified_basis(basis)
    if not isinstance(representation, AbstractLinearRepresentation):
        raise TypeError("representation must be an AbstractLinearRepresentation.")
    if not isinstance(representation.coefficient_space, ArraySpace) or (
        representation.coefficient_space.shape != (basis.support.term_count,)
    ):
        raise ValueError(
            "Polynomial lowering requires a flat coefficient space matching term_count."
        )
    if representation.coefficient_space.dtype != np.dtype(basis.basis.dtype):
        raise TypeError("Representation coefficients and polynomial basis dtypes differ.")
    reduced_space = ArraySpace((basis.dimension,), dtype=basis.basis.dtype)
    certificate = LinearRepresentationCertificate(
        field_spec_id=representation.field_spec.field_spec_id,
        field_names=representation.field_spec.sources,
        native_coefficient_space_id=reduced_space.space_id,
        coefficient_space_id=reduced_space.space_id,
        extraction_id=canonical_fingerprint(
            {
                "kind": "polynomial-subspace-extraction-v1",
                "base": representation.certificate.extraction_id,
                "basis": basis.basis_id,
            }
        ),
        replacement_id=canonical_fingerprint(
            {
                "kind": "polynomial-subspace-replacement-v1",
                "base": representation.certificate.replacement_id,
                "basis": basis.basis_id,
            }
        ),
        synthesis_id=canonical_fingerprint(
            {
                "kind": "polynomial-subspace-synthesis-v1",
                "base": representation.certificate.synthesis_id,
                "basis": basis.basis_id,
            }
        ),
        support_ids=tuple(
            dict.fromkeys(
                (*representation.certificate.support_ids, basis.support.support_id)
            )
        ),
        layout_ids=representation.certificate.layout_ids,
        topology_ids=representation.certificate.topology_ids,
        maximum_derivative_orders=representation.certificate.maximum_derivative_orders,
        construction_dependencies=(
            *representation.certificate.construction_dependencies,
            "verified-polynomial-subspace",
        ),
        source_certificate_ids=tuple(
            dict.fromkeys(
                (
                    *representation.certificate.source_certificate_ids,
                    representation.representation_id,
                    basis.evidence.evidence_id,
                )
            )
        ),
        proof="verified-polynomial-subspace-numerical-lowering",
        zero_preserving=True,
        round_trip_exact=False,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "polynomial-subspace-linear-representation-v1",
            "base": representation.prepared_id,
            "basis": basis.basis_id,
            "certificate": certificate.representation_id,
            "numeric_version": representation.numeric_version,
        }
    )

    def extract(values):
        return basis.coordinates(representation.extract(values))

    def replace(values, coordinates):
        return representation.replace(values, basis.coefficients(coordinates))

    def synthesize(coordinates):
        return representation.synthesize(basis.coefficients(coordinates))

    def assemble(bound):
        full = representation.assemble(bound)
        operator = lower_polynomial_enforcement_operator(full.operator, basis)
        full_evidence = full.evidence
        evidence = LinearAssemblyEvidence(
            bound_condition_id=full_evidence.bound_condition_id,
            operator_id=operator.operator_id,
            coefficient_space_id=reduced_space.space_id,
            codomain_id=full_evidence.codomain_id,
            quantifier_id=full_evidence.quantifier_id,
            representation_id=certificate.representation_id,
            prepared_id=prepared_id,
            row_shape=full_evidence.row_shape,
            row_dtype=full_evidence.row_dtype,
            support_id=canonical_fingerprint(
                {
                    "kind": "polynomial-subspace-assembly-support-v1",
                    "base": full_evidence.support_id,
                    "polynomial": basis.support.support_id,
                }
            ),
            geometry_revision=full_evidence.geometry_revision,
            assembly_method="verified-polynomial-subspace-composition",
            exactness="numerical",
            numeric_fingerprint=canonical_fingerprint(
                {
                    "kind": "polynomial-subspace-assembly-numerics-v1",
                    "base": full_evidence.numeric_fingerprint,
                    "basis": basis.basis_id,
                }
            ),
            coordinate_evidence_id=full_evidence.coordinate_evidence_id,
            derivative_orders=full_evidence.derivative_orders,
            integration_evidence_ids=full_evidence.integration_evidence_ids,
            preserved_certificate_ids=tuple(
                dict.fromkeys(
                    (
                        *full_evidence.preserved_certificate_ids,
                        representation.representation_id,
                        basis.evidence.evidence_id,
                    )
                )
            ),
            error_bound=jnp.maximum(
                full_evidence.error_bound,
                basis.evidence.maximum_selected_residual,
            ),
            tolerance=jnp.maximum(
                full_evidence.tolerance,
                jnp.asarray(basis.evidence.verification_tolerance),
            ),
            zero_preserving=full_evidence.zero_preserving,
        )
        return LinearConditionAssembly(
            operator,
            evidence,
            codomain_coordinates=full.codomain_coordinates,
            numeric_version=representation.numeric_version,
        )

    return CallableLinearRepresentation(
        representation.field_spec,
        reduced_space,
        reduced_space,
        extract,
        replace,
        synthesize,
        assemble,
        certificate=certificate,
        numeric_version=representation.numeric_version,
        prepared_id=prepared_id,
    )


__all__ = [
    "DeclaredReductivePolynomialAction",
    "FinitePolynomialAction",
    "PolynomialActionConstraints",
    "PolynomialActionEvidence",
    "PolynomialScalingAction",
    "PolynomialSubspaceBasis",
    "PolynomialSubspaceEvidence",
    "casimir_isotypic_blocks",
    "extract_equivariant_subspace",
    "extract_invariant_subspace",
    "lower_polynomial_enforcement_operator",
    "lower_polynomial_linear_representation",
    "polynomial_action_constraints",
    "scaling_polynomial_action",
    "weighted_polynomial_action",
]
