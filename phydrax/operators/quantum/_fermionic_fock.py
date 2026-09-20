#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


class FermionModeOrder(StrictModule):
    """Canonical global mode order used for every fermionic sign decision."""

    labels: tuple[str, ...] = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    order_id: str = eqx.field(static=True)

    def __init__(self, labels: Sequence[str], /):
        values = tuple(str(label) for label in labels)
        if not values or any(not label for label in values):
            raise ValueError("Fermion mode labels must be nonempty.")
        if len(set(values)) != len(values):
            raise ValueError("Fermion mode labels must be unique.")
        self.labels = values
        self.mode_count = len(values)
        self.order_id = canonical_fingerprint(
            {"kind": "fermion-mode-order", "labels": values}
        )

    def ordinal(self, label: str, /) -> int:
        value = str(label)
        if value not in self.labels:
            raise ValueError(f"Fermion mode {value!r} is absent from the order.")
        return self.labels.index(value)

    def permutation(self, target: FermionModeOrder, /) -> tuple[int, ...]:
        """Return source ordinals in target order, rejecting different mode sets."""

        if not isinstance(target, FermionModeOrder):
            raise TypeError("target must be FermionModeOrder.")
        if set(target.labels) != set(self.labels):
            raise ValueError("Fermion mode orders must contain exactly the same labels.")
        return tuple(self.ordinal(label) for label in target.labels)

    def permutation_sign(
        self, target: FermionModeOrder, occupations: Sequence[int], /
    ) -> int:
        """Sign induced by reordering the occupied exterior-product modes."""

        permutation = self.permutation(target)
        values = tuple(occupations)
        if len(values) != self.mode_count or any(value not in (0, 1) for value in values):
            raise ValueError("occupations must contain one binary value per mode.")
        occupied_source = tuple(index for index, value in enumerate(values) if value)
        target_positions = tuple(permutation.index(index) for index in occupied_source)
        inversions = sum(
            target_positions[left] > target_positions[right]
            for left in range(len(target_positions))
            for right in range(left + 1, len(target_positions))
        )
        return -1 if inversions % 2 else 1


class FermionicFockBasis(StrictModule):
    """Occupation basis inseparably tied to one canonical fermion mode order."""

    mode_order: FermionModeOrder = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(self, mode_order: FermionModeOrder, /):
        if not isinstance(mode_order, FermionModeOrder):
            raise TypeError("mode_order must be FermionModeOrder.")
        self.mode_order = mode_order
        self.mode_count = mode_order.mode_count
        self.dimension = 1 << mode_order.mode_count
        self.basis_id = canonical_fingerprint(
            {"kind": "fermionic-fock-basis", "mode_order": mode_order.order_id}
        )

    def basis_index(self, occupations: Sequence[int], /) -> int:
        values = tuple(occupations)
        if len(values) != self.mode_count or any(value not in (0, 1) for value in values):
            raise ValueError("occupations must contain one binary value per mode.")
        index = 0
        for value in values:
            index = 2 * index + value
        return index

    def occupation_tuple(self, index: int, /) -> tuple[int, ...]:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("index must be an integer.")
        value = int(index)
        if not 0 <= value < self.dimension:
            raise ValueError("Fock basis index is out of range.")
        return tuple(
            (value >> (self.mode_count - site - 1)) & 1 for site in range(self.mode_count)
        )

    def occupations(self, /, *, maximum_basis_states: int = 1 << 20) -> Array:
        maximum = int(maximum_basis_states)
        if maximum <= 0 or self.dimension > maximum:
            raise ValueError(
                f"Fock enumeration requires {self.dimension} basis states; capacity is {maximum}."
            )
        indices = jnp.arange(self.dimension, dtype=jnp.uint32)
        shifts = jnp.arange(self.mode_count - 1, -1, -1, dtype=jnp.uint32)
        return ((indices[:, None] >> shifts[None, :]) & 1).astype(jnp.int32)


class FermionLadderOperator(StrictModule):
    """One creation or annihilation symbol in a declared mode order."""

    mode_order: FermionModeOrder = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    action: str = eqx.field(static=True)
    mode_index: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(self, mode_order: FermionModeOrder, mode: str, action: str, /):
        if not isinstance(mode_order, FermionModeOrder):
            raise TypeError("mode_order must be FermionModeOrder.")
        action_ = str(action)
        if action_ not in ("create", "annihilate"):
            raise ValueError("action must be create or annihilate.")
        mode_ = str(mode)
        index = mode_order.ordinal(mode_)
        self.mode_order = mode_order
        self.mode = mode_
        self.action = action_
        self.mode_index = index
        self.operator_id = canonical_fingerprint(
            {
                "kind": "fermion-ladder-symbol",
                "mode_order": mode_order.order_id,
                "mode": mode_,
                "action": action_,
            }
        )

    def adjoint(self, /) -> FermionLadderOperator:
        return FermionLadderOperator(
            self.mode_order,
            self.mode,
            "annihilate" if self.action == "create" else "create",
        )

    def dense_matrix(self, /, *, maximum_elements: int = 1 << 26) -> Array:
        return fermion_ladder_matrix(
            FermionicFockBasis(self.mode_order),
            self.mode,
            self.action,
            maximum_elements=maximum_elements,
        )


class CARMonomial(StrictModule):
    """Ordered product of fermion ladder symbols; the rightmost acts first."""

    mode_order: FermionModeOrder = eqx.field(static=True)
    operations: tuple[FermionLadderOperator, ...]
    monomial_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_order: FermionModeOrder,
        operations: Sequence[FermionLadderOperator | tuple[str, str]] = (),
        /,
    ):
        if not isinstance(mode_order, FermionModeOrder):
            raise TypeError("mode_order must be FermionModeOrder.")
        selected: list[FermionLadderOperator] = []
        for operation in operations:
            if isinstance(operation, FermionLadderOperator):
                if operation.mode_order.order_id != mode_order.order_id:
                    raise ValueError(
                        "All ladder symbols must use the monomial mode order."
                    )
                selected.append(operation)
            else:
                mode, action = operation
                selected.append(FermionLadderOperator(mode_order, mode, action))
        values = tuple(selected)
        self.mode_order = mode_order
        self.operations = values
        self.monomial_id = canonical_fingerprint(
            {
                "kind": "car-monomial",
                "mode_order": mode_order.order_id,
                "operations": tuple(
                    (operation.mode, operation.action) for operation in values
                ),
            }
        )

    def adjoint(self, /) -> CARMonomial:
        return CARMonomial(
            self.mode_order,
            tuple(operation.adjoint() for operation in reversed(self.operations)),
        )

    def dense_matrix(self, /, *, maximum_elements: int = 1 << 26) -> Array:
        dimension = 1 << self.mode_order.mode_count
        required = dimension * dimension
        maximum = int(maximum_elements)
        if maximum <= 0 or required > maximum:
            raise ValueError(
                f"CAR monomial materialization requires {required} elements; capacity is {maximum}."
            )
        result = jnp.eye(dimension, dtype=jnp.complex128)
        basis = FermionicFockBasis(self.mode_order)
        for operation in self.operations:
            result = result @ fermion_ladder_matrix(
                basis,
                operation.mode,
                operation.action,
                maximum_elements=maximum,
            )
        return result


class CARPolynomial(StrictModule):
    """Finite canonical sum of distinct ordered CAR monomials."""

    mode_order: FermionModeOrder = eqx.field(static=True)
    monomials: tuple[CARMonomial, ...]
    coefficients: Array
    polynomial_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_order: FermionModeOrder,
        terms: Sequence[tuple[ArrayLike, CARMonomial | Sequence[tuple[str, str]]]],
        /,
    ):
        if not isinstance(mode_order, FermionModeOrder):
            raise TypeError("mode_order must be FermionModeOrder.")
        values = tuple(terms)
        if not values:
            raise ValueError("A CAR polynomial requires at least one term.")
        records: list[tuple[tuple[tuple[str, str], ...], Array]] = []
        for coefficient, monomial in values:
            resolved = (
                monomial
                if isinstance(monomial, CARMonomial)
                else CARMonomial(mode_order, monomial)
            )
            if resolved.mode_order.order_id != mode_order.order_id:
                raise ValueError("Every CAR monomial must use the polynomial mode order.")
            scalar = jnp.asarray(coefficient)
            if scalar.shape != ():
                raise ValueError("Every CAR polynomial coefficient must be scalar.")
            key = tuple(
                (operation.mode, operation.action) for operation in resolved.operations
            )
            records.append((key, scalar))
        keys = tuple(sorted({record[0] for record in records}))
        monomials = tuple(CARMonomial(mode_order, key) for key in keys)
        coefficients = jnp.stack(
            tuple(
                sum(
                    (
                        coefficient
                        for record_key, coefficient in records
                        if record_key == key
                    ),
                    jnp.asarray(0.0),
                )
                for key in keys
            )
        )
        if not jnp.issubdtype(coefficients.dtype, jnp.complexfloating):
            coefficients = coefficients.astype(jnp.result_type(coefficients.dtype, 1j))
        self.mode_order = mode_order
        self.monomials = monomials
        self.coefficients = coefficients
        self.polynomial_id = canonical_fingerprint(
            {
                "kind": "car-polynomial",
                "mode_order": mode_order.order_id,
                "monomials": tuple(monomial.monomial_id for monomial in monomials),
                "coefficient_shape": coefficients.shape,
                "coefficient_dtype": str(coefficients.dtype),
            }
        )

    def dense_matrix(self, /, *, maximum_elements: int = 1 << 26) -> Array:
        dimension = 1 << self.mode_order.mode_count
        required = dimension * dimension
        maximum = int(maximum_elements)
        if maximum <= 0 or required > maximum:
            raise ValueError(
                f"CAR polynomial materialization requires {required} elements; capacity is {maximum}."
            )
        result = jnp.zeros((dimension, dimension), dtype=self.coefficients.dtype)
        for coefficient, monomial in zip(self.coefficients, self.monomials, strict=True):
            result = result + coefficient * monomial.dense_matrix(
                maximum_elements=maximum
            )
        return result

    def apply(self, state: ArrayLike, /, *, maximum_elements: int = 1 << 26) -> Array:
        vector = jnp.asarray(state)
        dimension = 1 << self.mode_order.mode_count
        if vector.shape[-1:] != (dimension,):
            raise ValueError("state has the wrong trailing Fock dimension.")
        return vector @ self.dense_matrix(maximum_elements=maximum_elements).T

    def adjoint(self, /) -> CARPolynomial:
        return CARPolynomial(
            self.mode_order,
            tuple(
                (jnp.conj(coefficient), monomial.adjoint())
                for coefficient, monomial in zip(
                    self.coefficients, self.monomials, strict=True
                )
            ),
        )


class CAREvidence(StrictModule):
    """Numerical evidence for canonical anticommutation relations."""

    annihilation_residual: Array
    mixed_residual: Array
    permutation_unitarity_residual: Array
    valid: Array
    mode_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def fermion_ladder_matrix(
    basis: FermionicFockBasis,
    mode: str | int,
    action: str,
    /,
    *,
    maximum_elements: int = 1 << 26,
) -> Array:
    """Materialize one exact Jordan--Wigner ladder matrix in the declared basis."""

    if not isinstance(basis, FermionicFockBasis):
        raise TypeError("basis must be FermionicFockBasis.")
    index = (
        int(mode)
        if isinstance(mode, Integral) and not isinstance(mode, bool)
        else basis.mode_order.ordinal(str(mode))
    )
    if not 0 <= index < basis.mode_count:
        raise ValueError("Fermion mode index is out of range.")
    action_ = str(action)
    if action_ not in ("create", "annihilate"):
        raise ValueError("action must be create or annihilate.")
    required = basis.dimension * basis.dimension
    maximum = int(maximum_elements)
    if maximum <= 0 or required > maximum:
        raise ValueError(
            f"Fermion ladder materialization requires {required} elements; capacity is {maximum}."
        )
    columns = np.arange(basis.dimension, dtype=np.int64)
    bit_shift = basis.mode_count - index - 1
    occupied = (columns >> bit_shift) & 1
    eligible = occupied == (0 if action_ == "create" else 1)
    rows = columns ^ (1 << bit_shift)
    earlier_mask = sum(1 << shift for shift in range(bit_shift + 1, basis.mode_count))
    parity = np.asarray(
        [int(int(column & earlier_mask).bit_count() % 2) for column in columns],
        dtype=np.int8,
    )
    signs = 1 - 2 * parity
    matrix = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
    matrix[rows[eligible], columns[eligible]] = signs[eligible]
    return jnp.asarray(matrix)


def fermion_mode_permutation_matrix(
    source: FermionModeOrder,
    target: FermionModeOrder,
    /,
    *,
    maximum_elements: int = 1 << 26,
) -> Array:
    """Unitary exterior-algebra coordinate map between two mode orders."""

    if not isinstance(source, FermionModeOrder) or not isinstance(
        target, FermionModeOrder
    ):
        raise TypeError("source and target must be FermionModeOrder values.")
    permutation = source.permutation(target)
    basis = FermionicFockBasis(source)
    required = basis.dimension * basis.dimension
    maximum = int(maximum_elements)
    if maximum <= 0 or required > maximum:
        raise ValueError(
            f"Mode permutation materialization requires {required} elements; capacity is {maximum}."
        )
    matrix = np.zeros((basis.dimension, basis.dimension), dtype=np.float64)
    target_basis = FermionicFockBasis(target)
    for source_index in range(basis.dimension):
        occupations = basis.occupation_tuple(source_index)
        reordered = tuple(occupations[index] for index in permutation)
        target_index = target_basis.basis_index(reordered)
        matrix[target_index, source_index] = source.permutation_sign(target, occupations)
    return jnp.asarray(matrix, dtype=jnp.complex128)


def car_evidence(
    mode_order: FermionModeOrder,
    /,
    *,
    maximum_elements: int = 1 << 26,
    tolerance: float = 1e-10,
) -> CAREvidence:
    """Compute all finite CAR residuals for one mode order."""

    if not isinstance(mode_order, FermionModeOrder):
        raise TypeError("mode_order must be FermionModeOrder.")
    basis = FermionicFockBasis(mode_order)
    required = (mode_order.mode_count + 4) * basis.dimension * basis.dimension
    if int(maximum_elements) <= 0 or required > int(maximum_elements):
        raise ValueError(
            f"CAR evidence requires at most {required} resident elements; capacity is {int(maximum_elements)}."
        )
    annihilation = tuple(
        fermion_ladder_matrix(
            basis, index, "annihilate", maximum_elements=maximum_elements
        )
        for index in range(mode_order.mode_count)
    )
    identity = jnp.eye(basis.dimension, dtype=jnp.complex128)
    zero_residuals = []
    mixed_residuals = []
    for left in range(mode_order.mode_count):
        for right in range(mode_order.mode_count):
            first = annihilation[left]
            second = annihilation[right]
            zero_residuals.append(jnp.max(jnp.abs(first @ second + second @ first)))
            mixed = first @ jnp.conj(second.T) + jnp.conj(second.T) @ first
            expected = identity if left == right else jnp.zeros_like(identity)
            mixed_residuals.append(jnp.max(jnp.abs(mixed - expected)))
    reversed_order = FermionModeOrder(tuple(reversed(mode_order.labels)))
    permutation = fermion_mode_permutation_matrix(
        mode_order, reversed_order, maximum_elements=maximum_elements
    )
    permutation_residual = jnp.max(
        jnp.abs(jnp.conj(permutation.T) @ permutation - identity)
    )
    annihilation_residual = jnp.max(jnp.stack(zero_residuals))
    mixed_residual = jnp.max(jnp.stack(mixed_residuals))
    tolerance_ = float(tolerance)
    valid = (
        jnp.isfinite(annihilation_residual)
        & jnp.isfinite(mixed_residual)
        & jnp.isfinite(permutation_residual)
        & (annihilation_residual <= tolerance_)
        & (mixed_residual <= tolerance_)
        & (permutation_residual <= tolerance_)
    )
    return CAREvidence(
        annihilation_residual,
        mixed_residual,
        permutation_residual,
        valid,
        mode_order.mode_count,
        canonical_fingerprint(
            {
                "kind": "car-evidence",
                "mode_order": mode_order.order_id,
                "tolerance": tolerance_,
            }
        ),
    )


__all__ = [
    "CAREvidence",
    "CARMonomial",
    "CARPolynomial",
    "FermionLadderOperator",
    "FermionModeOrder",
    "FermionicFockBasis",
    "car_evidence",
    "fermion_ladder_matrix",
    "fermion_mode_permutation_matrix",
]
