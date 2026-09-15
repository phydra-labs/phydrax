#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


class LinkAlgebraEvidence(StrictModule):
    """Finite matrix evidence for one regulated link Hilbert space."""

    generator_commutator_residual: Array
    left_right_commutator_residual: Array
    casimir_residual: Array
    link_covariance_residual: Array
    finite: Array
    valid: Array
    dimension: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class LinkCutoffEvidence(StrictModule):
    """State-resolved evidence for a nonunitary hard representation cutoff."""

    boundary_probability: Array
    state_norm_residual: Array
    unitarity_defect_norm: Array
    state_defect_norm: Array
    finite: Array
    valid: Array
    cutoff_label: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class FiniteGroupLinkHilbertSpace(StrictModule):
    """Exact group-element link basis for an explicitly tabulated finite group."""

    multiplication_table: Array
    inverse_indices: Array
    left_translations: Array
    right_translations: Array
    evidence: LinkAlgebraEvidence
    labels: tuple[str, ...] = eqx.field(static=True)
    identity_index: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        multiplication_table: ArrayLike,
        /,
        *,
        labels: Sequence[str] | None = None,
        maximum_operator_elements: int = 1 << 26,
    ):
        table = np.asarray(multiplication_table)
        if (
            table.ndim != 2
            or table.shape[0] == 0
            or table.shape[0] != table.shape[1]
            or not np.issubdtype(table.dtype, np.integer)
        ):
            raise TypeError(
                "multiplication_table must be one nonempty square integer array."
            )
        order = int(table.shape[0])
        table = np.asarray(table, dtype=np.int32)
        if np.any(table < 0) or np.any(table >= order):
            raise ValueError("Finite-group products must be valid element indices.")
        if any(
            int(table[int(table[a, b]), c]) != int(table[a, int(table[b, c])])
            for a in range(order)
            for b in range(order)
            for c in range(order)
        ):
            raise ValueError("multiplication_table is not associative.")
        identities = tuple(
            candidate
            for candidate in range(order)
            if all(
                int(table[candidate, value]) == value
                and int(table[value, candidate]) == value
                for value in range(order)
            )
        )
        if len(identities) != 1:
            raise ValueError("multiplication_table must have exactly one identity.")
        identity = identities[0]
        inverse = []
        for value in range(order):
            candidates = tuple(
                candidate
                for candidate in range(order)
                if int(table[value, candidate]) == identity
                and int(table[candidate, value]) == identity
            )
            if len(candidates) != 1:
                raise ValueError("Every finite-group element must have one inverse.")
            inverse.append(candidates[0])
        labels_ = (
            tuple(f"g:{index}" for index in range(order))
            if labels is None
            else tuple(str(label) for label in labels)
        )
        if len(labels_) != order or any(not label for label in labels_):
            raise ValueError("labels must provide one nonempty label per group element.")
        if len(set(labels_)) != order:
            raise ValueError("Finite-group labels must be unique.")
        required = 2 * order**3
        maximum = int(maximum_operator_elements)
        if maximum <= 0 or required > maximum:
            raise ValueError(
                f"Finite-group translations require {required} elements; "
                f"capacity is {maximum}."
            )
        left = np.zeros((order, order, order), dtype=np.complex128)
        right = np.zeros_like(left)
        columns = np.arange(order)
        for element in range(order):
            left[element, table[element], columns] = 1.0
            right[element, table[columns, inverse[element]], columns] = 1.0
        left_jax = jnp.asarray(left)
        right_jax = jnp.asarray(right)
        identity_matrix = jnp.eye(order, dtype=jnp.complex128)
        unitary_residual = jnp.max(
            jnp.stack(
                tuple(
                    jnp.max(
                        jnp.abs(
                            jnp.conj(left_jax[element].T) @ left_jax[element]
                            - identity_matrix
                        )
                    )
                    for element in range(order)
                )
                + tuple(
                    jnp.max(
                        jnp.abs(
                            jnp.conj(right_jax[element].T) @ right_jax[element]
                            - identity_matrix
                        )
                    )
                    for element in range(order)
                )
            )
        )
        left_right = jnp.max(
            jnp.stack(
                tuple(
                    jnp.max(
                        jnp.abs(
                            left_jax[first] @ right_jax[second]
                            - right_jax[second] @ left_jax[first]
                        )
                    )
                    for first in range(order)
                    for second in range(order)
                )
            )
        )
        space_id = canonical_fingerprint(
            {
                "kind": "finite-group-link-hilbert-space",
                "labels": labels_,
                "multiplication_table": table,
            }
        )
        finite = jnp.all(jnp.isfinite(left_jax)) & jnp.all(jnp.isfinite(right_jax))
        evidence = LinkAlgebraEvidence(
            unitary_residual,
            left_right,
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            finite,
            finite & (unitary_residual <= 1e-12) & (left_right <= 1e-12),
            order,
            canonical_fingerprint(
                {"kind": "finite-group-link-algebra-evidence", "space": space_id}
            ),
        )
        self.multiplication_table = jnp.asarray(table)
        self.inverse_indices = jnp.asarray(inverse, dtype=jnp.int32)
        self.left_translations = left_jax
        self.right_translations = right_jax
        self.evidence = evidence
        self.labels = labels_
        self.identity_index = identity
        self.dimension = order
        self.space_id = space_id

    def left_translation(self, element: int, /) -> Array:
        index = int(element)
        if not 0 <= index < self.dimension:
            raise ValueError("Finite-group element index is out of range.")
        return self.left_translations[index]

    def right_translation(self, element: int, /) -> Array:
        index = int(element)
        if not 0 <= index < self.dimension:
            raise ValueError("Finite-group element index is out of range.")
        return self.right_translations[index]

    def representation_link_operator(self, representation: ArrayLike, /) -> Array:
        """Return matrix-valued multiplication U_ab in the group-element basis."""

        values = jnp.asarray(representation)
        if (
            values.ndim != 3
            or values.shape[0] != self.dimension
            or values.shape[1] != values.shape[2]
        ):
            raise ValueError("representation must have shape (group, d, d).")
        return jnp.stack(
            tuple(
                jnp.stack(
                    tuple(
                        jnp.diag(values[:, row, column])
                        for column in range(values.shape[2])
                    )
                )
                for row in range(values.shape[1])
            )
        )


def cyclic_group_link_hilbert(
    order: int, /, *, maximum_operator_elements: int = 1 << 26
) -> FiniteGroupLinkHilbertSpace:
    """Construct the exact regular link Hilbert space of Z_order."""

    value = int(order)
    if value < 2:
        raise ValueError("Cyclic group order must be at least two.")
    indices = np.arange(value, dtype=np.int32)
    table = (indices[:, None] + indices[None, :]) % value
    return FiniteGroupLinkHilbertSpace(
        table,
        labels=tuple(f"z{value}:{index}" for index in range(value)),
        maximum_operator_elements=maximum_operator_elements,
    )


class TruncatedU1LinkHilbertSpace(StrictModule):
    """Hard electric-flux truncation of one compact U(1) rotor link."""

    electric_levels: Array
    electric_field: Array
    electric_casimir: Array
    link_operator: Array
    link_adjoint: Array
    boundary_projector: Array
    algebra: LinkAlgebraEvidence
    maximum_flux: int = eqx.field(static=True)
    center_flux: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_flux: int,
        /,
        *,
        center_flux: int = 0,
        maximum_operator_elements: int = 1 << 26,
    ):
        cutoff = int(maximum_flux)
        center = int(center_flux)
        if cutoff < 1:
            raise ValueError("maximum_flux must be at least one.")
        dimension = 2 * cutoff + 1
        required = 7 * dimension * dimension
        maximum = int(maximum_operator_elements)
        if maximum <= 0 or required > maximum:
            raise ValueError(
                f"Truncated U(1) link operators require {required} elements; "
                f"capacity is {maximum}."
            )
        levels = jnp.arange(center - cutoff, center + cutoff + 1, dtype=jnp.float64)
        electric = jnp.diag(levels).astype(jnp.complex128)
        link = jnp.diag(jnp.ones((dimension - 1,), dtype=jnp.complex128), -1)
        adjoint = jnp.conj(link.T)
        casimir = electric @ electric
        boundary = jnp.zeros((dimension, dimension), dtype=jnp.complex128)
        boundary = boundary.at[0, 0].set(1.0).at[-1, -1].set(1.0)
        commutator = jnp.max(jnp.abs(electric @ link - link @ electric - link))
        unitarity_defect = jnp.max(jnp.abs(jnp.conj(link.T) @ link - jnp.eye(dimension)))
        space_id = canonical_fingerprint(
            {
                "kind": "truncated-u1-link-hilbert-space",
                "maximum_flux": cutoff,
                "center_flux": center,
                "dimension": dimension,
            }
        )
        finite = jnp.all(jnp.isfinite(electric)) & jnp.all(jnp.isfinite(link))
        self.electric_levels = levels
        self.electric_field = electric
        self.electric_casimir = casimir
        self.link_operator = link
        self.link_adjoint = adjoint
        self.boundary_projector = boundary
        self.algebra = LinkAlgebraEvidence(
            commutator,
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            commutator,
            finite,
            finite & (commutator <= 1e-12),
            dimension,
            canonical_fingerprint(
                {"kind": "truncated-u1-link-algebra", "space": space_id}
            ),
        )
        self.maximum_flux = cutoff
        self.center_flux = center
        self.dimension = dimension
        self.space_id = space_id

    def cutoff_evidence(
        self,
        state: ArrayLike,
        /,
        *,
        tolerance: float = 1e-8,
    ) -> LinkCutoffEvidence:
        vector = jnp.asarray(state)
        if vector.shape != (self.dimension,):
            raise ValueError("state must have the truncated U(1) link dimension.")
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        norm = jnp.real(jnp.vdot(vector, vector))
        norm_residual = jnp.abs(norm - 1.0)
        boundary_probability = jnp.real(
            jnp.vdot(vector, self.boundary_projector @ vector)
        )
        defect = (
            jnp.eye(self.dimension, dtype=vector.dtype)
            - self.link_adjoint @ self.link_operator
        )
        defect_norm = jnp.linalg.norm(defect)
        state_defect = jnp.linalg.norm(defect @ vector)
        finite = (
            jnp.all(jnp.isfinite(vector))
            & jnp.isfinite(boundary_probability)
            & jnp.isfinite(state_defect)
        )
        valid = (
            finite & (norm_residual <= tolerance_) & (boundary_probability <= tolerance_)
        )
        return LinkCutoffEvidence(
            boundary_probability,
            norm_residual,
            defect_norm,
            state_defect,
            finite,
            valid,
            f"u1-electric-abs-m-minus-center<={self.maximum_flux}",
            canonical_fingerprint(
                {
                    "kind": "truncated-u1-cutoff-evidence",
                    "space": self.space_id,
                    "tolerance": tolerance_,
                }
            ),
        )


def _spin_matrices(twice_spin: int, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = twice_spin + 1
    spin = 0.5 * twice_spin
    magnetic = np.arange(-twice_spin, twice_spin + 1, 2, dtype=float) / 2.0
    raising = np.zeros((dimension, dimension), dtype=np.complex128)
    for column, value in enumerate(magnetic[:-1]):
        raising[column + 1, column] = np.sqrt(spin * (spin + 1.0) - value * (value + 1.0))
    lowering = raising.T
    x = 0.5 * (raising + lowering)
    y = (raising - lowering) / (2.0j)
    z = np.diag(magnetic).astype(np.complex128)
    return x, y, z


def _half_clebsch(
    twice_spin: int,
    twice_magnetic: int,
    spinor_index: int,
    target_twice_spin: int,
    /,
) -> float:
    denominator = 2.0 * (twice_spin + 1)
    if spinor_index == 1:
        numerator = twice_spin + twice_magnetic + 2
        sign = 1.0 if target_twice_spin == twice_spin + 1 else -1.0
    else:
        numerator = twice_spin - twice_magnetic + 2
        sign = 1.0
    if target_twice_spin == twice_spin - 1:
        numerator = (
            twice_spin - twice_magnetic
            if spinor_index == 1
            else twice_spin + twice_magnetic
        )
    if numerator < 0:
        return 0.0
    return sign * float(np.sqrt(numerator / denominator))


class SU2IrrepTruncatedLinkHilbertSpace(StrictModule):
    """Peter--Weyl SU(2) link basis truncated at one maximum doubled spin."""

    twice_spins: Array
    left_magnetic: Array
    right_magnetic: Array
    block_offsets: Array
    left_electric_generators: Array
    right_electric_generators: Array
    electric_casimir: Array
    fundamental_link_operator: Array
    boundary_projector: Array
    algebra: LinkAlgebraEvidence
    maximum_twice_spin: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_twice_spin: int,
        /,
        *,
        maximum_dimension: int = 4096,
        maximum_operator_elements: int = 1 << 27,
    ):
        maximum_spin = int(maximum_twice_spin)
        if maximum_spin < 0:
            raise ValueError("maximum_twice_spin must be non-negative.")
        dimensions = tuple(spin + 1 for spin in range(maximum_spin + 1))
        dimension = sum(value * value for value in dimensions)
        if int(maximum_dimension) <= 0 or dimension > int(maximum_dimension):
            raise ValueError(
                f"SU(2) link dimension {dimension} exceeds maximum_dimension "
                f"{int(maximum_dimension)}."
            )
        required = (6 + 4) * dimension * dimension
        if int(maximum_operator_elements) <= 0 or required > int(
            maximum_operator_elements
        ):
            raise ValueError(
                f"SU(2) link operators require {required} elements; capacity is "
                f"{int(maximum_operator_elements)}."
            )
        offsets = [0]
        twice_spins: list[int] = []
        left_magnetic: list[int] = []
        right_magnetic: list[int] = []
        for twice_spin, block_dimension in enumerate(dimensions):
            for left in range(-twice_spin, twice_spin + 1, 2):
                for right in range(-twice_spin, twice_spin + 1, 2):
                    twice_spins.append(twice_spin)
                    left_magnetic.append(left)
                    right_magnetic.append(right)
            offsets.append(offsets[-1] + block_dimension * block_dimension)
        left_generators = np.zeros((3, dimension, dimension), dtype=np.complex128)
        right_generators = np.zeros_like(left_generators)
        casimir = np.zeros((dimension, dimension), dtype=np.complex128)
        for twice_spin, block_dimension in enumerate(dimensions):
            start, stop = offsets[twice_spin], offsets[twice_spin + 1]
            x, y, z = _spin_matrices(twice_spin)
            identity = np.eye(block_dimension, dtype=np.complex128)
            for generator, matrix in enumerate((x, y, z)):
                left_generators[generator, start:stop, start:stop] = np.kron(
                    matrix, identity
                )
                right_generators[generator, start:stop, start:stop] = np.kron(
                    identity, matrix
                )
            spin = 0.5 * twice_spin
            casimir[start:stop, start:stop] = (
                spin * (spin + 1.0) * np.eye(block_dimension * block_dimension)
            )
        basis_lookup = {
            (spin, left, right): index
            for index, (spin, left, right) in enumerate(
                zip(twice_spins, left_magnetic, right_magnetic, strict=True)
            )
        }
        link = np.zeros((2, 2, dimension, dimension), dtype=np.complex128)
        for source, (spin, left, right) in enumerate(
            zip(twice_spins, left_magnetic, right_magnetic, strict=True)
        ):
            for left_spinor in range(2):
                left_delta = -1 if left_spinor == 0 else 1
                for right_spinor in range(2):
                    right_delta = -1 if right_spinor == 0 else 1
                    for target_spin in (spin - 1, spin + 1):
                        target_left = left + left_delta
                        target_right = right + right_delta
                        key = (target_spin, target_left, target_right)
                        if key not in basis_lookup:
                            continue
                        coefficient = _half_clebsch(
                            spin, left, left_spinor, target_spin
                        ) * _half_clebsch(spin, right, right_spinor, target_spin)
                        coefficient *= np.sqrt((spin + 1.0) / (target_spin + 1.0))
                        link[
                            left_spinor,
                            right_spinor,
                            basis_lookup[key],
                            source,
                        ] = coefficient
        left_jax = jnp.asarray(left_generators)
        right_jax = jnp.asarray(right_generators)
        casimir_jax = jnp.asarray(casimir)
        link_jax = jnp.asarray(link)
        boundary = jnp.diag(
            jnp.asarray(np.asarray(twice_spins) == maximum_spin, dtype=jnp.complex128)
        )
        epsilon = np.zeros((3, 3, 3), dtype=float)
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
        epsilon[1, 0, 2] = epsilon[2, 1, 0] = epsilon[0, 2, 1] = -1.0
        generator_residuals = []
        for first in range(3):
            for second in range(3):
                expected_left = sum(
                    1j * epsilon[first, second, third] * left_jax[third]
                    for third in range(3)
                )
                expected_right = sum(
                    1j * epsilon[first, second, third] * right_jax[third]
                    for third in range(3)
                )
                generator_residuals.extend(
                    (
                        jnp.max(
                            jnp.abs(
                                left_jax[first] @ left_jax[second]
                                - left_jax[second] @ left_jax[first]
                                - expected_left
                            )
                        ),
                        jnp.max(
                            jnp.abs(
                                right_jax[first] @ right_jax[second]
                                - right_jax[second] @ right_jax[first]
                                - expected_right
                            )
                        ),
                    )
                )
        left_right = jnp.max(
            jnp.stack(
                tuple(
                    jnp.max(
                        jnp.abs(
                            left_jax[first] @ right_jax[second]
                            - right_jax[second] @ left_jax[first]
                        )
                    )
                    for first in range(3)
                    for second in range(3)
                )
            )
        )
        left_casimir = sum(matrix @ matrix for matrix in left_jax)
        right_casimir = sum(matrix @ matrix for matrix in right_jax)
        casimir_residual = jnp.maximum(
            jnp.max(jnp.abs(left_casimir - casimir_jax)),
            jnp.max(jnp.abs(right_casimir - casimir_jax)),
        )
        generator_residual = jnp.max(jnp.stack(generator_residuals))
        fundamental_generators = tuple(jnp.asarray(value) for value in _spin_matrices(1))
        covariance_residuals = []
        for generator in range(3):
            for left_spinor in range(2):
                for right_spinor in range(2):
                    component = link_jax[left_spinor, right_spinor]
                    left_commutator = (
                        left_jax[generator] @ component - component @ left_jax[generator]
                    )
                    left_expected = sum(
                        fundamental_generators[generator][other, left_spinor]
                        * link_jax[other, right_spinor]
                        for other in range(2)
                    )
                    right_commutator = (
                        right_jax[generator] @ component
                        - component @ right_jax[generator]
                    )
                    right_expected = sum(
                        link_jax[left_spinor, other]
                        * fundamental_generators[generator][other, right_spinor]
                        for other in range(2)
                    )
                    covariance_residuals.extend(
                        (
                            jnp.max(jnp.abs(left_commutator - left_expected)),
                            jnp.max(jnp.abs(right_commutator - right_expected)),
                        )
                    )
        covariance_residual = jnp.max(jnp.stack(covariance_residuals))
        space_id = canonical_fingerprint(
            {
                "kind": "su2-irrep-truncated-link-hilbert-space",
                "maximum_twice_spin": maximum_spin,
                "dimension": dimension,
                "basis": tuple(
                    zip(twice_spins, left_magnetic, right_magnetic, strict=True)
                ),
            }
        )
        finite = (
            jnp.all(jnp.isfinite(left_jax))
            & jnp.all(jnp.isfinite(right_jax))
            & jnp.all(jnp.isfinite(link_jax))
        )
        self.twice_spins = jnp.asarray(twice_spins, dtype=jnp.int32)
        self.left_magnetic = jnp.asarray(left_magnetic, dtype=jnp.int32)
        self.right_magnetic = jnp.asarray(right_magnetic, dtype=jnp.int32)
        self.block_offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.left_electric_generators = left_jax
        self.right_electric_generators = right_jax
        self.electric_casimir = casimir_jax
        self.fundamental_link_operator = link_jax
        self.boundary_projector = boundary
        self.algebra = LinkAlgebraEvidence(
            generator_residual,
            left_right,
            casimir_residual,
            covariance_residual,
            finite,
            finite
            & (generator_residual <= 1e-10)
            & (left_right <= 1e-10)
            & (casimir_residual <= 1e-10)
            & (covariance_residual <= 1e-10),
            dimension,
            canonical_fingerprint(
                {"kind": "su2-link-algebra-evidence", "space": space_id}
            ),
        )
        self.maximum_twice_spin = maximum_spin
        self.dimension = dimension
        self.space_id = space_id

    def link_component(self, left_spinor: int, right_spinor: int, /) -> Array:
        left, right = int(left_spinor), int(right_spinor)
        if left not in (0, 1) or right not in (0, 1):
            raise ValueError("Fundamental SU(2) spinor indices must be zero or one.")
        return self.fundamental_link_operator[left, right]

    def cutoff_evidence(
        self,
        state: ArrayLike,
        /,
        *,
        tolerance: float = 1e-8,
    ) -> LinkCutoffEvidence:
        vector = jnp.asarray(state)
        if vector.shape != (self.dimension,):
            raise ValueError("state must have the truncated SU(2) link dimension.")
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        norm = jnp.real(jnp.vdot(vector, vector))
        norm_residual = jnp.abs(norm - 1.0)
        boundary_probability = jnp.real(
            jnp.vdot(vector, self.boundary_projector @ vector)
        )
        identity = jnp.eye(self.dimension, dtype=vector.dtype)
        defects = []
        for right in range(2):
            for column in range(2):
                gram = sum(
                    jnp.conj(self.fundamental_link_operator[left, right].T)
                    @ self.fundamental_link_operator[left, column]
                    for left in range(2)
                )
                defects.append(gram - (identity if right == column else 0.0))
        defect = jnp.stack(defects)
        defect_norm = jnp.linalg.norm(defect)
        state_defect = jnp.max(
            jnp.stack(tuple(jnp.linalg.norm(value @ vector) for value in defects))
        )
        finite = (
            jnp.all(jnp.isfinite(vector))
            & jnp.isfinite(boundary_probability)
            & jnp.isfinite(defect_norm)
            & jnp.isfinite(state_defect)
        )
        return LinkCutoffEvidence(
            boundary_probability,
            norm_residual,
            defect_norm,
            state_defect,
            finite,
            finite & (norm_residual <= tolerance_) & (boundary_probability <= tolerance_),
            f"su2-twice-spin<={self.maximum_twice_spin}",
            canonical_fingerprint(
                {
                    "kind": "su2-link-cutoff-evidence",
                    "space": self.space_id,
                    "tolerance": tolerance_,
                }
            ),
        )


__all__ = [
    "FiniteGroupLinkHilbertSpace",
    "LinkAlgebraEvidence",
    "LinkCutoffEvidence",
    "SU2IrrepTruncatedLinkHilbertSpace",
    "TruncatedU1LinkHilbertSpace",
    "cyclic_group_link_hilbert",
]
