#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-group matrix irreps, deterministic multiplicities, and covariant maps."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._limit_study import (
    run_scientific_limit_study,
    ScientificLimitAxis,
    ScientificLimitDatum,
    ScientificLimitStudyPlan,
    ScientificLimitStudyResult,
    ScientificLimitVariation,
)
from ...._strict import StrictModule
from ._orbit_sector import (
    CharacterSectorPlan,
    FiniteGroupActionPlan,
    prepare_finite_group_action,
    PreparedFiniteGroupAction,
)


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _unitary_residual(matrix: np.ndarray, /) -> float:
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return float(np.linalg.norm(matrix.conj().T @ matrix - identity))


def _compose_corepresentations(
    left: np.ndarray,
    left_antiunitary: bool,
    right: np.ndarray,
    right_antiunitary: bool,
    /,
) -> tuple[np.ndarray, bool]:
    return (
        left @ (np.conj(right) if left_antiunitary else right),
        bool(left_antiunitary) ^ bool(right_antiunitary),
    )


class FiniteGroupIrrepPlan(StrictModule):
    """Generator realization for one unitary irrep or antiunitary corepresentation."""

    label: str = eqx.field(static=True)
    generator_labels: tuple[str, ...] = eqx.field(static=True)
    generator_matrices: Array
    generator_antiunitary: tuple[bool, ...] = eqx.field(static=True)
    projective: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    irrep_dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        generator_matrices: Mapping[str, ArrayLike],
        /,
        *,
        antiunitary_generators: Sequence[str] = (),
        projective: bool = False,
        tolerance: float = 1e-10,
    ):
        label_ = _identifier(label, "irrep label")
        if not isinstance(generator_matrices, Mapping) or not generator_matrices:
            raise TypeError("generator_matrices must be a non-empty mapping.")
        values = tuple(
            sorted(
                (
                    _identifier(name, "generator label"),
                    np.asarray(matrix, dtype=np.complex128),
                )
                for name, matrix in generator_matrices.items()
            )
        )
        dimension = int(values[0][1].shape[0])
        if dimension < 1 or any(
            matrix.shape != (dimension, dimension) for _, matrix in values
        ):
            raise ValueError("Every generator matrix must have one common square shape.")
        tolerance_ = float(tolerance)
        if not math.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be positive and finite.")
        residuals = tuple(_unitary_residual(matrix) for _, matrix in values)
        if any(value > tolerance_ for value in residuals):
            raise ValueError("Every finite-group generator matrix must be unitary.")
        antiunitary = frozenset(
            _identifier(name, "antiunitary generator") for name in antiunitary_generators
        )
        labels = tuple(name for name, _ in values)
        if not antiunitary <= set(labels):
            raise ValueError(
                "Antiunitary generator labels must name supplied generators."
            )
        matrices = np.stack(tuple(matrix for _, matrix in values), axis=0)
        flags = tuple(name in antiunitary for name in labels)
        content = {
            "kind": "finite-group-irrep-plan",
            "label": label_,
            "generator_labels": labels,
            "generator_matrices": array_tree_fingerprint(matrices),
            "generator_antiunitary": flags,
            "projective": bool(projective),
            "tolerance": tolerance_,
        }
        self.label = label_
        self.generator_labels = labels
        self.generator_matrices = jnp.asarray(matrices)
        self.generator_antiunitary = flags
        self.projective = bool(projective)
        self.tolerance = tolerance_
        self.irrep_dimension = dimension
        self.plan_id = canonical_fingerprint(content)


class FiniteGroupIrrepEvidence(StrictModule):
    unitarity_residuals: Array
    multiplication_residual: Array
    cocycle_residual: Array
    maximum_cocycle_modulus_error: Array
    contains_antiunitary: bool = eqx.field(static=True)
    projective: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedFiniteGroupIrrep(StrictModule):
    """Complete group realization aligned with one prepared monomial action."""

    action: PreparedFiniteGroupAction
    plan: FiniteGroupIrrepPlan
    matrices: Array
    antiunitary: Array
    multiplication_table: Array
    cocycle: Array
    evidence: FiniteGroupIrrepEvidence
    prepared_id: str = eqx.field(static=True)


def _action_product_table(action: PreparedFiniteGroupAction, /) -> np.ndarray:
    permutations = np.asarray(action.permutations, dtype=np.int32)
    phases = np.asarray(action.phases, dtype=np.complex128)
    order = action.group_order
    table = np.empty((order, order), dtype=np.int32)
    for left in range(order):
        for right in range(order):
            permutation = permutations[left, permutations[right]]
            phase = phases[left, permutations[right]] * phases[right]
            matches = tuple(
                index
                for index in range(order)
                if np.array_equal(permutations[index], permutation)
                and np.allclose(
                    phases[index],
                    phase,
                    rtol=action.character.tolerance,
                    atol=action.character.tolerance,
                )
            )
            if len(matches) != 1:
                raise ValueError(
                    "Prepared action does not have a unique multiplication table."
                )
            table[left, right] = matches[0]
    return table


def prepare_finite_group_irrep(
    action_plan: FiniteGroupActionPlan,
    irrep_plan: FiniteGroupIrrepPlan,
    /,
) -> PreparedFiniteGroupIrrep:
    """Close the action and extend a generator irrep along the identical words."""

    if not isinstance(action_plan, FiniteGroupActionPlan):
        raise TypeError("action_plan must be FiniteGroupActionPlan.")
    if not isinstance(irrep_plan, FiniteGroupIrrepPlan):
        raise TypeError("irrep_plan must be FiniteGroupIrrepPlan.")
    action_labels = tuple(generator.label for generator in action_plan.generators)
    if set(action_labels) != set(irrep_plan.generator_labels):
        raise ValueError("Action and irrep generator labels must agree exactly.")
    trivial = CharacterSectorPlan(
        f"{irrep_plan.label}-closure",
        {label: 1.0 + 0.0j for label in action_labels},
        tolerance=irrep_plan.tolerance,
    )
    action = prepare_finite_group_action(action_plan, trivial)
    matrices_by_label = {
        label: np.asarray(irrep_plan.generator_matrices[index], dtype=np.complex128)
        for index, label in enumerate(irrep_plan.generator_labels)
    }
    antiunitary_by_label = dict(
        zip(
            irrep_plan.generator_labels,
            irrep_plan.generator_antiunitary,
            strict=True,
        )
    )
    matrices: list[np.ndarray] = []
    flags: list[bool] = []
    identity = np.eye(irrep_plan.irrep_dimension, dtype=np.complex128)
    for word in action.words:
        matrix = identity
        antiunitary = False
        for label in word:
            matrix, antiunitary = _compose_corepresentations(
                matrices_by_label[label],
                antiunitary_by_label[label],
                matrix,
                antiunitary,
            )
        matrices.append(matrix)
        flags.append(antiunitary)
    matrix_table = np.stack(matrices, axis=0)
    multiplication = _action_product_table(action)
    order = action.group_order
    cocycle = np.ones((order, order), dtype=np.complex128)
    maximum_residual = 0.0
    maximum_modulus_error = 0.0
    for left in range(order):
        for right in range(order):
            product, flag = _compose_corepresentations(
                matrix_table[left],
                flags[left],
                matrix_table[right],
                flags[right],
            )
            target_index = multiplication[left, right]
            if flag != flags[target_index]:
                raise ValueError("Corepresentation antiunitary parity violates closure.")
            target = matrix_table[target_index]
            phase = np.trace(product @ target.conj().T) / irrep_plan.irrep_dimension
            if abs(phase) <= irrep_plan.tolerance:
                raise ValueError("Projective product has an indeterminate cocycle phase.")
            phase /= abs(phase)
            residual = float(np.linalg.norm(product - phase * target))
            maximum_residual = max(maximum_residual, residual)
            maximum_modulus_error = max(maximum_modulus_error, abs(abs(phase) - 1.0))
            cocycle[left, right] = phase
            if not irrep_plan.projective and (
                residual > irrep_plan.tolerance or abs(phase - 1.0) > irrep_plan.tolerance
            ):
                raise ValueError(
                    "Generator matrices do not define a linear group representation."
                )
            if irrep_plan.projective and residual > irrep_plan.tolerance:
                raise ValueError(
                    "Generator matrices do not define a projective representation."
                )
    cocycle_residual = 0.0
    for first in range(order):
        for second in range(order):
            for third in range(order):
                left = (
                    cocycle[first, second] * cocycle[multiplication[first, second], third]
                )
                conjugated = (
                    np.conj(cocycle[second, third])
                    if flags[first]
                    else cocycle[second, third]
                )
                right = conjugated * cocycle[first, multiplication[second, third]]
                cocycle_residual = max(cocycle_residual, abs(left - right))
    if cocycle_residual > 10.0 * irrep_plan.tolerance:
        raise ValueError(
            "Projective representation violates the twisted cocycle identity."
        )
    unitarity = np.asarray([_unitary_residual(matrix) for matrix in matrix_table])
    evidence_id = canonical_fingerprint(
        {
            "kind": "finite-group-irrep-evidence",
            "plan": irrep_plan.plan_id,
            "action": action.prepared_id,
            "unitarity": array_tree_fingerprint(unitarity),
            "multiplication_residual": maximum_residual,
            "cocycle_residual": cocycle_residual,
            "cocycle_modulus_error": maximum_modulus_error,
        }
    )
    evidence = FiniteGroupIrrepEvidence(
        unitarity_residuals=jnp.asarray(unitarity),
        multiplication_residual=jnp.asarray(maximum_residual),
        cocycle_residual=jnp.asarray(cocycle_residual),
        maximum_cocycle_modulus_error=jnp.asarray(maximum_modulus_error),
        contains_antiunitary=any(flags),
        projective=irrep_plan.projective,
        evidence_id=evidence_id,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-finite-group-irrep",
            "action": action.prepared_id,
            "plan": irrep_plan.plan_id,
            "matrices": array_tree_fingerprint(matrix_table),
            "antiunitary": tuple(flags),
            "multiplication": array_tree_fingerprint(multiplication),
            "cocycle": array_tree_fingerprint(cocycle),
        }
    )
    return PreparedFiniteGroupIrrep(
        action=action,
        plan=irrep_plan,
        matrices=jnp.asarray(matrix_table),
        antiunitary=jnp.asarray(flags),
        multiplication_table=jnp.asarray(multiplication),
        cocycle=jnp.asarray(cocycle),
        evidence=evidence,
        prepared_id=prepared_id,
    )


class IrrepMultiplicityEvidence(StrictModule):
    expected_multiplicity: int = eqx.field(static=True)
    prepared_multiplicity: int = eqx.field(static=True)
    rank_gap: Array
    orthonormality_residual: Array
    projector_residual: Array
    covariance_residual: Array
    pivot_indices: tuple[int, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedFiniteGroupIrrepBasis(StrictModule):
    """Deterministically gauged multiplicity × irrep embedding into a direct sector."""

    representation: PreparedFiniteGroupIrrep
    embedding: Array
    projector: Array
    multiplicity: int = eqx.field(static=True)
    irrep_dimension: int = eqx.field(static=True)
    direct_dimension: int = eqx.field(static=True)
    evidence: IrrepMultiplicityEvidence
    basis_id: str = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return self.multiplicity * self.irrep_dimension

    def to_direct(self, value: ArrayLike, /) -> Array:
        coefficients = jnp.asarray(value, dtype=self.embedding.dtype)
        if coefficients.shape != (self.dimension,):
            raise ValueError("Irrep coefficients have the wrong shape.")
        return self.embedding @ coefficients

    def from_direct(self, value: ArrayLike, /) -> Array:
        direct = jnp.asarray(value, dtype=self.embedding.dtype)
        if direct.shape != (self.direct_dimension,):
            raise ValueError("Direct-sector vector has the wrong shape.")
        return jnp.conj(self.embedding.T) @ direct


def _direct_action_matrices(action: PreparedFiniteGroupAction, /) -> np.ndarray:
    permutations = np.asarray(action.permutations, dtype=np.int32)
    phases = np.asarray(action.phases, dtype=np.complex128)
    dimension = action.plan.basis.dimension
    matrices = np.zeros((action.group_order, dimension, dimension), dtype=np.complex128)
    columns = np.arange(dimension)
    for index in range(action.group_order):
        matrices[index, permutations[index], columns] = phases[index]
    return matrices


def _deterministic_projected_basis(
    projector: np.ndarray,
    multiplicity: int,
    tolerance: float,
    /,
) -> tuple[np.ndarray, tuple[int, ...]]:
    dimension = projector.shape[0]
    vectors: list[np.ndarray] = []
    pivots: list[int] = []
    for pivot in range(dimension):
        vector = projector[:, pivot].copy()
        for basis_vector in vectors:
            vector -= np.vdot(basis_vector, vector) * basis_vector
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            continue
        vector /= norm
        nonzero = np.flatnonzero(np.abs(vector) > tolerance)
        phase_pivot = int(nonzero[0])
        vector *= np.exp(-1j * np.angle(vector[phase_pivot]))
        vectors.append(vector)
        pivots.append(pivot)
        if len(vectors) == multiplicity:
            break
    if len(vectors) != multiplicity:
        raise ValueError(
            "Multiplicity projector does not admit the expected stable rank."
        )
    return np.stack(vectors, axis=1), tuple(pivots)


def prepare_finite_group_irrep_basis(
    representation: PreparedFiniteGroupIrrep,
    /,
    *,
    maximum_dimension: int = 100_000,
) -> PreparedFiniteGroupIrrepBasis:
    """Construct matrix-unit sectors with deterministic multiplicity gauges."""

    if not isinstance(representation, PreparedFiniteGroupIrrep):
        raise TypeError("representation must be PreparedFiniteGroupIrrep.")
    if bool(np.any(np.asarray(representation.antiunitary))):
        raise ValueError(
            "Linear irrep projectors do not apply to antiunitary corepresentations."
        )
    cocycle = np.asarray(representation.cocycle)
    if np.max(np.abs(cocycle - 1.0)) > representation.plan.tolerance:
        raise ValueError(
            "Projective irreps require a matching projective action projector."
        )
    direct = _direct_action_matrices(representation.action)
    matrices = np.asarray(representation.matrices, dtype=np.complex128)
    order = representation.action.group_order
    irrep_dimension = representation.plan.irrep_dimension
    direct_dimension = representation.action.plan.basis.dimension
    characters = np.trace(matrices, axis1=1, axis2=2)
    direct_characters = np.trace(direct, axis1=1, axis2=2)
    multiplicity_value = np.vdot(characters, direct_characters) / order
    multiplicity = round(float(multiplicity_value.real))
    if (
        multiplicity < 1
        or abs(multiplicity_value.imag) > representation.plan.tolerance
        or abs(multiplicity_value.real - multiplicity) > representation.plan.tolerance
    ):
        raise ValueError("The requested irrep has no stable integral multiplicity.")
    reduced_dimension = multiplicity * irrep_dimension
    if reduced_dimension > maximum_dimension:
        raise ValueError("Prepared irrep dimension exceeds maximum_dimension.")
    units = np.zeros(
        (irrep_dimension, irrep_dimension, direct_dimension, direct_dimension),
        dtype=np.complex128,
    )
    for row in range(irrep_dimension):
        for column in range(irrep_dimension):
            weights = np.conj(matrices[:, row, column])
            units[row, column] = (
                irrep_dimension / order * np.tensordot(weights, direct, axes=((0,), (0,)))
            )
    seed_projector = 0.5 * (units[0, 0] + units[0, 0].conj().T)
    eigenvalues = np.linalg.eigvalsh(seed_projector)
    positive = eigenvalues[eigenvalues > representation.plan.tolerance]
    if positive.size != multiplicity:
        raise ValueError("Matrix-unit rank does not match the character multiplicity.")
    rejected = eigenvalues[eigenvalues <= representation.plan.tolerance]
    rank_gap = float(positive.min() - (rejected.max() if rejected.size else 0.0))
    seeds, pivots = _deterministic_projected_basis(
        seed_projector,
        multiplicity,
        representation.plan.tolerance,
    )
    columns: list[np.ndarray] = []
    for copy_index in range(multiplicity):
        for component in range(irrep_dimension):
            vector = units[component, 0] @ seeds[:, copy_index]
            norm = float(np.linalg.norm(vector))
            if norm <= representation.plan.tolerance:
                raise ValueError("A matrix-unit component has zero norm.")
            columns.append(vector / norm)
    embedding = np.stack(columns, axis=1)
    projector = embedding @ embedding.conj().T
    identity = np.eye(reduced_dimension, dtype=np.complex128)
    orthonormality = float(np.linalg.norm(embedding.conj().T @ embedding - identity))
    central = sum(units[index, index] for index in range(irrep_dimension))
    projector_residual = float(np.linalg.norm(projector - central))
    covariance = 0.0
    target_action = np.stack(
        tuple(
            np.kron(np.eye(multiplicity), matrices[group_index])
            for group_index in range(order)
        ),
        axis=0,
    )
    for group_index in range(order):
        covariance = max(
            covariance,
            float(
                np.linalg.norm(
                    direct[group_index] @ embedding
                    - embedding @ target_action[group_index]
                )
            ),
        )
    tolerance = 100.0 * representation.plan.tolerance
    if max(orthonormality, projector_residual, covariance) > tolerance:
        raise ValueError("Prepared irrep basis fails matrix-unit consistency.")
    evidence_id = canonical_fingerprint(
        {
            "kind": "irrep-multiplicity-evidence",
            "representation": representation.prepared_id,
            "multiplicity": multiplicity,
            "rank_gap": rank_gap,
            "orthonormality": orthonormality,
            "projector_residual": projector_residual,
            "covariance_residual": covariance,
            "pivots": pivots,
        }
    )
    evidence = IrrepMultiplicityEvidence(
        expected_multiplicity=multiplicity,
        prepared_multiplicity=multiplicity,
        rank_gap=jnp.asarray(rank_gap),
        orthonormality_residual=jnp.asarray(orthonormality),
        projector_residual=jnp.asarray(projector_residual),
        covariance_residual=jnp.asarray(covariance),
        pivot_indices=pivots,
        evidence_id=evidence_id,
    )
    basis_id = canonical_fingerprint(
        {
            "kind": "prepared-finite-group-irrep-basis",
            "representation": representation.prepared_id,
            "embedding": array_tree_fingerprint(embedding),
            "pivots": pivots,
        }
    )
    return PreparedFiniteGroupIrrepBasis(
        representation=representation,
        embedding=jnp.asarray(embedding),
        projector=jnp.asarray(projector),
        multiplicity=multiplicity,
        irrep_dimension=irrep_dimension,
        direct_dimension=direct_dimension,
        evidence=evidence,
        basis_id=basis_id,
    )


class CovariantOperatorEvidence(StrictModule):
    covariance_residuals: Array
    selection_rule_residual: Array
    evidence_id: str = eqx.field(static=True)


class ReducedCovariantOperator(StrictModule):
    """Sparse COO components between prepared finite-group irrep sectors."""

    source: PreparedFiniteGroupIrrepBasis
    target: PreparedFiniteGroupIrrepBasis
    component_representation: PreparedFiniteGroupIrrep
    rows: Array
    columns: Array
    values: Array
    component_offsets: Array
    component_count: int = eqx.field(static=True)
    source_dimension: int = eqx.field(static=True)
    target_dimension: int = eqx.field(static=True)
    evidence: CovariantOperatorEvidence
    operator_id: str = eqx.field(static=True)

    def mv(self, component: int, value: ArrayLike, /) -> Array:
        component_ = int(component)
        if component_ < 0 or component_ >= self.component_count:
            raise ValueError("Covariant operator component is out of range.")
        vector = jnp.asarray(value, dtype=self.values.dtype)
        if vector.shape != (self.source_dimension,):
            raise ValueError("Covariant source vector has the wrong shape.")
        start = self.component_offsets[component_]
        stop = self.component_offsets[component_ + 1]
        mask = (jnp.arange(self.values.shape[0]) >= start) & (
            jnp.arange(self.values.shape[0]) < stop
        )
        contributions = jnp.where(mask, self.values * vector[self.columns], 0.0j)
        return jax.ops.segment_sum(
            contributions,
            self.rows,
            num_segments=self.target_dimension,
        )

    def dense(self, component: int, /) -> Array:
        identity = jnp.eye(self.source_dimension, dtype=self.values.dtype)
        return jax.vmap(lambda column: self.mv(component, column), in_axes=1, out_axes=1)(
            identity
        )


def prepare_covariant_irrep_operator(
    source: PreparedFiniteGroupIrrepBasis,
    target: PreparedFiniteGroupIrrepBasis,
    component_representation: PreparedFiniteGroupIrrep,
    direct_components: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
    maximum_nonzero: int = 1_000_000,
) -> ReducedCovariantOperator:
    """Audit a tensor operator and lower its source/target blocks to sparse COO."""

    if not isinstance(source, PreparedFiniteGroupIrrepBasis) or not isinstance(
        target, PreparedFiniteGroupIrrepBasis
    ):
        raise TypeError("source and target must be prepared finite-group irrep bases.")
    if not isinstance(component_representation, PreparedFiniteGroupIrrep):
        raise TypeError("component_representation must be PreparedFiniteGroupIrrep.")
    if source.direct_dimension != target.direct_dimension:
        raise ValueError("Covariant direct source and target dimensions must agree.")
    action_plan_id = source.representation.action.plan.plan_id
    if (
        target.representation.action.plan.plan_id != action_plan_id
        or component_representation.action.plan.plan_id != action_plan_id
    ):
        raise ValueError("All covariant operator representations must share one action.")
    if any(
        bool(np.any(np.asarray(item.antiunitary)))
        for item in (
            source.representation,
            target.representation,
            component_representation,
        )
    ):
        raise ValueError(
            "Antiunitary covariant projection requires a corepresentation map."
        )
    direct = np.asarray(direct_components, dtype=np.complex128)
    component_count = component_representation.plan.irrep_dimension
    dimension = source.direct_dimension
    if direct.shape != (component_count, dimension, dimension):
        raise ValueError("direct_components has the wrong component or direct shape.")
    actions = _direct_action_matrices(source.representation.action)
    component_matrices = np.asarray(component_representation.matrices)
    residuals: list[float] = []
    for group_index, action in enumerate(actions):
        transformed = np.stack(
            [action @ value @ action.conj().T for value in direct], axis=0
        )
        expected = np.tensordot(
            component_matrices[group_index],
            direct,
            axes=((1,), (0,)),
        )
        residuals.append(float(np.linalg.norm(transformed - expected)))
    tolerance_ = float(tolerance)
    if max(residuals, default=0.0) > tolerance_:
        raise ValueError("Direct operator components violate their declared covariance.")
    source_embedding = np.asarray(source.embedding)
    target_embedding = np.asarray(target.embedding)
    reduced = np.stack(
        [target_embedding.conj().T @ value @ source_embedding for value in direct],
        axis=0,
    )
    selection_residual = float(
        np.linalg.norm(
            direct
            - np.stack(
                [
                    target_embedding @ value @ source_embedding.conj().T
                    for value in reduced
                ]
            )
        )
    )
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    offsets = [0]
    for component in reduced:
        row, column = np.nonzero(np.abs(component) > tolerance_)
        rows.append(row.astype(np.int32))
        columns.append(column.astype(np.int32))
        values.append(component[row, column])
        offsets.append(offsets[-1] + row.size)
    if offsets[-1] > maximum_nonzero:
        raise ValueError("Reduced covariant route exceeds maximum_nonzero.")
    row_table = np.concatenate(rows) if rows else np.empty((0,), dtype=np.int32)
    column_table = np.concatenate(columns) if columns else np.empty((0,), dtype=np.int32)
    value_table = (
        np.concatenate(values) if values else np.empty((0,), dtype=np.complex128)
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "covariant-operator-evidence",
            "source": source.basis_id,
            "target": target.basis_id,
            "components": component_representation.prepared_id,
            "covariance": residuals,
            "selection_residual": selection_residual,
        }
    )
    evidence = CovariantOperatorEvidence(
        covariance_residuals=jnp.asarray(residuals),
        selection_rule_residual=jnp.asarray(selection_residual),
        evidence_id=evidence_id,
    )
    operator_id = canonical_fingerprint(
        {
            "kind": "reduced-covariant-operator",
            "source": source.basis_id,
            "target": target.basis_id,
            "components": component_representation.prepared_id,
            "rows": array_tree_fingerprint(row_table),
            "columns": array_tree_fingerprint(column_table),
            "values": array_tree_fingerprint(value_table),
        }
    )
    return ReducedCovariantOperator(
        source=source,
        target=target,
        component_representation=component_representation,
        rows=jnp.asarray(row_table),
        columns=jnp.asarray(column_table),
        values=jnp.asarray(value_table),
        component_offsets=jnp.asarray(offsets, dtype=jnp.int32),
        component_count=component_count,
        source_dimension=source.dimension,
        target_dimension=target.dimension,
        evidence=evidence,
        operator_id=operator_id,
    )


def run_symmetry_resolved_finite_size_study(
    inverse_sizes: Sequence[float],
    values: Sequence[float],
    standard_errors: Sequence[float],
    /,
    *,
    linear_and_quadratic: bool = True,
) -> ScientificLimitStudyResult:
    """Extrapolate one symmetry-resolved observable with explicit systematic spread."""

    sizes = tuple(float(value) for value in inverse_sizes)
    observations = tuple(float(value) for value in values)
    errors = tuple(float(value) for value in standard_errors)
    if not sizes or len(sizes) != len(observations) or len(sizes) != len(errors):
        raise ValueError("Finite-size coordinates, values, and errors must align.")
    axis = ScientificLimitAxis(
        "inverse-size",
        0.0,
        minimum_span=max(sizes) - min(sizes),
    )
    variations = [
        ScientificLimitVariation(
            "linear",
            {"inverse-size": 1},
            minimum_points=3,
        )
    ]
    if linear_and_quadratic:
        variations.append(
            ScientificLimitVariation(
                "quadratic",
                {"inverse-size": 2},
                minimum_points=4,
            )
        )
    plan = ScientificLimitStudyPlan((axis,), tuple(variations))
    data = tuple(
        ScientificLimitDatum(
            f"size-{index}",
            {"inverse-size": size},
            observation,
            error,
        )
        for index, (size, observation, error) in enumerate(
            zip(sizes, observations, errors, strict=True)
        )
    )
    return run_scientific_limit_study(plan, data)


__all__ = [
    "CovariantOperatorEvidence",
    "FiniteGroupIrrepEvidence",
    "FiniteGroupIrrepPlan",
    "IrrepMultiplicityEvidence",
    "PreparedFiniteGroupIrrep",
    "PreparedFiniteGroupIrrepBasis",
    "ReducedCovariantOperator",
    "prepare_covariant_irrep_operator",
    "prepare_finite_group_irrep",
    "prepare_finite_group_irrep_basis",
    "run_symmetry_resolved_finite_size_study",
]
