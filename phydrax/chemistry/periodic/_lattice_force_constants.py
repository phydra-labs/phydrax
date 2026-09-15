#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical real-space interatomic force constants and native IFC2 generation."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic._many_body import ManyBodyPotential
from ...atomistic._potential_program import PreparedAtomisticPotentialProgram
from ...discretization import AbstractPreparedParticleNeighborhood, PeriodicCell
from ...sparse import EdgeRelation
from ...units import derived_unit, ENERGY, LENGTH, UnitDefinition


_MAX_IFC2_ROUTES = 4096
_MAX_IFC3_ROUTES = 2_000_000


def second_order_force_constant_unit(
    energy: UnitDefinition, length: UnitDefinition
) -> UnitDefinition:
    """Return the explicit energy/length² unit used by IFC2 blocks."""

    return derived_unit(f"{energy.symbol}/{length.symbol}^2", ((energy, 1), (length, -2)))


def third_order_force_constant_unit(
    energy: UnitDefinition, length: UnitDefinition
) -> UnitDefinition:
    """Return the explicit energy/length³ unit used by IFC3 blocks."""

    return derived_unit(f"{energy.symbol}/{length.symbol}^3", ((energy, 1), (length, -3)))


def _identifier(value: str, name: str) -> str:
    result = str(value)
    if not result or result != result.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return result


def _ifc2_keys(
    relation: EdgeRelation, translations: np.ndarray
) -> dict[tuple[int, int, tuple[int, ...]], int]:
    valid = np.asarray(relation.valid, dtype=bool)
    sources = np.asarray(relation.source_indices)
    targets = np.asarray(relation.target_indices)
    result: dict[tuple[int, int, tuple[int, ...]], int] = {}
    for route in np.flatnonzero(valid):
        key = (
            int(sources[route]),
            int(targets[route]),
            tuple(int(x) for x in translations[route]),
        )
        if key in result:
            raise ValueError(
                "IFC2 routes must not contain duplicate atom/translation records."
            )
        result[key] = int(route)
    return result


def _ifc2_reverse_indices(relation: EdgeRelation, translations: np.ndarray) -> np.ndarray:
    keys = _ifc2_keys(relation, translations)
    reverse = np.arange(relation.capacity, dtype=np.int32)
    for key, route in keys.items():
        source, target, translation = key
        partner = (target, source, tuple(-value for value in translation))
        if partner not in keys:
            raise ValueError(
                "Every valid IFC2 route requires its explicit reverse route."
            )
        reverse[route] = keys[partner]
    return reverse


def _ifc3_permutation_indices(
    atom_triplets: np.ndarray, translations: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    keys: dict[tuple[int, int, int, tuple[int, ...], tuple[int, ...]], int] = {}
    for route in np.flatnonzero(valid):
        key = (
            *(int(value) for value in atom_triplets[route]),
            tuple(int(value) for value in translations[route, 0]),
            tuple(int(value) for value in translations[route, 1]),
        )
        if key in keys:
            raise ValueError("IFC3 routes must not contain duplicate triplet records.")
        keys[key] = int(route)
    permutations = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    result = np.zeros((atom_triplets.shape[0], 6), dtype=np.int32)
    zero = np.zeros((translations.shape[-1],), dtype=np.int64)
    for route in np.flatnonzero(valid):
        atoms = atom_triplets[route]
        cells = (zero, translations[route, 0], translations[route, 1])
        for column, permutation in enumerate(permutations):
            origin = cells[permutation[0]]
            key = (
                int(atoms[permutation[0]]),
                int(atoms[permutation[1]]),
                int(atoms[permutation[2]]),
                tuple(int(value) for value in cells[permutation[1]] - origin),
                tuple(int(value) for value in cells[permutation[2]] - origin),
            )
            if key not in keys:
                raise ValueError(
                    "Every valid IFC3 route requires all six explicit permutations."
                )
            result[route, column] = keys[key]
    return result


class IFCConstraintPolicy(StrictModule, NonTrainableState):
    """Fail-closed linear projection policy for IFC2 invariants."""

    enforce_pair_symmetry: bool = eqx.field(static=True)
    enforce_acoustic_sum_rule: bool = eqx.field(static=True)
    enforce_rotational_sum_rule: bool = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_relative_correction: float = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        enforce_pair_symmetry: bool = True,
        enforce_acoustic_sum_rule: bool = True,
        enforce_rotational_sum_rule: bool = True,
        residual_tolerance: float = 1.0e-8,
        maximum_relative_correction: float = 5.0e-2,
        maximum_condition_number: float = 1.0e12,
    ):
        tolerance = float(residual_tolerance)
        correction = float(maximum_relative_correction)
        condition = float(maximum_condition_number)
        if (
            not isfinite(tolerance)
            or tolerance <= 0.0
            or not isfinite(correction)
            or correction < 0.0
            or not isfinite(condition)
            or condition < 1.0
        ):
            raise ValueError("IFC constraint tolerances and condition limit are invalid.")
        self.enforce_pair_symmetry = bool(enforce_pair_symmetry)
        self.enforce_acoustic_sum_rule = bool(enforce_acoustic_sum_rule)
        self.enforce_rotational_sum_rule = bool(enforce_rotational_sum_rule)
        self.residual_tolerance = tolerance
        self.maximum_relative_correction = correction
        self.maximum_condition_number = condition
        self.policy_id = canonical_fingerprint(
            {
                "kind": "ifc-constraint-policy",
                "pair": self.enforce_pair_symmetry,
                "asr": self.enforce_acoustic_sum_rule,
                "rotation": self.enforce_rotational_sum_rule,
                "tolerance": tolerance,
                "maximum_relative_correction": correction,
                "maximum_condition_number": condition,
            }
        )


class IFCConstraintEvidence(StrictModule, NonTrainableState):
    """Raw and corrected residuals; correction is never hidden."""

    raw_pair_residual: Array
    raw_acoustic_residual: Array
    raw_rotational_residual: Array
    corrected_pair_residual: Array
    corrected_acoustic_residual: Array
    corrected_rotational_residual: Array
    relative_correction: Array
    constraint_rank: int = eqx.field(static=True)
    constraint_condition_number: float = eqx.field(static=True)
    successful: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        raw_residuals,
        corrected_residuals,
        relative_correction,
        rank,
        condition,
        successful,
        /,
    ):
        raw = jnp.asarray(raw_residuals).reshape((3,))
        corrected = jnp.asarray(corrected_residuals, dtype=raw.dtype).reshape((3,))
        (
            self.raw_pair_residual,
            self.raw_acoustic_residual,
            self.raw_rotational_residual,
        ) = raw
        (
            self.corrected_pair_residual,
            self.corrected_acoustic_residual,
            self.corrected_rotational_residual,
        ) = corrected
        self.relative_correction = jnp.asarray(
            relative_correction, dtype=raw.dtype
        ).reshape(())
        self.constraint_rank = int(rank)
        self.constraint_condition_number = float(condition)
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "ifc-constraint-evidence",
                "rank": self.constraint_rank,
                "condition": self.constraint_condition_number,
                "arrays": array_tree_fingerprint(
                    {
                        "raw": np.asarray(raw),
                        "corrected": np.asarray(corrected),
                        "relative_correction": np.asarray(self.relative_correction),
                        "successful": np.asarray(self.successful),
                    }
                ),
            }
        )


def _constraint_matrix(
    relation: EdgeRelation,
    translations: np.ndarray,
    fractional_positions: np.ndarray,
    cell: PeriodicCell,
    reverse: np.ndarray,
    policy: IFCConstraintPolicy,
) -> np.ndarray:
    routes = relation.capacity
    columns = routes * 9
    rows: list[np.ndarray] = []
    valid = np.asarray(relation.valid, dtype=bool)
    source = np.asarray(relation.source_indices)
    target = np.asarray(relation.target_indices)
    if policy.enforce_pair_symmetry:
        for route in np.flatnonzero(valid):
            partner = int(reverse[route])
            if route > partner:
                continue
            for alpha in range(3):
                for beta in range(3):
                    row = np.zeros((columns,), dtype=float)
                    row[route * 9 + alpha * 3 + beta] = 1.0
                    row[partner * 9 + beta * 3 + alpha] -= 1.0
                    if np.any(row):
                        rows.append(row)
    if policy.enforce_acoustic_sum_rule:
        for atom in range(relation.source_size):
            selected = np.flatnonzero(valid & (target == atom))
            for alpha in range(3):
                for beta in range(3):
                    row = np.zeros((columns,), dtype=float)
                    row[selected * 9 + alpha * 3 + beta] = 1.0
                    if np.any(row):
                        rows.append(row)
    if policy.enforce_rotational_sum_rule:
        cartesian = np.asarray(cell.cartesian(fractional_positions))
        lattice = translations @ np.asarray(cell.vectors)
        displacement = cartesian[source] + lattice - cartesian[target]
        for atom in range(relation.target_size):
            selected = np.flatnonzero(valid & (target == atom))
            for alpha in range(3):
                for beta in range(3):
                    for gamma in range(beta + 1, 3):
                        row = np.zeros((columns,), dtype=float)
                        row[selected * 9 + alpha * 3 + beta] += displacement[
                            selected, gamma
                        ]
                        row[selected * 9 + alpha * 3 + gamma] -= displacement[
                            selected, beta
                        ]
                        if np.any(row):
                            rows.append(row)
    return np.stack(rows) if rows else np.zeros((0, columns), dtype=float)


def _residuals(
    values: np.ndarray,
    relation: EdgeRelation,
    translations: np.ndarray,
    fractional_positions: np.ndarray,
    cell: PeriodicCell,
    reverse: np.ndarray,
) -> np.ndarray:
    valid = np.asarray(relation.valid, dtype=bool)
    source = np.asarray(relation.source_indices)
    target = np.asarray(relation.target_indices)
    pair = float(
        np.max(np.abs(values - np.transpose(values[reverse], (0, 2, 1))), initial=0.0)
    )
    acoustic = 0.0
    rotation = 0.0
    cartesian = np.asarray(cell.cartesian(fractional_positions))
    displacement = (
        cartesian[source] + translations @ np.asarray(cell.vectors) - cartesian[target]
    )
    for atom in range(relation.target_size):
        selected = valid & (target == atom)
        acoustic = max(
            acoustic, float(np.max(np.abs(np.sum(values[selected], axis=0)), initial=0.0))
        )
        moment = np.einsum("eab,eg->abg", values[selected], displacement[selected])
        rotation = max(
            rotation,
            float(np.max(np.abs(moment - np.swapaxes(moment, 1, 2)), initial=0.0)),
        )
    return np.asarray((pair, acoustic, rotation), dtype=float)


def _project_ifc2(
    raw: np.ndarray,
    relation: EdgeRelation,
    translations: np.ndarray,
    fractional_positions: np.ndarray,
    cell: PeriodicCell,
    reverse: np.ndarray,
    policy: IFCConstraintPolicy,
) -> tuple[np.ndarray, IFCConstraintEvidence]:
    matrix = _constraint_matrix(
        relation, translations, fractional_positions, cell, reverse, policy
    )
    vector = raw.reshape((-1,)).astype(float, copy=False)
    if matrix.shape[0]:
        _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        threshold = np.finfo(float).eps * max(matrix.shape) * singular[0]
        rank = int(np.count_nonzero(singular > threshold))
        basis = vh[:rank]
        corrected_vector = vector - basis.T @ (basis @ vector)
        condition = float(singular[0] / singular[rank - 1]) if rank else 1.0
    else:
        corrected_vector = vector.copy()
        rank = 0
        condition = 1.0
    corrected = corrected_vector.reshape(raw.shape).astype(raw.dtype, copy=False)
    raw_residuals = _residuals(
        raw, relation, translations, fractional_positions, cell, reverse
    )
    corrected_residuals = _residuals(
        corrected, relation, translations, fractional_positions, cell, reverse
    )
    norm = np.linalg.norm(vector)
    relative = float(
        np.linalg.norm(corrected_vector - vector) / max(norm, np.finfo(float).tiny)
    )
    selected = np.asarray(
        (
            policy.enforce_pair_symmetry,
            policy.enforce_acoustic_sum_rule,
            policy.enforce_rotational_sum_rule,
        )
    )
    successful = (
        condition <= policy.maximum_condition_number
        and relative <= policy.maximum_relative_correction
        and np.all(corrected_residuals[selected] <= policy.residual_tolerance)
    )
    return corrected, IFCConstraintEvidence(
        raw_residuals, corrected_residuals, relative, rank, condition, successful
    )


class SecondOrderForceConstants(StrictModule, NonTrainableState):
    """Canonical Φ(iα,jβ,R) on edges target=i, source=j with explicit reverse routes."""

    relation: EdgeRelation
    translations: Array
    reverse_indices: Array
    raw_values: Array
    values: Array
    fractional_positions: Array
    cell: PeriodicCell
    unit: UnitDefinition
    constraints: IFCConstraintEvidence
    system_id: str = eqx.field(static=True)
    source_kind: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    ifc_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation,
        translations: ArrayLike,
        raw_values: ArrayLike,
        values: ArrayLike,
        fractional_positions: ArrayLike,
        cell: PeriodicCell,
        unit: UnitDefinition,
        constraints: IFCConstraintEvidence,
        /,
        *,
        system_id: str,
        source_kind: str,
        source_id: str,
        convention_id: str = "ifc-real-space-source-at-zero-target-at-R",
    ):
        if (
            not isinstance(relation, EdgeRelation)
            or relation.source_size != relation.target_size
        ):
            raise TypeError(
                "IFC2 relation must be a square EdgeRelation over primitive atoms."
            )
        if relation.capacity > _MAX_IFC2_ROUTES:
            raise ValueError(f"IFC2 route capacity exceeds {_MAX_IFC2_ROUTES}.")
        if not isinstance(cell, PeriodicCell) or cell.ambient_dimension != 3:
            raise TypeError("IFC2 requires a three-dimensional embedding PeriodicCell.")
        translation = np.asarray(translations)
        raw = np.asarray(raw_values)
        corrected = np.asarray(values)
        positions = np.asarray(fractional_positions)
        if not np.issubdtype(translation.dtype, np.integer) or translation.shape != (
            relation.capacity,
            cell.rank,
        ):
            raise ValueError(
                "IFC2 translations must be integer routes with shape (E, cell.rank)."
            )
        if raw.shape != (relation.capacity, 3, 3) or corrected.shape != raw.shape:
            raise ValueError("IFC2 values must have shape (E,3,3).")
        if positions.shape != (relation.source_size, cell.rank):
            raise ValueError(
                "IFC2 fractional positions must match primitive atoms and cell rank."
            )
        if (
            np.any(~np.isfinite(raw))
            or np.any(~np.isfinite(corrected))
            or np.any(~np.isfinite(positions))
        ):
            raise ValueError("IFC2 arrays must be finite.")
        if not isinstance(unit, UnitDefinition) or unit.dimension != ENERGY / LENGTH**2:
            raise ValueError("IFC2 unit must have energy/length² dimension.")
        if not isinstance(constraints, IFCConstraintEvidence):
            raise TypeError("constraints must be IFCConstraintEvidence.")
        reverse = _ifc2_reverse_indices(
            relation, translation.astype(np.int64, copy=False)
        )
        self.relation = relation
        self.translations = jnp.asarray(translation, dtype=jnp.int32)
        self.reverse_indices = jnp.asarray(reverse)
        self.raw_values = jnp.asarray(raw)
        self.values = jnp.asarray(corrected)
        self.fractional_positions = jnp.asarray(positions)
        self.cell = cell
        self.unit = unit
        self.constraints = constraints
        self.system_id = _identifier(system_id, "system_id")
        self.source_kind = _identifier(source_kind, "source_kind")
        self.source_id = _identifier(source_id, "source_id")
        self.convention_id = _identifier(convention_id, "convention_id")
        self.ifc_id = canonical_fingerprint(
            {
                "kind": "second-order-force-constants",
                "system": self.system_id,
                "cell": cell.cell_id,
                "unit": unit.unit_id,
                "source_kind": self.source_kind,
                "source": self.source_id,
                "convention": self.convention_id,
                "constraints": constraints.evidence_id,
                "arrays": array_tree_fingerprint(
                    {
                        "source": np.asarray(relation.source_indices),
                        "target": np.asarray(relation.target_indices),
                        "valid": np.asarray(relation.valid),
                        "translations": translation,
                        "raw": raw,
                        "values": corrected,
                        "positions": positions,
                    }
                ),
            }
        )


class ThirdOrderForceConstants(StrictModule, NonTrainableState):
    """Canonical Ψ(iα,jβ,R;kγ,S) with explicit sixfold permutation closure."""

    atom_triplets: Array
    translations: Array
    valid: Array
    permutation_indices: Array
    raw_values: Array
    values: Array
    unit: UnitDefinition
    permutation_residual: Array
    acoustic_residual: Array
    constraints_successful: Array
    atom_count: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    source_kind: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    ifc_id: str = eqx.field(static=True)

    def __init__(
        self,
        atom_triplets: ArrayLike,
        translations: ArrayLike,
        raw_values: ArrayLike,
        values: ArrayLike,
        unit: UnitDefinition,
        /,
        *,
        atom_count: int,
        system_id: str,
        source_kind: str,
        source_id: str,
        valid: ArrayLike | None = None,
        residual_tolerance: float = 1.0e-8,
    ):
        triplets = np.asarray(atom_triplets)
        translation = np.asarray(translations)
        raw = np.asarray(raw_values)
        corrected = np.asarray(values)
        if (
            triplets.ndim != 2
            or triplets.shape[1] != 3
            or not np.issubdtype(triplets.dtype, np.integer)
        ):
            raise ValueError("IFC3 atom_triplets must be an integer (E,3) array.")
        routes = triplets.shape[0]
        if routes == 0 or routes > _MAX_IFC3_ROUTES:
            raise ValueError(
                "IFC3 route count is empty or exceeds the admitted capacity."
            )
        if (
            translation.ndim != 3
            or translation.shape[:2] != (routes, 2)
            or not np.issubdtype(translation.dtype, np.integer)
        ):
            raise ValueError("IFC3 translations must have integer shape (E,2,rank).")
        valid_ = (
            np.ones((routes,), dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if valid_.shape != (routes,) or np.any(
            valid_
            & ((triplets < 0).any(axis=1) | (triplets >= int(atom_count)).any(axis=1))
        ):
            raise ValueError("IFC3 route mask or atom indices are invalid.")
        if raw.shape != (routes, 3, 3, 3) or corrected.shape != raw.shape:
            raise ValueError("IFC3 values must have shape (E,3,3,3).")
        if np.any(~np.isfinite(raw)) or np.any(~np.isfinite(corrected)):
            raise ValueError("IFC3 values must be finite.")
        if not isinstance(unit, UnitDefinition) or unit.dimension != ENERGY / LENGTH**3:
            raise ValueError("IFC3 unit must have energy/length³ dimension.")
        permutations = _ifc3_permutation_indices(triplets, translation, valid_)
        axes = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
        permutation_residual = 0.0
        for column, permutation in enumerate(axes):
            permuted = np.transpose(
                corrected[permutations[:, column]],
                (0,) + tuple(value + 1 for value in permutation),
            )
            permutation_residual = max(
                permutation_residual,
                float(np.max(np.abs(corrected[valid_] - permuted[valid_]), initial=0.0)),
            )
        acoustic = 0.0
        for first in range(int(atom_count)):
            chosen = valid_ & (triplets[:, 0] == first)
            acoustic = max(
                acoustic,
                float(np.max(np.abs(np.sum(corrected[chosen], axis=0)), initial=0.0)),
            )
        tolerance = float(residual_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("IFC3 residual_tolerance must be finite and positive.")
        self.atom_triplets = jnp.asarray(triplets, dtype=jnp.int32)
        self.translations = jnp.asarray(translation, dtype=jnp.int32)
        self.valid = jnp.asarray(valid_)
        self.permutation_indices = jnp.asarray(permutations)
        self.raw_values = jnp.asarray(raw)
        self.values = jnp.asarray(corrected)
        self.unit = unit
        self.permutation_residual = jnp.asarray(
            permutation_residual, dtype=self.values.dtype
        )
        self.acoustic_residual = jnp.asarray(acoustic, dtype=self.values.dtype)
        self.constraints_successful = jnp.asarray(
            permutation_residual <= tolerance and acoustic <= tolerance
        )
        self.atom_count = int(atom_count)
        self.residual_tolerance = tolerance
        self.system_id = _identifier(system_id, "system_id")
        self.source_kind = _identifier(source_kind, "source_kind")
        self.source_id = _identifier(source_id, "source_id")
        self.ifc_id = canonical_fingerprint(
            {
                "kind": "third-order-force-constants",
                "system": self.system_id,
                "unit": unit.unit_id,
                "source_kind": self.source_kind,
                "atom_count": self.atom_count,
                "source": self.source_id,
                "permutation_residual": permutation_residual,
                "acoustic_residual": acoustic,
                "tolerance": tolerance,
                "arrays": array_tree_fingerprint(
                    {
                        "triplets": triplets,
                        "translations": translation,
                        "valid": valid_,
                        "raw": raw,
                        "values": corrected,
                    }
                ),
            }
        )


class PrimitiveSupercellImageMap(StrictModule, NonTrainableState):
    """Exact primitive-site/image identity for extracting IFC routes from a supercell."""

    primitive_particle_ids: Array
    supercell_particle_ids: Array
    supercell_to_primitive: Array
    image_translations: Array
    primitive_fractional_positions: Array
    primitive_cell: PeriodicCell
    primitive_system_id: str = eqx.field(static=True)
    zero_image_indices: Array
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        primitive_particle_ids: ArrayLike,
        supercell_particle_ids: ArrayLike,
        supercell_to_primitive: ArrayLike,
        image_translations: ArrayLike,
        primitive_fractional_positions: ArrayLike,
        primitive_cell: PeriodicCell,
        /,
        *,
        primitive_system_id: str,
    ):
        primitive_ids = np.asarray(primitive_particle_ids)
        supercell_ids = np.asarray(supercell_particle_ids)
        mapping = np.asarray(supercell_to_primitive)
        translations = np.asarray(image_translations)
        fractional = np.asarray(primitive_fractional_positions)
        if (
            primitive_ids.ndim != 1
            or primitive_ids.size == 0
            or supercell_ids.ndim != 1
            or not np.issubdtype(primitive_ids.dtype, np.integer)
            or not np.issubdtype(supercell_ids.dtype, np.integer)
            or np.unique(primitive_ids).size != primitive_ids.size
            or np.unique(supercell_ids).size != supercell_ids.size
        ):
            raise ValueError(
                "Primitive and supercell particle IDs must be unique integer vectors."
            )
        if not isinstance(primitive_cell, PeriodicCell) or primitive_cell.rank != 3:
            raise TypeError("primitive_cell must be a rank-3 PeriodicCell.")
        if (
            mapping.shape != supercell_ids.shape
            or not np.issubdtype(mapping.dtype, np.integer)
            or np.any(mapping < 0)
            or np.any(mapping >= primitive_ids.size)
            or translations.shape != (supercell_ids.size, primitive_cell.rank)
            or not np.issubdtype(translations.dtype, np.integer)
            or fractional.shape != (primitive_ids.size, primitive_cell.rank)
            or np.any(~np.isfinite(fractional))
        ):
            raise ValueError(
                "Primitive-supercell map arrays have incompatible shapes or values."
            )
        keys = np.concatenate((mapping[:, None], translations), axis=1)
        if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
            raise ValueError("Each primitive-atom/image pair must occur at most once.")
        zero_indices = np.empty((primitive_ids.size,), dtype=np.int32)
        for atom in range(primitive_ids.size):
            matches = np.flatnonzero(
                (mapping == atom) & np.all(translations == 0, axis=1)
            )
            if matches.size != 1:
                raise ValueError(
                    "Every primitive atom requires exactly one zero-image representative."
                )
            zero_indices[atom] = int(matches[0])
        self.primitive_particle_ids = jnp.asarray(primitive_ids, dtype=jnp.int64)
        self.supercell_particle_ids = jnp.asarray(supercell_ids, dtype=jnp.int64)
        self.supercell_to_primitive = jnp.asarray(mapping, dtype=jnp.int32)
        self.image_translations = jnp.asarray(translations, dtype=jnp.int32)
        self.primitive_fractional_positions = jnp.asarray(fractional)
        self.primitive_cell = primitive_cell
        self.primitive_system_id = _identifier(primitive_system_id, "primitive_system_id")
        self.zero_image_indices = jnp.asarray(zero_indices)
        self.map_id = canonical_fingerprint(
            {
                "kind": "primitive-supercell-image-map",
                "primitive_system": self.primitive_system_id,
                "cell": primitive_cell.cell_id,
                "arrays": array_tree_fingerprint(
                    {
                        "primitive_ids": primitive_ids,
                        "supercell_ids": supercell_ids,
                        "mapping": mapping,
                        "translations": translations,
                        "fractional": fractional,
                    }
                ),
            }
        )

    def route_supercell_indices(
        self, relation: EdgeRelation, translations: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray]:
        translation = np.asarray(translations)
        if (
            relation.source_size != self.primitive_particle_ids.size
            or relation.target_size != self.primitive_particle_ids.size
            or translation.shape != (relation.capacity, self.primitive_cell.rank)
        ):
            raise ValueError("Primitive IFC routes do not align with the image map.")
        mapping = np.asarray(self.supercell_to_primitive)
        images = np.asarray(self.image_translations)
        source_atoms = np.asarray(relation.source_indices)
        target_atoms = np.asarray(relation.target_indices)
        source_indices = np.empty((relation.capacity,), dtype=np.int32)
        target_indices = np.asarray(self.zero_image_indices)[target_atoms]
        for route in range(relation.capacity):
            matches = np.flatnonzero(
                (mapping == source_atoms[route])
                & np.all(images == translation[route], axis=1)
            )
            if matches.size != 1:
                raise ValueError(
                    "Every IFC source atom/translation requires exactly one supercell image."
                )
            source_indices[route] = int(matches[0])
        return source_indices, target_indices


class FiniteDisplacementIFC2Result(StrictModule, NonTrainableState):
    force_constants: SecondOrderForceConstants
    equilibrium_force_residual: Array
    central_antisymmetry_residual: Array
    refinement_residual: Array
    force_evaluations: int = eqx.field(static=True)
    successful: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        force_constants,
        equilibrium,
        antisymmetry,
        refinement,
        force_evaluations,
        successful,
        /,
    ):
        self.force_constants = force_constants
        dtype = force_constants.values.dtype
        self.equilibrium_force_residual = jnp.asarray(equilibrium, dtype=dtype).reshape(
            ()
        )
        self.central_antisymmetry_residual = jnp.asarray(
            antisymmetry, dtype=dtype
        ).reshape(())
        self.refinement_residual = jnp.asarray(refinement, dtype=dtype).reshape(())
        self.force_evaluations = int(force_evaluations)
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.result_id = canonical_fingerprint(
            {
                "kind": "finite-displacement-ifc2-result",
                "ifc": force_constants.ifc_id,
                "force_evaluations": self.force_evaluations,
                "arrays": array_tree_fingerprint(
                    {
                        "equilibrium": np.asarray(self.equilibrium_force_residual),
                        "antisymmetry": np.asarray(self.central_antisymmetry_residual),
                        "refinement": np.asarray(self.refinement_residual),
                        "successful": np.asarray(self.successful),
                    }
                ),
            }
        )


class FiniteDisplacementIFC2Plan(StrictModule, NonTrainableState):
    """Native central h/h2 IFC2 plan over a fixed Phydrax force/neighborhood path."""

    potential: PreparedAtomisticPotentialProgram
    neighborhood: AbstractPreparedParticleNeighborhood
    image_map: PrimitiveSupercellImageMap
    equilibrium_positions: Array
    relation: EdgeRelation
    translations: Array
    constraint_policy: IFCConstraintPolicy
    displacement: float = eqx.field(static=True)
    equilibrium_force_tolerance: float = eqx.field(static=True)
    refinement_tolerance: float = eqx.field(static=True)
    maximum_force_evaluations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        potential: PreparedAtomisticPotentialProgram,
        neighborhood: AbstractPreparedParticleNeighborhood,
        equilibrium_positions: ArrayLike,
        image_map: PrimitiveSupercellImageMap,
        relation: EdgeRelation,
        translations: ArrayLike,
        /,
        *,
        displacement: float,
        constraint_policy: IFCConstraintPolicy | None = None,
        equilibrium_force_tolerance: float = 1.0e-7,
        refinement_tolerance: float = 1.0e-5,
        maximum_force_evaluations: int = 100_000,
    ):
        if not isinstance(potential, PreparedAtomisticPotentialProgram):
            raise TypeError("potential must be PreparedAtomisticPotentialProgram.")
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError("neighborhood must be AbstractPreparedParticleNeighborhood.")
        if not all(isinstance(term, ManyBodyPotential) for term in potential.plan.terms):
            raise ValueError(
                "Native crystalline IFC2 is bounded to EAM, SW, and Tersoff terms."
            )
        cell = potential.system.cell
        if cell is None or cell.rank != 3 or not cell.fully_periodic:
            raise ValueError(
                "Native crystalline IFC2 requires a fully periodic rank-3 cell."
            )
        positions = np.asarray(equilibrium_positions)
        if positions.shape != (potential.system.capacity, 3) or np.any(
            ~np.isfinite(positions)
        ):
            raise ValueError(
                "equilibrium_positions must be finite Cartesian system positions."
            )
        if not isinstance(image_map, PrimitiveSupercellImageMap):
            raise TypeError("image_map must be PrimitiveSupercellImageMap.")
        if not np.array_equal(
            np.asarray(image_map.supercell_particle_ids),
            np.asarray(potential.system.plan.particle_ids),
        ):
            raise ValueError(
                "Image-map supercell particle IDs differ from the prepared system."
            )
        if (
            relation.source_size != image_map.primitive_particle_ids.size
            or relation.target_size != image_map.primitive_particle_ids.size
        ):
            raise ValueError("IFC2 relation must index primitive atoms from image_map.")
        step = float(displacement)
        equilibrium_tolerance = float(equilibrium_force_tolerance)
        refinement = float(refinement_tolerance)
        calls = 1 + 4 * positions.size
        if (
            not isfinite(step)
            or step <= 0.0
            or not isfinite(equilibrium_tolerance)
            or equilibrium_tolerance <= 0.0
            or not isfinite(refinement)
            or refinement <= 0.0
            or calls > int(maximum_force_evaluations)
        ):
            raise ValueError(
                "Finite-displacement tolerances or force-call capacity are invalid."
            )
        translation = np.asarray(translations)
        _ifc2_reverse_indices(relation, translation)
        policy = IFCConstraintPolicy() if constraint_policy is None else constraint_policy
        if not isinstance(policy, IFCConstraintPolicy):
            raise TypeError("constraint_policy must be IFCConstraintPolicy or None.")
        self.potential = potential
        self.neighborhood = neighborhood
        self.equilibrium_positions = jnp.asarray(positions)
        self.image_map = image_map
        self.relation = relation
        self.translations = jnp.asarray(translation, dtype=jnp.int32)
        self.constraint_policy = policy
        self.displacement = step
        self.equilibrium_force_tolerance = equilibrium_tolerance
        self.refinement_tolerance = refinement
        self.maximum_force_evaluations = int(maximum_force_evaluations)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-displacement-ifc2-plan",
                "potential": potential.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "cell": image_map.primitive_cell.cell_id,
                "image_map": image_map.map_id,
                "policy": policy.policy_id,
                "displacement": step,
                "equilibrium_force_tolerance": equilibrium_tolerance,
                "refinement_tolerance": refinement,
                "maximum_force_evaluations": self.maximum_force_evaluations,
                "arrays": array_tree_fingerprint(
                    {"positions": positions, "translations": translation}
                ),
            }
        )

    def prepare(self, /) -> "PreparedFiniteDisplacementIFC2":
        return PreparedFiniteDisplacementIFC2(self)


class PreparedFiniteDisplacementIFC2(StrictModule, NonTrainableState):
    plan: FiniteDisplacementIFC2Plan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: FiniteDisplacementIFC2Plan, /):
        if not isinstance(plan, FiniteDisplacementIFC2Plan):
            raise TypeError("plan must be FiniteDisplacementIFC2Plan.")
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-finite-displacement-ifc2", "plan": plan.plan_id}
        )

    def evaluate(self, /) -> FiniteDisplacementIFC2Result:
        plan = self.plan
        positions = np.asarray(plan.equilibrium_positions)
        equilibrium = plan.potential.evaluate(
            plan.equilibrium_positions,
            plan.neighborhood.build(plan.equilibrium_positions),
        )
        equilibrium_residual = float(
            np.max(np.abs(np.asarray(equilibrium.forces)), initial=0.0)
        )
        hessians: list[np.ndarray] = []
        antisymmetry = 0.0
        all_successful = bool(equilibrium.successful)
        for step in (plan.displacement, 0.5 * plan.displacement):
            columns: list[np.ndarray] = []
            for coordinate in range(positions.size):
                shift = np.zeros_like(positions).reshape((-1,))
                shift[coordinate] = step
                shift = shift.reshape(positions.shape)
                plus_position = jnp.asarray(positions + shift)
                minus_position = jnp.asarray(positions - shift)
                plus = plan.potential.evaluate(
                    plus_position, plan.neighborhood.build(plus_position)
                )
                minus = plan.potential.evaluate(
                    minus_position, plan.neighborhood.build(minus_position)
                )
                plus_force = np.asarray(plus.forces).reshape((-1,))
                minus_force = np.asarray(minus.forces).reshape((-1,))
                columns.append(-(plus_force - minus_force) / (2.0 * step))
                antisymmetry = max(
                    antisymmetry,
                    float(
                        np.max(
                            np.abs(
                                plus_force
                                + minus_force
                                - 2.0 * np.asarray(equilibrium.forces).reshape((-1,))
                            ),
                            initial=0.0,
                        )
                    ),
                )
                all_successful = (
                    all_successful and bool(plus.successful) and bool(minus.successful)
                )
            hessians.append(np.stack(columns, axis=1))
        refinement = float(np.max(np.abs(hessians[1] - hessians[0]), initial=0.0))
        source, target = plan.image_map.route_supercell_indices(
            plan.relation, plan.translations
        )
        dense = hessians[1].reshape((positions.shape[0], 3, positions.shape[0], 3))
        raw = dense[target, :, source, :]
        cell = plan.image_map.primitive_cell
        fractional = np.asarray(plan.image_map.primitive_fractional_positions)
        reverse = _ifc2_reverse_indices(plan.relation, np.asarray(plan.translations))
        corrected, evidence = _project_ifc2(
            raw,
            plan.relation,
            np.asarray(plan.translations),
            fractional,
            cell,
            reverse,
            plan.constraint_policy,
        )
        units = plan.potential.system.plan.units
        artifact = SecondOrderForceConstants(
            plan.relation,
            plan.translations,
            raw,
            corrected,
            fractional,
            cell,
            second_order_force_constant_unit(
                units.scale.energy_unit, units.scale.length_unit
            ),
            evidence,
            system_id=plan.image_map.primitive_system_id,
            source_kind="native-finite-displacement-eam-sw-tersoff",
            source_id=plan.potential.prepared_id,
        )
        successful = (
            all_successful
            and equilibrium_residual <= plan.equilibrium_force_tolerance
            and refinement <= plan.refinement_tolerance
            and bool(evidence.successful)
        )
        return FiniteDisplacementIFC2Result(
            artifact,
            equilibrium_residual,
            antisymmetry,
            refinement,
            1 + 4 * positions.size,
            successful,
        )


def prepare_finite_displacement_ifc2(
    plan: FiniteDisplacementIFC2Plan, /
) -> PreparedFiniteDisplacementIFC2:
    return plan.prepare()


def evaluate_finite_displacement_ifc2(
    prepared: PreparedFiniteDisplacementIFC2, /
) -> FiniteDisplacementIFC2Result:
    if not isinstance(prepared, PreparedFiniteDisplacementIFC2):
        raise TypeError("prepared must be PreparedFiniteDisplacementIFC2.")
    return prepared.evaluate()


def normalize_second_order_force_constants(
    relation: EdgeRelation,
    translations: ArrayLike,
    raw_values: ArrayLike,
    fractional_positions: ArrayLike,
    cell: PeriodicCell,
    unit: UnitDefinition,
    /,
    *,
    system_id: str,
    source_kind: str,
    source_id: str,
    constraint_policy: IFCConstraintPolicy | None = None,
) -> SecondOrderForceConstants:
    """Normalize provider/native blocks through one explicit linear constraint map."""

    policy = IFCConstraintPolicy() if constraint_policy is None else constraint_policy
    if not isinstance(policy, IFCConstraintPolicy):
        raise TypeError("constraint_policy must be IFCConstraintPolicy or None.")
    translation = np.asarray(translations)
    raw = np.asarray(raw_values)
    positions = np.asarray(fractional_positions)
    reverse = _ifc2_reverse_indices(relation, translation)
    corrected, evidence = _project_ifc2(
        raw, relation, translation, positions, cell, reverse, policy
    )
    return SecondOrderForceConstants(
        relation,
        translation,
        raw,
        corrected,
        positions,
        cell,
        unit,
        evidence,
        system_id=system_id,
        source_kind=source_kind,
        source_id=source_id,
    )


def normalize_third_order_force_constants(*args, **kwargs) -> ThirdOrderForceConstants:
    """Construct a provider-normalized canonical IFC3 after full closure checks."""

    return ThirdOrderForceConstants(*args, **kwargs)


__all__ = [
    "PrimitiveSupercellImageMap",
    "FiniteDisplacementIFC2Plan",
    "FiniteDisplacementIFC2Result",
    "IFCConstraintEvidence",
    "IFCConstraintPolicy",
    "PreparedFiniteDisplacementIFC2",
    "SecondOrderForceConstants",
    "ThirdOrderForceConstants",
    "evaluate_finite_displacement_ifc2",
    "normalize_second_order_force_constants",
    "normalize_third_order_force_constants",
    "prepare_finite_displacement_ifc2",
    "second_order_force_constant_unit",
    "third_order_force_constant_unit",
]
