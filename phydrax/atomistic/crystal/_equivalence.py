#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded certified crystal equivalence with explicit undetermined outcomes.

Two periodic structures are *equivalent* under an `EquivalencePolicy` when a
lattice basis change, an origin shift and a species-preserving atom bijection
carry one onto the other with every Cartesian residual at most
``position_tolerance`` and every reduced-metric entry within
``lattice_tolerance``; for molecular structures the bijection must also map
molecules onto molecules of the same component through an admitted
topology/stereo correspondence.  The search is a bounded host preparation:

1. native Niggli reduction of each cell with its integer basis change;
2. under ``supercell="primitive-reduction"`` the certified pure translations
   of each structure, exact rationalization of the translation group and the
   primitive basis from the native Smith decomposition;
3. every integer basis of the left lattice whose metric matches the right
   reduced metric, from coefficient boxes derived from the interplanar
   spacings (counted before enumeration);
4. origin shifts that align the least frequent species, followed by the
   native threshold assignment (Hungarian) at twice the tolerance, a
   least-squares shift refinement and a certifying assignment;
5. molecular correspondences from the native bounded graph search.

Step 4 is exhaustive: a certificate with residuals at most ``tol`` implies an
anchor-aligned assignment at ``2 tol``, so an infeasible ``2 tol`` assignment
excludes the candidate.  Distances use the componentwise nearest fractional
image, which decides ``distance <= 2 tol`` exactly because ``2 tol`` is
required to be below half the smallest interplanar spacing.  ``DISTINCT`` is
returned only when every bound was respected and no candidate was ambiguous;
exhausting any budget, or a candidate feasible at ``2 tol`` but not certified
at ``tol``, yields ``UNDETERMINED``.  Structure identities stay distinct from
the tolerance-based certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from fractions import Fraction
from math import isfinite
from typing import final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._assignment_core import hungarian_assignment_one
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...algebraic._exact_integer import (
    exact_integer_determinant,
    invert_unimodular,
    smith_normal_decomposition,
)
from ...typing import Dim, HostFloat64, HostInt64, parse
from ._asu import AsymmetricUnitPlan, ExpandedCrystal
from ._atom_types import molecular_correspondences
from ._input import MolecularComponent
from ._lattice import niggli_reduce


SupercellPolicy: TypeAlias = Literal["identical-lattice", "primitive-reduction"]
"""``identical-lattice`` compares the given translation lattices only;
``primitive-reduction`` compares certified primitive cells."""
EnantiomorphPolicy: TypeAlias = Literal["proper-only", "allow-improper"]
"""Whether a mirror (improper) isometry may establish equivalence."""

_ASSIGN = jax.jit(hungarian_assignment_one)


class EquivalenceStatus(StrEnum):
    EQUIVALENT = "equivalent"
    DISTINCT = "distinct"
    UNDETERMINED = "undetermined"


class StructureAtomDim(Dim, minimum=1):
    """Atoms of one compared structure."""


class ComparedAtomDim(Dim, minimum=1):
    """Atoms of one compared (reduced) cell."""


def _positive_float(value: float, name: str, /) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _positive_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


@final
class CrystalStructure(StrictModule, NonTrainableState):
    """Host periodic structure with explicit identity for certified comparison.

    ``vectors`` is the row cell in angstrom, ``fractional`` the fractional
    coordinates and ``species`` the atomic numbers.  ``molecules`` optionally
    lists ``(component, atoms)`` with ``atoms[i]`` the structure atom of local
    atom ``i``; species must agree with the component.  ``structure_id`` is the
    caller's actual identity (for example a structure-store record id);
    ``content_id`` fingerprints the exact numerical and molecular content.
    """

    __strict_contract__ = True

    vectors: HostFloat64[Literal[3], Literal[3]]
    fractional: HostFloat64[StructureAtomDim, Literal[3]]
    species: HostInt64[StructureAtomDim]
    molecule_index: HostInt64[StructureAtomDim]
    molecule_local: HostInt64[StructureAtomDim]
    molecule_components: tuple[MolecularComponent, ...]
    structure_id: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)

    def __init__(
        self,
        vectors: ArrayLike,
        fractional: ArrayLike,
        species: Sequence[int] | ArrayLike,
        /,
        *,
        structure_id: str,
        molecules: Sequence[tuple[MolecularComponent, Sequence[int]]] = (),
    ) -> None:
        cell = np.array(jax.device_get(vectors), dtype=np.float64)
        positions = np.array(jax.device_get(fractional), dtype=np.float64)
        numbers = np.array(jax.device_get(species), dtype=np.int64)
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            raise ValueError("vectors must be a finite 3x3 row cell.")
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
            raise ValueError("fractional must have shape (atoms, 3).")
        if not np.all(np.isfinite(positions)):
            raise ValueError("fractional coordinates must be finite.")
        if numbers.shape != (positions.shape[0],) or np.any(numbers < 1):
            raise ValueError("species must hold one positive atomic number per atom.")
        if not isinstance(structure_id, str) or not structure_id:
            raise ValueError("structure_id must be a non-empty string.")
        count = positions.shape[0]
        index = np.full((count,), -1, dtype=np.int64)
        local = np.full((count,), -1, dtype=np.int64)
        components = []
        for m, (component, atoms) in enumerate(molecules):
            if not isinstance(component, MolecularComponent):
                raise TypeError(
                    "molecules must pair a MolecularComponent with atom indices."
                )
            members = tuple(int(a) for a in atoms)
            if len(members) != component.atom_count:
                raise ValueError(
                    "A molecule must list one structure atom per component atom."
                )
            for i, a in enumerate(members):
                if not 0 <= a < count or index[a] >= 0:
                    raise ValueError("Molecule atoms must be distinct structure atoms.")
                if numbers[a] != component.atomic_numbers[i]:
                    raise ValueError("Molecule species disagree with the component.")
                index[a] = m
                local[a] = i
            components.append(component)
        for array in (cell, positions, numbers, index, local):
            array.setflags(write=False)
        self.vectors = cell
        self.fractional = positions
        self.species = numbers
        self.molecule_index = index
        self.molecule_local = local
        self.molecule_components = tuple(components)
        self.structure_id = structure_id
        self.content_id = canonical_fingerprint(
            {
                "kind": "crystal-structure-content",
                "arrays": array_tree_fingerprint(
                    {
                        "vectors": cell,
                        "fractional": positions,
                        "species": numbers,
                        "molecule_index": index,
                        "molecule_local": local,
                    }
                ),
                "components": [component.component_id for component in components],
            }
        )

    @property
    def atom_count(self) -> int:
        return int(self.species.shape[0])

    @property
    def molecular(self) -> bool:
        return bool(self.molecule_components)


@final
class EquivalencePolicy(StrictModule):
    """Declared tolerances, lattice/enantiomorph semantics and search budgets."""

    position_tolerance: float = eqx.field(static=True)
    lattice_tolerance: float = eqx.field(static=True)
    supercell: SupercellPolicy = eqx.field(static=True)
    enantiomorph: EnantiomorphPolicy = eqx.field(static=True)
    maximum_atoms: int = eqx.field(static=True)
    maximum_translation_order: int = eqx.field(static=True)
    maximum_lattice_vectors: int = eqx.field(static=True)
    maximum_basis_checks: int = eqx.field(static=True)
    maximum_bases: int = eqx.field(static=True)
    maximum_assignments: int = eqx.field(static=True)
    maximum_mappings: int = eqx.field(static=True)
    maximum_nodes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        position_tolerance: float,
        lattice_tolerance: float,
        supercell: SupercellPolicy,
        enantiomorph: EnantiomorphPolicy,
        maximum_atoms: int,
        maximum_translation_order: int,
        maximum_lattice_vectors: int,
        maximum_basis_checks: int,
        maximum_bases: int,
        maximum_assignments: int,
        maximum_mappings: int,
        maximum_nodes: int,
    ) -> None:
        values = {
            "position_tolerance": _positive_float(
                position_tolerance, "position_tolerance"
            ),
            "lattice_tolerance": _positive_float(lattice_tolerance, "lattice_tolerance"),
            "supercell": parse(supercell, SupercellPolicy, "supercell"),
            "enantiomorph": parse(enantiomorph, EnantiomorphPolicy, "enantiomorph"),
            "maximum_atoms": _positive_int(maximum_atoms, "maximum_atoms"),
            "maximum_translation_order": _positive_int(
                maximum_translation_order, "maximum_translation_order"
            ),
            "maximum_lattice_vectors": _positive_int(
                maximum_lattice_vectors, "maximum_lattice_vectors"
            ),
            "maximum_basis_checks": _positive_int(
                maximum_basis_checks, "maximum_basis_checks"
            ),
            "maximum_bases": _positive_int(maximum_bases, "maximum_bases"),
            "maximum_assignments": _positive_int(
                maximum_assignments, "maximum_assignments"
            ),
            "maximum_mappings": _positive_int(maximum_mappings, "maximum_mappings"),
            "maximum_nodes": _positive_int(maximum_nodes, "maximum_nodes"),
        }
        if values["lattice_tolerance"] >= 0.5:
            raise ValueError("lattice_tolerance must be below 0.5.")
        for name, value in values.items():
            setattr(self, name, value)
        self.policy_id = canonical_fingerprint(
            {"kind": "crystal-equivalence-policy", **values}
        )


@final
class EquivalenceCertificate(StrictModule, NonTrainableState):
    """Explicit isometry and atom bijection establishing one equivalence.

    ``left_basis`` and ``right_basis`` are the exact row-major ``3 x 3``
    matrices whose columns are the compared (reduced, possibly primitive)
    basis vectors in each input's fractional coordinates.  With the
    unimodular ``transform`` ``N`` and ``origin_shift`` ``tau``, compared
    coordinates satisfy ``s_right[mapping[i]] ~ N^-1 s_left[i] + tau`` modulo
    the lattice.  ``left_atoms``/``right_atoms`` name the input atoms of the
    compared atoms.  Residuals are Cartesian angstrom in the right metric;
    ``metric_residual`` is the largest relative reduced-metric deviation.
    """

    __strict_contract__ = True

    left_basis: tuple[Fraction, ...] = eqx.field(static=True)
    right_basis: tuple[Fraction, ...] = eqx.field(static=True)
    transform: tuple[int, ...] = eqx.field(static=True)
    origin_shift: HostFloat64[Literal[3]]
    left_atoms: HostInt64[ComparedAtomDim]
    right_atoms: HostInt64[ComparedAtomDim]
    mapping: HostInt64[ComparedAtomDim]
    improper: bool = eqx.field(static=True)
    max_residual: float = eqx.field(static=True)
    rms_residual: float = eqx.field(static=True)
    metric_residual: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)


@final
class EquivalenceResult(StrictModule, NonTrainableState):
    """Three-valued equivalence outcome with search and resource evidence.

    ``enantiomorphic`` is ``True`` when a certified improper isometry exists
    (under ``proper-only`` the structures are then mirror images and the status
    reflects only proper isometries), ``False`` when the improper search was
    complete without one, and ``None`` when it was not determined.
    ``exhausted`` names every bound that ended part of the search and
    ``ambiguous`` counts candidates feasible at ``2 tol`` but not certified.
    """

    status: EquivalenceStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    left_id: str = eqx.field(static=True)
    right_id: str = eqx.field(static=True)
    left_content_id: str = eqx.field(static=True)
    right_content_id: str = eqx.field(static=True)
    certificate: EquivalenceCertificate | None
    enantiomorphic: bool | None = eqx.field(static=True)
    exhausted: tuple[str, ...] = eqx.field(static=True)
    ambiguous: int = eqx.field(static=True)
    lattice_vectors: int = eqx.field(static=True)
    bases: int = eqx.field(static=True)
    assignments: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class _Compared(NamedTuple):
    vectors: np.ndarray
    fractional: np.ndarray
    species: np.ndarray
    atoms: np.ndarray
    molecule: np.ndarray
    local: np.ndarray
    components: tuple[MolecularComponent, ...]
    basis: tuple[Fraction, ...]


class _Match(NamedTuple):
    mapping: np.ndarray
    displacement: np.ndarray
    unique: bool


class _Budget:
    """Mutable host counters of one comparison (never part of a result identity)."""

    def __init__(self, policy: EquivalencePolicy, /) -> None:
        self.policy = policy
        self.assignments = 0
        self.lattice_vectors = 0
        self.basis_checks = 0
        self.bases = 0
        self.ambiguous = 0
        self.exhausted: list[str] = []

    def exhaust(self, name: str, /) -> None:
        if name not in self.exhausted:
            self.exhausted.append(name)

    def assignment(self) -> bool:
        if self.assignments >= self.policy.maximum_assignments:
            self.exhaust("assignments")
            return False
        self.assignments += 1
        return True


def _rational_matrix(
    values: Sequence[Sequence[int | Fraction]], /
) -> tuple[Fraction, ...]:
    return tuple(Fraction(values[r][c]) for r in range(3) for c in range(3))


def _rational_product(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...], /
) -> tuple[Fraction, ...]:
    return tuple(
        sum((left[3 * r + k] * right[3 * k + c] for k in range(3)), Fraction(0))
        for r in range(3)
        for c in range(3)
    )


def _interplanar_spacings(vectors: np.ndarray, /) -> np.ndarray:
    """Return ``V / |a_j x a_k|`` for the three lattice planes of a row cell."""
    volume = abs(_triple_product(vectors))
    crosses = np.stack(
        (
            np.cross(vectors[1], vectors[2]),
            np.cross(vectors[2], vectors[0]),
            np.cross(vectors[0], vectors[1]),
        )
    )
    return volume / np.linalg.norm(crosses, axis=1)


def _triple_product(vectors: np.ndarray, /) -> float:
    """Return the triple product ``a . (b x c)`` of a row cell."""
    return float(np.dot(vectors[0], np.cross(vectors[1], vectors[2])))


def _match(
    vectors: np.ndarray,
    left: np.ndarray,
    left_species: np.ndarray,
    right: np.ndarray,
    right_species: np.ndarray,
    shift: np.ndarray,
    threshold: float,
    /,
) -> _Match | None:
    """Return a species-preserving bijection with residuals at most ``threshold``."""
    difference = right[None, :, :] - (left[:, None, :] + shift)
    difference = difference - np.round(difference)
    cartesian = difference @ vectors
    squared = np.sum(cartesian * cartesian, axis=-1)
    valid = (left_species[:, None] == right_species[None, :]) & (
        squared <= threshold * threshold
    )
    if not np.all(np.any(valid, axis=1)) or not np.all(np.any(valid, axis=0)):
        return None
    assigned, _, _, feasible, _ = _ASSIGN(jnp.asarray(squared), jnp.asarray(valid))
    mapping = np.asarray(jax.device_get(assigned), dtype=np.int64)
    if not bool(jax.device_get(feasible)):
        return None
    rows = np.arange(mapping.shape[0])
    if np.any(mapping < 0) or len(set(mapping.tolist())) != mapping.shape[0]:
        return None
    if not np.all(valid[rows, mapping]):
        return None
    unique = bool(np.all(np.sum(valid, axis=1) == 1))
    return _Match(mapping, cartesian[rows, mapping], unique)


def _refined_shift(
    left: np.ndarray, right: np.ndarray, match: _Match, shift: np.ndarray, /
) -> np.ndarray:
    difference = right[match.mapping] - (left + shift)
    difference = difference - np.round(difference)
    return shift + np.mean(difference, axis=0)


def _molecules_consistent(
    left: _Compared,
    right: _Compared,
    mapping: np.ndarray,
    improper: bool,
    budget: _Budget,
    /,
) -> bool | None:
    """Check molecule-to-molecule topology/stereo correspondence; ``None`` if undecided."""
    if not left.components and not right.components:
        return True
    if not left.components or not right.components:
        return False
    images: dict[int, int] = {}
    local_maps: dict[int, dict[int, int]] = {}
    for i, j in enumerate(mapping.tolist()):
        m, n = int(left.molecule[i]), int(right.molecule[j])
        if (m < 0) != (n < 0):
            return False
        if m < 0:
            continue
        if images.setdefault(m, n) != n:
            return False
        local_maps.setdefault(m, {})[int(left.local[i])] = int(right.local[j])
    if len(set(images.values())) != len(images):
        return False
    undecided = False
    for m, n in images.items():
        source = left.components[m]
        target = right.components[n]
        local = local_maps[m]
        if len(local) != source.atom_count or source.atom_count != target.atom_count:
            return False
        induced = tuple(local[i] for i in range(source.atom_count))
        result = molecular_correspondences(
            source,
            target,
            max_mappings=budget.policy.maximum_mappings,
            max_nodes=budget.policy.maximum_nodes,
            mode="stereo",
            mirror=improper,
        )
        if induced in result.mappings:
            continue
        if not result.complete:
            budget.exhaust("molecular-correspondence")
            undecided = True
            continue
        return False
    return None if undecided else True


def _reduced(structure: CrystalStructure, /) -> _Compared | str:
    reduction = niggli_reduce(structure.vectors)
    if not reduction.successful:
        return "Niggli reduction of an input cell did not certify."
    transform = np.asarray(reduction.transformation, dtype=np.int64)
    inverse = np.asarray(invert_unimodular(transform), dtype=np.float64)
    fractional = structure.fractional @ inverse.T
    fractional = fractional - np.floor(fractional)
    components = structure.molecule_components
    return _Compared(
        np.asarray(reduction.vectors, dtype=np.float64),
        fractional,
        structure.species.copy(),
        np.arange(structure.atom_count, dtype=np.int64),
        structure.molecule_index.copy(),
        structure.molecule_local.copy(),
        components,
        _rational_matrix(transform.tolist()),
    )


def _translations(
    cell: _Compared, budget: _Budget, /
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str | None]:
    """Return certified pure translations with permutations, or an undetermined reason."""
    policy = budget.policy
    tolerance = policy.position_tolerance
    values, counts = np.unique(cell.species, return_counts=True)
    anchor_species = int(values[np.argmin(counts)])
    anchors = np.flatnonzero(cell.species == anchor_species)
    origin = int(anchors[0])
    identity = np.arange(cell.species.shape[0], dtype=np.int64)
    found: list[tuple[np.ndarray, np.ndarray]] = [(np.zeros(3), identity)]
    ambiguous = False
    for b in anchors[1:]:
        if not budget.assignment():
            return found, "The assignment budget ended the translation search."
        shift = cell.fractional[b] - cell.fractional[origin]
        shift = shift - np.round(shift)
        loose = _match(
            cell.vectors, cell.fractional, cell.species, cell.fractional, cell.species,
            shift, 2.0 * tolerance,
        )  # fmt: skip
        if loose is None:
            continue
        refined = _refined_shift(cell.fractional, cell.fractional, loose, shift)
        if not budget.assignment():
            return found, "The assignment budget ended the translation search."
        strict = _match(
            cell.vectors, cell.fractional, cell.species, cell.fractional, cell.species,
            refined, tolerance,
        )  # fmt: skip
        if strict is None:
            ambiguous = True
            budget.ambiguous += 1
            continue
        consistent = _molecules_consistent(cell, cell, strict.mapping, False, budget)
        if consistent is None:
            ambiguous = True
            continue
        if not consistent:
            if strict.unique:
                continue
            ambiguous = True
            budget.ambiguous += 1
            continue
        found.append((refined - np.round(refined), strict.mapping))
    if ambiguous:
        return (
            found,
            "A pure translation is feasible at twice the tolerance but not certified.",
        )
    return found, None


def _translation_group(
    cell: _Compared,
    found: Sequence[tuple[np.ndarray, np.ndarray]],
    order: int,
    tolerance: float,
    /,
) -> set[tuple[int, ...]] | str:
    """Exact numerators ``order * t mod order`` of a closed translation group."""
    numerators = []
    for shift, _ in found:
        scaled = np.round(shift * order).astype(np.int64)
        residual = (shift - scaled / order) @ cell.vectors
        if float(np.linalg.norm(residual)) > tolerance:
            return "A certified translation is not a rational of the group order."
        numerators.append(tuple(int(v) % order for v in scaled))
    members = set(numerators)
    for left in members:
        for right in members:
            if tuple((left[i] + right[i]) % order for i in range(3)) not in members:
                return "The certified translations are not closed under addition."
    return members


def _primitive_basis(
    members: set[tuple[int, ...]], order: int, /
) -> tuple[list[list[Fraction]], np.ndarray] | str:
    """Right-handed exact primitive basis ``B`` (columns) and ``B^-1`` as floats.

    The Smith form ``U [order I | numerators] = diag(d)`` of the lattice
    generated by ``Z^3`` and the translations gives ``B = U^-1 diag(d) / order``.
    """
    generators = [[order if r == c else 0 for c in range(3)] for r in range(3)]
    for vector in sorted(members):
        for r in range(3):
            generators[r].append(vector[r])
    diagonal, left_matrix, _ = smith_normal_decomposition(generators)
    pivots = [int(diagonal[i, i]) for i in range(3)]
    left_inverse = invert_unimodular(left_matrix)
    basis = [
        [Fraction(int(left_inverse[r, c]) * pivots[c], order) for c in range(3)]
        for r in range(3)
    ]
    scaled_rows = [[int(basis[r][c] * order) for c in range(3)] for r in range(3)]
    determinant = Fraction(exact_integer_determinant(scaled_rows), order**3)
    if abs(determinant) != Fraction(1, order):
        return "The primitive basis does not have the certified lattice index."
    # B = U^-1 diag(d) / order, so B^-1 = order diag(1 / d) U exactly.
    inverse_rows = [
        [Fraction(order * int(left_matrix[r, c]), pivots[r]) for c in range(3)]
        for r in range(3)
    ]
    if determinant < 0:
        for r in range(3):
            basis[r][0] = -basis[r][0]
        inverse_rows = [[-value for value in inverse_rows[0]], *inverse_rows[1:]]
    inverse = np.asarray(
        [[float(v) for v in row] for row in inverse_rows], dtype=np.float64
    )
    return basis, inverse


def _orbit_representatives(
    cell: _Compared, found: Sequence[tuple[np.ndarray, np.ndarray]], /
) -> np.ndarray:
    """Keep mask of translation-orbit representatives.

    Molecules keep the least molecule of each orbit, unbound atoms the least atom.
    """
    keep = np.zeros((cell.species.shape[0],), dtype=np.bool_)
    seen_molecules: set[int] = set()
    seen_atoms: set[int] = set()
    for a in range(cell.species.shape[0]):
        m = int(cell.molecule[a])
        if m >= 0:
            if m in seen_molecules:
                continue
            orbit = {int(cell.molecule[mapping[a]]) for _, mapping in found}
            seen_molecules.update(orbit)
            keep |= cell.molecule == m
        else:
            if a in seen_atoms:
                continue
            seen_atoms.update(int(mapping[a]) for _, mapping in found)
            keep[a] = True
    return keep


def _primitive_subcell(
    cell: _Compared,
    keep: np.ndarray,
    basis: list[list[Fraction]],
    inverse: np.ndarray,
    /,
) -> _Compared:
    """Kept atoms in the primitive basis, molecules renumbered in kept order."""
    fractional = cell.fractional[keep] @ inverse.T
    fractional = fractional - np.floor(fractional)
    vectors = np.asarray([[float(v) for v in row] for row in basis]).T @ cell.vectors
    kept_molecules = sorted({int(m) for m in cell.molecule[keep] if m >= 0})
    renumber = {m: i for i, m in enumerate(kept_molecules)}
    molecule = np.asarray(
        [renumber[int(m)] if m >= 0 else -1 for m in cell.molecule[keep]], dtype=np.int64
    )
    return _Compared(
        vectors,
        fractional,
        cell.species[keep],
        cell.atoms[keep],
        molecule,
        cell.local[keep],
        tuple(cell.components[m] for m in kept_molecules),
        _rational_product(cell.basis, _rational_matrix(basis)),
    )


def _primitive(cell: _Compared, budget: _Budget, /) -> _Compared | str:
    """Return the certified primitive reduced cell of a Niggli-reduced structure."""
    policy = budget.policy
    found, reason = _translations(cell, budget)
    if reason is not None:
        return reason
    order = len(found)
    if order == 1:
        return cell
    if order > policy.maximum_translation_order:
        budget.exhaust("translation-order")
        return f"The translation group order {order} exceeds maximum_translation_order."
    count = cell.species.shape[0]
    if count % order:
        return "The certified translations do not divide the atom count."
    members = _translation_group(cell, found, order, policy.position_tolerance)
    if isinstance(members, str):
        return members
    lattice = _primitive_basis(members, order)
    if isinstance(lattice, str):
        return lattice
    basis, inverse = lattice
    keep = _orbit_representatives(cell, found)
    if int(np.sum(keep)) * order != count:
        return "Translation orbits do not partition the atoms evenly."
    primitive = _primitive_subcell(cell, keep, basis, inverse)
    reduction = niggli_reduce(primitive.vectors)
    if not reduction.successful:
        return "Niggli reduction of a primitive cell did not certify."
    transform = np.asarray(reduction.transformation, dtype=np.int64)
    reduced = (
        primitive.fractional
        @ np.asarray(invert_unimodular(transform), dtype=np.float64).T
    )
    return primitive._replace(
        vectors=np.asarray(reduction.vectors, dtype=np.float64),
        fractional=reduced - np.floor(reduced),
        basis=_rational_product(primitive.basis, _rational_matrix(transform.tolist())),
    )


def _bases(
    left: _Compared, right: _Compared, budget: _Budget, /
) -> tuple[list[tuple[np.ndarray, float]], bool]:
    """Return integer bases ``N`` of the left lattice matching the right metric."""
    policy = budget.policy
    tolerance = policy.lattice_tolerance
    target = right.vectors @ right.vectors.T
    lengths = np.sqrt(np.diag(target))
    spacing = _interplanar_spacings(left.vectors)
    bound = np.floor(np.sqrt(1.0 + tolerance) * np.max(lengths) / spacing).astype(
        np.int64
    )
    box = int(np.prod(2 * bound + 1))
    if box > policy.maximum_lattice_vectors:
        budget.exhaust("lattice-vectors")
        return [], False
    budget.lattice_vectors += box
    grids = np.meshgrid(*(np.arange(-b, b + 1) for b in bound), indexing="ij")
    integers = np.stack([grid.reshape(-1) for grid in grids], axis=1)
    cartesian = integers @ left.vectors
    squared = np.sum(cartesian * cartesian, axis=1)
    candidates = [
        integers[np.abs(squared - target[j, j]) <= tolerance * target[j, j]]
        for j in range(3)
    ]
    found: list[tuple[np.ndarray, float]] = []
    complete = True
    for first in candidates[0]:
        u = first @ left.vectors
        for second in candidates[1]:
            v = second @ left.vectors
            budget.basis_checks += 1
            if budget.basis_checks > policy.maximum_basis_checks:
                budget.exhaust("basis-checks")
                return found, False
            if abs(u @ v - target[0, 1]) > tolerance * lengths[0] * lengths[1]:
                continue
            for third in candidates[2]:
                budget.basis_checks += 1
                if budget.basis_checks > policy.maximum_basis_checks:
                    budget.exhaust("basis-checks")
                    return found, False
                w = third @ left.vectors
                if abs(u @ w - target[0, 2]) > tolerance * lengths[0] * lengths[2]:
                    continue
                if abs(v @ w - target[1, 2]) > tolerance * lengths[1] * lengths[2]:
                    continue
                matrix = np.stack((first, second, third), axis=1)
                if abs(exact_integer_determinant(matrix.tolist())) != 1:
                    continue
                gram = np.stack((u, v, w)) @ np.stack((u, v, w)).T
                residual = float(
                    np.max(
                        np.abs(gram - target)
                        / np.sqrt(np.outer(np.diag(target), np.diag(target)))
                    )
                )
                if len(found) >= policy.maximum_bases:
                    budget.exhaust("bases")
                    return found, False
                found.append((matrix, residual))
                budget.bases += 1
    return found, complete


def _certificate(
    left: _Compared,
    right: _Compared,
    transform: np.ndarray,
    shift: np.ndarray,
    match: _Match,
    improper: bool,
    metric_residual: float,
    /,
) -> EquivalenceCertificate:
    norms = np.linalg.norm(match.displacement, axis=1)
    origin = np.asarray(shift - np.floor(shift), dtype=np.float64)
    payload = {
        "kind": "crystal-equivalence-certificate",
        "left_basis": [str(v) for v in left.basis],
        "right_basis": [str(v) for v in right.basis],
        "transform": [int(v) for v in transform.reshape(-1)],
        "origin_shift": [float(v).hex() for v in origin],
        "left_atoms": [int(v) for v in left.atoms],
        "right_atoms": [int(v) for v in right.atoms],
        "mapping": [int(v) for v in match.mapping],
        "improper": improper,
    }
    return EquivalenceCertificate(
        left.basis,
        right.basis,
        tuple(int(v) for v in transform.reshape(-1)),
        origin,
        left.atoms.copy(),
        right.atoms.copy(),
        match.mapping.copy(),
        improper,
        float(np.max(norms)),
        float(np.sqrt(np.mean(norms * norms))),
        metric_residual,
        canonical_fingerprint(payload),
    )


def _search(
    left: _Compared,
    right: _Compared,
    bases: list[tuple[np.ndarray, float]],
    improper_flags: list[bool],
    budget: _Budget,
    /,
) -> tuple[EquivalenceCertificate | None, bool]:
    """Return the first certificate among ``bases`` and whether the search was exhaustive."""
    tolerance = budget.policy.position_tolerance
    values, counts = np.unique(left.species, return_counts=True)
    anchor_species = int(values[np.argmin(counts)])
    origin = int(np.flatnonzero(left.species == anchor_species)[0])
    targets = np.flatnonzero(right.species == anchor_species)
    exhaustive = True
    for (matrix, metric_residual), improper in zip(bases, improper_flags, strict=True):
        inverse = np.asarray(invert_unimodular(matrix), dtype=np.float64)
        moved = left.fractional @ inverse.T
        for b in targets:
            if not budget.assignment():
                return None, False
            shift = right.fractional[b] - moved[origin]
            loose = _match(
                right.vectors, moved, left.species, right.fractional, right.species,
                shift, 2.0 * tolerance,
            )  # fmt: skip
            if loose is None:
                continue
            refined = _refined_shift(moved, right.fractional, loose, shift)
            if not budget.assignment():
                return None, False
            strict = _match(
                right.vectors, moved, left.species, right.fractional, right.species,
                refined, tolerance,
            )  # fmt: skip
            if strict is None:
                budget.ambiguous += 1
                exhaustive = False
                continue
            consistent = _molecules_consistent(
                left, right, strict.mapping, improper, budget
            )
            if consistent is True:
                return (
                    _certificate(
                        left, right, matrix, refined, strict, improper, metric_residual
                    ),
                    exhaustive,
                )
            if consistent is None or not strict.unique:
                budget.ambiguous += 1
                exhaustive = False
    return None, exhaustive


def _composition(structure: CrystalStructure, /) -> dict[int, int]:
    values, counts = np.unique(structure.species, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts, strict=True)}


def _result(
    status: EquivalenceStatus,
    reason: str,
    left: CrystalStructure,
    right: CrystalStructure,
    policy: EquivalencePolicy,
    budget: _Budget,
    /,
    *,
    certificate: EquivalenceCertificate | None = None,
    enantiomorphic: bool | None = None,
) -> EquivalenceResult:
    exhausted = tuple(budget.exhausted)
    payload = {
        "kind": "crystal-equivalence-result",
        "status": status.value,
        "reason": reason,
        "left_content_id": left.content_id,
        "right_content_id": right.content_id,
        "certificate": None if certificate is None else certificate.certificate_id,
        "enantiomorphic": enantiomorphic,
        "exhausted": list(exhausted),
        "policy_id": policy.policy_id,
    }
    return EquivalenceResult(
        status,
        reason,
        left.structure_id,
        right.structure_id,
        left.content_id,
        right.content_id,
        certificate,
        enantiomorphic,
        exhausted,
        budget.ambiguous,
        budget.lattice_vectors,
        budget.bases,
        budget.assignments,
        policy.policy_id,
        canonical_fingerprint(payload),
    )


def certify_equivalence(
    left: CrystalStructure, right: CrystalStructure, policy: EquivalencePolicy, /
) -> EquivalenceResult:
    """Decide equivalence of two periodic structures within declared bounds."""
    if not isinstance(left, CrystalStructure) or not isinstance(right, CrystalStructure):
        raise TypeError("left and right must be CrystalStructure values.")
    if not isinstance(policy, EquivalencePolicy):
        raise TypeError("policy must be an EquivalencePolicy.")
    if left.molecular != right.molecular:
        raise ValueError("Both structures must declare molecules, or neither.")
    budget = _Budget(policy)
    statuses = EquivalenceStatus
    if max(left.atom_count, right.atom_count) > policy.maximum_atoms:
        budget.exhaust("atoms")
        return _result(
            statuses.UNDETERMINED,
            "A structure exceeds maximum_atoms.",
            left,
            right,
            policy,
            budget,
        )
    left_composition, right_composition = _composition(left), _composition(right)
    if set(left_composition) != set(right_composition):
        return _result(
            statuses.DISTINCT, "The element sets differ.", left, right, policy, budget
        )
    divisor_left = np.gcd.reduce(np.asarray(list(left_composition.values())))
    divisor_right = np.gcd.reduce(np.asarray(list(right_composition.values())))
    if any(
        left_composition[z] // divisor_left != right_composition[z] // divisor_right
        for z in left_composition
    ):
        return _result(
            statuses.DISTINCT,
            "The reduced compositions differ.",
            left,
            right,
            policy,
            budget,
        )
    if policy.supercell == "identical-lattice" and left.atom_count != right.atom_count:
        return _result(
            statuses.DISTINCT,
            "The atom counts differ under the identical-lattice policy.",
            left,
            right,
            policy,
            budget,
        )
    compared = []
    for structure in (left, right):
        reduced = _reduced(structure)
        if isinstance(reduced, str):
            return _result(statuses.UNDETERMINED, reduced, left, right, policy, budget)
        if policy.supercell == "primitive-reduction":
            spacing = float(np.min(_interplanar_spacings(reduced.vectors)))
            if 4.0 * policy.position_tolerance >= spacing:
                return _result(
                    statuses.UNDETERMINED,
                    "Twice the position tolerance reaches half an interplanar spacing.",
                    left,
                    right,
                    policy,
                    budget,
                )
            reduced = _primitive(reduced, budget)
            if isinstance(reduced, str):
                return _result(
                    statuses.UNDETERMINED, reduced, left, right, policy, budget
                )
        compared.append(reduced)
    left_cell, right_cell = compared
    if left_cell.species.shape[0] != right_cell.species.shape[0]:
        return _result(
            statuses.DISTINCT,
            "The compared cells hold different numbers of atoms.",
            left,
            right,
            policy,
            budget,
        )
    if 4.0 * policy.position_tolerance >= float(
        np.min(_interplanar_spacings(right_cell.vectors))
    ):
        return _result(
            statuses.UNDETERMINED,
            "Twice the position tolerance reaches half an interplanar spacing.",
            left,
            right,
            policy,
            budget,
        )
    bases, complete = _bases(left_cell, right_cell, budget)
    sign = np.sign(_triple_product(left_cell.vectors)) * np.sign(
        _triple_product(right_cell.vectors)
    )
    flags = [
        bool(sign * exact_integer_determinant(matrix.tolist()) < 0) for matrix, _ in bases
    ]
    proper = [basis for basis, flag in zip(bases, flags, strict=True) if not flag]
    mirror = [basis for basis, flag in zip(bases, flags, strict=True) if flag]
    certificate, proper_exhaustive = _search(
        left_cell, right_cell, proper, [False] * len(proper), budget
    )
    if certificate is not None:
        return _result(
            statuses.EQUIVALENT,
            "A proper isometry and bijection were certified.",
            left,
            right,
            policy,
            budget,
            certificate=certificate,
        )
    proper_complete = complete and proper_exhaustive and not budget.exhausted
    exhausted_before = len(budget.exhausted)
    mirror_certificate, mirror_exhaustive = _search(
        left_cell, right_cell, mirror, [True] * len(mirror), budget
    )
    mirror_complete = (
        complete and mirror_exhaustive and len(budget.exhausted) == exhausted_before
    )
    enantiomorphic = (
        True if mirror_certificate is not None else (False if mirror_complete else None)
    )
    if mirror_certificate is not None and policy.enantiomorph == "allow-improper":
        return _result(
            statuses.EQUIVALENT,
            "An improper isometry and bijection were certified.",
            left,
            right,
            policy,
            budget,
            certificate=mirror_certificate,
            enantiomorphic=True,
        )
    decided = proper_complete and (
        policy.enantiomorph == "proper-only" or mirror_complete
    )
    if decided:
        return _result(
            statuses.DISTINCT,
            "Every admissible basis and origin shift was excluded"
            + (
                "; an improper isometry exists, so the structures are enantiomorphs."
                if enantiomorphic
                else "."
            ),
            left,
            right,
            policy,
            budget,
            enantiomorphic=enantiomorphic,
        )
    return _result(
        statuses.UNDETERMINED,
        "A bounded search ended or a candidate was ambiguous; this is not proof of "
        "distinction.",
        left,
        right,
        policy,
        budget,
        enantiomorphic=enantiomorphic,
    )


def _copy_molecules(
    plan: AsymmetricUnitPlan, /
) -> list[tuple[MolecularComponent, tuple[int, ...]]]:
    """Return ``(component, atoms)`` of every expanded copy with its stereo identity.

    A mirror copy of a stereo-chiral component (``plan.copy_enantiomer``) is the
    opposite stereoisomer: its declaration is ``component.enantiomer()`` (every
    tetrahedral parity inverted, constructed once per component), so a racemate
    exports opposite-handed molecules with opposite declared stereo.
    """
    mirrored: dict[str, MolecularComponent] = {}
    molecules = []
    for copy in range(plan.copy_count):
        component = plan.sites[int(plan.copy_site[copy])].pool.component
        if bool(plan.copy_enantiomer[copy]):
            key = component.component_id
            if key not in mirrored:
                mirrored[key] = component.enantiomer()
            component = mirrored[key]
        atoms = np.flatnonzero(plan.atom_copy == copy)
        molecules.append((component, tuple(int(a) for a in atoms)))
    return molecules


def crystal_structure_from_expansion(
    plan: AsymmetricUnitPlan, expansion: ExpandedCrystal, /, *, structure_id: str
) -> CrystalStructure:
    """Return the host structure of an expansion with one molecule per copy.

    Mirror copies of stereo-chiral components carry ``component.enantiomer()``.
    """
    if not isinstance(plan, AsymmetricUnitPlan) or not isinstance(
        expansion, ExpandedCrystal
    ):
        raise TypeError(
            "plan and expansion must be an AsymmetricUnitPlan and ExpandedCrystal."
        )
    if expansion.plan_id != plan.plan_id:
        raise ValueError("expansion belongs to another plan.")
    species = [
        plan.sites[int(site)].pool.component.atomic_numbers[int(local)]
        for site, local in zip(plan.atom_site, plan.atom_local, strict=True)
    ]
    return CrystalStructure(
        np.asarray(jax.device_get(expansion.cell.vectors), dtype=np.float64),
        np.asarray(jax.device_get(expansion.fractional), dtype=np.float64),
        species,
        structure_id=structure_id,
        molecules=_copy_molecules(plan),
    )


__all__ = [
    "certify_equivalence",
    "crystal_structure_from_expansion",
    "CrystalStructure",
    "EnantiomorphPolicy",
    "EquivalenceCertificate",
    "EquivalencePolicy",
    "EquivalenceResult",
    "EquivalenceStatus",
    "SupercellPolicy",
]
