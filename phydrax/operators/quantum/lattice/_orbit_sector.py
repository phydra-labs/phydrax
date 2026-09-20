#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite monomial group actions and character-projected sector bases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ._sector import AbstractSectorBasis, SectorBasisResourcePolicy


class OrbitSectorResourcePolicy(StrictModule):
    """Hard closure, projected-dimension, and prepared-table limits."""

    maximum_group_order: int = eqx.field(static=True)
    maximum_orbit_dimension: int = eqx.field(static=True)
    maximum_table_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_group_order: int,
        maximum_orbit_dimension: int,
        maximum_table_bytes: int,
    ):
        values = (
            int(maximum_group_order),
            int(maximum_orbit_dimension),
            int(maximum_table_bytes),
        )
        if any(value < 1 for value in values):
            raise ValueError("Orbit-sector resource limits must be positive.")
        self.maximum_group_order = values[0]
        self.maximum_orbit_dimension = values[1]
        self.maximum_table_bytes = values[2]
        self.policy_id = canonical_fingerprint(
            {
                "kind": "orbit-sector-resource-policy",
                "maximum_group_order": values[0],
                "maximum_orbit_dimension": values[1],
                "maximum_table_bytes": values[2],
            }
        )


class MonomialConfigurationGenerator(StrictModule):
    """One configuration permutation with local phases and optional CAR sign.

    ``site_permutation[source]`` is the target site receiving the transformed
    local state. Local maps and phases are indexed by source site and incoming
    local-state index. Fermionic sites must use binary identity local maps; the
    induced occupied-mode permutation contributes its exact parity.
    """

    label: str = eqx.field(static=True)
    site_ids: tuple[str, ...] = eqx.field(static=True)
    site_dimensions: tuple[int, ...] = eqx.field(static=True)
    site_permutation: tuple[int, ...] = eqx.field(static=True)
    local_state_maps: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    local_phases: tuple[tuple[complex, ...], ...] = eqx.field(static=True)
    order: int = eqx.field(static=True)
    fermionic_sites: tuple[int, ...] = eqx.field(static=True)
    generator_id: str = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        site_ids: Sequence[str],
        site_dimensions: Sequence[int],
        site_permutation: Sequence[int],
        /,
        *,
        local_state_maps: Sequence[Sequence[int]] | None = None,
        order: int,
        local_phases: Sequence[Sequence[complex]] | None = None,
        fermionic_sites: Sequence[int] = (),
    ):
        name = str(label)
        sites = tuple(str(value) for value in site_ids)
        dimensions = tuple(site_dimensions)
        permutation = tuple(site_permutation)
        declared_order = int(order)
        if not name:
            raise ValueError("A symmetry-generator label must be non-empty.")
        if (
            not sites
            or any(not value for value in sites)
            or len(set(sites)) != len(sites)
        ):
            raise ValueError("Generator site IDs must be unique and non-empty.")
        if len(dimensions) != len(sites) or any(value < 1 for value in dimensions):
            raise ValueError("Generator site dimensions must be positive and complete.")
        if len(permutation) != len(sites) or tuple(sorted(permutation)) != tuple(
            range(len(sites))
        ):
            raise ValueError("site_permutation must be a permutation of all sites.")
        if declared_order < 1:
            raise ValueError("A symmetry-generator order must be positive.")

        maps = (
            tuple(tuple(range(dimension)) for dimension in dimensions)
            if local_state_maps is None
            else tuple(tuple(values) for values in local_state_maps)
        )
        if len(maps) != len(sites):
            raise ValueError("One local-state map is required per source site.")
        for source, values in enumerate(maps):
            target = permutation[source]
            if len(values) != dimensions[source]:
                raise ValueError("Local-state maps must cover every incoming state.")
            if any(value < 0 or value >= dimensions[target] for value in values):
                raise ValueError("A local-state map leaves the target local space.")
            if len(set(values)) != len(values) or len(values) != dimensions[target]:
                raise ValueError(
                    "Local-state maps must be bijections onto target spaces."
                )

        phases = (
            tuple(tuple(1.0 + 0.0j for _ in range(dimension)) for dimension in dimensions)
            if local_phases is None
            else tuple(tuple(complex(item) for item in values) for values in local_phases)
        )
        if len(phases) != len(sites) or any(
            len(values) != dimensions[index] for index, values in enumerate(phases)
        ):
            raise ValueError("Local phases must cover every source local state.")
        flat_phases = np.asarray([item for values in phases for item in values])
        if not np.all(np.isfinite(flat_phases)) or not np.allclose(
            np.abs(flat_phases), 1.0, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Local symmetry phases must be finite and unit magnitude.")

        fermions = tuple(fermionic_sites)
        if len(set(fermions)) != len(fermions) or any(
            value < 0 or value >= len(sites) for value in fermions
        ):
            raise ValueError("fermionic_sites must contain unique valid site indices.")
        fermion_set = set(fermions)
        if {permutation[index] for index in fermions} != fermion_set:
            raise ValueError(
                "A fermionic generator must preserve the fermionic site set."
            )
        for index in fermions:
            if dimensions[index] != 2 or maps[index] != (0, 1):
                raise ValueError(
                    "Fermionic site permutations require binary identity local maps."
                )

        self.label = name
        self.site_ids = sites
        self.site_dimensions = dimensions
        self.site_permutation = permutation
        self.local_state_maps = maps
        self.local_phases = phases
        self.order = declared_order
        self.fermionic_sites = fermions
        self.generator_id = canonical_fingerprint(
            {
                "kind": "monomial-configuration-generator",
                "label": name,
                "site_ids": sites,
                "site_dimensions": dimensions,
                "site_permutation": permutation,
                "order": declared_order,
                "local_state_maps": maps,
                "local_phases": tuple(
                    tuple((value.real, value.imag) for value in row) for row in phases
                ),
                "fermionic_sites": fermions,
            }
        )

    def apply(self, coordinate: ArrayLike, /) -> tuple[Array, Array]:
        """Apply the generator to one coordinate with its monomial phase."""
        raw = jnp.asarray(coordinate)
        if not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("Symmetry coordinates must use an integer dtype.")
        value = raw.astype(jnp.int32)
        if value.shape != (len(self.site_ids),):
            raise ValueError("A symmetry coordinate must provide one state per site.")
        dimensions = jnp.asarray(self.site_dimensions, dtype=jnp.int32)
        value = eqx.error_if(
            value,
            jnp.any((value < 0) | (value >= dimensions)),
            "A symmetry coordinate lies outside its local space.",
        )
        result = jnp.zeros_like(value)
        phase = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128)
        for source, target in enumerate(self.site_permutation):
            state = value[source]
            state_map = jnp.asarray(self.local_state_maps[source], dtype=jnp.int32)
            state_phase = jnp.asarray(self.local_phases[source], dtype=jnp.complex128)
            result = result.at[target].set(state_map[state])
            phase = phase * state_phase[state]
        if self.fermionic_sites:
            target_order = {
                site: ordinal for ordinal, site in enumerate(self.fermionic_sites)
            }
            inversion_parity = jnp.asarray(0, dtype=jnp.int32)
            for left_index, left in enumerate(self.fermionic_sites):
                for right in self.fermionic_sites[left_index + 1 :]:
                    reverses_order = (
                        target_order[self.site_permutation[left]]
                        > target_order[self.site_permutation[right]]
                    )
                    if reverses_order:
                        inversion_parity = inversion_parity + value[left] * value[right]
            phase = phase * jnp.where(inversion_parity % 2 == 0, 1.0, -1.0)
        return result, phase

    def _apply_host(self, coordinate: np.ndarray, /) -> tuple[np.ndarray, complex]:
        if coordinate.shape != (len(self.site_ids),):
            raise ValueError("A symmetry coordinate must provide one state per site.")
        result = np.zeros_like(coordinate, dtype=np.int32)
        phase = 1.0 + 0.0j
        for source, target in enumerate(self.site_permutation):
            state = int(coordinate[source])
            result[target] = self.local_state_maps[source][state]
            phase *= self.local_phases[source][state]
        if self.fermionic_sites:
            target_order = {
                site: ordinal for ordinal, site in enumerate(self.fermionic_sites)
            }
            occupied_targets = [
                self.site_permutation[source]
                for source in self.fermionic_sites
                if int(coordinate[source]) == 1
            ]
            inversions = sum(
                target_order[left] > target_order[right]
                for index, left in enumerate(occupied_targets)
                for right in occupied_targets[index + 1 :]
            )
            if inversions % 2:
                phase = -phase
        return result, phase


class FiniteGroupActionPlan(StrictModule):
    """Finite closure plan for monomial actions on one direct sector."""

    basis: AbstractSectorBasis
    generators: tuple[MonomialConfigurationGenerator, ...]
    resources: OrbitSectorResourcePolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: AbstractSectorBasis,
        generators: Sequence[MonomialConfigurationGenerator],
        resources: OrbitSectorResourcePolicy,
        /,
    ):
        if not isinstance(basis, AbstractSectorBasis):
            raise TypeError("basis must be an AbstractSectorBasis.")
        values = tuple(generators)
        if not values or any(
            not isinstance(value, MonomialConfigurationGenerator) for value in values
        ):
            raise TypeError("At least one monomial symmetry generator is required.")
        if not isinstance(resources, OrbitSectorResourcePolicy):
            raise TypeError("resources must be OrbitSectorResourcePolicy.")
        if len({value.label for value in values}) != len(values):
            raise ValueError("Symmetry-generator labels must be unique.")
        for value in values:
            if (
                value.site_ids != basis.site_ids
                or value.site_dimensions != basis.site_dimensions
            ):
                raise ValueError("A symmetry generator does not match the basis layout.")
        self.basis = basis
        self.generators = values
        self.resources = resources
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-group-action-plan",
                "basis": basis.basis_id,
                "generators": tuple(value.generator_id for value in values),
                "resources": resources.policy_id,
            }
        )


class CharacterSectorPlan(StrictModule):
    """One-dimensional unitary character specified on action generators."""

    label: str = eqx.field(static=True)
    generator_characters: tuple[tuple[str, complex], ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    character_id: str = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        generator_characters: Mapping[str, complex],
        /,
        *,
        tolerance: float = 1e-10,
    ):
        name = str(label)
        values = tuple(
            sorted(
                (str(key), complex(value)) for key, value in generator_characters.items()
            )
        )
        tol = float(tolerance)
        if not name:
            raise ValueError("A character-sector label must be non-empty.")
        if not values or any(not key for key, _ in values):
            raise ValueError("Generator characters must be non-empty and labeled.")
        raw = np.asarray([value for _, value in values], dtype=np.complex128)
        if not np.all(np.isfinite(raw)) or not np.allclose(
            np.abs(raw), 1.0, rtol=tol, atol=tol
        ):
            raise ValueError("Generator characters must be finite and unit magnitude.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tolerance must be positive and finite.")
        self.label = name
        self.generator_characters = values
        self.tolerance = tol
        self.character_id = canonical_fingerprint(
            {
                "kind": "character-sector-plan",
                "label": name,
                "generator_characters": tuple(
                    (key, value.real, value.imag) for key, value in values
                ),
                "tolerance": tol,
            }
        )

    def character_for(self, generator_label: str, /) -> complex:
        values = dict(self.generator_characters)
        try:
            return values[str(generator_label)]
        except KeyError as error:
            raise ValueError(
                f"No character value was supplied for generator {generator_label!r}."
            ) from error


class PreparedFiniteGroupAction(StrictModule):
    """Complete finite action and character tables on a direct sector."""

    plan: FiniteGroupActionPlan
    character: CharacterSectorPlan
    permutations: Array
    phases: Array
    characters: Array
    words: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    group_order: int = eqx.field(static=True)
    table_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PreparedOrbitSectorBasis(AbstractSectorBasis):
    """Normalized one-character projection of a direct sector basis."""

    base: AbstractSectorBasis
    action: PreparedFiniteGroupAction
    representatives: Array
    raw_to_orbit: Array
    embedding_coefficients: Array
    orbit_sizes: Array
    projection_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        base: AbstractSectorBasis,
        action: PreparedFiniteGroupAction,
        representatives: ArrayLike,
        raw_to_orbit: ArrayLike,
        embedding_coefficients: ArrayLike,
        orbit_sizes: ArrayLike,
        resources: SectorBasisResourcePolicy,
        /,
        *,
        projection_tolerance: float,
        basis_id: str,
    ):
        if not isinstance(base, AbstractSectorBasis):
            raise TypeError("base must be an AbstractSectorBasis.")
        if not isinstance(action, PreparedFiniteGroupAction):
            raise TypeError("action must be PreparedFiniteGroupAction.")
        if not isinstance(resources, SectorBasisResourcePolicy):
            raise TypeError("resources must be SectorBasisResourcePolicy.")
        representative_table = jnp.asarray(representatives, dtype=jnp.int32)
        raw_map = jnp.asarray(raw_to_orbit, dtype=jnp.int32)
        embedding = jnp.asarray(embedding_coefficients, dtype=jnp.complex128)
        sizes = jnp.asarray(orbit_sizes, dtype=jnp.int32)
        if (
            representative_table.ndim != 2
            or representative_table.shape[1] != len(base.site_ids)
            or representative_table.shape[0] < 1
        ):
            raise ValueError("representatives must be a nonempty coordinate table.")
        dimension = representative_table.shape[0]
        if raw_map.shape != (base.dimension,) or embedding.shape != (base.dimension,):
            raise ValueError("Raw orbit maps must cover the complete direct basis.")
        if sizes.shape != (dimension,):
            raise ValueError("orbit_sizes must provide one value per representative.")
        tolerance = float(projection_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("projection_tolerance must be positive and finite.")
        identifier = str(basis_id)
        if not identifier:
            raise ValueError("basis_id must be non-empty.")
        self.base = base
        self.action = action
        self.representatives = representative_table
        self.raw_to_orbit = raw_map
        self.embedding_coefficients = embedding
        self.orbit_sizes = sizes
        self.projection_tolerance = tolerance
        self.site_ids = base.site_ids
        self.site_dimensions = base.site_dimensions
        self.quantum_number_label = base.quantum_number_label
        self.quantum_number = base.quantum_number
        self.dimension = dimension
        self.resources = resources
        self.basis_id = identifier

    def coordinate(self, index: int | Array, /) -> Array:
        raw = jnp.asarray(index)
        if not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("An orbit-sector index must use an integer dtype.")
        rank = raw.astype(jnp.int32)
        if rank.shape != ():
            raise ValueError("An orbit-sector index must be scalar.")
        rank = eqx.error_if(
            rank,
            (rank < 0) | (rank >= self.dimension),
            "Orbit-sector index is out of range.",
        )
        return self._coordinate_unchecked(rank)

    def _coordinate_unchecked(self, rank: Array, /) -> Array:
        return self.representatives[rank]

    def contains(self, coordinate: ArrayLike, /) -> Array:
        raw = jnp.asarray(coordinate)
        if not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("Orbit-sector coordinates must use an integer dtype.")
        value = raw.astype(jnp.int32)
        if value.shape != (len(self.site_ids),):
            raise ValueError("An orbit coordinate must provide one state per site.")
        return self._contains_unchecked(value)

    def _contains_unchecked(self, coordinate: Array, /) -> Array:
        base_valid = self.base._contains_unchecked(coordinate)
        safe_coordinate = jnp.where(
            base_valid,
            coordinate,
            self.base._coordinate_unchecked(jnp.asarray(0, dtype=jnp.int64)),
        )
        raw_rank = self.base._rank_unchecked(safe_coordinate)
        orbit_rank = self.raw_to_orbit[raw_rank]
        safe_orbit = jnp.clip(orbit_rank, 0, self.dimension - 1)
        is_representative = jnp.all(self.representatives[safe_orbit] == safe_coordinate)
        return base_valid & (orbit_rank >= 0) & is_representative

    def rank(self, coordinate: ArrayLike, /) -> Array:
        raw = jnp.asarray(coordinate)
        if not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("Orbit-sector coordinates must use an integer dtype.")
        value = raw.astype(jnp.int32)
        if value.shape != (len(self.site_ids),):
            raise ValueError("An orbit coordinate must provide one state per site.")
        value = eqx.error_if(
            value,
            ~self._contains_unchecked(value),
            "Coordinate is not an orbit-sector representative.",
        )
        return self._rank_unchecked(value)

    def _rank_unchecked(self, coordinate: Array, /) -> Array:
        raw_rank = self.base._rank_unchecked(coordinate)
        return self.raw_to_orbit[raw_rank]

    def reduce(self, coordinate: ArrayLike, /) -> tuple[Array, Array, Array]:
        """Return orbit rank, normalized embedding coefficient, and validity."""
        raw = jnp.asarray(coordinate)
        if not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("Orbit reduction requires an integer coordinate.")
        value = raw.astype(jnp.int32)
        if value.shape != (len(self.site_ids),):
            raise ValueError("An orbit coordinate must provide one state per site.")
        valid_base = self.base._contains_unchecked(value)
        safe = jnp.where(
            valid_base,
            value,
            self.base._coordinate_unchecked(jnp.asarray(0, dtype=jnp.int64)),
        )
        raw_rank = self.base._rank_unchecked(safe)
        rank = self.raw_to_orbit[raw_rank]
        coefficient = self.embedding_coefficients[raw_rank]
        valid = (
            valid_base & (rank >= 0) & (jnp.abs(coefficient) > self.projection_tolerance)
        )
        return jnp.maximum(rank, 0), coefficient, valid


def _base_coordinates(basis: AbstractSectorBasis, /) -> np.ndarray:
    return np.stack(
        [
            np.asarray(basis.coordinate(index), dtype=np.int32)
            for index in range(basis.dimension)
        ],
        axis=0,
    )


def _generator_table(
    basis: AbstractSectorBasis,
    generator: MonomialConfigurationGenerator,
    coordinates: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    permutation = np.empty((basis.dimension,), dtype=np.int32)
    phases = np.empty((basis.dimension,), dtype=np.complex128)
    for rank, coordinate in enumerate(coordinates):
        target, phase = generator._apply_host(coordinate)
        if not bool(np.asarray(basis.contains(target))):
            raise ValueError(
                f"Generator {generator.label!r} does not preserve the direct sector."
            )
        permutation[rank] = int(np.asarray(basis.rank(target)))
        phases[rank] = phase
    if tuple(sorted(permutation.tolist())) != tuple(range(basis.dimension)):
        raise ValueError(f"Generator {generator.label!r} is not bijective on the sector.")
    if not np.allclose(np.abs(phases), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError(f"Generator {generator.label!r} is not unitary on the basis.")
    return permutation, phases


def _compose_actions(
    left_permutation: np.ndarray,
    left_phases: np.ndarray,
    right_permutation: np.ndarray,
    right_phases: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the action ``left`` after ``right``."""
    permutation = left_permutation[right_permutation]
    phases = right_phases * left_phases[right_permutation]
    return permutation, phases


def _matching_action(
    permutations: list[np.ndarray],
    phases: list[np.ndarray],
    characters: list[complex],
    candidate_permutation: np.ndarray,
    candidate_phases: np.ndarray,
    candidate_character: complex,
    tolerance: float,
    /,
) -> int | None:
    for index, (permutation, phase, character) in enumerate(
        zip(permutations, phases, characters, strict=True)
    ):
        if (
            np.array_equal(permutation, candidate_permutation)
            and np.allclose(phase, candidate_phases, rtol=tolerance, atol=tolerance)
            and np.allclose(
                character, candidate_character, rtol=tolerance, atol=tolerance
            )
        ):
            return index
    return None


def prepare_finite_group_action(
    plan: FiniteGroupActionPlan,
    character: CharacterSectorPlan,
    /,
) -> PreparedFiniteGroupAction:
    """Enumerate complete action closure and extend the generator character."""
    if not isinstance(plan, FiniteGroupActionPlan):
        raise TypeError("plan must be FiniteGroupActionPlan.")
    if not isinstance(character, CharacterSectorPlan):
        raise TypeError("character must be CharacterSectorPlan.")
    labels = {generator.label for generator in plan.generators}
    if {label for label, _ in character.generator_characters} != labels:
        raise ValueError("Character generator labels must exactly match the action plan.")

    coordinates = _base_coordinates(plan.basis)
    generator_tables = tuple(
        _generator_table(plan.basis, generator, coordinates)
        for generator in plan.generators
    )
    identity_permutation = np.arange(plan.basis.dimension, dtype=np.int32)
    identity_phases = np.ones((plan.basis.dimension,), dtype=np.complex128)
    for generator, (generator_permutation, generator_phases) in zip(
        plan.generators, generator_tables, strict=True
    ):
        power_permutation = identity_permutation
        power_phases = identity_phases
        for _ in range(generator.order):
            power_permutation, power_phases = _compose_actions(
                generator_permutation,
                generator_phases,
                power_permutation,
                power_phases,
            )
        if not np.array_equal(power_permutation, identity_permutation) or not np.allclose(
            power_phases,
            identity_phases,
            rtol=character.tolerance,
            atol=character.tolerance,
        ):
            raise ValueError(
                f"Generator {generator.label!r} does not satisfy its declared order."
            )
        if not np.allclose(
            character.character_for(generator.label) ** generator.order,
            1.0,
            rtol=character.tolerance,
            atol=character.tolerance,
        ):
            raise ValueError(
                f"Character for generator {generator.label!r} violates its order."
            )
    identity_permutation = np.arange(plan.basis.dimension, dtype=np.int32)
    identity_phases = np.ones((plan.basis.dimension,), dtype=np.complex128)
    permutations = [identity_permutation]
    phases = [identity_phases]
    characters = [1.0 + 0.0j]
    words: list[tuple[str, ...]] = [()]
    cursor = 0
    while cursor < len(permutations):
        base_permutation = permutations[cursor]
        base_phases = phases[cursor]
        base_character = characters[cursor]
        base_word = words[cursor]
        for generator, (generator_permutation, generator_phases) in zip(
            plan.generators, generator_tables, strict=True
        ):
            candidate_permutation, candidate_phases = _compose_actions(
                generator_permutation,
                generator_phases,
                base_permutation,
                base_phases,
            )
            candidate_character = base_character * character.character_for(
                generator.label
            )
            found = _matching_action(
                permutations,
                phases,
                characters,
                candidate_permutation,
                candidate_phases,
                candidate_character,
                character.tolerance,
            )
            if found is None:
                if len(permutations) >= plan.resources.maximum_group_order:
                    raise ValueError("Finite-group closure exceeds maximum_group_order.")
                permutations.append(candidate_permutation)
                phases.append(candidate_phases)
                characters.append(candidate_character)
                words.append(base_word + (generator.label,))
            else:
                continue
        cursor += 1

    permutation_table = np.stack(permutations, axis=0)
    phase_table = np.stack(phases, axis=0)
    character_table = np.asarray(characters, dtype=np.complex128)
    required = permutation_table.nbytes + phase_table.nbytes + character_table.nbytes
    if required > plan.resources.maximum_table_bytes:
        raise ValueError(
            f"Prepared finite-group tables require {required} bytes, exceeding "
            f"maximum_table_bytes {plan.resources.maximum_table_bytes}."
        )
    return PreparedFiniteGroupAction(
        plan=plan,
        character=character,
        permutations=jnp.asarray(permutation_table),
        phases=jnp.asarray(phase_table),
        characters=jnp.asarray(character_table),
        words=tuple(words),
        group_order=len(permutations),
        table_bytes=required,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-finite-group-action",
                "plan": plan.plan_id,
                "character": character.character_id,
                "permutations": array_tree_fingerprint(permutation_table),
                "phases": array_tree_fingerprint(phase_table),
                "characters": array_tree_fingerprint(character_table),
            }
        ),
    )


def prepare_orbit_sector_basis(
    action: PreparedFiniteGroupAction,
    /,
) -> PreparedOrbitSectorBasis:
    """Project a direct sector into one normalized character sector."""
    if not isinstance(action, PreparedFiniteGroupAction):
        raise TypeError("action must be PreparedFiniteGroupAction.")
    basis = action.plan.basis
    permutations = np.asarray(action.permutations, dtype=np.int32)
    phases = np.asarray(action.phases, dtype=np.complex128)
    characters = np.asarray(action.characters, dtype=np.complex128)
    visited = np.zeros((basis.dimension,), dtype=np.bool_)
    raw_to_orbit = np.full((basis.dimension,), -1, dtype=np.int32)
    embedding = np.zeros((basis.dimension,), dtype=np.complex128)
    representatives: list[np.ndarray] = []
    orbit_sizes: list[int] = []
    coordinates = _base_coordinates(basis)
    tolerance = action.character.tolerance

    for seed in range(basis.dimension):
        if visited[seed]:
            continue
        orbit = np.unique(permutations[:, seed])
        visited[orbit] = True
        representative = int(np.min(orbit))
        projected = np.zeros((basis.dimension,), dtype=np.complex128)
        np.add.at(
            projected,
            permutations[:, representative],
            np.conj(characters) * phases[:, representative],
        )
        norm = float(np.linalg.norm(projected))
        if norm <= tolerance:
            continue
        projected /= norm
        rank = len(representatives)
        if rank >= action.plan.resources.maximum_orbit_dimension:
            raise ValueError("Projected orbit dimension exceeds resource admission.")
        support = np.flatnonzero(np.abs(projected) > tolerance)
        raw_to_orbit[support] = rank
        embedding[support] = projected[support]
        representatives.append(coordinates[representative])
        orbit_sizes.append(support.size)

    if not representatives:
        raise ValueError("The requested character sector is empty.")
    representative_table = np.stack(representatives, axis=0).astype(np.int32)
    orbit_size_table = np.asarray(orbit_sizes, dtype=np.int32)
    required = (
        action.table_bytes
        + representative_table.nbytes
        + raw_to_orbit.nbytes
        + embedding.nbytes
        + orbit_size_table.nbytes
    )
    if required > action.plan.resources.maximum_table_bytes:
        raise ValueError(
            f"Orbit-sector tables require {required} bytes, exceeding "
            f"maximum_table_bytes {action.plan.resources.maximum_table_bytes}."
        )
    resources = SectorBasisResourcePolicy(
        maximum_dimension=action.plan.resources.maximum_orbit_dimension,
        maximum_table_bytes=action.plan.resources.maximum_table_bytes,
    )
    identifier = canonical_fingerprint(
        {
            "kind": "character-projected-orbit-sector-basis",
            "base": basis.basis_id,
            "action": action.prepared_id,
            "representatives": array_tree_fingerprint(representative_table),
            "embedding": array_tree_fingerprint(embedding),
        }
    )
    return PreparedOrbitSectorBasis(
        basis,
        action,
        representative_table,
        raw_to_orbit,
        embedding,
        orbit_size_table,
        resources,
        projection_tolerance=tolerance,
        basis_id=identifier,
    )


__all__ = [
    "CharacterSectorPlan",
    "FiniteGroupActionPlan",
    "MonomialConfigurationGenerator",
    "OrbitSectorResourcePolicy",
    "PreparedFiniteGroupAction",
    "PreparedOrbitSectorBasis",
    "prepare_finite_group_action",
    "prepare_orbit_sector_basis",
]
