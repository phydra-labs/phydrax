"""One executable per prepared-Hamiltonian layout across material/parameter revisions."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._identity import IdentityWords
from phydrax.discretization import ParticlePairKeySpace
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_program,
    role_capacity,
    search_capacity,
)


_FRAME = phx.discretization.PeriodicCell(
    20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
)
_POSITIONS = jnp.asarray(
    [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [0.3, 1.05, 0.0]], dtype=jnp.float64
)
_MATERIALS = {
    "a": ([10, 20, 30], [6, 8, 1], [12.0, 16.0, 1.0], [0, 1, 0], [0.3, -0.4, 0.1]),
    "b": ([11, 21, 31], [7, 7, 1], [14.0, 14.0, 1.0], [1, 0, 1], [-0.2, 0.2, 0.0]),
}
_REVISIONS = {"1": ([1.0, 0.5], [1.0, 1.2]), "2": ([0.8, 0.7], [1.1, 0.9])}


def _program(material: str, revision: str) -> Any:
    ids, numbers, masses, types, charges = _MATERIALS[material]
    epsilon, sigma = _REVISIONS[revision]
    system = phx.atomistic.AtomisticSystemPlan(
        np.asarray(ids, dtype=np.int64),
        np.asarray(numbers, dtype=np.int32),
        np.asarray(masses, dtype=np.float64),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.asarray(types, dtype=np.int32),
        charges=np.asarray(charges, dtype=np.float64),
    ).prepare()
    return prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            (
                phx.atomistic.LennardJonesPotential(
                    np.asarray(epsilon), np.asarray(sigma), 2.5, switch_distance=2.0
                ),
                phx.atomistic.DirectCoulombPotential(),
            )
        ),
        system,
        search=phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
        skin=0.5,
        search_capacities=(search_capacity(16, particles_per_cell=3, images=27),),
        role_capacities=(role_capacity(3, 2),),
        maximum_pairs=3,
        frame=_FRAME,
    )


def test_identity_words_are_a_lossless_digest_representation() -> None:
    digest = "0123456789abcdef" * 4
    words = IdentityWords.from_fingerprint(digest)
    assert words.words.dtype == jnp.uint32 and words.words.shape == (8,)
    assert words.fingerprint() == digest
    assert bool(words.equals(IdentityWords.from_fingerprint(digest)))
    assert not bool(words.equals(IdentityWords.from_fingerprint("f" * 64)))
    with pytest.raises(ValueError):
        IdentityWords.from_fingerprint("a-label-is-not-a-digest")

    @jax.jit
    def decode(value: IdentityWords) -> Any:
        return value.fingerprint()

    with pytest.raises(TypeError):
        decode(words)


def test_schema_pair_key_space_has_content_free_statics() -> None:
    first = ParticlePairKeySpace(particle_ids=np.asarray([4, 2, 9]), support_id="schema")
    second = ParticlePairKeySpace(particle_ids=np.asarray([5, 1, 8]), support_id="schema")
    assert first.particle_discretization_id is None
    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)
    assert not np.array_equal(first.sorted_particle_ids, second.sorted_particle_ids)


def test_revisions_of_one_layout_share_treedef_and_one_trace() -> None:
    programs = {
        (material, revision): _program(material, revision)
        for material in _MATERIALS
        for revision in _REVISIONS
    }
    structures = {jax.tree_util.tree_structure(value) for value in programs.values()}
    assert len(structures) == 1
    assert len({value.layout_id for value in programs.values()}) == 1
    assert len({value.scientific_id for value in programs.values()}) == len(programs)
    assert len({value.prepared_id for value in programs.values()}) == len(programs)
    traces = []

    @eqx.filter_jit
    def evaluate(program: Any, state: Any) -> Any:
        traces.append(None)
        result = program.evaluate(_POSITIONS, state)
        return result.energy, result.forces, result.successful

    energies = []
    for program in programs.values():
        state = bundle_state(program, _POSITIONS)
        energy, forces, successful = evaluate(program, state)
        eager = program.evaluate(_POSITIONS, state)
        assert bool(successful)
        np.testing.assert_allclose(energy, eager.energy, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(forces, eager.forces, rtol=1e-12, atol=1e-12)
        assert eager.program_id == program.prepared_id
        energies.append(float(energy))
    assert len(traces) == 1
    assert len(set(np.round(energies, 12).tolist())) == len(energies)


def test_foreign_material_state_is_refused_by_status() -> None:
    owner = _program("a", "1")
    foreign = _program("b", "1")
    result = owner.evaluate(_POSITIONS, bundle_state(foreign, _POSITIONS))
    assert not bool(result.successful)
    assert bool(jnp.isnan(result.energy))
    same_material = _program("a", "2")
    accepted = owner.evaluate(_POSITIONS, bundle_state(same_material, _POSITIONS))
    assert bool(accepted.successful)


def test_equal_shape_algorithm_selectors_have_distinct_layouts() -> None:
    system = _program("a", "1").system
    plane = phx.atomistic.ScalarWallPotential(
        phx.atomistic.WallKind.PLANE,
        np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float64),
        1.0,
    ).prepare(system)
    sphere = phx.atomistic.ScalarWallPotential(
        phx.atomistic.WallKind.SPHERE,
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        1.0,
    ).prepare(system)
    revised_plane = phx.atomistic.ScalarWallPotential(
        phx.atomistic.WallKind.PLANE,
        np.asarray([0.0, 1.0, 0.0, 1.5], dtype=np.float64),
        2.0,
    ).prepare(system)
    assert plane.layout_id != sphere.layout_id
    assert plane.layout_id == revised_plane.layout_id
    assert plane.prepared_id != revised_plane.prepared_id
    assert (
        phx.atomistic.atomistic_executable_signature(
            plane, layout_id=plane.layout_id
        ).signature_id
        != phx.atomistic.atomistic_executable_signature(
            sphere, layout_id=sphere.layout_id
        ).signature_id
    )
