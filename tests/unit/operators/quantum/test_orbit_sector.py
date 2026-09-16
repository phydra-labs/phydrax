#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.lifecycle._array_artifact import ArrayArtifactProvenance
from phydrax.operators.quantum import FermionModeOrder
from phydrax.operators.quantum.lattice import (
    CharacterSectorPlan,
    FiniteGroupActionPlan,
    FixedCardinalityFermionBasis,
    FixedSpinProjectionBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    MonomialConfigurationGenerator,
    OrbitOperatorResourcePolicy,
    OrbitSectorResourcePolicy,
    prepare_finite_group_action,
    prepare_orbit_sector_basis,
    prepare_quantum_lattice,
    prepare_quantum_orbit_sector_operator,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    read_quantum_lattice_artifact_archive,
    SectorBasisResourcePolicy,
    write_quantum_lattice_artifact_archive,
)


def _basis_resources():
    return SectorBasisResourcePolicy(maximum_dimension=128, maximum_table_bytes=100_000)


def _orbit_resources():
    return OrbitSectorResourcePolicy(
        maximum_group_order=32,
        maximum_orbit_dimension=128,
        maximum_table_bytes=200_000,
    )


def _operator_resources():
    return OrbitOperatorResourcePolicy(
        maximum_routes=1_024,
        maximum_workspace_bytes=1_000_000,
    )


def _compiler_resources():
    return QuantumLatticeResourcePolicy(
        maximum_terms=32,
        maximum_factors_per_term=4,
        maximum_branches_per_input=256,
        maximum_sector_dimension=128,
        maximum_workspace_bytes=1_000_000,
    )


def _translation_generator(site_ids, dimensions):
    count = len(site_ids)
    return MonomialConfigurationGenerator(
        "translation",
        site_ids,
        dimensions,
        tuple((index + 1) % count for index in range(count)),
        order=count,
    )


def _prepare_character_basis(base, generator, character):
    action = prepare_finite_group_action(
        FiniteGroupActionPlan(base, (generator,), _orbit_resources()),
        CharacterSectorPlan("momentum", {generator.label: character}),
    )
    return prepare_orbit_sector_basis(action)


def _ring_exchange(site_count):
    spaces = tuple(LocalSpacePlan.spin(f"s{index}", 1) for index in range(site_count))
    raising = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    lowering = raising.T
    terms = []
    for left in range(site_count):
        right = (left + 1) % site_count
        terms.append(
            QuantumLatticeTerm(
                (
                    LocalOperatorPlan(spaces[left], f"raise-{left}", raising, (2,)),
                    LocalOperatorPlan(spaces[right], f"lower-{right}", lowering, (-2,)),
                ),
                coefficient=1.0,
                add_adjoint=True,
                label=f"exchange-{left}-{right}",
            )
        )
    prepared = prepare_quantum_lattice(
        QuantumLatticeSpecification(spaces, tuple(terms)), _compiler_resources()
    )
    basis = FixedSpinProjectionBasis(
        tuple(space.site_id for space in spaces),
        (1,) * site_count,
        2 - site_count,
        resources=_basis_resources(),
    )
    return prepared, basis


def _matrix(operator):
    identity = jnp.eye(operator.source.size, dtype=jnp.complex128)
    return jnp.stack(tuple(operator.mv(column) for column in identity), axis=1)


def test_translation_character_sectors_project_three_site_ring_exactly():
    prepared, direct = _ring_exchange(3)
    generator = _translation_generator(direct.site_ids, direct.site_dimensions)
    eigenvalues = []
    for momentum in range(3):
        character = np.exp(2.0j * np.pi * momentum / 3.0)
        basis = _prepare_character_basis(direct, generator, character)
        assert basis.dimension == 1
        operator = prepare_quantum_orbit_sector_operator(
            prepared, basis, _operator_resources()
        )
        value = complex(np.asarray(operator.mv(jnp.ones((1,), dtype=jnp.complex128)))[0])
        eigenvalues.append(value)
        assert bool(operator.evidence.accepted)
        assert float(operator.evidence.maximum_invariance_residual) < 1e-12
        compiled = eqx.filter_jit(operator.mv)(jnp.asarray((1.0 + 0.0j,)))
        np.testing.assert_allclose(compiled, (value,), atol=1e-12)
    np.testing.assert_allclose(
        sorted(np.real(eigenvalues)), (-1.0, -1.0, 2.0), atol=1e-12
    )
    np.testing.assert_allclose(np.imag(eigenvalues), 0.0, atol=1e-12)


def test_two_site_exchange_even_and_odd_sectors_match_full_projection():
    prepared, direct = _ring_exchange(2)
    generator = _translation_generator(direct.site_ids, direct.site_dimensions)
    even = _prepare_character_basis(direct, generator, 1.0)
    odd = _prepare_character_basis(direct, generator, -1.0)
    even_operator = prepare_quantum_orbit_sector_operator(
        prepared, even, _operator_resources()
    )
    odd_operator = prepare_quantum_orbit_sector_operator(
        prepared, odd, _operator_resources()
    )
    np.testing.assert_allclose(_matrix(even_operator), ((2.0,),), atol=1e-12)
    np.testing.assert_allclose(_matrix(odd_operator), ((-2.0,),), atol=1e-12)
    assert not bool(even.contains((1, 0)))


def test_stabilizer_incompatible_character_is_rejected_as_empty():
    direct = FixedSpinProjectionBasis(
        ("a", "b", "c"),
        (1, 1, 1),
        -3,
        resources=_basis_resources(),
    )
    generator = _translation_generator(direct.site_ids, direct.site_dimensions)
    action = prepare_finite_group_action(
        FiniteGroupActionPlan(direct, (generator,), _orbit_resources()),
        CharacterSectorPlan("nontrivial", {"translation": np.exp(2.0j * np.pi / 3.0)}),
    )
    with pytest.raises(ValueError, match="empty"):
        prepare_orbit_sector_basis(action)


def test_fermion_site_permutation_retains_car_parity():
    order = FermionModeOrder(("a", "b"))
    direct = FixedCardinalityFermionBasis(order, 2, resources=_basis_resources())
    swap = MonomialConfigurationGenerator(
        "swap",
        direct.site_ids,
        direct.site_dimensions,
        (1, 0),
        order=2,
        fermionic_sites=(0, 1),
    )
    coordinate, phase = swap.apply((1, 1))
    np.testing.assert_array_equal(coordinate, (1, 1))
    np.testing.assert_allclose(phase, -1.0)
    compatible = _prepare_character_basis(direct, swap, -1.0)
    assert compatible.dimension == 1
    with pytest.raises(ValueError, match="empty"):
        _prepare_character_basis(direct, swap, 1.0)


def test_noninvariant_action_is_rejected_before_reduced_execution():
    spaces = (LocalSpacePlan.spin("a", 1), LocalSpacePlan.spin("b", 1))
    sz = np.diag((-1.0, 1.0))
    prepared = prepare_quantum_lattice(
        QuantumLatticeSpecification(
            spaces,
            (
                QuantumLatticeTerm(
                    (LocalOperatorPlan(spaces[0], "field-a", sz, (0,)),),
                    coefficient=1.0,
                    label="asymmetric-field",
                ),
            ),
        ),
        _compiler_resources(),
    )
    direct = FixedSpinProjectionBasis(("a", "b"), (1, 1), 0, resources=_basis_resources())
    swap = _translation_generator(direct.site_ids, direct.site_dimensions)
    basis = _prepare_character_basis(direct, swap, 1.0)
    with pytest.raises(ValueError, match="not invariant"):
        prepare_quantum_orbit_sector_operator(prepared, basis, _operator_resources())


def test_orbit_basis_archive_round_trip_preserves_projection(tmp_path):
    direct = FixedSpinProjectionBasis(("a", "b"), (1, 1), 0, resources=_basis_resources())
    swap = _translation_generator(direct.site_ids, direct.site_dimensions)
    basis = _prepare_character_basis(direct, swap, -1.0)
    provenance = ArrayArtifactProvenance(
        "phydrax-test-producer",
        ("source:independent-fixture",),
        ("profile:orbit-sector",),
        ("unit:explicit",),
    )
    path = tmp_path / "orbit-basis.pxa"
    written = write_quantum_lattice_artifact_archive(path, basis, provenance)
    restored, reopened = read_quantum_lattice_artifact_archive(path, basis, provenance)
    assert reopened.artifact_id == written.artifact_id
    assert restored.basis_id == basis.basis_id
    np.testing.assert_array_equal(restored.raw_to_orbit, basis.raw_to_orbit)
    np.testing.assert_allclose(
        restored.embedding_coefficients, basis.embedding_coefficients
    )
