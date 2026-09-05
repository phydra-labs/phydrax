"""Numerical bridge evidence only: these are not calibrated protein parameters."""

import hashlib
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.protein_folding._construct import ProteinConstruct
from phydrax.applications.protein_folding.potentials import (
    RotamerFreeEnergyStatus,
    RotamerFreeEnergyTerm,
    RotamerGeometryPlan,
    RotamerParameterPlan,
)
from phydrax.atomistic import (
    AtomisticPotentialProgram,
    AtomisticSystemPlan,
    AtomisticUnitSystem,
)
from phydrax.discretization import DenseParticleNeighborhoodPlan
from phydrax.qualification import ReferenceArtifactManifest


def _source():
    payload = b"Deterministic analytical Gaussian rotamer numerical fixture, not biological calibration."
    return ReferenceArtifactManifest(
        "analytical-rotamer-numerics",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="test-author-owned",
        commercial_use_permitted=False,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted-test-fixture",
        nondimensionalization={"reduced_length": 1.0, "reduced_energy": 1.0},
        uncertainty=None,
        lineage_ids=("analytical-model-not-protein-calibration",),
    )


def _fixture(
    *,
    loop=False,
    method="exact",
    tolerance=1e-10,
    amplitude=0.2,
    maximum_steps=100,
    permutation=None,
    unary_gap=None,
):
    units = AtomisticUnitSystem.reduced()
    ids = np.arange(9, dtype=np.int64) * 17 + 100
    positions = np.asarray(
        [
            [1.3 * i + dx, dy, 0.2 * i]
            for i in range(3)
            for dx, dy in ((0, 0), (1, 0), (0, 1))
        ]
    )
    cards = (2, 3, 2)
    sites = tuple(
        np.asarray([[0.1 * k, -0.2 * k, 0.4 * (-1) ** k] for k in range(card)])
        for card in cards
    )
    source = _source()
    geometry = RotamerGeometryPlan(
        ProteinConstruct(("A",), ("AAA",)),
        ids.reshape((3, 3)),
        sites,
        source,
        units=units,
    )
    pairs = ((0, 1), (1, 2), (0, 2)) if loop else ((0, 1), (1, 2))
    parameters = RotamerParameterPlan(
        units,
        1.0,
        tuple(
            np.linspace(-0.1, 0.2, card)
            if unary_gap is None
            else np.arange(card) * unary_gap
            for card in cards
        ),
        pairs,
        tuple(
            amplitude
            * np.asarray(
                [[(-1) ** (i + j) for j in range(cards[b])] for i in range(cards[a])]
            )
            for a, b in pairs
        ),
        tuple(np.full((cards[a], cards[b]), 1.3) for a, b in pairs),
        source,
    )
    term = RotamerFreeEnergyTerm(
        geometry,
        parameters,
        ids[::3],
        [0.2, 0.3, 0.5],
        sampling_temperature=1.0,
        inference_method=method,
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
        maximum_steps=maximum_steps,
    )
    if permutation is not None:
        ids = ids[permutation]
        positions = positions[permutation]
    system = AtomisticSystemPlan(ids, np.full(9, 6), np.ones(9), units).prepare()
    return term, system, jnp.asarray(positions)


def test_heterogeneous_exact_energy_matches_independent_enumeration():
    term, system, positions = _fixture(loop=True)
    prepared = term.prepare(system)
    tables, valid = prepared.log_factors(positions)
    energies = []
    for states in itertools.product(range(2), range(3), range(2)):
        score = sum(float(tables[i][0, states[i]]) for i in range(3))
        score += sum(
            float(tables[3 + k][0, states[i], states[j]])
            for k, (i, j) in enumerate(term.parameters.pair_indices)
        )
        energies.append(-score)
    expected = -np.log(np.exp(-np.asarray(energies)).sum())
    result = eqx.filter_jit(prepared.evaluate)(positions)
    assert bool(valid & result.successful)
    np.testing.assert_allclose(result.energy, expected, atol=1e-12)
    np.testing.assert_allclose(result.atom_energy.sum(), result.energy, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(result.atom_energy)[[0, 3, 6]],
        np.asarray(result.energy) * [0.2, 0.3, 0.5],
        atol=1e-12,
    )
    for marginal in np.split(np.asarray(result.variable_probabilities), [2, 5]):
        np.testing.assert_allclose(marginal.sum(), 1.0, atol=1e-12)


def test_tree_implicit_jitted_energy_marginals_and_forces_match_exact():
    exact_term, system, positions = _fixture()
    bethe_term, _, _ = _fixture(method="bethe")
    exact = exact_term.prepare(system)
    bethe = bethe_term.prepare(system)
    exact_value, exact_gradient = jax.jit(
        jax.value_and_grad(lambda x: exact.evaluate(x).energy)
    )(positions)
    bethe_value, bethe_gradient = jax.jit(
        jax.value_and_grad(lambda x: bethe.evaluate(x).energy)
    )(positions)
    result = eqx.filter_jit(bethe.evaluate)(positions)
    assert bool(result.successful & result.derivative_qualified)
    assert result.inference.inference.log_normalizer_exact
    np.testing.assert_allclose(bethe_value, exact_value, atol=2e-9)
    np.testing.assert_allclose(bethe_gradient, exact_gradient, atol=2e-8)
    np.testing.assert_allclose(
        result.variable_probabilities,
        exact.evaluate(positions).variable_probabilities,
        atol=2e-9,
    )


def test_native_force_path_is_scalar_gradient_and_rigid_covariant():
    term, system, positions = _fixture(loop=True, method="bethe")
    program = AtomisticPotentialProgram((term,)).prepare(system)
    relation = (
        DenseParticleNeighborhoodPlan(system.capacity * (system.capacity - 1) // 2)
        .prepare(system.particles)
        .build(positions)
    )
    result = jax.jit(lambda q: program.evaluate(q, relation))(positions)
    prepared = program.terms[0]
    direction = jnp.arange(27, dtype=float).reshape((9, 3)) / 27 - 0.5
    h = 1e-5
    finite_difference = (
        prepared.evaluate(positions + h * direction).energy
        - prepared.evaluate(positions - h * direction).energy
    ) / (2 * h)
    assert bool(result.successful)
    np.testing.assert_allclose(
        -jnp.sum(result.forces * direction), finite_difference, rtol=2e-5, atol=2e-8
    )
    rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = positions @ rotation.T + jnp.asarray([2.0, -1.0, 0.7])
    moved = program.evaluate(transformed, relation)
    np.testing.assert_allclose(moved.energy, result.energy, atol=1e-9)
    np.testing.assert_allclose(moved.forces, result.forces @ rotation.T, atol=2e-8)
    np.testing.assert_allclose(result.forces.sum(axis=0), 0.0, atol=2e-8)
    np.testing.assert_allclose(
        jnp.cross(positions, result.forces).sum(axis=0), 0.0, atol=2e-8
    )


def test_loopy_approximation_and_force_error_at_tighter_tolerance():
    exact_term, system, positions = _fixture(loop=True, amplitude=0.1)
    tight_term, _, _ = _fixture(loop=True, method="bethe", amplitude=0.1, tolerance=1e-12)
    loose_term, _, _ = _fixture(loop=True, method="bethe", amplitude=0.1, tolerance=1e-5)
    exact, tight, loose = (
        term.prepare(system) for term in (exact_term, tight_term, loose_term)
    )
    results = [
        jax.jit(jax.value_and_grad(lambda q: model.evaluate(q).energy))(positions)
        for model in (exact, tight, loose)
    ]
    for model in (tight, loose):
        for evaluation in (
            model.evaluate(positions),
            eqx.filter_jit(model.evaluate)(positions),
        ):
            assert bool(evaluation.successful & evaluation.derivative_qualified)
            assert not evaluation.inference.inference.log_normalizer_exact
            root = evaluation.inference.nonlinear
            assert bool(root.successful)
            threshold = (
                model.plan.method.absolute_tolerance
                + model.plan.method.relative_tolerance
                * root.diagnostics.initial_residual_norm
            )
            assert float(jnp.linalg.norm(root.residual)) <= float(threshold)
    assert abs(float(results[1][0] - results[0][0])) < 2e-3
    assert float(jnp.max(jnp.abs(results[1][1] - results[0][1]))) < 2e-3
    assert float(jnp.max(jnp.abs(results[2][1] - results[1][1]))) < 1e-4
    np.testing.assert_allclose(
        tight.evaluate(positions).variable_probabilities,
        exact.evaluate(positions).variable_probabilities,
        atol=2e-3,
    )


def test_atom_reordering_preserves_identity_bound_observables():
    term, system, positions = _fixture()
    permutation = np.asarray([8, 0, 6, 2, 5, 1, 7, 3, 4])
    permuted_term, permuted_system, reordered = _fixture(permutation=permutation)
    first = term.prepare(system).evaluate(positions)
    second = permuted_term.prepare(permuted_system).evaluate(reordered)
    np.testing.assert_allclose(first.energy, second.energy, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(first.atom_energy)[permutation], second.atom_energy, atol=1e-12
    )


def test_invalid_geometry_and_unqualified_loopy_branch_fail_without_fallback():
    term, system, positions = _fixture()
    result = eqx.filter_jit(term.prepare(system).evaluate)(
        positions.at[1].set(positions[0])
    )
    assert int(result.status) == int(RotamerFreeEnergyStatus.INVALID_GEOMETRY)
    assert not bool(result.successful)
    assert bool(jnp.isnan(result.energy))
    strong, _, _ = _fixture(loop=True, method="bethe", amplitude=30.0)
    result = strong.prepare(system).evaluate(positions)
    assert int(result.status) == int(RotamerFreeEnergyStatus.UNQUALIFIED_BRANCH)
    assert not bool(result.derivative_qualified)
    assert bool(jnp.isnan(result.energy))


def test_nonconvergence_and_model_contract_refusals():
    term, system, positions = _fixture(
        loop=True, method="bethe", maximum_steps=1, tolerance=1e-14
    )
    result = eqx.filter_jit(term.prepare(system).evaluate)(positions)
    assert not bool(result.successful)
    assert int(result.status) == int(RotamerFreeEnergyStatus.INFERENCE_FAILED)
    assert not bool(result.inference.nonlinear.successful)
    assert not bool(result.derivative_qualified)
    assert bool(jnp.isnan(result.energy))
    with pytest.raises(ValueError, match="temperature"):
        RotamerFreeEnergyTerm(
            term.geometry, term.parameters, [100], [1.0], sampling_temperature=2.0
        )
    with pytest.raises(ValueError, match="sum to one"):
        RotamerFreeEnergyTerm(
            term.geometry, term.parameters, [100], [0.5], sampling_temperature=1.0
        )
    missing = AtomisticSystemPlan(
        np.arange(9) + 1000, np.full(9, 6), np.ones(9), term.parameters.units
    ).prepare()
    with pytest.raises(ValueError, match="active atom"):
        term.prepare(missing)
    with pytest.raises(PermissionError):
        RotamerGeometryPlan(
            term.geometry.construct,
            term.geometry.frame_atom_ids,
            term.geometry.local_sites,
            _source(),
            units=term.parameters.units,
            commercial_use=True,
        )
    with pytest.raises(ValueError):
        term.parameters.source.require_uncertainty()


@pytest.mark.parametrize("loop", [False, True])
def test_underflowed_finite_unary_populations_preserve_native_forces(loop):
    exact_term, system, positions = _fixture(loop=loop, unary_gap=1000.0)
    bethe_term, _, _ = _fixture(loop=loop, method="bethe", unary_gap=1000.0)
    exact = AtomisticPotentialProgram((exact_term,)).prepare(system)
    bethe = AtomisticPotentialProgram((bethe_term,)).prepare(system)
    relation = (
        DenseParticleNeighborhoodPlan(system.capacity * (system.capacity - 1) // 2)
        .prepare(system.particles)
        .build(positions)
    )
    reference = jax.jit(lambda q: exact.evaluate(q, relation))(positions)
    result = jax.jit(lambda q: bethe.evaluate(q, relation))(positions)
    beliefs = bethe.terms[0].evaluate(positions)
    assert bool(reference.successful & result.successful & beliefs.derivative_qualified)
    np.testing.assert_array_equal(
        beliefs.variable_probabilities, [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    )
    np.testing.assert_allclose(result.energy, reference.energy, atol=2e-8)
    np.testing.assert_allclose(result.forces, reference.forces, atol=2e-8)
