from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from opt_einsum import contract

from phydrax.atomistic import (
    AtomicStructure,
    atomistic_potential_revision,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    AtomisticStatus,
    energy_and_forces,
)
from phydrax.nn.atomistic import PaiNNPotential
from phydrax.nn.atomistic._painn import _PaiNNInteraction
from phydrax.units import ANGSTROM, ELECTRONVOLT


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)


def _execution(maximum_neighbors: Any = 3) -> Any:
    return AtomisticGraphExecutionPlan(
        maximum_neighbors,
        maximum_dense_atoms=4,
    )


def _model(*, precision: Any = None, seed: Any = 7) -> Any:
    return PaiNNPotential(
        SCALE,
        cutoff=2.5,
        feature_count=8,
        interaction_count=2,
        radial_basis_count=6,
        precision=precision,
        key=jr.key(seed),
    )


def _structure(positions: Any = None) -> Any:
    if positions is None:
        positions = [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 0.8, 0.2]]
    # ty: ignore[invalid-argument-type]
    return AtomicStructure([8, 1, 1], positions, [15.999, 1.008, 1.008], SCALE)


def test_painn_scenario_1() -> None:
    model = _model()
    structure = _structure()
    reference = energy_and_forces(model, structure, _execution())
    rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = AtomicStructure(
        structure.atomic_numbers,
        structure.positions @ rotation.T + jnp.asarray([3.0, -2.0, 1.0]),
        structure.masses,
        SCALE,
    )
    observed = energy_and_forces(model, transformed, _execution())
    np.testing.assert_allclose(observed.energy, reference.energy, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(
        observed.forces[0], reference.forces[0] @ rotation.T, rtol=2e-9, atol=2e-9
    )
    model = _model()
    structure = _structure()
    permutation = np.asarray([2, 0, 1])
    permuted = AtomicStructure(
        np.asarray(structure.atomic_numbers)[permutation],
        np.asarray(structure.positions)[permutation],
        np.asarray(structure.masses)[permutation],
        SCALE,
        particle_ids=np.asarray(structure.particle_ids)[permutation],
    )
    reference = energy_and_forces(model, structure, _execution())
    observed = energy_and_forces(model, permuted, _execution())
    np.testing.assert_allclose(observed.energy, reference.energy, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(
        observed.forces[0], reference.forces[0][permutation], rtol=2e-9, atol=2e-9
    )
    model = _model()
    batch = AtomisticBatch.from_structure(_structure())
    prediction = energy_and_forces(model, batch, _execution())
    step = 1e-5
    direction = jnp.zeros_like(batch.positions).at[0, 1, 0].set(1.0)
    plus = model.energy(
        batch, _execution(), positions=batch.positions + step * direction
    )[0]
    minus = model.energy(
        batch, _execution(), positions=batch.positions - step * direction
    )[0]
    finite_difference = -(plus - minus) / (2.0 * step)
    np.testing.assert_allclose(
        finite_difference, prediction.forces[0, 1, 0], rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(prediction.net_force, 0.0, atol=2e-9)
    np.testing.assert_allclose(prediction.net_torque, 0.0, atol=2e-9)
    assert prediction.provenance.conservative_forces
    assert prediction.provenance.frozen_candidate_topology
    assert not prediction.provenance.stress_available


def test_painn_scenario_2() -> None:
    model = _model()
    structure = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        SCALE,
    )
    prediction = energy_and_forces(model, structure, _execution())
    np.testing.assert_allclose(prediction.forces, 0.0, atol=1e-10)
    radial, envelope = model._radial_basis(jnp.asarray([2.5]))
    biased_filter = model.interactions[0].filter_out(
        model.interactions[0].filter_in(radial)
    )
    assert float(jnp.linalg.norm(biased_filter)) > 0.0
    np.testing.assert_allclose(envelope, 0.0, atol=0.0)
    np.testing.assert_allclose(biased_filter * envelope, 0.0, atol=0.0)
    below = model(
        AtomicStructure(
            # ty: ignore[invalid-argument-type]
            [1, 1],
            # ty: ignore[invalid-argument-type]
            [[0.0, 0.0, 0.0], [2.5 - 1e-5, 0.0, 0.0]],
            # ty: ignore[invalid-argument-type]
            [1.0, 1.0],
            SCALE,
        ),
        _execution(),
    )
    at = model(structure, _execution())
    assert abs(float(below - at)) < 1e-7
    model = _model()
    # ty: ignore[invalid-argument-type]
    one = AtomicStructure([1], [[0.0, 0.0, 0.0]], [1.0], SCALE)
    one_prediction = energy_and_forces(model, one, _execution())
    np.testing.assert_allclose(one_prediction.forces, 0.0, atol=0.0)
    coincident = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 8],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 16.0],
        SCALE,
    )
    assert bool(
        jnp.all(jnp.isfinite(energy_and_forces(model, coincident, _execution()).forces))
    )
    disconnected = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 8],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 16.0],
        SCALE,
    )
    separate = model(one, _execution()) + model(
        # ty: ignore[invalid-argument-type]
        AtomicStructure([8], [[10.0, 0.0, 0.0]], [16.0], SCALE),
        _execution(),
    )
    np.testing.assert_allclose(
        model(disconnected, _execution()), separate, rtol=1e-12, atol=1e-12
    )
    precision = AtomisticPrecisionPolicy(
        coordinate_dtype="float32",
        compute_dtype="float32",
        reduction_dtype="float32",
        output_dtype="float32",
    )
    structure = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1],
        np.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        SCALE,
        coordinate_dtype="float32",
    )
    prediction = energy_and_forces(_model(precision=precision), structure, _execution())
    assert prediction.energy.dtype == jnp.float32
    assert prediction.forces.dtype == jnp.float32


def test_jit_vjp_and_second_order_parameter_derivative() -> None:
    model = _model()
    batch = AtomisticBatch.from_structure(_structure())
    compiled = jax.jit(
        lambda position: model.energy(batch, _execution(), positions=position)
    )
    energy = compiled(batch.positions)
    assert energy.shape == (1,)
    _, pullback = jax.vjp(
        lambda position: model.energy(batch, _execution(), positions=position),
        batch.positions,
    )
    position_gradient = pullback(jnp.ones((1,), dtype=energy.dtype))[0]
    assert position_gradient.shape == batch.positions.shape

    def embedding_energy(embedding: Any) -> Any:
        candidate = eqx.tree_at(lambda value: value.embedding, model, embedding)
        return jnp.sum(candidate.energy(batch, _execution()))

    gradient = jax.grad(embedding_energy)
    first = gradient(model.embedding)
    second = jax.jvp(gradient, (model.embedding,), (jnp.ones_like(model.embedding),))[1]
    assert bool(jnp.all(jnp.isfinite(first)))
    assert bool(jnp.all(jnp.isfinite(second)))


def test_painn_scenario_3() -> None:
    periodic = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0],
        SCALE,
        cell=np.eye(3),
        # ty: ignore[invalid-argument-type]
        periodic_axes=[True, False, False],
    )
    assert periodic.has_periodic_metadata
    with pytest.raises(ValueError, match="nonperiodic"):
        energy_and_forces(_model(), periodic, _execution())
    structure = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        SCALE,
    )
    model = _model()
    prediction = energy_and_forces(model, structure, _execution(0))
    assert not bool(prediction.valid[0])
    assert int(prediction.status[0]) == int(AtomisticStatus.NEIGHBOR_OVERFLOW)
    assert bool(jnp.isnan(prediction.energy[0]))
    with pytest.raises(Exception, match="overflow"):
        model(structure, _execution(0))
    interaction = _PaiNNInteraction(4, 3, jr.key(21))
    scalar = jr.normal(jr.key(22), (3, 4))
    vector = jr.normal(jr.key(23), (3, 3, 4))
    observed_scalar, observed_vector = interaction.atomwise_update(scalar, vector)
    vector_u = interaction.vector_u(vector)
    vector_v = interaction.vector_v(vector)
    squared_norm = contract("ndf,ndf->nf", vector_v, vector_v)
    tiny = jnp.asarray(jnp.finfo(vector.dtype).tiny, dtype=vector.dtype)
    vector_norm = jnp.where(
        squared_norm > 0.0,
        jnp.sqrt(jnp.maximum(squared_norm, tiny)),
        0.0,
    )
    invariant = contract("ndf,ndf->nf", vector_u, vector_v)
    heads = interaction.update_out(
        interaction.update_in(jnp.concatenate((scalar, vector_norm), axis=-1))
    )
    scalar_scalar, scalar_vector, vector_vector = jnp.split(heads, 3, axis=-1)
    expected_scalar = scalar + scalar_scalar + scalar_vector * invariant
    expected_vector = vector + vector_vector[:, None, :] * vector_u
    np.testing.assert_allclose(observed_scalar, expected_scalar, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(observed_vector, expected_vector, rtol=1e-12, atol=1e-12)


def test_painn_scenario_4() -> None:
    structure = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1, 0],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [np.nan, np.nan, np.nan]],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 0.0],
        SCALE,
        # ty: ignore[invalid-argument-type]
        active_mask=[True, True, False],
    )
    prediction = energy_and_forces(_model(), structure, _execution())
    assert bool(prediction.valid[0])
    assert bool(jnp.all(jnp.isfinite(prediction.energy)))
    assert bool(jnp.all(jnp.isfinite(prediction.forces)))
    np.testing.assert_allclose(prediction.forces[0, 2], 0.0)
    first = _model(seed=31)
    second = _model(seed=32)
    first_revision = atomistic_potential_revision(first)
    second_revision = atomistic_potential_revision(second)
    assert first.architecture_id == second.architecture_id
    assert first_revision.semantic_id == second_revision.semantic_id
    assert first_revision.revision_id != second_revision.revision_id
    assert atomistic_potential_revision(_model(seed=31)).revision_id == (
        first_revision.revision_id
    )
    provenance = energy_and_forces(first, _structure(), _execution()).provenance
    assert provenance.architecture_id == first.architecture_id
    assert provenance.potential_revision_id == first_revision.revision_id
    precision = AtomisticPrecisionPolicy(
        coordinate_dtype="float64",
        compute_dtype="float32",
        reduction_dtype="float64",
        output_dtype="float32",
    )
    prediction = energy_and_forces(
        _model(precision=precision), _structure(), _execution()
    )
    assert prediction.energy.dtype == jnp.float32
    assert prediction.forces.dtype == jnp.float32
    assert prediction.net_force.dtype == jnp.float32
    assert prediction.net_torque.dtype == jnp.float32


def test_painn_scenario_5() -> None:
    original = _model(seed=51)
    updated = eqx.tree_at(
        lambda potential: potential.embedding,
        original,
        original.embedding + 0.125,
    )
    original_revision = atomistic_potential_revision(original)
    updated_revision = atomistic_potential_revision(updated)
    assert updated_revision.revision_id != original_revision.revision_id
    prediction = energy_and_forces(updated, _structure(), _execution())
    assert prediction.provenance.potential_revision_id == updated_revision.revision_id
    fixed_update = eqx.tree_at(
        lambda potential: potential.configuration.radial_frequencies,
        original,
        original.configuration.radial_frequencies * 2.0,
    )
    assert atomistic_potential_revision(fixed_update).revision_id == (
        original_revision.revision_id
    )
    model = _model()
    structure = _structure()
    with pytest.raises(TypeError):
        jax.jit(lambda potential: energy_and_forces(potential, structure, _execution()))(
            model
        )
    precision = AtomisticPrecisionPolicy(
        coordinate_dtype="float64",
        compute_dtype="float32",
        reduction_dtype="float64",
        output_dtype="float32",
    )
    structure = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1],
        # ty: ignore[invalid-argument-type]
        [[1e30, -1e30, 1e30]],
        # ty: ignore[invalid-argument-type]
        [1.0],
        SCALE,
        coordinate_dtype="float64",
    )
    prediction = energy_and_forces(_model(precision=precision), structure, _execution())
    np.testing.assert_allclose(prediction.forces, 0.0, atol=0.0)
    np.testing.assert_allclose(prediction.net_torque, 0.0, atol=0.0)
    assert bool(jnp.all(jnp.isfinite(prediction.net_torque)))
    assert prediction.net_torque.dtype == jnp.float32
