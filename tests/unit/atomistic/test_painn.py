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
    atomistic_energy_derivatives,
    atomistic_potential_revision,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    AtomisticStatus,
    energy_and_forces,
)
from phydrax.atomistic._graph import prepare_atomistic_graph_topology
from phydrax.discretization import ParticleImageCapacity
from phydrax.nn.atomistic import PaiNNPotential
from phydrax.nn.atomistic._painn import _PaiNNInteraction
from phydrax.sparse import StreamedRelationPlan
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
    # The numerical helper admits the same batch precision contract as the
    # host path: a float32 batch never silently feeds a float64 model.
    batch = AtomisticBatch.from_structure(structure)
    topology = prepare_atomistic_graph_topology(batch, _execution(), cutoff=2.5)
    with pytest.raises(ValueError, match="precision contract"):
        atomistic_energy_derivatives(
            _model(), batch, _execution(), batch.positions, topology=topology
        )


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
    # Image routes are a charged resource; a direct call needs a host topology.
    with pytest.raises(ValueError, match="image_capacity"):
        energy_and_forces(_model(), periodic, _execution())
    with pytest.raises(ValueError, match="require a topology"):
        _model()(periodic, _execution())
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


def _dense_painn_energy(model: Any, positions: Any, numbers: Any, active: Any) -> Any:
    """Independent all-pairs PaiNN energy: full edge messages and receiver sums."""
    count = positions.shape[0]
    send, receive = np.nonzero(~np.eye(count, dtype=bool))
    displacement = positions[receive] - positions[send]
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    live = active[send] & active[receive] & (distance < model.configuration.cutoff)
    safe = jnp.where(live, distance, 1.0)
    direction = displacement / safe[:, None]
    radial, envelope = model._radial_basis(safe)
    scalar = model.embedding[numbers] * active[:, None]
    vector = jnp.zeros((count, 3, model.configuration.feature_count))
    for interaction in model.interactions:
        filtered = interaction.filter_out(interaction.filter_in(radial)) * envelope
        message = interaction.message_out(interaction.message_in(scalar[send])) * filtered
        message = jnp.where(live[:, None], message, 0.0)
        scalar_part, vector_part, direction_part = jnp.split(message, 3, axis=-1)
        vector_message = (
            vector_part[:, None, :] * vector[send]
            + direction_part[:, None, :] * direction[:, :, None]
        )
        scalar, vector = interaction.atomwise_update(
            scalar + jax.ops.segment_sum(scalar_part, receive, count),
            vector + jax.ops.segment_sum(vector_message, receive, count),
        )
        scalar = scalar * active[:, None]
        vector = vector * active[:, None, None]
    atom_energy = model.readout_energy(model.readout_hidden(scalar)) * active
    return jnp.sum(atom_energy)


def test_streamed_painn_matches_dense_energy_forces_and_force_loss_gradient() -> None:
    structure = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [8, 1, 1, 1, 0],
        # ty: ignore[invalid-argument-type]
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [-0.2, 0.8, 0.2],
            [2.6, 0.2, 0.1],
            [0.3, 0.3, 0.3],
        ],
        # ty: ignore[invalid-argument-type]
        [15.999, 1.008, 1.008, 1.008, 0.0],
        SCALE,
        # ty: ignore[invalid-argument-type]
        active_mask=[True, True, True, True, False],
    )
    batch = AtomisticBatch.from_structure(structure)
    numbers = batch.atomic_numbers[0]
    active = batch.atom_mask[0]
    model = PaiNNPotential(
        SCALE,
        cutoff=2.5,
        feature_count=8,
        interaction_count=2,
        radial_basis_count=6,
        key=jr.key(71),
    )
    # A one-receiver, two-event tile fragments every multi-neighbor receiver.
    tiny = StreamedRelationPlan(receiver_tile=1, edge_tile=2)
    for plan in (None, tiny):
        execution = AtomisticGraphExecutionPlan(4, maximum_dense_atoms=5, streamed=plan)
        topology = prepare_atomistic_graph_topology(batch, execution, cutoff=2.5)

        def streamed(model: Any, position: Any) -> Any:
            return model.energy(
                batch, execution, positions=position[None], topology=topology
            )[0]

        def dense(model: Any, position: Any) -> Any:
            return _dense_painn_energy(model, position, numbers, active)

        position = batch.positions[0]
        np.testing.assert_allclose(
            streamed(model, position), dense(model, position), rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(
            jax.grad(streamed, argnums=1)(model, position),
            jax.grad(dense, argnums=1)(model, position),
            rtol=1e-10,
            atol=1e-10,
        )

        def force_loss(energy: Any) -> Any:
            def loss(model: Any) -> Any:
                forces = -jax.grad(energy, argnums=1)(model, position)
                return jnp.sum(forces * forces)

            return loss

        observed = eqx.filter_grad(force_loss(streamed))(model)
        expected = eqx.filter_grad(force_loss(dense))(model)
        for actual, reference in zip(
            jax.tree_util.tree_leaves(observed),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-10)
        tangent = jnp.zeros_like(position).at[1, 0].set(1.0)
        np.testing.assert_allclose(
            jax.jvp(
                lambda x: jax.grad(streamed, argnums=1)(model, x), (position,), (tangent,)
            )[1],
            jax.jvp(
                lambda x: jax.grad(dense, argnums=1)(model, x), (position,), (tangent,)
            )[1],
            rtol=1e-9,
            atol=1e-10,
        )


PERIODIC_CELL = np.asarray([[2.8, 0.0, 0.0], [0.3, 2.9, 0.0], [0.2, 0.1, 3.0]])
PERIODIC_POSITIONS = np.asarray([[0.0, 0.0, 0.0], [1.1, 0.3, 0.2]])


def _periodic_execution() -> Any:
    return AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=2048,
            maximum_degree=64,
            maximum_images=125,
        ),
    )


def _periodic_structure(positions: Any, cell: Any, numbers: Any) -> Any:
    return AtomicStructure(
        numbers,
        positions,
        np.where(np.asarray(numbers) == 8, 15.999, 1.008),
        SCALE,
        cell=cell,
        # ty: ignore[invalid-argument-type]
        periodic_axes=[True, True, True],
    )


def test_periodic_painn_is_extensive_wrap_invariant_and_has_intensive_stress() -> None:
    model = _model()
    unit = energy_and_forces(
        model,
        _periodic_structure(PERIODIC_POSITIONS, PERIODIC_CELL, [1, 8]),
        _periodic_execution(),
        compute_stress=True,
    )
    supercell = PERIODIC_CELL * np.asarray([[2.0], [1.0], [1.0]])
    doubled = energy_and_forces(
        model,
        _periodic_structure(
            np.concatenate((PERIODIC_POSITIONS, PERIODIC_POSITIONS + PERIODIC_CELL[0])),
            supercell,
            [1, 8, 1, 8],
        ),
        _periodic_execution(),
        compute_stress=True,
    )
    assert bool(unit.valid[0]) and bool(doubled.valid[0])
    # Self images and distinct-pair images enter exactly once per directed route.
    np.testing.assert_allclose(doubled.energy, 2.0 * unit.energy, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        doubled.forces[0], np.tile(unit.forces[0], (2, 1)), rtol=1e-10, atol=1e-10
    )
    assert unit.stress is not None and doubled.stress is not None
    np.testing.assert_allclose(doubled.stress, unit.stress, rtol=1e-10, atol=1e-10)
    wrapped = energy_and_forces(
        model,
        _periodic_structure(
            PERIODIC_POSITIONS
            + np.stack((np.zeros(3), PERIODIC_CELL[1] - PERIODIC_CELL[2])),
            PERIODIC_CELL,
            [1, 8],
        ),
        _periodic_execution(),
    )
    np.testing.assert_allclose(wrapped.energy, unit.energy, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(wrapped.forces, unit.forces, rtol=1e-10, atol=1e-10)
