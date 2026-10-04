from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.atomistic import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
    AtomisticStatus,
    energy_and_forces,
)
from phydrax.atomistic._graph import prepare_atomistic_graph_topology
from phydrax.discretization import ParticleImageCapacity
from phydrax.nn.atomistic import NequIPPotential
from phydrax.nn.operator.layers import o3_gated_activation
from phydrax.sparse import StreamedRelationPlan
from phydrax.units import ANGSTROM, ELECTRONVOLT


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)


def _execution(maximum_neighbors: Any = 3) -> Any:
    return AtomisticGraphExecutionPlan(
        maximum_neighbors,
        maximum_dense_atoms=4,
    )


def _model(*, interaction_count: Any = 2) -> Any:
    return NequIPPotential(
        SCALE,
        cutoff=2.5,
        feature_count=3,
        interaction_count=interaction_count,
        radial_basis_count=4,
        key=jr.key(27),
    )


def _structure(positions: Any = None) -> Any:
    if positions is None:
        positions = [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 0.8, 0.2]]
    # ty: ignore[invalid-argument-type]
    return AtomicStructure([8, 1, 1], positions, [15.999, 1.008, 1.008], SCALE)


def test_nequip_scenario_1() -> None:
    model = _model()
    structure = _structure()
    reference = energy_and_forces(model, structure, _execution())
    rotation = jnp.asarray([[0.36, -0.48, 0.80], [0.80, 0.60, 0.00], [-0.48, 0.64, 0.60]])
    transformed = AtomicStructure(
        structure.atomic_numbers,
        structure.positions @ rotation.T + jnp.asarray([2.0, -3.0, 1.0]),
        structure.masses,
        SCALE,
    )
    observed = energy_and_forces(model, transformed, _execution())
    np.testing.assert_allclose(observed.energy, reference.energy, rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(
        observed.forces[0], reference.forces[0] @ rotation.T, rtol=3e-9, atol=3e-9
    )
    assert observed.provenance.method_id.endswith("nequip-energy")
    assert observed.provenance.conservative_forces
    assert observed.provenance.frozen_candidate_topology
    model = _model(interaction_count=1)
    batch = AtomisticBatch.from_structure(_structure())
    prediction = energy_and_forces(model, batch, _execution())
    step = 1e-5
    direction = jnp.zeros_like(batch.positions).at[0, 1, 2].set(1.0)
    plus = model.energy(
        batch, _execution(), positions=batch.positions + step * direction
    )[0]
    minus = model.energy(
        batch, _execution(), positions=batch.positions - step * direction
    )[0]
    finite_difference = -(plus - minus) / (2.0 * step)
    np.testing.assert_allclose(
        finite_difference, prediction.forces[0, 1, 2], rtol=3e-4, atol=3e-5
    )
    np.testing.assert_allclose(prediction.net_force, 0.0, atol=3e-9)
    np.testing.assert_allclose(prediction.net_torque, 0.0, atol=3e-9)
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
    np.testing.assert_allclose(observed.energy, reference.energy, rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(
        observed.forces[0], reference.forces[0][permutation], rtol=3e-9, atol=3e-9
    )


def test_three_atom_energy_is_continuous_when_one_edge_crosses_cutoff() -> None:
    model = _model(interaction_count=1)

    def energy(distance: Any) -> Any:
        structure = AtomicStructure(
            # ty: ignore[invalid-argument-type]
            [1, 6, 8],
            # ty: ignore[invalid-argument-type]
            [[0.0, 0.0, 0.0], [0.7, 0.2, 0.0], [distance, 0.0, 0.0]],
            # ty: ignore[invalid-argument-type]
            [1.0, 12.0, 16.0],
            SCALE,
        )
        return model(structure, _execution())

    step = 1e-7
    below = energy(2.5 - step)
    at = energy(2.5)
    above = energy(2.5 + step)
    assert abs(float(below - at)) < 1e-5
    assert abs(float(above - at)) < 1e-5


def test_nequip_scenario_2() -> None:
    model = _model()
    # ty: ignore[invalid-argument-type]
    hydrogen = AtomicStructure([1], [[0.0, 0.0, 0.0]], [1.0], SCALE)
    water = _structure()
    batch = AtomisticBatch.from_structures((hydrogen, water), atom_capacity=4)
    batched = energy_and_forces(model, batch, _execution())
    np.testing.assert_allclose(
        batched.energy[0], model(hydrogen, _execution()), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        batched.energy[1], model(water, _execution()), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(batched.atom_energy[~batch.atom_mask], 0.0, atol=0.0)
    np.testing.assert_allclose(batched.forces[~batch.atom_mask], 0.0, atol=0.0)

    overflow_model = _model()
    overflow = energy_and_forces(overflow_model, water, _execution(0))
    assert not bool(overflow.valid[0])
    assert int(overflow.status[0]) == int(AtomisticStatus.NEIGHBOR_OVERFLOW)
    assert bool(jnp.isnan(overflow.energy[0]))
    with pytest.raises(Exception, match="overflow"):
        overflow_model(water, _execution(0))
    model = _model()
    reference = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 8],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 16.0],
        SCALE,
    )
    padded = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 8, 0, 0],
        # ty: ignore[invalid-argument-type]
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.1, 0.0],
            [np.nan, np.nan, np.nan],
            [np.inf, -np.inf, np.inf],
        ],
        # ty: ignore[invalid-argument-type]
        [1.0, 16.0, 0.0, 0.0],
        SCALE,
        # ty: ignore[invalid-argument-type]
        active_mask=[True, True, False, False],
    )
    observed = energy_and_forces(model, padded, _execution())
    expected = energy_and_forces(model, reference, _execution())
    assert bool(observed.valid[0])
    assert bool(jnp.all(jnp.isfinite(observed.energy)))
    assert bool(jnp.all(jnp.isfinite(observed.forces)))
    assert bool(jnp.all(jnp.isfinite(observed.net_force)))
    assert bool(jnp.all(jnp.isfinite(observed.net_torque)))
    np.testing.assert_allclose(observed.energy, expected.energy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        observed.forces[0, :2], expected.forces[0], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(observed.forces[0, 2:], 0.0, atol=0.0)
    model = _model(interaction_count=1)
    interaction = model.interactions[0]
    plan = interaction.tensor_product.plan
    assert plan.path_count > 6
    assert plan.parameter_count > plan.path_count
    assert interaction.radial_out.out_size == plan.parameter_count
    assert interaction.radial_out.weight.shape[0] == plan.parameter_count
    assert model.configuration.maximum_degree == 2


def test_jit_position_vjp_and_second_parameter_derivative_are_finite() -> None:
    model = _model(interaction_count=1)
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
    assert pullback(jnp.ones_like(energy))[0].shape == batch.positions.shape

    def embedding_energy(embedding: Any) -> Any:
        candidate = eqx.tree_at(lambda value: value.embedding, model, embedding)
        return jnp.sum(candidate.energy(batch, _execution()))

    gradient = jax.grad(embedding_energy)
    first = gradient(model.embedding)
    second = jax.jvp(gradient, (model.embedding,), (jnp.ones_like(model.embedding),))[1]
    assert bool(jnp.all(jnp.isfinite(first)))
    assert bool(jnp.all(jnp.isfinite(second)))


def test_periodic_resource_refusals_and_tensor_product_resource_overflow() -> None:
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
    # Image routes are a charged resource; a direct call needs a host topology.
    with pytest.raises(ValueError, match="image_capacity"):
        energy_and_forces(_model(), periodic, _execution())
    with pytest.raises(ValueError, match="require a topology"):
        _model()(periodic, _execution())
    with pytest.raises(ValueError, match="parameters"):
        NequIPPotential(
            SCALE,
            cutoff=2.5,
            feature_count=3,
            interaction_count=1,
            radial_basis_count=4,
            maximum_tensor_product_parameters=1,
        )


def _dense_nequip_energy(model: Any, positions: Any, numbers: Any, active: Any) -> Any:
    """Independent all-pairs NequIP energy: full edge messages and receiver sums."""
    count = positions.shape[0]
    send, receive = np.nonzero(~np.eye(count, dtype=bool))
    displacement = positions[receive] - positions[send]
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    live = active[send] & active[receive] & (distance < model.configuration.cutoff)
    safe = jnp.where(live, distance, 1.0)
    radial, envelope = model._radial_basis(safe)
    edge_features = model._edge_features(displacement / safe[:, None])
    representation = model.configuration.hidden_representation
    mask = active.astype(radial.dtype)
    values = jnp.zeros((count, representation.packed_size), dtype=radial.dtype)
    values = values.at[:, : model.configuration.feature_count].set(
        model.embedding[numbers]
    )
    values = values * mask[:, None]
    for interaction in model.interactions:
        weights = interaction.radial_out(interaction.radial_in(radial))
        weights = weights * envelope[:, None]
        messages = interaction.tensor_product(values[send], edge_features, weights)
        messages = jnp.where(live[:, None], messages, 0.0)
        connected = interaction.self_connection(values, numbers) + jax.ops.segment_sum(
            messages, receive, count
        )
        values = o3_gated_activation(connected, representation) * mask[:, None]
    scalars = representation.split(values).scalars
    return jnp.sum(model.readout_energy(model.readout_hidden(scalars)) * mask)


def test_streamed_nequip_matches_dense_energy_forces_and_force_loss_gradient() -> None:
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
    position = batch.positions[0]
    model = NequIPPotential(
        SCALE,
        cutoff=2.5,
        feature_count=3,
        interaction_count=2,
        radial_basis_count=4,
        key=jr.key(73),
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
            return _dense_nequip_energy(model, position, numbers, active)

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


def test_periodic_nequip_is_extensive_with_intensive_stress() -> None:
    cell = np.asarray([[2.8, 0.0, 0.0], [0.3, 2.9, 0.0], [0.2, 0.1, 3.0]])
    positions = np.asarray([[0.0, 0.0, 0.0], [1.1, 0.3, 0.2]])
    execution = AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=2048,
            maximum_degree=64,
            maximum_images=125,
        ),
    )

    def structure(positions: Any, cell: Any, numbers: Any) -> Any:
        return AtomicStructure(
            numbers,
            positions,
            np.where(np.asarray(numbers) == 8, 15.999, 1.008),
            SCALE,
            cell=cell,
            # ty: ignore[invalid-argument-type]
            periodic_axes=[True, True, True],
        )

    model = _model()
    unit = energy_and_forces(
        model, structure(positions, cell, [1, 8]), execution, compute_stress=True
    )
    doubled = energy_and_forces(
        model,
        structure(
            np.concatenate((positions, positions + cell[0])),
            cell * np.asarray([[2.0], [1.0], [1.0]]),
            [1, 8, 1, 8],
        ),
        execution,
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
