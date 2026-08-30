import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from opt_einsum import contract

from phydrax.atomistic import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    AtomisticStatus,
    energy_and_forces,
)
from phydrax.nn.atomistic import checkpoint_atomistic_potential, PaiNNPotential
from phydrax.nn.atomistic._painn import _PaiNNInteraction


SCALE = AtomisticScaleContract("angstrom", "electronvolt")


def _model(*, maximum_neighbors=3, precision=None, seed=7):
    return PaiNNPotential(
        SCALE,
        cutoff=2.5,
        maximum_neighbors=maximum_neighbors,
        maximum_dense_atoms=4,
        feature_count=8,
        interaction_count=2,
        radial_basis_count=6,
        precision=precision,
        key=jr.key(seed),
    )


def _structure(positions=None):
    if positions is None:
        positions = [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 0.8, 0.2]]
    return AtomicStructure([8, 1, 1], positions, [15.999, 1.008, 1.008], SCALE)


def test_energy_invariant_force_equivariant_under_rigid_motion():
    model = _model()
    structure = _structure()
    reference = energy_and_forces(model, structure)
    rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = AtomicStructure(
        structure.atomic_numbers,
        structure.positions @ rotation.T + jnp.asarray([3.0, -2.0, 1.0]),
        structure.masses,
        SCALE,
    )
    observed = energy_and_forces(model, transformed)
    np.testing.assert_allclose(observed.energy, reference.energy, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(
        observed.forces[0], reference.forces[0] @ rotation.T, rtol=2e-9, atol=2e-9
    )


def test_atom_permutation_preserves_energy_and_permutes_force():
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
    reference = energy_and_forces(model, structure)
    observed = energy_and_forces(model, permuted)
    np.testing.assert_allclose(observed.energy, reference.energy, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(
        observed.forces[0], reference.forces[0][permutation], rtol=2e-9, atol=2e-9
    )


def test_conservative_force_matches_scalar_energy_finite_difference():
    model = _model()
    batch = AtomisticBatch.from_structure(_structure())
    prediction = energy_and_forces(model, batch)
    step = 1e-5
    direction = jnp.zeros_like(batch.positions).at[0, 1, 0].set(1.0)
    plus = model.energy(batch, positions=batch.positions + step * direction)[0]
    minus = model.energy(batch, positions=batch.positions - step * direction)[0]
    finite_difference = -(plus - minus) / (2.0 * step)
    np.testing.assert_allclose(
        finite_difference, prediction.forces[0, 1, 0], rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(prediction.net_force, 0.0, atol=2e-9)
    np.testing.assert_allclose(prediction.net_torque, 0.0, atol=2e-9)
    assert prediction.provenance.conservative_forces
    assert prediction.provenance.frozen_candidate_topology
    assert not prediction.provenance.stress_available


def test_smooth_cutoff_has_zero_force_at_boundary():
    model = _model()
    structure = AtomicStructure(
        [1, 1], [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], [1.0, 1.0], SCALE
    )
    prediction = energy_and_forces(model, structure)
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
            [1, 1], [[0.0, 0.0, 0.0], [2.5 - 1e-5, 0.0, 0.0]], [1.0, 1.0], SCALE
        )
    )
    at = model(structure)
    assert abs(float(below - at)) < 1e-7


def test_one_two_disconnected_and_coincident_atoms_are_well_defined():
    model = _model()
    one = AtomicStructure([1], [[0.0, 0.0, 0.0]], [1.0], SCALE)
    one_prediction = energy_and_forces(model, one)
    np.testing.assert_allclose(one_prediction.forces, 0.0, atol=0.0)
    coincident = AtomicStructure(
        [1, 8], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [1.0, 16.0], SCALE
    )
    assert bool(jnp.all(jnp.isfinite(energy_and_forces(model, coincident).forces)))
    disconnected = AtomicStructure(
        [1, 8], [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [1.0, 16.0], SCALE
    )
    separate = model(one) + model(AtomicStructure([8], [[10.0, 0.0, 0.0]], [16.0], SCALE))
    np.testing.assert_allclose(model(disconnected), separate, rtol=1e-12, atol=1e-12)


def test_jit_vjp_and_second_order_parameter_derivative():
    model = _model()
    batch = AtomisticBatch.from_structure(_structure())
    compiled = jax.jit(lambda position: model.energy(batch, positions=position))
    energy = compiled(batch.positions)
    assert energy.shape == (1,)
    _, pullback = jax.vjp(
        lambda position: model.energy(batch, positions=position), batch.positions
    )
    position_gradient = pullback(jnp.ones((1,), dtype=energy.dtype))[0]
    assert position_gradient.shape == batch.positions.shape

    def embedding_energy(embedding):
        candidate = eqx.tree_at(lambda value: value.embedding, model, embedding)
        return jnp.sum(candidate.energy(batch))

    gradient = jax.grad(embedding_energy)
    first = gradient(model.embedding)
    second = jax.jvp(gradient, (model.embedding,), (jnp.ones_like(model.embedding),))[1]
    assert bool(jnp.all(jnp.isfinite(first)))
    assert bool(jnp.all(jnp.isfinite(second)))


def test_precision_policy_controls_prediction_storage():
    precision = AtomisticPrecisionPolicy(
        coordinate_dtype="float32",
        compute_dtype="float32",
        reduction_dtype="float32",
        output_dtype="float32",
    )
    structure = AtomicStructure(
        [1, 1],
        np.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        SCALE,
        coordinate_dtype="float32",
    )
    prediction = energy_and_forces(_model(precision=precision), structure)
    assert prediction.energy.dtype == jnp.float32
    assert prediction.forces.dtype == jnp.float32


def test_periodic_metadata_is_preserved_then_rejected_by_painn():
    periodic = AtomicStructure(
        [1],
        [[0.0, 0.0, 0.0]],
        [1.0],
        SCALE,
        cell=np.eye(3),
        periodic_axes=[True, False, False],
    )
    assert periodic.has_periodic_metadata
    with pytest.raises(ValueError, match="nonperiodic"):
        energy_and_forces(_model(), periodic)


def test_neighbor_overflow_returns_invalid_typed_prediction_and_direct_energy_fails():
    structure = AtomicStructure(
        [1, 1], [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [1.0, 1.0], SCALE
    )
    model = _model(maximum_neighbors=0)
    prediction = energy_and_forces(model, structure)
    assert not bool(prediction.valid[0])
    assert int(prediction.status[0]) == int(AtomisticStatus.NEIGHBOR_OVERFLOW)
    assert bool(jnp.isnan(prediction.energy[0]))
    with pytest.raises(Exception, match="overflow"):
        model(structure)


def test_atomwise_update_is_canonical_norm_conditioned_three_head_equation():
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


def test_nonfinite_padding_geometry_is_inert_before_neural_messages():
    structure = AtomicStructure(
        [1, 1, 0],
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [np.nan, np.nan, np.nan]],
        [1.0, 1.0, 0.0],
        SCALE,
        active_mask=[True, True, False],
    )
    prediction = energy_and_forces(_model(), structure)
    assert bool(prediction.valid[0])
    assert bool(jnp.all(jnp.isfinite(prediction.energy)))
    assert bool(jnp.all(jnp.isfinite(prediction.forces)))
    np.testing.assert_allclose(prediction.forces[0, 2], 0.0)


def test_parameter_state_provenance_distinguishes_same_architecture_models():
    first = _model(seed=31)
    second = _model(seed=32)
    assert first.architecture_id == second.architecture_id
    assert first.parameter_state_id != second.parameter_state_id
    assert first.potential_id != second.potential_id
    provenance = energy_and_forces(first, _structure()).provenance
    assert provenance.architecture_id == first.architecture_id
    assert provenance.parameter_state_id == first.parameter_state_id


def test_force_is_cast_to_output_dtype_when_coordinate_dtype_differs():
    precision = AtomisticPrecisionPolicy(
        coordinate_dtype="float64",
        compute_dtype="float32",
        reduction_dtype="float64",
        output_dtype="float32",
    )
    prediction = energy_and_forces(_model(precision=precision), _structure())
    assert prediction.energy.dtype == jnp.float32
    assert prediction.forces.dtype == jnp.float32
    assert prediction.net_force.dtype == jnp.float32
    assert prediction.net_torque.dtype == jnp.float32


def test_external_parameter_updates_require_immutable_public_checkpoint():
    original = _model(seed=51)
    updated = eqx.tree_at(
        lambda potential: potential.embedding,
        original,
        original.embedding + 0.125,
    )
    assert updated.parameter_state_id == original.parameter_state_id
    checkpointed = checkpoint_atomistic_potential(updated)
    assert checkpointed is not updated
    assert updated.parameter_state_id == original.parameter_state_id
    assert checkpointed.parameter_state_id != updated.parameter_state_id
    prediction = energy_and_forces(checkpointed, _structure())
    assert prediction.provenance.parameter_state_id == checkpointed.parameter_state_id


def test_torque_uses_reduction_precision_before_output_cast_for_huge_coordinates():
    precision = AtomisticPrecisionPolicy(
        coordinate_dtype="float64",
        compute_dtype="float32",
        reduction_dtype="float64",
        output_dtype="float32",
    )
    structure = AtomicStructure(
        [1],
        [[1e30, -1e30, 1e30]],
        [1.0],
        SCALE,
        coordinate_dtype="float64",
    )
    prediction = energy_and_forces(_model(precision=precision), structure)
    np.testing.assert_allclose(prediction.forces, 0.0, atol=0.0)
    np.testing.assert_allclose(prediction.net_torque, 0.0, atol=0.0)
    assert bool(jnp.all(jnp.isfinite(prediction.net_torque)))
    assert prediction.net_torque.dtype == jnp.float32
