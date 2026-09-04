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
from phydrax.nn.atomistic import NequIPPotential
from phydrax.units import ANGSTROM, ELECTRONVOLT


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)


def _execution(maximum_neighbors=3):
    return AtomisticGraphExecutionPlan(
        maximum_neighbors,
        maximum_dense_atoms=4,
    )


def _model(*, interaction_count=2):
    return NequIPPotential(
        SCALE,
        cutoff=2.5,
        feature_count=3,
        interaction_count=interaction_count,
        radial_basis_count=4,
        key=jr.key(27),
    )


def _structure(positions=None):
    if positions is None:
        positions = [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 0.8, 0.2]]
    return AtomicStructure([8, 1, 1], positions, [15.999, 1.008, 1.008], SCALE)


def test_energy_is_rigid_motion_invariant_and_force_is_equivariant():
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


def test_conservative_force_matches_energy_finite_difference():
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


def test_three_atom_energy_is_continuous_when_one_edge_crosses_cutoff():
    model = _model(interaction_count=1)

    def energy(distance):
        structure = AtomicStructure(
            [1, 6, 8],
            [[0.0, 0.0, 0.0], [0.7, 0.2, 0.0], [distance, 0.0, 0.0]],
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


def test_atom_and_species_permutation_preserves_energy_and_permutes_force():
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


def test_padding_is_masked_and_neighbor_overflow_fails_closed_without_truncation():
    model = _model()
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


def test_nonfinite_padding_geometry_is_sanitized_before_radial_and_angular_maps():
    model = _model()
    reference = AtomicStructure(
        [1, 8], [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]], [1.0, 16.0], SCALE
    )
    padded = AtomicStructure(
        [1, 8, 0, 0],
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.1, 0.0],
            [np.nan, np.nan, np.nan],
            [np.inf, -np.inf, np.inf],
        ],
        [1.0, 16.0, 0.0, 0.0],
        SCALE,
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


def test_radial_modulation_has_one_output_per_actual_tensor_product_weight():
    model = _model(interaction_count=1)
    interaction = model.interactions[0]
    plan = interaction.tensor_product.plan
    assert plan.path_count > 6
    assert plan.parameter_count > plan.path_count
    assert interaction.radial_out.out_size == plan.parameter_count
    assert interaction.radial_out.weight.shape[0] == plan.parameter_count
    assert model.configuration.maximum_degree == 2


def test_jit_position_vjp_and_second_parameter_derivative_are_finite():
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

    def embedding_energy(embedding):
        candidate = eqx.tree_at(lambda value: value.embedding, model, embedding)
        return jnp.sum(candidate.energy(batch, _execution()))

    gradient = jax.grad(embedding_energy)
    first = gradient(model.embedding)
    second = jax.jvp(gradient, (model.embedding,), (jnp.ones_like(model.embedding),))[1]
    assert bool(jnp.all(jnp.isfinite(first)))
    assert bool(jnp.all(jnp.isfinite(second)))


def test_periodic_metadata_and_tensor_product_resource_overflow_are_rejected():
    periodic = AtomicStructure(
        [1],
        [[0.0, 0.0, 0.0]],
        [1.0],
        SCALE,
        cell=np.eye(3),
        periodic_axes=[True, False, False],
    )
    with pytest.raises(ValueError, match="nonperiodic"):
        energy_and_forces(_model(), periodic, _execution())
    with pytest.raises(ValueError, match="parameters"):
        NequIPPotential(
            SCALE,
            cutoff=2.5,
            feature_count=3,
            interaction_count=1,
            radial_basis_count=4,
            maximum_tensor_product_parameters=1,
        )
