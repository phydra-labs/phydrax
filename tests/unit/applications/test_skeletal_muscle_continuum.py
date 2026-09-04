#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.ein as ein
from phydrax._trainable import partition_trainable
from phydrax.applications.skeletal_muscle.continuum import (
    affine_mesh_power_evidence,
    EngelhardtGasam2025Parameters,
    EngelhardtGasam2025Plan,
    GasamQualificationPlan,
    UniformFiberArchitecturePlan,
)


def _material(activation=0.0, *, material_id="test-gasam"):
    architecture = UniformFiberArchitecturePlan("test-longitudinal").prepare(
        jnp.asarray((2.0, 0.0, 0.0))
    )
    return EngelhardtGasam2025Plan(material_id).prepare(
        EngelhardtGasam2025Parameters.published_multiload_fit(),
        architecture,
        activation,
    )


def _isochoric_fiber_stretch(stretch):
    transverse = stretch ** -0.5
    return jnp.diag(jnp.asarray((stretch, transverse, transverse)))


def test_material_parameters_are_dynamic_jax_leaves():
    parameters = EngelhardtGasam2025Parameters.published_multiload_fit()
    leaves = jax.tree_util.tree_leaves(parameters)

    assert len(leaves) == 7
    assert all(isinstance(value, jax.Array) for value in leaves)
    material = _material(0.5)
    trainable, fixed = partition_trainable(material)
    assert trainable.parameters.alpha is not None
    assert trainable.parameters.peak_active_nominal_stress_pa is not None
    assert fixed.plan is material.plan


def test_uniform_architecture_normalizes_and_is_sign_indifferent():
    plan = UniformFiberArchitecturePlan("longitudinal")
    positive = plan.prepare(jnp.asarray((4.0, 0.0, 0.0)))
    negative = plan.prepare(jnp.asarray((-4.0, 0.0, 0.0)))

    assert bool(positive.evidence.valid)
    np.testing.assert_allclose(
        positive.reference_direction, (1.0, 0.0, 0.0)
    )
    np.testing.assert_allclose(
        positive.structural_tensor, negative.structural_tensor
    )


def test_passive_reference_and_source_force_length_limits():
    passive = _material(0.0)
    identity = jnp.eye(3)
    response = passive.evaluate(identity, 0.0)
    terms_below = passive.source_terms(_isochoric_fiber_stretch(0.55))
    terms_optimal = passive.source_terms(
        _isochoric_fiber_stretch(passive.parameters.optimal_active_stretch)
    )

    np.testing.assert_allclose(
        response.reference_energy_density, 0.0, atol=2.0e-3
    )
    np.testing.assert_allclose(
        response.first_piola, jnp.zeros((3, 3)), atol=2.0e-2
    )
    np.testing.assert_allclose(terms_below[3], 0.0, atol=0.0)
    np.testing.assert_allclose(terms_below[4], 0.0, atol=0.0)
    np.testing.assert_allclose(terms_below[5], 0.0, atol=0.0)
    np.testing.assert_allclose(terms_optimal[3], 1.0, rtol=2.0e-5)


def test_active_energy_derivative_recovers_source_nominal_force_length():
    activation = 0.6
    active = _material(activation)
    passive = _material(0.0)
    stretch = jnp.asarray(1.0)

    def active_energy(value):
        return active.reference_energy_density(_isochoric_fiber_stretch(value))

    def passive_energy(value):
        return passive.reference_energy_density(_isochoric_fiber_stretch(value))

    active_increment = jax.grad(
        lambda value: active_energy(value) - passive_energy(value)
    )(stretch)
    force_length = active.source_terms(_isochoric_fiber_stretch(stretch))[3]
    expected = (
        active.parameters.peak_active_nominal_stress_pa * activation * force_length
    )
    np.testing.assert_allclose(
        active_increment, expected, rtol=5.0e-5, atol=5.0e-2
    )


def test_complete_active_potential_is_objective_and_has_consistent_tangent():
    material = _material(0.8)
    deformation = jnp.asarray(
        ((1.08, 0.06, 0.0), (0.01, 0.97, 0.03), (0.0, 0.02, 0.96))
    )
    direction = jnp.asarray(
        ((0.02, -0.01, 0.0), (0.01, 0.0, 0.015), (0.0, -0.01, -0.02))
    )
    angle = 0.31
    rotation = jnp.asarray(
        (
            (jnp.cos(angle), -jnp.sin(angle), 0.0),
            (jnp.sin(angle), jnp.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    response = material.evaluate(deformation, 1200.0)
    rotated = material.evaluate(rotation @ deformation, 1200.0)
    tangent = material.block_tangent(
        deformation, 1200.0
    ).deformation_deformation
    stress_jvp = jax.jvp(
        lambda value: material.evaluate(value, 1200.0).first_piola,
        (deformation,),
        (direction,),
    )[1]
    tangent_jvp = ein.contract("iJkL,kL->iJ", tangent, direction)

    np.testing.assert_allclose(
        response.reference_energy_density,
        rotated.reference_energy_density,
        rtol=2.0e-5,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        rotated.first_piola,
        rotation @ response.first_piola,
        rtol=5.0e-5,
        atol=3.0e-2,
    )
    np.testing.assert_allclose(
        stress_jvp, tangent_jvp, rtol=5.0e-5, atol=5.0e-2
    )


def test_invalid_activation_candidate_rolls_back_whole_material_state():
    material = _material(0.25)
    candidate = material.propose_activation(1.25)
    commit = candidate.commit()
    selected = material.with_commit(commit)

    assert not commit.committed
    assert commit.rollback_applied
    np.testing.assert_array_equal(
        selected.state.activation, material.state.activation
    )
    compiled = eqx.filter_jit(
        lambda activation: material.propose_activation(activation).commit()
    )(jnp.asarray(1.25))
    assert not bool(compiled.committed)
    np.testing.assert_array_equal(
        compiled.state.activation, material.state.activation
    )
    assert selected.state.state_id == material.state.state_id


def test_material_commit_rejects_a_foreign_prepared_owner_without_mutation():
    material = _material(0.25)
    foreign = _material(0.5, material_id="foreign-gasam")
    commit = foreign.propose_activation(0.75).commit()

    assert commit.prepared_id == foreign.prepared_id
    np.testing.assert_array_equal(commit.source_state_id, foreign.state.state_id)
    np.testing.assert_array_equal(commit.source_activation, foreign.state.activation)
    with pytest.raises(ValueError, match="different prepared material"):
        material.with_commit(commit)

    np.testing.assert_array_equal(material.state.activation, 0.25)
    np.testing.assert_array_equal(material.state.state_id, 0)


def test_material_commit_rejects_a_stale_source_state_without_mutation():
    material = _material(0.25)
    first = material.propose_activation(0.5).commit()
    stale = material.propose_activation(0.75).commit()
    advanced = material.with_commit(first)

    assert first.prepared_id == material.prepared_id
    np.testing.assert_array_equal(first.source_state_id, material.state.state_id)
    np.testing.assert_array_equal(first.source_activation, material.state.activation)
    assert advanced.state.state_id != material.state.state_id
    with pytest.raises(ValueError, match="stale or different source state"):
        advanced.with_commit(stale)

    np.testing.assert_array_equal(advanced.state.activation, 0.5)
    np.testing.assert_array_equal(advanced.state.state_id, first.state.state_id)


def test_material_commit_rejects_a_source_mismatched_sibling_state():
    material = _material(0.25)
    left = material.with_commit(material.propose_activation(0.5).commit())
    right = material.with_commit(material.propose_activation(0.75).commit())
    left_commit = left.propose_activation(0.9).commit()

    np.testing.assert_array_equal(left_commit.source_state_id, right.state.state_id)
    with pytest.raises(ValueError, match="stale or different source state"):
        right.with_commit(left_commit)

    np.testing.assert_array_equal(right.state.activation, 0.75)
    np.testing.assert_array_equal(right.state.state_id, 1)


def test_qualification_reports_local_not_global_active_stability():
    material = _material(0.65)
    deformation = _isochoric_fiber_stretch(1.0)
    rate = jnp.asarray(
        ((0.01, 0.002, 0.0), (0.0, -0.005, 0.0), (0.0, 0.0, -0.005))
    )
    evidence = GasamQualificationPlan().evaluate(material, deformation, rate)

    assert bool(evidence.valid)
    assert evidence.minimum_acoustic_value_pa > 0.0
    assert not evidence.active_global_stability_claimed
    assert "polyconvex" in evidence.passive_polyconvexity_source


def test_affine_mesh_energy_and_power_are_capacity_mask_invariant():
    material = _material(0.5)
    volumes = jnp.asarray(
        (
            (1.0e-6, 0.0, 0.0, 0.0),
            (0.5e-6, 0.5e-6, 0.0, 0.0),
            (0.25e-6, 0.25e-6, 0.25e-6, 0.25e-6),
        )
    )
    mask = volumes > 0.0
    evidence = affine_mesh_power_evidence(
        material,
        volumes,
        mask,
        _isochoric_fiber_stretch(1.05),
        jnp.diag(jnp.asarray((0.02, -0.01, -0.01))),
    )

    assert bool(evidence.valid)
    np.testing.assert_array_equal(evidence.active_cell_counts, (1, 2, 4))
    np.testing.assert_allclose(evidence.energy_errors_j, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(evidence.power_errors_w, 0.0, atol=1.0e-10)
