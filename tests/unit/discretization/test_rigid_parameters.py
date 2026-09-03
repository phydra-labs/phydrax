#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.particle._core import ParticleSetPlan
from phydrax.discretization.particle._rigid_body import RigidBodySetPlan
from phydrax.discretization.particle._rigid_parameters import (
    realize_rigid_body_plans,
    RigidInertialCoordinates,
    RigidInertialParameterization,
    RigidInertialParameters,
)


def _inertia_from_covariance(masses, covariance):
    identity = jnp.eye(3, dtype=covariance.dtype)
    return masses[:, None, None] * (
        jnp.trace(covariance, axis1=-2, axis2=-1)[:, None, None] * identity
        - covariance
    )


def _body_origin_inertia(masses, offsets, inertia_com):
    identity = jnp.eye(3, dtype=inertia_com.dtype)
    outer = offsets[:, :, None] * offsets[:, None, :]
    return inertia_com + masses[:, None, None] * (
        jnp.sum(offsets * offsets, axis=-1)[:, None, None] * identity - outer
    )


def _prepared_source():
    masses = jnp.asarray([2.0, 3.5])
    offsets = jnp.asarray([[0.2, -0.1, 0.3], [-0.15, 0.25, 0.05]])
    factors = jnp.asarray(
        [
            [[0.7, 0.0, 0.0], [0.1, 0.8, 0.0], [-0.2, 0.05, 0.9]],
            [[0.5, 0.0, 0.0], [-0.15, 0.65, 0.0], [0.1, 0.2, 0.75]],
        ]
    )
    covariance = factors @ jnp.swapaxes(factors, -1, -2)
    inertia_com = _inertia_from_covariance(masses, covariance)
    inertia_body = _body_origin_inertia(masses, offsets, inertia_com)
    particle_plan = ParticleSetPlan(
        jnp.asarray([17, 29], dtype=jnp.int64),
        masses,
        ambient_dimension=3,
        active_mask=jnp.asarray([True, True]),
        name="parameterized-particles",
    )
    particles = particle_plan.prepare(numeric_version="source-numerics")
    rigid_plan = RigidBodySetPlan(
        jnp.asarray([4, 8], dtype=jnp.int32),
        inertia_body,
        fixed_mask=jnp.asarray([False, True]),
        name="parameterized-bodies",
    )
    return rigid_plan.prepare(particles), offsets, inertia_com


def test_prepared_inverse_round_trip_exposes_reconstruction_evidence():
    source, offsets, expected_inertia_com = _prepared_source()
    parameterization = RigidInertialParameterization(source)

    coordinates = parameterization.inverse(offsets)
    evaluation = parameterization.evaluate(coordinates)

    assert bool(evaluation.valid)
    np.testing.assert_allclose(
        evaluation.parameters.masses,
        source.particles.safe_masses,
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        evaluation.parameters.inertia_com,
        expected_inertia_com,
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        evaluation.parameters.inertia_body_origin,
        source.inertia_body,
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(evaluation.source_mass_residual, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(evaluation.source_inertia_residual, 0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        evaluation.mass_reconstruction_residual, 0.0, atol=1.0e-7
    )
    np.testing.assert_allclose(
        evaluation.center_of_mass_reconstruction_residual, 0.0, atol=1.0e-7
    )
    np.testing.assert_allclose(
        evaluation.inertia_reconstruction_residual, 0.0, atol=1.0e-6
    )
    np.testing.assert_allclose(
        evaluation.body_origin_reconstruction_residual, 0.0, atol=1.0e-6
    )
    assert np.all(np.asarray(evaluation.inertia_condition_number) >= 1.0)
    assert np.all(np.asarray(evaluation.pseudo_inertia_condition_number) >= 1.0)

    repeated = RigidInertialParameterization(source)
    repeated_coordinates = repeated.inverse(offsets)
    repeated_evaluation = repeated.evaluate(repeated_coordinates)
    assert repeated.parameterization_id == parameterization.parameterization_id
    assert repeated_coordinates.coordinates_id == coordinates.coordinates_id
    assert repeated_evaluation.evaluation_id == evaluation.evaluation_id


def test_all_decoded_inertias_are_spd_and_obey_strict_triangle_inequalities():
    source, _, _ = _prepared_source()
    parameterization = RigidInertialParameterization(source)
    coordinates = parameterization.coordinates(
        jnp.asarray([-4.0, 2.5]),
        jnp.asarray([[1.5, -0.7, 0.2], [-2.0, 0.3, 1.1]]),
        jnp.asarray(
            [
                [-1.5, 0.4, 0.7, -0.2, 0.6, 1.1],
                [0.3, -0.8, -0.4, 0.5, -0.1, 1.7],
            ]
        ),
    )

    evaluation = parameterization.evaluate(coordinates)
    inertia_com = np.asarray(evaluation.parameters.inertia_com)
    inertia_body = np.asarray(evaluation.parameters.inertia_body_origin)
    pseudo_inertia = np.asarray(evaluation.parameters.pseudo_inertia_body_origin)

    for inertia in (inertia_com, inertia_body):
        np.testing.assert_allclose(inertia, np.swapaxes(inertia, -1, -2))
        principal_moments = np.linalg.eigvalsh(inertia)
        assert np.all(principal_moments > 0.0)
        assert np.all(
            np.sum(principal_moments, axis=-1)
            - 2.0 * np.max(principal_moments, axis=-1)
            > 0.0
        )
    assert np.all(np.linalg.eigvalsh(pseudo_inertia) > 0.0)
    assert evaluation.inertia_spd_mask.tolist() == [True, True]
    assert evaluation.triangle_inequality_mask.tolist() == [True, True]
    assert evaluation.pseudo_inertia_spd_mask.tolist() == [True, True]
    assert evaluation.body_origin_inertia_spd_mask.tolist() == [True, True]
    assert evaluation.body_origin_triangle_inequality_mask.tolist() == [True, True]
    assert np.all(np.asarray(evaluation.minimum_inertia_eigenvalue) > 0.0)
    assert np.all(np.asarray(evaluation.minimum_triangle_margin) > 0.0)


def test_invalid_coordinates_and_nonphysical_direct_parameters_reject():
    source, _, _ = _prepared_source()
    parameterization = RigidInertialParameterization(source)
    with pytest.raises(ValueError, match="shape \\(N,3\\)"):
        parameterization.coordinates(
            jnp.zeros((2,)), jnp.zeros((2, 2)), jnp.zeros((2, 6))
        )
    with pytest.raises(ValueError, match="shape \\(N,6\\)"):
        parameterization.coordinates(
            jnp.zeros((2,)), jnp.zeros((2, 3)), jnp.zeros((2, 5))
        )
    with pytest.raises(ValueError, match="finite"):
        parameterization.coordinates(
            jnp.asarray([0.0, jnp.nan]), jnp.zeros((2, 3)), jnp.zeros((2, 6))
        )
    with pytest.raises(ValueError, match="triangle inequalities"):
        RigidInertialParameters(
            jnp.asarray([1.0]),
            jnp.zeros((1, 3)),
            jnp.diag(jnp.asarray([1.0, 1.0, 3.0]))[None, :, :],
            parameterization_id=parameterization.parameterization_id,
            coordinates_id="invalid-principal-moments",
        )
    with pytest.raises(ValueError, match="physically realizable"):
        parameterization.inverse(jnp.full((2, 3), 100.0))

    foreign = RigidInertialParameterization(source, parameterization_id="foreign")
    with pytest.raises(ValueError, match="different parameterization"):
        parameterization.evaluate(foreign.inverse())

    planar_particles = ParticleSetPlan(
        jnp.asarray([1]), jnp.asarray([1.0]), ambient_dimension=2
    ).prepare()
    planar_bodies = RigidBodySetPlan(
        jnp.asarray([0]), jnp.asarray([1.0])
    ).prepare(planar_particles)
    with pytest.raises(ValueError, match="three dimensions"):
        RigidInertialParameterization(planar_bodies)


def test_realization_returns_fresh_plans_and_preserves_prepared_owner():
    source, offsets, _ = _prepared_source()
    parameterization = RigidInertialParameterization(source)
    coordinates = parameterization.inverse(offsets)
    original_masses = np.asarray(source.particles.safe_masses).copy()
    original_inertia = np.asarray(source.inertia_body).copy()

    particle_plan, rigid_plan, evidence = realize_rigid_body_plans(
        parameterization, coordinates
    )

    assert particle_plan is not source.particles.plan
    assert rigid_plan is not source.plan
    assert particle_plan.plan_id != source.particles.plan.plan_id
    assert rigid_plan.plan_id != source.plan.plan_id
    assert evidence.requires_repreparation
    assert evidence.source_prepared_id == source.prepared_id
    np.testing.assert_array_equal(source.particles.safe_masses, original_masses)
    np.testing.assert_array_equal(source.inertia_body, original_inertia)

    changed_coordinates = parameterization.coordinates(
        coordinates.mass_coordinates.at[0].add(0.5),
        coordinates.center_of_mass_offsets.at[1, 0].add(0.125),
        coordinates.covariance_coordinates,
    )
    changed_particle_plan, changed_rigid_plan, changed_evidence = (
        parameterization.realize(changed_coordinates)
    )
    assert changed_evidence.evaluation_id != evidence.evaluation_id
    assert changed_particle_plan.plan_id != particle_plan.plan_id
    assert changed_rigid_plan.plan_id != rigid_plan.plan_id

    changed_particles = changed_particle_plan.prepare(
        numeric_version="realized-numerics"
    )
    changed_prepared = changed_rigid_plan.prepare(changed_particles)
    assert changed_prepared.prepared_id != source.prepared_id
    assert changed_prepared.particles.prepared_id != source.particles.prepared_id
    np.testing.assert_array_equal(source.particles.safe_masses, original_masses)
    np.testing.assert_array_equal(source.inertia_body, original_inertia)


def test_coordinate_constructor_requires_explicit_nonempty_binding():
    with pytest.raises(ValueError, match="parameterization_id"):
        RigidInertialCoordinates(
            jnp.asarray([0.0]),
            jnp.zeros((1, 3)),
            jnp.zeros((1, 6)),
            parameterization_id="",
        )
