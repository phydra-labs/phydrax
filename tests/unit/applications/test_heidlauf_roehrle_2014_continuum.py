#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.skeletal_muscle.continuum import (
    HeidlaufRoehrle2014Parameters,
    HeidlaufRoehrle2014Plan,
    HeidlaufRoehrle2014QualificationPlan,
    HeidlaufRoehrle2014StressInput,
    UniformFiberArchitecturePlan,
)


def _input(gamma, token=0, *, source="prescribed-test-stress", successful=True):
    return HeidlaufRoehrle2014StressInput(
        gamma,
        jnp.full((8,), token, dtype=jnp.uint32),
        source,
        source_successful=successful,
    )


def _material(gamma=0.0, *, name="test-2014"):
    architecture = UniformFiberArchitecturePlan("source-x-fibers").prepare(
        jnp.asarray((1.0, 0.0, 0.0))
    )
    return HeidlaufRoehrle2014Plan(name, "prescribed-test-stress").prepare(
        HeidlaufRoehrle2014Parameters.published_table_2(),
        architecture,
        _input(gamma),
    )


def _numpy_source_equation(deformation, pressure, gamma):
    """Literal Eq. 1 transcription, independent of the energy and JAX kernel."""
    c10, c01, b1, d1, maximum = 6.352e-7, 3627.0, 0.02756, 43.373, 73000.0
    c = deformation.T @ deformation
    direction = np.asarray((1.0, 0.0, 0.0))
    stretch = np.sqrt(direction @ c @ direction)
    structure = np.outer(direction, direction)
    second = (
        2 * c10 * np.eye(3)
        + 2 * c01 * (np.trace(c) * np.eye(3) - c)
        + b1 * (stretch ** (d1 - 2) - stretch**-2) * structure
        + maximum * gamma / stretch * structure
        - pressure * np.linalg.inv(c)
    )
    return deformation @ second


@pytest.mark.parametrize("stretch", (0.79, 1.0, 1.26))
def test_source_stress_matches_independent_equation_including_compression(stretch):
    material = _material(1.35)
    deformation = np.asarray(
        ((stretch, 0.03, 0.0), (0.0, stretch**-0.5, 0.02), (0.0, 0.0, stretch**-0.5))
    )
    actual = material.evaluate(jnp.asarray(deformation), 8600.0)
    expected = _numpy_source_equation(deformation, 8600.0, 1.35)
    np.testing.assert_allclose(actual.first_piola, expected, rtol=3e-5, atol=0.2)
    assert bool(actual.evidence.valid)


def test_passive_energy_gradient_and_source_pressure_are_unprojected_off_constraint():
    material = _material()
    deformation = jnp.diag(jnp.asarray((0.82, 1.1, 1.04)))
    pressure = 8500.0
    response = material.evaluate(deformation, pressure)
    gradient = jax.grad(material.passive_energy_density)(deformation)
    np.testing.assert_allclose(
        gradient, response.passive_first_piola, rtol=2e-5, atol=0.02
    )
    np.testing.assert_allclose(
        response.pressure_first_piola,
        -pressure * np.linalg.inv(deformation).T,
        rtol=2e-6,
        atol=0.01,
    )
    np.testing.assert_allclose(
        response.constraint_residual, -np.log(np.linalg.det(deformation)), rtol=2e-6
    )
    anisotropic_axial = (
        response.passive_first_piola[0, 0]
        - 2 * material.parameters.c10_pa * deformation[0, 0]
        - 2
        * material.parameters.c01_pa
        * deformation[0, 0]
        * (deformation[1, 1] ** 2 + deformation[2, 2] ** 2)
    )
    assert float(anisotropic_axial) < 0.0


def test_passive_rest_requires_the_source_hydrostatic_pressure_not_zero():
    material = _material()
    rest_pressure = 2 * material.parameters.c10_pa + 4 * material.parameters.c01_pa
    np.testing.assert_allclose(
        material.evaluate(jnp.eye(3), rest_pressure).first_piola, 0.0, atol=0.003
    )
    assert float(material.evaluate(jnp.eye(3), 0.0).first_piola[1, 1]) > 14000.0


def test_active_input_is_not_an_activation_bound_or_second_length_multiplier():
    material = _material(1.7)
    compressed = material.evaluate(jnp.diag(jnp.asarray((0.8, 1.0, 1.0))), 0.0)
    stretched = material.evaluate(jnp.diag(jnp.asarray((1.2, 1.0, 1.0))), 0.0)
    np.testing.assert_allclose(
        compressed.active_first_piola[0, 0], 73000 * 1.7, rtol=2e-6
    )
    np.testing.assert_allclose(
        stretched.active_first_piola[0, 0], compressed.active_first_piola[0, 0], rtol=2e-6
    )
    signed = material.with_commit(
        material.propose_active_stress(_input(-0.2, 1)).commit()
    )
    assert float(signed.evaluate(jnp.eye(3), 0.0).active_first_piola[0, 0]) < 0.0
    np.testing.assert_allclose(
        signed.passive_energy_density(jnp.eye(3)),
        material.passive_energy_density(jnp.eye(3)),
    )


def test_objectivity_passive_gradient_active_power_and_full_tangent_difference():
    material = _material(0.6)
    deformation = jnp.asarray(((0.94, 0.04, 0.01), (0.0, 1.03, 0.02), (0.01, 0.0, 1.01)))
    rate = jnp.asarray(((0.04, -0.02, 0.01), (0.01, -0.03, 0.02), (0.0, 0.01, 0.01)))
    evidence = HeidlaufRoehrle2014QualificationPlan().evaluate(
        material, deformation, 1000.0, rate
    )
    assert bool(evidence.valid)
    tangent = material.block_tangent(deformation, 1000.0)
    current = deformation @ material.architecture.reference_direction
    expected = 73000 * jnp.outer(
        current / jnp.linalg.norm(current), material.architecture.reference_direction
    )
    np.testing.assert_allclose(
        tangent.deformation_active_stress, expected, rtol=3e-6, atol=0.01
    )


@pytest.mark.parametrize(
    "cause", ("nonfinite", "source-failure", "foreign-source", "outer-failure")
)
def test_every_constitutive_input_failure_rolls_back_all_state_leaves_under_jit(cause):
    material = _material(0.3)
    source = "other-source" if cause == "foreign-source" else "prescribed-test-stress"
    incoming = _input(
        float("nan") if cause == "nonfinite" else 0.7,
        9,
        source=source,
        successful=cause != "source-failure",
    )
    commit = eqx.filter_jit(
        lambda value: material.propose_active_stress(value).commit(
            successful=cause != "outer-failure"
        )
    )(incoming)
    selected = material.with_commit(commit)
    assert bool(commit.rollback_applied)
    for before, after in zip(
        jax.tree_util.tree_leaves(material.state),
        jax.tree_util.tree_leaves(selected.state),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)


def test_stale_foreign_and_changed_numeric_revision_cannot_apply_commit():
    material = _material(0.3)
    first = material.propose_active_stress(_input(0.6, 1)).commit()
    advanced = material.with_commit(first)
    stale = material.propose_active_stress(_input(0.8, 2)).commit()
    with pytest.raises(Exception, match="stale source"):
        advanced.with_commit(stale)
    with pytest.raises(ValueError, match="foreign prepared"):
        _material(0.3, name="other").with_commit(first)
    changed = eqx.tree_at(
        lambda value: value.parameters.c01_pa, material, material.parameters.c01_pa * 1.1
    )
    with pytest.raises(Exception, match="parameter revision"):
        changed.with_commit(first)
    assert advanced.prepared_id == material.prepared_id
    assert (
        advanced.numeric_revision().revision_id != material.numeric_revision().revision_id
    )


def test_reflection_and_partial_quadrature_coverage_fail_without_repair():
    material = _material(0.4)
    response = material.evaluate(jnp.diag(jnp.asarray((-1.0, 1.0, 1.0))), 0.0)
    assert not bool(response.evidence.valid)
    assert bool(jnp.all(jnp.isnan(response.first_piola)))
    tangent = eqx.filter_jit(material.block_tangent)(
        jnp.diag(jnp.asarray((-1.0, 1.0, 1.0))), 0.0
    )
    assert all(
        bool(jnp.all(jnp.isnan(block))) for block in jax.tree_util.tree_leaves(tangent)
    )
    invalid = jnp.diag(jnp.asarray((-1.0, 1.0, 1.0)))
    stress_jvp = jax.jit(
        lambda deformation: jax.jvp(
            lambda value: material.first_piola(value, 0.0),
            (deformation,),
            (jnp.eye(3),),
        )[1]
    )(invalid)
    stress_jacobian = jax.jit(
        jax.jacfwd(lambda deformation: material.first_piola(deformation, 0.0))
    )(invalid)
    stress_vjp = jax.jit(
        jax.grad(lambda deformation: jnp.sum(material.first_piola(deformation, 0.0)))
    )(invalid)
    constraint_gradient = jax.jit(jax.grad(material.constraint))(invalid)
    constraint_jvp = jax.jit(
        lambda deformation: jax.jvp(material.constraint, (deformation,), (jnp.eye(3),))[1]
    )(invalid)
    for derivative in (
        stress_jvp,
        stress_jacobian,
        stress_vjp,
        constraint_gradient,
        constraint_jvp,
    ):
        assert not bool(jnp.all(jnp.isfinite(derivative)))
    with pytest.raises(ValueError, match="exactly cover"):
        material.first_piola_points(
            jnp.broadcast_to(jnp.eye(3), (2, 4, 3, 3)),
            jnp.zeros((2, 4)),
            jnp.zeros((2, 1)),
        )
