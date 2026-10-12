#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.applications.solid_mechanics._mpm_coupled_fields import (
    effective_pore_pressure_first_piola,
)
from phydrax.discretization.mpm._dynamics import PreparedMPMDynamics
from phydrax.discretization.mpm._schedule import AbstractExplicitMPMSchedule
from phydrax.discretization.mpm._types import MPMRuntimeState, MPMStepResult
from phydrax.equations._material_point import (
    CompiledMaterialPointProblem,
    MaterialPointArguments,
)


def _case(
    *,
    clamp_x: bool = False,
    schedule: AbstractExplicitMPMSchedule | None = None,
    stateful: bool = False,
) -> tuple[CompiledMaterialPointProblem, MaterialPointArguments, MPMRuntimeState]:
    dimension = 3 if stateful else 2
    bounds = jnp.stack(
        (
            jnp.zeros((dimension,), dtype=jnp.float64),
            jnp.ones((dimension,), dtype=jnp.float64),
        )
    )
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(12, periodic=True, endpoint=False)
            for _ in range(dimension)
        ),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(bounds)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4, dtype=jnp.int32),
        jnp.full((4,), 0.01, dtype=jnp.float64),
        ambient_dimension=dimension,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    boundary = None
    if clamp_x:
        boundary = phx.discretization.PrescribedGridVelocityPlan(
            jnp.zeros(grid.vertices().shape + (dimension,), dtype=jnp.bool_)
            .at[..., 0]
            .set(True)
        )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "predictor-split-test",
            phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
            if stateful
            else phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(
                dimension
            ),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(schedule=schedule),
        phx.discretization.MPMParticleDomainPlan(
            bounds,
            periodic=(True,) * dimension,
            support_margin=0.0,
        ),
        boundary=boundary,
    )
    arguments = MaterialPointArguments(
        phx.applications.solid_mechanics.FiniteStrainJ2Parameters(10.0, 30.0, 0.15, 2.0)
        if stateful
        else phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
            2.0, 8.0
        )
    )
    position = jnp.asarray(
        ((0.28, 0.31, 0.35), (0.42, 0.36, 0.44), (0.34, 0.49, 0.53), (0.48, 0.52, 0.61)),
        dtype=jnp.float64,
    )[:, :dimension]
    velocity = jnp.broadcast_to(
        jnp.asarray((0.08, -0.03, 0.01), dtype=jnp.float64)[:dimension], position.shape
    )
    state = compiled.initialize_state(
        position, velocity, jnp.full((4,), 0.01, dtype=jnp.float64), arguments
    )
    return compiled, arguments, state


@pytest.mark.parametrize("clamp_x", (False, True), ids=("free", "constrained"))
def test_uncorrected_split_is_exactly_the_detailed_step(clamp_x: bool) -> None:
    compiled, arguments, state = _case(clamp_x=clamp_x)
    dynamics = compiled.dynamics
    dt = jnp.asarray(0.001, dtype=jnp.float64)
    expected = eqx.filter_jit(PreparedMPMDynamics.step_detailed)(
        dynamics, state, dt, arguments
    )

    @eqx.filter_jit
    def split(
        prepared: PreparedMPMDynamics,
        value: MPMRuntimeState,
        step_size: Array,
        material_arguments: MaterialPointArguments,
    ) -> MPMStepResult:
        # Both routes receive the plan and material arrays as dynamic leaves;
        # captured constants would compare different XLA numerical programs.
        predictor = prepared.predict_grid(value, step_size, material_arguments)
        return prepared.finalize_grid_velocity(predictor, predictor.grid.velocity)

    actual = split(dynamics, state, dt, arguments)
    assert bool(actual.successful)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)


def test_corrected_velocity_updates_deformation_and_material_once() -> None:
    compiled, arguments, state = _case()
    dynamics = compiled.dynamics
    dt = jnp.asarray(0.001, dtype=jnp.float64)
    predictor = dynamics.predict_grid(state, dt, arguments)
    assert bool(predictor.successful)
    assert predictor.state.time == state.time
    assert predictor.state.accepted_step == state.accepted_step
    for frozen, committed in zip(
        jax.tree.leaves(predictor.state.particles),
        jax.tree.leaves(state.particles),
        strict=True,
    ):
        np.testing.assert_array_equal(frozen, committed)
    gradient = jnp.asarray(((0.2, 0.1), (-0.05, -0.1)), dtype=jnp.float64)
    correction = (dynamics.grid_coordinates @ gradient.T).reshape(
        dynamics.nodal_shape + (2,)
    )
    corrected = predictor.grid.velocity + correction[None, ...]
    result = dynamics.finalize_grid_velocity(predictor, corrected)
    expected_deformation = jnp.broadcast_to(
        jnp.eye(2, dtype=jnp.float64) + dt * gradient,
        state.particles.deformation_gradient.shape,
    )
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.accepted_state.particles.deformation_gradient,
        expected_deformation,
        rtol=1e-12,
        atol=1e-12,
    )
    material = dynamics.material.evaluate(
        expected_deformation,
        state.particles.material_state,
        dynamics.particles.safe_masses / state.particles.reference_volume,
        arguments.material_parameters,
        state.time + dt,
        dt,
    )
    np.testing.assert_allclose(
        result.accepted_state.particles.first_piola, material.first_piola
    )
    np.testing.assert_array_equal(
        result.accepted_state.particles.material_state, material.trial_state
    )
    assert result.accepted_state.time == state.time + dt
    assert result.accepted_state.accepted_step == state.accepted_step + 1
    assert result.diagnostics.transfer.valid_routes == predictor.routes.valid_route_count


@pytest.mark.parametrize("stateful", (False, True), ids=("elastic", "j2-history"))
def test_failed_corrected_material_rolls_back_the_committed_state(stateful: bool) -> None:
    compiled, arguments, state = _case(stateful=stateful)
    dynamics = compiled.dynamics
    predictor = dynamics.predict_grid(state, 0.001, arguments)
    correction = (
        jnp.zeros_like(dynamics.grid_coordinates)
        .at[:, 0]
        .set(-2000.0 * dynamics.grid_coordinates[:, 0])
        .reshape(dynamics.nodal_shape + (dynamics.dimension,))
    )
    result = dynamics.finalize_grid_velocity(
        predictor, predictor.grid.velocity + correction[None, ...]
    )
    assert not bool(result.successful)
    assert not bool(result.diagnostics.material_admissible)
    assert int(result.rejection_reasons) & int(
        phx.discretization.MPMRejectionReason.MATERIAL
    )
    assert result.accepted_state.time == state.time
    assert result.accepted_state.accepted_step == state.accepted_step
    for accepted, committed in zip(
        jax.tree.leaves(result.accepted_state.particles),
        jax.tree.leaves(state.particles),
        strict=True,
    ):
        np.testing.assert_array_equal(accepted, committed)


def test_nodal_force_increment_is_applied_only_to_the_frozen_attempt() -> None:
    compiled, arguments, state = _case()
    dynamics = compiled.dynamics
    dt = jnp.asarray(0.001, dtype=jnp.float64)
    baseline = dynamics.predict_grid(state, dt, arguments)
    acceleration = jnp.asarray((0.3, -0.2), dtype=jnp.float64)
    predictor = dynamics.predict_grid(
        state,
        dt,
        arguments,
        nodal_force_increment=baseline.grid.mass[..., None] * acceleration,
    )
    expected = baseline.grid.velocity + jnp.where(
        baseline.grid.active[..., None], dt * acceleration, 0.0
    )
    np.testing.assert_allclose(predictor.grid.velocity, expected, rtol=1e-12, atol=1e-12)
    result = dynamics.finalize_grid_velocity(predictor, predictor.grid.velocity)
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.accepted_state.particles.velocity,
        state.particles.velocity + dt * acceleration,
        rtol=1e-12,
        atol=1e-12,
    )


def test_first_piola_increment_matches_one_force_predictor_evaluation() -> None:
    compiled, arguments, state = _case()
    dynamics = compiled.dynamics
    increment = jnp.broadcast_to(
        jnp.asarray(((0.4, 0.1), (0.1, -0.2)), dtype=jnp.float64),
        state.particles.first_piola.shape,
    )
    predictor = dynamics.predict_grid(
        state, 0.001, arguments, first_piola_increment=increment
    )
    force_state = eqx.tree_at(
        lambda value: value.particles.first_piola,
        state,
        state.particles.first_piola + increment,
    )
    expected = dynamics.step_detailed(force_state, 0.001, arguments)
    actual = dynamics.finalize_grid_velocity(predictor, predictor.grid.velocity)
    assert bool(actual.successful)
    np.testing.assert_array_equal(
        actual.grid.internal_force, expected.grid.internal_force
    )
    np.testing.assert_allclose(
        actual.accepted_state.particles.deformation_gradient,
        expected.accepted_state.particles.deformation_gradient,
        rtol=8.0 * np.finfo(np.float64).eps,
        atol=8.0 * np.finfo(np.float64).eps,
    )
    np.testing.assert_array_equal(
        predictor.state.particles.first_piola, state.particles.first_piola
    )


@pytest.mark.parametrize(
    "schedule",
    (phx.discretization.USFMPMSchedule(), phx.discretization.MUSLMPMSchedule()),
    ids=("usf", "musl"),
)
def test_split_refuses_schedules_that_do_not_finalize_the_corrected_gradient(
    schedule: AbstractExplicitMPMSchedule,
) -> None:
    compiled, arguments, state = _case(schedule=schedule)
    with pytest.raises(ValueError, match="USL"):
        compiled.dynamics.predict_grid(state, 0.001, arguments)
    assert bool(compiled.dynamics.step_detailed(state, 0.001, arguments).successful)


@pytest.mark.strict_jax
def test_pore_pressure_piola_integer_identity_preserves_fractional_pressure() -> None:
    deformation = jnp.eye(2, dtype=jnp.int32)
    result = effective_pore_pressure_first_piola(deformation, 0.5, 0.7)
    np.testing.assert_allclose(result, -0.35 * jnp.eye(2, dtype=jnp.float64), rtol=1e-14)


@pytest.mark.strict_jax
def test_pore_pressure_piola_matches_the_cofactor_and_has_finite_derivatives() -> None:
    deformation = jnp.asarray(
        (((1.2, 0.3), (-0.1, 0.8)), ((0.9, -0.2), (0.1, 1.1))), dtype=jnp.float64
    )
    pressure = jnp.asarray((2.0, 3.0), dtype=jnp.float64)
    alpha = jnp.asarray(0.7, dtype=jnp.float64)
    cofactor = np.asarray((((0.8, 0.1), (-0.3, 1.2)), ((1.1, -0.1), (0.2, 0.9))))
    expected = -0.7 * np.asarray(pressure)[:, None, None] * cofactor
    actual = jax.jit(effective_pore_pressure_first_piola)(deformation, pressure, alpha)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    derivative = jax.jacfwd(effective_pore_pressure_first_piola)(
        deformation, pressure, alpha
    )
    assert bool(jnp.all(jnp.isfinite(derivative)))
    np.testing.assert_array_equal(
        effective_pore_pressure_first_piola(deformation, jnp.zeros_like(pressure), alpha),
        jnp.zeros_like(deformation),
    )


@pytest.mark.parametrize("step_size", (0.0, 10.0), ids=("invalid-step", "unstable-step"))
def test_failed_predictor_preserves_detailed_step_rollback(step_size: float) -> None:
    compiled, arguments, state = _case()
    dynamics = compiled.dynamics
    expected = dynamics.step_detailed(state, step_size, arguments)
    predictor = dynamics.predict_grid(state, step_size, arguments)
    assert not bool(predictor.successful)
    actual = dynamics.finalize_grid_velocity(predictor, predictor.grid.velocity)
    assert not bool(actual.successful)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)
