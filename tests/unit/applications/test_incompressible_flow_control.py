import io

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.incompressible_flow._control import (
    MACFlowControlPlan,
    MACFlowControlTarget,
)
from phydrax.applications.incompressible_flow._production import (
    MACConstantPressureGradientForcing,
)


def _periodic(*, count=4, viscosity=0.05, forcing=None):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0 * jnp.pi, 2.0 * jnp.pi]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    problem = phx.equations.IncompressibleFlowProblem(
        2,
        viscosity,
        forcing=forcing,
        forcing_id=None if forcing is None else forcing.forcing_id,
    )
    dynamics = phx.equations.compile_mac_incompressible_flow(
        problem,
        momentum,
        phx.solver.MACPressureProjectionPlan(
            operators, solve_method="transform", tolerance=1.0e-10
        ),
    )
    zero = dynamics.project_state(
        tuple(jnp.zeros(layout.shape) for layout in discretization.face_layouts)
    )
    return discretization, operators, momentum, dynamics, zero


def _channel(*, count=6, viscosity=0.1):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=False),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, -1.0], [2.0 * jnp.pi, 1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    dynamics = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, viscosity),
        momentum,
        phx.solver.MACPressureProjectionPlan(
            operators,
            solve_method="transform",
            tolerance=1.0e-10,
        ),
    )
    return discretization, operators, dynamics


def _all_dynamic_leaves_equal(left, right):
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_ssprk_exact_discrete_poiseuille_bulk_target():
    discretization, _, dynamics = _channel()
    viscosity = 0.1
    acceleration = 0.2
    x_faces = discretization.face_centers[0]
    y_faces = discretization.face_centers[1]
    velocity = (
        acceleration * (1.0 - x_faces[..., 1] ** 2) / (2.0 * viscosity),
        jnp.zeros(y_faces.shape[:-1]),
    )
    initial = dynamics.project_state(velocity)
    cell = dynamics.unpack_velocity(initial)[0]
    volumes = discretization.cell_volumes
    exact_discrete_bulk = jnp.sum(volumes * cell) / jnp.sum(volumes)
    prepared = MACFlowControlPlan(
        phx.solver.SSPRK33FixedStepMethod(dynamics),
        MACFlowControlTarget.bulk_velocity(exact_discrete_bulk, axes=(0,)),
        target_absolute_tolerance=1.0e-9,
        response_tolerance=1.0e-9,
        projection_tolerance=1.0e-8,
    ).prepare()

    result = prepared.step(prepared.initialize(0.0, initial), step_size=0.01)

    assert result.successful
    np.testing.assert_allclose(
        result.diagnostics.achieved, jnp.asarray((exact_discrete_bulk,)), atol=1.0e-9
    )
    assert result.diagnostics.resources.stage_map_evaluations == 3


def test_prescribed_gradient_matches_compiler_forcing():
    discretization, operators, momentum, dynamics, initial = _periodic()
    gradient = jnp.asarray((-0.3, 0.0))
    forcing = MACConstantPressureGradientForcing(operators, gradient, density=1.0)
    direct = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(
            2, 0.05, forcing=forcing, forcing_id=forcing.forcing_id
        ),
        momentum,
        dynamics.projection,
    )
    direct_step = phx.solver.SSPRK33FixedStepMethod(direct).step(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.05), None
    )
    controlled = MACFlowControlPlan(
        phx.solver.SSPRK33FixedStepMethod(dynamics),
        MACFlowControlTarget.prescribed_pressure_gradient(gradient),
        projection_tolerance=1.0e-9,
    ).prepare()

    result = controlled.step(controlled.initialize(0.0, initial), step_size=0.05)

    assert result.successful
    np.testing.assert_allclose(
        result.state.state, direct_step.accepted_state, atol=1.0e-10
    )
    np.testing.assert_allclose(result.diagnostics.control, gradient, atol=0.0)


def test_multi_axis_response_and_frozen_density_mass_flux():
    discretization, _, _, dynamics, initial = _periodic()
    method = phx.solver.SSPRK33FixedStepMethod(dynamics)
    bulk = MACFlowControlPlan(
        method,
        MACFlowControlTarget.bulk_velocity((0.03, -0.02), axes=(0, 1)),
        target_absolute_tolerance=1.0e-10,
        response_tolerance=1.0e-10,
        projection_tolerance=1.0e-9,
    ).prepare()
    bulk_result = bulk.step(bulk.initialize(0.0, initial), step_size=0.1)

    assert bulk_result.successful
    np.testing.assert_allclose(
        bulk_result.diagnostics.response_matrix,
        -0.1 * jnp.eye(2),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        bulk_result.diagnostics.achieved, (0.03, -0.02), atol=1e-10
    )

    density = 2.0 * jnp.ones(discretization.cell_shape)
    mass = MACFlowControlPlan(
        method,
        MACFlowControlTarget.frozen_density_mass_flux(
            ((0.04,), density), axes=(0,), density_id="frozen-rho-two"
        ),
        target_absolute_tolerance=1.0e-10,
        response_tolerance=1.0e-10,
        projection_tolerance=1.0e-9,
    ).prepare()
    mass_result = mass.step(mass.initialize(0.0, initial), step_size=0.1)

    assert mass_result.successful
    np.testing.assert_allclose(mass_result.diagnostics.achieved, (0.04,), atol=1e-10)
    np.testing.assert_allclose(
        mass_result.diagnostics.response_matrix, ((-0.1,),), atol=1e-10
    )
    assert mass.plan.target.kind == "frozen_density_mass_flux"
    with pytest.raises(ValueError, match="finite and positive"):
        MACFlowControlTarget.frozen_density_mass_flux(
            ((0.04,), density.at[0, 0].set(0.0)), axes=(0,)
        )


def test_singular_wall_normal_response_fails_and_rolls_back_for_retry():
    discretization, _, dynamics = _channel(count=4)
    initial = dynamics.project_state(
        tuple(jnp.zeros(layout.shape) for layout in discretization.face_layouts)
    )
    prepared = MACFlowControlPlan(
        phx.solver.SSPRK33FixedStepMethod(dynamics),
        MACFlowControlTarget.bulk_velocity(0.1, axes=(1,)),
        projection_tolerance=1.0e-8,
    ).prepare()
    state = prepared.initialize(0.0, initial)

    failed = prepared.step(state, step_size=0.01)
    retried = prepared.step(failed.state, step_size=0.005)

    assert not failed.successful
    assert not failed.diagnostics.conditioning.full_rank
    assert _all_dynamic_leaves_equal(failed.state, state)
    assert not retried.successful
    assert _all_dynamic_leaves_equal(retried.state, state)


def test_imex_euler_constant_density_bulk_control_executes_full_stage_map():
    _, _, _, dynamics, initial = _periodic()
    prepared = MACFlowControlPlan(
        phx.solver.MACIMEXEulerMethod(
            dynamics, fixed_step_size=0.02, solve_method="transform"
        ),
        MACFlowControlTarget.bulk_velocity(0.01, axes=(0,), density=1.5),
        target_absolute_tolerance=1.0e-10,
        response_tolerance=1.0e-10,
        projection_tolerance=1.0e-9,
    ).prepare()

    result = prepared.step(prepared.initialize(0.0, initial))

    assert result.successful
    np.testing.assert_allclose(result.diagnostics.achieved, (0.01,), atol=1.0e-10)
    np.testing.assert_allclose(
        result.diagnostics.response_matrix, ((-0.02 / 1.5,),), atol=1.0e-10
    )


def test_sbdf2_startup_restart_preserves_complete_control_history():
    _, _, _, dynamics, initial = _periodic()
    prepared = MACFlowControlPlan(
        phx.solver.MACSBDF2Method(
            dynamics, 0.01, solve_method="transform", tolerance=1.0e-10
        ),
        MACFlowControlTarget.bulk_velocity(0.02, axes=(0,)),
        target_absolute_tolerance=1.0e-8,
        response_tolerance=1.0e-8,
        projection_tolerance=1.0e-8,
    ).prepare()
    startup = prepared.step(prepared.initialize(0.0, initial))
    assert startup.successful
    assert not startup.state.startup_pending

    checkpoint = io.BytesIO()
    eqx.tree_serialise_leaves(checkpoint, startup.state)
    checkpoint.seek(0)
    restored = eqx.tree_deserialise_leaves(checkpoint, startup.state)
    uninterrupted = prepared.step(startup.state)
    restarted = prepared.step(restored)

    assert uninterrupted.successful
    assert restarted.successful
    assert uninterrupted.state.accepted_steps == 2
    assert _all_dynamic_leaves_equal(uninterrupted.state, restarted.state)


def test_prescribed_gradient_schedule_is_evaluated_at_ssprk_stages():
    _, _, _, dynamics, initial = _periodic()
    target = MACFlowControlTarget.prescribed_pressure_gradient(
        lambda time: jnp.asarray((-time,)),
        axes=(0,),
        schedule_id="linear-pressure-gradient-stage-schedule",
    )
    prepared = MACFlowControlPlan(
        phx.solver.SSPRK33FixedStepMethod(dynamics),
        target,
        projection_tolerance=1.0e-9,
    ).prepare()

    result = prepared.step(prepared.initialize(0.0, initial), step_size=0.1)

    assert result.successful
    np.testing.assert_allclose(result.diagnostics.control, (-0.1,), atol=1.0e-12)
    np.testing.assert_allclose(result.diagnostics.observed_flux, (0.005,), atol=1.0e-10)


def test_target_schedule_control_and_prepared_identities_change():
    _, _, _, dynamics, initial = _periodic()
    method = phx.solver.MACIMEXEulerMethod(
        dynamics, fixed_step_size=0.01, solve_method="transform"
    )
    first_target = MACFlowControlTarget.bulk_velocity(0.1, axes=(0,))
    second_target = MACFlowControlTarget.bulk_velocity(0.2, axes=(0,))
    first = MACFlowControlPlan(method, first_target).prepare()
    second = MACFlowControlPlan(method, second_target).prepare()
    scheduled = MACFlowControlTarget.bulk_velocity(
        lambda time: jnp.asarray((0.1 + time,)),
        axes=(0,),
        schedule_id="linear-bulk-schedule",
    )
    continuation = first.initialize(0.0, initial)

    assert first_target.target_id != second_target.target_id
    assert first.plan.plan_id != second.plan.plan_id
    assert first.prepared_id != second.prepared_id
    assert first.control_space_id == second.control_space_id
    assert continuation.control_id == first.control_space_id
    assert scheduled.target_id != first_target.target_id
    assert scheduled.schedule_id == "linear-bulk-schedule"
