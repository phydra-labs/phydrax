#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.equations._les_closures import (
    LESParameterProvenance,
    ResolvedLESFilter,
    SmagorinskyLESPlan,
)
from phydrax.equations._mac_incompressible import compile_mac_incompressible_flow
from phydrax.equations._mac_les import MACAlgebraicLESPlan
from phydrax.linalg import DenseLU, LinearSolvePolicy, TolerancePolicy
from phydrax.solver._mac_stage_inverse_general import MACVariableViscosityStagePlan
from phydrax.solver._mac_viscous import MACSBDF2GStabilityLedger


def _linear_policy():
    return LinearSolvePolicy(
        DenseLU(),
        tolerance=TolerancePolicy(relative=2.0e-8, absolute=2.0e-8, max_steps=40),
    )


def _grid(*, count=2, boundary_provider=None):
    if boundary_provider is None:
        specs = tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(3)
        )
        boundary_class = "periodic"
    else:
        specs = (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count),
        )
        boundary_class = "wall-bounded"
    grid = phx.discretization.TensorGridPlan(specs, axis_names=("x", "y", "z")).prepare(
        jnp.asarray([[0.0, 0.0, 0.0], [2.0 * jnp.pi] * 3])
    )
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    if boundary_provider is None:
        momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    else:
        boundaries = phx.discretization.MACBoundaryPlan(
            operators,
            (
                phx.discretization.MACBoundarySide(
                    "z", "lower", "free-slip", provider=boundary_provider
                ),
                phx.discretization.MACBoundarySide(
                    "z", "upper", "free-slip", provider=boundary_provider
                ),
            ),
        ).prepare()
        momentum = phx.discretization.MACMomentumPlan(
            operators, boundaries=boundaries
        ).prepare()
    return discretization, operators, momentum, boundary_class


def _les(discretization, coefficient, boundary_class):
    resolved_filter = ResolvedLESFilter(
        "mac-cell-volume",
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class=boundary_class,
        scale_rule="volume-equivalent",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    provenance = LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id,
        "incompressible-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    return MACAlgebraicLESPlan(SmagorinskyLESPlan(coefficient).prepare(provenance))


def _compiled(*, coefficient=None, count=2, viscosity=0.01, provider=None):
    discretization, operators, momentum, boundary_class = _grid(
        count=count, boundary_provider=provider
    )
    projection = phx.solver.MACPressureProjectionPlan(
        operators,
        boundaries=momentum.boundaries,
        solve_method="transform",
        tolerance=2.0e-8,
    )
    dynamics = compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, viscosity),
        momentum,
        projection,
        algebraic_les=(
            None
            if coefficient is None
            else _les(discretization, coefficient, boundary_class)
        ),
    )
    return discretization, operators, dynamics


def _taylor_green(discretization):
    x_faces, y_faces, z_faces = discretization.face_centers
    return (
        jnp.sin(x_faces[..., 1]),
        jnp.zeros(y_faces.shape[:-1], dtype=y_faces.dtype),
        jnp.zeros(z_faces.shape[:-1], dtype=z_faces.dtype),
    )


def _method(dynamics, step=1.0e-3):
    return phx.solver.MACIMEXEulerMethod(
        dynamics,
        fixed_step_size=step,
        solve_method="iterative",
        tolerance=2.0e-8,
        maximum_iterations=40,
        linear_policy=_linear_policy(),
    )


def test_zero_coefficient_imex_and_sbdf2_select_exact_constant_profiles():
    discretization, _, base = _compiled(coefficient=None)
    _, _, zero = _compiled(coefficient=0.0)
    state = base.pack_velocity(_taylor_green(discretization))

    base_imex = phx.solver.MACIMEXEulerMethod(
        base, fixed_step_size=1.0e-3, solve_method="transform"
    )
    zero_imex = phx.solver.MACIMEXEulerMethod(
        zero, fixed_step_size=1.0e-3, solve_method="transform"
    )
    base_step = base_imex.step(0.0, state)
    zero_step = zero_imex.step(0.0, state)

    assert not zero_imex.implicit_les
    assert zero_imex.temporal_profile == "mac-constant-laplacian-imex-euler"
    np.testing.assert_array_equal(zero_step.state, base_step.state)
    np.testing.assert_array_equal(zero_step.pressure, base_step.pressure)

    base_sbdf = phx.solver.MACSBDF2Method(base, 1.0e-3, solve_method="transform")
    zero_sbdf = phx.solver.MACSBDF2Method(zero, 1.0e-3, solve_method="transform")
    base_startup = base_sbdf.initialize(0.0, state)
    zero_startup = zero_sbdf.initialize(0.0, state)
    np.testing.assert_array_equal(zero_startup.history.state, base_startup.history.state)
    np.testing.assert_array_equal(
        zero_startup.history.explicit_rate, base_startup.history.explicit_rate
    )


def test_frozen_imex_uses_one_inverse_for_predictor_and_composite_projection():
    discretization, operators, dynamics = _compiled(coefficient=0.12)
    state = dynamics.pack_velocity(_taylor_green(discretization))

    result = _method(dynamics).step(0.0, state)

    assert result.accepted
    assert result.helmholtz is None
    assert result.les_stage is not None
    assert result.stage_inverse is not None
    assert result.predictor_inverse_id == result.projection_inverse_id
    assert result.stage_inverse.operator_id == result.predictor_inverse_id
    assert result.les_stage.prepared_id == dynamics.algebraic_les.prepared_id
    assert result.coefficient_refresh == "accepted-state-once-per-attempt"
    assert jnp.max(jnp.abs(operators.divergence(result.velocity))) < 2.0e-7


def test_frozen_sbdf2_restart_extrapolation_and_g_stability_identity():
    discretization, _, dynamics = _compiled(coefficient=0.12)
    state = dynamics.pack_velocity(_taylor_green(discretization))
    method = phx.solver.MACSBDF2Method(
        dynamics,
        1.0e-3,
        solve_method="iterative",
        tolerance=2.0e-8,
        maximum_iterations=40,
        linear_policy=_linear_policy(),
    )
    startup = method.initialize(0.0, state)
    restarted = method.step(startup.history)
    first = method.step(startup.history)

    assert startup.accepted
    assert first.accepted
    assert first.coefficient_projection is not None
    assert first.coefficient_time == first.attempted_time
    assert first.predictor_inverse_id == first.projection_inverse_id
    assert isinstance(first.g_stability, MACSBDF2GStabilityLedger)
    assert first.g_stability.successful
    assert first.g_stability.temporal_dissipation >= 0.0
    np.testing.assert_allclose(first.history.state, restarted.history.state)
    np.testing.assert_allclose(first.history.pressure, restarted.history.pressure)
    assert method.coefficient_extrapolation == "projected-2*u[n]-u[n-1]-at-t[n+1]"
    assert method.capabilities.method_id == method.method_id
    assert method.capabilities.order == 2
    assert not method.capabilities.adaptive
    assert not method.allows_adaptive_step


def test_manufactured_variable_viscosity_refresh_has_declared_temporal_orders():
    discretization, _, dynamics = _compiled(coefficient=0.12, count=4)
    base = _taylor_green(discretization)
    stage = dynamics.boundary_stage(1.0)

    def viscosity(amplitude):
        result = dynamics.algebraic_les.evaluate(
            tuple(amplitude * value for value in base), stage
        )
        return result.model_result.kinematic_viscosity

    exact = viscosity(np.exp(1.0))
    errors_imex = []
    errors_sbdf = []
    for step in (0.08, 0.04, 0.02):
        current = np.exp(1.0 - step)
        previous = np.exp(1.0 - 2.0 * step)
        errors_imex.append(float(jnp.linalg.norm(viscosity(current) - exact)))
        errors_sbdf.append(
            float(jnp.linalg.norm(viscosity(2.0 * current - previous) - exact))
        )

    assert errors_imex[0] / errors_imex[1] > 1.8
    assert errors_imex[1] / errors_imex[2] > 1.8
    assert errors_sbdf[0] / errors_sbdf[1] > 3.4
    assert errors_sbdf[1] / errors_sbdf[2] > 3.4


def test_frozen_imex_preserves_affine_boundary_data_and_rolls_back_failure():
    provider = phx.discretization.MACBoundaryProvider(0.125)
    discretization, _, dynamics = _compiled(coefficient=0.12, provider=provider)
    velocity = tuple(
        jnp.zeros(layout.shape, dtype=dynamics.momentum.operators.pressure_space.dtype)
        for layout in discretization.face_layouts
    )
    velocity = velocity[:2] + (jnp.full_like(velocity[2], 0.125),)
    state = dynamics.pack_velocity(velocity)

    result = _method(dynamics).step(0.0, state)

    assert result.accepted
    np.testing.assert_allclose(result.velocity[2][..., 0], 0.125)
    np.testing.assert_allclose(result.velocity[2][..., -1], 0.125)
    assert result.stage_inverse.boundary_defect < 1.0e-12

    def failed_boundary(time, coordinates, args):
        del coordinates, args
        value = jnp.where(time > 0.0, jnp.nan, 0.125)
        return value, jnp.asarray(0.0)

    failing_provider = phx.discretization.MACBoundaryProvider(
        function=failed_boundary, provider_id="failed-frozen-les-boundary"
    )
    failed_discretization, _, failed_dynamics = _compiled(
        coefficient=0.12, provider=failing_provider
    )
    failed_velocity = tuple(
        jnp.zeros(
            layout.shape,
            dtype=failed_dynamics.momentum.operators.pressure_space.dtype,
        )
        for layout in failed_discretization.face_layouts
    )
    failed_velocity = failed_velocity[:2] + (jnp.full_like(failed_velocity[2], 0.125),)
    failed_state = failed_dynamics.pack_velocity(failed_velocity)
    failed = _method(failed_dynamics).step(0.0, failed_state)

    assert not failed.accepted
    np.testing.assert_array_equal(failed.state, failed_state)
    assert failed.time == 0.0


def test_failed_sbdf2_attempt_retains_complete_restart_history_atomically():
    def boundary(time, coordinates, args):
        del coordinates, args
        value = jnp.where(time > 1.5e-3, jnp.nan, 0.125)
        return value, jnp.asarray(0.0)

    provider = phx.discretization.MACBoundaryProvider(
        function=boundary, provider_id="failed-sbdf2-frozen-les-boundary"
    )
    discretization, _, dynamics = _compiled(coefficient=0.12, provider=provider)
    velocity = tuple(
        jnp.zeros(layout.shape, dtype=dynamics.momentum.operators.pressure_space.dtype)
        for layout in discretization.face_layouts
    )
    velocity = velocity[:2] + (jnp.full_like(velocity[2], 0.125),)
    state = dynamics.pack_velocity(velocity)
    method = phx.solver.MACSBDF2Method(
        dynamics,
        1.0e-3,
        solve_method="iterative",
        tolerance=2.0e-8,
        maximum_iterations=40,
        linear_policy=_linear_policy(),
    )
    startup = method.initialize(0.0, state)

    failed = method.step(startup.history)

    assert startup.accepted
    assert not failed.accepted
    assert failed.history.valid
    assert failed.history.status == startup.history.status
    assert failed.history.accepted_steps == startup.history.accepted_steps
    np.testing.assert_array_equal(failed.history.time, startup.history.time)
    np.testing.assert_array_equal(failed.history.state, startup.history.state)
    np.testing.assert_array_equal(
        failed.history.previous_state, startup.history.previous_state
    )
    for failed_rate, startup_rate in zip(
        failed.history.explicit_rate,
        startup.history.explicit_rate,
        strict=True,
    ):
        np.testing.assert_array_equal(failed_rate, startup_rate)
    for failed_rate, startup_rate in zip(
        failed.history.previous_explicit_rate,
        startup.history.previous_explicit_rate,
        strict=True,
    ):
        np.testing.assert_array_equal(failed_rate, startup_rate)
    np.testing.assert_array_equal(failed.history.pressure, startup.history.pressure)


def test_mismatched_variational_action_and_unsupported_routes_are_rejected():
    discretization, _, dynamics = _compiled(coefficient=0.12)
    other_discretization, _, other = _compiled(coefficient=0.12, count=3)
    assert discretization.prepared_id != other_discretization.prepared_id
    viscosity = jnp.zeros(discretization.cell_shape)
    density = tuple(jnp.ones_like(value) for value in _taylor_green(discretization))

    with pytest.raises(ValueError, match="action and momentum IDs differ"):
        MACVariableViscosityStagePlan(
            dynamics.momentum,
            density,
            viscosity,
            1.0e-3,
            viscosity_action=other.algebraic_les.viscosity_action,
            stage_id="mismatched-action",
        )
    with pytest.raises(ValueError, match="only the iterative"):
        phx.solver.MACIMEXEulerMethod(
            dynamics, fixed_step_size=1.0e-3, solve_method="transform"
        )


def test_sbdf2_is_explicitly_rejected_by_adaptive_rollout():
    discretization, _, dynamics = _compiled(coefficient=0.0)
    state = dynamics.pack_velocity(_taylor_green(discretization))
    del state
    method = phx.solver.MACSBDF2Method(dynamics, 1.0e-3, solve_method="transform")

    with pytest.raises(ValueError, match="fixed-step"):
        phx.solver.MACAdaptiveRolloutPlan(
            dynamics,
            method,
            None,
            None,
            final_time=0.01,
            initial_step_size=1.0e-3,
        )
