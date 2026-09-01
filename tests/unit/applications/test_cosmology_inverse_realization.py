import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cosmology = phx.applications.cosmology


def _case(*, geometry_ad="piecewise"):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), jnp.asarray([0.5, 0.5]), ambient_dimension=1
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(
        grid,
        execution=phx.discretization.SplatExecutionPolicy(geometry_ad=geometry_ad),
    ).prepare(particles)
    target_positions = jnp.asarray([[0.25], [0.75]])
    target = transfer.deposit_content(
        transfer.build(target_positions), particles.masses
    ).density
    layout = phx.observation.CoordinateLayout(
        tuple(f"field:{index}" for index in range(target.size))
    )
    observation = phx.solver.FieldObservationPlan(
        lambda field, args: field,
        target,
        phx.observation.CholeskyCovarianceAction(jnp.eye(target.size), layout),
        observation_id="inverse-target",
    )
    plan = cosmology.ParticleFieldRealizationPlan(
        transfer,
        observation,
        target_kind="density",
        plan_id="inverse-test",
    )
    return plan, target_positions, target


def test_exact_particle_realization_has_zero_residual_and_conserves_mass():
    plan, target_positions, target = _case()
    result = plan.evaluate(target_positions)
    assert bool(result.successful)
    np.testing.assert_allclose(result.predicted_density, target, atol=1e-12)
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.mass_balance_defect, 0.0, atol=1e-12)
    assert result.support_complete
    assert result.captured_fraction_minimum == 1.0


def test_inverse_objective_gradient_and_sensitivity_are_finite():
    plan, _, _ = _case()
    positions = jnp.asarray([[0.2], [0.7]])
    value, gradient = plan.value_and_gradient(positions)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    direction = jnp.asarray([[0.1], [-0.1]])
    report = plan.sensitivity(positions, direction, epsilon=1e-5)
    assert bool(report.finite)
    assert report.jvp_residual < 1e-4


def test_periodic_parameterization_and_optimizer_descent():
    plan, _, _ = _case()
    wrapped = plan.positions(jnp.asarray([[1.2], [-0.3]]))
    np.testing.assert_allclose(wrapped, [[0.2], [0.7]])
    initial = jnp.asarray([[0.2], [0.7]])
    initial_value = plan.objective(initial)
    result = phx.optim.minimize(
        lambda positions, args: plan.objective(positions),
        initial,
        method=phx.optim.NonlinearConjugateGradient(),
        termination=phx.optim.OptimizationTermination(maximum_steps=12),
    )
    assert result.objective <= initial_value
    assert bool(plan.evaluate(result.parameters).successful)


def test_frozen_geometry_is_rejected():
    with pytest.raises(ValueError, match="piecewise"):
        _case(geometry_ad="frozen")
