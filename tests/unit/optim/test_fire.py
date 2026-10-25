#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import dataclasses
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


_ITERATING = int(phx.optim.OptimizationStatus.ITERATING)
_SUCCESS = int(phx.optim.OptimizationStatus.SUCCESS)
_NONFINITE = int(phx.optim.OptimizationStatus.NONFINITE_EVALUATION)


def _termination(*, steps: int = 5000, tolerance: float = 1e-9) -> Any:
    return phx.optim.OptimizationTermination(
        absolute_optimality=tolerance,
        relative_optimality=0.0,
        maximum_steps=steps,
    )


def _fast_fire2() -> Any:
    return phx.optim.FIRE2(
        initial_time_step=0.05,
        minimum_time_step=0.005,
        maximum_time_step=0.3,
        delay_steps=5,
        maximum_step=0.2,
    )


def _fire() -> Any:
    return phx.optim.FIRE(maximum_time_step=0.5)


def _force_driven_methods() -> list[Any]:
    return [
        _fire(),
        phx.optim.FIRE(maximum_time_step=0.5, downhill_check=True),
        _fast_fire2(),
        phx.optim.TrustScaledLBFGS(maximum_step=0.5),
    ]


def _stepwise(
    method: Any,
    state: Any,
    group_energy: Callable[[Any], Any],
    steps: int,
    /,
    *,
    termination: Any = None,
    bounds: Any = None,
    freeze: bool = True,
) -> Any:
    """Drive the public stepwise protocol for a fixed number of evaluations."""

    def body(_: Any, current: Any) -> Any:
        values = group_energy(current.parameters)
        gradient = jax.grad(lambda x: jnp.sum(group_energy(x)))(current.parameters)
        return method.step(
            current,
            values,
            gradient,
            bounds=bounds,
            termination=termination,
            active=(current.status == _ITERATING) if freeze else None,
        )

    return eqx.filter_jit(lambda initial: jax.lax.fori_loop(0, steps, body, initial))(
        state
    )


def _anisotropic(x: Any, center: Any) -> Any:
    return 0.5 * jnp.sum(jnp.array([1.0, 4.0, 9.0]) * (x - center) ** 2)


def _double_well(x: Any, _: Any) -> Any:
    return (x[0] ** 2 - 1.0) ** 2 + 2.0 * x[1] ** 2


@pytest.mark.parametrize("method", _force_driven_methods(), ids=lambda m: m.method_id)
def test_force_driven_methods_reach_anisotropic_quadratic_minimum(method: Any) -> None:
    center = jnp.array([0.5, -1.0, 2.0], dtype=jnp.float64)
    result = phx.optim.minimize(
        _anisotropic,
        jnp.array([3.0, 2.0, -1.0], dtype=jnp.float64),
        method=method,
        args=center,
        termination=_termination(),
    )

    assert int(result.status) == _SUCCESS
    np.testing.assert_allclose(result.parameters, center, atol=1e-8)
    # The reported value and gradient belong to the reported iterate.
    np.testing.assert_allclose(
        result.objective, _anisotropic(result.parameters, center), rtol=0, atol=1e-15
    )
    assert result.diagnostics.final_optimality_norm <= 1e-9
    evidence = result.method_evidence
    np.testing.assert_array_equal(evidence.accepted_parameters, result.parameters)
    assert int(evidence.status[0]) == _SUCCESS
    assert (
        int(result.diagnostics.objective_evaluations) == int(evidence.evaluations[0]) + 1
    )


# Both starts lie below the unit barrier at x = 0, inside one basin each.
@pytest.mark.parametrize(
    ("start", "minimum"),
    [([0.6, 0.3], [1.0, 0.0]), ([-0.5, -0.2], [-1.0, 0.0])],
    ids=["right-basin", "left-basin"],
)
@pytest.mark.parametrize("method", _force_driven_methods(), ids=lambda m: m.method_id)
def test_force_driven_methods_relax_nonconvex_double_well(
    method: Any, start: list[float], minimum: list[float]
) -> None:
    result = phx.optim.minimize(
        _double_well,
        jnp.array(start, dtype=jnp.float64),
        method=method,
        termination=_termination(tolerance=1e-8),
    )

    assert int(result.status) == _SUCCESS
    np.testing.assert_allclose(result.parameters, jnp.array(minimum), atol=1e-7)


@pytest.mark.strict_jax
def test_fire2_first_step_and_uphill_retreat_follow_the_controller_equations() -> None:
    dt, a0, increase, decrease, decay = 0.1, 0.2, 1.5, 0.5, 0.5
    method = phx.optim.FIRE2(
        initial_time_step=dt,
        minimum_time_step=0.01,
        maximum_time_step=1.0,
        time_step_increase=increase,
        time_step_decrease=decrease,
        initial_mixing=a0,
        mixing_decay=decay,
        delay_steps=0,
        maximum_step=10.0,
    )
    x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
    force = -x0  # objective 0.5 |x|^2
    state = method.step(
        method.init(x0),
        jnp.array([0.5 * jnp.sum(x0**2)]),
        x0,
    )

    # From rest the half kick is dt F, the power is positive and growth starts
    # after the zero-step delay; the mixed velocity equals the kick.
    grown = dt * increase
    np.testing.assert_allclose(state.parameters, x0 + grown * dt * force, rtol=1e-12)
    np.testing.assert_allclose(state.velocity, dt * force, rtol=1e-12)
    np.testing.assert_allclose(state.time_step, [grown])
    np.testing.assert_allclose(state.mixing, [a0 * decay])
    assert int(state.positive_steps[0]) == 1
    assert not bool(state.uphill[0])

    x1 = jnp.array([1.0, 0.0], dtype=jnp.float64)
    velocity = jnp.array([3.0, 0.0], dtype=jnp.float64)
    moving_away = dataclasses.replace(method.init(x1), velocity=velocity)
    retreated = method.step(moving_away, jnp.array([0.5]), x1)

    kicked = velocity + dt * (-x1)
    power = float(jnp.dot(kicked, -x1))
    assert power < 0.0
    shrunk = dt * decrease
    speed = float(jnp.linalg.norm(kicked) / jnp.linalg.norm(x1))
    mixed = (1.0 - a0) * velocity + ((1.0 - a0) * dt + a0 * speed) * (-x1)
    np.testing.assert_allclose(retreated.parameters, x1 - 0.5 * shrunk * mixed)
    np.testing.assert_array_equal(retreated.velocity, jnp.zeros(2))
    np.testing.assert_allclose(retreated.time_step, [shrunk])
    np.testing.assert_allclose(retreated.mixing, [a0])
    np.testing.assert_allclose(retreated.power, [power])
    assert bool(retreated.uphill[0]) and int(retreated.positive_steps[0]) == 0


@pytest.mark.strict_jax
def test_fire_caps_each_site_while_fire2_scales_the_whole_group() -> None:
    weights = jnp.array([[100.0], [1.0]], dtype=jnp.float64)

    def energy(x: Any) -> Any:
        return jnp.reshape(0.5 * jnp.sum(weights * x**2), (1,))

    x0 = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float64)
    gradient = weights * x0
    layout = phx.optim.OptimizerGroupLayout(jnp.array([0, 0]), group_count=1)
    cap, dt = 0.05, 0.1

    fire = phx.optim.FIRE(initial_time_step=dt, maximum_step=cap)
    fire_state = fire.step(fire.init(x0, layout=layout), energy(x0), gradient)
    # From rest the displacement is dt^2 F; only the stiff site exceeds the cap.
    np.testing.assert_allclose(
        fire_state.parameters,
        jnp.array([[1.0 - cap, 0.0, 0.0], [0.0, 1.0 - dt**2, 0.0]]),
        rtol=1e-12,
    )
    assert bool(fire_state.trust_limited[0])

    fire2 = phx.optim.FIRE2(
        initial_time_step=dt,
        minimum_time_step=0.01,
        maximum_time_step=1.0,
        delay_steps=10,
        maximum_step=cap,
    )
    fire2_state = fire2.step(fire2.init(x0, layout=layout), energy(x0), gradient)
    raw = dt * dt * (-gradient)
    factor = cap / float(jnp.max(jnp.linalg.norm(raw, axis=1)))
    np.testing.assert_allclose(fire2_state.parameters, x0 + factor * raw, rtol=1e-12)
    np.testing.assert_allclose(fire2_state.time_step, [dt * factor], rtol=1e-12)
    assert bool(fire2_state.trust_limited[0])


def test_fire_inverse_mass_scales_kicks_and_holds_massless_sites() -> None:
    x0 = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=jnp.float64)
    method = phx.optim.FIRE(initial_time_step=0.1, maximum_step=10.0)
    layout = phx.optim.OptimizerGroupLayout(jnp.array([0, 0, 0]), group_count=1)
    state = method.init(x0, layout=layout, inverse_mass=jnp.array([1.0, 0.25, 0.0]))
    moved = method.step(state, jnp.array([0.5 * jnp.sum(x0**2)]), x0)

    np.testing.assert_allclose(
        moved.parameters - x0,
        -0.01 * jnp.array([[1.0, 1.0], [0.25, 0.25], [0.0, 0.0]]),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="non-negative"):
        method.init(x0, layout=layout, inverse_mass=jnp.array([1.0, -1.0, 1.0]))


def test_fire_downhill_check_restores_the_accepted_iterate_and_shrinks_the_step() -> None:
    dt, decrease = 1.8, 0.5
    method = phx.optim.FIRE(
        initial_time_step=dt,
        minimum_time_step=0.01,
        maximum_time_step=2.0,
        time_step_decrease=decrease,
        maximum_step=10.0,
        downhill_check=True,
    )

    def energy(x: Any) -> Any:
        return jnp.reshape(0.5 * jnp.sum(x**2), (1,))

    x0 = jnp.array([1.0], dtype=jnp.float64)
    first = method.step(method.init(x0), energy(x0), x0)
    # From rest the power is zero: the uphill branch shrinks dt but the incoming
    # step moves the iterate, overshooting to a higher objective.
    np.testing.assert_allclose(first.parameters, x0 - dt**2)
    x1 = first.parameters
    assert float(energy(x1)[0]) > float(energy(x0)[0])

    second = method.step(first, energy(x1), x1)
    shrunk = dt * decrease * decrease
    assert bool(second.rolled_back[0]) and not bool(second.evaluation_accepted[0])
    assert int(second.rollbacks[0]) == 1
    np.testing.assert_array_equal(second.accepted_parameters, x0)
    np.testing.assert_allclose(second.accepted_value, energy(x0))
    np.testing.assert_allclose(second.time_step, [shrunk])
    np.testing.assert_allclose(second.parameters, x0 - shrunk**2)

    plain = phx.optim.FIRE(
        initial_time_step=dt,
        minimum_time_step=0.01,
        maximum_time_step=2.0,
        time_step_decrease=decrease,
        maximum_step=10.0,
    )
    accepted = plain.step(plain.step(plain.init(x0), energy(x0), x0), energy(x1), x1)
    assert bool(accepted.evaluation_accepted[0])
    np.testing.assert_array_equal(accepted.accepted_parameters, x1)


@pytest.mark.parametrize("method", _force_driven_methods(), ids=lambda m: m.method_id)
def test_force_driven_methods_reject_nonfinite_evaluations(method: Any) -> None:
    def objective(x: Any, _: Any) -> Any:
        # The minimizer at 3 lies in a region where the objective is undefined.
        return jnp.sum(jnp.where(x > 2.0, jnp.nan, (x - 3.0) ** 2))

    result = phx.optim.minimize(
        objective,
        jnp.array([0.0], dtype=jnp.float64),
        method=method,
        termination=_termination(),
    )

    assert int(result.status) == _NONFINITE
    assert bool(jnp.all(result.parameters <= 2.0))
    assert bool(jnp.isfinite(result.objective))
    np.testing.assert_allclose(result.objective, objective(result.parameters, None))
    evidence = result.method_evidence
    assert int(evidence.nonfinite_evaluations[0]) == 1
    # The pending point was returned to the accepted iterate.
    np.testing.assert_array_equal(evidence.parameters, evidence.accepted_parameters)


@pytest.mark.parametrize("method", _force_driven_methods(), ids=lambda m: m.method_id)
def test_force_driven_methods_stepwise_protocol_matches_solve(method: Any) -> None:
    center = jnp.array([0.5, -1.0, 2.0], dtype=jnp.float64)
    x0 = jnp.array([3.0, 2.0, -1.0], dtype=jnp.float64)
    termination = _termination(steps=3000, tolerance=1e-7)

    def stepwise(initial: Any) -> Any:
        state = method.init(initial, value_dtype=jnp.float64)

        def condition(current: Any) -> Any:
            return (current.status[0] == _ITERATING) & (
                current.evaluations[0] < termination.maximum_steps
            )

        def body(current: Any) -> Any:
            value, gradient = jax.value_and_grad(_anisotropic)(current.parameters, center)
            return method.step(
                current, jnp.reshape(value, (1,)), gradient, termination=termination
            )

        return jax.lax.while_loop(condition, body, state)

    state = eqx.filter_jit(stepwise)(x0)
    result = phx.optim.minimize(
        _anisotropic, x0, method=method, args=center, termination=termination
    )

    np.testing.assert_allclose(result.parameters, state.accepted_parameters, atol=1e-12)
    assert int(result.method_evidence.evaluations[0]) == int(state.evaluations[0])
    assert int(result.status) == int(state.status[0]) == _SUCCESS


@pytest.mark.parametrize("method", _force_driven_methods(), ids=lambda m: m.method_id)
def test_force_driven_groups_evolve_as_independent_systems(method: Any) -> None:
    stiffness = jnp.array([[1.0, 3.0], [20.0, 0.5]], dtype=jnp.float64)
    centers = jnp.array([[0.5, -0.5], [2.0, 1.0]], dtype=jnp.float64)
    x0 = jnp.array([[2.0, 1.0], [-1.0, 3.0], [7.0, 7.0]], dtype=jnp.float64)

    def group_energy(x: Any) -> Any:
        # Row 2 is a padded site: it belongs to no group and contributes nothing.
        return 0.5 * jnp.sum(stiffness * (x[:2] - centers) ** 2, axis=1)

    layout = phx.optim.OptimizerGroupLayout(jnp.array([0, 1, -1]), group_count=2)
    grouped = _stepwise(
        method, method.init(x0, layout=layout), group_energy, 30, freeze=False
    )

    for group in range(2):

        def single_energy(x: Any, group: int = group) -> Any:
            return jnp.reshape(
                0.5 * jnp.sum(stiffness[group] * (x - centers[group]) ** 2), (1,)
            )

        alone = _stepwise(method, method.init(x0[group]), single_energy, 30, freeze=False)
        np.testing.assert_allclose(
            grouped.parameters[group], alone.parameters, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            grouped.accepted_value[group], alone.accepted_value[0], rtol=1e-10
        )
    np.testing.assert_array_equal(grouped.parameters[2], x0[2])


@pytest.mark.parametrize("method", _force_driven_methods(), ids=lambda m: m.method_id)
def test_force_driven_groups_converge_and_freeze_individually(method: Any) -> None:
    stiffness = jnp.array([[1.0, 1.0], [50.0, 0.2]], dtype=jnp.float64)
    x0 = jnp.array([[1.0, -1.0], [1.0, -1.0]], dtype=jnp.float64)

    def group_energy(x: Any) -> Any:
        return 0.5 * jnp.sum(stiffness * x**2, axis=1)

    layout = phx.optim.OptimizerGroupLayout(jnp.array([0, 1]), group_count=2)
    state = _stepwise(
        method,
        method.init(x0, layout=layout),
        group_energy,
        4000,
        termination=_termination(tolerance=1e-8),
    )

    np.testing.assert_array_equal(state.status, jnp.array([_SUCCESS, _SUCCESS]))
    assert int(state.evaluations[0]) != int(state.evaluations[1])
    # A converged group stays at its accepted, converged evaluation.
    np.testing.assert_array_equal(state.parameters, state.accepted_parameters)
    assert bool(jnp.all(state.optimality <= 1e-8))
    # |x| <= optimality / smallest stiffness.
    np.testing.assert_allclose(state.accepted_parameters, jnp.zeros((2, 2)), atol=1e-7)


@pytest.mark.parametrize("method", _force_driven_methods(), ids=lambda m: m.method_id)
def test_force_driven_methods_admit_bounds_and_unbounded_domains(method: Any) -> None:
    target = jnp.array([2.0, -1.0, 0.5], dtype=jnp.float64)

    def objective(x: Any, _: Any) -> Any:
        return jnp.sum((x - target) ** 2)

    boxed = phx.optim.minimize(
        objective,
        jnp.array([0.2, 0.8, 0.9], dtype=jnp.float64),
        method=method,
        bounds=phx.optim.Bounds(0.0, 1.0),
        termination=_termination(tolerance=1e-8),
    )

    assert int(boxed.status) == _SUCCESS
    np.testing.assert_allclose(boxed.parameters, jnp.array([1.0, 0.0, 0.5]), atol=1e-8)
    assert boxed.diagnostics.primal_feasibility == 0.0
    assert int(boxed.diagnostics.active_constraints) == 2
    assert boxed.certificate is not None
    assert boxed.certificate.complementarity <= 1e-12

    unbounded = phx.optim.minimize(
        objective,
        jnp.array([0.2, 0.8, 0.9], dtype=jnp.float64),
        method=method,
        bounds=phx.optim.Bounds(),
        termination=_termination(tolerance=1e-8),
    )
    assert int(unbounded.status) == _SUCCESS
    np.testing.assert_allclose(unbounded.parameters, target, atol=1e-8)
    assert int(unbounded.diagnostics.active_constraints) == 0
    assert bool(jnp.isfinite(unbounded.diagnostics.complementarity))


def test_trust_scaled_lbfgs_caps_steps_and_accepts_uphill_evaluations() -> None:
    method = phx.optim.TrustScaledLBFGS(maximum_step=0.05)
    center = jnp.array([0.5, -1.0, 2.0], dtype=jnp.float64)

    def energy(x: Any) -> Any:
        return jnp.reshape(_anisotropic(x, center), (1,))

    state = _stepwise(
        method,
        method.init(jnp.array([3.0, 2.0, -1.0], dtype=jnp.float64)),
        energy,
        20,
    )
    assert int(state.trust_limited_steps[0]) == 20
    np.testing.assert_allclose(state.step_norm, [0.05], rtol=1e-12)

    # Without a line search the unit steepest step across a stiff valley is taken
    # even though it raises the objective.
    stiff = phx.optim.TrustScaledLBFGS(maximum_step=float("inf"))
    curvature = jnp.array([1.0, 100.0], dtype=jnp.float64)

    def valley(x: Any) -> Any:
        return jnp.reshape(0.5 * jnp.sum(curvature * x**2), (1,))

    x0 = jnp.array([0.0, 1e-3], dtype=jnp.float64)
    first = stiff.step(stiff.init(x0), valley(x0), curvature * x0)
    x1 = first.parameters
    assert float(valley(x1)[0]) > float(valley(x0)[0])
    second = stiff.step(first, valley(x1), curvature * x1)
    assert bool(second.evaluation_accepted[0])
    np.testing.assert_array_equal(second.accepted_parameters, x1)


def test_trust_scaled_lbfgs_rejects_negative_curvature_pairs() -> None:
    method = phx.optim.TrustScaledLBFGS(maximum_step=0.5)

    def energy(x: Any) -> Any:
        return jnp.reshape(jnp.sum(0.25 * x**4 - x**2), (1,))

    def gradient(x: Any) -> Any:
        return x**3 - 2.0 * x

    x0 = jnp.array([0.1], dtype=jnp.float64)
    first = method.step(method.init(x0), energy(x0), gradient(x0))
    np.testing.assert_allclose(first.parameters, [0.6])
    second = method.step(first, energy(first.parameters), gradient(first.parameters))
    # Between 0.1 and 0.6 the double well has negative curvature: s·y < 0.
    assert bool(second.curvature_rejected[0])
    assert int(second.history.count[0]) == 0
    assert not bool(second.direction_reset[0])

    result = phx.optim.minimize(
        lambda x, _: energy(x)[0],
        x0,
        method=method,
        termination=_termination(tolerance=1e-9),
    )
    assert int(result.status) == _SUCCESS
    np.testing.assert_allclose(result.parameters, [np.sqrt(2.0)], atol=1e-9)
    assert int(result.method_evidence.curvature_rejections[0]) >= 1


@pytest.mark.strict_jax
def test_optimizer_group_layout_reduces_sites_and_excludes_padding() -> None:
    values = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    layout = phx.optim.OptimizerGroupLayout(jnp.array([1, 0, 1, -1]), group_count=2)

    np.testing.assert_allclose(layout.squared_norm(values), [25.0, 66.0])
    np.testing.assert_allclose(layout.site_norm_max(values), [5.0, np.sqrt(61.0)])
    np.testing.assert_array_equal(
        layout.group_all_finite(values.at[3, 0].set(jnp.nan)), [True, True]
    )
    np.testing.assert_array_equal(
        layout.group_all_finite(values.at[2, 1].set(jnp.inf)), [True, False]
    )
    selected = layout.select(jnp.array([False, True]), -values, values)
    np.testing.assert_array_equal(
        selected, jnp.array([[-1.0, -2.0], [3.0, 4.0], [-5.0, -6.0], [7.0, 8.0]])
    )
    clamped, limited = layout.clamp_sites(values, 4.0)
    np.testing.assert_allclose(jnp.linalg.norm(clamped[:3], axis=1), [2.236068, 4.0, 4.0])
    np.testing.assert_array_equal(clamped[3], values[3])
    np.testing.assert_array_equal(limited, [True, True])

    with pytest.raises(ValueError, match="group_count"):
        phx.optim.OptimizerGroupLayout(jnp.array([0, 2]), group_count=2)
    with pytest.raises(ValueError, match="prefix"):
        phx.optim.OptimizerGroupLayout(jnp.array([0, 1, 0]), group_count=2).validate(
            values
        )


@pytest.mark.strict_jax
def test_optimizer_group_layout_completes_sharded_reductions() -> None:
    sites = jnp.arange(24.0).reshape(2, 4, 3)
    full = phx.optim.OptimizerGroupLayout(
        jnp.array([0, 1, 1, 0, 0, 1, 1, 0]), group_count=2
    )
    shard = phx.optim.OptimizerGroupLayout(
        jnp.array([0, 1, 1, 0]), group_count=2, collective_axis="shards"
    )
    flat = sites.reshape(8, 3)

    inner = jax.vmap(lambda block: shard.inner(block, block), axis_name="shards")(sites)
    largest = jax.vmap(shard.site_norm_max, axis_name="shards")(sites)

    for index in range(2):
        np.testing.assert_allclose(inner[index], full.inner(flat, flat))
        np.testing.assert_allclose(largest[index], full.site_norm_max(flat))


@pytest.mark.strict_jax
def test_optimizer_group_layout_partitions_multi_leaf_trees_with_empty_groups() -> None:
    tree = {
        "cell": jnp.array([[2.0, 0.0], [0.0, 3.0]]),
        "coordinates": jnp.arange(12.0).reshape(4, 3),
    }
    # Coordinate rows are sites (row 2 is padding); the whole cell is one site of
    # group 1, and group 2 owns no site.
    layout = phx.optim.OptimizerGroupLayout(
        {"cell": jnp.array(1), "coordinates": jnp.array([0, 1, -1, 1])},
        group_count=3,
    )
    layout.validate(tree)

    squared = layout.squared_norm(tree)
    np.testing.assert_allclose(squared, [5.0, 50.0 + 302.0 + 13.0, 0.0])
    np.testing.assert_allclose(
        layout.site_norm_max(tree), [np.sqrt(5.0), np.sqrt(302.0), 0.0]
    )
    np.testing.assert_array_equal(
        layout.count_true(jax.tree.map(lambda leaf: leaf > 4.0, tree)), [0, 4, 0]
    )
    staged = eqx.filter_jit(lambda layout_, tree_: layout_.squared_norm(tree_))
    np.testing.assert_allclose(staged(layout, tree), squared)

    scaled = layout.scale(jnp.array([2.0, -1.0, 5.0]), tree)
    coordinates = tree["coordinates"]
    np.testing.assert_array_equal(scaled["cell"], -tree["cell"])
    np.testing.assert_array_equal(
        scaled["coordinates"],
        jnp.stack((2.0 * coordinates[0], -coordinates[1], jnp.zeros(3), -coordinates[3])),
    )
    clamped, limited = layout.clamp_sites(tree, 4.0)
    np.testing.assert_array_equal(limited, [False, True, False])
    np.testing.assert_array_equal(clamped["cell"], tree["cell"])
    np.testing.assert_array_equal(clamped["coordinates"][2], coordinates[2])
    np.testing.assert_allclose(
        jnp.linalg.norm(clamped["coordinates"][jnp.array([1, 3])], axis=1), [4.0, 4.0]
    )

    with pytest.raises(ValueError, match="structure"):
        layout.validate(tree["coordinates"])
