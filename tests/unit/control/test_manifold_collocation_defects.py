from typing import Any

import jax
import jax.numpy as jnp

from phydrax.control._advanced_collocation import (
    manifold_radau_collocation_defects,
    radau_collocation_defects,
)
from phydrax.metrix import (
    EmbeddedStateGeometry,
    EuclideanStateGeometry,
    QuaternionPoseStateGeometry,
    SpecialOrthogonalStateGeometry,
)
from phydrax.solver._radau_iia import RadauIIAMethod


_CONFIGURATION = "retraction"
_TANGENT = "shared-local"


def test_euclidean_manifold_defects_agree_with_existing_radau_evaluator() -> None:
    method = RadauIIAMethod(3)
    geometry = EuclideanStateGeometry()
    times = jnp.asarray([0.0, 0.4, 1.0])
    states = jnp.asarray([[0.2, -0.3], [0.8, 0.5], [1.1, -0.1]])
    rates = jnp.asarray(
        [
            [[0.4, -0.2], [0.7, 0.3], [-0.1, 0.9]],
            [[0.2, 0.1], [-0.3, 0.6], [0.8, -0.4]],
        ]
    )
    controls = jnp.asarray([[0.25], [-0.5]])

    def dynamics(time: Any, state: Any, control: Any, args: Any) -> Any:
        del args
        return (1.0 + time) * state + control[0]

    expected = radau_collocation_defects(method, dynamics, times, states, rates, controls)
    actual = manifold_radau_collocation_defects(
        method,
        geometry,
        dynamics,
        times,
        states,
        rates,
        controls,
        configuration_convention=_CONFIGURATION,
        tangent_convention=_TANGENT,
    )

    assert jnp.allclose(actual.stage_times, expected.stage_times)
    assert jnp.allclose(actual.stage_states, expected.stage_states)
    assert jnp.allclose(actual.stage_defects, expected.stage_defects)
    assert jnp.allclose(actual.endpoint_defects, expected.endpoint_defects)
    assert actual.finite == expected.finite
    assert actual.configuration_convention == _CONFIGURATION
    assert actual.tangent_convention == _TANGENT


def test_euclidean_implicit_dae_residual_uses_stage_rates() -> None:
    method = RadauIIAMethod(2)
    times = jnp.asarray([0.0, 1.0])
    states = jnp.asarray([[0.0], [1.0]])
    rates = jnp.ones((1, method.stage_count, 1))
    controls = jnp.zeros((1, 1))

    def dae(time: Any, state: Any, state_rate: Any, control: Any, args: Any) -> Any:
        del time, control, args
        return state_rate - jnp.ones_like(state)

    expected = radau_collocation_defects(
        method, dae, times, states, rates, controls, implicit=True
    )

    result = manifold_radau_collocation_defects(
        method,
        EuclideanStateGeometry(),
        dae,
        times,
        states,
        rates,
        controls,
        configuration_convention=_CONFIGURATION,
        tangent_convention=_TANGENT,
        implicit=True,
    )

    assert jnp.allclose(result.stage_defects, expected.stage_defects)
    assert jnp.allclose(result.endpoint_defects, expected.endpoint_defects)
    assert result.equation_valid
    assert result.valid


def test_manifold_collocation_defects_scenario_1() -> None:
    for dimension in [2, 3]:
        method = RadauIIAMethod(3)
        geometry = SpecialOrthogonalStateGeometry(dimension, tolerance=2.0e-5)
        generator = jnp.zeros((dimension, dimension))
        generator = generator.at[0, 1].set(-0.2)
        generator = generator.at[1, 0].set(0.2)
        anchor = jnp.eye(dimension)
        endpoint = geometry.retract(anchor, generator)
        states = jnp.stack((anchor, endpoint))
        rates = jnp.broadcast_to(generator, (1, method.stage_count, dimension, dimension))

        result = manifold_radau_collocation_defects(
            method,
            geometry,
            lambda time, state, control, args: state @ generator,
            jnp.asarray([0.0, 1.0]),
            states,
            rates,
            jnp.zeros((1, 1)),
            configuration_convention=_CONFIGURATION,
            tangent_convention=_TANGENT,
            equation_tolerance=2.0e-5,
        )

        stage_membership = jnp.asarray(
            [geometry.contains(stage) for stage in result.stage_states[0]]
        )
        assert jnp.all(stage_membership)
        assert jnp.allclose(result.stage_defects, 0.0, atol=2.0e-6)
        assert jnp.allclose(result.endpoint_defects, 0.0, atol=2.0e-6)
        assert jnp.all(result.contained)
        assert result.chart_valid
        assert result.equation_valid
        assert result.valid
    method = RadauIIAMethod(2)
    geometry = QuaternionPoseStateGeometry()
    pose = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.2, -0.4, 0.7])
    equivalent = pose.at[:4].multiply(-1.0)
    rates = jnp.zeros((1, method.stage_count, 6))

    result = manifold_radau_collocation_defects(
        method,
        geometry,
        lambda time, state, control, args: jnp.zeros_like(state),
        jnp.asarray([0.0, 1.0]),
        jnp.stack((pose, equivalent)),
        rates,
        jnp.zeros((1, 1)),
        configuration_convention=_CONFIGURATION,
        tangent_convention=_TANGENT,
    )

    assert result.stage_states.shape == (1, method.stage_count, 7)
    assert result.stage_defects.shape == (1, method.stage_count, 6)
    assert result.endpoint_defects.shape == (1, 6)
    assert jnp.allclose(result.stage_defects, 0.0)
    assert jnp.allclose(result.endpoint_defects, 0.0)
    assert result.valid
    method = RadauIIAMethod(2)

    def unavailable_dynamics(time: Any, state: Any, control: Any, args: Any) -> None:
        del time, state, control, args
        raise AssertionError("dynamics must not run without an exact differential")

    result = manifold_radau_collocation_defects(
        method,
        _additive_embedded_geometry(exact_differential=False),
        # ty: ignore[invalid-argument-type]
        unavailable_dynamics,
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.0], [1.0]]),
        jnp.ones((1, method.stage_count, 1)),
        jnp.zeros((1, 1)),
        configuration_convention=_CONFIGURATION,
        tangent_convention=_TANGENT,
    )

    assert result.stage_states.shape == (1, method.stage_count, 1)
    assert result.stage_defects.shape == (1, method.stage_count, 1)
    assert jnp.all(~jnp.isfinite(result.stage_defects))
    assert not result.finite
    assert not result.chart_valid
    assert not result.equation_valid
    assert not result.valid


def test_noncommuting_so3_chart_trajectory_uses_anchored_differential() -> None:
    for implicit in [False, True]:
        method = RadauIIAMethod(2)
        geometry = SpecialOrthogonalStateGeometry(3, tolerance=2.0e-5)
        anchor = jnp.eye(3)
        first = jnp.asarray(
            [
                [0.0, -0.3, 0.0],
                [0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        second = jnp.asarray(
            [
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0],
                [-0.2, 0.0, 0.0],
            ]
        )
        assert jnp.linalg.norm(first @ second - second @ first) > 0.0

        stage_coordinates = (
            method.c[:, None, None] * first + method.c[:, None, None] ** 2 * second
        )
        stage_coordinate_rates = first + 2.0 * method.c[:, None, None] * second
        rates = stage_coordinate_rates[None, ...]
        endpoint = geometry.retract(anchor, first + second)
        states = jnp.stack((anchor, endpoint))

        def physical_rate(time: Any) -> Any:
            coordinates = time * first + time**2 * second
            coordinate_rate = first + 2.0 * time * second
            rate = geometry.retraction_jvp(anchor, coordinates, coordinate_rate)
            return rate

        physical_stage_rates = jax.vmap(physical_rate)(method.c)
        stage_states = jax.vmap(lambda local: geometry.retract(anchor, local))(
            stage_coordinates
        )
        recovered_coordinate_rates = jax.vmap(
            lambda point, rate: geometry.retraction_inverse_jvp(anchor, point, rate)
        )(stage_states, physical_stage_rates)
        assert jnp.allclose(
            recovered_coordinate_rates,
            stage_coordinate_rates,
            atol=2.0e-5,
        )

        if implicit:

            def dynamics(
                time: Any, state: Any, state_rate: Any, control: Any, args: Any
            ) -> Any:
                del state, control, args
                return state_rate - physical_rate(time)

        else:

            def dynamics(time: Any, state: Any, control: Any, args: Any) -> Any:
                del state, control, args
                return physical_rate(time)

        result = manifold_radau_collocation_defects(
            method,
            geometry,
            dynamics,
            jnp.asarray([0.0, 1.0]),
            states,
            rates,
            jnp.zeros((1, 1)),
            configuration_convention=_CONFIGURATION,
            tangent_convention=_TANGENT,
            implicit=implicit,
            chart_tolerance=2.0e-5,
            equation_tolerance=2.0e-5,
        )

        expected_stages = jax.vmap(lambda local: geometry.retract(anchor, local))(
            stage_coordinates
        )
        assert jnp.allclose(result.stage_states[0], expected_stages, atol=2.0e-6)
        assert jnp.allclose(result.stage_defects, 0.0, atol=2.0e-5)
        assert jnp.allclose(result.endpoint_defects, 0.0, atol=2.0e-6)
        assert result.chart_valid
        assert result.equation_valid
        assert result.valid


def _additive_embedded_geometry(
    *,
    exact_differential: Any,
    inverse_jvp_scale: Any = 1.0,
) -> Any:
    differential_actions = (
        {
            "retraction_jvp_action": (lambda state, local, velocity: velocity),
            "retraction_inverse_jvp_action": (
                lambda state, point, tangent: inverse_jvp_scale * tangent
            ),
            "retraction_vjp_action": (lambda state, local, cotangent: cotangent),
        }
        if exact_differential
        else {}
    )
    return EmbeddedStateGeometry(
        membership=lambda state: jnp.all(jnp.isfinite(state)),
        tangent_projection=lambda state, vector: vector,
        retraction=lambda state, local: state + local,
        inverse_retraction=lambda state, point: point - state,
        geometry_id=(
            "geometry:test-additive-exact"
            if exact_differential
            else "geometry:test-additive-no-differential"
        ),
        retraction_method="addition",
        **differential_actions,
    )


def test_manifold_collocation_defects_scenario_2() -> None:
    method = RadauIIAMethod(2)
    result = manifold_radau_collocation_defects(
        method,
        _additive_embedded_geometry(
            exact_differential=True,
            inverse_jvp_scale=2.0,
        ),
        lambda time, state, control, args: jnp.ones_like(state),
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.0], [1.0]]),
        jnp.ones((1, method.stage_count, 1)),
        jnp.zeros((1, 1)),
        configuration_convention=_CONFIGURATION,
        tangent_convention=_TANGENT,
    )

    assert result.finite
    assert not result.chart_valid
    assert not result.equation_valid
    assert not result.valid
    method = RadauIIAMethod(2)
    rates = jnp.zeros((1, method.stage_count, 1)).at[0, 0, 0].set(jnp.nan)
    result = manifold_radau_collocation_defects(
        method,
        EuclideanStateGeometry(),
        lambda time, state, control, args: jnp.zeros_like(state),
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.0], [0.0]]),
        rates,
        jnp.zeros((1, 1)),
        configuration_convention=_CONFIGURATION,
        tangent_convention=_TANGENT,
    )

    assert not result.finite
    assert not jnp.all(result.contained)
    assert not result.chart_valid
    assert not result.equation_valid
    assert not result.valid
