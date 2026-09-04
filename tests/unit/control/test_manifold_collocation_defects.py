import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.control._advanced_collocation import (
    manifold_radau_collocation_defects,
    radau_collocation_defects,
)
from phydrax.metrix import (
    AbstractStateGeometry,
    EmbeddedStateGeometry,
    EuclideanStateGeometry,
    SpecialOrthogonalStateGeometry,
)
from phydrax.solver._radau_iia import RadauIIAMethod


_CONFIGURATION = "retraction"
_TANGENT = "shared-local"


def test_euclidean_manifold_defects_agree_with_existing_radau_evaluator():
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

    def dynamics(time, state, control, args):
        del args
        return (1.0 + time) * state + control[0]

    expected = radau_collocation_defects(
        method, dynamics, times, states, rates, controls
    )
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


def test_euclidean_implicit_dae_residual_uses_stage_rates():
    method = RadauIIAMethod(2)
    times = jnp.asarray([0.0, 1.0])
    states = jnp.asarray([[0.0], [1.0]])
    rates = jnp.ones((1, method.stage_count, 1))
    controls = jnp.zeros((1, 1))

    def dae(time, state, state_rate, control, args):
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


@pytest.mark.parametrize("dimension", [2, 3])
def test_constant_lie_algebra_velocity_preserves_so_stages_and_endpoint(dimension):
    method = RadauIIAMethod(3)
    geometry = SpecialOrthogonalStateGeometry(dimension, tolerance=2.0e-5)
    generator = jnp.zeros((dimension, dimension))
    generator = generator.at[0, 1].set(-0.2)
    generator = generator.at[1, 0].set(0.2)
    anchor = jnp.eye(dimension)
    endpoint = geometry.retract(anchor, generator)
    states = jnp.stack((anchor, endpoint))
    rates = jnp.broadcast_to(
        generator, (1, method.stage_count, dimension, dimension)
    )

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


@pytest.mark.parametrize("implicit", [False, True])
def test_noncommuting_so3_chart_trajectory_uses_anchored_differential(implicit):
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
        method.c[:, None, None] * first
        + method.c[:, None, None] ** 2 * second
    )
    stage_coordinate_rates = first + 2.0 * method.c[:, None, None] * second
    rates = stage_coordinate_rates[None, ...]
    endpoint = geometry.retract(anchor, first + second)
    states = jnp.stack((anchor, endpoint))

    def physical_rate(time):
        coordinates = time * first + time**2 * second
        coordinate_rate = first + 2.0 * time * second
        _, rate = jax.jvp(
            lambda local: geometry.retract(anchor, local),
            (coordinates,),
            (coordinate_rate,),
        )
        return rate

    physical_stage_rates = jax.vmap(physical_rate)(method.c)
    recovered_coordinate_rates = jax.vmap(
        lambda local, rate: geometry.pullback(anchor, local, rate)
    )(stage_coordinates, physical_stage_rates)
    assert jnp.allclose(
        recovered_coordinate_rates,
        stage_coordinate_rates,
        atol=2.0e-5,
    )

    if implicit:

        def dynamics(time, state, state_rate, control, args):
            del state, control, args
            return state_rate - physical_rate(time)

    else:

        def dynamics(time, state, control, args):
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


class _InvalidInverseGeometry(AbstractStateGeometry):
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_pullback: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self):
        self.geometry_id = "geometry:test-invalid-inverse"
        self.retraction_method = "invalid-addition"
        self.trivial = False
        self.supports_exact_pullback = True
        self.supports_commutator_free = True

    def contains(self, state, /):
        return jnp.all(jnp.isfinite(jnp.asarray(state)))

    def project_tangent(self, state, vector, /):
        del state
        return jnp.asarray(vector)

    def to_local(self, state, tangent, /):
        del state
        return jnp.asarray(tangent)

    def from_local(self, state, local_tangent, /):
        del state
        return jnp.asarray(local_tangent)

    def retract(self, state, local_tangent, /):
        return jnp.asarray(state) + jnp.asarray(local_tangent)

    def inverse_retract(self, state, point, /):
        del state
        return jnp.zeros_like(jnp.asarray(point))

    def pullback(self, state, local_tangent, tangent, /):
        del state, local_tangent
        return jnp.asarray(tangent)


class _LinearDifferentialGeometry(AbstractStateGeometry):
    pullback_scale: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_pullback: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, *, exact: bool, pullback_scale: float = 1.0):
        self.pullback_scale = pullback_scale
        self.geometry_id = "geometry:test-linear-differential"
        self.retraction_method = "addition"
        self.trivial = False
        self.supports_exact_pullback = exact
        self.supports_commutator_free = True

    def contains(self, state, /):
        return jnp.all(jnp.isfinite(jnp.asarray(state)))

    def project_tangent(self, state, vector, /):
        del state
        return jnp.asarray(vector)

    def to_local(self, state, tangent, /):
        del state, tangent
        raise AssertionError("manifold collocation must not call to_local")

    def from_local(self, state, local_tangent, /):
        del state, local_tangent
        raise AssertionError("manifold collocation must not call from_local")

    def retract(self, state, local_tangent, /):
        return jnp.asarray(state) + jnp.asarray(local_tangent)

    def inverse_retract(self, state, point, /):
        return jnp.asarray(point) - jnp.asarray(state)

    def pullback(self, state, local_tangent, tangent, /):
        del state, local_tangent
        return self.pullback_scale * jnp.asarray(tangent)


def test_unavailable_or_inconsistent_chart_differential_fails_evidence():
    method = RadauIIAMethod(2)
    times = jnp.asarray([0.0, 1.0])
    states = jnp.asarray([[0.0], [1.0]])
    rates = jnp.ones((1, method.stage_count, 1))
    controls = jnp.zeros((1, 1))

    def unavailable_dynamics(time, state, control, args):
        del time, state, control, args
        raise AssertionError("dynamics must not run without exact differential")

    unavailable = manifold_radau_collocation_defects(
        method,
        _LinearDifferentialGeometry(exact=False),
        unavailable_dynamics,
        times,
        states,
        rates,
        controls,
        configuration_convention=_CONFIGURATION,
        tangent_convention=_TANGENT,
    )
    assert jnp.all(~jnp.isfinite(unavailable.stage_defects))
    assert not unavailable.finite
    assert not unavailable.chart_valid
    assert not unavailable.equation_valid
    assert not unavailable.valid

    inconsistent = manifold_radau_collocation_defects(
        method,
        _LinearDifferentialGeometry(exact=True, pullback_scale=2.0),
        lambda time, state, control, args: jnp.ones_like(state),
        times,
        states,
        rates,
        controls,
        configuration_convention=_CONFIGURATION,
        tangent_convention=_TANGENT,
    )
    assert inconsistent.finite
    assert not inconsistent.chart_valid
    assert not inconsistent.equation_valid
    assert not inconsistent.valid


def test_invalid_chart_and_nonfinite_stage_fail_typed_evidence():
    method = RadauIIAMethod(2)
    times = jnp.asarray([0.0, 1.0])
    states = jnp.asarray([[0.0], [1.0]])
    controls = jnp.zeros((1, 1))
    rates = jnp.zeros((1, method.stage_count, 1))

    invalid_chart = manifold_radau_collocation_defects(
        method,
        _InvalidInverseGeometry(),
        lambda time, state, control, args: jnp.zeros_like(state),
        times,
        states,
        rates,
        controls,
        configuration_convention=_CONFIGURATION,
        tangent_convention=_TANGENT,
    )
    assert invalid_chart.finite
    assert jnp.all(invalid_chart.contained)
    assert not invalid_chart.chart_valid
    assert invalid_chart.equation_valid
    assert not invalid_chart.valid

    nonfinite_stage = manifold_radau_collocation_defects(
        method,
        EuclideanStateGeometry(),
        lambda time, state, control, args: jnp.zeros_like(state),
        times,
        states,
        rates.at[0, 0, 0].set(jnp.nan),
        controls,
        configuration_convention=_CONFIGURATION,
        tangent_convention=_TANGENT,
    )
    assert not nonfinite_stage.finite
    assert not jnp.all(nonfinite_stage.contained)
    assert not nonfinite_stage.chart_valid
    assert not nonfinite_stage.equation_valid
    assert not nonfinite_stage.valid


def test_exact_chart_differential_suffices_and_rate_shape_remains_equal():
    method = RadauIIAMethod(2)
    exact_embedded = EmbeddedStateGeometry(
        membership=lambda state: jnp.asarray(True),
        tangent_projection=lambda state, vector: vector,
        retraction=lambda state, local: state + local,
        inverse_retraction=lambda state, point: point - state,
        retraction_pullback=lambda state, local, tangent: tangent,
        geometry_id="geometry:test-exact-collocation",
        retraction_method="addition",
    )
    common = (
        method,
        exact_embedded,
        lambda time, state, control, args: jnp.zeros_like(state),
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.0], [0.0]]),
        jnp.zeros((1, method.stage_count, 1)),
        jnp.zeros((1, 1)),
    )
    result = manifold_radau_collocation_defects(
        *common,
        configuration_convention=_CONFIGURATION,
        tangent_convention=_TANGENT,
    )
    assert exact_embedded.supports_exact_pullback
    assert not exact_embedded.supports_commutator_free
    assert result.valid

    with pytest.raises(ValueError, match="stage_local_rates must have shape"):
        manifold_radau_collocation_defects(
            method,
            EuclideanStateGeometry(),
            common[2],
            common[3],
            common[4],
            jnp.zeros((1, 1, 1)),
            common[6],
            configuration_convention=_CONFIGURATION,
            tangent_convention=_TANGENT,
        )
