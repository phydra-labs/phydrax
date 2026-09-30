from collections.abc import Callable
from dataclasses import dataclass
from math import factorial
from typing import Any, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from phydrax._strict import StrictModule
from phydrax.domain import (
    CallbackDerivativeRule,
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
    DomainFunction,
    HyperRectangle,
)
from phydrax.domain._derivative import DerivativeRuleProvider
from phydrax.metrix import CoordinateChart, RiemannianMetric
from phydrax.operators import (
    domain_codifferential,
    domain_exterior_derivative,
    domain_hodge_laplacian,
    domain_hodge_star,
    domain_interior_product,
    domain_lie_derivative,
    domain_pullback_form,
    domain_to_twisted,
    domain_wedge,
    DomainDifferentialForm,
    partial_n,
)


@eqx.filter_jit
def _compiled_value(function: Callable[[Array], Array], point: Array) -> Array:
    return function(point)


@final
class _StoppedPolynomial(StrictModule, DerivativeRuleProvider):
    amplitude: Array
    powers: tuple[tuple[int, int], ...] = eqx.field(static=True)
    factors: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        amplitude: Array,
        powers: tuple[tuple[int, int], ...],
        factors: tuple[float, ...],
        /,
    ) -> None:
        self.amplitude = amplitude
        self.powers = powers
        self.factors = factors

    def __call__(self, x: Array, /, *, key: Any = None, **kwargs: Any) -> Array:
        del key, kwargs
        values = jnp.stack(
            [
                factor * x[0] ** powers[0] * x[1] ** powers[1]
                for factor, powers in zip(self.factors, self.powers, strict=True)
            ]
        )
        return self.amplitude * jax.lax.stop_gradient(values)

    def derivative_rule_for(self, function: DomainFunction, /) -> DerivativeRule:
        return _PolynomialDerivativeRule(function)


@dataclass(frozen=True, slots=True, eq=False)
class _PolynomialDerivativeRule(DerivativeRule):
    function: DomainFunction

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        del mode, basis, periodic
        if backend not in ("ad", "jet"):
            return None
        if var != "x" or axis not in (0, 1):
            raise ValueError("Polynomial rule requires a plane coordinate axis.")
        source = self.function.func
        if not isinstance(source, _StoppedPolynomial):
            raise TypeError("Polynomial rule requires its native evaluator.")
        powers: list[tuple[int, int]] = []
        factors: list[float] = []
        for factor, power in zip(source.factors, source.powers, strict=True):
            if power[axis] < order:
                factor = 0.0
            else:
                factor *= factorial(power[axis]) / factorial(power[axis] - order)
            next_power = list(power)
            next_power[axis] = max(0, next_power[axis] - order)
            powers.append((next_power[0], next_power[1]))
            factors.append(factor)
        return DomainFunction(
            domain=self.function.domain,
            deps=self.function.deps,
            func=_StoppedPolynomial(source.amplitude, tuple(powers), tuple(factors)),
        )


def _setup() -> tuple[HyperRectangle, CoordinateChart, RiemannianMetric]:
    domain = HyperRectangle(jnp.array([-2.0, -2.0]), jnp.array([2.0, 2.0]), label="x")
    chart = CoordinateChart("analytic_domain_forms", ("x", "y"))
    return domain, chart, RiemannianMetric(lambda q: jnp.eye(2), chart=chart)


def _field(
    geometry: HyperRectangle,
    amplitude: Array,
    powers: tuple[tuple[int, int], ...],
    factors: tuple[float, ...],
    /,
) -> DomainFunction:
    domain = geometry.Function("x")(jnp.zeros((1,))).domain
    return DomainFunction(
        domain=domain,
        deps=("x",),
        func=_StoppedPolynomial(amplitude, powers, factors),
    )


@pytest.mark.parametrize("backend", ["ad", "jet"])
@pytest.mark.parametrize("mode", ["forward", "reverse"])
def test_hodge_codifferential_and_laplacian_preserve_analytic_coordinate_rules(
    backend: Literal["ad", "jet"],
    mode: Literal["forward", "reverse"],
) -> None:
    geometry, chart, metric = _setup()
    amplitude = jnp.asarray(1.3)
    scalar = DomainDifferentialForm(
        _field(geometry, amplitude, ((3, 2),), (1.0,)),
        chart=chart,
        degree=0,
    )
    covector = DomainDifferentialForm(
        _field(geometry, amplitude, ((2, 0), (0, 3)), (1.0, 1.0)),
        chart=chart,
        degree=1,
    )
    point = jnp.array([0.4, 0.7])
    star_d = domain_exterior_derivative(
        domain_hodge_star(covector, metric), mode=mode, backend=backend
    )
    delta = domain_codifferential(covector, metric, mode=mode, backend=backend)
    laplacian = domain_hodge_laplacian(scalar, metric, mode=mode, backend=backend)
    assert jnp.allclose(
        _compiled_value(star_d.coefficients.func, point),
        jnp.array([1.3 * (0.8 + 1.47)]),
    )
    assert jnp.allclose(
        _compiled_value(delta.coefficients.func, point),
        jnp.array([-1.3 * (0.8 + 1.47)]),
    )
    assert jnp.allclose(
        _compiled_value(laplacian.coefficients.func, point),
        jnp.array([-1.3 * (6.0 * 0.4 * 0.7**2 + 2.0 * 0.4**3)]),
    )
    second = partial_n(
        domain_to_twisted(scalar, -1).coefficients,
        var="x",
        axis=0,
        order=2,
        backend=backend,
        mode=mode,
    )
    assert jnp.allclose(second.func(point), jnp.array([-1.3 * 6.0 * 0.4 * 0.7**2]))


def test_wedge_twist_interior_and_lie_compose_both_operands_analytic_rules() -> None:
    geometry, chart, _ = _setup()
    scalar = DomainDifferentialForm(
        _field(geometry, jnp.asarray(1.3), ((2, 0),), (1.0,)),
        chart=chart,
        degree=0,
    )
    covector = DomainDifferentialForm(
        _field(geometry, jnp.asarray(0.6), ((0, 0), (0, 1)), (0.0, 1.0)),
        chart=chart,
        degree=1,
    )
    product = domain_wedge(domain_to_twisted(scalar, -1), covector)
    derivative = domain_exterior_derivative(product)
    point = jnp.array([0.4, 0.7])
    assert derivative.twist == "twisted"
    assert jnp.allclose(
        derivative.coefficients.func(point), jnp.array([-2.0 * 1.3 * 0.6 * 0.4 * 0.7])
    )
    vector = _field(geometry, jnp.asarray(0.8), ((1, 0), (0, 0)), (1.0, 0.0))
    covector = DomainDifferentialForm(
        _field(geometry, jnp.asarray(1.3), ((2, 0), (0, 3)), (1.0, 1.0)),
        chart=chart,
        degree=1,
    )
    contraction = domain_interior_product(vector, covector)
    contraction_d = domain_exterior_derivative(contraction)
    lie = domain_lie_derivative(vector, covector)
    expected = jnp.array([3.0 * 0.8 * 1.3 * 0.4**2, 0.0])
    assert jnp.allclose(contraction_d.coefficients.func(point), expected)
    assert jnp.allclose(lie.coefficients.func(point), expected)


def test_pullback_chain_rule_preserves_analytic_form_and_mapping_rules() -> None:
    geometry, chart, _ = _setup()
    scalar = DomainDifferentialForm(
        _field(geometry, jnp.asarray(1.3), ((3, 2),), (1.0,)),
        chart=chart,
        degree=0,
    )
    mapping = _field(geometry, jnp.asarray(1.0), ((2, 0), (0, 1)), (1.0, 1.0))
    source_chart = CoordinateChart("analytic_pullback", ("u", "v"))
    pullback = domain_pullback_form(scalar, mapping, source_chart=source_chart)
    derivative = domain_exterior_derivative(pullback)
    point = jnp.array([0.4, 0.7])
    expected = jnp.array([6.0 * 1.3 * 0.4**5 * 0.7**2, 2.0 * 1.3 * 0.4**6 * 0.7])
    assert jnp.allclose(_compiled_value(derivative.coefficients.func, point), expected)
    after = domain_pullback_form(
        domain_exterior_derivative(scalar), mapping, source_chart=source_chart
    )
    assert jnp.allclose(after.coefficients.func(point), expected)


def test_composed_analytic_rules_keep_live_coefficient_parameter_gradients() -> None:
    geometry, chart, metric = _setup()
    point = jnp.array([0.4, 0.7])

    def value(amplitude: Array) -> Array:
        form = DomainDifferentialForm(
            _field(geometry, amplitude, ((3, 2),), (1.0,)),
            chart=chart,
            degree=0,
        )
        laplacian = domain_hodge_laplacian(form, metric)
        return laplacian.coefficients.func(point)[0]

    derivative = jax.jit(jax.grad(value))(jnp.asarray(1.3))
    assert jnp.allclose(derivative, -(6.0 * 0.4 * 0.7**2 + 2.0 * 0.4**3))
    assert jnp.allclose(jax.jit(value)(jnp.asarray(2.6)), 2.6 * derivative)


@pytest.mark.parametrize("backend", ["ad", "jet"])
def test_laplacian_requests_complete_order_from_original_analytic_callback(
    backend: Literal["ad", "jet"],
) -> None:
    geometry, chart, metric = _setup()

    def derive(**request: Any) -> DomainFunction:
        order = request["order"]
        axis = request["axis"]
        if axis == 1 or order > 3:
            return geometry.Function("x")(lambda x: jnp.zeros((1,)))
        factor = factorial(3) / factorial(3 - order)
        return geometry.Function("x")(
            lambda x: jax.lax.stop_gradient(jnp.array([factor * x[0] ** (3 - order)]))
        )

    field = geometry.Function("x")(
        lambda x: jax.lax.stop_gradient(jnp.array([x[0] ** 3]))
    ).with_derivative_rule(CallbackDerivativeRule(derive))
    form = DomainDifferentialForm(field, chart=chart, degree=0)
    laplacian = domain_hodge_laplacian(form, metric, backend=backend)
    assert jnp.allclose(
        _compiled_value(laplacian.coefficients.func, jnp.array([0.4, 0.7])),
        jnp.array([-2.4]),
    )
