from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from phydrax.rheology import (
    GeneralizedNewtonianLaw,
    MuILaw,
    MuIPackingFractionLaw,
    MuIParameters,
)


pytestmark = pytest.mark.strict_jax


def _parameters(*, barker: bool = False) -> MuIParameters:
    values = tuple(
        jnp.asarray(x, dtype=jnp.float64) for x in (0.383, 0.643, 0.279, 0.001, 2500.0)
    )
    return MuIParameters(
        *values,
        barker_alpha=1.9 if barker else None,
        barker_mu_infinity=0.05 if barker else None,
    )


def _neutral_condition(inertial: float) -> float:
    mu = 0.383 + 0.26 * inertial / (0.279 + inertial)
    derivative = 0.26 * 0.279 / (0.279 + inertial) ** 2
    r = inertial * derivative / mu
    return 4 * r * r - 4 * r + mu * mu * (1 - r / 2) ** 2


def test_jop_values_limits_status_and_packing() -> None:
    law = MuILaw(packing_fraction_law=MuIPackingFractionLaw(0.6, 0.2))
    rate = jnp.asarray([0.0, 20.0, 100.0, 1e9], dtype=jnp.float64)
    pressure = jnp.asarray(100.0, dtype=jnp.float64)
    evaluation = jax.jit(law.evaluate)(rate, pressure, _parameters())
    inertial = np.asarray(rate) * 0.001 * 5
    reference = 0.383 + 0.26 * inertial / (0.279 + inertial)
    np.testing.assert_allclose(evaluation.inertial_number, inertial, rtol=1e-14)
    np.testing.assert_allclose(evaluation.friction_coefficient, reference, rtol=1e-14)
    np.testing.assert_allclose(
        evaluation.shear_stress_magnitude_pa, 100 * reference, rtol=1e-14
    )
    np.testing.assert_allclose(
        evaluation.apparent_viscosity_pa_s[1:],
        100 * reference[1:] / np.asarray(rate)[1:],
        rtol=1e-14,
    )
    np.testing.assert_allclose(evaluation.packing_fraction, 0.6 - 0.2 * inertial)
    assert not evaluation.viscosity_defined[0]
    assert not evaluation.successful[0]
    assert jnp.isinf(evaluation.apparent_viscosity_pa_s[0])
    assert abs(float(evaluation.friction_coefficient[-1]) - 0.643) < 2e-8
    separation = law.evaluate(
        0.0, jnp.asarray([0.0, -100.0], dtype=jnp.float64), _parameters()
    )
    assert jnp.all(separation.separated & separation.successful)
    np.testing.assert_array_equal(separation.shear_stress_magnitude_pa, np.zeros(2))
    np.testing.assert_array_equal(separation.apparent_viscosity_pa_s, np.zeros(2))
    assert not jnp.any(separation.hadamard_applicable)


def test_jop_hand_derivatives_and_parameter_leaves() -> None:
    law, parameters = MuILaw(), _parameters()
    assert len(jax.tree_util.tree_leaves(parameters)) == 5
    rate, pressure = (
        jnp.asarray(20.0, dtype=jnp.float64),
        jnp.asarray(100.0, dtype=jnp.float64),
    )

    def friction(rate_: jax.Array, pressure_: jax.Array, p: MuIParameters) -> jax.Array:
        return law.evaluate(rate_, pressure_, p).friction_coefficient

    dr = jax.jit(jax.grad(friction, argnums=0))(rate, pressure, parameters)
    dp = jax.grad(friction, argnums=1)(rate, pressure, parameters)
    inertial, derivative = 0.1, 0.26 * 0.279 / (0.279 + 0.1) ** 2
    np.testing.assert_allclose(dr, derivative * 0.005, rtol=1e-13)
    np.testing.assert_allclose(dp, -derivative * inertial / 200.0, rtol=1e-13)
    gradient = eqx.filter_grad(lambda p: friction(rate, pressure, p))(parameters)
    np.testing.assert_allclose(gradient.mu_s, 0.279 / 0.379, rtol=1e-13)
    np.testing.assert_allclose(gradient.mu_2, 0.1 / 0.379, rtol=1e-13)
    np.testing.assert_allclose(gradient.I_0, -0.26 * 0.1 / 0.379**2, rtol=1e-13)
    np.testing.assert_allclose(gradient.grain_diameter_m, derivative * 100.0, rtol=1e-13)
    np.testing.assert_allclose(
        gradient.grain_density_kg_m3, derivative * inertial / 5000.0, rtol=1e-13
    )
    rates = jnp.asarray([2.0, 20.0, 200.0], dtype=jnp.float64)
    batched = jax.jit(jax.vmap(lambda r: law.evaluate(r, pressure, parameters)))(rates)
    assert jnp.all(batched.successful)
    np.testing.assert_allclose(
        batched.friction_coefficient,
        0.383 + 0.26 * np.asarray(rates) * 0.005 / (0.279 + np.asarray(rates) * 0.005),
    )


def test_barker_gray_exact_two_branch_equations_and_ad() -> None:
    parameters = _parameters(barker=True)
    assert len(jax.tree_util.tree_leaves(parameters)) == 7
    law = MuILaw(regularization="barker-gray-2017")
    neutral = brentq(_neutral_condition, 1e-8, 0.1, xtol=1e-15)
    inertial = np.asarray([neutral / 100, neutral, 0.1, 1.0, 20.0])
    upper = (0.383 * 0.279 + 0.643 * inertial + 0.05 * inertial**2) / (0.279 + inertial)
    match = (0.383 * 0.279 + 0.643 * neutral + 0.05 * neutral**2) / (0.279 + neutral)
    low = np.sqrt(1.9 / (np.log(neutral / inertial) + 1.9 / match**2))
    reference = np.where(inertial <= neutral, low, upper)
    rate = jnp.asarray(inertial / 0.005, dtype=jnp.float64)
    evaluation = jax.jit(law.evaluate)(rate, 100.0, parameters)
    assert evaluation.neutral_point_valid
    np.testing.assert_allclose(
        evaluation.lower_neutral_inertial_number, neutral, rtol=2e-10
    )
    np.testing.assert_allclose(evaluation.friction_coefficient, reference, rtol=2e-10)
    slope_upper = (0.26 * 0.279 + 0.05 * inertial * (2 * 0.279 + inertial)) / (
        0.279 + inertial
    ) ** 2
    slope_low = reference / (2 * inertial * (np.log(neutral / inertial) + 1.9 / match**2))
    slope = np.where(inertial <= neutral, slope_low, slope_upper)
    r = inertial * slope / reference
    np.testing.assert_allclose(
        evaluation.hadamard_margin,
        -(4 * r * r - 4 * r + reference**2 * (1 - r / 2) ** 2),
        rtol=1e-8,
        atol=1e-10,
    )

    # Coefficient differentiation includes the neutral root's implicit derivative.
    def value(p: MuIParameters) -> jax.Array:
        return law.evaluate(0.01, 100.0, p).friction_coefficient

    gradient = eqx.filter_grad(value)(parameters)

    def coefficients(p: MuIParameters) -> tuple[jax.Array | None, ...]:
        return (
            p.mu_s,
            p.mu_2,
            p.I_0,
            p.grain_diameter_m,
            p.grain_density_kg_m3,
            p.barker_alpha,
            p.barker_mu_infinity,
        )

    for index, (coefficient, derivative) in enumerate(
        zip(coefficients(parameters), coefficients(gradient), strict=True)
    ):
        if coefficient is None or derivative is None:
            raise RuntimeError("Barker coefficients are required by this test.")
        h = abs(float(coefficient)) * 1e-5

        def select(p: MuIParameters) -> jax.Array | None:
            return coefficients(p)[index]

        plus = eqx.tree_at(select, parameters, coefficient + h)
        minus = eqx.tree_at(select, parameters, coefficient - h)
        reference_derivative = (float(value(plus)) - float(value(minus))) / (2 * h)
        np.testing.assert_allclose(
            derivative, reference_derivative, rtol=2e-5, atol=2e-10
        )
    assert not law.evaluate(0.0, 100.0, parameters).viscosity_defined
    assert law.evaluate(0.0, 100.0, parameters).friction_coefficient == 0


@pytest.mark.parametrize(
    "kind,n,yield_stress,regularization,expected,defined",
    [
        ("power-law", 1.0, 0.0, 0.0, 2.0, True),
        ("power-law", 0.5, 0.0, 0.0, np.inf, False),
        ("power-law", 1.5, 0.0, 0.0, 0.0, True),
        ("bingham", 1.0, 3.0, 0.0, np.inf, False),
        ("herschel-bulkley", 0.5, 3.0, 0.0, np.inf, False),
        ("bingham", 1.0, 3.0, 10.0, 32.0, True),
        ("herschel-bulkley", 0.5, 3.0, 10.0, np.inf, False),
        ("carreau-yasuda", 0.5, 0.0, 0.0, 7.0, True),
    ],
)
def test_generalized_newtonian_zero_rate(
    kind: str,
    n: float,
    yield_stress: float,
    regularization: float,
    expected: float,
    defined: bool,
) -> None:
    from phydrax.rheology import GeneralizedNewtonianKind
    from phydrax.typing import parse

    law = GeneralizedNewtonianLaw(
        parse(kind, GeneralizedNewtonianKind, "kind"),
        7.0,
        consistency_pa_s_n=2.0,
        power_index=n,
        yield_stress_pa=yield_stress,
        regularization_s=regularization,
    )
    result = law.viscosity(jnp.asarray(0.0, dtype=jnp.float64))
    assert result.viscosity_defined == defined
    assert result.successful == defined
    np.testing.assert_equal(np.asarray(result.viscosity_pa_s), expected)


def test_regularized_bingham_zero_limit_derivative_and_tiny_rate() -> None:
    law = GeneralizedNewtonianLaw(
        "bingham", 1.0, consistency_pa_s_n=2.0, yield_stress_pa=3.0, regularization_s=10.0
    )
    derivative = jax.jit(jax.grad(lambda r: law.viscosity(r).viscosity_pa_s))(
        jnp.asarray(0.0, dtype=jnp.float64)
    )
    np.testing.assert_allclose(derivative, -150.0, rtol=1e-14)
    values = law.viscosity(jnp.asarray([0.0, 1e-100, 0.2], dtype=jnp.float64))
    np.testing.assert_allclose(
        values.viscosity_pa_s,
        [32.0, 32.0, 2.0 + 3.0 * (-np.expm1(-2.0)) / 0.2],
        rtol=1e-14,
    )
    singular = GeneralizedNewtonianLaw("power-law", 1.0, power_index=0.5)
    assert singular.viscosity(1e-100).viscosity_pa_s > 1e49


def test_missing_barker_coefficients_and_invalid_selector() -> None:
    with pytest.raises(ValueError, match="explicit"):
        MuILaw(regularization="barker-gray-2017").evaluate(1.0, 100.0, _parameters())
    with pytest.raises(ValueError):
        MuILaw(regularization="unknown")  # ty: ignore[invalid-argument-type]


def test_no_neutral_interval_reports_failure() -> None:
    parameters = MuIParameters(
        0.383, 0.39, 0.279, 0.001, 2500.0, barker_alpha=1.9, barker_mu_infinity=0.05
    )
    result = MuILaw(regularization="barker-gray-2017").evaluate(1.0, 100.0, parameters)
    assert not result.neutral_point_valid
    assert not result.successful
