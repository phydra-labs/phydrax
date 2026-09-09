import jax.numpy as jnp

from phydrax._strict import StrictModule
from phydrax.finance.contracts._options import OptionType, VanillaPayoff
from phydrax.finance.models._diffusion import (
    BachelierModel,
    Black76Model,
    BlackScholesModel,
    HestonModel,
)
from phydrax.finance.valuation._analytic import (
    evaluate_bachelier_european,
    evaluate_black76_european,
    evaluate_black_scholes_european,
    invert_bachelier_implied_volatility,
    invert_black76_implied_volatility,
    invert_black_scholes_implied_volatility,
)
from phydrax.finance.valuation._fourier import (
    evaluate_heston_cos,
    evaluate_heston_fourier,
    heston_log_price_characteristic_function,
    HestonCOSPlan,
    HestonFourierPlan,
)
from phydrax.finance.valuation._hedging import evaluate_hedge_replay
from phydrax.finance.valuation._lattice import (
    evaluate_lattice,
    LatticePlan,
    LatticeProblem,
)
from phydrax.finance.valuation._pde import evaluate_pde, FiniteDifferencePlan, PDEProblem
from phydrax.finance.valuation._sensitivity import (
    evaluate_aad_greeks,
    evaluate_bump_greeks,
    GreekRequest,
)


def test_black_scholes_benchmark_parity_and_all_implied_volatility_inversions():
    call = evaluate_black_scholes_european(
        BlackScholesModel(0.2), 100.0, 100.0, 1.0, 0.05
    )
    put = evaluate_black_scholes_european(
        BlackScholesModel(0.2),
        100.0,
        100.0,
        1.0,
        0.05,
        option_type=OptionType.PUT,
    )
    assert jnp.allclose(call.value, 10.450583572, rtol=2e-6)
    assert jnp.allclose(call.value - put.value, 100.0 - 100.0 * jnp.exp(-0.05), rtol=2e-6)
    recovered = invert_black_scholes_implied_volatility(
        call.value, 100.0, 100.0, 1.0, 0.05
    )
    assert jnp.allclose(recovered.volatility, 0.2, rtol=2e-5)

    black = evaluate_black76_european(Black76Model(0.35), 95.0, 100.0, 2.0, 0.9)
    recovered_black = invert_black76_implied_volatility(
        black.value, 95.0, 100.0, 2.0, 0.9
    )
    assert jnp.allclose(recovered_black.volatility, 0.35, rtol=2e-5)

    normal = evaluate_bachelier_european(BachelierModel(8.0), -2.0, 1.0, 1.5, 0.95)
    recovered_normal = invert_bachelier_implied_volatility(
        normal.value, -2.0, 1.0, 1.5, 0.95
    )
    assert jnp.allclose(recovered_normal.volatility, 8.0, rtol=2e-5)


def test_lattice_refinement_and_pde_cross_engine_agree_with_analytic_value():
    model = BlackScholesModel(0.2)
    payoff = VanillaPayoff(100.0, OptionType.PUT)
    problem = LatticeProblem(model, payoff, 100.0, 1.0, 0.05)
    analytic = evaluate_black_scholes_european(
        model, 100.0, 100.0, 1.0, 0.05, option_type=OptionType.PUT
    ).value
    coarse = evaluate_lattice(problem, LatticePlan(50)).value
    fine = evaluate_lattice(problem, LatticePlan(400)).value
    assert jnp.abs(fine - analytic) < jnp.abs(coarse - analytic)

    pde_problem = PDEProblem(model, payoff, 100.0, 1.0, 0.05)
    pde = evaluate_pde(
        pde_problem,
        FiniteDifferencePlan(
            space_steps=200,
            time_steps=200,
            spot_maximum=400.0,
        ),
    )
    assert jnp.allclose(pde.value, analytic, atol=8e-2)


def test_heston_characteristic_function_and_transform_engines_agree():
    model = HestonModel(2.0, 0.04, 0.2, -0.5, 0.04)
    assert jnp.allclose(
        heston_log_price_characteristic_function(model, 0.0, 100.0, 1.0, 0.03, 0.01),
        1.0,
    )
    fourier = evaluate_heston_fourier(
        model,
        100.0,
        100.0,
        1.0,
        0.03,
        HestonFourierPlan(num_nodes=2048),
        dividend_yield=0.01,
    )
    cos = evaluate_heston_cos(
        model,
        100.0,
        100.0,
        1.0,
        0.03,
        HestonCOSPlan(num_terms=256),
        dividend_yield=0.01,
    )
    assert jnp.allclose(fourier.value, cos.value, atol=8e-2)


class _QuadraticValuation(StrictModule):
    def __call__(self, parameters):
        return parameters[0] ** 2 + 3.0 * parameters[1]


def test_aad_and_bump_greeks_defend_first_and_second_order_contracts():
    request = GreekRequest(
        ("spot", "rate"),
        second_order_names=("spot",),
        bump_sizes=jnp.array([1.0e-3, 1.0e-4]),
    )
    parameters = jnp.array([2.0, 0.1])
    aad = evaluate_aad_greeks(
        _QuadraticValuation(), parameters, request, valuation_id="quadratic"
    )
    base = _QuadraticValuation()(parameters)
    up = jnp.stack(
        tuple(
            _QuadraticValuation()(parameters.at[index].add(request.bump_sizes[index]))
            for index in range(2)
        )
    )
    down = jnp.stack(
        tuple(
            _QuadraticValuation()(parameters.at[index].add(-request.bump_sizes[index]))
            for index in range(2)
        )
    )
    bumped = evaluate_bump_greeks(base, up, down, request, valuation_id="quadratic")
    assert jnp.allclose(aad.first_order, jnp.array([4.0, 3.0]))
    assert jnp.allclose(aad.second("spot"), 2.0)
    assert jnp.allclose(bumped.first_order, aad.first_order, rtol=1e-3)
    assert jnp.allclose(bumped.second("spot"), 2.0, rtol=2e-2)


def test_self_financing_hedge_replay_is_exact_for_unit_underlying_claim():
    times = jnp.array([0.0, 0.5, 1.0])
    spots = jnp.array([100.0, 110.0, 105.0])
    replay = evaluate_hedge_replay(
        times,
        spots,
        spots,
        jnp.ones_like(spots),
        0.0,
        replay_id="unit-underlying",
    )
    assert replay.evidence.self_financing
    assert jnp.allclose(replay.replication_errors, 0.0)
