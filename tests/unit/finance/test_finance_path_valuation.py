import jax.numpy as jnp

from phydrax.finance.contracts._options import (
    AsianPayoff,
    AverageType,
    BarrierActivation,
    BarrierDirection,
    BasketPayoff,
    OptionType,
    PathBarrierPayoff,
    VarianceSwapPayoff,
)
from phydrax.finance.valuation._monte_carlo import (
    evaluate_monte_carlo,
    evaluate_path_payoff,
    MonteCarloPathBatch,
    MonteCarloValuationPlan,
    prepare_monte_carlo,
)


def _paths():
    values = jnp.array(
        [
            [[100.0, 100.0], [110.0, 90.0], [120.0, 80.0]],
            [[100.0, 100.0], [90.0, 110.0], [80.0, 120.0]],
            [[100.0, 100.0], [105.0, 105.0], [110.0, 110.0]],
            [[100.0, 100.0], [95.0, 95.0], [90.0, 90.0]],
        ]
    )
    return MonteCarloPathBatch(
        jnp.array([0.0, 0.5, 1.0]),
        values,
        jnp.ones((4, 3), dtype=bool),
        path_id="deterministic-paths",
    )


def test_path_payoffs_cover_barrier_asian_basket_and_variance_swap_semantics():
    paths = _paths()
    asian, valid = evaluate_path_payoff(
        paths,
        AsianPayoff(100.0, OptionType.CALL, average_type=AverageType.ARITHMETIC),
    )
    assert jnp.all(valid)
    assert jnp.allclose(asian, jnp.array([10.0, 0.0, 5.0, 0.0]))

    barrier, _ = evaluate_path_payoff(
        paths,
        PathBarrierPayoff(
            100.0,
            115.0,
            OptionType.CALL,
            BarrierDirection.UP,
            BarrierActivation.KNOCK_OUT,
            rebate=2.0,
        ),
    )
    assert jnp.allclose(barrier, jnp.array([2.0, 0.0, 10.0, 0.0]))

    basket, _ = evaluate_path_payoff(
        paths,
        BasketPayoff(100.0, jnp.array([0.5, 0.5]), OptionType.CALL),
    )
    assert jnp.allclose(basket, jnp.array([0.0, 0.0, 10.0, 0.0]))

    single = MonteCarloPathBatch(
        paths.times,
        paths.values[:, :, :1],
        paths.valid,
        path_id="single-asset",
    )
    variance, _ = evaluate_path_payoff(single, VarianceSwapPayoff(0.0, 1.0))
    assert jnp.all(variance >= 0.0)


def test_monte_carlo_route_reports_exact_deterministic_sample_moments():
    paths = _paths()
    payoff = AsianPayoff(100.0, OptionType.CALL)
    prepared = prepare_monte_carlo(
        paths,
        payoff,
        1.0,
        MonteCarloValuationPlan(minimum_paths=4),
    )
    result = evaluate_monte_carlo(prepared)
    assert jnp.allclose(result.value, 3.75)
    assert result.standard_error > 0.0
