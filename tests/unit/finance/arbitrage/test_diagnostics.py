import jax.numpy as jnp

from phydrax.finance.arbitrage import (
    evaluate_static_option_arbitrage,
    NumeraireOptionMarginal,
    option_marginal_convex_order,
    OptionCallSlice,
)
from phydrax.finance.core import PricingLaw


def _law():
    return PricingLaw(
        "usd-q",
        "synthetic option marginal",
        "equity-factors",
        "market-filtration",
        "risk-neutral",
        "usd-money-market",
        "usd-collateral",
    )


def _marginal(values, probabilities, maturity):
    return NumeraireOptionMarginal(
        jnp.asarray(values),
        jnp.asarray(probabilities),
        jnp.ones((len(values),)),
        1.0,
        _law(),
        maturity=maturity,
        asset_id="asset",
        currency_code="USD",
        market_snapshot_id=f"snapshot-{maturity}",
        option_evidence_id=f"options-{maturity}",
    )


def test_option_marginals_are_checked_in_declared_numeraire_units():
    feasible = option_marginal_convex_order(
        _marginal([-1.0, 1.0], [0.5, 0.5], 1.0),
        _marginal([-2.0, 2.0], [0.5, 0.5], 2.0),
    )
    reversed_order = option_marginal_convex_order(
        _marginal([-2.0, 2.0], [0.5, 0.5], 1.0),
        _marginal([-1.0, 1.0], [0.5, 0.5], 2.0),
    )

    assert bool(feasible.feasible)
    assert not bool(reversed_order.feasible)


def test_static_option_grid_reports_butterfly_violation_without_global_claim():
    valid = OptionCallSlice(
        [80.0, 100.0, 120.0],
        [22.0, 10.0, 4.0],
        100.0,
        1.0,
        maturity=1.0,
        asset_id="asset",
        currency_code="USD",
        numeraire_id="usd-money-market",
        market_snapshot_id="snapshot",
    )
    invalid = OptionCallSlice(
        [80.0, 100.0, 120.0],
        [22.0, 10.0, -4.0],
        100.0,
        1.0,
        maturity=1.0,
        asset_id="asset",
        currency_code="USD",
        numeraire_id="usd-money-market",
        market_snapshot_id="snapshot",
    )

    valid_evidence = evaluate_static_option_arbitrage(valid)
    invalid_evidence = evaluate_static_option_arbitrage(invalid)
    assert bool(valid_evidence.valid)
    assert valid_evidence.support_limited
    assert not bool(invalid_evidence.valid)
    assert invalid_evidence.butterfly_convexity_violation > 0.0
