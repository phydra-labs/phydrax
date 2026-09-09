import jax.numpy as jnp

from phydrax.finance.contracts._exercise import SettlementTerms
from phydrax.finance.contracts._options import (
    BasketOption,
    EuropeanOption,
    OptionType,
)
from phydrax.finance.core._currency import Currency, CurrencyAmount
from phydrax.finance.core._identifiers import FinancialIdentifier
from phydrax.finance.core._time import BusinessDayRule, FinanceDate


def _settlement(currency):
    return SettlementTerms(
        currency,
        calendar_id="NYC",
        business_day_rule=BusinessDayRule.FOLLOWING,
    )


def test_exact_contract_terms_lower_to_typed_device_payoffs():
    usd = Currency("USD", 2)
    underlying = FinancialIdentifier("ticker", "XYZ")
    option = EuropeanOption(
        underlying,
        CurrencyAmount(usd, 10_000),
        FinanceDate.from_iso("2027-09-08"),
        OptionType.CALL,
        _settlement(usd),
        quantity=3.0,
    )
    payoff = option.payoff()
    assert jnp.isclose(payoff.strike, 100.0)
    assert jnp.isclose(payoff.notional, 3.0)


def test_basket_contract_rejects_weight_underlying_shape_mismatch():
    usd = Currency("USD", 2)
    first = FinancialIdentifier("ticker", "AAA")
    second = FinancialIdentifier("ticker", "BBB")
    try:
        BasketOption(
            (first, second),
            jnp.array([1.0]),
            CurrencyAmount(usd, 10_000),
            FinanceDate.from_iso("2027-09-08"),
            OptionType.CALL,
            _settlement(usd),
        )
    except ValueError:
        return
    raise AssertionError("basket contract accepted mismatched weights")
