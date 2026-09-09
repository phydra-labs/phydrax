import equinox as eqx
import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.finance.core import (
    add_currency_amounts,
    AssetReference,
    Currency,
    CurrencyAmount,
    FinancialIdentifier,
    FXPair,
    InstrumentReference,
    MonetaryArray,
    MonetaryRounding,
    round_to_minor_atoms,
    subtract_currency_amounts,
)


def test_financial_references_preserve_namespaced_identity_and_quote_semantics():
    usd = Currency("USD", 2)
    equity_id = FinancialIdentifier("figi", "BBG000B9XRY4")
    option_id = FinancialIdentifier("internal", "AAPL-202612-C200")
    asset = AssetReference(equity_id, "equity", usd, "Apple common equity")
    option = InstrumentReference(
        option_id,
        (equity_id,),
        usd,
        "currency_per_share",
        "December call",
    )

    assert equity_id.canonical == "figi:BBG000B9XRY4"
    assert asset.asset_id == equity_id.canonical
    assert option.instrument_id == option_id.canonical
    assert option.underlying_ids == (equity_id,)
    assert option.settlement_currency.currency_id == usd.currency_id

    with pytest.raises(ValueError, match="unique"):
        InstrumentReference(
            option_id,
            (equity_id, equity_id),
            usd,
            "currency_per_share",
            "duplicate underlying",
        )


def test_exact_same_currency_arithmetic_rejects_cross_currency_substitution():
    usd = Currency("USD", 2)
    eur = Currency("EUR", 2)
    left = CurrencyAmount(usd, 125)
    right = CurrencyAmount(usd, -25)

    total = add_currency_amounts(left, right)
    difference = subtract_currency_amounts(left, right)
    assert total.atoms.dtype == jnp.dtype(jnp.int64)
    assert int(total.atoms) == 100
    assert int(difference.atoms) == 150

    with pytest.raises(ValueError, match="matching currencies"):
        add_currency_amounts(left, CurrencyAmount(eur, 25))

    with pytest.raises(OverflowError, match="signed int64"):
        CurrencyAmount(usd, 2**63)


def test_monetary_rounding_is_explicit_and_padding_is_neutral():
    usd = Currency("USD", 2)
    jpy = Currency("JPY", 0)
    monetary = MonetaryArray(
        jnp.asarray((1.125, 2.5, 0.0)),
        (usd, jpy),
        jnp.asarray((0, 1, 0), dtype=jnp.int32),
        jnp.asarray((True, True, False)),
    )

    np.testing.assert_array_equal(
        round_to_minor_atoms(monetary, MonetaryRounding.HALF_EVEN),
        np.asarray((112, 2, 0), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        round_to_minor_atoms(monetary, MonetaryRounding.HALF_AWAY_FROM_ZERO),
        np.asarray((113, 3, 0), dtype=np.int64),
    )

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="neutral zero padding"
    ):
        invalid = MonetaryArray(
            jnp.asarray((1.0, 0.25)),
            (usd,),
            jnp.asarray((0, 0), dtype=jnp.int32),
            jnp.asarray((True, False)),
        )
        jax.block_until_ready(invalid.values)


def test_fx_pair_is_ordered_identity_and_carries_no_numerical_rate():
    usd = Currency("USD", 2)
    eur = Currency("EUR", 2)
    direct = FXPair(eur, usd)
    reverse = FXPair(usd, eur)

    assert direct.base is eur
    assert direct.quote is usd
    assert direct.pair_id != reverse.pair_id
