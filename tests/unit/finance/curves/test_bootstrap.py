#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.finance.core import Currency, FinanceDate
from phydrax.finance.curves._bootstrap import (
    BootstrapSolverPolicy,
    DepositBootstrapInstrument,
    MultiCurveBootstrapPlan,
    ParSwapBootstrapInstrument,
    SingleCurveBootstrapPlan,
)
from phydrax.finance.curves._core import (
    CurveDefinition,
    CurveGrid,
    InterpolationPolicy,
)


USD = Currency("USD", 2)
VALUATION_DATE = FinanceDate.from_iso("2026-09-08")
POLICY = InterpolationPolicy(
    "linear", left_extrapolation="forbid", right_extrapolation="flat_forward"
)
SOLVER = BootstrapSolverPolicy(
    absolute_optimality=1e-8,
    relative_optimality=1e-8,
    maximum_steps=100,
    repricing_tolerance=2e-6,
    rank_tolerance=1e-7,
)


def _log_discount_definition(curve_id, role):
    return CurveDefinition(
        curve_id=curve_id,
        role=role,
        valuation_date=VALUATION_DATE,
        currency=USD,
        representation="log_discount",
        grid=CurveGrid(jnp.array([0.0, 1.0, 2.0])),
        interpolation=POLICY,
    )


def test_single_curve_bootstrap_recovers_a_flat_curve_and_quote_jacobian():
    definition = _log_discount_definition("usd-discount", "discount")
    instruments = (
        DepositBootstrapInstrument(
            instrument_id="deposit-1y",
            discount_curve_id="usd-discount",
            start_time=0.0,
            end_time=1.0,
            accrual_fraction=1.0,
        ),
        DepositBootstrapInstrument(
            instrument_id="deposit-2y",
            discount_curve_id="usd-discount",
            start_time=0.0,
            end_time=2.0,
            accrual_fraction=2.0,
        ),
    )
    true_rate = 0.025
    quotes = jnp.array([jnp.exp(true_rate) - 1.0, (jnp.exp(2.0 * true_rate) - 1.0) / 2.0])
    result = SingleCurveBootstrapPlan(
        definition,
        instruments,
        jnp.array([0.0, -0.01, -0.02]),
        solver_policy=SOLVER,
    ).solve(quotes)

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.curves.curve("usd-discount").node_values,
        jnp.array([0.0, -true_rate, -2.0 * true_rate]),
        atol=2e-6,
    )
    assert result.curves.curve("usd-discount").node_quote_jacobian.shape == (3, 2)


def test_joint_two_curve_bootstrap_uses_discounted_projection_cashflows():
    discount_definition = _log_discount_definition("discount", "discount")
    projection_definition = _log_discount_definition("projection", "projection")
    instruments = (
        DepositBootstrapInstrument(
            instrument_id="discount-1y",
            discount_curve_id="discount",
            start_time=0.0,
            end_time=1.0,
            accrual_fraction=1.0,
        ),
        DepositBootstrapInstrument(
            instrument_id="discount-2y",
            discount_curve_id="discount",
            start_time=0.0,
            end_time=2.0,
            accrual_fraction=2.0,
        ),
        ParSwapBootstrapInstrument(
            instrument_id="projection-swap-1y",
            discount_curve_id="discount",
            projection_curve_id="projection",
            start_times=jnp.array([0.0, 0.0]),
            end_times=jnp.array([1.0, 0.0]),
            payment_times=jnp.array([1.0, 0.0]),
            accrual_fractions=jnp.array([1.0, 0.0]),
            valid=jnp.array([True, False]),
        ),
        ParSwapBootstrapInstrument(
            instrument_id="projection-swap-2y",
            discount_curve_id="discount",
            projection_curve_id="projection",
            start_times=jnp.array([0.0, 1.0]),
            end_times=jnp.array([1.0, 2.0]),
            payment_times=jnp.array([1.0, 2.0]),
            accrual_fractions=jnp.ones((2,)),
            valid=jnp.ones((2,), dtype=bool),
        ),
    )
    discount_rate = 0.02
    projection_rate = 0.035
    d1, d2 = jnp.exp(-discount_rate), jnp.exp(-2.0 * discount_rate)
    forward = jnp.exp(projection_rate) - 1.0
    quotes = jnp.array(
        [
            1.0 / d1 - 1.0,
            (1.0 / d2 - 1.0) / 2.0,
            forward,
            (d1 * forward + d2 * forward) / (d1 + d2),
        ]
    )
    result = MultiCurveBootstrapPlan(
        (discount_definition, projection_definition),
        instruments,
        (
            jnp.array([0.0, -0.01, -0.02]),
            jnp.array([0.0, -0.02, -0.04]),
        ),
        solver_policy=SOLVER,
    ).solve(quotes)

    assert bool(result.successful)
    assert int(result.jacobian_rank) == 4
    np.testing.assert_allclose(
        result.curves.curve("discount").node_values,
        jnp.array([0.0, -discount_rate, -2.0 * discount_rate]),
        atol=3e-6,
    )
    np.testing.assert_allclose(
        result.curves.curve("projection").node_values,
        jnp.array([0.0, -projection_rate, -2.0 * projection_rate]),
        atol=3e-6,
    )


def test_bootstrap_success_requires_independent_instrument_repricing():
    definition = _log_discount_definition("discount", "discount")
    instruments = tuple(
        DepositBootstrapInstrument(
            instrument_id=instrument_id,
            discount_curve_id="discount",
            start_time=0.0,
            end_time=1.0,
            accrual_fraction=1.0,
        )
        for instrument_id in ("duplicate-a", "duplicate-b")
    )
    result = SingleCurveBootstrapPlan(
        definition,
        instruments,
        jnp.array([0.0, -0.01, -0.02]),
        solver_policy=SOLVER,
    ).solve(jnp.array([0.02, 0.02]))

    assert not bool(result.successful)
    assert not bool(result.independent)
    with pytest.raises(RuntimeError, match="independently identify"):
        result.require_success()


def test_replay_rejects_quote_layout_changes_even_when_shapes_match():
    definition = _log_discount_definition("discount", "discount")
    base = (
        DepositBootstrapInstrument(
            instrument_id="one",
            discount_curve_id="discount",
            start_time=0.0,
            end_time=1.0,
            accrual_fraction=1.0,
        ),
        DepositBootstrapInstrument(
            instrument_id="two",
            discount_curve_id="discount",
            start_time=0.0,
            end_time=2.0,
            accrual_fraction=2.0,
        ),
    )
    changed = (
        base[0],
        DepositBootstrapInstrument(
            instrument_id="different-two",
            discount_curve_id="discount",
            start_time=0.0,
            end_time=2.0,
            accrual_fraction=2.0,
        ),
    )
    initial = jnp.array([0.0, -0.01, -0.02])
    plan = SingleCurveBootstrapPlan(definition, base, initial, solver_policy=SOLVER)
    replay = plan.prepare_replay()
    changed_plan = SingleCurveBootstrapPlan(
        definition, changed, initial, solver_policy=SOLVER
    )

    with pytest.raises(ValueError, match="topology"):
        replay.refresh(changed_plan, jnp.array([0.01, 0.02]))
