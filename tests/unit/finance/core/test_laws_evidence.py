import equinox as eqx
import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from phydrax.finance.core import (
    FinanceEvidenceBinding,
    FinancialScenarioSet,
    laws_compatible,
    PhysicalLaw,
    PricingLaw,
    require_law_compatible,
    scenario_set_compatible,
    StressLaw,
)


def _physical() -> PhysicalLaw:
    return PhysicalLaw("usd-history", "audited history", "usd-factors", "market-close")


def _pricing(measure_id: str = "usd-risk-neutral") -> PricingLaw:
    return PricingLaw(
        "usd-pricing",
        "calibrated pricing law",
        "usd-factors",
        "market-close",
        measure_id,
        "usd-money-market",
        "csa-usd-ois",
    )


def test_physical_pricing_and_stress_laws_are_not_substitutable():
    physical = PhysicalLaw(
        "shared-law", "shared provenance", "usd-factors", "market-close"
    )
    pricing = PricingLaw(
        "shared-law",
        "shared provenance",
        "usd-factors",
        "market-close",
        "usd-risk-neutral",
        "usd-money-market",
        "csa-usd-ois",
    )
    stress = StressLaw("shared-law", "shared provenance", "usd-factors", "market-close")

    assert not laws_compatible(physical, pricing)
    assert not laws_compatible(pricing, stress)
    with pytest.raises(TypeError, match="not substitutable"):
        require_law_compatible(pricing, physical)
    assert not laws_compatible(_pricing(), _pricing("usd-forward-measure"))


def test_scenario_masks_weights_and_padding_are_observable_contracts():
    scenarios = FinancialScenarioSet(
        jnp.asarray(
            (
                ((1.0, 2.0), (1.5, 2.5), (0.0, 0.0)),
                ((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)),
                ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            )
        ),
        jnp.asarray((0.6, 0.4, 0.0)),
        jnp.asarray((0.0, 1.0, 0.0)),
        jnp.asarray(((True, True, False), (True, False, False), (False, False, False))),
        "usd-history",
        "usd-factors",
        "history-2026q3",
        "history-2026q3-r17",
    )

    assert scenario_set_compatible(scenarios, _physical())
    assert scenarios.scenario_active.tolist() == [True, True, False]
    assert scenarios.time_active.tolist() == [True, True, False]

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="zero when inactive"):
        invalid = FinancialScenarioSet(
            scenarios.values.at[2, 2, 0].set(1.0),
            scenarios.weights,
            scenarios.times,
            scenarios.valid,
            scenarios.law_id,
            scenarios.factor_layout_id,
            scenarios.semantic_id,
            "history-2026q3-r18",
        )
        jax.block_until_ready(invalid.values)


def test_scenario_semantic_and_numeric_identities_are_independent():
    values = jnp.asarray((((1.0,),),))
    weights = jnp.asarray((1.0,))
    times = jnp.asarray((0.0,))
    valid = jnp.asarray(((True,),))
    first = FinancialScenarioSet(
        values,
        weights,
        times,
        valid,
        "usd-history",
        "usd-factors",
        "history-meaning",
        "numeric-r1",
    )
    second = FinancialScenarioSet(
        values + 1.0,
        weights,
        times,
        valid,
        "usd-history",
        "usd-factors",
        "history-meaning",
        "numeric-r2",
    )

    assert first.semantic_id == second.semantic_id
    assert first.numeric_id != second.numeric_id
    assert scenario_set_compatible(first, _physical())
    assert scenario_set_compatible(second, _physical())


def test_finance_evidence_groups_are_disjoint():
    binding = FinanceEvidenceBinding(
        ("data-evidence",),
        ("model-evidence",),
        ("numeric-evidence",),
        ("use-evidence",),
    )

    assert binding.data_evidence_ids == ("data-evidence",)
    with pytest.raises(ValueError, match="must be disjoint"):
        FinanceEvidenceBinding(
            ("shared-evidence",),
            ("shared-evidence",),
            (),
            (),
        )
