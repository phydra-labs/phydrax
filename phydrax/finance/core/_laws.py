#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule

from ._identifiers import _canonical_text, _token


class PhysicalLaw(StrictModule):
    """Descriptor for a physical/statistical P-law, separate from model numerics."""

    law_id: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    filtration_id: str = eqx.field(static=True)

    def __init__(
        self,
        law_id: str,
        provenance: str,
        factor_layout_id: str,
        filtration_id: str,
        /,
    ):
        self.law_id = _token(law_id, "physical law_id")
        self.provenance = _canonical_text(provenance, "physical law provenance")
        self.factor_layout_id = _token(factor_layout_id, "factor_layout_id")
        self.filtration_id = _token(filtration_id, "filtration_id")


class PricingLaw(StrictModule):
    """Descriptor for one pricing Q-law with explicit measure and numeraire."""

    law_id: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    filtration_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    numeraire_id: str = eqx.field(static=True)
    collateral_convention_id: str = eqx.field(static=True)

    def __init__(
        self,
        law_id: str,
        provenance: str,
        factor_layout_id: str,
        filtration_id: str,
        measure_id: str,
        numeraire_id: str,
        collateral_convention_id: str,
        /,
    ):
        self.law_id = _token(law_id, "pricing law_id")
        self.provenance = _canonical_text(provenance, "pricing law provenance")
        self.factor_layout_id = _token(factor_layout_id, "factor_layout_id")
        self.filtration_id = _token(filtration_id, "filtration_id")
        self.measure_id = _token(measure_id, "measure_id")
        self.numeraire_id = _token(numeraire_id, "numeraire_id")
        self.collateral_convention_id = _token(
            collateral_convention_id, "collateral_convention_id"
        )


class StressLaw(StrictModule):
    """Descriptor for a stress law, neither a P-law nor a pricing Q-law."""

    law_id: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    filtration_id: str = eqx.field(static=True)

    def __init__(
        self,
        law_id: str,
        provenance: str,
        factor_layout_id: str,
        filtration_id: str,
        /,
    ):
        self.law_id = _token(law_id, "stress law_id")
        self.provenance = _canonical_text(provenance, "stress law provenance")
        self.factor_layout_id = _token(factor_layout_id, "factor_layout_id")
        self.filtration_id = _token(filtration_id, "filtration_id")


FinancialLaw = PhysicalLaw | PricingLaw | StressLaw


def _law(value: object, name: str, /) -> FinancialLaw:
    if not isinstance(value, (PhysicalLaw, PricingLaw, StressLaw)):
        raise TypeError(f"{name} must be a PhysicalLaw, PricingLaw, or StressLaw.")
    return value


def laws_compatible(expected: FinancialLaw, supplied: FinancialLaw, /) -> bool:
    """Return exact semantic compatibility without P/Q/stress substitution."""
    expected_ = _law(expected, "expected law")
    supplied_ = _law(supplied, "supplied law")
    if type(expected_) is not type(supplied_):
        return False
    if expected_.law_id != supplied_.law_id:
        return False
    if expected_.provenance != supplied_.provenance:
        return False
    if expected_.factor_layout_id != supplied_.factor_layout_id:
        return False
    if expected_.filtration_id != supplied_.filtration_id:
        return False
    if isinstance(expected_, PricingLaw) and isinstance(supplied_, PricingLaw):
        return (
            expected_.measure_id == supplied_.measure_id
            and expected_.numeraire_id == supplied_.numeraire_id
            and expected_.collateral_convention_id == supplied_.collateral_convention_id
        )
    return True


def require_law_compatible(expected: FinancialLaw, supplied: FinancialLaw, /) -> None:
    """Require exact law-family and semantic-ID compatibility."""
    expected_ = _law(expected, "expected law")
    supplied_ = _law(supplied, "supplied law")
    if type(expected_) is not type(supplied_):
        raise TypeError("physical, pricing, and stress laws are not substitutable.")
    if not laws_compatible(expected_, supplied_):
        raise ValueError("financial law descriptors are semantically incompatible.")


class FinancialScenarioSet(StrictModule):
    """Fixed-capacity scenario/time/factor values with explicit semantic identity.

    ``values`` has shape ``(scenario, time, factor)`` and ``valid`` has shape
    ``(scenario, time)``. Inactive rows and nodes are neutral zero padding.
    ``semantic_id`` identifies meaning; ``numeric_id`` independently identifies the
    numerical realization and therefore is not used for semantic compatibility.
    """

    values: Array
    weights: Array
    times: Array
    valid: Array
    law_id: str = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    semantic_id: str = eqx.field(static=True)
    numeric_id: str = eqx.field(static=True)
    scenario_capacity: int = eqx.field(static=True)
    time_capacity: int = eqx.field(static=True)
    factor_count: int = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        weights: ArrayLike,
        times: ArrayLike,
        valid: ArrayLike,
        law_id: str,
        factor_layout_id: str,
        semantic_id: str,
        numeric_id: str,
        /,
    ):
        values_ = jnp.asarray(values)
        if values_.ndim != 3 or any(int(size) <= 0 for size in values_.shape):
            raise ValueError(
                "scenario values must have non-empty (scenario, time, factor) shape."
            )
        if not jnp.issubdtype(values_.dtype, jnp.floating):
            raise TypeError("scenario values must have a floating dtype.")
        scenario_capacity, time_capacity, factor_count = map(int, values_.shape)
        weights_ = jnp.asarray(weights)
        times_ = jnp.asarray(times)
        mask = jnp.asarray(valid)
        if weights_.shape != (scenario_capacity,) or not jnp.issubdtype(
            weights_.dtype, jnp.floating
        ):
            raise TypeError("scenario weights must be a floating scenario-axis array.")
        if times_.shape != (time_capacity,) or not jnp.issubdtype(
            times_.dtype, jnp.floating
        ):
            raise TypeError("scenario times must be a floating time-axis array.")
        if mask.shape != (scenario_capacity, time_capacity) or mask.dtype != jnp.dtype(
            bool
        ):
            raise TypeError("scenario valid must be a boolean scenario/time mask.")

        scenario_active = jnp.any(mask, axis=1)
        time_active = jnp.any(mask, axis=0)
        scenario_count = jnp.sum(scenario_active, dtype=jnp.int32)
        time_count = jnp.sum(time_active, dtype=jnp.int32)
        scenario_prefix = jnp.arange(scenario_capacity) < scenario_count
        time_prefix = jnp.arange(time_capacity) < time_count
        mask_prefix = (
            jnp.arange(time_capacity)[None, :]
            < jnp.sum(mask, axis=1, dtype=jnp.int32)[:, None]
        )
        mask = eqx.error_if(
            mask,
            ~jnp.any(scenario_active)
            | ~jnp.all(scenario_active == scenario_prefix)
            | ~jnp.all(time_active == time_prefix)
            | ~jnp.all(mask == (scenario_active[:, None] & mask_prefix)),
            "scenario validity must contain non-empty prefix rows and time paths.",
        )
        values_ = eqx.error_if(
            values_,
            jnp.any(mask[..., None] & ~jnp.isfinite(values_))
            | jnp.any(~mask[..., None] & (values_ != 0)),
            "scenario values must be finite when active and zero when inactive.",
        )
        weights_ = eqx.error_if(
            weights_,
            jnp.any(scenario_active & (~jnp.isfinite(weights_) | (weights_ <= 0)))
            | jnp.any(~scenario_active & (weights_ != 0))
            | ~jnp.isclose(jnp.sum(weights_), 1.0, rtol=1e-6, atol=1e-7),
            "active scenario weights must be finite, positive, and sum to one; "
            "inactive weights must be zero.",
        )
        active_times = jnp.where(time_active, times_, 0)
        times_ = eqx.error_if(
            times_,
            jnp.any(time_active & ~jnp.isfinite(times_))
            | jnp.any(~time_active & (times_ != 0))
            | jnp.any(time_active[1:] & (active_times[1:] <= active_times[:-1])),
            "active scenario times must be finite and strictly increasing; inactive "
            "times must be zero.",
        )
        self.values = values_
        self.weights = weights_
        self.times = times_
        self.valid = mask
        law_id_ = _token(law_id, "scenario law_id")
        factor_layout_id_ = _token(factor_layout_id, "factor_layout_id")
        semantic_id_ = _token(semantic_id, "scenario semantic_id")
        numeric_id_ = _token(numeric_id, "scenario numeric_id")
        if semantic_id_ == numeric_id_:
            raise ValueError("scenario semantic_id and numeric_id must be distinct.")
        self.law_id = law_id_
        self.factor_layout_id = factor_layout_id_
        self.semantic_id = semantic_id_
        self.numeric_id = numeric_id_
        self.scenario_capacity = scenario_capacity
        self.time_capacity = time_capacity
        self.factor_count = factor_count

    @property
    def scenario_active(self) -> Array:
        return jnp.any(self.valid, axis=1)

    @property
    def time_active(self) -> Array:
        return jnp.any(self.valid, axis=0)


def scenario_set_compatible(
    scenarios: FinancialScenarioSet,
    law: FinancialLaw,
    /,
) -> bool:
    """Check semantic law/layout compatibility, independent of numeric revision."""
    if not isinstance(scenarios, FinancialScenarioSet):
        raise TypeError("scenarios must be a FinancialScenarioSet.")
    law_ = _law(law, "law")
    return (
        scenarios.law_id == law_.law_id
        and scenarios.factor_layout_id == law_.factor_layout_id
    )


__all__ = [
    "FinancialLaw",
    "FinancialScenarioSet",
    "laws_compatible",
    "PhysicalLaw",
    "PricingLaw",
    "require_law_compatible",
    "scenario_set_compatible",
    "StressLaw",
]
