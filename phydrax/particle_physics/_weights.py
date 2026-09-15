#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum, StrEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class WeightVariationKind(StrEnum):
    NOMINAL = "nominal"
    NORMALIZATION = "normalization"
    SHAPE = "shape"
    REPLICA = "replica"
    HESSIAN = "hessian"
    ENVELOPE = "envelope"


class EventAccountingStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_WEIGHT = 1
    INVALID_COUNT = 2
    INVALID_CROSS_SECTION = 3
    OVERFLOW = 4


def _labels(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if (
        not result
        or any(not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain distinct non-empty values.")
    return result


class EventWeightSet(StrictModule, NonTrainableState):
    """Fixed named event weights with explicit variation and correlation semantics."""

    values: Array
    event_active: Array
    finite: Array
    names: tuple[str, ...] = eqx.field(static=True)
    variation_kinds: tuple[WeightVariationKind, ...] = eqx.field(static=True)
    correlation_groups: tuple[str, ...] = eqx.field(static=True)
    nominal_index: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    weight_count: int = eqx.field(static=True)
    weight_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        /,
        *,
        names: Sequence[str],
        variation_kinds: Sequence[WeightVariationKind | str],
        correlation_groups: Sequence[str],
        event_active: ArrayLike | None = None,
        nominal_name: str = "nominal",
    ):
        values_ = jnp.asarray(values)
        if values_.ndim != 2 or values_.shape[0] < 1 or values_.shape[1] < 1:
            raise ValueError("values must have shape (event_capacity, weight_count).")
        names_ = _labels(names, "names")
        groups = tuple(str(value).strip() for value in correlation_groups)
        kinds = tuple(WeightVariationKind(value) for value in variation_kinds)
        if (
            len(names_) != values_.shape[1]
            or len(kinds) != len(names_)
            or len(groups) != len(names_)
        ):
            raise ValueError("Weight metadata must align with the weight axis.")
        if any(not value for value in groups):
            raise ValueError("correlation_groups must be non-empty strings.")
        nominal = str(nominal_name).strip()
        if nominal not in names_:
            raise ValueError("nominal_name must identify one named weight.")
        nominal_index = names_.index(nominal)
        if kinds[nominal_index] is not WeightVariationKind.NOMINAL:
            raise ValueError("The nominal weight must use WeightVariationKind.NOMINAL.")
        if sum(kind is WeightVariationKind.NOMINAL for kind in kinds) != 1:
            raise ValueError("Exactly one nominal weight is required.")
        active = (
            jnp.ones((values_.shape[0],), dtype=bool)
            if event_active is None
            else jnp.asarray(event_active, dtype=bool)
        )
        if active.shape != (values_.shape[0],):
            raise ValueError("event_active must align with the event axis.")
        self.values = values_
        self.event_active = active
        self.finite = jnp.all(jnp.isfinite(values_), axis=-1)
        self.names = names_
        self.variation_kinds = kinds
        self.correlation_groups = groups
        self.nominal_index = nominal_index
        self.event_capacity = int(values_.shape[0])
        self.weight_count = int(values_.shape[1])
        self.weight_set_id = canonical_fingerprint(
            {
                "kind": "hep-event-weight-set",
                "names": list(names_),
                "variation_kinds": [kind.value for kind in kinds],
                "correlation_groups": list(groups),
                "nominal": nominal,
                "capacity": self.event_capacity,
            }
        )

    @property
    def nominal(self) -> Array:
        return self.values[:, self.nominal_index]


class CrossSectionLedger(StrictModule, NonTrainableState):
    attempted_count: Array
    generated_count: Array
    selected_count: Array
    positive_count: Array
    negative_count: Array
    zero_count: Array
    sum_weights: Array
    sum_absolute_weights: Array
    sum_squared_weights: Array
    cross_section: Array
    cross_section_uncertainty: Array
    filter_efficiency: Array
    overflow_count: Array
    status: Array
    weight_set_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(EventAccountingStatus.SUCCESS)


def summarize_event_weights(
    weights: EventWeightSet,
    /,
    *,
    selected: ArrayLike | None = None,
    attempted_count: ArrayLike | None = None,
    cross_section: ArrayLike = jnp.nan,
    cross_section_uncertainty: ArrayLike = jnp.nan,
    overflow: ArrayLike | None = None,
) -> CrossSectionLedger:
    """Summarize signed nominal weights without hiding invalid or overflowed events."""
    if not isinstance(weights, EventWeightSet):
        raise TypeError("weights must be EventWeightSet.")
    selected_ = (
        weights.event_active
        if selected is None
        else jnp.asarray(selected, dtype=bool) & weights.event_active
    )
    if selected_.shape != (weights.event_capacity,):
        raise ValueError("selected must align with event capacity.")
    overflow_ = (
        jnp.zeros((weights.event_capacity,), dtype=bool)
        if overflow is None
        else jnp.asarray(overflow, dtype=bool)
    )
    if overflow_.shape != (weights.event_capacity,):
        raise ValueError("overflow must align with event capacity.")
    generated = jnp.sum(weights.event_active, dtype=jnp.int32)
    attempted = (
        generated
        if attempted_count is None
        else jnp.asarray(attempted_count, dtype=jnp.int32)
    )
    active = selected_ & weights.finite & ~overflow_
    nominal = jnp.where(active, weights.nominal, 0.0)
    selected_count = jnp.sum(selected_, dtype=jnp.int32)
    overflow_count = jnp.sum(overflow_ & weights.event_active, dtype=jnp.int32)
    finite = jnp.all(jnp.where(weights.event_active, weights.finite, True))
    count_valid = (attempted >= generated) & (generated >= selected_count)
    cross_section_ = jnp.asarray(cross_section, dtype=weights.values.dtype)
    uncertainty = jnp.asarray(cross_section_uncertainty, dtype=weights.values.dtype)
    cross_section_valid = (jnp.isnan(cross_section_) & jnp.isnan(uncertainty)) | (
        jnp.isfinite(cross_section_) & jnp.isfinite(uncertainty) & (uncertainty >= 0.0)
    )
    status = jnp.where(
        overflow_count > 0,
        int(EventAccountingStatus.OVERFLOW),
        jnp.where(
            ~finite,
            int(EventAccountingStatus.NONFINITE_WEIGHT),
            jnp.where(
                ~count_valid,
                int(EventAccountingStatus.INVALID_COUNT),
                jnp.where(
                    ~cross_section_valid,
                    int(EventAccountingStatus.INVALID_CROSS_SECTION),
                    int(EventAccountingStatus.SUCCESS),
                ),
            ),
        ),
    )
    safe_attempted = jnp.maximum(attempted, 1)
    return CrossSectionLedger(
        attempted,
        generated,
        selected_count,
        jnp.sum((nominal > 0.0).astype(jnp.int32)),
        jnp.sum((nominal < 0.0).astype(jnp.int32)),
        jnp.sum(active & (nominal == 0.0), dtype=jnp.int32),
        jnp.sum(nominal),
        jnp.sum(jnp.abs(nominal)),
        jnp.sum(nominal * nominal),
        cross_section_,
        uncertainty,
        generated.astype(weights.values.dtype)
        / safe_attempted.astype(weights.values.dtype),
        overflow_count,
        status.astype(jnp.int32),
        weights.weight_set_id,
    )


__all__ = [
    "CrossSectionLedger",
    "EventAccountingStatus",
    "EventWeightSet",
    "WeightVariationKind",
    "summarize_event_weights",
]
