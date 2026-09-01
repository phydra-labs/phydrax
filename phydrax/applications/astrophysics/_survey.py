#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._observation_status import AstrophysicsObservationStatus


class SurveyVisitPlan(StrictModule, NonTrainableState):
    times: Array
    exposure: Array
    dither: Array
    depth: Array
    selection_width: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        times,
        exposure,
        dither,
        depth,
        selection_width,
        /,
        *,
        survey_id="survey-visits",
    ):
        self.times = jnp.asarray(times)
        self.exposure = jnp.asarray(exposure)
        self.dither = jnp.asarray(dither)
        self.depth = jnp.asarray(depth)
        self.selection_width = jnp.asarray(selection_width)
        count = int(self.times.size)
        if (
            self.exposure.shape != (count,)
            or self.dither.shape != (count, 2)
            or self.depth.shape != (count,)
            or self.selection_width.shape != (count,)
        ):
            raise ValueError("Survey visit arrays are inconsistent.")
        self.plan_id = canonical_fingerprint(
            {"kind": "survey-visit-plan", "survey_id": str(survey_id), "visits": count}
        )

    def selection_probability(
        self, measured_flux: ArrayLike, visit_index: ArrayLike, /
    ) -> Array:
        flux = jnp.asarray(measured_flux)
        index = jnp.asarray(visit_index, dtype=jnp.int32)
        threshold = self.depth[index]
        width = self.selection_width[index]
        return jax_sigmoid((flux - threshold) / jnp.maximum(width, 1.0e-30))


class SurveyCatalogResult(StrictModule):
    selected_values: Array
    selected_weight: Array
    selected_count: Array
    overflow: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class SurveyCatalogPlan(StrictModule, NonTrainableState):
    capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, capacity: int, /, *, catalog_id="survey-catalog"):
        if int(capacity) <= 0:
            raise ValueError("Survey catalog capacity must be positive.")
        self.capacity = int(capacity)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "survey-catalog-plan",
                "catalog_id": str(catalog_id),
                "capacity": int(capacity),
            }
        )

    def select(
        self, values: ArrayLike, probability: ArrayLike, threshold: ArrayLike, /
    ) -> SurveyCatalogResult:
        values_ = jnp.asarray(values)
        probability_ = jnp.asarray(probability)
        selected = probability_ >= jnp.asarray(threshold)
        order = jnp.argsort(~selected)
        selected_values = values_[order[: self.capacity]]
        selected_weight = probability_[order[: self.capacity]]
        count = jnp.sum(selected.astype(jnp.int32))
        overflow = count > self.capacity
        valid = jnp.all(jnp.isfinite(selected_values)) & jnp.all(
            (probability_ >= 0.0) & (probability_ <= 1.0)
        )
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.NONPHYSICAL_MODEL),
        ).astype(jnp.int32)
        return SurveyCatalogResult(
            selected_values,
            selected_weight,
            jnp.minimum(count, self.capacity),
            overflow,
            valid,
            status,
            self.plan_id,
        )


def jax_sigmoid(value: Array, /) -> Array:
    return 0.5 * (jnp.tanh(0.5 * value) + 1.0)


__all__ = ["SurveyCatalogPlan", "SurveyCatalogResult", "SurveyVisitPlan"]
