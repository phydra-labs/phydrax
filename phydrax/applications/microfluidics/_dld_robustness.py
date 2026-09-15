#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._admissibility import AdmissibilityHeader, AdmissibilityReason
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class DLDRobustnessResult(StrictModule):
    nominal: Array
    mean: Array
    worst: Array
    conditional_value_at_risk: Array
    chance_satisfaction: Array
    invalid_sample_count: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class DLDRobustnessPlan(StrictModule, NonTrainableState):
    tail_fraction: float = eqx.field(static=True)
    required_value: float = eqx.field(static=True)
    maximize: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        tail_fraction: float = 0.1,
        required_value: float,
        maximize: bool = True,
    ) -> None:
        tail = float(tail_fraction)
        required = float(required_value)
        if not np.isfinite(tail) or not 0.0 < tail <= 1.0 or not np.isfinite(required):
            raise ValueError("DLD robustness tail and requirement are invalid.")
        self.tail_fraction = tail
        self.required_value = required
        self.maximize = bool(maximize)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dld-robustness",
                "tail_fraction": tail,
                "required_value": required,
                "maximize": bool(maximize),
            }
        )

    def evaluate(
        self,
        values: ArrayLike,
        valid_samples: ArrayLike,
        /,
        *,
        nominal_index: int = 0,
    ) -> DLDRobustnessResult:
        metric = jnp.asarray(values)
        valid = jnp.asarray(valid_samples, dtype=bool)
        nominal = int(nominal_index)
        if (
            metric.ndim != 1
            or valid.shape != metric.shape
            or not 0 <= nominal < metric.size
        ):
            raise ValueError("DLD robustness samples or nominal index are invalid.")
        finite = jnp.isfinite(metric)
        valid = valid & finite
        invalid_count = jnp.sum(~valid)
        safe = jnp.where(valid, metric, jnp.nan)
        count = jnp.maximum(jnp.sum(valid), 1)
        mean = jnp.nansum(safe) / count
        ordered = jnp.sort(
            jnp.where(
                valid,
                metric if self.maximize else -metric,
                jnp.inf,
            )
        )
        tail_count = max(1, int(np.ceil(self.tail_fraction * metric.size)))
        tail_values = ordered[:tail_count]
        cvar_oriented = jnp.mean(tail_values)
        cvar = cvar_oriented if self.maximize else -cvar_oriented
        worst = jnp.nanmin(safe) if self.maximize else jnp.nanmax(safe)
        satisfied = (
            metric >= self.required_value
            if self.maximize
            else metric <= self.required_value
        )
        chance = jnp.sum(valid & satisfied) / count
        supported = (invalid_count == 0) & jnp.all(finite)
        reasons = jnp.where(
            supported,
            jnp.asarray(0, dtype=jnp.uint32),
            jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), dtype=jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(supported, 1.0, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "dld-robustness-evidence", "plan": self.plan_id}
            ),
        )
        return DLDRobustnessResult(
            metric[nominal],
            mean,
            worst,
            cvar,
            chance,
            invalid_count,
            header,
            self.plan_id,
        )


__all__ = ["DLDRobustnessPlan", "DLDRobustnessResult"]
