#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit retained heat, never a conversion of total metabolic power."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._uchida_umberger_2010 import UchidaUmberger2010Plan, UchidaUmberger2010Result


RETAINED_HEAT_CATEGORIES = (
    "activation-maintenance",
    "intrinsic-shortening-lengthening",
    "immediate-negative-work-dissipation",
    "heat-floor-adjustment",
    "separately-sourced-basal",
)


class RetainedHeatLedger(StrictModule, NonTrainableState):
    """Interval-constant source powers in W; transfers are not thermal sources.

    Rows follow ``RETAINED_HEAT_CATEGORIES``; columns follow ``source_ids``.
    No perfusion, boundary exchange, or total metabolism slot exists. Chemical
    storage/export are reported separately, and are never added to retained heat.
    The caller attests that the named source state has already been accepted.
    """

    category_power_W: Array
    chemical_storage_power_W: Array
    chemical_export_power_W: Array
    time_start_s: Array
    time_end_s: Array
    successful: Array
    source_ids: tuple[str, ...] = eqx.field(static=True)
    source_model_id: str = eqx.field(static=True)
    source_state_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    basal_evidence_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        category_power_W: ArrayLike,
        source_ids: tuple[str, ...],
        /,
        *,
        source_model_id: str,
        source_state_id: str,
        evidence_id: str,
        time_start_s: ArrayLike,
        time_end_s: ArrayLike,
        chemical_storage_power_W: ArrayLike,
        chemical_export_power_W: ArrayLike,
        basal_evidence_id: str | None = None,
        source_successful: ArrayLike = True,
    ):
        ids = tuple(source_ids)
        if not ids or len(set(ids)) != len(ids) or any(not x.strip() for x in ids):
            raise ValueError("Heat source IDs must be unique and nonempty.")
        if any(not x.strip() for x in (source_model_id, source_state_id, evidence_id)):
            raise ValueError(
                "Retained heat requires model, accepted-state and evidence IDs."
            )
        power = jnp.asarray(category_power_W)
        if not jnp.issubdtype(power.dtype, jnp.floating):
            power = power.astype(float)
        if power.shape != (len(RETAINED_HEAT_CATEGORIES), len(ids)):
            raise ValueError("Retained heat requires five category rows by source.")
        storage = jnp.asarray(chemical_storage_power_W, dtype=power.dtype)
        export = jnp.asarray(chemical_export_power_W, dtype=power.dtype)
        start = jnp.asarray(time_start_s, dtype=power.dtype)
        end = jnp.asarray(time_end_s, dtype=power.dtype)
        accepted = jnp.asarray(source_successful, dtype=bool)
        if storage.shape != (len(ids),) or export.shape != storage.shape:
            raise ValueError("Chemical transfer powers must match source IDs.")
        if start.shape != () or end.shape != () or accepted.shape != ():
            raise ValueError("Heat interval and acceptance must be scalar.")
        if basal_evidence_id is not None and not basal_evidence_id.strip():
            raise ValueError("Basal evidence must be nonempty when supplied.")
        self.category_power_W = power
        self.chemical_storage_power_W = storage
        self.chemical_export_power_W = export
        self.time_start_s = start
        self.time_end_s = end
        self.source_ids = ids
        self.source_model_id = source_model_id
        self.source_state_id = source_state_id
        self.evidence_id = evidence_id
        self.basal_evidence_id = basal_evidence_id
        self.successful = (
            accepted
            & jnp.all(jnp.isfinite(power))
            & jnp.all(power >= 0)
            & jnp.all(jnp.isfinite(storage))
            & jnp.all(jnp.isfinite(export))
            & jnp.all(export >= 0)
            & jnp.isfinite(start)
            & jnp.isfinite(end)
            & (end > start)
            & (jnp.all(power[4] == 0) if basal_evidence_id is None else True)
        )

    @property
    def retained_power_W(self) -> Array:
        return jnp.sum(self.category_power_W, axis=0)


def uchida_umberger_retained_heat(
    plan: UchidaUmberger2010Plan,
    result: UchidaUmberger2010Result,
    /,
    *,
    source_state_id: str,
    evidence_id: str,
    time_start_s: ArrayLike,
    time_end_s: ArrayLike,
) -> RetainedHeatLedger:
    """Use mass × corrected heat once, exposing—not adding—the corrections.

    ``evidence_id`` must identify the caller's local retention/registration
    justification. This adapter does not supply anatomy or a physiological
    allocation default. No basal heat or chemical flux is inferred.
    """
    if result.model_id != plan.model_id:
        raise ValueError("Uchida result belongs to another muscle/model identity.")
    mass = result.muscle_mass_kg
    specific = jnp.stack(
        (
            result.activation_maintenance_heat_W_per_kg,
            result.intrinsic_shortening_lengthening_heat_W_per_kg,
            result.negative_work_dissipation_W_per_kg,
            result.heat_floor_adjustment_W_per_kg,
            jnp.zeros_like(result.heat_rate_W_per_kg),
        )
    )
    discrepancy = jnp.abs(jnp.sum(specific, axis=0) - result.heat_rate_W_per_kg)
    tolerance = (
        32
        * jnp.finfo(specific.dtype).eps
        * jnp.maximum(1, jnp.abs(result.heat_rate_W_per_kg))
    )
    return RetainedHeatLedger(
        mass[None, :] * specific,
        plan.muscle_ids,
        source_model_id=result.model_id,
        source_state_id=source_state_id,
        evidence_id=evidence_id,
        time_start_s=time_start_s,
        time_end_s=time_end_s,
        chemical_storage_power_W=jnp.zeros_like(mass),
        chemical_export_power_W=jnp.zeros_like(mass),
        source_successful=result.evidence.successful & jnp.all(discrepancy <= tolerance),
    )


__all__ = [
    "RETAINED_HEAT_CATEGORIES",
    "RetainedHeatLedger",
    "uchida_umberger_retained_heat",
]
