#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity event-to-MUAP template superposition."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class TemplateEMGEvidence(StrictModule, NonTrainableState):
    finite: Array
    event_times_valid: Array
    output_grid_valid: Array
    template_support_complete: Array
    fixed_event_topology: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    differentiation_scope: str = eqx.field(
        static=True,
        default="template samples/amplitudes only for fixed event indices and masks",
    )


class TemplateEMGResult(StrictModule, NonTrainableState):
    sample_times_s: Array
    voltage_V: Array
    evidence: TemplateEMGEvidence
    plan_id: str = eqx.field(static=True)


class MotorUnitActionPotentialTemplatePlan(StrictModule, NonTrainableState):
    """Explicit supplied MUAP bank; no activation-to-EMG substitution."""

    template_V: Array
    sample_period_s: float = eqx.field(static=True)
    zero_index: int = eqx.field(static=True)
    unit_ids: tuple[str, ...] = eqx.field(static=True)
    channel_ids: tuple[str, ...] = eqx.field(static=True)
    template_source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        template_V: ArrayLike,
        sample_period_s: float,
        zero_index: int,
        unit_ids: tuple[str, ...],
        channel_ids: tuple[str, ...],
        /,
        *,
        template_source_id: str,
    ):
        template = jnp.asarray(template_V)
        if template.ndim != 3:
            raise ValueError("template_V must have shape (motor_unit, channel, sample).")
        if not jnp.issubdtype(template.dtype, jnp.inexact):
            template = template.astype(float)
        units = tuple(str(value).strip() for value in unit_ids)
        channels = tuple(str(value).strip() for value in channel_ids)
        if (
            template.shape[:2] != (len(units), len(channels))
            or not units
            or not channels
            or any(not value for value in units + channels)
            or len(set(units)) != len(units)
            or len(set(channels)) != len(channels)
        ):
            raise ValueError("Template axes must match unique nonempty unit/channel IDs.")
        period = float(sample_period_s)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("sample_period_s must be positive and finite.")
        origin = int(zero_index)
        if origin < 0 or origin >= template.shape[-1]:
            raise ValueError("zero_index must select a template sample.")
        source = str(template_source_id).strip()
        if not source:
            raise ValueError("template_source_id must be nonempty.")
        if not np.all(np.isfinite(np.asarray(template))):
            raise ValueError("MUAP templates must be finite.")
        self.template_V = template
        self.sample_period_s = period
        self.zero_index = origin
        self.unit_ids = units
        self.channel_ids = channels
        self.template_source_id = source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "supplied-motor-unit-action-potential-templates",
                "template": array_tree_fingerprint(template),
                "sample_period_s": period.hex(),
                "zero_index": origin,
                "unit_ids": units,
                "channel_ids": channels,
                "template_source_id": source,
            }
        )

    def prepare(self, /) -> PreparedMotorUnitActionPotentialTemplates:
        return PreparedMotorUnitActionPotentialTemplates(self)


class PreparedMotorUnitActionPotentialTemplates(StrictModule):
    plan: MotorUnitActionPotentialTemplatePlan

    def synthesize(
        self,
        event_times_s: ArrayLike,
        event_mask: ArrayLike,
        sample_times_s: ArrayLike,
        /,
    ) -> TemplateEMGResult:
        events = jnp.asarray(event_times_s, dtype=self.plan.template_V.dtype)
        mask = jnp.asarray(event_mask, dtype=bool)
        times = jnp.asarray(sample_times_s, dtype=self.plan.template_V.dtype)
        if events.ndim != 2 or events.shape[0] != len(self.plan.unit_ids):
            raise ValueError("event_times_s must have shape (motor_unit, event_slot).")
        if mask.shape != events.shape:
            raise ValueError("event_mask must match event_times_s.")
        if times.ndim != 1 or times.shape[0] == 0:
            raise ValueError("sample_times_s must be one nonempty vector.")
        topology_events = jax.lax.stop_gradient(events)
        topology_mask = jax.lax.stop_gradient(mask)
        template = self.plan.template_V
        period = self.plan.sample_period_s
        origin = self.plan.zero_index
        sample_count = template.shape[-1]
        active_event_times = jnp.where(topology_mask, topology_events, 0.0)

        def one_template(values, unit_event_times, unit_mask):
            coordinate = (
                times[None, :] - unit_event_times[:, None]
            ) / period + origin
            lower = jax.lax.stop_gradient(
                jnp.floor(coordinate).astype(jnp.int32)
            )
            fraction = coordinate - lower
            valid = (
                unit_mask[:, None]
                & (coordinate >= 0)
                & (coordinate <= sample_count - 1)
            )
            lower_safe = jnp.clip(lower, 0, sample_count - 1)
            upper_safe = jnp.clip(lower + 1, 0, sample_count - 1)
            interpolated = (
                (1.0 - fraction) * values[lower_safe]
                + fraction * values[upper_safe]
            )
            return jnp.sum(jnp.where(valid, interpolated, 0.0), axis=0)

        def one_unit(unit_index):
            def one_channel(channel_index):
                return one_template(
                    template[unit_index, channel_index],
                    active_event_times[unit_index],
                    topology_mask[unit_index],
                )

            return jax.vmap(one_channel)(
                jnp.arange(len(self.plan.channel_ids), dtype=jnp.int32)
            )

        unit_voltage = jax.vmap(one_unit)(
            jnp.arange(len(self.plan.unit_ids), dtype=jnp.int32)
        )
        voltage = jnp.sum(unit_voltage, axis=0)
        finite = jnp.all(jnp.isfinite(voltage)) & jnp.all(
            jnp.isfinite(jnp.where(mask, events, 0.0))
        )
        event_valid = jnp.all(~mask | jnp.isfinite(events))
        grid_valid = jnp.all(jnp.isfinite(times)) & jnp.all(jnp.diff(times) > 0.0)
        support_start_s = active_event_times - origin * period
        support_end_s = active_event_times + (sample_count - 1 - origin) * period
        event_support_complete = (times[0] <= support_start_s) & (
            support_end_s <= times[-1]
        )
        complete = jnp.all(~topology_mask | event_support_complete)
        successful = finite & event_valid & grid_valid & complete
        evidence = TemplateEMGEvidence(
            finite,
            event_valid,
            grid_valid,
            complete,
            jnp.asarray(True),
            successful,
            self.plan.plan_id,
        )
        return TemplateEMGResult(times, voltage, evidence, self.plan.plan_id)


__all__ = [
    "MotorUnitActionPotentialTemplatePlan",
    "PreparedMotorUnitActionPotentialTemplates",
    "TemplateEMGEvidence",
    "TemplateEMGResult",
]
