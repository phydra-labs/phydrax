#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class TriggerLine(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    seed_names: tuple[str, ...] = eqx.field(static=True)
    stream_names: tuple[str, ...] = eqx.field(static=True)
    prescale: int = eqx.field(static=True)
    maximum_latency: float = eqx.field(static=True)
    maximum_resource_units: float = eqx.field(static=True)
    line_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        seed_names: Sequence[str] = (),
        stream_names: Sequence[str],
        prescale: int = 1,
        maximum_latency: float,
        maximum_resource_units: float,
    ):
        name_ = str(name).strip()
        seeds = tuple(sorted(str(value).strip() for value in seed_names))
        streams = tuple(sorted(str(value).strip() for value in stream_names))
        prescale_ = int(prescale)
        latency = float(maximum_latency)
        resources = float(maximum_resource_units)
        if (
            not name_
            or any(not value for value in seeds + streams)
            or len(set(seeds)) != len(seeds)
            or not streams
            or len(set(streams)) != len(streams)
        ):
            raise ValueError("Trigger line identity, seeds, and streams are invalid.")
        if (
            prescale_ < 1
            or not jnp.isfinite(latency)
            or latency < 0.0
            or not jnp.isfinite(resources)
            or resources < 0.0
        ):
            raise ValueError("Trigger prescale and resource limits are invalid.")
        self.name = name_
        self.seed_names = seeds
        self.stream_names = streams
        self.prescale = prescale_
        self.maximum_latency = latency
        self.maximum_resource_units = resources
        self.line_id = canonical_fingerprint(
            {
                "kind": "trigger-line",
                "name": name_,
                "seeds": list(seeds),
                "streams": list(streams),
                "prescale": prescale_,
                "maximum_latency": latency,
                "maximum_resource_units": resources,
            }
        )


class TriggerMenu(StrictModule, NonTrainableState):
    lines: tuple[TriggerLine, ...]
    stream_names: tuple[str, ...] = eqx.field(static=True)
    menu_id: str = eqx.field(static=True)

    def __init__(self, lines: Sequence[TriggerLine], /):
        lines_ = tuple(lines)
        if not lines_ or any(not isinstance(value, TriggerLine) for value in lines_):
            raise TypeError("lines must contain typed non-empty trigger lines.")
        names = tuple(value.name for value in lines_)
        if len(set(names)) != len(names):
            raise ValueError("Trigger line names must be unique.")
        seen: set[str] = set()
        for line in lines_:
            if not set(line.seed_names) <= seen:
                raise ValueError(
                    "Trigger seeds must reference earlier lines, forming a DAG."
                )
            seen.add(line.name)
        streams = tuple(
            sorted({stream for line in lines_ for stream in line.stream_names})
        )
        self.lines = lines_
        self.stream_names = streams
        self.menu_id = canonical_fingerprint(
            {
                "kind": "trigger-menu",
                "lines": [value.line_id for value in lines_],
                "streams": list(streams),
            }
        )


class TriggerReplayResult(StrictModule, NonTrainableState):
    pre_prescale: Array
    post_prescale: Array
    stream_membership: Array
    line_weighted_acceptance: Array
    line_resource_valid: Array
    successful: Array
    menu_id: str = eqx.field(static=True)


def _mix_uint32(value, salt):
    mixed = value.astype(jnp.uint32) ^ jnp.asarray(salt, dtype=jnp.uint32)
    mixed ^= mixed >> jnp.uint32(16)
    mixed *= jnp.uint32(0x7FEB352D)
    mixed ^= mixed >> jnp.uint32(15)
    mixed *= jnp.uint32(0x846CA68B)
    mixed ^= mixed >> jnp.uint32(16)
    return mixed


def replay_trigger_menu(
    menu: TriggerMenu,
    event_ids: ArrayLike,
    raw_decisions: ArrayLike,
    /,
    *,
    event_weights: ArrayLike | None = None,
    measured_latency: ArrayLike | None = None,
    measured_resource_units: ArrayLike | None = None,
) -> TriggerReplayResult:
    """Replay seed, prescale, stream, and resource semantics offline."""
    if not isinstance(menu, TriggerMenu):
        raise TypeError("menu must be TriggerMenu.")
    event_ids_ = jnp.asarray(event_ids)
    raw = jnp.asarray(raw_decisions, dtype=jnp.bool_)
    event_count = event_ids_.shape[0]
    line_count = len(menu.lines)
    if event_ids_.ndim != 1 or raw.shape != (event_count, line_count):
        raise ValueError(
            "Event identities and raw line decisions have incompatible shape."
        )
    weights = (
        jnp.ones((event_count,), dtype=jnp.float64)
        if event_weights is None
        else jnp.asarray(event_weights)
    )
    latency = (
        jnp.zeros((event_count, line_count), dtype=weights.dtype)
        if measured_latency is None
        else jnp.asarray(measured_latency, dtype=weights.dtype)
    )
    resources = (
        jnp.zeros((event_count, line_count), dtype=weights.dtype)
        if measured_resource_units is None
        else jnp.asarray(measured_resource_units, dtype=weights.dtype)
    )
    if (
        weights.shape != (event_count,)
        or latency.shape != raw.shape
        or resources.shape != raw.shape
    ):
        raise ValueError(
            "Trigger weights, latency, and resources must align with decisions."
        )
    pre = jnp.zeros_like(raw)
    post = jnp.zeros_like(raw)
    resource_valid = jnp.zeros_like(raw)
    line_indices = {line.name: index for index, line in enumerate(menu.lines)}
    for index, line in enumerate(menu.lines):
        seeded = raw[:, index]
        for seed_name in line.seed_names:
            seeded &= post[:, line_indices[seed_name]]
        pre = pre.at[:, index].set(seeded)
        keep = (_mix_uint32(event_ids_, index + 1) % jnp.uint32(line.prescale)) == 0
        valid_resource = (
            (latency[:, index] >= 0)
            & (latency[:, index] <= line.maximum_latency)
            & (resources[:, index] >= 0)
            & (resources[:, index] <= line.maximum_resource_units)
        )
        resource_valid = resource_valid.at[:, index].set(valid_resource)
        post = post.at[:, index].set(seeded & keep & valid_resource)
    membership = jnp.zeros((event_count, len(menu.stream_names)), dtype=jnp.bool_)
    stream_indices = {name: index for index, name in enumerate(menu.stream_names)}
    for line_index, line in enumerate(menu.lines):
        for stream_name in line.stream_names:
            stream_index = stream_indices[stream_name]
            membership = membership.at[:, stream_index].set(
                membership[:, stream_index] | post[:, line_index]
            )
    weighted_acceptance = jnp.sum(weights[:, None] * post, axis=0)
    successful = (
        jnp.all(jnp.isfinite(weights))
        & jnp.all(jnp.isfinite(latency))
        & jnp.all(jnp.isfinite(resources))
        & jnp.all(latency >= 0)
        & jnp.all(resources >= 0)
    )
    return TriggerReplayResult(
        pre,
        post,
        membership,
        weighted_acceptance,
        resource_valid,
        successful,
        menu.menu_id,
    )


__all__ = ["TriggerLine", "TriggerMenu", "TriggerReplayResult", "replay_trigger_menu"]
