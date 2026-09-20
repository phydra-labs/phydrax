#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import inspect
import marshal
from collections.abc import Callable
from enum import IntEnum
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._identity import callable_payload
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


GRRayEventMargin: TypeAlias = Callable[[Array, Array, Array], Array]


def _event_margin_identity(margin: GRRayEventMargin | None, /) -> object:
    if margin is None:
        return None
    if inspect.isfunction(margin):
        closure = (
            ()
            if margin.__closure__ is None
            else tuple(cell.cell_contents for cell in margin.__closure__)
        )
        return {
            "function": f"{margin.__module__}.{margin.__qualname__}",
            "code": hashlib.sha256(marshal.dumps(margin.__code__)).hexdigest(),
            "defaults": margin.__defaults__,
            "closure": closure,
        }
    return callable_payload(margin)


class GRRayEventCode(IntEnum):
    """Ordered terminal events for one relativistic trajectory."""

    NONE = 0
    CAPTURE = 1
    ESCAPE = 2
    DOMAIN = 3
    WORK = 4


class GRRayEventSurfaces(StrictModule, NonTrainableState):
    """Optional signed event surfaces, ordered capture, escape, then domain.

    Each callable receives ``(affine_parameter, coordinates, tangent)`` and returns
    a scalar margin. A trajectory is inside the admissible side while the margin is
    positive and triggers on a down-crossing through zero.
    """

    capture_margin: GRRayEventMargin | None
    escape_margin: GRRayEventMargin | None
    domain_margin: GRRayEventMargin | None
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        capture_margin: GRRayEventMargin | None = None,
        escape_margin: GRRayEventMargin | None = None,
        domain_margin: GRRayEventMargin | None = None,
        event_id: str | None = None,
    ):
        margins = (capture_margin, escape_margin, domain_margin)
        if any(margin is not None and not callable(margin) for margin in margins):
            raise TypeError("GR ray event margins must be callable or None.")
        event_id_ = (
            canonical_fingerprint(
                {
                    "kind": "gr-ray-event-surfaces",
                    "capture": _event_margin_identity(capture_margin),
                    "escape": _event_margin_identity(escape_margin),
                    "domain": _event_margin_identity(domain_margin),
                }
            )
            if event_id is None
            else str(event_id)
        )
        if not event_id_:
            raise ValueError("event_id must be non-empty.")
        self.capture_margin = capture_margin
        self.escape_margin = escape_margin
        self.domain_margin = domain_margin
        self.event_id = event_id_

    @property
    def enabled(self) -> bool:
        return any(
            margin is not None
            for margin in (
                self.capture_margin,
                self.escape_margin,
                self.domain_margin,
            )
        )

    def margins(
        self,
        affine_parameter: ArrayLike,
        coordinates: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        """Evaluate all ordered surfaces; disabled surfaces have infinite margin."""

        affine = jnp.asarray(affine_parameter)
        point = jnp.asarray(coordinates)
        velocity = jnp.asarray(tangent)
        if point.shape[-1:] != (4,) or velocity.shape != point.shape:
            raise ValueError("GR event coordinates and tangents must end in shape (4,).")
        infinity = jnp.full(point.shape[:-1], jnp.inf, dtype=point.dtype)
        values = (
            infinity
            if self.capture_margin is None
            else jnp.asarray(self.capture_margin(affine, point, velocity)),
            infinity
            if self.escape_margin is None
            else jnp.asarray(self.escape_margin(affine, point, velocity)),
            infinity
            if self.domain_margin is None
            else jnp.asarray(self.domain_margin(affine, point, velocity)),
        )
        if any(value.shape != point.shape[:-1] for value in values):
            raise ValueError("Each GR ray event margin must return the ray batch shape.")
        return jnp.stack(values, axis=-1)


class GRRayEventLedger(StrictModule, NonTrainableState):
    """One fixed terminal-event slot per ray, including work exhaustion."""

    event_code: Array
    affine_parameter: Array
    history_index: Array
    state: Array
    state_valid: Array
    recorded: Array
    simultaneous: Array
    ledger_id: str = eqx.field(static=True)

    def __init__(
        self,
        event_code: ArrayLike,
        affine_parameter: ArrayLike,
        history_index: ArrayLike,
        state: ArrayLike,
        state_valid: ArrayLike,
        recorded: ArrayLike,
        simultaneous: ArrayLike,
        /,
        *,
        ledger_id: str,
    ):
        code = jnp.asarray(event_code, dtype=jnp.int32)
        affine = jnp.asarray(affine_parameter)
        index = jnp.asarray(history_index, dtype=jnp.int32)
        state_ = jnp.asarray(state)
        state_valid_ = jnp.asarray(state_valid, dtype=jnp.bool_)
        if state_.shape[:-1] != code.shape or state_.shape[-1] < 8:
            raise ValueError(
                "GR event ledger state must have shape ray_batch_shape + (state_size >= 8,)."
            )
        recorded_ = jnp.asarray(recorded, dtype=jnp.bool_)
        simultaneous_ = jnp.asarray(simultaneous, dtype=jnp.bool_)
        if any(
            value.shape != code.shape
            for value in (
                affine,
                index,
                state_valid_,
                recorded_,
                simultaneous_,
            )
        ):
            raise ValueError("GR event ledger fields must have equal ray batch shapes.")
        if not isinstance(ledger_id, str) or not ledger_id:
            raise ValueError("ledger_id must be a non-empty string.")
        self.event_code = code
        self.affine_parameter = affine
        self.history_index = index
        self.state = state_
        self.state_valid = state_valid_
        self.recorded = recorded_
        self.simultaneous = simultaneous_
        self.ledger_id = ledger_id

    @property
    def coordinates(self) -> Array:
        return self.state[..., :4]

    @property
    def tangent(self) -> Array:
        return self.state[..., 4:8]


def ordered_gr_ray_event_code(
    triggered: ArrayLike,
    work_exhausted: ArrayLike,
    /,
) -> Array:
    """Resolve terminal event masks with capture > escape > domain > work priority."""

    masks = jnp.asarray(triggered, dtype=jnp.bool_)
    work = jnp.asarray(work_exhausted, dtype=jnp.bool_)
    if masks.shape[-1:] != (3,) or work.shape != masks.shape[:-1]:
        raise ValueError("triggered must end in three ordered event masks.")
    none = jnp.full(work.shape, int(GRRayEventCode.NONE), dtype=jnp.int32)
    code = jnp.where(work, int(GRRayEventCode.WORK), none)
    code = jnp.where(masks[..., 2], int(GRRayEventCode.DOMAIN), code)
    code = jnp.where(masks[..., 1], int(GRRayEventCode.ESCAPE), code)
    return jnp.where(masks[..., 0], int(GRRayEventCode.CAPTURE), code).astype(jnp.int32)


__all__ = [
    "GRRayEventCode",
    "GRRayEventLedger",
    "GRRayEventMargin",
    "GRRayEventSurfaces",
    "ordered_gr_ray_event_code",
]
