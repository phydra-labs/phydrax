#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._space import TensorSpectralDiscretization


class ModalDecayReport(StrictModule, NonTrainableState):
    """Physical tail norms and raw coefficient-decay evidence."""

    total_norm: Array
    tail_norms: Array
    relative_tail_norms: Array
    head_tail_inner_products: Array
    coefficient_envelope: Array
    local_log_slopes: Array
    rounding_floor_mask: Array
    zero_reference_norm: Array
    finite: Array
    active_mode_mask: Array
    diagnostics_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        total_norm: ArrayLike,
        tail_norms: ArrayLike,
        relative_tail_norms: ArrayLike,
        head_tail_inner_products: ArrayLike,
        coefficient_envelope: ArrayLike,
        local_log_slopes: ArrayLike,
        rounding_floor_mask: ArrayLike,
        zero_reference_norm: ArrayLike,
        finite: ArrayLike,
        active_mode_mask: ArrayLike,
        diagnostics_id: str,
        prepared_id: str,
    ):
        self.total_norm = jnp.asarray(total_norm)
        self.tail_norms = jnp.asarray(tail_norms)
        self.relative_tail_norms = jnp.asarray(relative_tail_norms)
        self.head_tail_inner_products = jnp.asarray(head_tail_inner_products)
        self.coefficient_envelope = jnp.asarray(coefficient_envelope)
        self.local_log_slopes = jnp.asarray(local_log_slopes)
        self.rounding_floor_mask = jnp.asarray(rounding_floor_mask, dtype=bool)
        self.zero_reference_norm = jnp.asarray(zero_reference_norm, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.active_mode_mask = jnp.asarray(active_mode_mask, dtype=bool)
        self.diagnostics_id = str(diagnostics_id)
        self.prepared_id = str(prepared_id)


class SpectralModalDiagnosticsPlan(StrictModule, NonTrainableState):
    """Static tail masks, rounding policy, and workspace budget."""

    discretization: TensorSpectralDiscretization
    tail_masks: tuple[Array, ...]
    active_mode_mask: Array
    tail_fraction: float = eqx.field(static=True)
    minimum_tail_modes: int = eqx.field(static=True)
    floor_multiplier: float = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_mode_count: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        tail_fraction: float = 1.0 / 3.0,
        minimum_tail_modes: int = 2,
        floor_multiplier: float = 32.0,
        maximum_workspace_bytes: int = 512 * 1024**2,
    ):
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        fraction = float(tail_fraction)
        minimum = int(minimum_tail_modes)
        multiplier = float(floor_multiplier)
        maximum = int(maximum_workspace_bytes)
        if (
            not 0.0 < fraction <= 1.0
            or minimum <= 0
            or not math.isfinite(multiplier)
            or multiplier <= 0.0
            or maximum <= 0
        ):
            raise ValueError("Modal diagnostics policy values are invalid.")
        masks = tuple(
            jnp.asarray(
                _tail_mask(
                    axis.family, np.asarray(axis.modes.mode_numbers), fraction, minimum
                )
            )
            for axis in discretization.axes
        )
        largest = max(discretization.modal_shape)
        active = np.zeros((len(discretization.axes), largest), dtype=bool)
        for axis, count in enumerate(discretization.modal_shape):
            active[axis, :count] = True
        itemsize = np.dtype(discretization.plan.precision.coefficient_dtype).itemsize
        workspace = (
            (len(discretization.axes) + 2)
            * int(prod(discretization.physical_shape))
            * itemsize
        )
        if workspace > maximum:
            raise ValueError("Modal diagnostics exceed maximum_workspace_bytes.")
        self.discretization = discretization
        self.tail_masks = masks
        self.active_mode_mask = jnp.asarray(active)
        self.tail_fraction = fraction
        self.minimum_tail_modes = minimum
        self.floor_multiplier = multiplier
        self.maximum_workspace_bytes = maximum
        self.maximum_mode_count = largest
        self.workspace_bytes = workspace
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-modal-diagnostics-plan",
                "discretization": discretization.prepared_id,
                "tail_fraction": fraction,
                "minimum_tail_modes": minimum,
                "floor_multiplier": multiplier,
                "maximum_workspace_bytes": maximum,
                "tail_masks": [np.asarray(mask).astype(int).tolist() for mask in masks],
            }
        )

    def prepare(self, /) -> "PreparedSpectralModalDiagnostics":
        return PreparedSpectralModalDiagnostics(self)


class PreparedSpectralModalDiagnostics(StrictModule, NonTrainableState):
    plan: SpectralModalDiagnosticsPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SpectralModalDiagnosticsPlan, /):
        if not isinstance(plan, SpectralModalDiagnosticsPlan):
            raise TypeError("plan must be a SpectralModalDiagnosticsPlan.")
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-modal-diagnostics",
                "plan": plan.plan_id,
            }
        )

    def evaluate(self, coefficients: ArrayLike, /) -> ModalDecayReport:
        discretization = self.plan.discretization
        modal = discretization._validate_leading(
            coefficients,
            discretization.modal_shape,
            "Modal diagnostics coefficients",
        )
        physical = discretization.reconstruct(modal, real_output=False)
        total_inner = _weighted_inner(discretization, physical, physical)
        total_norm = jnp.sqrt(jnp.maximum(jnp.real(total_inner), 0.0))
        zero = total_norm == 0.0
        tail_norms = []
        relative = []
        overlaps = []
        envelopes = []
        slopes = []
        floors = []
        spectral_rank = len(discretization.modal_shape)
        for axis, mask in enumerate(self.plan.tail_masks):
            shape = [1] * modal.ndim
            shape[axis] = mask.size
            tail_coefficients = modal * mask.reshape(tuple(shape))
            tail = discretization.reconstruct(tail_coefficients, real_output=False)
            head = physical - tail
            tail_inner = _weighted_inner(discretization, tail, tail)
            tail_norm = jnp.sqrt(jnp.maximum(jnp.real(tail_inner), 0.0))
            overlap = jnp.real(_weighted_inner(discretization, head, tail))
            tail_norms.append(tail_norm)
            relative.append(jnp.where(zero, 0.0, tail_norm / total_norm))
            overlaps.append(overlap)

            reduction_axes = tuple(
                index for index in range(spectral_rank) if index != axis
            )
            envelope = jnp.max(jnp.abs(modal), axis=reduction_axes, initial=0.0)
            padding = [(0, self.plan.maximum_mode_count - mask.size)] + [(0, 0)] * (
                envelope.ndim - 1
            )
            envelope = jnp.pad(envelope, tuple(padding))
            tiny = jnp.finfo(envelope.dtype).tiny
            log_envelope = jnp.log(jnp.maximum(envelope, tiny))
            slope = jnp.zeros_like(envelope)
            slope = slope.at[1:].set(log_envelope[1:] - log_envelope[:-1])
            maximum = jnp.max(envelope, axis=0, initial=0.0)
            floor = envelope <= (
                self.plan.floor_multiplier * jnp.finfo(envelope.dtype).eps * maximum[None]
            )
            envelopes.append(envelope)
            slopes.append(slope)
            floors.append(floor)

        tail_norm_array = jnp.stack(tail_norms, axis=0)
        relative_array = jnp.stack(relative, axis=0)
        overlap_array = jnp.stack(overlaps, axis=0)
        envelope_array = jnp.stack(envelopes, axis=0)
        slope_array = jnp.stack(slopes, axis=0)
        floor_array = (
            jnp.stack(floors, axis=0)
            | ~self.plan.active_mode_mask[(...,) + (None,) * (envelope_array.ndim - 2)]
        )
        finite = (
            jnp.all(jnp.isfinite(modal))
            & jnp.all(jnp.isfinite(total_norm))
            & jnp.all(jnp.isfinite(tail_norm_array))
            & jnp.all(jnp.isfinite(overlap_array))
            & jnp.all(jnp.isfinite(envelope_array))
        )
        diagnostics_id = canonical_fingerprint(
            {
                "kind": "modal-decay-report",
                "prepared": self.prepared_id,
                "coefficient_shape": list(modal.shape),
            }
        )
        return ModalDecayReport(
            total_norm=total_norm,
            tail_norms=tail_norm_array,
            relative_tail_norms=relative_array,
            head_tail_inner_products=overlap_array,
            coefficient_envelope=envelope_array,
            local_log_slopes=slope_array,
            rounding_floor_mask=floor_array,
            zero_reference_norm=zero,
            finite=finite,
            active_mode_mask=self.plan.active_mode_mask,
            diagnostics_id=diagnostics_id,
            prepared_id=self.prepared_id,
        )


def _weighted_inner(
    discretization: TensorSpectralDiscretization,
    left: Array,
    right: Array,
    /,
) -> Array:
    weights = discretization.quadrature_weights
    payload_rank = left.ndim - len(discretization.physical_shape)
    weighted = weights[(...,) + (None,) * payload_rank]
    axes = tuple(range(len(discretization.physical_shape)))
    return jnp.sum(jnp.conj(left) * right * weighted, axis=axes)


def _tail_mask(
    family: str,
    mode_numbers: np.ndarray,
    fraction: float,
    minimum: int,
    /,
) -> np.ndarray:
    count = int(mode_numbers.size)
    minimum_ = min(minimum, count)
    if family == "fourier":
        magnitudes = np.abs(mode_numbers)
        cutoff = (1.0 - fraction) * np.max(magnitudes)
        mask = magnitudes > cutoff
        if np.count_nonzero(mask) < minimum_:
            selected = np.argsort(magnitudes, kind="stable")[-minimum_:]
            mask[selected] = True
        return mask
    start = max(0, count - max(minimum_, int(math.ceil(fraction * count))))
    mask = np.zeros((count,), dtype=bool)
    mask[start:] = True
    return mask


__all__ = [
    "ModalDecayReport",
    "PreparedSpectralModalDiagnostics",
    "SpectralModalDiagnosticsPlan",
]
