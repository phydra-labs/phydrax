#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coordinate-aware modal and frequency-response correlation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ._core import modal_assurance_criterion
from ._modal import modal_pairing_cost


@dataclass(frozen=True, slots=True)
class ModalCorrelationResult:
    modal_assurance: Array
    reference_to_candidate: tuple[int, ...]
    matched_modal_assurance: Array
    relative_frequency_error: Array
    accepted: Array


@dataclass(frozen=True, slots=True)
class FrequencyResponseCorrelationResult:
    assurance: Array
    amplitude_error: Array
    accepted: Array


def _minimum_cost_assignment(cost: np.ndarray) -> tuple[int, ...]:
    rows, columns = cost.shape
    if rows > columns:
        raise ValueError("Modal pairing requires at least as many candidate modes.")
    if columns > 16:
        raise ValueError("Exact modal pairing is bounded to sixteen candidate modes.")
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row in range(rows):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (total, assignment) in states.items():
            for column in range(columns):
                bit = 1 << column
                if mask & bit:
                    continue
                candidate = (total + float(cost[row, column]), assignment + (column,))
                existing = next_states.get(mask | bit)
                if existing is None or candidate[0] < existing[0]:
                    next_states[mask | bit] = candidate
        states = next_states
    return min(states.values(), key=lambda value: value[0])[1]


def correlate_modes(
    reference_frequencies_hz: ArrayLike,
    reference_modes: ArrayLike,
    candidate_frequencies_hz: ArrayLike,
    candidate_modes: ArrayLike,
    coordinate_map: ArrayLike,
    /,
    *,
    minimum_mac: float = 0.9,
    maximum_relative_frequency_error: float = 0.05,
    frequency_weight: float = 1.0,
) -> ModalCorrelationResult:
    if (
        np.iscomplexobj(reference_frequencies_hz)
        or np.iscomplexobj(candidate_frequencies_hz)
        or not isfinite(minimum_mac)
        or not 0 <= minimum_mac <= 1
        or not isfinite(maximum_relative_frequency_error)
        or maximum_relative_frequency_error < 0
        or not isfinite(frequency_weight)
        or frequency_weight < 0
    ):
        raise ValueError("Modal correlation frequencies and thresholds are invalid.")
    reference_frequency = jnp.asarray(reference_frequencies_hz, dtype=jnp.float64)
    candidate_frequency = jnp.asarray(candidate_frequencies_hz, dtype=jnp.float64)
    reference = jnp.asarray(reference_modes)
    candidate = jnp.asarray(candidate_modes)
    mapping = jnp.asarray(coordinate_map)
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("Modal correlation requires coordinate-by-mode matrices.")
    if reference_frequency.ndim != 1 or candidate_frequency.ndim != 1:
        raise ValueError("Modal correlation frequencies must be rank-one vectors.")
    if reference.shape[1] != reference_frequency.size:
        raise ValueError("Reference frequencies and modes do not align.")
    if candidate.shape[1] != candidate_frequency.size:
        raise ValueError("Candidate frequencies and modes do not align.")
    if mapping.shape != (reference.shape[0], candidate.shape[0]):
        raise ValueError("Correlation coordinate map has incompatible shape.")
    if (
        not all(
            np.all(np.isfinite(np.asarray(value)))
            for value in (
                reference_frequency,
                candidate_frequency,
                reference,
                candidate,
                mapping,
            )
        )
        or np.any(np.asarray(reference_frequency) < 0)
        or np.any(np.asarray(candidate_frequency) < 0)
    ):
        raise ValueError(
            "Modal correlation inputs must be finite with nonnegative frequencies."
        )
    mapped_candidate = mapping @ candidate
    mac = modal_assurance_criterion(reference, mapped_candidate)
    cost = modal_pairing_cost(
        mac,
        reference_frequency,
        candidate_frequency,
        frequency_weight,
    )
    pairing = _minimum_cost_assignment(np.asarray(cost))
    columns = jnp.asarray(pairing, dtype=jnp.int32)
    rows = jnp.arange(reference_frequency.size)
    matched_mac = mac[rows, columns]
    frequency_error = jnp.abs(
        candidate_frequency[columns] - reference_frequency
    ) / jnp.maximum(
        jnp.abs(reference_frequency), jnp.finfo(reference_frequency.dtype).tiny
    )
    accepted = (matched_mac >= float(minimum_mac)) & (
        frequency_error <= float(maximum_relative_frequency_error)
    )
    return ModalCorrelationResult(mac, pairing, matched_mac, frequency_error, accepted)


def correlate_frequency_responses(
    reference_response: ArrayLike,
    candidate_response: ArrayLike,
    /,
    *,
    minimum_assurance: float = 0.9,
    maximum_relative_amplitude_error: float = 0.1,
) -> FrequencyResponseCorrelationResult:
    reference = jnp.asarray(reference_response)
    if (
        not isfinite(minimum_assurance)
        or not 0 <= minimum_assurance <= 1
        or not isfinite(maximum_relative_amplitude_error)
        or maximum_relative_amplitude_error < 0
    ):
        raise ValueError("Frequency-response correlation thresholds are invalid.")
    candidate = jnp.asarray(candidate_response)
    if reference.shape != candidate.shape or reference.ndim < 1:
        raise ValueError("Frequency responses must have identical non-scalar shapes.")
    reference = eqx.error_if(
        reference,
        jnp.any(~jnp.isfinite(reference) | ~jnp.isfinite(candidate)),
        "Frequency responses must be finite.",
    )
    numerator = jnp.abs(jnp.sum(jnp.conj(reference) * candidate, axis=0)) ** 2
    reference_power = jnp.sum(jnp.abs(reference) ** 2, axis=0)
    candidate_power = jnp.sum(jnp.abs(candidate) ** 2, axis=0)
    assurance = numerator / jnp.maximum(
        reference_power * candidate_power, jnp.finfo(reference_power.dtype).tiny
    )
    amplitude_error = jnp.sqrt(
        jnp.sum(jnp.abs(candidate - reference) ** 2, axis=0)
        / jnp.maximum(reference_power, jnp.finfo(reference_power.dtype).tiny)
    )
    accepted = (assurance >= float(minimum_assurance)) & (
        amplitude_error <= float(maximum_relative_amplitude_error)
    )
    return FrequencyResponseCorrelationResult(assurance, amplitude_error, accepted)


__all__ = [
    "FrequencyResponseCorrelationResult",
    "ModalCorrelationResult",
    "correlate_frequency_responses",
    "correlate_modes",
]
