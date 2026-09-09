#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact first-moment pulse/chase dynamics for four transcript channels."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.linalg import matrix_exponential_action, MatrixFunctionPolicy
from phydrax.units import convert_value, derived_unit, SECOND, UnitDefinition


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return value


def _identifiers(values: tuple[str, ...], name: str, /) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple of identifiers.")
    result = tuple(_identifier(value, name) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique.")
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True, init=False)
class PulseChaseSchedule:
    """Known piecewise-constant labeling and kinetic schedule.

    Latent channels are ordered labeled-U, labeled-S, unlabeled-U, unlabeled-S.
    ``rates`` is ordered synthesis, splicing, degradation, dilution and has shape
    ``(interval, gene, 4)``. Label fractions affect newly synthesized transcripts
    only; a schedule transition never relabels molecules already present.
    """

    boundaries: tuple[float, ...]
    rates: Array
    label_fractions: Array
    time_unit: UnitDefinition
    schedule_id: str

    def __init__(
        self,
        boundaries: ArrayLike,
        rates: ArrayLike,
        label_fractions: ArrayLike,
        /,
        *,
        rate_unit: UnitDefinition,
        time_unit: UnitDefinition = SECOND,
    ):
        raw_boundaries = np.asarray(boundaries, dtype=float)
        raw_rates = np.asarray(rates)
        fractions = np.asarray(label_fractions, dtype=float)
        if (
            raw_boundaries.ndim != 1
            or raw_boundaries.size < 2
            or not np.all(np.isfinite(raw_boundaries))
            or np.any(np.diff(raw_boundaries) <= 0.0)
        ):
            raise ValueError(
                "Pulse/chase boundaries must be finite and strictly increasing."
            )
        if (
            raw_rates.dtype.kind not in "ifu"
            or raw_rates.ndim != 3
            or raw_rates.shape[0] != raw_boundaries.size - 1
            or raw_rates.shape[1] == 0
            or raw_rates.shape[2] != 4
            or not np.all(np.isfinite(raw_rates))
            or np.any(raw_rates < 0.0)
        ):
            raise ValueError(
                "Pulse/chase rates must be finite nonnegative (interval, gene, 4) values."
            )
        if np.any(raw_rates[..., 1:] <= 0.0):
            raise ValueError(
                "Splicing, degradation, and dilution rates must be strictly positive."
            )
        if (
            fractions.shape != (raw_boundaries.size - 1,)
            or not np.all(np.isfinite(fractions))
            or np.any((fractions < 0.0) | (fractions > 1.0))
        ):
            raise ValueError("Each interval needs a finite labeling fraction in [0, 1].")
        runtime_unit = derived_unit(f"1/({time_unit.symbol})", ((time_unit, -1),))
        runtime_rates = convert_value(
            jnp.asarray(raw_rates, dtype=float), source=rate_unit, target=runtime_unit
        )
        runtime_host = np.asarray(runtime_rates)
        if not np.all(np.isfinite(runtime_host)) or np.any(runtime_host < 0.0):
            raise ValueError("Rate conversion must preserve finite nonnegative values.")
        object.__setattr__(
            self, "boundaries", tuple(float(value) for value in raw_boundaries)
        )
        object.__setattr__(self, "rates", runtime_rates)
        object.__setattr__(self, "label_fractions", jnp.asarray(fractions))
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(
            self,
            "schedule_id",
            canonical_fingerprint(
                {
                    "kind": "single-cell-pulse-chase-schedule",
                    "boundaries": raw_boundaries.tolist(),
                    "rates": array_tree_fingerprint(runtime_host),
                    "label_fractions": fractions.tolist(),
                    "time_unit": time_unit.unit_id,
                    "channel_order": (
                        "labeled-unspliced",
                        "labeled-spliced",
                        "unlabeled-unspliced",
                        "unlabeled-spliced",
                    ),
                }
            ),
        )


def transient_labeled_transcript_mean(
    rates: ArrayLike,
    label_fraction: ArrayLike,
    initial_mean: ArrayLike,
    duration: ArrayLike,
    /,
) -> Array:
    """Apply one exact affine first-moment interval to labeled/unlabeled U/S."""

    synthesis, splicing, degradation, dilution = jnp.asarray(rates)
    label = jnp.asarray(label_fraction, dtype=jnp.asarray(rates).dtype)
    zero = jnp.zeros_like(synthesis)
    loss_u = splicing + dilution
    loss_s = degradation + dilution
    matrix = jnp.stack(
        (
            jnp.stack((-loss_u, zero, zero, zero, label * synthesis)),
            jnp.stack((splicing, -loss_s, zero, zero, zero)),
            jnp.stack((zero, zero, -loss_u, zero, (1.0 - label) * synthesis)),
            jnp.stack((zero, zero, splicing, -loss_s, zero)),
            jnp.stack((zero, zero, zero, zero, zero)),
        )
    )
    initial = jnp.concatenate(
        (
            jnp.asarray(initial_mean, dtype=matrix.dtype),
            jnp.ones((1,), dtype=matrix.dtype),
        )
    )
    if initial.shape != (5,):
        raise ValueError("initial_mean must have four labeled/unlabeled U/S values.")
    result = matrix_exponential_action(
        lambda value: matrix @ value,
        initial,
        duration,
        policy=MatrixFunctionPolicy("arnoldi", max_dimension=5),
    )
    return jnp.where(result.converged, result.value[:4], jnp.nan)


def scheduled_labeled_transcript_mean(
    schedule: PulseChaseSchedule,
    initial_mean: ArrayLike,
    /,
    *,
    gene_index: int = 0,
) -> Array:
    """Return latent means at all exact boundaries of a declared schedule."""

    if not isinstance(schedule, PulseChaseSchedule):
        raise TypeError("schedule must be PulseChaseSchedule.")
    if not 0 <= gene_index < schedule.rates.shape[1]:
        raise ValueError("gene_index is outside the schedule's gene support.")
    state = jnp.asarray(initial_mean, dtype=schedule.rates.dtype)
    if state.shape != (4,) or not np.all(np.isfinite(np.asarray(state))):
        raise ValueError("initial_mean must contain four finite channel means.")
    if np.any(np.asarray(state) < 0.0):
        raise ValueError("Initial transcript means must be nonnegative.")
    values = [state]
    for interval, (start, stop) in enumerate(
        zip(schedule.boundaries[:-1], schedule.boundaries[1:], strict=True)
    ):
        state = transient_labeled_transcript_mean(
            schedule.rates[interval, gene_index],
            schedule.label_fractions[interval],
            state,
            stop - start,
        )
        values.append(state)
    return jnp.stack(values)


@dataclass(frozen=True, slots=True, init=False)
class PulseChasePrediction:
    """Frozen latent prediction with explicit fit lineage and observation identity."""

    model_id: str
    schedule_id: str
    assay_id: str
    preprocessing_id: str
    fit_observation_ids: tuple[str, ...]
    fit_culture_ids: tuple[str, ...]
    fit_plate_ids: tuple[str, ...]
    latent_means: Array
    latent_covariance: Array
    prediction_id: str

    def __init__(
        self,
        latent_means: ArrayLike,
        latent_covariance: ArrayLike,
        /,
        *,
        model_id: str,
        schedule_id: str,
        assay_id: str,
        preprocessing_id: str,
        fit_observation_ids: tuple[str, ...],
        fit_culture_ids: tuple[str, ...],
        fit_plate_ids: tuple[str, ...],
    ):
        means = np.asarray(latent_means, dtype=float)
        covariance = np.asarray(latent_covariance, dtype=float)
        expected_covariance_shape = (*means.shape[:-1], 4, 4)
        if (
            means.ndim < 2
            or means.shape[-1] != 4
            or covariance.shape != expected_covariance_shape
            or not np.all(np.isfinite(means))
            or np.any(means < 0.0)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(
                covariance,
                np.swapaxes(covariance, -1, -2),
                rtol=1e-12,
                atol=1e-12,
            )
            or np.any(np.linalg.eigvalsh(covariance) < -1e-12)
        ):
            raise ValueError(
                "Latent means and positive-semidefinite covariance must be finite, "
                "aligned, and end in the four pulse/chase channels."
            )
        model = _identifier(model_id, "model_id")
        schedule = _identifier(schedule_id, "schedule_id")
        assay = _identifier(assay_id, "assay_id")
        preprocessing = _identifier(preprocessing_id, "preprocessing_id")
        observations = _identifiers(fit_observation_ids, "fit_observation_ids")
        cultures = _identifiers(fit_culture_ids, "fit_culture_ids")
        plates = _identifiers(fit_plate_ids, "fit_plate_ids")
        object.__setattr__(self, "model_id", model)
        object.__setattr__(self, "schedule_id", schedule)
        object.__setattr__(self, "assay_id", assay)
        object.__setattr__(self, "preprocessing_id", preprocessing)
        object.__setattr__(self, "fit_observation_ids", observations)
        object.__setattr__(self, "fit_culture_ids", cultures)
        object.__setattr__(self, "fit_plate_ids", plates)
        object.__setattr__(self, "latent_means", jnp.asarray(means))
        object.__setattr__(self, "latent_covariance", jnp.asarray(covariance))
        object.__setattr__(
            self,
            "prediction_id",
            canonical_fingerprint(
                {
                    "kind": "pulse-chase-frozen-prediction",
                    "model": model,
                    "schedule": schedule,
                    "assay": assay,
                    "preprocessing": preprocessing,
                    "fit_observations": observations,
                    "fit_cultures": cultures,
                    "fit_plates": plates,
                    "latent_means": array_tree_fingerprint(means),
                    "latent_covariance": array_tree_fingerprint(covariance),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PulseChaseIdentifiability:
    """Row-space evidence bound to one fitted model, schedule, assay, and data."""

    parameter_names: tuple[str, ...]
    sensitivity: Array
    singular_values: Array
    identified_combinations: Array
    unidentifiable_combinations: Array
    rank: int
    relative_tolerance: float
    model_id: str
    schedule_id: str
    assay_id: str
    preprocessing_id: str
    fit_observation_ids: tuple[str, ...]
    fit_culture_ids: tuple[str, ...]
    fit_plate_ids: tuple[str, ...]
    evidence_id: str

    @property
    def all_parameters_identifiable(self) -> bool:
        return self.rank == len(self.parameter_names)


def pulse_chase_identifiability(
    sensitivity: ArrayLike,
    parameter_names: tuple[str, ...],
    /,
    *,
    model_id: str,
    schedule_id: str,
    assay_id: str,
    preprocessing_id: str,
    fit_observation_ids: tuple[str, ...],
    fit_culture_ids: tuple[str, ...],
    fit_plate_ids: tuple[str, ...],
    relative_tolerance: float = 1e-8,
) -> PulseChaseIdentifiability:
    """Diagnose combinations for one exact pulse/chase fit and no other."""

    names = tuple(parameter_names)
    values = np.asarray(sensitivity, dtype=float)
    if (
        not names
        or len(set(names)) != len(names)
        or any(not name or name != name.strip() for name in names)
    ):
        raise ValueError("Parameter names must be unique nonempty canonical strings.")
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != len(names)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "Sensitivity must be a finite nonempty (observation, parameter) matrix."
        )
    if not np.isfinite(relative_tolerance) or not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be finite and in (0, 1).")
    model = _identifier(model_id, "model_id")
    schedule = _identifier(schedule_id, "schedule_id")
    assay = _identifier(assay_id, "assay_id")
    preprocessing = _identifier(preprocessing_id, "preprocessing_id")
    observations = _identifiers(fit_observation_ids, "fit_observation_ids")
    cultures = _identifiers(fit_culture_ids, "fit_culture_ids")
    plates = _identifiers(fit_plate_ids, "fit_plate_ids")
    _, singular_values, right = np.linalg.svd(values, full_matrices=True)
    scale = singular_values[0] if singular_values.size else 0.0
    rank = int(np.sum(singular_values > relative_tolerance * scale)) if scale else 0
    identified = right[:rank]
    unidentified = right[rank:]
    evidence_id = canonical_fingerprint(
        {
            "kind": "pulse-chase-local-identifiability",
            "parameters": names,
            "sensitivity": array_tree_fingerprint(values),
            "relative_tolerance": relative_tolerance,
            "rank": rank,
            "model": model,
            "schedule": schedule,
            "assay": assay,
            "preprocessing": preprocessing,
            "fit_observations": observations,
            "fit_cultures": cultures,
            "fit_plates": plates,
        }
    )
    return PulseChaseIdentifiability(
        names,
        jnp.asarray(values),
        jnp.asarray(singular_values),
        jnp.asarray(identified),
        jnp.asarray(unidentified),
        rank,
        float(relative_tolerance),
        model,
        schedule,
        assay,
        preprocessing,
        observations,
        cultures,
        plates,
        evidence_id,
    )


__all__ = [
    "PulseChasePrediction",
    "PulseChaseIdentifiability",
    "PulseChaseSchedule",
    "pulse_chase_identifiability",
    "scheduled_labeled_transcript_mean",
    "transient_labeled_transcript_mean",
]
