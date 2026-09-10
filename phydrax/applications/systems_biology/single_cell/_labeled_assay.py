#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Four-channel pulse/chase count observation with calibrated label confusion."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax import ein
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import conversion_factor, SECOND, UnitDefinition

from ._scenario import _identity, _label, GeneIdentity


LABELED_CHANNELS = (
    "labeled-unspliced",
    "labeled-spliced",
    "unlabeled-unspliced",
    "unlabeled-spliced",
)


@dataclass(frozen=True, slots=True, init=False)
class LabeledTranscriptAssay:
    """Conditional count law for labeled/unlabeled U/S molecules.

    Columns of ``label_confusion`` are true label state and rows are reported
    label state, both ordered labeled, unlabeled. Capture is channel-specific and
    happens after classification. A true molecule is observed in at most one
    channel, so cross-label covariance is retained rather than four independent
    binomial laws being asserted.
    """

    capture_probabilities: Array
    background_rates: Array
    label_confusion: Array
    observation_probabilities: Array
    calibration_covariance: Array | None
    labeling_calibration: ReferenceArtifactManifest
    count_calibration: ReferenceArtifactManifest
    assay_id: str

    def __init__(
        self,
        capture_probabilities: ArrayLike,
        background_rates: ArrayLike,
        label_confusion: ArrayLike,
        /,
        *,
        labeling_calibration: ReferenceArtifactManifest,
        count_calibration: ReferenceArtifactManifest,
        calibration_covariance: ArrayLike | None = None,
    ):
        capture = np.asarray(capture_probabilities, dtype=float)
        background = np.asarray(background_rates, dtype=float)
        confusion = np.asarray(label_confusion, dtype=float)
        if (
            capture.shape != (4,)
            or not np.all(np.isfinite(capture))
            or np.any((capture < 0.0) | (capture > 1.0))
        ):
            raise ValueError(
                "Capture probabilities must be four finite values in [0, 1]."
            )
        if (
            background.shape != (4,)
            or not np.all(np.isfinite(background))
            or np.any(background < 0.0)
        ):
            raise ValueError("Background rates must be four finite nonnegative values.")
        if (
            confusion.shape != (2, 2)
            or not np.all(np.isfinite(confusion))
            or np.any((confusion < 0.0) | (confusion > 1.0))
            or not np.allclose(np.sum(confusion, axis=0), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(
                "Label-confusion columns must be calibrated probability distributions."
            )
        for manifest, name in (
            (labeling_calibration, "labeling calibration"),
            (count_calibration, "count calibration"),
        ):
            if not isinstance(manifest, ReferenceArtifactManifest):
                raise TypeError(f"{name} must be a ReferenceArtifactManifest.")
            manifest.require_rights()
            manifest.require_uncertainty()
        covariance = (
            None
            if calibration_covariance is None
            else np.asarray(calibration_covariance, dtype=float)
        )
        if covariance is not None and (
            covariance.shape != (12, 12)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12)
            or np.min(np.linalg.eigvalsh(covariance)) < -1e-12
        ):
            raise ValueError(
                "Assay calibration covariance must be a finite positive-semidefinite "
                "12x12 matrix over capture, background, and label-confusion parameters."
            )
        probabilities = np.zeros((4, 4), dtype=float)
        for true in range(4):
            true_label, splice = (0 if true < 2 else 1), true % 2
            for observed_label in range(2):
                observed = 2 * observed_label + splice
                probabilities[observed, true] = (
                    confusion[observed_label, true_label] * capture[observed]
                )
        if np.any(np.sum(probabilities, axis=0) > 1.0 + 1e-12):
            raise ValueError(
                "Classification and capture assign more than unit probability to a molecule."
            )
        object.__setattr__(self, "capture_probabilities", jnp.asarray(capture))
        object.__setattr__(self, "background_rates", jnp.asarray(background))
        object.__setattr__(self, "label_confusion", jnp.asarray(confusion))
        object.__setattr__(self, "observation_probabilities", jnp.asarray(probabilities))
        object.__setattr__(
            self,
            "calibration_covariance",
            None if covariance is None else jnp.asarray(covariance),
        )
        object.__setattr__(self, "labeling_calibration", labeling_calibration)
        object.__setattr__(self, "count_calibration", count_calibration)
        object.__setattr__(
            self,
            "assay_id",
            canonical_fingerprint(
                {
                    "kind": "labeled-transcript-count-assay",
                    "channels": LABELED_CHANNELS,
                    "capture": capture.tolist(),
                    "background": background.tolist(),
                    "label_confusion": confusion.tolist(),
                    "labeling_calibration": labeling_calibration.manifest_id,
                    "count_calibration": count_calibration.manifest_id,
                    "calibration_covariance": (
                        None if covariance is None else array_tree_fingerprint(covariance)
                    ),
                }
            ),
        )

    def conditional_moments(self, latent_counts: ArrayLike, /) -> tuple[Array, Array]:
        """Return conditional moments, including quantified assay calibration."""

        latent = jnp.asarray(latent_counts, dtype=float)
        if latent.shape[-1:] != (4,):
            raise ValueError("Latent counts must end in the four declared channels.")
        probabilities = self.observation_probabilities
        mean = latent @ probabilities.T + self.background_rates
        diagonal = ein.contract("...t,ot->...o", latent, probabilities)
        second = ein.contract("...t,ot,pt->...op", latent, probabilities, probabilities)
        covariance = -second
        indices = jnp.arange(4)
        covariance = covariance.at[..., indices, indices].add(diagonal)
        covariance = covariance.at[..., indices, indices].add(self.background_rates)
        if self.calibration_covariance is not None:
            parameters = jnp.concatenate(
                (
                    self.capture_probabilities,
                    self.background_rates,
                    self.label_confusion.reshape(-1),
                )
            )
            observed_labels = jnp.asarray((0, 0, 1, 1))
            splice_states = jnp.asarray((0, 1, 0, 1))

            def calibrated_mean(parameter_vector):
                capture = parameter_vector[:4]
                background = parameter_vector[4:8]
                confusion = parameter_vector[8:].reshape((2, 2))
                calibrated_probabilities = (
                    confusion[observed_labels[:, None], observed_labels[None, :]]
                    * capture[:, None]
                    * (splice_states[:, None] == splice_states[None, :])
                )
                return latent @ calibrated_probabilities.T + background

            sensitivity = jax.jacfwd(calibrated_mean)(parameters)
            covariance = covariance + ein.contract(
                "...oi,ij,...pj->...op",
                sensitivity,
                self.calibration_covariance,
                sensitivity,
            )
        return mean, covariance

    def sample(self, key: Key, latent_counts: ArrayLike, /) -> Array:
        """Sample exact mutually exclusive capture plus Poisson background."""

        raw = np.asarray(latent_counts)
        if (
            raw.dtype.kind not in "ifu"
            or raw.ndim < 1
            or raw.shape[-1] != 4
            or not np.all(np.isfinite(raw))
            or np.any(raw < 0.0)
            or np.any(raw != np.floor(raw))
        ):
            raise ValueError("Latent counts must be finite nonnegative integer values.")
        latent = jnp.asarray(raw, dtype=jnp.int32)
        observed = jnp.zeros(latent.shape, dtype=jnp.int32)
        keys = iter(jax.random.split(key, 9))
        for true in range(4):
            splice = true % 2
            destinations = (splice, splice + 2)
            remaining = latent[..., true]
            remaining_probability = jnp.ones(
                (), dtype=self.observation_probabilities.dtype
            )
            for destination in destinations:
                probability = self.observation_probabilities[destination, true]
                conditional = jnp.where(
                    remaining_probability > 0.0,
                    probability / remaining_probability,
                    0.0,
                )
                captured = jax.random.binomial(
                    next(keys), n=remaining, p=conditional
                ).astype(jnp.int32)
                observed = observed.at[..., destination].add(captured)
                remaining = remaining - captured
                remaining_probability = remaining_probability - probability
        background = jax.random.poisson(
            next(keys), self.background_rates, shape=latent.shape
        )
        return observed + background.astype(jnp.int32)


@dataclass(frozen=True, slots=True, init=False)
class LabeledTranscriptCounts:
    """Four-channel cell snapshots with physical time and preparation grouping."""

    gene: GeneIdentity
    cell_ids: tuple[int, ...]
    culture_ids: tuple[str, ...]
    plate_ids: tuple[str, ...]
    counts: Array
    valid: Array
    times: Array
    time_unit: UnitDefinition
    assay_id: str
    source_id: str
    source_parent_ids: tuple[str, ...]
    preprocessing_id: str
    preprocessing_parent_ids: tuple[str, ...]
    observation_id: str

    def __init__(
        self,
        gene: GeneIdentity,
        cell_ids: tuple[int, ...],
        counts: ArrayLike,
        /,
        *,
        culture_ids: tuple[str, ...],
        plate_ids: tuple[str, ...],
        times: ArrayLike,
        time_unit: UnitDefinition,
        assay_id: str,
        source_id: str,
        preprocessing_id: str,
        source_parent_ids: tuple[str, ...],
        preprocessing_parent_ids: tuple[str, ...],
        valid: ArrayLike | None = None,
    ):
        if not isinstance(gene, GeneIdentity):
            raise TypeError("gene must be GeneIdentity.")
        ids = tuple(_identity(value, "cell_id") for value in cell_ids)
        cultures = tuple(_label(value, "culture_id") for value in culture_ids)
        plates = tuple(_label(value, "plate_id") for value in plate_ids)
        source_parents = tuple(
            sorted(_label(value, "source_parent_id") for value in source_parent_ids)
        )
        preprocessing_parents = tuple(
            sorted(
                _label(value, "preprocessing_parent_id")
                for value in preprocessing_parent_ids
            )
        )
        if (
            not source_parents
            or len(set(source_parents)) != len(source_parents)
            or not preprocessing_parents
            or len(set(preprocessing_parents)) != len(preprocessing_parents)
        ):
            raise ValueError(
                "Source and preprocessing parent ancestry must be nonempty and unique."
            )
        if (
            not ids
            or len(set(ids)) != len(ids)
            or len(cultures) != len(ids)
            or len(plates) != len(ids)
        ):
            raise ValueError(
                "Unique cells and one culture/plate identity per snapshot are required."
            )
        raw = np.asarray(counts)
        mask = np.ones(raw.shape, dtype=bool) if valid is None else np.asarray(valid)
        if raw.shape != (len(ids), 4) or raw.dtype.kind not in "ifu":
            raise ValueError("Counts must have shape (cell, 4) in labeled U/S order.")
        if mask.shape != raw.shape or mask.dtype != bool:
            raise ValueError("Count validity must be a boolean mask matching counts.")
        active = raw[mask]
        if (
            np.any(~np.isfinite(active))
            or np.any(active < 0.0)
            or np.any(active != np.floor(active))
        ):
            raise ValueError("Active pulse/chase counts must be nonnegative integers.")
        coordinates = np.asarray(times, dtype=float)
        if coordinates.shape != (len(ids),) or not np.all(np.isfinite(coordinates)):
            raise ValueError("Physical times must be finite with one value per cell.")
        conversion_factor(time_unit, SECOND)
        object.__setattr__(self, "gene", gene)
        object.__setattr__(self, "cell_ids", ids)
        object.__setattr__(self, "culture_ids", cultures)
        object.__setattr__(self, "plate_ids", plates)
        object.__setattr__(
            self, "counts", jnp.asarray(np.where(mask, raw, 0), dtype=float)
        )
        object.__setattr__(self, "valid", jnp.asarray(mask))
        object.__setattr__(self, "times", jnp.asarray(coordinates))
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(self, "assay_id", _label(assay_id, "assay_id"))
        object.__setattr__(self, "source_id", _label(source_id, "source_id"))
        object.__setattr__(self, "source_parent_ids", source_parents)
        object.__setattr__(
            self, "preprocessing_id", _label(preprocessing_id, "preprocessing_id")
        )
        object.__setattr__(self, "preprocessing_parent_ids", preprocessing_parents)
        object.__setattr__(
            self,
            "observation_id",
            canonical_fingerprint(
                {
                    "kind": "labeled-transcript-counts",
                    "gene": (gene.gene_id, gene.label),
                    "cells": ids,
                    "cultures": cultures,
                    "plates": plates,
                    "counts": array_tree_fingerprint(np.asarray(self.counts)),
                    "valid": array_tree_fingerprint(mask),
                    "times": coordinates.tolist(),
                    "time_unit": time_unit.unit_id,
                    "assay": assay_id,
                    "source": source_id,
                    "source_parents": source_parents,
                    "preprocessing": preprocessing_id,
                    "preprocessing_parents": preprocessing_parents,
                    "channels": LABELED_CHANNELS,
                }
            ),
        )


def observe_labeled_transcripts(
    key: Key,
    latent_counts: ArrayLike,
    assay: LabeledTranscriptAssay,
    /,
    *,
    gene: GeneIdentity,
    cell_ids: tuple[int, ...],
    culture_ids: tuple[str, ...],
    plate_ids: tuple[str, ...],
    times: ArrayLike,
    time_unit: UnitDefinition,
    source_id: str,
    preprocessing_id: str,
    source_parent_ids: tuple[str, ...],
    preprocessing_parent_ids: tuple[str, ...],
) -> LabeledTranscriptCounts:
    """Observe supplied latent snapshots; no lineage is inferred between cells."""

    if not isinstance(assay, LabeledTranscriptAssay):
        raise TypeError("assay must be LabeledTranscriptAssay.")
    observed = assay.sample(key, latent_counts)
    return LabeledTranscriptCounts(
        gene,
        cell_ids,
        observed,
        culture_ids=culture_ids,
        plate_ids=plate_ids,
        times=times,
        time_unit=time_unit,
        assay_id=assay.assay_id,
        source_id=source_id,
        preprocessing_id=preprocessing_id,
        source_parent_ids=source_parent_ids,
        preprocessing_parent_ids=preprocessing_parent_ids,
    )


__all__ = [
    "LABELED_CHANNELS",
    "LabeledTranscriptAssay",
    "LabeledTranscriptCounts",
    "observe_labeled_transcripts",
]
