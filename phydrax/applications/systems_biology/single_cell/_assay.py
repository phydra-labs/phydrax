#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Count capture is a separate, calibrated observation of latent transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.series import SampledSeries, SeriesSupport
from phydrax.units import conversion_factor, SECOND, UnitDefinition

from .._gene_expression import CountMeasurementPlan, PreparedCountMeasurement
from ._scenario import _address, _identity, _label, GeneIdentity, TranscriptExperiment


@dataclass(frozen=True, slots=True)
class TranscriptCountAssay:
    """Independent U/S binomial capture + Poisson background, with source admission.

    The caller supplies independent calibration and its actual uncertainty. This
    class does not infer capture efficiency from the same expression counts or
    silently turn a reference citation into validated biological calibration.
    """

    unspliced: PreparedCountMeasurement
    spliced: PreparedCountMeasurement
    calibration: ReferenceArtifactManifest

    def __post_init__(self):
        if not isinstance(self.unspliced, PreparedCountMeasurement) or not isinstance(
            self.spliced, PreparedCountMeasurement
        ):
            raise TypeError("Assay channels must be prepared native count measurements.")
        if not isinstance(self.calibration, ReferenceArtifactManifest):
            raise TypeError("An independent count calibration manifest is required.")
        self.calibration.require_rights()
        self.calibration.require_uncertainty()

    @classmethod
    def from_plans(
        cls,
        unspliced: CountMeasurementPlan,
        spliced: CountMeasurementPlan,
        calibration: ReferenceArtifactManifest,
        /,
    ) -> TranscriptCountAssay:
        return cls(unspliced.prepare(), spliced.prepare(), calibration)

    @property
    def assay_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "transcript-count-assay",
                "U": self.unspliced.measurement_id,
                "S": self.spliced.measurement_id,
                "calibration": self.calibration.manifest_id,
            }
        )


@dataclass(frozen=True, slots=True, init=False)
class TranscriptCounts:
    """One gene's measured snapshots, not latent states or a inferred lineage.

    Masks remain separate from integer counts. Coordinates are explicitly physical
    time, pseudotime, or absent. Snapshot rows have no connected trajectory edges.
    """

    gene: GeneIdentity
    cell_ids: tuple[int, ...]
    counts: Array
    valid: Array
    coordinates: Array
    coordinate_semantics: Literal["physical_time", "pseudotime", "none"]
    time_unit: UnitDefinition | None
    assay_id: str
    source_id: str
    preprocessing_id: str
    observation_id: str

    def __init__(
        self,
        gene: GeneIdentity,
        cell_ids: tuple[int, ...],
        counts: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        coordinates: ArrayLike | None = None,
        coordinate_semantics: Literal["physical_time", "pseudotime", "none"],
        time_unit: UnitDefinition | None = None,
        assay_id: str,
        source_id: str,
        preprocessing_id: str,
    ):
        if not isinstance(gene, GeneIdentity):
            raise TypeError("gene must be GeneIdentity.")
        ids = tuple(_identity(x, "cell_id") for x in cell_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Count snapshots require nonempty unique cell identities.")
        raw = np.asarray(counts)
        if raw.shape != (len(ids), 2) or raw.dtype.kind not in "ifu":
            raise ValueError("Measured counts must have shape (cell, 2), ordered U,S.")
        mask = np.ones(raw.shape, dtype=bool) if valid is None else np.asarray(valid)
        if mask.dtype != bool or mask.shape != raw.shape:
            raise ValueError("Count validity must be a boolean mask matching counts.")
        active = raw[mask]
        if (
            np.any(~np.isfinite(active))
            or np.any(active < 0)
            or np.any(active != np.floor(active))
        ):
            raise ValueError(
                "Active measured counts must be nonnegative finite integers; transformed expression is not counts."
            )
        if coordinate_semantics not in ("physical_time", "pseudotime", "none"):
            raise ValueError("Coordinate meaning must be declared explicitly.")
        if coordinate_semantics == "physical_time":
            if time_unit is None:
                raise ValueError("Physical-time observations require an exact time unit.")
            conversion_factor(time_unit, SECOND)
        elif time_unit is not None:
            raise ValueError(
                "Pseudotime/absent coordinates cannot carry a physical-time unit."
            )
        if coordinate_semantics == "none":
            if coordinates is not None:
                raise ValueError("Absent coordinates must not contain values.")
            coords = np.zeros(len(ids))
        else:
            coords = np.asarray(coordinates, dtype=float)
            if coords.shape != (len(ids),) or np.any(~np.isfinite(coords)):
                raise ValueError(
                    "Declared coordinates must be finite with one value per cell."
                )
        for name, value in (
            ("gene", gene),
            ("cell_ids", ids),
            ("counts", jnp.asarray(np.where(mask, raw, 0), dtype=float)),
            ("valid", jnp.asarray(mask)),
            ("coordinates", jnp.asarray(coords)),
            ("coordinate_semantics", coordinate_semantics),
            ("time_unit", time_unit),
            ("assay_id", _label(assay_id, "assay_id")),
            ("source_id", _label(source_id, "source_id")),
            ("preprocessing_id", _label(preprocessing_id, "preprocessing_id")),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "observation_id",
            canonical_fingerprint(
                {
                    "kind": "measured-transcript-snapshots",
                    "gene": (gene.gene_id, gene.label),
                    "cells": ids,
                    "counts": array_tree_fingerprint(np.asarray(self.counts)),
                    "mask": array_tree_fingerprint(mask),
                    "coordinates": coords.tolist(),
                    "coordinate_semantics": coordinate_semantics,
                    "time_unit": None if time_unit is None else time_unit.unit_id,
                    "assay": assay_id,
                    "source": source_id,
                    "preprocessing": preprocessing_id,
                }
            ),
        )

    def to_series(self) -> SampledSeries:
        support = SeriesSupport(
            self.coordinates,
            node_valid=jnp.any(self.valid, axis=-1),
            edge_valid=jnp.zeros((len(self.cell_ids) - 1,), dtype=bool),
            coordinate_name=self.coordinate_semantics,
            coordinate_id=self.observation_id + ":coordinates",
        )
        return SampledSeries(
            support, self.counts, value_valid=self.valid, series_id=self.observation_id
        )


def observe_transcripts(
    experiment: TranscriptExperiment,
    assay: TranscriptCountAssay,
    key: Array,
    /,
    *,
    gene_id: int,
    segment_id: int,
    sample_time: float | None = None,
) -> TranscriptCounts:
    """Capture one actual saved snapshot per cell using independent assay randomness.

    Noise is addressed by cell/gene/segment, physical sample time, channel and
    capture/background. It is stable under cell workset changes and uses a namespace
    disjoint from latent SSA, even if the caller passes the same root key.
    """
    paths = tuple(
        p
        for p in experiment.paths
        if p.gene.gene_id == gene_id and p.segment.segment_id == segment_id
    )
    if not paths:
        raise ValueError("No generated paths match the requested gene and segment.")
    selected_time = (
        paths[0].segment.schedule.boundaries[-1]
        if sample_time is None
        else float(sample_time)
    )
    time_bits = int(np.asarray(selected_time, dtype=np.float64).view(np.uint64))
    observed = []
    for path in paths:
        matches = np.flatnonzero(
            np.asarray(path.latent.support.coordinates) == selected_time
        )
        if matches.size != 1:
            raise ValueError(
                "The observation time must be an actually saved physical node."
            )
        latent = path.latent.values[int(matches[0]), 1:]
        path_key = _address(key, 1, path.cell.cell_id, gene_id, segment_id, time_bits)
        channels = []
        for channel, measurement in enumerate((assay.unspliced, assay.spliced)):
            capture_key = _address(path_key, channel, 0)
            background_key = _address(path_key, channel, 1)
            captured = jax.random.binomial(
                capture_key, latent[channel], measurement.plan.capture_probability
            )
            background = jax.random.poisson(
                background_key, measurement.plan.background_rate
            )
            measured = captured + background
            if float(measured) > measurement.plan.observation_capacity:
                raise ValueError(
                    "Measured counts exceed the assay capacity; no clipping is permitted."
                )
            channels.append(measured)
        observed.append(jnp.stack(channels))
    source = canonical_fingerprint(
        {
            "paths": [p.path_id for p in paths],
            "assay_key": array_tree_fingerprint(np.asarray(jax.random.key_data(key))),
            "time": selected_time,
        }
    )
    return TranscriptCounts(
        paths[0].gene,
        tuple(p.cell.cell_id for p in paths),
        jnp.stack(observed),
        coordinates=jnp.full((len(paths),), selected_time),
        coordinate_semantics="physical_time",
        time_unit=paths[0].segment.schedule.time_unit,
        assay_id=assay.assay_id,
        source_id=source,
        preprocessing_id="native-independent-capture-background",
    )
