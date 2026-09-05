#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit array adapters for AnnData/scVelo-style exports, with no provider import.

Callers extract raw U/S columns and stable source-to-cell identities themselves.
Normalized/log counts are refused rather than silently cast back to molecules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.interchange import AdapterReport, AdapterStatus
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import UnitDefinition

from ._assay import TranscriptCounts
from ._scenario import _label, GeneIdentity


@dataclass(frozen=True, slots=True)
class ImportedTranscriptCounts:
    raw_unspliced: Array
    raw_spliced: Array
    counts: TranscriptCounts
    source: ReferenceArtifactManifest
    report: AdapterReport


def import_transcript_arrays(
    unspliced: ArrayLike,
    spliced: ArrayLike,
    /,
    *,
    gene: GeneIdentity,
    cell_ids: tuple[int, ...],
    source: ReferenceArtifactManifest,
    assay_id: str,
    preprocessing_id: str,
    coordinate_semantics: Literal["physical_time", "pseudotime", "none"],
    coordinates: ArrayLike | None = None,
    time_unit: UnitDefinition | None = None,
    valid: ArrayLike | None = None,
    commercial_use: bool = False,
    training_use: bool = False,
    redistribution: bool = False,
    export: bool = False,
) -> ImportedTranscriptCounts:
    """Import explicitly mapped raw count columns; missing entries retain a mask."""
    if not isinstance(source, ReferenceArtifactManifest):
        raise TypeError("source must be an admitted reference artifact manifest.")
    source.require_rights(
        commercial_use=commercial_use,
        training_use=training_use,
        redistribution=redistribution,
        export=export,
    )
    raw_u, raw_s = jnp.asarray(unspliced), jnp.asarray(spliced)
    if raw_u.shape != (len(cell_ids),) or raw_s.shape != raw_u.shape:
        raise ValueError(
            "Each extracted count column must match the declared stable cell support."
        )
    counts = TranscriptCounts(
        gene,
        cell_ids,
        jnp.stack((raw_u, raw_s), axis=-1),
        valid=valid,
        coordinates=coordinates,
        coordinate_semantics=coordinate_semantics,
        time_unit=time_unit,
        assay_id=assay_id,
        source_id=source.manifest_id,
        preprocessing_id=preprocessing_id,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "explicit-raw-transcript-columns",
        "native-transcript-counts",
        source_id=source.manifest_id,
        target_id=counts.observation_id,
        coordinate_mapping=(
            "source-row->declared-cell-id",
            "source-column->declared-gene-id",
        ),
        preserved_fields=(
            "raw_unspliced",
            "raw_spliced",
            "valid",
            "cell_ids",
            "gene_id",
            "coordinates",
            "coordinate_semantics",
            "preprocessing_id",
            "source_rights",
        ),
        assumptions=(
            "Caller-selected columns are raw transcript counts, not normalized expression.",
            "Cell rows are snapshots; no trajectory or physical lineage is inferred.",
        ),
    )
    return ImportedTranscriptCounts(raw_u, raw_s, counts, source, report)


@dataclass(frozen=True, slots=True)
class ImportedVelocityField:
    values: Array
    valid: Array
    standard_errors: Array | None
    observations: TranscriptCounts
    source: ReferenceArtifactManifest
    estimator_id: str
    preprocessing_id: str
    representation_id: str
    uncertainty_id: str
    field_id: str
    report: AdapterReport


def import_velocity_field(
    values: ArrayLike,
    observations: TranscriptCounts,
    /,
    *,
    source: ReferenceArtifactManifest,
    estimator_id: str,
    preprocessing_id: str,
    representation_id: str,
    uncertainty_id: str,
    valid: ArrayLike | None = None,
    standard_errors: ArrayLike | None = None,
    commercial_use: bool = False,
    training_use: bool = False,
    redistribution: bool = False,
    export: bool = False,
) -> ImportedVelocityField:
    """Retain an external estimator/embedding field without physical-time promotion.

    ``standard_errors=None`` explicitly retains unreported uncertainty. Arbitrary
    embedded arrows are neither a physical CTMC path nor an energy landscape.
    """
    source.require_rights(
        commercial_use=commercial_use,
        training_use=training_use,
        redistribution=redistribution,
        export=export,
    )
    labels = tuple(
        _label(x, "velocity lineage")
        for x in (estimator_id, preprocessing_id, representation_id, uncertainty_id)
    )
    raw = np.asarray(values)
    if (
        raw.ndim != 2
        or raw.shape[0] != len(observations.cell_ids)
        or raw.shape[1] == 0
        or raw.dtype.kind not in "ifu"
    ):
        raise ValueError(
            "Velocity fields must have shape (declared cell, representation dimension)."
        )
    mask = np.ones(raw.shape, dtype=bool) if valid is None else np.asarray(valid)
    if mask.shape != raw.shape or mask.dtype != bool or np.any(~np.isfinite(raw[mask])):
        raise ValueError("Velocity validity must preserve a finite active field.")
    errors = None if standard_errors is None else np.asarray(standard_errors, dtype=float)
    if errors is not None and (
        errors.shape != raw.shape
        or np.any(~np.isfinite(errors[mask]))
        or np.any(errors[mask] < 0)
    ):
        raise ValueError(
            "Velocity standard errors must be nonnegative, finite and aligned with the field."
        )
    identity = canonical_fingerprint(
        {
            "kind": "imported-inferred-velocity",
            "source": source.manifest_id,
            "observations": observations.observation_id,
            "lineage": labels,
            "values": array_tree_fingerprint(raw),
            "valid": array_tree_fingerprint(mask),
            "errors": None if errors is None else array_tree_fingerprint(errors),
        }
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "external-inferred-velocity-array",
        "native-inferred-velocity-field",
        source_id=source.manifest_id,
        target_id=identity,
        preserved_fields=(
            "values",
            "valid",
            "standard_errors",
            "estimator_id",
            "preprocessing_id",
            "representation_id",
            "uncertainty_id",
            "cell_ids",
            "coordinate_semantics",
            "source_rights",
        ),
        assumptions=(
            "No physical time, lineage, or energy claim is inferred from an embedded field.",
        ),
    )
    return ImportedVelocityField(
        jnp.asarray(raw),
        jnp.asarray(mask),
        None if errors is None else jnp.asarray(errors),
        observations,
        source,
        *labels,
        identity,
        report,
    )
