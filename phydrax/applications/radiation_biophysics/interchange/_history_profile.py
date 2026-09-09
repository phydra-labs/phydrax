#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Timed, zero-preserving external radiation history profile."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import conversion_factor, SECOND, UnitDefinition

from .._interactions import _text, PrimaryHistoryKey, RadiationSource


@dataclass(frozen=True, slots=True)
class RadiationHistoryCoverage:
    missing_references: tuple[str, ...]
    missing_species: tuple[str, ...]
    missing_time_species_rows: tuple[tuple[str, float], ...]
    zero_physical_histories: tuple[PrimaryHistoryKey, ...]
    zero_chemical_histories: tuple[PrimaryHistoryKey, ...]
    complete: bool


@dataclass(frozen=True, slots=True, init=False)
class TimedRadiationHistoryProfile:
    """External histories with explicit zero counts and timed species coverage.

    ``species_counts`` includes one cell for every declared history/species/time
    coordinate. ``species_valid`` distinguishes a measured zero from an unreported
    coordinate. This is an adapter record, not a transport or radiolysis engine.
    """

    source: RadiationSource
    histories: tuple[PrimaryHistoryKey, ...]
    physical_tuple_ids: tuple[str, ...]
    physical_event_counts: Array
    dose_gy: Array
    dose_standard_errors_gy: Array
    species_ids: tuple[str, ...]
    sample_times: Array
    species_counts: Array
    species_valid: Array
    time_unit: UnitDefinition
    dosimetry_reference: ReferenceArtifactManifest | None
    transport_reference: ReferenceArtifactManifest | None
    chemical_reference: ReferenceArtifactManifest | None
    profile_id: str

    def __init__(
        self,
        source: RadiationSource,
        histories: tuple[PrimaryHistoryKey, ...],
        /,
        *,
        physical_tuple_ids: tuple[str, ...],
        physical_event_counts: ArrayLike,
        dose_gy: ArrayLike,
        dose_standard_errors_gy: ArrayLike,
        species_ids: tuple[str, ...],
        sample_times: ArrayLike,
        species_counts: ArrayLike,
        species_valid: ArrayLike,
        time_unit: UnitDefinition,
        dosimetry_reference: ReferenceArtifactManifest | None,
        transport_reference: ReferenceArtifactManifest | None,
        chemical_reference: ReferenceArtifactManifest | None,
    ):
        if not isinstance(source, RadiationSource):
            raise TypeError("source must be RadiationSource.")
        if (
            not isinstance(histories, tuple)
            or not histories
            or not all(isinstance(item, PrimaryHistoryKey) for item in histories)
            or len(set(histories)) != len(histories)
        ):
            raise ValueError("Radiation histories must be unique and nonempty.")
        if any(item.source_id != source.artifact.artifact_id for item in histories):
            raise ValueError("Every history must belong to the external source artifact.")
        tuples = tuple(physical_tuple_ids)
        if len(tuples) != len(histories) or len(set(tuples)) != len(tuples):
            raise ValueError(
                "Each history requires one unique exact physical-tuple identity."
            )
        for value in tuples:
            _text(value, "physical tuple ID")
        event_counts = np.asarray(physical_event_counts)
        doses = np.asarray(dose_gy, dtype=float)
        dose_errors = np.asarray(dose_standard_errors_gy, dtype=float)
        if (
            event_counts.shape != (len(histories),)
            or event_counts.dtype.kind not in "iu"
            or np.any(event_counts < 0)
        ):
            raise ValueError(
                "Physical event counts must be nonnegative integers per history."
            )
        if (
            doses.shape != event_counts.shape
            or dose_errors.shape != event_counts.shape
            or not np.all(np.isfinite(doses))
            or not np.all(np.isfinite(dose_errors))
            or np.any(doses < 0.0)
            or np.any(dose_errors <= 0.0)
        ):
            raise ValueError(
                "Dose and positive dosimetric uncertainty are required per history."
            )
        species = tuple(species_ids)
        times = np.asarray(sample_times, dtype=float)
        if (
            not species
            or len(set(species)) != len(species)
            or any(not item or item != item.strip() for item in species)
        ):
            raise ValueError("Species IDs must be unique nonempty canonical strings.")
        if (
            times.ndim != 1
            or times.size == 0
            or not np.all(np.isfinite(times))
            or np.any(times < 0.0)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                "Chemical sample times must be finite, nonnegative, and increasing."
            )
        conversion_factor(time_unit, SECOND)
        counts = np.asarray(species_counts)
        valid = np.asarray(species_valid)
        expected = (len(histories), len(species), len(times))
        if counts.shape != expected or counts.dtype.kind not in "ifu":
            raise ValueError("Species counts must have shape (history, species, time).")
        if valid.shape != expected or valid.dtype != bool:
            raise ValueError("Species validity must be a boolean mask matching counts.")
        active = counts[valid]
        if (
            np.any(~np.isfinite(active))
            or np.any(active < 0.0)
            or np.any(active != np.floor(active))
        ):
            raise ValueError("Reported species counts must be nonnegative integers.")
        for reference in (dosimetry_reference, transport_reference, chemical_reference):
            if reference is not None and not isinstance(
                reference, ReferenceArtifactManifest
            ):
                raise TypeError(
                    "Stage references must be ReferenceArtifactManifest or None."
                )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "histories", histories)
        object.__setattr__(self, "physical_tuple_ids", tuples)
        object.__setattr__(self, "physical_event_counts", jnp.asarray(event_counts))
        object.__setattr__(self, "dose_gy", jnp.asarray(doses))
        object.__setattr__(self, "dose_standard_errors_gy", jnp.asarray(dose_errors))
        object.__setattr__(self, "species_ids", species)
        object.__setattr__(self, "sample_times", jnp.asarray(times))
        object.__setattr__(
            self, "species_counts", jnp.asarray(np.where(valid, counts, 0))
        )
        object.__setattr__(self, "species_valid", jnp.asarray(valid))
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(self, "dosimetry_reference", dosimetry_reference)
        object.__setattr__(self, "transport_reference", transport_reference)
        object.__setattr__(self, "chemical_reference", chemical_reference)
        object.__setattr__(
            self,
            "profile_id",
            canonical_fingerprint(
                {
                    "kind": "timed-radiation-history-profile",
                    "source": source.fingerprint(),
                    "histories": [
                        (
                            item.source_id,
                            item.run_id,
                            item.primary_id,
                            item.fraction_id,
                        )
                        for item in histories
                    ],
                    "physical_tuples": tuples,
                    "physical_event_counts": event_counts.tolist(),
                    "dose_gy": doses.tolist(),
                    "dose_standard_errors_gy": dose_errors.tolist(),
                    "species": species,
                    "times": times.tolist(),
                    "time_unit": time_unit.unit_id,
                    "species_counts": array_tree_fingerprint(np.where(valid, counts, 0)),
                    "species_valid": array_tree_fingerprint(valid),
                    "references": tuple(
                        None if item is None else item.manifest_id
                        for item in (
                            dosimetry_reference,
                            transport_reference,
                            chemical_reference,
                        )
                    ),
                }
            ),
        )

    def require_rights(
        self,
        /,
        *,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ) -> None:
        self.source.require_rights(
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
        for reference in (
            self.dosimetry_reference,
            self.transport_reference,
            self.chemical_reference,
        ):
            if reference is not None:
                reference.require_rights(
                    commercial_use=commercial_use,
                    redistribution=redistribution,
                    training_use=training_use,
                    export=export,
                )

    def coverage(
        self,
        required_species_ids: tuple[str, ...],
        required_sample_times: tuple[float, ...],
        /,
    ) -> RadiationHistoryCoverage:
        """Report exact stage gaps while retaining valid measured zeros."""

        required_species = tuple(required_species_ids)
        required_times = tuple(float(value) for value in required_sample_times)
        if (
            not required_species
            or len(set(required_species)) != len(required_species)
            or any(not value or value != value.strip() for value in required_species)
            or not required_times
            or len(set(required_times)) != len(required_times)
            or not np.all(np.isfinite(required_times))
            or any(value < 0.0 for value in required_times)
        ):
            raise ValueError(
                "Required species and sample-time grids must be nonempty, unique, and canonical."
            )
        missing_references = tuple(
            name
            for name, reference in (
                ("dosimetry", self.dosimetry_reference),
                ("transport", self.transport_reference),
                ("chemical-G", self.chemical_reference),
            )
            if reference is None
        )
        missing_species = tuple(
            value for value in required_species if value not in self.species_ids
        )
        missing_rows: list[tuple[str, float]] = []
        valid = np.asarray(self.species_valid)
        sample_times = np.asarray(self.sample_times)
        for species in required_species:
            if species not in self.species_ids:
                continue
            species_index = self.species_ids.index(species)
            for time in required_times:
                matches = np.flatnonzero(sample_times == time)
                if matches.size != 1 or not np.all(valid[:, species_index, matches[0]]):
                    missing_rows.append((species, time))
        events = np.asarray(self.physical_event_counts)
        counts = np.asarray(self.species_counts)
        chemical_reported = np.any(valid, axis=(1, 2))
        chemical_any = np.any(valid & (counts > 0), axis=(1, 2))
        zero_physical = tuple(
            history
            for history, count in zip(self.histories, events, strict=True)
            if count == 0
        )
        zero_chemical = tuple(
            history
            for history, reported, any_count in zip(
                self.histories, chemical_reported, chemical_any, strict=True
            )
            if reported and not any_count
        )
        complete = not missing_references and not missing_species and not missing_rows
        return RadiationHistoryCoverage(
            missing_references,
            missing_species,
            tuple(missing_rows),
            zero_physical,
            zero_chemical,
            complete,
        )


__all__ = ["RadiationHistoryCoverage", "TimedRadiationHistoryProfile"]
