#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING

import h5py
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...qualification import ReferenceArtifactManifest
from .._report import AdapterFormatProfile, AdapterReport, AdapterStatus


if TYPE_CHECKING:
    from ...applications.detector.calorimetry._geometry import CalorimeterGeometry


@dataclass(frozen=True, slots=True)
class CaloChallengeProfile:
    incident_energy_dataset: str
    shower_dataset: str
    geometry_id: str
    energy_unit: str
    maximum_records: int
    profile_id: str = field(init=False)

    def __post_init__(self) -> None:
        incident = str(self.incident_energy_dataset).strip().strip("/")
        shower = str(self.shower_dataset).strip().strip("/")
        geometry = str(self.geometry_id).strip()
        unit = str(self.energy_unit).strip()
        maximum = int(self.maximum_records)
        if not incident or not shower or not geometry or not unit or maximum < 1:
            raise ValueError("CaloChallenge profile fields must be explicit and bounded.")
        object.__setattr__(self, "incident_energy_dataset", incident)
        object.__setattr__(self, "shower_dataset", shower)
        object.__setattr__(self, "geometry_id", geometry)
        object.__setattr__(self, "energy_unit", unit)
        object.__setattr__(self, "maximum_records", maximum)
        object.__setattr__(
            self,
            "profile_id",
            canonical_fingerprint(
                {
                    "kind": "calochallenge-hdf5-profile",
                    "incident_energy_dataset": incident,
                    "shower_dataset": shower,
                    "geometry": geometry,
                    "energy_unit": unit,
                    "maximum_records": maximum,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CaloChallengeImport:
    incident_energies: object
    showers: object
    reference: ReferenceArtifactManifest
    report: AdapterReport
    profile_id: str


def import_calochallenge_hdf5(
    data: bytes,
    reference: ReferenceArtifactManifest,
    profile: CaloChallengeProfile,
    geometry: CalorimeterGeometry,
    /,
    *,
    training_use: bool = False,
    commercial_use: bool = False,
) -> CaloChallengeImport:
    """Import the explicit incident-energy/showers CaloChallenge HDF5 profile."""
    from ...applications.detector.calorimetry._geometry import CalorimeterGeometry

    if not isinstance(data, bytes):
        raise TypeError("CaloChallenge data must be exact bytes.")
    if not isinstance(reference, ReferenceArtifactManifest):
        raise TypeError("reference must be ReferenceArtifactManifest.")
    if not isinstance(profile, CaloChallengeProfile) or not isinstance(
        geometry, CalorimeterGeometry
    ):
        raise TypeError("profile and geometry must use calorimeter types.")
    if profile.geometry_id != geometry.geometry_id:
        raise ValueError("CaloChallenge profile and calorimeter geometry differ.")
    reference.verify_bytes(data)
    reference.require_rights(
        training_use=training_use,
        commercial_use=commercial_use,
    )
    with h5py.File(BytesIO(data), "r") as handle:
        if (
            profile.incident_energy_dataset not in handle
            or profile.shower_dataset not in handle
        ):
            raise ValueError("CaloChallenge HDF5 is missing a profiled dataset.")
        incident_dataset = handle[profile.incident_energy_dataset]
        shower_dataset = handle[profile.shower_dataset]
        if (
            incident_dataset.is_virtual
            or incident_dataset.external
            or shower_dataset.is_virtual
            or shower_dataset.external
        ):
            raise ValueError(
                "CaloChallenge datasets must be resident in the bounded resource."
            )
        incident = np.asarray(incident_dataset[()], dtype=np.float64).reshape((-1,))
        showers = np.asarray(shower_dataset[()], dtype=np.float64)
    if (
        incident.size < 1
        or incident.size > profile.maximum_records
        or showers.shape != (incident.size, geometry.cell_count)
    ):
        raise ValueError("CaloChallenge arrays exceed record/cell support.")
    if (
        np.any(~np.isfinite(incident))
        or np.any(incident <= 0.0)
        or np.any(~np.isfinite(showers))
        or np.any(showers < 0.0)
    ):
        raise ValueError(
            "CaloChallenge incident energies and showers must be finite and physical."
        )
    if np.any(showers[:, ~np.asarray(geometry.active)] != 0.0):
        raise ValueError("CaloChallenge inactive cells must be exactly zero.")
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "CaloChallenge-HDF5",
        "phydrax-calorimeter-corpus-arrays",
        source_id=reference.manifest_id,
        target_id=profile.profile_id,
        preserved_fields=("incident_energies", "showers"),
        source_profile=AdapterFormatProfile(
            "CaloChallenge-HDF5",
            qualifiers={
                "energy_unit": profile.energy_unit,
                "geometry_id": profile.geometry_id,
            },
        ),
        target_profile=AdapterFormatProfile(
            "phydrax-calorimeter-corpus-arrays",
            qualifiers={"cell_support": geometry.geometry_id},
        ),
        stage="calochallenge-import",
    )
    return CaloChallengeImport(
        jnp.asarray(incident),
        jnp.asarray(showers),
        reference,
        report,
        profile.profile_id,
    )


__all__ = [
    "CaloChallengeImport",
    "CaloChallengeProfile",
    "import_calochallenge_hdf5",
]
