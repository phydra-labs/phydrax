#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Offline-governed import from an optional Basis Set Exchange installation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from collections.abc import Sequence

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticSystemPlan
from ...operators.quantum.gaussian import (
    GaussianBasisPlan,
    GaussianShellRepresentation,
)
from ...qualification import ReferenceArtifactManifest


class GaussianBasisImport(StrictModule, NonTrainableState):
    basis: GaussianBasisPlan
    artifact: ReferenceArtifactManifest
    package_version: str = eqx.field(static=True)
    import_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: GaussianBasisPlan,
        artifact: ReferenceArtifactManifest,
        package_version: str,
        /,
    ):
        if not isinstance(basis, GaussianBasisPlan):
            raise TypeError("basis must be GaussianBasisPlan.")
        if not isinstance(artifact, ReferenceArtifactManifest):
            raise TypeError("artifact must be ReferenceArtifactManifest.")
        version = str(package_version).strip()
        if not version or basis.source_artifact_id != artifact.manifest_id:
            raise ValueError("Basis import artifact and package identity must match.")
        self.basis = basis
        self.artifact = artifact
        self.package_version = version
        self.import_id = canonical_fingerprint(
            {
                "kind": "gaussian-basis-import",
                "basis": basis.basis_id,
                "artifact": artifact.manifest_id,
                "package_version": version,
            }
        )


def is_basis_set_exchange_available() -> bool:
    return importlib.util.find_spec("basis_set_exchange") is not None


def import_basis_set_exchange(
    name: str,
    system: AtomisticSystemPlan,
    particle_ids: Sequence[int],
    /,
    *,
    representation: GaussianShellRepresentation = GaussianShellRepresentation.REAL_SPHERICAL,
    role: str = "orbital",
    maximum_basis_functions: int = 4096,
    license_id: str,
    commercial_use_permitted: bool,
    redistribution_permitted: bool,
    training_use_permitted: bool,
    export_permitted: bool,
    export_classification: str,
    lineage_ids: Sequence[str],
) -> GaussianBasisImport:
    """Import one exact record; all rights assertions are explicit caller evidence."""

    if not is_basis_set_exchange_available():
        raise ImportError(
            "Basis data import requires the optional 'basis_set_exchange' package."
        )
    if not isinstance(system, AtomisticSystemPlan):
        raise TypeError("system must be AtomisticSystemPlan.")
    identifiers = tuple(particle_ids)
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in identifiers
    ):
        raise TypeError("Basis import particle IDs must be exact integers.")
    active = np.asarray(system.active_mask, dtype=np.bool_)
    id_to_index = {
        int(value): index
        for index, value in enumerate(np.asarray(system.particle_ids))
        if active[index]
    }
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or any(value not in id_to_index for value in identifiers)
    ):
        raise ValueError("Basis import particle IDs must be unique active system IDs.")
    numbers = tuple(
        int(np.asarray(system.atomic_numbers)[id_to_index[value]])
        for value in identifiers
    )
    if any(value <= 0 for value in numbers):
        raise ValueError("Basis imports require positive atomic numbers.")
    module = importlib.import_module("basis_set_exchange")
    version = importlib.metadata.version("basis_set_exchange")
    payload = module.get_basis(
        str(name),
        elements=tuple(sorted(set(numbers))),
        fmt="json",
        header=False,
    )
    data = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    record = json.loads(data)
    artifact = ReferenceArtifactManifest(
        f"basis-set-exchange:{str(name).strip()}@{version}",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        license_id=license_id,
        commercial_use_permitted=commercial_use_permitted,
        redistribution_permitted=redistribution_permitted,
        training_use_permitted=training_use_permitted,
        export_permitted=export_permitted,
        export_classification=export_classification,
        nondimensionalization={"bohr": 1.0},
        uncertainty=None,
        lineage_ids=tuple(lineage_ids),
    )
    basis = GaussianBasisPlan.from_basis_exchange_record(
        identifiers,
        numbers,
        record,
        representation=representation,
        maximum_basis_functions=maximum_basis_functions,
        source_id=f"basis-set-exchange:{str(name).strip()}@{version}",
        source_artifact_id=artifact.manifest_id,
        role=role,
    )
    return GaussianBasisImport(basis, artifact, version)


__all__ = [
    "GaussianBasisImport",
    "import_basis_set_exchange",
    "is_basis_set_exchange_available",
]
