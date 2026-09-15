#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Complete source/provenance context for admitted periodic numeric bytes."""

from __future__ import annotations

import hashlib
from numbers import Integral

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import UnitDefinition
from ._orbital_model import PeriodicOrbitalBasisPlan


class PeriodicProvenanceManifest(StrictModule, NonTrainableState):
    source_id: str = eqx.field(static=True)
    sha256: str = eqx.field(static=True)
    byte_size: int = eqx.field(static=True)
    rights_id: str = eqx.field(static=True)
    parent_ids: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        sha256: str,
        byte_size: int,
        rights_id: str,
        /,
        *,
        parent_ids: tuple[str, ...] = (),
    ):
        source = str(source_id).strip()
        digest = str(sha256).strip().lower()
        rights = str(rights_id).strip()
        parents = tuple(str(value).strip() for value in parent_ids)
        if (
            not source
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(byte_size, bool)
            or not isinstance(byte_size, Integral)
            or int(byte_size) < 0
            or not rights
            or any(not value for value in parents)
            or len(set(parents)) != len(parents)
        ):
            raise ValueError(
                "Periodic provenance source, digest, size, rights, or parents are invalid."
            )
        self.source_id = source
        self.sha256 = digest
        self.byte_size = int(byte_size)
        self.rights_id = rights
        self.parent_ids = parents
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "periodic-provenance-manifest",
                "source": source,
                "sha256": digest,
                "byte_size": int(byte_size),
                "rights": rights,
                "parents": list(parents),
            }
        )

    @classmethod
    def for_bytes(
        cls,
        payload: bytes,
        source_id: str,
        rights_id: str,
        /,
        *,
        parent_ids: tuple[str, ...] = (),
    ) -> "PeriodicProvenanceManifest":
        if not isinstance(payload, bytes):
            raise TypeError("Periodic source payload must be bytes.")
        return cls(
            source_id,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            rights_id,
            parent_ids=parent_ids,
        )

    def admit_bytes(self, payload: bytes, /) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("Periodic source payload must be bytes.")
        if (
            len(payload) != self.byte_size
            or hashlib.sha256(payload).hexdigest() != self.sha256
        ):
            raise ValueError(
                "Periodic source bytes do not match the bound size and digest."
            )


class PeriodicSourceContext(StrictModule, NonTrainableState):
    """Cell, ordered basis, gauge, spin, statistics, units, and byte provenance."""

    basis: PeriodicOrbitalBasisPlan
    energy_unit: UnitDefinition
    provenance: PeriodicProvenanceManifest
    context_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: PeriodicOrbitalBasisPlan,
        energy_unit: UnitDefinition,
        provenance: PeriodicProvenanceManifest,
        /,
    ):
        if not isinstance(basis, PeriodicOrbitalBasisPlan):
            raise TypeError("Periodic source context requires PeriodicOrbitalBasisPlan.")
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("Periodic source context requires an energy UnitDefinition.")
        if not isinstance(provenance, PeriodicProvenanceManifest):
            raise TypeError(
                "Periodic source context requires PeriodicProvenanceManifest."
            )
        self.basis = basis
        self.energy_unit = energy_unit
        self.provenance = provenance
        self.context_id = canonical_fingerprint(
            {
                "kind": "periodic-source-context",
                "cell": basis.cell_id,
                "basis": basis.basis_id,
                "gauge": basis.gauge.gauge_id,
                "spin_order": basis.spin_order,
                "statistics": basis.statistics,
                "length_unit": basis.length_unit.unit_id,
                "energy_unit": energy_unit.unit_id,
                "provenance": provenance.manifest_id,
            }
        )

    def admit_bytes(self, payload: bytes, /) -> None:
        self.provenance.admit_bytes(payload)


__all__ = ["PeriodicProvenanceManifest", "PeriodicSourceContext"]
