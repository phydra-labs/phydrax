#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host electronic-site and parameter declarations; no parameter inference."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

from ...._fingerprint import canonical_fingerprint
from ....artifacts import ScientificArtifactEnvelope
from ....qualification import ReferenceArtifactManifest
from ....units import ENERGY, TIME, UnitDefinition
from .._construct import NucleicAcidConstruct, NucleotideKey


BasisKey = tuple[int, ...]
_RIGHTS_KEYS = frozenset(("commercial_use", "redistribution", "training_use", "export"))


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string.")


def _site_id(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError("Electronic site IDs must be nonnegative signed-int64 integers.")


def _basis_key(value: BasisKey) -> None:
    if not isinstance(value, tuple) or len(value) not in (1, 2):
        raise ValueError("A carrier basis key has one site ID or an electron/hole pair.")
    for site in value:
        _site_id(site)


def _admit(
    manifests: tuple[ReferenceArtifactManifest, ...], requested_use: Mapping[str, bool]
) -> None:
    if not isinstance(requested_use, Mapping) or set(requested_use) != _RIGHTS_KEYS:
        raise ValueError("Declare all four requested-use rights explicitly.")
    for manifest in manifests:
        manifest.require_rights(**requested_use)


@dataclass(frozen=True, slots=True)
class ElectronicSiteGraph:
    """Explicit orbital/site IDs bound to nucleotides, not atom IDs.

    Edges are undirected support; a parameter coupling records H[row, column]
    in its declared complex orbital gauge. Structure artifacts are optional for
    a sequence-only model and mandatory provenance for structure-derived inputs.
    """

    construct: NucleicAcidConstruct
    site_ids: tuple[int, ...]
    nucleotide_keys: tuple[NucleotideKey, ...]
    orbital_labels: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    structure_artifacts: tuple[ScientificArtifactEnvelope, ...] = ()
    structure_rights: tuple[ReferenceArtifactManifest, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.construct, NucleicAcidConstruct):
            raise TypeError("construct must be NucleicAcidConstruct.")
        for name in (
            "site_ids",
            "nucleotide_keys",
            "orbital_labels",
            "edges",
            "structure_artifacts",
            "structure_rights",
        ):
            if not isinstance(object.__getattribute__(self, name), tuple):
                raise TypeError(f"{name} must be an immutable tuple.")
        if not self.site_ids or len(set(self.site_ids)) != len(self.site_ids):
            raise ValueError("Electronic site IDs must be nonempty and unique.")
        if len(self.nucleotide_keys) != len(self.site_ids) or len(
            self.orbital_labels
        ) != len(self.site_ids):
            raise ValueError(
                "Every electronic site requires a nucleotide and orbital label."
            )
        valid_keys = set(self.construct.nucleotide_keys)
        for site, key, label in zip(
            self.site_ids, self.nucleotide_keys, self.orbital_labels, strict=True
        ):
            _site_id(site)
            if key not in valid_keys:
                raise ValueError("Electronic site nucleotide is outside the construct.")
            _text(label, "orbital label")
        if len(set(zip(self.nucleotide_keys, self.orbital_labels, strict=True))) != len(
            self.site_ids
        ):
            raise ValueError("Nucleotide/orbital identities must be unique.")
        known = set(self.site_ids)
        seen = set()
        for edge in self.edges:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise ValueError("Each edge must be a pair of site IDs.")
            left, right = edge
            if left == right or left not in known or right not in known:
                raise ValueError("Edges must join two distinct declared sites.")
            canonical = tuple(sorted(edge))
            if canonical in seen:
                raise ValueError("Duplicate undirected electronic edge.")
            seen.add(canonical)
        if len(self.structure_artifacts) != len(self.structure_rights):
            raise ValueError("Every structure artifact requires its own rights manifest.")
        for envelope, rights in zip(
            self.structure_artifacts, self.structure_rights, strict=True
        ):
            if (
                envelope.status != "complete"
                or envelope.content_digest != rights.checksum
            ):
                raise ValueError(
                    "Structure artifact and rights manifest must identify the same complete source."
                )
            if envelope.license_id != rights.license_id:
                raise ValueError("Structure envelope and rights license disagree.")

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "nucleotide-electronic-site-graph",
                "construct": self.construct.fingerprint(),
                "sites": sorted(
                    (site, key.strand_id, key.position, label)
                    for site, key, label in zip(
                        self.site_ids,
                        self.nucleotide_keys,
                        self.orbital_labels,
                        strict=True,
                    )
                ),
                "edges": sorted(tuple(sorted(edge)) for edge in self.edges),
                "structures": sorted(
                    item.artifact_id for item in self.structure_artifacts
                ),
                "rights": sorted(item.manifest_id for item in self.structure_rights),
            }
        )


@dataclass(frozen=True, slots=True)
class ElectronicChannel:
    """One declared Markovian channel with a separate inverse-time rate.

    dephasing: |source><source| (coherence ij decays at (rate_i+rate_j)/2).
    bath: |target><source|, a declared population-transfer bath, not an inferred
    thermal equilibrium law. recombination: |vacuum><source|. Recombination is
    loss of this electronic sector only, never a lesion or atom-charge model.
    """

    channel_id: str
    kind: str
    source: BasisKey
    target: BasisKey | None
    rate: float
    rate_unit: UnitDefinition

    def __post_init__(self) -> None:
        _text(self.channel_id, "channel_id")
        _basis_key(self.source)
        if self.kind not in ("dephasing", "bath", "recombination"):
            raise ValueError("Unknown electronic channel kind.")
        if self.kind == "bath":
            _basis_key(self.target)
            if self.target == self.source or len(self.target) != len(self.source):
                raise ValueError("Bath transfer requires a distinct same-sector target.")
        elif self.target is not None:
            raise ValueError("Dephasing/recombination channels do not take a target.")
        if not math.isfinite(self.rate) or self.rate < 0.0:
            raise ValueError("Electronic channel rates must be finite and nonnegative.")
        if not isinstance(
            self.rate_unit, UnitDefinition
        ) or self.rate_unit.dimension != TIME.power(-1):
            raise ValueError(
                "Channel rate units must be inverse time, without an angular factor."
            )

    def record(self) -> dict[str, object]:
        return {
            "id": self.channel_id,
            "kind": self.kind,
            "source": self.source,
            "target": self.target,
            "rate": self.rate,
            "unit": self.rate_unit.unit_id,
        }


@dataclass(frozen=True, slots=True)
class ElectronicParameterArtifact:
    """Source-pinned single-system energies and environmental parameters.

    Raw bytes are checksum-verified and retained separately from this normalized
    declaration. No bundled DNA/RNA calibration is supplied. A two-index basis
    denotes an explicitly declared electron/hole interaction artifact, including
    zero entries when a noninteracting model is intended.
    """

    basis_keys: tuple[BasisKey, ...]
    site_energies: tuple[float, ...]
    couplings: tuple[tuple[BasisKey, BasisKey, complex], ...]
    channels: tuple[ElectronicChannel, ...]
    energy_unit: UnitDefinition
    scope: str
    orbital_gauge: str
    source: ReferenceArtifactManifest
    raw_content: bytes
    structure_derived: bool = False

    def __post_init__(self) -> None:
        for values in (
            self.basis_keys,
            self.site_energies,
            self.couplings,
            self.channels,
        ):
            if not isinstance(values, tuple):
                raise TypeError("Electronic parameter arrays must be immutable tuples.")
        if not self.basis_keys or len(set(self.basis_keys)) != len(self.basis_keys):
            raise ValueError("Parameter basis keys must be nonempty and unique.")
        for key in self.basis_keys:
            _basis_key(key)
        if len({len(key) for key in self.basis_keys}) != 1:
            raise ValueError("A parameter artifact cannot mix carrier sectors.")
        if len(self.site_energies) != len(self.basis_keys) or any(
            not math.isfinite(value) for value in self.site_energies
        ):
            raise ValueError("Every parameter site requires a finite real energy.")
        if (
            not isinstance(self.energy_unit, UnitDefinition)
            or self.energy_unit.dimension != ENERGY
        ):
            raise ValueError(
                "Electronic parameters require single-system energy units, not molar energy or frequency."
            )
        _text(self.scope, "scope")
        _text(self.orbital_gauge, "orbital_gauge")
        if not isinstance(self.structure_derived, bool):
            raise TypeError("structure_derived must be a boolean.")
        if not isinstance(self.source, ReferenceArtifactManifest) or not isinstance(
            self.raw_content, bytes
        ):
            raise TypeError("Source manifest and raw parameter bytes are required.")
        if (
            len(self.raw_content) != self.source.size_bytes
            or hashlib.new(self.source.checksum_algorithm, self.raw_content).hexdigest()
            != self.source.checksum
        ):
            raise ValueError("Raw parameter bytes do not match the source manifest.")
        seen = set()
        known = set(self.basis_keys)
        for row, column, value in self.couplings:
            if row == column or row not in known or column not in known:
                raise ValueError("Couplings must connect distinct declared basis keys.")
            if not math.isfinite(complex(value).real) or not math.isfinite(
                complex(value).imag
            ):
                raise ValueError("Electronic couplings must be finite.")
            edge = tuple(sorted((row, column)))
            if edge in seen:
                raise ValueError(
                    "Each Hermitian edge must be supplied once, not in both directions."
                )
            seen.add(edge)
        if len({channel.channel_id for channel in self.channels}) != len(self.channels):
            raise ValueError("Electronic channel IDs must be unique.")
        for channel in self.channels:
            if channel.source not in known or (
                channel.target is not None and channel.target not in known
            ):
                raise ValueError("Channel source/target is outside parameter support.")

    def fingerprint(self) -> str:
        edges = []
        for row, column, value in self.couplings:
            value = complex(value)
            if row > column:
                row, column, value = column, row, value.conjugate()
            edges.append((row, column, value.real, value.imag))
        return canonical_fingerprint(
            {
                "kind": "declared-electronic-parameters",
                "sites": sorted(zip(self.basis_keys, self.site_energies, strict=True)),
                "couplings": sorted(edges),
                "channels": sorted(
                    (channel.record() for channel in self.channels),
                    key=lambda record: record["id"],
                ),
                "energy_unit": self.energy_unit.unit_id,
                "scope": self.scope,
                "gauge": self.orbital_gauge,
                "source": self.source.manifest_id,
                "structure_derived": self.structure_derived,
                "phase_convention": "exp(-i H t / hbar); angular generator; no cyclic-frequency conversion",
            }
        )
