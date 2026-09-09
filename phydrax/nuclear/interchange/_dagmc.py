#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Opaque, governed DAGMC geometry artifacts without CAD-kernel ownership."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..._fingerprint import canonical_fingerprint
from ...interchange import AdapterReport, AdapterStatus, BoundedResource
from ...qualification import ReferenceArtifactManifest
from ...units import LENGTH, UnitDefinition


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class DagmcGeometryArtifact:
    resource: BoundedResource
    reference: ReferenceArtifactManifest
    length_unit: UnitDefinition
    source_geometry_id: str
    converter: str
    converter_release: str
    material_regions: tuple[tuple[str, str], ...]
    overlap_checked: bool
    geometry_id: str = field(init=False)
    report: AdapterReport = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.resource, BoundedResource):
            raise TypeError("resource must be BoundedResource.")
        if not isinstance(self.reference, ReferenceArtifactManifest):
            raise TypeError("reference must be ReferenceArtifactManifest.")
        if (
            not isinstance(self.length_unit, UnitDefinition)
            or self.length_unit.dimension != LENGTH
        ):
            raise ValueError("length_unit must have physical length dimension.")
        self.reference.verify_bytes(self.resource.data)
        self.reference.require_rights()
        if not self.resource.data.startswith(_HDF5_SIGNATURE):
            raise ValueError("DAGMC geometry must be an HDF5 resource.")
        values = tuple(
            str(value).strip()
            for value in (
                self.source_geometry_id,
                self.converter,
                self.converter_release,
            )
        )
        if any(not value for value in values):
            raise ValueError("DAGMC source and converter identities must be non-empty.")
        regions = tuple(
            (str(region).strip(), str(material).strip())
            for region, material in self.material_regions
        )
        if any(not region or not material for region, material in regions):
            raise ValueError("DAGMC material-region mappings must be non-empty.")
        if len({region for region, _ in regions}) != len(regions):
            raise ValueError("DAGMC material-region identifiers must be unique.")
        if not isinstance(self.overlap_checked, bool):
            raise TypeError("overlap_checked must be boolean.")
        object.__setattr__(self, "source_geometry_id", values[0])
        object.__setattr__(self, "converter", values[1])
        object.__setattr__(self, "converter_release", values[2])
        object.__setattr__(self, "material_regions", regions)
        geometry_id = canonical_fingerprint(
            {
                "kind": "dagmc-geometry-artifact",
                "resource": self.resource.manifest.manifest_id,
                "reference": self.reference.manifest_id,
                "length_unit": self.length_unit.unit_id,
                "source_geometry": values[0],
                "converter": values[1],
                "converter_release": values[2],
                "material_regions": [list(value) for value in regions],
                "overlap_checked": self.overlap_checked,
            }
        )
        object.__setattr__(self, "geometry_id", geometry_id)
        object.__setattr__(
            self,
            "report",
            AdapterReport(
                AdapterStatus.LOSSLESS,
                "DAGMC-HDF5",
                "DagmcGeometryArtifact",
                source_id=self.resource.manifest.manifest_id,
                target_id=geometry_id,
                preserved_fields=(
                    "exact DAGMC HDF5 bytes",
                    "material-region mapping",
                    "source geometry lineage",
                    "converter identity",
                    "length unit",
                ),
                assumptions=(
                    "DAGMC geometry remains opaque to native JAX execution",
                    "overlap_checked records an external claim, not a Phydrax geometry proof",
                ),
            ),
        )


__all__ = ["DagmcGeometryArtifact"]
