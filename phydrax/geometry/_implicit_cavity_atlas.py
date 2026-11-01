from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
)
from ..discretization._cell_mesh import CellMesh
from ..typing import checked
from ._source_identity import source_entity_id
from ._volume_material_map import MaterialSourceMapCapability
from .implicit._analytic_profile import AnalyticImplicitProfile


class ImplicitCavityAtlas(StrictModule, NonTrainableState):
    """Stable source/material classes for one or more certified implicit regions."""

    source_profiles: tuple[AnalyticImplicitProfile, ...]
    source_fidelity_certificates: tuple[Any, ...]
    source_map: MaterialSourceMapCapability
    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    source_validity: CellValidityCertificate
    source_embedding: Any
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    region_ids: tuple[str, ...] = eqx.field(static=True)
    region_entity_ids: tuple[str, ...] = eqx.field(static=True)
    region_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    cell_region_ids: tuple[str, ...] = eqx.field(static=True)
    cell_entity_ids: tuple[str, ...] = eqx.field(static=True)
    cell_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    atlas_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        region_profiles: Mapping[str, AnalyticImplicitProfile],
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        cell_region_ids: Sequence[str],
        region_occurrence_paths: Mapping[str, tuple[str, ...]],
        source_fidelity_certificates: Mapping[str, Any],
        source_validity: CellValidityCertificate,
        source_embedding: Any,
        /,
    ) -> None:
        if any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in region_profiles
        ):
            raise ValueError(
                "Implicit source region keys must be stable canonical strings."
            )
        region_names = tuple(sorted(region_profiles))
        if not region_names or len(set(region_names)) != len(region_names):
            raise ValueError(
                "An implicit cavity atlas requires unique named source regions."
            )
        profiles = tuple(region_profiles[name] for name in region_names)
        if not all(isinstance(value, AnalyticImplicitProfile) for value in profiles):
            raise TypeError(
                "Every implicit source region requires AnalyticImplicitProfile authority."
            )
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "Implicit cavity atlas requires CellMesh and CellGeometrySpec owners."
            )
        if set(region_occurrence_paths) != set(region_names) or set(
            source_fidelity_certificates
        ) != set(region_names):
            raise ValueError(
                "Region occurrence and fidelity evidence must cover every source region."
            )
        identities = {(value.source_id, value.source_revision) for value in profiles}
        if len(identities) != 1:
            raise ValueError(
                "One implicit material map must retain one source identity and revision."
            )
        source_id, source_revision = next(iter(identities))
        paths = tuple(tuple(region_occurrence_paths[name]) for name in region_names)
        if any(
            any(
                not isinstance(part, str) or not part or part != part.strip()
                for part in path
            )
            for path in paths
        ):
            raise ValueError(
                "Implicit region occurrence paths must be immutable canonical strings."
            )
        entity_ids = tuple(
            source_entity_id(
                source_id,
                source_revision,
                "implicit-region",
                region,
                occurrence_path=path,
            )
            for region, path in zip(region_names, paths, strict=True)
        )
        cell_regions = tuple(str(value).strip() for value in cell_region_ids)
        cell_count = sum(block.cell_count for block in mesh.blocks)
        if len(cell_regions) != cell_count or any(
            value not in region_names for value in cell_regions
        ):
            raise ValueError(
                "Per-cell implicit region rows must align with the actual mesh cells."
            )
        region_entity = dict(zip(region_names, entity_ids, strict=True))
        region_path = dict(zip(region_names, paths, strict=True))
        cell_entities = tuple(region_entity[value] for value in cell_regions)
        cell_paths = tuple(region_path[value] for value in cell_regions)
        domain_id = canonical_fingerprint(
            {
                "kind": "implicit-cavity-material-domain",
                "source_id": source_id,
                "source_revision": source_revision,
                "regions": tuple(zip(region_names, entity_ids, paths, strict=True)),
            }
        )
        source_map = MaterialSourceMapCapability(
            source_id,
            source_revision,
            domain_id,
            entity_ids,
            region_names,
            occurrence_paths=paths,
        )
        certificates = tuple(source_fidelity_certificates[name] for name in region_names)
        for profile, certificate in zip(profiles, certificates, strict=True):
            if not callable(getattr(certificate, "require_current", None)):
                raise TypeError(
                    "Every implicit region requires a current source-fidelity certificate."
                )
            certificate.require_current(mesh, geometry, profile)
        source_validity.require_bound(geometry, mesh=mesh)
        if not source_validity.all_certified:
            raise ValueError(
                "Implicit cavity atlas requires certified whole-cell validity."
            )
        binding = getattr(source_embedding, "binding", None)
        if binding is None:
            raise ValueError(
                "Implicit cavity atlas requires a source-bound global embedding certificate."
            )
        binding.require(mesh, geometry)
        if getattr(source_embedding, "status", None) != "certified":
            raise ValueError("Implicit cavity atlas requires certified global embedding.")
        mesh_id = mesh.mesh_id
        topology_id = mesh.topology_id
        geometry_id = cell_geometry_id(geometry)
        atlas_id = canonical_fingerprint(
            {
                "kind": "implicit-cavity-atlas",
                "source_map": source_map.map_id,
                "mesh": mesh_id,
                "topology": topology_id,
                "geometry": geometry_id,
                "regions": tuple(
                    (name, profile.profile_id, certificate.certificate_id)
                    for name, profile, certificate in zip(
                        region_names, profiles, certificates, strict=True
                    )
                ),
                "validity": source_validity.certificate_id,
                "embedding": source_embedding.certificate_id,
                "cell_regions": cell_regions,
            }
        )
        self.source_profiles = profiles
        self.source_fidelity_certificates = certificates
        self.source_map = source_map
        self.source_mesh = mesh
        self.source_geometry = geometry
        self.source_validity = source_validity
        self.source_embedding = source_embedding
        self.source_id = source_id
        self.source_revision = source_revision
        self.domain_id = domain_id
        self.region_ids = region_names
        self.region_entity_ids = entity_ids
        self.region_occurrence_paths = paths
        self.cell_region_ids = cell_regions
        self.cell_entity_ids = cell_entities
        self.cell_occurrence_paths = cell_paths
        self.source_mesh_id = mesh_id
        self.source_topology_id = topology_id
        self.source_geometry_id = geometry_id
        self.atlas_id = atlas_id

    def require_bound(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        if (
            mesh.mesh_id != self.source_mesh_id
            or mesh.topology_id != self.source_topology_id
            or cell_geometry_id(geometry) != self.source_geometry_id
        ):
            raise ValueError("Implicit cavity atlas is stale for these endpoint maps.")
        for profile, certificate in zip(
            self.source_profiles, self.source_fidelity_certificates, strict=True
        ):
            profile.validate_source_integrity()
            certificate.binding.require(mesh, geometry, source=profile)
        self.source_validity.require_bound(geometry, mesh=mesh)
        self.source_embedding.binding.require(mesh, geometry)

    @classmethod
    def from_stored_fields(cls, value: ImplicitCavityAtlas, /) -> ImplicitCavityAtlas:
        if not isinstance(value, ImplicitCavityAtlas):
            raise TypeError("value must be ImplicitCavityAtlas.")
        profiles = dict(zip(value.region_ids, value.source_profiles, strict=True))
        certificates = dict(
            zip(value.region_ids, value.source_fidelity_certificates, strict=True)
        )
        rebuilt = cls(
            profiles,
            value.source_mesh,
            value.source_geometry,
            value.cell_region_ids,
            dict(zip(value.region_ids, value.region_occurrence_paths, strict=True)),
            certificates,
            value.source_validity,
            value.source_embedding,
        )
        if rebuilt.atlas_id != value.atlas_id:
            raise ValueError(
                "Stored implicit cavity atlas does not reproduce its source maps."
            )
        return rebuilt


__all__ = ["ImplicitCavityAtlas"]
