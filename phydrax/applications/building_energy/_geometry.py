# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Explicit semantic enrichment of existing surface and meshing contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ...ein import contract
from ...geometry.surface import SurfaceModel
from ...meshing import MeshingEntityKind, MeshingScope, MeshLabel
from ...units import conversion_factor, METER
from ._model import (
    _text,
    Aperture,
    BuildingBoundary,
    BuildingSource,
    Construction,
    Surface,
    Zone,
)


class SurfaceRole(StrictModule):
    """One gross surface label, explicit thermal adjacency and reduction.

    Aperture labels are subsets of the gross surface's triangles. They replace,
    rather than add to, opaque area. No role is inferred from CAD label names.
    """

    label: MeshLabel
    zone_id: str = eqx.field(static=True)
    adjacent_zone: str | None = eqx.field(static=True)
    boundary_id: str | None = eqx.field(static=True)
    adiabatic: bool = eqx.field(static=True)
    construction: Construction
    aperture_labels: tuple[MeshLabel, ...]
    aperture_templates: tuple[Aperture, ...]

    def __init__(
        self,
        label: MeshLabel,
        zone_id: str,
        construction: Construction,
        *,
        adjacent_zone: str | None = None,
        apertures: Sequence[tuple[MeshLabel, Aperture]] = (),
        boundary_id: str | None = None,
        adiabatic: bool = False,
    ):
        self.label, self.zone_id, self.construction = (
            label,
            _text(zone_id, "zone_id"),
            construction,
        )
        self.adjacent_zone = adjacent_zone
        self.boundary_id, self.adiabatic = boundary_id, bool(adiabatic)
        self.aperture_labels = tuple(item[0] for item in apertures)
        self.aperture_templates = tuple(item[1] for item in apertures)


def surface_tag_labels(model: SurfaceModel) -> tuple[MeshLabel, ...]:
    """Bind existing cell tags to authoritative meshing labels, without assigning roles."""
    mesh = model.mesh
    tags = model.metadata.cell_tags
    if not tags:
        raise ValueError(
            "Surface has no cell tags; provide explicit revision-bound labels."
        )
    entities = mesh.entity_set(2)
    ids = np.asarray(entities.entity_ids)
    return tuple(
        MeshLabel(
            tag,
            MeshingScope(
                mesh.mesh_id,
                mesh.numeric_version,
                MeshingEntityKind.MESH,
                2,
                entities.entity_set_id,
                ids[np.asarray(tags) == tag],
            ),
        )
        for tag in dict.fromkeys(tags)
    )


def _label_positions(model: SurfaceModel, label: MeshLabel) -> np.ndarray:
    mesh, scope = model.mesh, label.scope
    entity_set = mesh.entity_set(2)
    if (
        scope.source_id != mesh.mesh_id
        or scope.source_revision != mesh.numeric_version
        or scope.entity_kind is not MeshingEntityKind.MESH
        or scope.entity_dimension != 2
        or scope.entity_set_id != entity_set.entity_set_id
    ):
        raise ValueError("Building geometry label has a stale or foreign mesh revision.")
    authoritative = np.asarray(entity_set.entity_ids)
    requested = np.asarray(scope.entity_ids)
    if not np.all(np.isin(requested, authoritative)):
        raise ValueError("Geometry label contains unknown cell IDs.")
    return np.flatnonzero(np.isin(authoritative, requested))


def enrich_building_geometry(
    zones: Sequence[Zone],
    model: SurfaceModel,
    roles: Sequence[SurfaceRole],
    *,
    source_id: str,
    coordinate_contract: SpatialCoordinateContract | None = None,
    boundaries: Sequence[BuildingBoundary] | None = None,
) -> BuildingSource:
    """Derive SI areas from an existing triangular surface; no volume mesh is needed."""
    spatial = (
        model.metadata.coordinate_contract
        if coordinate_contract is None
        else coordinate_contract
    )
    if spatial.spatial_id != model.metadata.coordinate_contract.spatial_id:
        raise ValueError(
            "Geometry coordinate contract does not match SurfaceModel metadata."
        )
    if (
        spatial.length_coordinate_kind != "physical"
        or spatial.coordinate_system != "cartesian"
    ):
        raise ValueError("Building areas require physical Cartesian coordinates.")
    mesh = model.mesh
    faces = jnp.concatenate(tuple(block.vertices for block in mesh.blocks), axis=0)
    points = mesh.coordinates * float(conversion_factor(spatial.length_unit, METER))
    triangles = points[faces]
    crosses = jnp.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = 0.5 * jnp.sqrt(contract("ij,ij->i", crosses, crosses))
    areas = eqx.error_if(
        areas,
        jnp.any(~jnp.isfinite(areas) | (areas <= 0)),
        "Building geometry contains degenerate triangles.",
    )
    seen: set[int] = set()
    surfaces = []
    for role in roles:
        positions = _label_positions(model, role.label)
        cells = set(positions.tolist())
        if cells & seen:
            raise ValueError(
                "Gross building surface labels overlap; adjacency must be represented once."
            )
        seen.update(cells)
        apertures, aperture_seen = [], set()
        for label, template in zip(
            role.aperture_labels, role.aperture_templates, strict=True
        ):
            aperture_positions = _label_positions(model, label)
            aperture_cells = set(aperture_positions.tolist())
            if not aperture_cells <= cells or aperture_cells & aperture_seen:
                raise ValueError(
                    "Aperture labels must be disjoint subsets of their gross surface."
                )
            aperture_seen.update(aperture_cells)
            apertures.append(
                Aperture(
                    template.aperture_id,
                    jnp.sum(areas[aperture_positions]),
                    template.u_value,
                    template.solar_transmittance,
                )
            )
        binding = canonical_fingerprint(
            {
                "kind": "building-surface-binding",
                "model": model.model_id,
                "geometry": mesh.geometry_id,
                "revision": mesh.numeric_version,
                "spatial": spatial.spatial_id,
                "label": role.label.label_id,
                "apertures": [x.label_id for x in role.aperture_labels],
            }
        )
        surfaces.append(
            Surface(
                role.label.name,
                role.zone_id,
                jnp.sum(areas[positions]),
                role.construction,
                adjacent_zone=role.adjacent_zone,
                apertures=apertures,
                geometry_binding=binding,
                boundary_id=role.boundary_id,
                adiabatic=role.adiabatic,
            )
        )
    if not surfaces:
        raise ValueError(
            "Geometry enrichment requires at least one explicit surface role."
        )
    return BuildingSource(
        zones,
        surfaces=surfaces,
        source_id=source_id,
        boundaries=boundaries,
        provenance=(
            *model.metadata.provenance,
            f"surface-model:{model.model_id}",
            f"geometry:{mesh.geometry_id}",
        ),
    )


class BuildingArchetype(StrictModule):
    """User-supplied, licensed assumptions; no bundled archetype dataset."""

    constructions: tuple[Construction, ...]
    roles: tuple[str, ...] = eqx.field(static=True)
    archetype_id: str = eqx.field(static=True)
    source_url: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        archetype_id: str,
        constructions: Mapping[str, Construction],
        *,
        source_url: str,
        license_id: str,
        assumptions: Sequence[str],
    ):
        if not constructions or not assumptions:
            raise ValueError("Archetype requires explicit constructions and assumptions.")
        self.archetype_id, self.source_url, self.license_id = (
            _text(archetype_id, "archetype_id"),
            _text(source_url, "source_url"),
            _text(license_id, "license_id"),
        )
        self.roles, self.constructions = (
            tuple(constructions),
            tuple(constructions.values()),
        )
        self.assumptions = tuple(assumptions)

    def construction(self, role: str) -> Construction:
        return self.constructions[self.roles.index(role)]

    def enrich(
        self, source: BuildingSource, surface_roles: Mapping[str, str], *, source_id: str
    ) -> BuildingSource:
        replacements = {
            surface_id: self.construction(role)
            for surface_id, role in surface_roles.items()
        }
        return retrofit_building(
            source,
            construction_replacements=replacements,
            source_id=source_id,
            provenance=(
                f"archetype:{self.archetype_id}",
                self.source_url,
                f"license:{self.license_id}",
                *self.assumptions,
            ),
        )


def retrofit_building(
    source: BuildingSource,
    *,
    source_id: str,
    construction_replacements: Mapping[str, Construction] | None = None,
    zone_replacements: Mapping[str, Zone] | None = None,
    provenance: Sequence[str] = (),
) -> BuildingSource:
    """Return a new physical source, never patch solver matrices or mutate baseline."""
    constructions = (
        {} if construction_replacements is None else dict(construction_replacements)
    )
    zones = {} if zone_replacements is None else dict(zone_replacements)
    if not set(constructions) <= {s.surface_id for s in source.surfaces} or not set(
        zones
    ) <= {z.zone_id for z in source.zones}:
        raise ValueError("Retrofit references unknown source elements.")
    if any(key != value.zone_id for key, value in zones.items()):
        raise ValueError("Zone replacements must preserve source zone identity.")
    if source_id == source.source_id:
        raise ValueError("Retrofit source_id must distinguish the changed source.")
    surfaces = tuple(
        Surface(
            s.surface_id,
            s.zone_id,
            s.area,
            constructions.get(s.surface_id, s.construction),
            adjacent_zone=s.adjacent_zone,
            apertures=s.apertures,
            geometry_binding=s.geometry_binding,
            boundary_id=s.boundary_id,
            adiabatic=s.adiabatic,
        )
        for s in source.surfaces
    )
    return BuildingSource(
        tuple(zones.get(z.zone_id, z) for z in source.zones),
        surfaces=surfaces,
        adjacencies=source.adjacencies,
        source_id=source_id,
        boundaries=source.boundaries,
        ventilation=source.ventilation,
        provenance=(*source.provenance, f"retrofit-of:{source.source_id}", *provenance),
    )
