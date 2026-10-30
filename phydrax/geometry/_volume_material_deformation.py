from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from fractions import Fraction
from typing import Any

import equinox as eqx
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_transition_types import MaterialChartDeformation
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import axes
from ..discretization._nested_reference_types import (
    freeze_reference_chart_arguments,
    reference_chart_arguments_identity,
    ReferenceCellMapWitnesses,
    validate_reference_cell_map_witnesses,
)
from ..discretization._reference_cell import reference_cell_topology
from ._domain_certification import DomainCoverageCertificate


class MaterialReferencePolicy(str, Enum):
    """Explicit policy for declaring a material reference configuration."""

    ACCEPTED_CONFIGURATION = "accepted-configuration"


def _canonical_rows(values: tuple[str, ...] | list[str], role: str, /) -> tuple[str, ...]:
    rows = tuple(str(value).strip() for value in values)
    if not rows or any(not value for value in rows):
        raise ValueError(f"{role} must contain nonempty identities for every cell.")
    return rows


def _cell_count(mesh: CellMesh, /) -> int:
    return sum(block.cell_count for block in mesh.blocks)


def _identity_reference_maps(
    mesh: CellMesh, /
) -> tuple[tuple[str, ReferenceCellMapWitnesses], ...]:
    """Make the explicit reference configuration's exact identity chart bank."""
    records = []
    arguments = freeze_reference_chart_arguments((axes(3),))
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        if topology.dimension != 3:
            raise ValueError(
                "A material volume reference requires three-dimensional cells."
            )
        exact_vertices = tuple(
            tuple(
                tuple(Fraction(float(value)) for value in point)
                for point in topology.vertices
            )
            for _ in range(block.cell_count)
        )
        vertices = np.asarray(
            [
                [[float(value) for value in point] for point in cell]
                for cell in exact_vertices
            ],
            dtype=np.float64,
        )
        target_ids = np.arange(block.cell_count, dtype=np.int64)
        source_ids = np.array(block.global_ids, dtype=np.int64, copy=True)
        for bank in (vertices, target_ids, source_ids):
            bank.setflags(write=False)
        rows = ReferenceCellMapWitnesses(
            target_ids,
            source_ids,
            block.cell_kind,
            block.cell_kind,
            vertices,
            exact_vertices,
            tuple(arguments[0] for _ in range(block.cell_count)),
        )
        validate_reference_cell_map_witnesses(rows)
        records.append((block.name, rows))
    return tuple(records)


def _identity_reference_exact_vertices(
    witness: ReferenceCellMapWitnesses, /
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    exact_vertices = witness.source_reference_exact_vertices
    if exact_vertices is None:
        raise ValueError(
            "Identity reference maps require exact source-reference vertices."
        )
    return tuple(
        tuple(tuple(str(value) for value in point) for point in cell)
        for cell in exact_vertices
    )


def _canonical_reference_maps(
    mesh: CellMesh,
    maps: Mapping[str, ReferenceCellMapWitnesses],
    role: str,
    /,
) -> tuple[tuple[str, ReferenceCellMapWitnesses], ...]:
    if not isinstance(maps, Mapping):
        raise TypeError(f"{role} reference maps must be keyed by endpoint block name.")
    expected = tuple(block.name for block in mesh.blocks)
    if set(maps) != set(expected):
        raise ValueError(
            f"{role} reference maps must cover exactly the endpoint mesh blocks."
        )
    result = []
    for block in mesh.blocks:
        value = maps[block.name]
        validate_reference_cell_map_witnesses(value)
        target_ids = np.array(value.target_cell_ids, dtype=np.int64, copy=True)
        source_ids = np.array(value.source_cell_ids, dtype=np.int64, copy=True)
        vertices = np.array(value.source_reference_vertices, dtype=np.float64, copy=True)
        if value.target_cell_kind != block.cell_kind or not np.array_equal(
            target_ids, np.arange(block.cell_count, dtype=np.int64)
        ):
            raise ValueError(
                f"{role} reference map rows must bind local cells in canonical block order."
            )
        for bank in (target_ids, source_ids, vertices):
            bank.setflags(write=False)
        copied = ReferenceCellMapWitnesses(
            target_ids,
            source_ids,
            value.target_cell_kind,
            value.source_cell_kind,
            vertices,
            value.source_reference_exact_vertices,
            value.source_reference_chart_arguments,
        )
        validate_reference_cell_map_witnesses(copied)
        result.append((block.name, copied))
    return tuple(result)


def _require_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source_map: Any,
    coverage: DomainCoverageCertificate,
    role: str,
    /,
) -> DomainCoverageCertificate:
    if not isinstance(coverage, DomainCoverageCertificate):
        raise TypeError(f"{role} endpoint requires a DomainCoverageCertificate.")
    if coverage.status != "certified":
        raise ValueError(
            f"{role} physical-domain coverage is not certified ({coverage.status!r})."
        )
    if (coverage.source_id, coverage.source_revision) != (
        source_map.source_id,
        source_map.source_revision,
    ):
        raise ValueError(
            f"{role} coverage does not match the declared material source revision."
        )
    if coverage.cell_regions.shape != (_cell_count(mesh),):
        raise ValueError(
            f"{role} coverage region rows do not match the endpoint cell count."
        )
    try:
        coverage.require_current(
            mesh,
            geometry,
            coverage.domain,
            coverage.cell_regions,
            embedding=coverage.embedding,
            source=coverage.source,
            fidelity=coverage.fidelity,
            source_coverage_witness=coverage.source_coverage_witness,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{role} physical-domain coverage is stale for its endpoint map."
        ) from error
    return coverage


class MaterialReferenceDeclaration(StrictModule, NonTrainableState):
    """An explicitly authored material-coordinate origin at an accepted state."""

    root_mesh: CellMesh
    root_geometry: CellGeometrySpec
    source_map: Any
    root_coverage: DomainCoverageCertificate
    cell_entity_ids: tuple[str, ...] = eqx.field(static=True)
    region_ids: tuple[str, ...] = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    policy: MaterialReferencePolicy = eqx.field(static=True)
    identity_reference_maps: tuple[tuple[str, ReferenceCellMapWitnesses], ...]
    declaration_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_mesh: CellMesh,
        root_geometry: CellGeometrySpec,
        source_map: Any,
        cell_entity_ids: tuple[str, ...] | list[str],
        region_ids: tuple[str, ...] | list[str],
        root_coverage: DomainCoverageCertificate,
        /,
        *,
        frame_id: str,
        policy: MaterialReferencePolicy,
        occurrence_paths: tuple[tuple[str, ...], ...]
        | list[tuple[str, ...]]
        | None = None,
    ) -> None:
        from ._volume_material_map import (
            _material_occurrence_paths,
            _require_endpoint_source_coverage,
            MaterialSourceMapCapability,
        )

        if not isinstance(root_mesh, CellMesh) or not isinstance(
            root_geometry, CellGeometrySpec
        ):
            raise TypeError(
                "A material reference requires canonical root mesh and geometry owners."
            )
        if not isinstance(source_map, MaterialSourceMapCapability):
            raise TypeError("source_map must be MaterialSourceMapCapability.")
        source_map = MaterialSourceMapCapability.revalidate(source_map)
        if not isinstance(policy, MaterialReferencePolicy):
            raise TypeError("policy must explicitly select a MaterialReferencePolicy.")
        frame = str(frame_id).strip()
        if not frame or frame != frame_id:
            raise ValueError(
                "A material reference requires a canonical nonempty frame_id."
            )
        entities = _canonical_rows(cell_entity_ids, "Reference material entity rows")
        regions = _canonical_rows(region_ids, "Reference material region rows")
        if len(entities) != _cell_count(root_mesh) or len(regions) != len(entities):
            raise ValueError(
                "Reference material identity rows must align with every root cell."
            )
        paths = _material_occurrence_paths(
            source_map, entities, regions, occurrence_paths, "reference"
        )
        root_coverage = _require_endpoint_source_coverage(
            root_mesh, root_geometry, source_map, regions, root_coverage, "reference"
        )
        identity_maps = _identity_reference_maps(root_mesh)
        self.root_mesh = root_mesh
        self.root_geometry = root_geometry
        self.source_map = source_map
        self.root_coverage = root_coverage
        self.cell_entity_ids = entities
        self.region_ids = regions
        self.occurrence_paths = paths
        self.frame_id = frame
        self.policy = policy
        self.identity_reference_maps = identity_maps
        self.declaration_id = canonical_fingerprint(
            {
                "kind": "material-reference-declaration",
                "authoring_policy": policy.value,
                "frame": frame,
                "source_map": source_map.map_id,
                "root_mesh": root_mesh.mesh_id,
                "root_geometry": cell_geometry_id(root_geometry),
                "root_coverage": root_coverage.certificate_id,
                "entities": entities,
                "regions": regions,
                "occurrences": paths,
                "identity_reference_maps": tuple(
                    {
                        "block": name,
                        "target_cell_ids": array_tree_fingerprint(
                            witness.target_cell_ids
                        ),
                        "source_cell_ids": array_tree_fingerprint(
                            witness.source_cell_ids
                        ),
                        "target_cell_kind": witness.target_cell_kind,
                        "source_cell_kind": witness.source_cell_kind,
                        "exact_vertices": _identity_reference_exact_vertices(witness),
                        "chart_arguments": reference_chart_arguments_identity(
                            witness.source_reference_chart_arguments
                        ),
                    }
                    for name, witness in identity_maps
                ),
            }
        )

    @classmethod
    def revalidate(
        cls,
        value: MaterialReferenceDeclaration,
        root_mesh: CellMesh,
        root_geometry: CellGeometrySpec,
        source_map: Any,
        root_coverage: DomainCoverageCertificate,
        /,
    ) -> MaterialReferenceDeclaration:
        if not isinstance(value, MaterialReferenceDeclaration):
            raise TypeError("value must be MaterialReferenceDeclaration.")
        rebuilt = cls(
            root_mesh,
            root_geometry,
            source_map,
            value.cell_entity_ids,
            value.region_ids,
            root_coverage,
            frame_id=value.frame_id,
            policy=value.policy,
            occurrence_paths=value.occurrence_paths,
        )
        if rebuilt.declaration_id != value.declaration_id:
            raise ValueError("Stored material reference declaration does not revalidate.")
        return rebuilt


def declare_material_reference(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source_map: Any,
    cell_entity_ids: tuple[str, ...] | list[str],
    region_ids: tuple[str, ...] | list[str],
    coverage: DomainCoverageCertificate,
    /,
    *,
    frame_id: str,
    policy: MaterialReferencePolicy,
    occurrence_paths: tuple[tuple[str, ...], ...] | list[tuple[str, ...]] | None = None,
) -> MaterialReferenceDeclaration:
    """Explicitly author this accepted configuration as the material reference."""
    if policy is not MaterialReferencePolicy.ACCEPTED_CONFIGURATION:
        raise ValueError(
            "Only explicit accepted-configuration material reference authoring is supported."
        )
    return MaterialReferenceDeclaration(
        mesh,
        geometry,
        source_map,
        cell_entity_ids,
        region_ids,
        coverage,
        frame_id=frame_id,
        policy=policy,
        occurrence_paths=occurrence_paths,
    )


class VolumeMaterialDeformation(MaterialChartDeformation, NonTrainableState):
    """Endpoint-bound material deformation over one declared reference chart."""

    reference: MaterialReferenceDeclaration
    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    source_coverage: DomainCoverageCertificate
    target_coverage: DomainCoverageCertificate
    source_reference_maps: tuple[tuple[str, ReferenceCellMapWitnesses], ...]
    target_reference_maps: tuple[tuple[str, ReferenceCellMapWitnesses], ...]
    reference_witness: Any
    reference_witness_id: str = eqx.field(static=True)
    coordinate_transition: Any
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    source_domain_coverage: str = eqx.field(static=True)
    target_domain_coverage: str = eqx.field(static=True)
    source_coverage_id: str = eqx.field(static=True)
    target_coverage_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    maximum_displacement_bound: float = eqx.field(static=True)
    deformation_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: MaterialReferenceDeclaration,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
        *,
        source_reference_maps: Mapping[str, ReferenceCellMapWitnesses],
        target_reference_maps: Mapping[str, ReferenceCellMapWitnesses],
        source_coverage: DomainCoverageCertificate,
        target_coverage: DomainCoverageCertificate,
        source_frame_id: str,
        target_frame_id: str,
        coordinate_transition: Any = None,
    ) -> None:
        from ..discretization._cell_geometry_transition_types import (
            CellGeometryTransition,
        )
        from ._common_root_volume_overlap import MaterialReferenceVolumeWitness

        if not isinstance(reference, MaterialReferenceDeclaration):
            raise TypeError("reference must be an explicit MaterialReferenceDeclaration.")
        if not all(isinstance(value, CellMesh) for value in (source_mesh, target_mesh)):
            raise TypeError("Material deformation endpoints require CellMesh owners.")
        if not all(
            isinstance(value, CellGeometrySpec)
            for value in (source_geometry, target_geometry)
        ):
            raise TypeError(
                "Material deformation endpoints require CellGeometrySpec maps."
            )
        source_frame = str(source_frame_id).strip()
        target_frame = str(target_frame_id).strip()
        if (source_frame, target_frame) != (reference.frame_id, reference.frame_id):
            raise ValueError(
                "Material deformation endpoints must remain in the declared reference frame."
            )
        source_coverage = _require_coverage(
            source_mesh, source_geometry, reference.source_map, source_coverage, "source"
        )
        target_coverage = _require_coverage(
            target_mesh, target_geometry, reference.source_map, target_coverage, "target"
        )
        source_maps = _canonical_reference_maps(
            source_mesh, source_reference_maps, "source"
        )
        target_maps = _canonical_reference_maps(
            target_mesh, target_reference_maps, "target"
        )
        if coordinate_transition is not None:
            if not isinstance(coordinate_transition, CellGeometryTransition):
                raise TypeError(
                    "coordinate_transition must be CellGeometryTransition or None."
                )
            coordinate_transition.require_bound(
                source_mesh, source_geometry, target_mesh, target_geometry
            )
        witness = MaterialReferenceVolumeWitness(
            reference.root_mesh,
            reference.root_geometry,
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            dict(source_maps),
            dict(target_maps),
        )
        displacement = float(witness.maximum_displacement_bound)
        if not np.isfinite(displacement) or displacement < 0.0:
            raise ValueError(
                "Material reference witness must retain a finite certified displacement bound."
            )
        source_geometry_id = cell_geometry_id(source_geometry)
        target_geometry_id = cell_geometry_id(target_geometry)
        self.reference = reference
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.source_geometry = source_geometry
        self.target_geometry = target_geometry
        self.source_coverage = source_coverage
        self.target_coverage = target_coverage
        self.source_reference_maps = source_maps
        self.target_reference_maps = target_maps
        self.reference_witness = witness
        self.reference_witness_id = witness.witness_id
        self.coordinate_transition = coordinate_transition
        self.source_topology_id = source_mesh.topology_id
        self.target_topology_id = target_mesh.topology_id
        self.source_geometry_id = source_geometry_id
        self.target_geometry_id = target_geometry_id
        self.domain_id = reference.source_map.domain_id
        self.source_domain_coverage = source_coverage.status
        self.target_domain_coverage = target_coverage.status
        self.source_coverage_id = source_coverage.certificate_id
        self.target_coverage_id = target_coverage.certificate_id
        self.frame_id = reference.frame_id
        self.maximum_displacement_bound = displacement
        self.deformation_id = canonical_fingerprint(
            {
                "kind": "volume-material-deformation",
                "reference": reference.declaration_id,
                "reference_witness": witness.witness_id,
                "source_mesh": source_mesh.mesh_id,
                "source_topology": source_mesh.topology_id,
                "source_geometry": source_geometry_id,
                "target_mesh": target_mesh.mesh_id,
                "target_topology": target_mesh.topology_id,
                "target_geometry": target_geometry_id,
                "source_coverage": source_coverage.certificate_id,
                "target_coverage": target_coverage.certificate_id,
                "source_frame": source_frame,
                "target_frame": target_frame,
                "coordinate_transition": (
                    None
                    if coordinate_transition is None
                    else coordinate_transition.transition_id
                ),
                "maximum_displacement_bound": displacement,
            }
        )

    def require_bound(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
    ) -> None:
        if (
            source_mesh.mesh_id != self.source_mesh.mesh_id
            or source_mesh.topology_id != self.source_topology_id
            or cell_geometry_id(source_geometry) != self.source_geometry_id
            or target_mesh.mesh_id != self.target_mesh.mesh_id
            or target_mesh.topology_id != self.target_topology_id
            or cell_geometry_id(target_geometry) != self.target_geometry_id
        ):
            raise ValueError(
                "Material deformation is stale for these endpoint mesh/geometry maps."
            )
        if (
            self.reference_witness.root_mesh_id != self.reference.root_mesh.mesh_id
            or self.reference_witness.root_geometry_id
            != cell_geometry_id(self.reference.root_geometry)
        ):
            raise ValueError(
                "Material deformation's exact root witness changed its declared reference."
            )
        if self.reference_witness.witness_id != self.reference_witness_id:
            raise ValueError(
                "Material deformation's exact root witness changed its retained maps."
            )
        if self.coordinate_transition is not None:
            self.coordinate_transition.require_bound(
                source_mesh, source_geometry, target_mesh, target_geometry
            )

    @classmethod
    def revalidate(
        cls,
        value: VolumeMaterialDeformation,
        reference: MaterialReferenceDeclaration,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
        *,
        source_coverage: DomainCoverageCertificate,
        target_coverage: DomainCoverageCertificate,
        coordinate_transition: Any = None,
    ) -> VolumeMaterialDeformation:
        if not isinstance(value, VolumeMaterialDeformation):
            raise TypeError("value must be VolumeMaterialDeformation.")
        rebuilt = cls(
            reference,
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            source_reference_maps=dict(value.source_reference_maps),
            target_reference_maps=dict(value.target_reference_maps),
            source_coverage=source_coverage,
            target_coverage=target_coverage,
            source_frame_id=value.frame_id,
            target_frame_id=value.frame_id,
            coordinate_transition=(
                value.coordinate_transition
                if coordinate_transition is None
                else coordinate_transition
            ),
        )
        if rebuilt.deformation_id != value.deformation_id:
            raise ValueError("Stored volume material deformation does not revalidate.")
        return rebuilt


__all__ = [
    "MaterialReferenceDeclaration",
    "MaterialReferencePolicy",
    "VolumeMaterialDeformation",
    "declare_material_reference",
]
