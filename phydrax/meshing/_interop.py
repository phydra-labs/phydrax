#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np

from .._physical import SpatialCoordinateContract
from ..discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    FiniteElementSpec,
    lagrange_element,
)
from ..discretization._cell_ordering import MESHIO_CELL_TYPES, reference_node_permutation
from ..interchange import (
    AdapterCapability,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    AdapterWaiver,
    MeshArrayArtifact,
    MeshArrayAssociation,
    MeshArrayBlock,
    MeshArrayField,
)
from ..interchange._mesh_arrays import MeshArraySelection
from ._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshLabel,
    MeshZone,
    MeshZoneRole,
    validate_mesh_labels,
    validate_mesh_zones,
)
from ._scope import MeshingEntityKind, MeshingScope


_POINT_IDS = "phydrax_point_global_ids"
_CELL_IDS = "phydrax_cell_global_ids"


@dataclass(frozen=True, slots=True)
class MeshInteropPolicy:
    coordinate_contract: SpatialCoordinateContract
    allow_lossy: bool = False
    maximum_file_bytes: int = 1_000_000_000
    maximum_data_bytes: int = 4_000_000_000
    maximum_vertices: int = 20_000_000
    maximum_cells: int = 50_000_000

    def __post_init__(self):
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if not isinstance(self.allow_lossy, bool):
            raise TypeError("allow_lossy must be a boolean.")
        for value in (
            self.maximum_file_bytes,
            self.maximum_data_bytes,
            self.maximum_vertices,
            self.maximum_cells,
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError("Mesh interchange limits must be positive integers.")


@dataclass(frozen=True, slots=True)
class CellMeshImportResult:
    mesh: CellMesh
    geometry: CellGeometrySpec
    attributes: tuple[MeshAttribute, ...]
    artifact: MeshArrayArtifact
    report: AdapterReport
    zones: tuple[MeshZone, ...] = ()
    labels: tuple[MeshLabel, ...] = ()


@dataclass(frozen=True, slots=True)
class CellMeshExportResult:
    path: Path
    report: AdapterReport


def _decoded_bytes(source) -> int:
    def size(value):
        if isinstance(value, dict):
            return sum(size(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(size(item) for item in value)
        return 0 if value is None else int(np.asarray(value).nbytes)

    return sum(
        size(value)
        for value in (
            source.points,
            [block.data for block in source.cells],
            source.point_data,
            source.cell_data,
            source.field_data,
            source.point_sets,
            source.cell_sets,
            source.gmsh_periodic,
            source.info,
        )
    )


def _loss(path, direction, reason, *, category="dropped", changes_interpretation=True):
    return AdapterLoss(
        path, direction, category, reason, changes_interpretation=changes_interpretation
    )


def _require_permission(losses, policy):
    changing = tuple(loss for loss in losses if loss.changes_interpretation)
    if changing and not policy.allow_lossy:
        raise ValueError(
            "Mesh interchange requires explicit loss permission for "
            + ", ".join(loss.path for loss in changing)
            + "."
        )
    return tuple(
        AdapterWaiver(loss, "Explicit MeshInteropPolicy.allow_lossy") for loss in changing
    )


def _report(
    source_format,
    target_format,
    source_id,
    target_id,
    losses,
    policy,
    *,
    preserved=(),
    native=False,
    assumptions=(),
):
    return AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        source_format,
        target_format,
        source_id=source_id,
        target_id=target_id,
        coordinate_mapping=("identity",),
        preserved_fields=tuple(dict.fromkeys(preserved)),
        losses=tuple(losses),
        waivers=_require_permission(losses, policy),
        capabilities=tuple(
            AdapterCapability(name)
            for name in (
                (
                    "cell_connectivity",
                    "high_order_geometry",
                    "numeric_fields",
                    "spatial_coordinate_contract",
                    "organization",
                )
                if native
                else ("cell_connectivity", "high_order_geometry", "numeric_fields")
            )
        ),
        assumptions=assumptions,
    )


def _check_array_limits(artifact, policy):
    if artifact.points.shape[0] > policy.maximum_vertices:
        raise ValueError("Mesh artifact exceeds maximum_vertices.")
    if (
        sum(block.connectivity.shape[0] for block in artifact.blocks)
        > policy.maximum_cells
    ):
        raise ValueError("Mesh artifact exceeds maximum_cells.")
    arrays = [artifact.points, artifact.point_global_ids]
    for block in artifact.blocks:
        arrays.extend((block.connectivity, block.global_ids))
    for field in artifact.fields:
        arrays.append(field.values)
        if field.entity_ids is not None:
            arrays.append(field.entity_ids)
    arrays.extend(values for _, values in artifact.entity_global_ids)
    arrays.extend(
        selection.entity_ids for selection in (*artifact.zones, *artifact.labels)
    )
    if artifact.vertex_point_indices is not None:
        arrays.append(artifact.vertex_point_indices)
    if artifact.vertex_coordinates is not None:
        arrays.append(artifact.vertex_coordinates)
    if sum(array.nbytes for array in arrays) > policy.maximum_data_bytes:
        raise ValueError("Mesh artifact exceeds maximum_data_bytes.")


def read_mesh_array_artifact(
    path: str | Path, policy: MeshInteropPolicy, /
) -> tuple[MeshArrayArtifact, AdapterReport]:
    """Decode numeric meshio arrays; unsupported external semantics are explicit losses."""
    if not isinstance(policy, MeshInteropPolicy):
        raise TypeError("policy must be MeshInteropPolicy.")
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.stat().st_size > policy.maximum_file_bytes:
        raise ValueError("Mesh artifact exceeds maximum_file_bytes.")
    source = meshio.read(source_path)
    if _decoded_bytes(source) > policy.maximum_data_bytes:
        raise ValueError("Decoded mesh artifact exceeds maximum_data_bytes.")
    points = np.asarray(source.points)
    if points.ndim != 2 or 0 in points.shape or not np.all(np.isfinite(points)):
        raise ValueError("Mesh artifact points must be one non-empty finite matrix.")
    if points.shape[0] > policy.maximum_vertices:
        raise ValueError("Mesh artifact exceeds maximum_vertices.")
    if sum(len(block.data) for block in source.cells) > policy.maximum_cells:
        raise ValueError("Mesh artifact exceeds maximum_cells.")
    point_ids = source.point_data.get(_POINT_IDS)
    losses = []
    if point_ids is None:
        point_ids = np.arange(points.shape[0], dtype=np.int64)
        losses.append(
            _loss(
                "point_global_ids",
                "import",
                "Source supplied no Phydrax point identities.",
                category="synthesized",
                changes_interpretation=False,
            )
        )
    for name, metadata in (
        ("field_data", source.field_data),
        ("point_sets", source.point_sets),
        ("cell_sets", source.cell_sets),
        ("gmsh_periodic", source.gmsh_periodic),
        ("info", source.info),
    ):
        if metadata is not None and len(metadata):
            losses.append(
                _loss(
                    name,
                    "import",
                    f"External {name} semantics are not represented by meshio numeric fields.",
                )
            )
    for name, values in source.cell_data.items():
        if len(values) != len(source.cells):
            raise ValueError(
                f"Cell data {name!r} must contain one array per source block."
            )
    blocks, supported = [], []
    offset = 0
    for index, block in enumerate(source.cells):
        count = len(block.data)
        if block.type not in MESHIO_CELL_TYPES:
            losses.append(
                _loss(
                    f"cells[{index}]",
                    "import",
                    f"Cell type {block.type!r} is unsupported by CellMesh interchange.",
                )
            )
            offset += count
            continue
        kind, order, _ = MESHIO_CELL_TYPES[block.type]
        raw_ids = source.cell_data.get(_CELL_IDS)
        ids = (
            np.arange(offset, offset + count, dtype=np.int64)
            if raw_ids is None
            else raw_ids[index]
        )
        if raw_ids is None:
            losses.append(
                _loss(
                    f"cells[{index}].global_ids",
                    "import",
                    "Source supplied no Phydrax cell identities.",
                    category="synthesized",
                    changes_interpretation=False,
                )
            )
        blocks.append(
            MeshArrayBlock(f"{kind}-{index}", kind, block.type, order, block.data, ids)
        )
        supported.append(index)
        offset += count
    if not blocks:
        raise ValueError("Mesh artifact contains no supported CellMesh blocks.")
    fields = []
    for name, values in source.point_data.items():
        if name != _POINT_IDS:
            fields.append(MeshArrayField(name, MeshArrayAssociation.POINT, values))
    for name, values in source.cell_data.items():
        if name == _CELL_IDS:
            continue
        for block, index in zip(blocks, supported, strict=True):
            fields.append(
                MeshArrayField(
                    name, MeshArrayAssociation.CELL, values[index], block_name=block.name
                )
            )
    artifact = MeshArrayArtifact(
        points,
        point_ids,
        tuple(blocks),
        policy.coordinate_contract,
        source_id=str(source_path),
        source_format=source_path.suffix.lower() or "meshio",
        fields=tuple(fields),
    )
    _check_array_limits(artifact, policy)
    report = _report(
        artifact.source_format,
        "phydrax-mesh-arrays",
        str(source_path),
        artifact.artifact_id,
        losses,
        policy,
        preserved=(field.name for field in fields),
        assumptions=(
            f"coordinate-contract-supplied-by-caller:{policy.coordinate_contract.spatial_id}",
        ),
    )
    return artifact, report


def _scope(mesh, dimension, identifiers, kind="mesh"):
    entity_set = mesh.entity_set(dimension)
    if not np.all(np.isin(identifiers, np.asarray(entity_set.entity_ids))):
        raise ValueError("Mesh interchange scope indexes undeclared entity IDs.")
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind(kind),
        dimension,
        entity_set.entity_set_id,
        identifiers,
    )


def _validate_scope(scope, mesh):
    if scope.source_id != mesh.mesh_id or scope.source_revision != mesh.numeric_version:
        raise ValueError("Mesh interchange scope has a stale or foreign source binding.")
    entity_set = mesh.entity_set(scope.entity_dimension)
    if scope.entity_set_id != entity_set.entity_set_id or not np.all(
        np.isin(np.asarray(scope.entity_ids), np.asarray(entity_set.entity_ids))
    ):
        raise ValueError("Mesh interchange scope has a foreign entity set binding.")


def import_cell_mesh(
    path: str | Path | MeshArrayArtifact, policy: MeshInteropPolicy, /
) -> CellMeshImportResult:
    """Construct a CellMesh from either a native artifact or external numeric arrays."""
    if not isinstance(policy, MeshInteropPolicy):
        raise TypeError("policy must be MeshInteropPolicy.")
    if isinstance(path, MeshArrayArtifact):
        artifact = path
        if (
            artifact.coordinate_contract.spatial_id
            != policy.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Native artifact coordinate contract conflicts with import policy."
            )
        losses = []
        source_format, source_id = "phydrax-mesh-arrays", artifact.artifact_id
        assumptions = ()
    else:
        artifact, source_report = read_mesh_array_artifact(path, policy)
        losses = list(source_report.losses)
        source_format, source_id = source_report.source_format, source_report.source_id
        assumptions = source_report.assumptions
    _check_array_limits(artifact, policy)
    dimensions = set()
    corner_sets = []
    elements, routes = {}, {}
    for block in artifact.blocks:
        if block.source_cell_type not in MESHIO_CELL_TYPES:
            raise ValueError(
                f"Unsupported native source cell type {block.source_cell_type!r}."
            )
        kind, order, count = MESHIO_CELL_TYPES[block.source_cell_type]
        if (kind, order) != (block.cell_kind, block.geometry_order):
            raise ValueError("Native cell kind/order contradicts its source cell type.")
        element = lagrange_element(kind, order)
        permutation = reference_node_permutation(
            block.source_cell_type, element.reference_nodes
        )
        if block.connectivity.shape[1] != permutation.size:
            raise ValueError(
                "Geometry connectivity width does not match its reference cell ordering."
            )
        dimensions.add(
            1 if kind == "interval" else 2 if kind in ("triangle", "quadrilateral") else 3
        )
        corner_sets.append(block.connectivity[:, :count])
        elements[block.name] = element
        routes[block.name] = block.connectivity[:, permutation]
    if len(dimensions) != 1:
        raise ValueError("CellMesh import requires one topological dimension.")
    dimension = dimensions.pop()
    corner_nodes = artifact.vertex_point_indices
    if corner_nodes is None:
        corner_nodes = np.unique(
            np.concatenate([values.reshape(-1) for values in corner_sets])
        )
    if not np.all(
        np.isin(
            np.concatenate([values.reshape(-1) for values in corner_sets]), corner_nodes
        )
    ):
        raise ValueError("Native vertex point mapping omits cell corners.")
    source_to_corner = np.full(artifact.points.shape[0], -1, dtype=np.int64)
    source_to_corner[corner_nodes] = np.arange(corner_nodes.size)
    blocks = [
        CellBlock(
            block.name,
            block.cell_kind,
            source_to_corner[corners],
            global_ids=block.global_ids,
        )
        for block, corners in zip(artifact.blocks, corner_sets, strict=True)
    ]
    if dimension == 2:
        blocks.sort(key=lambda block: block.arity)
    mesh = CellMesh(
        artifact.points[corner_nodes]
        if artifact.vertex_coordinates is None
        else artifact.vertex_coordinates,
        tuple(blocks),
        vertex_global_ids=artifact.point_global_ids[corner_nodes],
        entity_global_ids=dict(artifact.entity_global_ids),
        numeric_version=artifact.numeric_version,
    )
    geometry = CellGeometrySpec(elements, routes, artifact.points)
    attributes = []
    projected_fields = set()
    block_by_name = {block.name: block for block in mesh.blocks}
    for field in artifact.fields:
        if field.association is MeshArrayAssociation.POINT:
            identifiers = artifact.point_global_ids[corner_nodes]
            values = field.values[corner_nodes]
            entity_dimension = 0
            if corner_nodes.size != artifact.points.shape[0]:
                projected_fields.add(field.name)
                losses.append(
                    _loss(
                        f"field.{field.name}.nonvertex_values",
                        "import",
                        "Non-vertex field values remain in the artifact, not in vertex MeshAttributes.",
                    )
                )
        elif field.association is MeshArrayAssociation.CELL:
            block_name = field.block_name
            if block_name is None:
                raise ValueError("Cell fields require a block_name.")
            identifiers = np.asarray(block_by_name[block_name].global_ids)
            values = field.values
            entity_dimension = dimension
        else:
            identifiers, values, entity_dimension = (
                field.entity_ids,
                field.values,
                field.entity_dimension,
            )
            if identifiers is None or entity_dimension is None:
                raise ValueError("Entity fields require a dimension and scoped IDs.")
        order = np.argsort(identifiers, kind="stable")
        attributes.append(
            MeshAttribute(
                field.name,
                MeshAttributeRole(field.role),
                _scope(mesh, entity_dimension, identifiers, field.entity_kind),
                values[order],
                unit=field.unit,
            )
        )
    zones = validate_mesh_zones(
        tuple(
            MeshZone(
                selection.name,
                MeshZoneRole(selection.role),
                _scope(
                    mesh,
                    selection.entity_dimension,
                    selection.entity_ids,
                    selection.entity_kind,
                ),
            )
            for selection in artifact.zones
        )
    )
    labels = validate_mesh_labels(
        tuple(
            MeshLabel(
                selection.name,
                _scope(
                    mesh,
                    selection.entity_dimension,
                    selection.entity_ids,
                    selection.entity_kind,
                ),
            )
            for selection in artifact.labels
        )
    )
    report = _report(
        source_format,
        "phydrax-cell-mesh",
        source_id,
        mesh.mesh_id,
        losses,
        policy,
        preserved=(
            field.name for field in artifact.fields if field.name not in projected_fields
        ),
        native=isinstance(path, MeshArrayArtifact),
        assumptions=assumptions,
    )
    return CellMeshImportResult(
        mesh, geometry, tuple(attributes), artifact, report, zones, labels
    )


def export_mesh_array_artifact(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    policy: MeshInteropPolicy,
    /,
    *,
    attributes: tuple[MeshAttribute, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
    point_global_ids=None,
) -> tuple[MeshArrayArtifact, AdapterReport]:
    """Preserve native mesh semantics without imposing external-file limitations.

    Supply point_global_ids from an imported artifact to retain identities of
    non-vertex geometry nodes, which CellGeometrySpec itself does not own.
    """
    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError("mesh and geometry must be CellMesh and CellGeometrySpec.")
    if not isinstance(policy, MeshInteropPolicy):
        raise TypeError("policy must be MeshInteropPolicy.")
    elements, routes, points = geometry.resolve(mesh)
    points = np.asarray(points)
    if (
        points.shape[0] > policy.maximum_vertices
        or sum(block.cell_count for block in mesh.blocks) > policy.maximum_cells
    ):
        raise ValueError("Mesh export exceeds maximum_vertices or maximum_cells.")
    reverse_types = {
        (kind, order): source for source, (kind, order, _) in MESHIO_CELL_TYPES.items()
    }
    blocks, vertices, nodes = [], [], []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if not isinstance(element, FiniteElementSpec):
            raise TypeError(
                "Mesh export requires FiniteElementSpec coordinate elements with a supported meshio ordering."
            )
        key = (block.cell_kind, element.degree)
        if key not in reverse_types:
            raise ValueError(f"No meshio cell ordering for geometry tuple {key!r}.")
        source_type = reverse_types[key]
        permutation = reference_node_permutation(source_type, element.reference_nodes)
        connectivity = np.asarray(route)[:, np.argsort(permutation)]
        blocks.append(
            MeshArrayBlock(
                block.name,
                block.cell_kind,
                source_type,
                element.degree,
                connectivity,
                np.asarray(block.global_ids),
            )
        )
        vertices.append(np.asarray(block.vertices).reshape(-1))
        nodes.append(connectivity[:, : block.arity].reshape(-1))
    vertex_array, node_array = np.concatenate(vertices), np.concatenate(nodes)
    pairs = np.unique(np.column_stack((vertex_array, node_array)), axis=0)
    if (
        np.unique(pairs[:, 0]).size != pairs.shape[0]
        or np.unique(pairs[:, 1]).size != pairs.shape[0]
    ):
        raise ValueError("Geometry corners must map mesh vertices one-to-one.")
    vertex_nodes = np.full(mesh.coordinates.shape[0], -1, dtype=np.int64)
    vertex_nodes[pairs[:, 0]] = pairs[:, 1]
    missing = np.flatnonzero(vertex_nodes < 0)
    if missing.size:
        if (
            points.shape != mesh.coordinates.shape
            or not np.array_equal(points, np.asarray(mesh.coordinates))
            or np.any(np.isin(missing, pairs[:, 1]))
        ):
            raise ValueError("Geometry does not identify isolated mesh vertices.")
        vertex_nodes[missing] = missing
    losses = []
    if point_global_ids is None:
        point_ids = np.full(points.shape[0], -1, dtype=np.int64)
        point_ids[vertex_nodes] = np.asarray(mesh.vertex_global_ids)
        other_nodes = np.flatnonzero(point_ids < 0)
        if other_nodes.size:
            # Smallest unused IDs avoid overflow even when a vertex uses INT64_MAX.
            candidates = np.arange(points.shape[0], dtype=np.int64)
            available = candidates[
                ~np.isin(candidates, np.asarray(mesh.vertex_global_ids))
            ]
            point_ids[other_nodes] = available[: other_nodes.size]
            losses.append(
                _loss(
                    "high_order_point_global_ids",
                    "export",
                    "CellGeometrySpec has no non-vertex point identities; supplied new unused IDs.",
                    category="synthesized",
                    changes_interpretation=False,
                )
            )
    else:
        point_ids = np.asarray(point_global_ids)
        if point_ids.shape != (points.shape[0],) or not np.array_equal(
            point_ids[vertex_nodes], np.asarray(mesh.vertex_global_ids)
        ):
            raise ValueError(
                "Geometry point IDs must preserve the mesh vertex identities."
            )
    fields = []
    for attribute in attributes:
        if not isinstance(attribute, MeshAttribute):
            raise TypeError("attributes must contain MeshAttribute values.")
        _validate_scope(attribute.scope, mesh)
        fields.append(
            MeshArrayField(
                attribute.name,
                MeshArrayAssociation.ENTITY,
                np.asarray(attribute.values),
                role=attribute.role.value,
                unit=attribute.unit,
                entity_dimension=attribute.scope.entity_dimension,
                entity_ids=np.asarray(attribute.scope.entity_ids),
                entity_kind=attribute.scope.entity_kind.value,
            )
        )
    zones, labels = validate_mesh_zones(tuple(zones)), validate_mesh_labels(tuple(labels))
    for selection in (*zones, *labels):
        _validate_scope(selection.scope, mesh)

    def selection(value):
        return MeshArraySelection(
            value.name,
            value.scope.entity_dimension,
            np.asarray(value.scope.entity_ids),
            role=value.role.value if isinstance(value, MeshZone) else None,
            entity_kind=value.scope.entity_kind.value,
        )

    artifact = MeshArrayArtifact(
        points,
        point_ids,
        tuple(blocks),
        policy.coordinate_contract,
        source_id=mesh.mesh_id,
        source_format="phydrax-cell-mesh",
        fields=tuple(fields),
        numeric_version=mesh.numeric_version,
        vertex_point_indices=vertex_nodes,
        vertex_coordinates=np.asarray(mesh.coordinates),
        entity_global_ids=tuple(
            (dimension, np.asarray(mesh.entity_set(dimension).entity_ids))
            for dimension in range(mesh.topological_dimension + 1)
        ),
        zones=tuple(selection(value) for value in zones),
        labels=tuple(selection(value) for value in labels),
    )
    _check_array_limits(artifact, policy)
    report = _report(
        "phydrax-cell-mesh",
        "phydrax-mesh-arrays",
        mesh.mesh_id,
        artifact.artifact_id,
        losses,
        policy,
        preserved=(field.name for field in fields),
        native=True,
    )
    return artifact, report


def _external_fields(artifact, dimension, losses):
    point_data = {_POINT_IDS: artifact.point_global_ids}
    cell_data = {_CELL_IDS: [block.global_ids for block in artifact.blocks]}
    groups = {}
    for field in artifact.fields:
        if field.name in (_POINT_IDS, _CELL_IDS):
            raise ValueError("Mesh attribute name collides with reserved identity data.")
        if field.role != MeshAttributeRole.USER.value:
            losses.append(
                _loss(
                    f"attribute.{field.name}.role",
                    "export",
                    "meshio numeric fields do not encode attribute roles.",
                )
            )
        if field.unit is not None:
            losses.append(
                _loss(
                    f"attribute.{field.name}.unit",
                    "export",
                    "meshio numeric fields do not encode attribute units.",
                )
            )
        if field.entity_kind != MeshingEntityKind.MESH.value:
            losses.append(
                _loss(
                    f"attribute.{field.name}.entity_kind",
                    "export",
                    "meshio numeric fields do not encode semantic scope kinds.",
                )
            )
        groups.setdefault((field.name, field.entity_dimension), []).append(field)
    preserved = []
    for (name, field_dimension), fields in groups.items():
        if field_dimension not in (0, dimension):
            losses.append(
                _loss(
                    f"attribute.{name}",
                    "export",
                    "meshio cell artifacts cannot port this entity association.",
                )
            )
            continue
        scoped_ids = []
        for field in fields:
            if field.entity_ids is None:
                raise ValueError("External attributes require scoped entity IDs.")
            scoped_ids.append(field.entity_ids)
        identifiers = np.concatenate(scoped_ids)
        shapes = {(field.values.shape[1:], field.values.dtype.str) for field in fields}
        if len(shapes) != 1 or np.unique(identifiers).size != identifiers.size:
            raise ValueError(
                "Same-name external attributes require disjoint scopes and identical value layouts."
            )
        values = np.concatenate([field.values for field in fields])
        if values.ndim > 2 or 0 in values.shape:
            losses.append(
                _loss(
                    f"attribute.{name}",
                    "export",
                    "meshio fields cannot preserve this component tensor shape.",
                )
            )
            continue
        if values.dtype.kind == "b":
            values = values.astype(np.uint8)
            losses.append(
                _loss(
                    f"attribute.{name}.dtype",
                    "export",
                    "meshio numeric fields encode booleans as unsigned bytes.",
                    category="transformed",
                )
            )
        expected = (
            artifact.point_global_ids
            if field_dimension == 0
            else np.concatenate([block.global_ids for block in artifact.blocks])
        )
        if identifiers.size != expected.size or not np.all(
            np.isin(expected, identifiers)
        ):
            losses.append(
                _loss(
                    f"attribute.{name}",
                    "export",
                    "Partial entity scopes cannot be represented as dense meshio fields without inventing values.",
                )
            )
            continue
        order = np.argsort(identifiers)
        ordered = values[order[np.searchsorted(identifiers[order], expected)]]
        if field_dimension == 0:
            point_data[name] = ordered
        else:
            offsets = np.cumsum(
                [block.connectivity.shape[0] for block in artifact.blocks]
            )[:-1]
            cell_data[name] = list(np.split(ordered, offsets))
        preserved.append(name)
    return point_data, cell_data, preserved


def _written_array_losses(target, decoded):
    """Check the actual codec rather than assuming all meshio writers preserve data."""
    losses = []
    expected_points = target.points
    actual_points = np.asarray(decoded.points)
    if (
        actual_points.shape[0] == expected_points.shape[0]
        and actual_points.shape[1] > expected_points.shape[1]
        and np.all(actual_points[:, expected_points.shape[1] :] == 0)
    ):
        actual_points = actual_points[:, : expected_points.shape[1]]
    if not np.array_equal(actual_points, expected_points):
        losses.append(
            _loss(
                "points",
                "export",
                "The external codec changes point coordinates or ordering.",
            )
        )
    elif decoded.points.shape != target.points.shape:
        losses.append(
            _loss(
                "ambient_dimension",
                "export",
                "The external codec pads coordinate dimensions with zeros.",
                category="transformed",
                changes_interpretation=False,
            )
        )
    if len(target.cells) != len(decoded.cells) or any(
        a.type != b.type or not np.array_equal(a.data, b.data)
        for a, b in zip(target.cells, decoded.cells)
    ):
        losses.append(
            _loss(
                "cell_connectivity",
                "export",
                "The external codec changes cell types, block grouping, or connectivity ordering.",
            )
        )
    for name, values in target.point_data.items():
        if (
            name not in decoded.point_data
            or values.dtype.kind != decoded.point_data[name].dtype.kind
            or values.dtype.itemsize != decoded.point_data[name].dtype.itemsize
            or not np.array_equal(values, decoded.point_data[name])
        ):
            losses.append(
                _loss(
                    f"point_data.{name}",
                    "export",
                    "The external codec drops or changes this point field.",
                )
            )
    for name, values in target.cell_data.items():
        actual = decoded.cell_data.get(name)
        if (
            actual is None
            or len(values) != len(actual)
            or any(
                a.dtype.kind != b.dtype.kind
                or a.dtype.itemsize != b.dtype.itemsize
                or not np.array_equal(a, b)
                for a, b in zip(values, actual)
            )
        ):
            losses.append(
                _loss(
                    f"cell_data.{name}",
                    "export",
                    "The external codec drops or changes this cell field.",
                )
            )
    return losses


def export_cell_mesh(
    path: str | Path,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    policy: MeshInteropPolicy,
    /,
    *,
    attributes: tuple[MeshAttribute, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
    point_global_ids=None,
) -> CellMeshExportResult:
    """Write meshio arrays with explicit accounting for native semantic losses."""
    artifact, native_report = export_mesh_array_artifact(
        mesh,
        geometry,
        policy,
        attributes=attributes,
        zones=zones,
        labels=labels,
        point_global_ids=point_global_ids,
    )
    losses = list(native_report.losses)
    losses.extend(
        (
            _loss(
                "coordinate_contract",
                "export",
                "meshio omits physical coordinate contracts; readers must supply them out of band.",
            ),
            _loss(
                "block_names",
                "export",
                "meshio cell blocks do not serialize native block names.",
            ),
            _loss(
                "source_revision",
                "export",
                "meshio arrays do not serialize native mesh identity or numeric revision.",
            ),
        )
    )
    for dimension, _ in artifact.entity_global_ids:
        if dimension not in (0, mesh.topological_dimension):
            losses.append(
                _loss(
                    f"entity_global_ids[{dimension}]",
                    "export",
                    "meshio top-dimensional cells do not serialize lower-dimensional entity IDs.",
                )
            )
    for name, selections in (("zones", artifact.zones), ("labels", artifact.labels)):
        if selections:
            losses.append(
                _loss(
                    name,
                    "export",
                    "meshio numeric cell artifacts do not encode native organization semantics.",
                )
            )
    vertex_coordinates, vertex_point_indices = (
        artifact.vertex_coordinates,
        artifact.vertex_point_indices,
    )
    if vertex_coordinates is None or vertex_point_indices is None:
        raise ValueError(
            "Native mesh export requires vertex coordinates and their geometry point mapping."
        )
    if not np.array_equal(vertex_coordinates, artifact.points[vertex_point_indices]):
        losses.append(
            _loss(
                "vertex_coordinates",
                "export",
                "The external file represents geometry coordinates, not separate topology vertex coordinates.",
            )
        )
    if not np.array_equal(
        vertex_point_indices,
        np.unique(
            np.concatenate(
                [
                    block.connectivity[
                        :, : MESHIO_CELL_TYPES[block.source_cell_type][2]
                    ].reshape(-1)
                    for block in artifact.blocks
                ]
            )
        ),
    ):
        losses.append(
            _loss(
                "vertex_order",
                "export",
                "The external file does not encode the native vertex ordering or isolated vertex membership.",
            )
        )
    point_data, cell_data, preserved = _external_fields(
        artifact, mesh.topological_dimension, losses
    )
    _require_permission(losses, policy)
    destination = Path(path).expanduser().resolve()
    if artifact.points.shape[1] > 3:
        raise ValueError(
            "External meshio export supports at most three coordinate dimensions."
        )
    external_points = np.zeros((artifact.points.shape[0], 3), dtype=artifact.points.dtype)
    external_points[:, : artifact.points.shape[1]] = artifact.points
    # Own arrays: meshio writers may pad coordinates or reshape data in place.
    target = meshio.Mesh(
        external_points,
        [
            (block.source_cell_type, np.array(block.connectivity, copy=True))
            for block in artifact.blocks
        ],
        point_data={
            name: np.array(value, copy=True) for name, value in point_data.items()
        },
        cell_data={
            name: [np.array(value, copy=True) for value in values]
            for name, values in cell_data.items()
        },
    )
    if _decoded_bytes(target) > policy.maximum_data_bytes:
        raise ValueError("Mesh export exceeds maximum_data_bytes.")
    # Some meshio formats create companion files, so retain the requested path.
    # Every external export already requires explicit native-metadata loss permission.
    meshio.write(destination, target)
    if destination.stat().st_size > policy.maximum_file_bytes:
        raise ValueError("Mesh export exceeds maximum_file_bytes.")
    decoded = meshio.read(destination)
    if _decoded_bytes(decoded) > policy.maximum_data_bytes:
        raise ValueError("Decoded mesh export exceeds maximum_data_bytes.")
    written_losses = _written_array_losses(
        meshio.Mesh(
            artifact.points,
            [(block.source_cell_type, block.connectivity) for block in artifact.blocks],
            point_data=point_data,
            cell_data=cell_data,
        ),
        decoded,
    )
    losses.extend(written_losses)
    changed_names = {
        loss.path.split(".", 1)[1]
        for loss in written_losses
        if loss.path.startswith(("point_data.", "cell_data."))
    }
    report = _report(
        "phydrax-cell-mesh",
        destination.suffix.lower() or "meshio",
        mesh.mesh_id,
        str(destination),
        losses,
        policy,
        preserved=(name for name in preserved if name not in changed_names),
    )
    return CellMeshExportResult(destination, report)


__all__ = [
    "CellMeshExportResult",
    "CellMeshImportResult",
    "MeshInteropPolicy",
    "export_cell_mesh",
    "export_mesh_array_artifact",
    "import_cell_mesh",
    "read_mesh_array_artifact",
]
