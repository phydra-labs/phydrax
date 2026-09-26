#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Mmg adaptation through the persistent ``phydrax-mmg-worker`` library worker.

Planar triangles run through mmg2d, surface triangles in space through mmgs and
tetrahedra through mmg3d (or ParMmg with the collective
``phydrax-parmmg-worker``). Region references ride on cells and boundary
references on facets; both are canonical encodings of the source result's cell
blocks, cell zones, facet patches, and facet zones, rebuilt by name on the
adapted mesh.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, final, NamedTuple

import equinox as eqx
import numpy as np

from ..._external_runtime import NativeWorkerCall, NativeWorkerPolicy
from ..._fingerprint import canonical_fingerprint
from ..._identity import SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ...interchange import AdapterLoss, AdapterReport, AdapterStatus
from ...logging import emit
from .._association import BRepAssociationTransfer
from .._audit import CellMeshAuditPolicy
from .._canonical import certify_cell_mesh
from .._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
)
from .._metric import MeshMetricField
from .._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshLabel,
    MeshPatch,
    MeshZone,
)
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._scope import MeshingEntityKind, MeshingScope
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)
from ._conversion import _check_arrays, _check_result_limits, _fresh_ids
from ._worker import ProviderWorker


# Mmg imposes Lagrangian displacement on boundary entities with this reference
# (MMG5_DISPREF) and extends it elastically into the volume.
_DISPLACED_BOUNDARY_REFERENCE = 10
# Mmg's gradation when none is requested (MMG5_HGRAD = log 1.3).
_MMG_DEFAULT_GRADATION = 1.3
_PARTITION_ATTRIBUTE = "parmmg-rank"
_BUILD_HINT = (
    "Build native/providers/mmg with CMake against an installed Mmg "
    "(find_package(mmg CONFIG)) and put phydrax-mmg-worker on PATH or set "
    "PHYDRAX_MMG_WORKER."
)


def _positive(value: float, name: str, /) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return number


def _flag(value: bool, name: str, /) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool.")
    return bool(value)


def _name(value: str, what: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string.")
    text = value.strip()
    if not text:
        raise ValueError(f"{what} must be non-empty.")
    return text


@dataclass(frozen=True, slots=True)
class MmgOptions:
    """Mmg operation controls in the source coordinate units.

    With a metric, its declared size bounds become Mmg's hmin and hmax, so
    ``minimum_size`` and ``maximum_size`` must stay None; without one they are
    optional explicit bounds (None keeps Mmg's bounding-box sizes). ``gradation``
    is Mmg's hgrad with or without a metric (None keeps Mmg's default 1.3).
    ``angle_detection`` is the ridge-detection dihedral threshold in degrees
    (None disables detection). ``insertion``/``swapping``/``relocation``/
    ``surface_modification`` switch off Mmg's noinsert/noswap/nomove/nosurf
    operators; ``optimize`` keeps the current edge sizes (Mmg ``-optim``).
    Metric sizes are adaptation targets, not hard output edge-length bounds.
    """

    hausdorff_distance: float = 0.01
    minimum_size: float | None = None
    maximum_size: float | None = None
    gradation: float | None = None
    angle_detection: float | None = 45.0
    insertion: bool = True
    swapping: bool = True
    relocation: bool = True
    surface_modification: bool = True
    optimize: bool = False

    def __post_init__(self):
        _positive(self.hausdorff_distance, "hausdorff_distance")
        for value, name in (
            (self.minimum_size, "minimum_size"),
            (self.maximum_size, "maximum_size"),
        ):
            if value is not None:
                _positive(value, name)
        if (
            self.minimum_size is not None
            and self.maximum_size is not None
            and self.minimum_size > self.maximum_size
        ):
            raise ValueError("minimum_size cannot exceed maximum_size.")
        if self.gradation is not None and (
            not np.isfinite(self.gradation) or self.gradation < 1.0
        ):
            raise ValueError("gradation must be finite and at least one.")
        if self.angle_detection is not None and not (
            0.0 < float(self.angle_detection) < 180.0
        ):
            raise ValueError(
                "angle_detection must lie strictly between 0 and 180 degrees."
            )
        for value, name in (
            (self.insertion, "insertion"),
            (self.swapping, "swapping"),
            (self.relocation, "relocation"),
            (self.surface_modification, "surface_modification"),
            (self.optimize, "optimize"),
        ):
            _flag(value, name)


def _option_parameters(options: MmgOptions, /) -> dict[str, Any]:
    def optional(value: float | None) -> float | None:
        return None if value is None else float(value)

    return {
        "hausdorff_distance": float(options.hausdorff_distance),
        "minimum_size": optional(options.minimum_size),
        "maximum_size": optional(options.maximum_size),
        "gradation": optional(options.gradation),
        "angle_detection": optional(options.angle_detection),
        "insertion": bool(options.insertion),
        "swapping": bool(options.swapping),
        "relocation": bool(options.relocation),
        "surface_modification": bool(options.surface_modification),
        "optimize": bool(options.optimize),
    }


@dataclass(frozen=True, slots=True)
class MmgLevelSet:
    """Discretize the ``isovalue`` contour of a scalar vertex field.

    Every source region (cell block and cell zone) keeps its name; cells below
    the isovalue form the ``interior`` label and cells above it the
    ``exterior`` label, the discrete contour becomes the ``interface`` patch.
    Regions named in ``unsplit_regions`` (block or cell-zone names) are not cut.
    """

    values: MeshAttribute
    isovalue: float = 0.0
    interface: str = "level-set"
    interior: str = "level-set-interior"
    exterior: str = "level-set-exterior"
    unsplit_regions: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.values, MeshAttribute):
            raise TypeError("values must be MeshAttribute.")
        if not np.isfinite(float(self.isovalue)):
            raise ValueError("isovalue must be finite.")
        names = tuple(
            _name(value, what)
            for value, what in (
                (self.interface, "interface"),
                (self.interior, "interior"),
                (self.exterior, "exterior"),
            )
        )
        if names[1] == names[2]:
            raise ValueError("interior and exterior label names must differ.")
        if isinstance(self.unsplit_regions, str):
            raise TypeError("unsplit_regions must be a tuple of region names.")
        for value in self.unsplit_regions:
            _name(value, "unsplit region")


class MmgLagrangianMode(StrEnum):
    """Mmg ``-lag`` modes: displacement only, plus swaps and moves, plus insertion."""

    DISPLACE = "displace"
    DISPLACE_AND_OPTIMIZE = "displace_and_optimize"
    DISPLACE_AND_REMESH = "displace_and_remesh"


def _lagrangian_code(mode: MmgLagrangianMode, /) -> int:
    match mode:
        case MmgLagrangianMode.DISPLACE:
            return 0
        case MmgLagrangianMode.DISPLACE_AND_OPTIMIZE:
            return 1
        case MmgLagrangianMode.DISPLACE_AND_REMESH:
            return 2
        case _:
            raise ValueError(f"Unknown Mmg Lagrangian mode {mode!r}.")


@dataclass(frozen=True, slots=True)
class MmgLagrangianMotion:
    """Move ``moving_boundary`` facets by ``displacement`` (Mmg ``-lag``).

    Mmg imposes the vertex displacement on the moving boundary only and
    extends it into the domain by linear elasticity (ISCD LinearElasticity);
    other boundaries stay fixed. Requires Mmg built with USE_ELAS.
    """

    displacement: MeshAttribute
    moving_boundary: MeshingScope
    mode: MmgLagrangianMode = MmgLagrangianMode.DISPLACE_AND_REMESH

    def __post_init__(self):
        if not isinstance(self.displacement, MeshAttribute):
            raise TypeError("displacement must be MeshAttribute.")
        if not isinstance(self.moving_boundary, MeshingScope):
            raise TypeError("moving_boundary must be MeshingScope.")
        if not isinstance(self.mode, MmgLagrangianMode):
            raise TypeError("mode must be MmgLagrangianMode.")


class _Region(NamedTuple):
    reference: int
    block: str
    zone: MeshZone | None
    source_count: int


class _Boundary(NamedTuple):
    reference: int
    patches: tuple[MeshPatch, ...]
    zone: MeshZone | None
    moving: bool
    source_count: int


class _Encoding(NamedTuple):
    backend: str
    program: str
    arrays: dict[str, np.ndarray]
    parameters: dict[str, Any]
    regions: tuple[_Region, ...]
    boundaries: tuple[_Boundary, ...]
    interface_reference: int
    metric_representation: str
    gradation: float
    field_layout: tuple[tuple[MeshAttribute, int, int], ...]
    required_vertices: int
    losses: tuple[AdapterLoss, ...]


def _backend(mesh: CellMesh, /) -> str:
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not all(isinstance(block, CellBlock) for block in mesh.blocks):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Mmg adapts triangle or tetrahedron blocks, not polyhedra.",
        )
    kinds = {block.cell_kind for block in mesh.blocks}
    if kinds == {"triangle"} and mesh.ambient_dimension in (2, 3):
        return "mmg2d" if mesh.ambient_dimension == 2 else "mmgs"
    if kinds == {"tetrahedron"} and mesh.ambient_dimension == 3:
        return "mmg3d"
    raise MeshingFailure(
        MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
        "Mmg requires affine triangles in 2D/3D or affine tetrahedra in 3D.",
    )


def _cell_ids(mesh: CellMesh, /) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )


def _entity_ids(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """IDs aligned with the rows used here: coordinates, block cells, or connectivity."""
    if dimension == 0:
        return np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if dimension == mesh.topological_dimension:
        return _cell_ids(mesh)
    return np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)


def _entity_vertices(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Vertex rows of edges or faces, aligned with ``mesh.entity_set(dimension)``."""
    connectivity = mesh.connectivity
    if dimension == 1:
        assert isinstance(
            connectivity,
            (PolygonalConnectivity, TetrahedralConnectivity, PolyhedralConnectivity),
        )
        return np.asarray(connectivity.edges, dtype=np.int64)
    if isinstance(connectivity, TetrahedralConnectivity):
        return np.asarray(connectivity.faces, dtype=np.int64)
    # Several tetrahedron blocks share packed polyhedral incidence of triangles.
    assert isinstance(connectivity, PolyhedralConnectivity) and dimension == 2
    return np.asarray(connectivity.face_vertex_values, dtype=np.int64).reshape(-1, 3)


def _scope_mask(mesh: CellMesh, scope: MeshingScope, what: str, /) -> np.ndarray:
    """Membership of an exactly bound mesh scope over this module's entity rows."""
    if not isinstance(scope, MeshingScope):
        raise TypeError(f"{what} must be MeshingScope.")
    dimension = scope.entity_dimension
    if (
        dimension > mesh.topological_dimension
        or scope.source_id != mesh.mesh_id
        or scope.source_revision != mesh.numeric_version
        or scope.entity_kind is not MeshingEntityKind.MESH
        or scope.entity_set_id != mesh.entity_set(dimension).entity_set_id
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            f"{what} must select entities of this exact mesh and numeric revision.",
        )
    requested = np.asarray(scope.entity_ids, dtype=np.int64)
    mask = np.isin(_entity_ids(mesh, dimension), requested)
    if np.count_nonzero(mask) != requested.size:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            f"{what} contains undeclared mesh entity IDs.",
        )
    return mask


def _vertex_order(mesh: CellMesh, scope: MeshingScope, what: str, /) -> np.ndarray:
    """Rows of a whole-vertex scope in CellMesh coordinate order."""
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    scope_ids = np.asarray(scope.entity_ids, dtype=np.int64)
    if (
        scope.source_id != mesh.mesh_id
        or scope.source_revision != mesh.numeric_version
        or scope.entity_kind is not MeshingEntityKind.MESH
        or scope.entity_dimension != 0
        or scope.entity_set_id != mesh.entity_set(0).entity_set_id
        or not np.array_equal(scope_ids, np.sort(vertex_ids))
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            f"{what} must bind every vertex of this exact mesh and numeric revision.",
        )
    # MeshingScope sorts IDs; values follow that order, not CellMesh row order.
    return np.searchsorted(scope_ids, vertex_ids)


def _metric_rows(mesh: CellMesh, metric: MeshMetricField, /) -> np.ndarray:
    if not isinstance(metric, MeshMetricField):
        raise TypeError("metric must be MeshMetricField or None.")
    order = _vertex_order(mesh, metric.scope, "Mmg metric")
    values = np.asarray(metric.values, dtype=np.float64)
    if values.shape[1:] != (mesh.ambient_dimension, mesh.ambient_dimension):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "Mmg requires an ambient-coordinate SPD metric at every vertex.",
        )
    return values[order]


def _mmg_metric(rows: np.ndarray, /) -> tuple[str, np.ndarray]:
    """Exactly isotropic metrics become Mmg sizes, others its symmetric storage."""
    dimension = rows.shape[1]
    diagonal = np.diagonal(rows, axis1=1, axis2=2)
    off_diagonal = rows[:, ~np.eye(dimension, dtype=np.bool_)]
    if np.all(off_diagonal == 0.0) and np.all(diagonal == diagonal[:, :1]):
        return "scalar", 1.0 / np.sqrt(diagonal[:, 0])
    # Mmg's in-memory order is the upper triangle by rows (m11 m12 m13 m22 m23
    # m33), unlike the Medit .sol file's lower-triangular order.
    rows_, columns = np.triu_indices(dimension)
    return "tensor", np.ascontiguousarray(rows[:, rows_, columns])


def _vertex_values(mesh: CellMesh, attribute: MeshAttribute, what: str, /) -> np.ndarray:
    if not isinstance(attribute, MeshAttribute):
        raise TypeError(f"{what} must be MeshAttribute.")
    order = _vertex_order(mesh, attribute.scope, what)
    values = np.asarray(attribute.values)
    if values.dtype.kind != "f":
        raise ValueError(f"{what} must hold floating-point values.")
    return np.asarray(values, dtype=np.float64)[order]


def _regions(
    mesh: CellMesh, zones: tuple[MeshZone, ...], /
) -> tuple[np.ndarray, tuple[_Region, ...]]:
    """Canonical region references: sorted (block name, cell-zone name) pairs."""
    dimension = mesh.topological_dimension
    cell_ids = _cell_ids(mesh)
    block_index = np.repeat(
        np.arange(len(mesh.blocks), dtype=np.int64),
        [block.cell_count for block in mesh.blocks],
    )
    cell_zones = tuple(zone for zone in zones if zone.scope.entity_dimension == dimension)
    zone_index = np.full(cell_ids.shape, -1, dtype=np.int64)
    for index, zone in enumerate(cell_zones):
        zone_index[np.isin(cell_ids, np.asarray(zone.scope.entity_ids))] = index
    keys = block_index * (len(cell_zones) + 1) + zone_index + 1
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    members = [
        (
            mesh.blocks[key // (len(cell_zones) + 1)].name,
            None
            if key % (len(cell_zones) + 1) == 0
            else cell_zones[key % (len(cell_zones) + 1) - 1],
        )
        for key in unique.tolist()
    ]
    order = sorted(
        range(len(members)),
        key=lambda item: (
            members[item][0],
            "" if members[item][1] is None else members[item][1].name,
        ),
    )
    references = np.empty(len(members), dtype=np.int64)
    references[order] = np.arange(1, len(members) + 1, dtype=np.int64)
    regions = tuple(
        _Region(
            int(references[item]), members[item][0], members[item][1], int(counts[item])
        )
        for item in order
    )
    return references[inverse], regions


def _boundary_references(
    count: int, moving: tuple[bool, ...], lagrangian: bool, /
) -> list[int]:
    if not lagrangian:
        return list(range(1, count + 1))
    if sum(moving) != 1:
        raise ValueError(
            "The Lagrangian moving boundary must coincide with exactly one class of "
            "boundary patches and zones; Mmg displaces a single boundary reference."
        )
    free = (
        value for value in range(1, count + 2) if value != _DISPLACED_BOUNDARY_REFERENCE
    )
    return [_DISPLACED_BOUNDARY_REFERENCE if flag else next(free) for flag in moving]


def _boundaries(
    source: CellMeshingResult, moving: np.ndarray | None, /
) -> tuple[np.ndarray, tuple[_Boundary, ...]]:
    """Canonical facet references from patch/facet-zone/moving-boundary membership."""
    mesh = source.mesh
    dimension = mesh.topological_dimension - 1
    facet_ids = _entity_ids(mesh, dimension)
    patches = tuple(
        patch for patch in source.patches if patch.scope.entity_dimension == dimension
    )
    zones = tuple(
        zone for zone in source.zones if zone.scope.entity_dimension == dimension
    )
    columns = [np.isin(facet_ids, np.asarray(item.scope.entity_ids)) for item in patches]
    columns += [np.isin(facet_ids, np.asarray(item.scope.entity_ids)) for item in zones]
    if moving is not None:
        columns.append(moving)
    if not columns:
        return np.zeros(facet_ids.shape, dtype=np.int64), ()
    membership = np.column_stack(columns)
    unique, inverse, counts = np.unique(
        membership, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    classes = []
    for row, count in zip(unique, counts, strict=True):
        if not row.any():
            continue
        patch_flags = row[: len(patches)]
        zone_flags = row[len(patches) : len(patches) + len(zones)]
        members = tuple(
            item for item, flag in zip(patches, patch_flags, strict=True) if flag
        )
        zone = next(
            (item for item, flag in zip(zones, zone_flags, strict=True) if flag), None
        )
        flag = bool(row[-1]) if moving is not None else False
        key = (
            tuple(sorted(item.name for item in members)),
            "" if zone is None else zone.name,
            flag,
        )
        classes.append((key, row, members, zone, flag, int(count)))
    classes.sort(key=lambda item: item[0])
    references = _boundary_references(
        len(classes), tuple(item[4] for item in classes), moving is not None
    )
    lookup = np.zeros(len(unique), dtype=np.int64)
    for reference, item in zip(references, classes, strict=True):
        lookup[np.flatnonzero((unique == item[1]).all(axis=1))[0]] = reference
    boundaries = tuple(
        _Boundary(reference, item[2], item[3], item[4], item[5])
        for reference, item in zip(references, classes, strict=True)
    )
    return lookup[inverse], boundaries


def _declared_losses(
    source: CellMeshingResult,
    fields: tuple[MeshAttribute, ...],
    transferred: bool,
    /,
) -> tuple[AdapterLoss, ...]:
    mesh = source.mesh
    kept = {mesh.topological_dimension, mesh.topological_dimension - 1}
    losses = [
        AdapterLoss(
            "entity_global_ids",
            "import",
            "synthesized",
            "Generated output IDs are not source entity identities; lineage is unknown.",
            changes_interpretation=False,
        )
    ]
    dropped = [
        *(
            f"zone {zone.name!r}"
            for zone in source.zones
            if zone.scope.entity_dimension not in kept
        ),
        *(
            f"patch {patch.name!r}"
            for patch in source.patches
            if patch.scope.entity_dimension != mesh.topological_dimension - 1
        ),
        *(f"label {label.name!r}" for label in source.labels),
        *(
            f"attribute {attribute.name!r}"
            for attribute in source.attributes
            if attribute.attribute_id not in {item.attribute_id for item in fields}
        ),
    ]
    if dropped:
        losses.append(
            AdapterLoss(
                "organization",
                "import",
                "dropped",
                "Mmg carries only cell regions and facet references; not transferred: "
                + ", ".join(sorted(dropped))
                + ".",
                changes_interpretation=True,
            )
        )
    if (source.associations and not transferred) or source.boundary is not None:
        losses.append(
            AdapterLoss(
                "geometry_associations",
                "import",
                "dropped",
                "Boundary models, and geometry associations without an association "
                "transfer, are not re-associated to the adapted mesh.",
                changes_interpretation=True,
            )
        )
    return tuple(losses)


def _require_affine(source: CellMeshingResult, /) -> None:
    affine = CellGeometrySpec.affine(source.mesh)
    if (
        affine.geometry_layout_id != source.geometry.geometry_layout_id
        or not np.array_equal(
            np.asarray(source.geometry.coordinates), np.asarray(source.mesh.coordinates)
        )
    ):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Mmg adapts affine simplices; curved (high-order) geometry would be discarded.",
        )


def _required_masks(
    mesh: CellMesh,
    backend: str,
    required: tuple[MeshingScope, ...],
    ridges: MeshingScope | None,
    /,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Required-entity membership per dimension and the ridge-edge membership."""
    masks = [
        np.zeros(_entity_ids(mesh, dimension).shape, dtype=np.bool_)
        for dimension in range(mesh.topological_dimension + 1)
    ]
    for scope in required:
        masks[scope.entity_dimension] |= _scope_mask(mesh, scope, "Required entity scope")
    if ridges is None:
        return masks, np.zeros(masks[1].shape, dtype=np.bool_)
    if backend == "mmg2d":
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Planar meshes (mmg2d) have no ridge edges.",
        )
    if ridges.entity_dimension != 1:
        raise ValueError("ridges must select edges.")
    return masks, _scope_mask(mesh, ridges, "Ridge scope")


def _explicit_facets(mesh: CellMesh, cell_references: np.ndarray, /) -> np.ndarray:
    """Boundary and region-interface facets.

    Mmg gives facets it creates itself the reference of an adjacent cell, which
    would alias a boundary reference; declaring them with reference zero keeps
    every facet reference exactly the encoded one.
    """
    connectivity = mesh.connectivity
    if isinstance(connectivity, TetrahedralConnectivity):
        incidence = np.asarray(connectivity.cell_faces, dtype=np.int64)
        counts = np.asarray(connectivity.face_cell_counts)
    elif isinstance(connectivity, PolyhedralConnectivity):
        incidence = np.asarray(connectivity.cell_face_values, dtype=np.int64).reshape(
            -1, 4
        )
        counts = np.asarray(connectivity.face_cell_counts)
        cell_ids = _cell_ids(mesh)
        order = np.argsort(cell_ids)
        rows = order[
            np.searchsorted(
                cell_ids[order], np.asarray(connectivity.cell_global_ids, dtype=np.int64)
            )
        ]
        cell_references = cell_references[rows]
    else:
        assert isinstance(connectivity, PolygonalConnectivity)
        incidence = np.asarray(connectivity.cell_edges, dtype=np.int64)
        counts = np.asarray(connectivity.edge_cell_counts)
    references = np.repeat(cell_references, incidence.shape[1])
    lowest = np.full(counts.shape, np.iinfo(np.int64).max, dtype=np.int64)
    highest = np.zeros(counts.shape, dtype=np.int64)
    np.minimum.at(lowest, incidence.reshape(-1), references)
    np.maximum.at(highest, incidence.reshape(-1), references)
    return (counts == 1) | (lowest != highest)


def _geometry_arrays(
    source: CellMeshingResult,
    backend: str,
    cell_references: np.ndarray,
    facet_references: np.ndarray,
    masks: list[np.ndarray],
    ridges: np.ndarray,
    /,
) -> dict[str, np.ndarray]:
    mesh = source.mesh
    dimension = mesh.topological_dimension
    arrays = {
        "vertices": np.ascontiguousarray(mesh.coordinates, dtype=np.float64),
        "vertex_required": masks[0].astype(np.uint8),
        "cells": np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
        ),
        "cell_references": cell_references,
        "cell_required": masks[dimension].astype(np.uint8),
    }
    edges = _entity_vertices(mesh, 1)
    explicit = _explicit_facets(mesh, cell_references)
    if backend == "mmg3d":
        faces = _entity_vertices(mesh, 2)
        include = (facet_references != 0) | masks[2] | explicit
        arrays.update(
            triangles=faces[include],
            triangle_references=facet_references[include],
            triangle_required=masks[2][include].astype(np.uint8),
        )
        edge_references = np.zeros(edges.shape[0], dtype=np.int64)
        explicit_edges = np.zeros(edges.shape[0], dtype=np.bool_)
    else:
        arrays.update(
            triangles=np.zeros((0, 3), dtype=np.int64),
            triangle_references=np.zeros(0, dtype=np.int64),
            triangle_required=np.zeros(0, dtype=np.uint8),
        )
        edge_references = facet_references
        explicit_edges = explicit
    include = (edge_references != 0) | masks[1] | ridges | explicit_edges
    arrays.update(
        edges=edges[include],
        edge_references=edge_references[include],
        edge_required=masks[1][include].astype(np.uint8),
        edge_ridges=ridges[include].astype(np.uint8),
    )
    return arrays


def _level_set_parameters(
    level_set: MmgLevelSet,
    regions: tuple[_Region, ...],
    boundaries: tuple[_Boundary, ...],
    /,
) -> tuple[dict[str, Any], int]:
    names = {region.block for region in regions} | {
        region.zone.name for region in regions if region.zone is not None
    }
    unknown = sorted(set(level_set.unsplit_regions) - names)
    if unknown:
        raise ValueError(f"unsplit_regions names unknown regions: {', '.join(unknown)}.")
    count = len(regions)
    materials = [
        [
            region.reference,
            region.block not in level_set.unsplit_regions
            and (
                region.zone is None or region.zone.name not in level_set.unsplit_regions
            ),
            count + region.reference,
            2 * count + region.reference,
        ]
        for region in regions
    ]
    interface = max((item.reference for item in boundaries), default=0) + 1
    return (
        {
            "isovalue": float(level_set.isovalue),
            "interface_reference": interface,
            "materials": materials,
        },
        interface,
    )


def _metric_encoding(
    plan: MmgAdaptationPlan, arrays: dict[str, np.ndarray], parameters: dict[str, Any], /
) -> tuple[str, float]:
    """Metric representation and the gradation Mmg applies (hgrad of the options)."""
    options = plan.options
    gradation = (
        _MMG_DEFAULT_GRADATION if options.gradation is None else float(options.gradation)
    )
    if plan.metric is None:
        return "none", gradation
    if options.minimum_size is not None or options.maximum_size is not None:
        raise ValueError("A metric owns Mmg's size bounds; leave the option sizes None.")
    representation, arrays["metric"] = _mmg_metric(
        _metric_rows(plan.source.mesh, plan.metric)
    )
    # Mmg requires hmin < hmax even for a constant metric: round outward.
    parameters.update(
        minimum_size=float(np.nextafter(plan.metric.minimum_size, 0.0)),
        maximum_size=float(np.nextafter(plan.metric.maximum_size, np.inf)),
    )
    return representation, gradation


def _motion_encoding(
    plan: MmgAdaptationPlan,
    backend: str,
    required_vertices: np.ndarray,
    arrays: dict[str, np.ndarray],
    parameters: dict[str, Any],
    /,
) -> np.ndarray | None:
    """Displacement arrays and the moving-facet membership of a Lagrangian request."""
    motion = plan.motion
    if motion is None:
        return None
    mesh = plan.source.mesh
    if backend == "mmgs":
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Mmg has no Lagrangian motion for surface meshes (mmgs).",
        )
    if required_vertices.any():
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Required vertices cannot be combined with Lagrangian motion.",
        )
    if motion.moving_boundary.entity_dimension != mesh.topological_dimension - 1:
        raise ValueError("The moving boundary must select facets.")
    moving = _scope_mask(mesh, motion.moving_boundary, "Moving boundary")
    displacement = _vertex_values(mesh, motion.displacement, "Displacement")
    if displacement.shape[1:] != (mesh.ambient_dimension,):
        raise ValueError("Displacement must hold one ambient vector per vertex.")
    arrays["displacement"] = np.ascontiguousarray(displacement)
    parameters["lagrangian_mode"] = _lagrangian_code(motion.mode)
    return moving


def _level_set_encoding(
    plan: MmgAdaptationPlan,
    regions: tuple[_Region, ...],
    boundaries: tuple[_Boundary, ...],
    arrays: dict[str, np.ndarray],
    parameters: dict[str, Any],
    /,
) -> int:
    """Level-set values and materials; returns the interface reference."""
    level_set = plan.level_set
    if level_set is None:
        return 0
    source = plan.source
    values = _vertex_values(source.mesh, level_set.values, "Level-set values")
    if values.ndim != 1:
        raise ValueError("Level-set values must be scalar.")
    taken = {patch.name for patch in source.patches} | {
        zone.name for zone in source.zones
    }
    if level_set.interface in taken:
        raise ValueError("The level-set interface name is already a zone or patch name.")
    arrays["level_set"] = values
    level_parameters, interface = _level_set_parameters(level_set, regions, boundaries)
    parameters.update(level_parameters)
    return interface


def _field_encoding(
    plan: MmgAdaptationPlan, arrays: dict[str, np.ndarray], /
) -> tuple[tuple[MeshAttribute, int, int], ...]:
    names = [attribute.name for attribute in plan.fields]
    if len(set(names)) != len(names) or _PARTITION_ATTRIBUTE in names:
        raise ValueError("Transferred field names must be unique and not reserved.")
    layout = []
    columns = []
    start = 0
    for attribute in plan.fields:
        values = _vertex_values(plan.source.mesh, attribute, f"Field {attribute.name!r}")
        flat = values.reshape(values.shape[0], -1)
        layout.append((attribute, start, flat.shape[1]))
        columns.append(flat)
        start += flat.shape[1]
    if columns:
        arrays["fields"] = np.ascontiguousarray(np.concatenate(columns, axis=1))
    return tuple(layout)


def _encode(plan: MmgAdaptationPlan, /) -> _Encoding:
    source = plan.source
    mesh = source.mesh
    backend = _backend(mesh)
    _require_affine(source)
    if plan.level_set is not None and plan.motion is not None:
        raise ValueError("Mmg runs either level-set discretization or Lagrangian motion.")
    masks, ridges = _required_masks(mesh, backend, plan.required, plan.ridges)
    parameters = _option_parameters(plan.options)
    arrays: dict[str, np.ndarray] = {}
    representation, gradation = _metric_encoding(plan, arrays, parameters)
    moving = _motion_encoding(plan, backend, masks[0], arrays, parameters)
    cell_references, regions = _regions(mesh, source.zones)
    facet_references, boundaries = _boundaries(source, moving)
    interface = _level_set_encoding(plan, regions, boundaries, arrays, parameters)
    program = (
        "lagrangian"
        if plan.motion is not None
        else "levelset"
        if plan.level_set is not None
        else "remesh"
    )
    parameters["program"] = program
    arrays.update(
        _geometry_arrays(
            source, backend, cell_references, facet_references, masks, ridges
        )
    )
    return _Encoding(
        backend,
        program,
        arrays,
        parameters,
        regions,
        boundaries,
        interface,
        representation,
        gradation,
        _field_encoding(plan, arrays),
        int(np.count_nonzero(masks[0])),
        _declared_losses(
            source,
            plan.fields,
            plan.association_transfer is not None and bool(source.associations),
        ),
    )


@dataclass(frozen=True, slots=True)
class MmgAdaptationPlan:
    """Validated Mmg request bound to one exact source result revision."""

    source: CellMeshingResult
    options: MmgOptions
    limits: MeshingLimits
    audit_policy: CellMeshAuditPolicy
    metric: MeshMetricField | None = None
    level_set: MmgLevelSet | None = None
    motion: MmgLagrangianMotion | None = None
    required: tuple[MeshingScope, ...] = ()
    ridges: MeshingScope | None = None
    fields: tuple[MeshAttribute, ...] = ()
    association_transfer: BRepAssociationTransfer | None = None
    plan_id: str = field(init=False)
    _encoding: _Encoding = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        if not isinstance(self.options, MmgOptions):
            raise TypeError("options must be MmgOptions.")
        if not isinstance(self.limits, MeshingLimits):
            raise TypeError("limits must be MeshingLimits.")
        if not isinstance(self.audit_policy, CellMeshAuditPolicy):
            raise TypeError("audit_policy must be CellMeshAuditPolicy.")
        if self.metric is not None and not isinstance(self.metric, MeshMetricField):
            raise TypeError("metric must be MeshMetricField or None.")
        if self.level_set is not None and not isinstance(self.level_set, MmgLevelSet):
            raise TypeError("level_set must be MmgLevelSet or None.")
        if self.motion is not None and not isinstance(self.motion, MmgLagrangianMotion):
            raise TypeError("motion must be MmgLagrangianMotion or None.")
        if not isinstance(self.required, tuple) or not all(
            isinstance(scope, MeshingScope) for scope in self.required
        ):
            raise TypeError("required must be a tuple of MeshingScope values.")
        if not isinstance(self.fields, tuple) or not all(
            isinstance(attribute, MeshAttribute) for attribute in self.fields
        ):
            raise TypeError("fields must be a tuple of MeshAttribute values.")
        if self.association_transfer is not None:
            if not isinstance(self.association_transfer, BRepAssociationTransfer):
                raise TypeError(
                    "association_transfer must be BRepAssociationTransfer or None."
                )
            if self.source.associations:
                self.association_transfer.source_associations(self.source)
        encoding = _encode(self)
        object.__setattr__(self, "_encoding", encoding)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "mmg-adaptation-plan",
                    "source": self.source.result_id,
                    "metric": None if self.metric is None else self.metric.metric_id,
                    "level_set": (
                        None
                        if self.level_set is None
                        else {
                            "values": self.level_set.values.attribute_id,
                            "isovalue": float(self.level_set.isovalue),
                            "names": [
                                self.level_set.interface,
                                self.level_set.interior,
                                self.level_set.exterior,
                            ],
                            "unsplit": sorted(self.level_set.unsplit_regions),
                        }
                    ),
                    "motion": (
                        None
                        if self.motion is None
                        else {
                            "displacement": self.motion.displacement.attribute_id,
                            "moving_boundary": self.motion.moving_boundary.scope_id,
                            "mode": self.motion.mode.value,
                        }
                    ),
                    "required": sorted(scope.scope_id for scope in self.required),
                    "ridges": None if self.ridges is None else self.ridges.scope_id,
                    "fields": [attribute.attribute_id for attribute in self.fields],
                    "association_transfer": None
                    if self.association_transfer is None
                    else self.association_transfer.transfer_id,
                    "parameters": encoding.parameters,
                    "limits": self.limits.limits_id,
                    "audit_policy": self.audit_policy.policy_id,
                }
            ),
        )


@final
class MmgFieldTransfer(StrictModule, NonTrainableState):
    """P1 transfer of declared vertex fields onto the adapted vertices.

    ``located_count`` output vertices lie in a source simplex (within
    ``tolerance``) and use its exact barycentric weights; ``projected_count``
    lie outside the source (boundary approximation) and use the clamped weights
    of their closest source point, at most ``maximum_projection_distance`` away.
    ``source_configuration`` is ``"source"`` or, for Lagrangian motion, the
    source displaced by Mmg's own motion.
    """

    method: str = eqx.field(static=True)
    source_configuration: str = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    located_count: int = eqx.field(static=True)
    projected_count: int = eqx.field(static=True)
    maximum_projection_distance: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: str,
        source_configuration: str,
        field_names: tuple[str, ...],
        /,
        *,
        located_count: int,
        projected_count: int,
        maximum_projection_distance: float,
        tolerance: float,
    ):
        method_ = _name(method, "method")
        configuration = _name(source_configuration, "source_configuration")
        names = tuple(_name(value, "field name") for value in field_names)
        located, projected = int(located_count), int(projected_count)
        distance, tolerance_ = float(maximum_projection_distance), float(tolerance)
        if located < 0 or projected < 0:
            raise ValueError("Transfer counts must be non-negative.")
        if not (np.isfinite(distance) and distance >= 0.0) or not (
            np.isfinite(tolerance_) and tolerance_ >= 0.0
        ):
            raise ValueError("Transfer distances must be finite and non-negative.")
        self.method = method_
        self.source_configuration = configuration
        self.field_names = names
        self.located_count = located
        self.projected_count = projected
        self.maximum_projection_distance = distance
        self.tolerance = tolerance_
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "mmg-field-transfer",
                "method": method_,
                "source_configuration": configuration,
                "field_names": names,
                "located": located,
                "projected": projected,
                "maximum_projection_distance": distance,
                "tolerance": tolerance_,
            }
        )


@final
class MmgReference(StrictModule, NonTrainableState):
    """One Mmg reference: the source organization it encodes and its entity counts.

    ``kind`` is ``region`` (cells of a block/cell-zone pair), ``boundary``
    (facets of one patch/facet-zone class), ``interior``/``exterior`` (level-set
    sides of a region), or ``interface`` (the discretized level set).
    """

    kind: str = eqx.field(static=True)
    reference: int = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    source_count: int = eqx.field(static=True)
    target_count: int = eqx.field(static=True)

    def __init__(
        self,
        kind: str,
        reference: int,
        names: tuple[str, ...],
        /,
        *,
        source_count: int,
        target_count: int,
    ):
        if kind not in ("region", "boundary", "interior", "exterior", "interface"):
            raise ValueError(f"Unknown Mmg reference kind {kind!r}.")
        if int(reference) <= 0 or int(source_count) < 0 or int(target_count) < 0:
            raise ValueError("Mmg references and counts must be positive/non-negative.")
        self.kind = kind
        self.reference = int(reference)
        self.names = tuple(str(value) for value in names)
        self.source_count = int(source_count)
        self.target_count = int(target_count)


@final
class MmgReferenceRetention(StrictModule, NonTrainableState):
    """Evidence that every encoded reference and required vertex survived."""

    references: tuple[MmgReference, ...]
    required_vertices: int = eqx.field(static=True)
    retained_required_vertices: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        references: tuple[MmgReference, ...],
        /,
        *,
        required_vertices: int,
        retained_required_vertices: int,
    ):
        if not all(isinstance(item, MmgReference) for item in references):
            raise TypeError("references must contain MmgReference values.")
        if not 0 <= int(retained_required_vertices) <= int(required_vertices):
            raise ValueError(
                "Retained required vertices must not exceed the required count."
            )
        self.references = tuple(references)
        self.required_vertices = int(required_vertices)
        self.retained_required_vertices = int(retained_required_vertices)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "mmg-reference-retention",
                "references": [
                    [
                        item.kind,
                        item.reference,
                        list(item.names),
                        item.source_count,
                        item.target_count,
                    ]
                    for item in self.references
                ],
                "required_vertices": self.required_vertices,
                "retained_required_vertices": self.retained_required_vertices,
            }
        )


@final
class MmgSessionEvidence(StrictModule, NonTrainableState):
    """The persistent worker session and call that produced one adaptation."""

    session_id: str = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)
    sequence: int = eqx.field(static=True)
    ranks: int = eqx.field(static=True)
    peak_rss_bytes: int = eqx.field(static=True)
    elapsed_seconds: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        session_id: str,
        identity_id: str,
        /,
        *,
        sequence: int,
        ranks: int,
        peak_rss_bytes: int,
        elapsed_seconds: float,
    ):
        session = _name(session_id, "session_id")
        identity = _name(identity_id, "identity_id")
        if int(sequence) <= 0 or int(ranks) <= 0 or int(peak_rss_bytes) < 0:
            raise ValueError(
                "Worker sequence, ranks, and peak memory must be valid counts."
            )
        self.session_id = session
        self.identity_id = identity
        self.sequence = int(sequence)
        self.ranks = int(ranks)
        self.peak_rss_bytes = int(peak_rss_bytes)
        self.elapsed_seconds = float(elapsed_seconds)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "mmg-session-evidence",
                "session_id": session,
                "identity_id": identity,
                "sequence": self.sequence,
                "ranks": self.ranks,
                "peak_rss_bytes": self.peak_rss_bytes,
            }
        )


@final
class MmgAdaptationResult(StrictModule, NonTrainableState):
    """Certified adapted mesh with its metric, field transfer, and evidence.

    ``metric_representation`` records how the source metric reached Mmg
    (``scalar`` for exactly isotropic metrics, ``tensor`` otherwise, ``none``
    without a metric); ``metric`` is Mmg's adapted metric on the target
    vertices when Mmg returns one. Transferred fields are attributes of
    ``mesh`` with their source names.
    """

    plan_id: str = eqx.field(static=True)
    mesh: CellMeshingResult
    metric: MeshMetricField | None
    metric_representation: str = eqx.field(static=True)
    fields: MmgFieldTransfer | None
    references: MmgReferenceRetention
    session: MmgSessionEvidence
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan_id: str,
        mesh: CellMeshingResult,
        metric: MeshMetricField | None,
        metric_representation: str,
        fields: MmgFieldTransfer | None,
        references: MmgReferenceRetention,
        session: MmgSessionEvidence,
        /,
    ):
        plan = _name(plan_id, "plan_id")
        if not isinstance(mesh, CellMeshingResult):
            raise TypeError("mesh must be CellMeshingResult.")
        if metric is not None and (
            not isinstance(metric, MeshMetricField)
            or metric.scope.source_id != mesh.mesh.mesh_id
            or metric.scope.source_revision != mesh.mesh.numeric_version
        ):
            raise ValueError("The adapted metric must be bound to the adapted mesh.")
        if metric_representation not in ("none", "scalar", "tensor"):
            raise ValueError("metric_representation must be none, scalar, or tensor.")
        if fields is not None and not isinstance(fields, MmgFieldTransfer):
            raise TypeError("fields must be MmgFieldTransfer or None.")
        if not isinstance(references, MmgReferenceRetention):
            raise TypeError("references must be MmgReferenceRetention.")
        if not isinstance(session, MmgSessionEvidence):
            raise TypeError("session must be MmgSessionEvidence.")
        self.plan_id = plan
        self.mesh = mesh
        self.metric = metric
        self.metric_representation = metric_representation
        self.fields = fields
        self.references = references
        self.session = session
        self.result_id = canonical_fingerprint(
            {
                "kind": "mmg-adaptation-result",
                "plan": plan,
                "mesh": mesh.result_id,
                "metric": None if metric is None else metric.metric_id,
                "metric_representation": metric_representation,
                "fields": None if fields is None else fields.transfer_id,
                "references": references.evidence_id,
                "session": session.evidence_id,
            }
        )


class _Output(NamedTuple):
    vertices: np.ndarray
    cells: np.ndarray
    cell_references: np.ndarray
    facets: np.ndarray
    facet_references: np.ndarray
    metric: np.ndarray | None
    fields: np.ndarray | None
    required_targets: np.ndarray
    partition: np.ndarray | None


def _rank(name: str, /) -> int:
    prefix, _, number = name.partition("-")
    if prefix != "rank" or not number.isdigit():
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            f"Unexpected collective worker part {name!r}.",
        )
    return int(number)


def _stitch(parts: Mapping[str, Mapping[str, np.ndarray]], /) -> _Output:
    """Assemble ParMmg's distributed parts through its global vertex numbering."""
    ordered = sorted(parts.items(), key=lambda item: _rank(item[0]))
    total = max(int(np.max(part["global_vertex_ids"], initial=0)) for _, part in ordered)
    dimension = ordered[0][1]["vertices"].shape[1]
    vertices = np.full((total, dimension), np.nan, dtype=np.float64)
    owned = np.zeros(total, dtype=np.int64)
    metric = fields = None
    cells, cell_references, partition, facets, facet_references = [], [], [], [], []
    required = []
    for name, part in ordered:
        rank = _rank(name)
        globals_ = np.asarray(part["global_vertex_ids"], dtype=np.int64) - 1
        own = np.asarray(part["vertex_owners"]) == rank
        vertices[globals_[own]] = part["vertices"][own]
        owned[globals_[own]] += 1
        if "metric" in part:
            values = np.asarray(part["metric"], dtype=np.float64)
            if metric is None:
                metric = np.full((total, *values.shape[1:]), np.nan, dtype=np.float64)
            metric[globals_[own]] = values[own]
        if "fields" in part:
            values = np.asarray(part["fields"], dtype=np.float64)
            if fields is None:
                fields = np.full((total, values.shape[1]), np.nan, dtype=np.float64)
            fields[globals_[own]] = values[own]
        cells.append(globals_[np.asarray(part["cells"], dtype=np.int64)])
        cell_references.append(np.asarray(part["cell_references"], dtype=np.int64))
        partition.append(np.full(part["cells"].shape[0], rank, dtype=np.int32))
        references = np.asarray(part["facet_references"], dtype=np.int64)
        # Parallel interface faces carry reference zero; boundary faces are owned once.
        keep = references != 0
        facets.append(globals_[np.asarray(part["facets"], dtype=np.int64)[keep]])
        facet_references.append(references[keep])
        targets = np.asarray(part["required_vertex_targets"], dtype=np.int64)
        required.append(np.where(targets >= 0, globals_[np.maximum(targets, 0)], -1))
    if np.any(owned != 1):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "ParMmg parts do not own every global vertex exactly once.",
        )
    return _Output(
        vertices,
        np.concatenate(cells),
        np.concatenate(cell_references),
        np.concatenate(facets),
        np.concatenate(facet_references),
        metric,
        fields,
        np.max(np.stack(required), axis=0),
        np.concatenate(partition),
    )


def _collect(call: NativeWorkerCall, /) -> _Output:
    if call.parts:
        return _stitch(call.parts)
    arrays = call.arrays
    return _Output(
        np.asarray(arrays["vertices"], dtype=np.float64),
        np.asarray(arrays["cells"], dtype=np.int64),
        np.asarray(arrays["cell_references"], dtype=np.int64),
        np.asarray(arrays["facets"], dtype=np.int64),
        np.asarray(arrays["facet_references"], dtype=np.int64),
        None
        if "metric" not in arrays
        else np.asarray(arrays["metric"], dtype=np.float64),
        None
        if "fields" not in arrays
        else np.asarray(arrays["fields"], dtype=np.float64),
        np.asarray(arrays["required_vertex_targets"], dtype=np.int64),
        None,
    )


def _compact(output: _Output, backend: str, limits: MeshingLimits, /) -> _Output:
    """Drop vertices no cell uses and orient tetrahedra positively."""
    _check_arrays(output.vertices, output.cells, limits)
    used, remapped = np.unique(output.cells, return_inverse=True)
    cells = remapped.reshape(output.cells.shape)
    vertices = output.vertices[used]
    renumber = np.full(output.vertices.shape[0], -1, dtype=np.int64)
    renumber[used] = np.arange(used.size, dtype=np.int64)
    facets = renumber[output.facets]
    if np.any(facets < 0):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Mmg returned boundary facets on vertices no cell uses.",
        )
    if backend == "mmg3d":
        determinants = np.linalg.det(vertices[cells[:, 1:]] - vertices[cells[:, :1]])
        negative = determinants < 0
        cells[negative, :2] = cells[negative, 1::-1]
    targets = output.required_targets
    return _Output(
        vertices,
        cells,
        output.cell_references,
        facets,
        output.facet_references,
        None if output.metric is None else output.metric[used],
        None if output.fields is None else output.fields[used],
        np.where(targets >= 0, renumber[np.maximum(targets, 0)], -1),
        output.partition,
    )


def _row_positions(table: np.ndarray, queries: np.ndarray, /) -> np.ndarray:
    """Row index of each query in ``table`` (unordered vertex sets), or -1."""
    if queries.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    rows = np.concatenate([np.sort(table, axis=1), np.sort(queries, axis=1)])
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    where = np.full(inverse.max() + 1, -1, dtype=np.int64)
    where[inverse[: table.shape[0]]] = np.arange(table.shape[0], dtype=np.int64)
    return where[inverse[table.shape[0] :]]


def _entity_scope(
    mesh: CellMesh, dimension: int, identifiers: np.ndarray, /
) -> MeshingScope:
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        mesh.entity_set(dimension).entity_set_id,
        identifiers,
    )


def _lost(what: str, /) -> MeshingFailure:
    return MeshingFailure(
        MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
        f"Mmg did not retain {what}.",
    )


def _decode_regions(
    encoding: _Encoding, references: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Region index and level-set side (0 none, 1 interior, 2 exterior) per cell."""
    count = len(encoding.regions)
    code = np.full(3 * count + 1, -1, dtype=np.int64)
    for index, region in enumerate(encoding.regions):
        code[region.reference] = 3 * index
        if encoding.program == "levelset":
            code[count + region.reference] = 3 * index + 1
            code[2 * count + region.reference] = 3 * index + 2
    valid = (references > 0) & (references < code.size)
    decoded = np.where(valid, code[np.where(valid, references, 0)], -1)
    if np.any(decoded < 0):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Mmg returned cells with an unencoded region reference.",
        )
    return decoded // 3, decoded % 3


def _build_mesh(
    plan: MmgAdaptationPlan, output: _Output, region: np.ndarray, /
) -> tuple[CellMesh, np.ndarray]:
    """Blocks by name (canonical order, ascending fresh IDs) and the cell permutation."""
    encoding = plan._encoding
    source = plan.source.mesh
    kind = "tetrahedron" if encoding.backend == "mmg3d" else "triangle"
    block_of_region = np.array(
        [item.block for item in encoding.regions], dtype=np.object_
    )
    names = sorted({block.name for block in source.blocks})
    order = np.concatenate(
        [np.flatnonzero(block_of_region[region] == name) for name in names]
    )
    cell_ids = _fresh_ids(_cell_ids(source), order.size)
    blocks = []
    start = 0
    for name in names:
        count = int(np.count_nonzero(block_of_region[region] == name))
        if count == 0:
            raise _lost(f"cell block {name!r}")
        rows = order[start : start + count]
        blocks.append(
            CellBlock(
                name,
                kind,
                output.cells[rows],
                global_ids=cell_ids[start : start + count],
            )
        )
        start += count
    mesh = CellMesh(
        output.vertices,
        tuple(blocks),
        vertex_global_ids=_fresh_ids(
            np.asarray(source.vertex_global_ids), output.vertices.shape[0]
        ),
        numeric_version=plan.plan_id,
    )
    return mesh, order


def _cell_organization(
    plan: MmgAdaptationPlan,
    mesh: CellMesh,
    region: np.ndarray,
    side: np.ndarray,
    /,
) -> tuple[list[MeshZone], list[MeshLabel], list[MmgReference], dict[str, str]]:
    encoding = plan._encoding
    dimension = mesh.topological_dimension
    cell_ids = _cell_ids(mesh)
    zones, labels, references = [], [], []
    renamed: dict[str, str] = {}
    source_zones = [
        zone for zone in plan.source.zones if zone.scope.entity_dimension == dimension
    ]
    zone_of_region = np.array(
        [None if item.zone is None else item.zone.name for item in encoding.regions],
        dtype=np.object_,
    )
    for zone in source_zones:
        members = cell_ids[zone_of_region[region] == zone.name]
        if members.size == 0:
            raise _lost(f"cell zone {zone.name!r}")
        target = MeshZone(
            zone.name,
            zone.role,
            _entity_scope(mesh, dimension, members),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        renamed[zone.zone_id] = target.zone_id
        zones.append(target)
    count = len(encoding.regions)
    for index, item in enumerate(encoding.regions):
        names = (item.block, *(() if item.zone is None else (item.zone.name,)))
        in_region = region == index
        target_count = int(np.count_nonzero(in_region))
        if target_count == 0:
            raise _lost(f"region reference {item.reference} ({', '.join(names)})")
        references.append(
            MmgReference(
                "region",
                item.reference,
                names,
                source_count=item.source_count,
                target_count=target_count,
            )
        )
        if encoding.program == "levelset":
            for code, kind, offset in (
                (1, "interior", count),
                (2, "exterior", 2 * count),
            ):
                references.append(
                    MmgReference(
                        kind,
                        offset + item.reference,
                        names,
                        source_count=0,
                        target_count=int(np.count_nonzero(in_region & (side == code))),
                    )
                )
    if encoding.program == "levelset":
        level_set = plan.level_set
        assert level_set is not None
        for code, name in ((1, level_set.interior), (2, level_set.exterior)):
            members = cell_ids[side == code]
            if members.size:
                labels.append(MeshLabel(name, _entity_scope(mesh, dimension, members)))
    return zones, labels, references, renamed


def _facet_organization(
    plan: MmgAdaptationPlan,
    mesh: CellMesh,
    output: _Output,
    renamed: dict[str, str],
    /,
) -> tuple[list[MeshZone], list[MeshPatch], list[MmgReference]]:
    encoding = plan._encoding
    dimension = mesh.topological_dimension - 1
    referenced = output.facet_references != 0
    rows = _row_positions(_entity_vertices(mesh, dimension), output.facets[referenced])
    if np.any(rows < 0):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Mmg returned boundary facets that are not faces of its cells.",
        )
    facet_ids = _entity_ids(mesh, dimension)[rows]
    facet_references = output.facet_references[referenced]
    known = {item.reference for item in encoding.boundaries} | (
        {encoding.interface_reference} if encoding.program == "levelset" else set()
    )
    if not set(np.unique(facet_references).tolist()) <= known:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Mmg returned facets with an unencoded boundary reference.",
        )
    members: dict[str, list[np.ndarray]] = {}
    zone_members: dict[str, list[np.ndarray]] = {}
    references = []
    for item in encoding.boundaries:
        selected = np.unique(facet_ids[facet_references == item.reference])
        names = tuple(sorted(patch.name for patch in item.patches)) + (
            () if item.zone is None else (item.zone.name,)
        )
        if selected.size == 0:
            raise _lost(f"boundary reference {item.reference} ({', '.join(names)})")
        references.append(
            MmgReference(
                "boundary",
                item.reference,
                names,
                source_count=item.source_count,
                target_count=selected.size,
            )
        )
        for patch in item.patches:
            members.setdefault(patch.name, []).append(selected)
        if item.zone is not None:
            zone_members.setdefault(item.zone.name, []).append(selected)
    patches = []
    for patch in plan.source.patches:
        if patch.name not in members:
            continue
        unknown = set(patch.adjacent_zone_ids) - set(renamed)
        if unknown:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                f"Patch {patch.name!r} is adjacent to zones not carried by Mmg.",
            )
        patches.append(
            MeshPatch(
                patch.name,
                _entity_scope(mesh, dimension, np.concatenate(members[patch.name])),
                connected=patch.connected,
                adjacent_zone_ids=tuple(
                    renamed[value] for value in patch.adjacent_zone_ids
                ),
            )
        )
    zones = [
        MeshZone(
            zone.name,
            zone.role,
            _entity_scope(mesh, dimension, np.concatenate(zone_members[zone.name])),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        for zone in plan.source.zones
        if zone.name in zone_members
    ]
    if encoding.program == "levelset":
        level_set = plan.level_set
        assert level_set is not None
        interface = np.unique(facet_ids[facet_references == encoding.interface_reference])
        references.append(
            MmgReference(
                "interface",
                encoding.interface_reference,
                (level_set.interface,),
                source_count=0,
                target_count=interface.size,
            )
        )
        if interface.size:
            patches.append(
                MeshPatch(level_set.interface, _entity_scope(mesh, dimension, interface))
            )
    return zones, patches, references


def _output_attributes(
    plan: MmgAdaptationPlan, mesh: CellMesh, output: _Output, order: np.ndarray, /
) -> tuple[MeshAttribute, ...]:
    """Transferred fields by source name, plus the ParMmg partition of the cells."""
    attributes = []
    if output.fields is not None:
        scope = MmgProvider.vertex_scope(mesh)
        count = output.vertices.shape[0]
        attributes = [
            MeshAttribute(
                attribute.name,
                attribute.role,
                scope,
                output.fields[:, start : start + width].reshape(
                    count, *attribute.component_shape
                ),
                unit=attribute.unit,
            )
            for attribute, start, width in plan._encoding.field_layout
        ]
    if output.partition is not None:
        attributes.append(
            MeshAttribute(
                _PARTITION_ATTRIBUTE,
                MeshAttributeRole.PARTITION,
                _entity_scope(mesh, mesh.topological_dimension, _cell_ids(mesh)),
                output.partition[order],
            )
        )
    return tuple(attributes)


def _field_transfer(
    encoding: _Encoding, transfer: Mapping[str, Any] | None, /
) -> MmgFieldTransfer | None:
    if transfer is None:
        return None
    return MmgFieldTransfer(
        transfer["method"],
        transfer["source_configuration"],
        tuple(attribute.name for attribute, _, _ in encoding.field_layout),
        located_count=transfer["located"],
        projected_count=transfer["projected"],
        maximum_projection_distance=transfer["maximum_projection_distance"],
        tolerance=transfer["tolerance"],
    )


def _adapter_report(
    encoding: _Encoding, source: CellMesh, mesh: CellMesh, /
) -> AdapterReport:
    return AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        encoding.backend,
        "phydrax-cell-mesh",
        source_id=source.mesh_id,
        target_id=mesh.mesh_id,
        coordinate_mapping=("identity",),
        preserved_fields=tuple(
            attribute.name for attribute, _, _ in encoding.field_layout
        ),
        assumptions=(
            "Blocks, cell zones, facet patches, and facet zones are rebuilt by "
            "name from Mmg references; entity correspondence is not supplied.",
            "Declared vertex fields are P1-interpolated in the source "
            "(Lagrangian: displaced source) configuration.",
        ),
        losses=encoding.losses,
    )


def _adapted_metric(
    mesh: CellMesh, values: np.ndarray | None, /
) -> MeshMetricField | None:
    if values is None:
        return None
    dimension = mesh.ambient_dimension
    if values.ndim == 1:
        tensors = np.eye(dimension)[None] / values[:, None, None] ** 2
    else:
        rows, columns = np.triu_indices(dimension)
        tensors = np.zeros((values.shape[0], dimension, dimension), dtype=np.float64)
        tensors[:, rows, columns] = values
        tensors[:, columns, rows] = values
    eigenvalues = np.linalg.eigvalsh(tensors)
    if not np.all(eigenvalues > 0.0) or not np.all(np.isfinite(eigenvalues)):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Mmg returned a metric that is not positive definite.",
        )
    sizes = 1.0 / np.sqrt(eigenvalues)
    return MeshMetricField(
        MmgProvider.vertex_scope(mesh),
        tensors,
        minimum_size=float(np.nextafter(sizes.min(), 0.0)),
        maximum_size=float(np.nextafter(sizes.max(), np.inf)),
        maximum_anisotropy=max(
            1.0, float(np.nextafter(np.max(sizes[:, 0] / sizes[:, -1]), np.inf))
        ),
    )


class MmgProvider:
    """Persistent Mmg library worker: mmg2d, mmgs, and mmg3d (or collective ParMmg).

    The worker executable is ``executable``, else ``PHYDRAX_MMG_WORKER``, else
    ``phydrax-mmg-worker`` on PATH; one session is launched lazily and reused
    across calls. ``launcher`` (for example ``("mpiexec", "-n", "4")``) runs
    the collective ``phydrax-parmmg-worker``.
    """

    def __init__(
        self,
        *,
        executable: str | os.PathLike[str] | None = None,
        launcher: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        policy: NativeWorkerPolicy | None = None,
    ):
        self.worker = ProviderWorker(
            "mmg",
            executable=executable,
            environment_variable="PHYDRAX_MMG_WORKER",
            default_executable="phydrax-mmg-worker",
            build_hint=_BUILD_HINT,
            launcher=launcher,
            policy=policy,
            environment=environment,
        )

    def close(self) -> None:
        self.worker.close()

    def __enter__(self) -> MmgProvider:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    @property
    def info(self) -> MeshingProviderInfo:
        reported = self.worker.identity.reported
        collective = bool(reported["collective"])
        version = f"{reported['release']} ({str(reported['git_commit'])[:12]})"
        if collective:
            parmmg = reported["parmmg"]
            version = f"ParMmg {parmmg['release']} ({str(parmmg['git_commit'])[:12]}) / Mmg {version}"
        capabilities = (
            MeshingCapability.ANISOTROPIC_METRIC,
            MeshingCapability.MULTI_MATERIAL,
            MeshingCapability.IMPLICIT_CONFORMING,
        )
        return MeshingProviderInfo(
            "mmg",
            version,
            "LGPL-3.0-or-later",
            operations=(MeshingOperation.REMESH_SURFACE, MeshingOperation.ADAPT_VOLUME),
            source_kinds=(MeshingSourceKind.CELL_MESH,),
            capabilities=capabilities
            + (
                (MeshingCapability.PARALLEL, MeshingCapability.DISTRIBUTED)
                if collective
                else ()
            ),
            cell_kinds=("tetrahedron",) if collective else ("triangle", "tetrahedron"),
            dimensions=(3,) if collective else (2, 3),
            execution_modes=(MeshingExecutionMode.SUBPROCESS,),
        )

    @staticmethod
    def vertex_scope(mesh: CellMesh, /) -> MeshingScope:
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        vertices = mesh.entity_set(0)
        return MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            0,
            vertices.entity_set_id,
            vertices.entity_ids,
        )

    def plan(
        self,
        source: CellMeshingResult,
        /,
        *,
        metric: MeshMetricField | None = None,
        level_set: MmgLevelSet | None = None,
        motion: MmgLagrangianMotion | None = None,
        required: tuple[MeshingScope, ...] = (),
        ridges: MeshingScope | None = None,
        fields: tuple[MeshAttribute, ...] = (),
        options: MmgOptions | None = None,
        limits: MeshingLimits | None = None,
        audit_policy: CellMeshAuditPolicy | None = None,
        association_transfer: BRepAssociationTransfer | None = None,
    ) -> MmgAdaptationPlan:
        return MmgAdaptationPlan(
            source,
            MmgOptions() if options is None else options,
            MeshingLimits() if limits is None else limits,
            CellMeshAuditPolicy() if audit_policy is None else audit_policy,
            metric=metric,
            level_set=level_set,
            motion=motion,
            required=required,
            ridges=ridges,
            fields=fields,
            association_transfer=association_transfer,
        )

    def adapt(
        self,
        source: CellMeshingResult,
        /,
        *,
        metric: MeshMetricField | None = None,
        level_set: MmgLevelSet | None = None,
        motion: MmgLagrangianMotion | None = None,
        required: tuple[MeshingScope, ...] = (),
        ridges: MeshingScope | None = None,
        fields: tuple[MeshAttribute, ...] = (),
        options: MmgOptions | None = None,
        limits: MeshingLimits | None = None,
        audit_policy: CellMeshAuditPolicy | None = None,
        association_transfer: BRepAssociationTransfer | None = None,
    ) -> MmgAdaptationResult:
        return self.execute(
            self.plan(
                source,
                metric=metric,
                level_set=level_set,
                motion=motion,
                required=required,
                ridges=ridges,
                fields=fields,
                options=options,
                limits=limits,
                audit_policy=audit_policy,
                association_transfer=association_transfer,
            )
        )

    def execute(self, plan: MmgAdaptationPlan, /) -> MmgAdaptationResult:
        if not isinstance(plan, MmgAdaptationPlan):
            raise TypeError("plan must be MmgAdaptationPlan.")
        encoding = plan._encoding
        emit(
            "DEBUG",
            "provider.worker.call",
            "Mmg worker call started",
            plan_id=plan.plan_id,
            provider=encoding.backend,
            program=encoding.program,
        )
        call = self.worker.call(
            encoding.backend, encoding.parameters, encoding.arrays, limits=plan.limits
        )
        emit(
            "DEBUG",
            "provider.worker.completed",
            "Mmg worker call completed",
            plan_id=plan.plan_id,
            provider=encoding.backend,
            sequence=call.sequence,
            elapsed_seconds=call.evidence["elapsed_seconds"],
            peak_rss_bytes=call.evidence["peak_rss_bytes"],
        )
        return self._result(plan, call)

    def _result(
        self, plan: MmgAdaptationPlan, call: NativeWorkerCall, /
    ) -> MmgAdaptationResult:
        encoding = plan._encoding
        source = plan.source
        output = _compact(_collect(call), encoding.backend, plan.limits)
        region, side = _decode_regions(encoding, output.cell_references)
        mesh, order = _build_mesh(plan, output, region)
        region, side = region[order], side[order]
        cell_zones, labels, references, renamed = _cell_organization(
            plan, mesh, region, side
        )
        facet_zones, patches, boundary_references = _facet_organization(
            plan, mesh, output, renamed
        )
        retained = int(np.count_nonzero(output.required_targets >= 0))
        if retained != encoding.required_vertices:
            raise _lost("every required vertex at its source position")
        attributes = _output_attributes(plan, mesh, output, order)
        zones = tuple(cell_zones + facet_zones)
        certified = certify_cell_mesh(
            mesh,
            source.coordinate_contract,
            audit_policy=plan.audit_policy,
            patches=tuple(patches),
            zones=zones,
            labels=tuple(labels),
            attributes=attributes,
        )
        associations = ()
        if plan.association_transfer is not None and source.associations:
            # Unknown lineage: re-derive B-Rep associations by classification
            # transfer through the rebuilt facet references, then projection.
            associations = plan.association_transfer.rederive(source, certified)
            certified = certify_cell_mesh(
                mesh,
                source.coordinate_contract,
                audit_policy=plan.audit_policy,
                patches=tuple(patches),
                zones=zones,
                labels=tuple(labels),
                attributes=attributes,
                associations=associations,
            )
        _check_result_limits(certified, plan.limits)
        identity = self.worker.identity
        provider = self.info
        session = MmgSessionEvidence(
            call.evidence["session_id"],
            identity.identity_id,
            sequence=call.sequence,
            ranks=identity.ranks,
            peak_rss_bytes=call.evidence["peak_rss_bytes"],
            elapsed_seconds=call.evidence["elapsed_seconds"],
        )
        fields = _field_transfer(encoding, call.result["interpolation"])
        retention = MmgReferenceRetention(
            tuple(references + boundary_references),
            required_vertices=encoding.required_vertices,
            retained_required_vertices=retained,
        )
        metric = _adapted_metric(mesh, output.metric)
        compliance = _compliance(plan, mesh)
        result = CellMeshingResult(
            mesh,
            certified.geometry,
            source.coordinate_contract,
            certified.audit,
            certified.quality,
            compliance,
            _trace(plan, mesh, certified, compliance),
            provider,
            MeshingRuntimeInfo(
                provider.provider_id,
                provider.version,
                MeshingExecutionMode.SUBPROCESS,
                deterministic=False,
                enforced_limits=(
                    "wall_time",
                    "exchange_bytes",
                    self.worker.memory_limit_evidence(),
                    "output_vertices",
                    "output_cells",
                    "output_incidence",
                ),
            ),
            MeshingDerivativeMode.NONDIFFERENTIABLE,
            SemanticProvenance(
                {
                    "kind": "mmg-adaptation",
                    "plan": plan.plan_id,
                    "source": source.result_id,
                    "mesh": mesh.mesh_id,
                    "backend": encoding.backend,
                    "program": encoding.program,
                    "metric_representation": encoding.metric_representation,
                    "adapted_metric": call.result["adapted_metric"],
                    "worker": identity.reported,
                    "worker_identity": identity.identity_id,
                    "session": session.evidence_id,
                    "fields": None if fields is None else fields.transfer_id,
                    "references": retention.evidence_id,
                    "lineage": "unknown",
                    "output_ids": "generated",
                }
            ),
            patches=tuple(patches),
            zones=zones,
            labels=tuple(labels),
            attributes=attributes,
            associations=associations,
            adapter_reports=(_adapter_report(encoding, source.mesh, mesh),),
        )
        return MmgAdaptationResult(
            plan.plan_id,
            result,
            metric,
            encoding.metric_representation,
            fields,
            retention,
            session,
        )


def _compliance(plan: MmgAdaptationPlan, mesh: CellMesh, /) -> MeshingComplianceReport:
    points = np.asarray(mesh.coordinates)
    edges = _entity_vertices(mesh, 1)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    # Mmg applies its gradation (hgrad) with or without a metric.
    requested = [
        ("hausdorff_distance", plan.options.hausdorff_distance),
        ("gradation", plan._encoding.gradation),
    ]
    if plan.metric is not None:
        requested += [
            ("metric_minimum_size", plan.metric.minimum_size),
            ("metric_maximum_size", plan.metric.maximum_size),
        ]
    return MeshingComplianceReport(
        plan.plan_id,
        requested=tuple(requested),
        achieved=(
            ("minimum_edge", float(lengths.min())),
            ("maximum_edge", float(lengths.max())),
        ),
    )


def _trace(
    plan: MmgAdaptationPlan,
    mesh: CellMesh,
    certified: CellMeshingResult,
    compliance: MeshingComplianceReport,
    /,
) -> MeshingTrace:
    return MeshingTrace(
        (
            MeshingStageReport(
                MeshingStageKind.CONTROL_RESOLUTION,
                MeshingStageStatus.PASSED,
                input_ids=(plan.source.result_id,),
                output_ids=(plan.plan_id,),
            ),
            MeshingStageReport(
                MeshingStageKind.VOLUME_FILL
                if plan._encoding.backend == "mmg3d"
                else MeshingStageKind.SURFACE_MESHING,
                MeshingStageStatus.PASSED,
                input_ids=(plan.source.mesh.mesh_id,),
                output_ids=(mesh.mesh_id,),
                created_count=mesh.entity_set(mesh.topological_dimension).count,
            ),
            MeshingStageReport(
                MeshingStageKind.TOPOLOGY_AUDIT,
                MeshingStageStatus.PASSED,
                output_ids=(certified.audit.report_id,),
            ),
            MeshingStageReport(
                MeshingStageKind.SPECIFICATION_COMPLIANCE,
                MeshingStageStatus.PASSED,
                output_ids=(compliance.report_id,),
            ),
        )
    )


__all__ = [
    "MmgAdaptationPlan",
    "MmgAdaptationResult",
    "MmgFieldTransfer",
    "MmgLagrangianMode",
    "MmgLagrangianMotion",
    "MmgLevelSet",
    "MmgOptions",
    "MmgProvider",
    "MmgReference",
    "MmgReferenceRetention",
    "MmgSessionEvidence",
]
