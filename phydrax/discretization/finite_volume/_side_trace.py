#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared finite-volume face traces: cell averages and reconstructed face states.

A finite-volume field is a set of cell averages. Its value on a face is not
unique: the first-order face state of a side is that side cell's average
(`representation="cell-average"`), and a reconstruction evaluates a face state
from the side cell and its stencil (`representation="face-state"`). Facets are
the owner's faces in canonical order: structured grids enumerate the faces of
each axis in turn (row-major within an axis) and orient interior faces from the
lower to the upper cell; unstructured and triangular meshes use their face
tables, whose owner is the cell the stored area vector points out of.

Sites follow the owner cell's facet parametrization, so owner and neighbor
traces of an interior face are evaluated at the same physical points; normals
point out of the side cell and weights are the physical face measure. Linear
reconstructions publish per-facet gather routes whose transpose is exact.
Nonlinear reconstructions (WENO, limited MUSCL) publish
`PreparedNonlinearFaceTrace`, whose only derivative contract is the
linearization at a supplied state.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from math import prod
from typing import assert_never, final, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import fixed_field, NonTrainableState
from ...linalg import (
    ArraySpace,
    DiagonalPairing,
    prepare_linearization,
    PreparedLinearization,
)
from ...typing import checked, parse
from .._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from .._integration_domain import IntegrationDomain
from .._reference_cell import FacetShape
from .._side_actions import (
    AbstractSideRoute,
    FacetTraceRule,
    PreparedTraceAction,
    SideActionDescriptor,
    SideGatherRoute,
    SideRepresentation,
    SideTraceQuantity,
)
from .._topology import EntitySelection
from .._triangular import TriangleConnectivity
from .._views import FieldTraceSide
from ._cell_polynomial import PreparedCellPolynomialReconstruction
from ._reconstruction import (
    MUSCLReconstruction,
    PiecewiseConstantReconstruction,
    UnlimitedLimiter,
)
from ._structured import FiniteVolumeDiscretization
from ._triangle_fv import TriangleFiniteVolumeDiscretization
from ._triangle_polynomial import TriangleKExactReconstructionPlan
from ._triangle_reconstruction import TriangleMUSCLReconstructionPlan
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_weno import PreparedUnstructuredWENOZReconstruction
from ._weno import WENOReconstructionPlan


FiniteVolumeFacetOwner: TypeAlias = (
    FiniteVolumeDiscretization
    | UnstructuredFiniteVolumeDiscretization
    | TriangleFiniteVolumeDiscretization
)
FiniteVolumeFaceReconstruction: TypeAlias = (
    PiecewiseConstantReconstruction
    | MUSCLReconstruction
    | WENOReconstructionPlan
    | PreparedCellPolynomialReconstruction
    | PreparedUnstructuredWENOZReconstruction
    | TriangleKExactReconstructionPlan
    | TriangleMUSCLReconstructionPlan
)
_MeshOwner: TypeAlias = (
    UnstructuredFiniteVolumeDiscretization | TriangleFiniteVolumeDiscretization
)
_EMBEDDING_TOLERANCE = 64.0 * float(np.finfo(np.float64).eps)


def _require_owner(discretization: object, /) -> None:
    if not isinstance(
        discretization,
        (
            FiniteVolumeDiscretization,
            UnstructuredFiniteVolumeDiscretization,
            TriangleFiniteVolumeDiscretization,
        ),
    ):
        raise TypeError(
            "discretization must be a FiniteVolumeDiscretization, "
            "UnstructuredFiniteVolumeDiscretization, or "
            "TriangleFiniteVolumeDiscretization."
        )


def finite_volume_field_space_id(discretization: FiniteVolumeFacetOwner, /) -> str:
    """Identity of the cell-average coefficient space of one finite-volume owner."""
    return canonical_fingerprint(
        {
            "kind": "finite-volume-field-space",
            "discretization": discretization.prepared_id,
            "field": discretization.cell_space.name,
            "components": list(discretization.component_names),
        }
    )


def structured_axis_edges(
    discretization: FiniteVolumeDiscretization, /
) -> tuple[np.ndarray, ...]:
    """Host cell edges of every structured axis."""
    edges = []
    for axis in discretization.grid.structured_axes:
        widths = np.asarray(axis.interval_widths, dtype=np.float64)
        lower = float(np.asarray(axis.bounds)[0])
        edges.append(lower + np.concatenate(([0.0], np.cumsum(widths))))
    return tuple(edges)


@final
@dataclass(frozen=True)
class _FacetTable:
    """Host owner/neighbor incidence of every facet in canonical order."""

    entity_set_id: str
    owner: np.ndarray
    neighbor: np.ndarray
    owner_local: np.ndarray
    neighbor_local: np.ndarray
    periodic: np.ndarray


@final
@dataclass(frozen=True)
class _FacetFrame:
    """Host sites, measures, and outward normals of one side of selected facets."""

    shape: FacetShape
    side: FieldTraceSide
    cells: np.ndarray
    local: np.ndarray
    sites: np.ndarray
    weights: np.ndarray
    normals: np.ndarray
    revision_id: str


def _structured_table(discretization: FiniteVolumeDiscretization, /) -> _FacetTable:
    cell_shape = discretization.cell_shape
    owners, neighbors, owner_locals, neighbor_locals, periodics = [], [], [], [], []
    for axis, (layout, structured) in enumerate(
        zip(discretization.face_layouts, discretization.grid.structured_axes, strict=True)
    ):
        count = cell_shape[axis]
        periodic = bool(structured.periodic)
        if layout.shape[axis] != (count if periodic else count + 1):
            raise ValueError("Structured face layouts do not match the cell layout.")
        faces = np.indices(layout.shape).reshape((len(cell_shape), -1))
        index = faces[axis]
        lower = faces.copy()
        lower[axis] = (index - 1) % count
        upper = faces.copy()
        upper[axis] = np.minimum(index, count - 1)
        lower_cells = np.ravel_multi_index(tuple(lower), cell_shape)
        upper_cells = np.ravel_multi_index(tuple(upper), cell_shape)
        first = index == 0
        last = index == count
        exterior_lower = first & (not periodic)
        owners.append(np.where(exterior_lower, upper_cells, lower_cells))
        neighbors.append(np.where(exterior_lower | last, -1, upper_cells))
        owner_locals.append(np.where(exterior_lower, 2 * axis, 2 * axis + 1))
        neighbor_locals.append(np.where(exterior_lower | last, -1, 2 * axis))
        periodics.append(first & periodic)
    return _FacetTable(
        entity_set_id=canonical_fingerprint(
            {
                "kind": "structured-finite-volume-facets",
                "layouts": [
                    layout.entity_set_id for layout in discretization.face_layouts
                ],
            }
        ),
        owner=np.concatenate(owners).astype(np.int32),
        neighbor=np.concatenate(neighbors).astype(np.int32),
        owner_local=np.concatenate(owner_locals).astype(np.int32),
        neighbor_local=np.concatenate(neighbor_locals).astype(np.int32),
        periodic=np.concatenate(periodics),
    )


def _cell_facets(
    discretization: _MeshOwner, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host `(cell facets, incidence signs, valid slots)` of a mesh owner."""
    connectivity = discretization.connectivity
    match connectivity:
        case TriangleConnectivity():
            facets = np.asarray(connectivity.cell_edges, dtype=np.int32)[:, :3]
            signs = np.asarray(connectivity.cell_edge_signs, dtype=np.float64)[:, :3]
            return facets, signs, np.ones(facets.shape, dtype=np.bool_)
        case PolygonalConnectivity():
            return (
                np.asarray(connectivity.cell_edges, dtype=np.int32),
                np.asarray(connectivity.cell_edge_signs, dtype=np.float64),
                np.asarray(connectivity.cell_edge_valid, dtype=np.bool_),
            )
        case TetrahedralConnectivity():
            facets = np.asarray(connectivity.cell_faces, dtype=np.int32)
            return (
                facets,
                np.asarray(connectivity.cell_face_signs, dtype=np.float64),
                np.ones(facets.shape, dtype=np.bool_),
            )
        case _:
            raise ValueError(
                "Finite-volume side traces cover edge and triangular faces; general "
                "polyhedral faces have no facet trace rule."
            )


def mesh_cell_dimension(discretization: _MeshOwner, /) -> int:
    match discretization:
        case TriangleFiniteVolumeDiscretization():
            return 2
        case UnstructuredFiniteVolumeDiscretization():
            return discretization.cell_dimension
        case _:
            assert_never(discretization)


def _local_positions(
    facets: np.ndarray, valid: np.ndarray, cells: np.ndarray, faces: np.ndarray, /
) -> np.ndarray:
    match = (facets[cells] == faces[:, None]) & valid[cells]
    if not np.all(np.any(match, axis=1)):
        raise ValueError("The face table names cells that do not contain the face.")
    return np.argmax(match, axis=1).astype(np.int32)


def _mesh_table(discretization: _MeshOwner, /) -> _FacetTable:
    facets, _, valid = _cell_facets(discretization)
    owner = np.asarray(discretization.owner_cells, dtype=np.int32)
    neighbor = np.asarray(discretization.neighbor_cells, dtype=np.int32)
    faces = np.arange(owner.size, dtype=np.int32)
    interior = neighbor >= 0
    neighbor_local = np.full(owner.shape, -1, dtype=np.int32)
    neighbor_local[interior] = _local_positions(
        facets, valid, neighbor[interior], faces[interior]
    )
    dimension = mesh_cell_dimension(discretization)
    return _FacetTable(
        entity_set_id=discretization.topology.entity_sets[dimension - 1].entity_set_id,
        owner=owner,
        neighbor=neighbor,
        owner_local=_local_positions(facets, valid, owner, faces),
        neighbor_local=neighbor_local,
        periodic=np.zeros(owner.shape, dtype=np.bool_),
    )


def _facet_table(discretization: FiniteVolumeFacetOwner, /) -> _FacetTable:
    match discretization:
        case FiniteVolumeDiscretization():
            return _structured_table(discretization)
        case (
            UnstructuredFiniteVolumeDiscretization()
            | TriangleFiniteVolumeDiscretization()
        ):
            return _mesh_table(discretization)
        case _:
            assert_never(discretization)


def _cell_entity_set_id(discretization: FiniteVolumeFacetOwner, /) -> str:
    match discretization:
        case FiniteVolumeDiscretization():
            return discretization.cell_layout.entity_set_id
        case (
            UnstructuredFiniteVolumeDiscretization()
            | TriangleFiniteVolumeDiscretization()
        ):
            dimension = mesh_cell_dimension(discretization)
            return discretization.topology.entity_sets[dimension].entity_set_id
        case _:
            assert_never(discretization)


def _cell_count(discretization: FiniteVolumeFacetOwner, /) -> int:
    match discretization:
        case FiniteVolumeDiscretization():
            return prod(discretization.cell_shape)
        case (
            UnstructuredFiniteVolumeDiscretization()
            | TriangleFiniteVolumeDiscretization()
        ):
            return discretization.cell_count
        case _:
            assert_never(discretization)


def _selected(
    rows: np.ndarray, count: int, entity_set_id: str, selection: EntitySelection | None, /
) -> tuple[np.ndarray, str | None]:
    if selection is None:
        return rows, None
    if not isinstance(selection, EntitySelection):
        raise TypeError("selection must be an EntitySelection or None.")
    mask = np.asarray(selection.mask, dtype=np.bool_)
    if selection.entity_set_id != entity_set_id or mask.shape != (count,):
        raise ValueError("The selection belongs to another entity set of this owner.")
    return rows[mask[rows]], selection.selection_id


def finite_volume_integration_domain(
    discretization: FiniteVolumeFacetOwner,
    kind: str,
    selection: EntitySelection | None,
    /,
) -> IntegrationDomain:
    """Cell, exterior-facet, or interior-facet domain of a finite-volume owner."""
    _require_owner(discretization)
    kind_ = str(kind)
    support_id = discretization.support.support_id
    match kind_:
        case "cell":
            count = _cell_count(discretization)
            entity_set_id = _cell_entity_set_id(discretization)
            rows, selection_id = _selected(
                np.arange(count, dtype=np.int32), count, entity_set_id, selection
            )
            return IntegrationDomain(
                "cell", rows, support_id, entity_set_id, selection_id=selection_id
            )
        case "exterior_facet" | "interior_facet":
            table = _facet_table(discretization)
            exterior = kind_ == "exterior_facet"
            candidates = np.flatnonzero(
                (table.neighbor < 0) if exterior else (table.neighbor >= 0)
            ).astype(np.int32)
            rows, selection_id = _selected(
                candidates, table.owner.size, table.entity_set_id, selection
            )
            return IntegrationDomain(
                kind_,
                rows,
                support_id,
                table.entity_set_id,
                owner_cells=table.owner[rows],
                neighbor_cells=None if exterior else table.neighbor[rows],
                owner_local_entities=table.owner_local[rows],
                neighbor_local_entities=None if exterior else table.neighbor_local[rows],
                periodic_face_mask=None if exterior else table.periodic[rows],
                selection_id=selection_id,
            )
        case _:
            raise ValueError(
                "Finite-volume integration domains are 'cell', 'exterior_facet', or "
                "'interior_facet'."
            )


def _checked_facets(
    discretization: FiniteVolumeFacetOwner,
    table: _FacetTable,
    domain: IntegrationDomain,
    /,
) -> np.ndarray:
    if not isinstance(domain, IntegrationDomain):
        raise TypeError("domain must be an IntegrationDomain.")
    match domain.kind:
        case "exterior_facet" | "interior_facet":
            pass
        case _:
            raise ValueError(
                "Finite-volume side traces act on exterior or interior facets."
            )
    if (
        domain.support_id != discretization.support.support_id
        or domain.entity_set_id != table.entity_set_id
    ):
        raise ValueError(
            "The facet domain belongs to another owner's support; prepare it from "
            "this discretization."
        )
    facets = np.asarray(domain.entity_indices, dtype=np.int32)
    if facets.size == 0 or np.any(facets >= table.owner.size):
        raise ValueError("The facet domain names no facets of this discretization.")
    interior = domain.kind == "interior_facet"
    routes_match = np.array_equal(
        np.asarray(domain.owner_cells), table.owner[facets]
    ) and np.array_equal(
        np.asarray(domain.owner_local_entities), table.owner_local[facets]
    )
    if interior:
        routes_match = (
            routes_match
            and np.array_equal(np.asarray(domain.neighbor_cells), table.neighbor[facets])
            and np.array_equal(
                np.asarray(domain.neighbor_local_entities), table.neighbor_local[facets]
            )
        )
    if not routes_match or np.any((table.neighbor[facets] >= 0) != interior):
        raise ValueError(
            "The facet domain routes do not match this discretization's face table."
        )
    if np.any(table.periodic[facets]):
        raise ValueError(
            "Periodic faces join two physically distinct images; side traces need "
            "one shared facet embedding."
        )
    return facets


def _side_route(
    table: _FacetTable,
    facets: np.ndarray,
    domain: IntegrationDomain,
    side: FieldTraceSide,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    match side:
        case "owner":
            return table.owner[facets], table.owner_local[facets]
        case "neighbor":
            if domain.kind != "interior_facet":
                raise ValueError(
                    "Exterior facets have no neighbor side; prepare the owner trace."
                )
            return table.neighbor[facets], table.neighbor_local[facets]
        case "average":
            raise ValueError(
                "Side actions are one-sided; compose the owner and neighbor actions "
                "instead of side='average'."
            )
        case _:
            assert_never(side)


def _revision(owner_id: str, coordinates: object, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "finite-volume-side-revision",
            "owner": owner_id,
            "coordinates": array_tree_fingerprint(coordinates),
        }
    )


def _structured_sites(
    edges: tuple[np.ndarray, ...],
    cell_shape: tuple[int, ...],
    cells: np.ndarray,
    local: np.ndarray,
    parameters: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sites, measure factors, and outward normals of axis-aligned cell faces."""
    dimension = len(cell_shape)
    index = np.stack(np.unravel_index(cells, cell_shape), axis=-1)
    axis = local // 2
    upper = local % 2
    sites = np.zeros((cells.size, parameters.shape[0], dimension), dtype=np.float64)
    measure = np.ones((cells.size,), dtype=np.float64)
    normals = np.zeros((cells.size, dimension), dtype=np.float64)
    rows = np.arange(cells.size)
    normals[rows, axis] = np.where(upper == 1, 1.0, -1.0)
    for facet in range(cells.size):
        normal_axis = int(axis[facet])
        transverse = [other for other in range(dimension) if other != normal_axis]
        position = index[facet, normal_axis] + upper[facet]
        sites[facet, :, normal_axis] = edges[normal_axis][position]
        for column, other in enumerate(transverse):
            start = edges[other][index[facet, other]]
            width = edges[other][index[facet, other] + 1] - start
            sites[facet, :, other] = start + parameters[:, column] * width
            measure[facet] *= width
    return sites, measure, normals


def _structured_frame(
    discretization: FiniteVolumeDiscretization,
    table: _FacetTable,
    facets: np.ndarray,
    domain: IntegrationDomain,
    rule: FacetTraceRule,
    side: FieldTraceSide,
    /,
) -> _FacetFrame:
    edges = structured_axis_edges(discretization)
    cell_shape = discretization.cell_shape
    shape: FacetShape
    match len(cell_shape):
        case 1:
            shape = "point"
        case 2:
            shape = "edge"
        case 3:
            shape = "quadrilateral"
        case _:
            raise ValueError("Structured side traces cover one to three dimensions.")
    parameters, reference = rule.reference(shape)
    cells, local = _side_route(table, facets, domain, side)
    sites, measure, normals = _structured_sites(
        edges, cell_shape, cells, local, parameters
    )
    owner_sites, _, _ = _structured_sites(
        edges, cell_shape, table.owner[facets], table.owner_local[facets], parameters
    )
    if domain.kind == "interior_facet":
        neighbor_sites, _, _ = _structured_sites(
            edges,
            cell_shape,
            table.neighbor[facets],
            table.neighbor_local[facets],
            parameters,
        )
        scale = max(1.0, max(float(np.max(np.abs(values))) for values in edges))
        if float(np.max(np.abs(neighbor_sites - owner_sites))) > (
            _EMBEDDING_TOLERANCE * scale
        ):
            raise ValueError(
                "The owner and neighbor embeddings of an interior facet disagree; the "
                "side sites are not shared physical points."
            )
    return _FacetFrame(
        shape=shape,
        side=side,
        cells=cells,
        local=local,
        sites=owner_sites,
        weights=measure[:, None] * reference[None, :],
        normals=np.broadcast_to(normals[:, None, :], sites.shape).copy(),
        revision_id=_revision(discretization.prepared_id, edges),
    )


def _mesh_frame(
    discretization: _MeshOwner,
    table: _FacetTable,
    facets: np.ndarray,
    domain: IntegrationDomain,
    rule: FacetTraceRule,
    side: FieldTraceSide,
    /,
) -> _FacetFrame:
    _, signs, _ = _cell_facets(discretization)
    points = np.asarray(discretization.vertices, dtype=np.float64)
    owner_sign = signs[table.owner[facets], table.owner_local[facets]]
    if domain.kind == "interior_facet":
        neighbor_sign = signs[table.neighbor[facets], table.neighbor_local[facets]]
        if np.any(owner_sign * neighbor_sign >= 0.0):
            raise ValueError(
                "The owner and neighbor embeddings of an interior facet disagree; the "
                "side sites are not shared physical points."
            )
    cells, local = _side_route(table, facets, domain, side)
    forward = owner_sign > 0.0
    shape: FacetShape
    connectivity = discretization.connectivity
    match connectivity:
        case TriangleConnectivity() | PolygonalConnectivity():
            shape = "edge"
            vertices = np.asarray(connectivity.edges, dtype=np.int32)[facets]
            ordered = np.where(forward[:, None], vertices, vertices[:, ::-1])
            start, stop = points[ordered[:, 0]], points[ordered[:, 1]]
            parameters, reference = rule.reference(shape)
            t = parameters[:, 0]
            sites = start[:, None, :] + t[None, :, None] * (stop - start)[:, None, :]
            tangent = stop - start
            length = np.linalg.norm(tangent, axis=-1)
            outward = np.stack((tangent[:, 1], -tangent[:, 0]), axis=-1) / length[:, None]
            jacobian = length
        case TetrahedralConnectivity():
            shape = "triangle"
            vertices = np.asarray(connectivity.faces, dtype=np.int32)[facets]
            ordered = np.where(forward[:, None], vertices, vertices[:, (0, 2, 1)])
            first, second, third = (points[ordered[:, column]] for column in range(3))
            parameters, reference = rule.reference(shape)
            sites = (
                first[:, None, :]
                + parameters[None, :, 0, None] * (second - first)[:, None, :]
                + parameters[None, :, 1, None] * (third - first)[:, None, :]
            )
            area = np.cross(second - first, third - first)
            jacobian = np.linalg.norm(area, axis=-1)
            outward = area / jacobian[:, None]
        case _:
            raise ValueError(
                "Finite-volume side traces cover edge and triangular faces; general "
                "polyhedral faces have no facet trace rule."
            )
    if np.any(jacobian <= 0.0):
        raise ValueError("Finite-volume side traces require faces of positive measure.")
    orientation = 1.0 if side == "owner" else -1.0
    return _FacetFrame(
        shape=shape,
        side=side,
        cells=cells,
        local=local,
        sites=sites,
        weights=jacobian[:, None] * reference[None, :],
        normals=np.broadcast_to(orientation * outward[:, None, :], sites.shape).copy(),
        revision_id=_revision(discretization.prepared_id, points),
    )


def _facet_frame(
    discretization: FiniteVolumeFacetOwner,
    domain: IntegrationDomain,
    rule: FacetTraceRule,
    side: FieldTraceSide,
    /,
) -> _FacetFrame:
    if not isinstance(rule, FacetTraceRule):
        raise TypeError("rule must be a FacetTraceRule.")
    side_ = parse(side, FieldTraceSide, "side")
    table = _facet_table(discretization)
    facets = _checked_facets(discretization, table, domain)
    match discretization:
        case FiniteVolumeDiscretization():
            return _structured_frame(discretization, table, facets, domain, rule, side_)
        case (
            UnstructuredFiniteVolumeDiscretization()
            | TriangleFiniteVolumeDiscretization()
        ):
            return _mesh_frame(discretization, table, facets, domain, rule, side_)
        case _:
            assert_never(discretization)


def _require_prepared_on(
    prepared: FiniteVolumeFacetOwner, discretization: FiniteVolumeFacetOwner, /
) -> None:
    if prepared.prepared_id != discretization.prepared_id:
        raise ValueError(
            "The reconstruction was prepared on a different finite-volume discretization."
        )


def _structured_radius(
    reconstruction: MUSCLReconstruction | WENOReconstructionPlan, /
) -> int:
    match reconstruction:
        case MUSCLReconstruction():
            return reconstruction.ghost_width - 1
        case WENOReconstructionPlan():
            return 1 if reconstruction.order == 3 else 2
        case _:
            assert_never(reconstruction)


def _require_reconstruction(
    discretization: FiniteVolumeFacetOwner, reconstruction: object, /
) -> tuple[SideRepresentation, int, bool]:
    """Representation, trace degree along a facet, and coefficient linearity."""
    structured = isinstance(discretization, FiniteVolumeDiscretization)
    match reconstruction:
        case PiecewiseConstantReconstruction():
            return "cell-average", 0, True
        case MUSCLReconstruction() | WENOReconstructionPlan():
            if not structured:
                raise ValueError(
                    "Directional MUSCL/WENO face plans act on structured grids; "
                    "unstructured meshes use cell polynomial, WENO-Z, or triangle "
                    "reconstructions."
                )
            linear = isinstance(reconstruction, MUSCLReconstruction) and isinstance(
                reconstruction.limiter, UnlimitedLimiter
            )
            return "face-state", 0, linear
        case PreparedCellPolynomialReconstruction():
            _require_prepared_on(reconstruction.discretization, discretization)
            return "face-state", reconstruction.basis.degree, True
        case PreparedUnstructuredWENOZReconstruction():
            _require_prepared_on(reconstruction.discretization, discretization)
            return "face-state", reconstruction.optimal.basis.degree, False
        case TriangleKExactReconstructionPlan():
            _require_prepared_on(reconstruction.prepared.discretization, discretization)
            return "face-state", 2, True
        case TriangleMUSCLReconstructionPlan():
            _require_prepared_on(reconstruction.gradient.discretization, discretization)
            return "face-state", 1, reconstruction.limiter == "unlimited"
        case _:
            raise TypeError(
                "reconstruction must be PiecewiseConstantReconstruction, "
                "MUSCLReconstruction, WENOReconstructionPlan, "
                "PreparedCellPolynomialReconstruction, "
                "PreparedUnstructuredWENOZReconstruction, "
                "TriangleKExactReconstructionPlan, or TriangleMUSCLReconstructionPlan."
            )


def _axis_stencils(
    discretization: FiniteVolumeDiscretization,
    cells: np.ndarray,
    local: np.ndarray,
    radius: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-major cells and widths of each side cell's normal-axis stencil."""
    cell_shape = discretization.cell_shape
    edges = structured_axis_edges(discretization)
    index = np.stack(np.unravel_index(cells, cell_shape), axis=-1)
    axis = local // 2
    offsets = np.arange(-radius, radius + 1)
    stencils = np.empty((cells.size, offsets.size), dtype=np.int32)
    widths = np.empty((cells.size, offsets.size), dtype=np.float64)
    for facet in range(cells.size):
        normal_axis = int(axis[facet])
        count = cell_shape[normal_axis]
        along = index[facet, normal_axis] + offsets
        if discretization.grid.structured_axes[normal_axis].periodic:
            along = along % count
        elif np.any(along < 0) or np.any(along >= count):
            raise ValueError(
                "The reconstruction stencil of a selected facet leaves the grid; that "
                "face state depends on boundary ghost states owned by the boundary "
                "condition."
            )
        stencil = np.repeat(index[facet][None, :], offsets.size, axis=0)
        stencil[:, normal_axis] = along
        stencils[facet] = np.ravel_multi_index(tuple(stencil.T), cell_shape)
        widths[facet] = np.diff(edges[normal_axis])[along]
    return stencils, widths


def _axis_trace(
    plan: MUSCLReconstruction | WENOReconstructionPlan,
    values: Array,
    widths: Array,
    upper: Array,
    /,
) -> Array:
    """Face state of the stencil's center cell on its lower or upper face."""
    match plan:
        case WENOReconstructionPlan():
            radius = 1 if plan.order == 3 else 2
            minimum = 3 if plan.order == 3 else 6
            padded = jnp.concatenate(
                (values, jnp.repeat(values[-1:], max(0, minimum - values.shape[0]), 0))
            )
            left, right = plan.reconstruct(padded)
            upper_trace, lower_trace = left[radius], right[radius - 1]
        case MUSCLReconstruction():
            radius = (values.shape[0] - 1) // 2
            left, right = plan.reconstruct_axis(
                values,
                0,
                periodic=False,
                lower_exterior=values[0],
                upper_exterior=values[-1],
                cell_widths=widths,
            )
            upper_trace, lower_trace = left[radius + 1], right[radius]
        case _:
            assert_never(plan)
    return jnp.where(upper, upper_trace, lower_trace)


def _stencil_weights(
    cells: np.ndarray,
    basis: np.ndarray,
    factors: np.ndarray,
    stencils: np.ndarray,
    valid: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather route `u_c + sum_s (b . F_s)(u_s - u_c)` of stencil reconstructions."""
    neighbor = np.einsum("fqk,fks->fqs", basis, factors) * valid[:, None, :]
    center = 1.0 - np.sum(neighbor, axis=-1)
    dofs = np.concatenate(
        (cells[:, None], np.where(valid, stencils, cells[:, None])), axis=1
    )
    return dofs.astype(np.int32), np.concatenate((center[..., None], neighbor), axis=-1)


def _linear_route(
    discretization: FiniteVolumeFacetOwner,
    reconstruction: FiniteVolumeFaceReconstruction,
    frame: _FacetFrame,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-facet `(dofs, weights)` of a coefficient-linear face state."""
    cells = frame.cells
    count = frame.sites.shape[1]
    match reconstruction:
        case PiecewiseConstantReconstruction():
            return cells[:, None], np.ones((cells.size, count, 1), dtype=np.float64)
        case PreparedCellPolynomialReconstruction():
            basis = reconstruction.basis_values(
                jnp.asarray(cells), jnp.asarray(frame.sites)
            )
            return _stencil_weights(
                cells,
                np.asarray(basis, dtype=np.float64),
                np.asarray(reconstruction.factors, dtype=np.float64)[cells],
                np.asarray(reconstruction.stencil_cells, dtype=np.int32)[cells],
                np.asarray(reconstruction.stencil_valid, dtype=np.bool_)[cells],
            )
        case TriangleKExactReconstructionPlan():
            prepared = reconstruction.prepared
            basis = prepared.basis_derivative(
                jnp.asarray(cells), jnp.asarray(frame.sites), (0, 0)
            )
            return _stencil_weights(
                cells,
                np.asarray(basis, dtype=np.float64),
                np.asarray(prepared.factors, dtype=np.float64)[cells],
                np.asarray(prepared.neighbor_cells, dtype=np.int32)[cells],
                np.asarray(prepared.valid, dtype=np.bool_)[cells],
            )
        case TriangleMUSCLReconstructionPlan():
            gradient = reconstruction.gradient
            centers = np.asarray(gradient.discretization.cell_centers, dtype=np.float64)
            return _stencil_weights(
                cells,
                frame.sites - centers[cells][:, None, :],
                np.asarray(gradient.factors, dtype=np.float64)[cells],
                np.asarray(gradient.neighbor_cells, dtype=np.int32)[cells],
                np.asarray(gradient.valid, dtype=np.bool_)[cells],
            )
        case MUSCLReconstruction():
            if not isinstance(discretization, FiniteVolumeDiscretization):
                raise TypeError("Directional plans act on structured grids.")
            stencils, widths = _axis_stencils(
                discretization, cells, frame.local, _structured_radius(reconstruction)
            )
            weights = jax.vmap(
                jax.jacfwd(
                    lambda values, width, upper: _axis_trace(
                        reconstruction, values[:, None], width, upper
                    )[0]
                )
            )(
                jnp.zeros(stencils.shape, dtype=jnp.float64),
                jnp.asarray(widths),
                jnp.asarray(frame.local % 2 == 1),
            )
            host = np.asarray(weights, dtype=np.float64)
            return stencils, np.broadcast_to(
                host[:, None, :], (cells.size, count, host.shape[1])
            ).copy()
        case WENOReconstructionPlan() | PreparedUnstructuredWENOZReconstruction():
            raise TypeError("Nonlinear reconstructions have no linear gather route.")
        case _:
            assert_never(reconstruction)


@final
class _TensorCellSideRoute(AbstractSideRoute, NonTrainableState):
    """Side gather over the row-major cells of a structured state array."""

    cells: SideGatherRoute
    global_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, cells: SideGatherRoute, global_shape: tuple[int, ...], /) -> None:
        if prod(global_shape) != prod(cells.coefficient_shape):
            raise ValueError("The structured state does not flatten onto the route.")
        self.cells = cells
        self.global_shape = global_shape

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        return self.global_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.cells.output_shape

    def apply(self, coefficients: Array, /) -> Array:
        return self.cells.apply(coefficients.reshape(self.cells.coefficient_shape))

    def transpose(self, values: Array, /) -> Array:
        return self.cells.transpose(values).reshape(self.global_shape)


def _support_rows(
    discretization: FiniteVolumeFacetOwner, dofs: np.ndarray, /
) -> np.ndarray:
    match discretization:
        case FiniteVolumeDiscretization():
            return np.unique(np.unravel_index(dofs, discretization.cell_shape)[0])
        case (
            UnstructuredFiniteVolumeDiscretization()
            | TriangleFiniteVolumeDiscretization()
        ):
            return np.unique(dofs)
        case _:
            assert_never(discretization)


def _coefficient_space(discretization: FiniteVolumeFacetOwner, /) -> ArraySpace:
    space = discretization.cell_space.vector_space
    if not isinstance(space, ArraySpace):
        raise TypeError("Finite-volume cell averages are array valued.")
    return space


def _descriptor(
    discretization: FiniteVolumeFacetOwner,
    domain: IntegrationDomain,
    rule: FacetTraceRule,
    frame: _FacetFrame,
    reconstruction: FiniteVolumeFaceReconstruction,
    representation: SideRepresentation,
    degree: int,
    /,
) -> SideActionDescriptor:
    match reconstruction:
        case (
            PiecewiseConstantReconstruction()
            | MUSCLReconstruction()
            | WENOReconstructionPlan()
            | TriangleKExactReconstructionPlan()
            | TriangleMUSCLReconstructionPlan()
        ):
            identity = reconstruction.plan_id
        case (
            PreparedCellPolynomialReconstruction()
            | PreparedUnstructuredWENOZReconstruction()
        ):
            identity = reconstruction.prepared_id
        case _:
            assert_never(reconstruction)
    return SideActionDescriptor(
        owner_id=discretization.prepared_id,
        field_space_id=finite_volume_field_space_id(discretization),
        quantity="value",
        representation=representation,
        orientation="unoriented",
        approximation="exact",
        side=frame.side,
        domain=domain,
        revision_id=canonical_fingerprint(
            {
                "kind": "finite-volume-face-state-revision",
                "geometry": frame.revision_id,
                "reconstruction": identity,
            }
        ),
        rule=rule,
        trace_degree=degree,
        quadrature_exact_degree=rule.exact_degree(frame.shape),
    )


def _prepared_frame(
    discretization: FiniteVolumeFacetOwner,
    field_name: str,
    domain: IntegrationDomain,
    rule: FacetTraceRule,
    quantity: SideTraceQuantity,
    side: FieldTraceSide,
    /,
) -> _FacetFrame:
    _require_owner(discretization)
    if str(field_name) != discretization.cell_space.name:
        raise KeyError(f"Unknown finite-volume field {field_name!r}.")
    quantity_ = parse(quantity, SideTraceQuantity, "quantity")
    match quantity_:
        case "value":
            pass
        case "normal" | "tangential":
            raise ValueError(
                "Finite-volume cell averages publish value face states; normal "
                "fluxes belong to the conservation-law owner, not to the trace."
            )
        case "conormal-flux":
            raise ValueError(
                "Conormal fluxes are published by compiled physics owners, not by "
                "discretization traces."
            )
        case _:
            assert_never(quantity_)
    return _facet_frame(discretization, domain, rule, side)


def prepare_finite_volume_side_trace(
    discretization: FiniteVolumeFacetOwner,
    field_name: str,
    domain: IntegrationDomain,
    /,
    *,
    rule: FacetTraceRule,
    quantity: SideTraceQuantity = "value",
    side: FieldTraceSide = "owner",
    reconstruction: FiniteVolumeFaceReconstruction | None = None,
) -> PreparedTraceAction:
    """Prepare the linear face state of a finite-volume field on selected facets.

    `reconstruction=None` (or `PiecewiseConstantReconstruction()`) publishes
    the side cell average at every site (`representation="cell-average"`,
    `trace_degree=0`). Coefficient-linear reconstructions publish reconstructed
    face states (`representation="face-state"`) through per-facet gather routes
    whose transpose is exact. Nonlinear reconstructions are refused here; use
    `prepare_finite_volume_nonlinear_face_trace`.
    """
    frame = _prepared_frame(discretization, field_name, domain, rule, quantity, side)
    reconstruction_ = (
        PiecewiseConstantReconstruction() if reconstruction is None else reconstruction
    )
    representation, degree, linear = _require_reconstruction(
        discretization, reconstruction_
    )
    if not linear:
        raise ValueError(
            "The reconstruction is nonlinear in the cell averages and has no exact "
            "transpose; prepare_nonlinear_face_trace publishes its face states with a "
            "linearization at a supplied state."
        )
    space = _coefficient_space(discretization)
    dofs, weights = _linear_route(discretization, reconstruction_, frame)
    cells = SideGatherRoute(
        dofs,
        np.asarray(weights, dtype=space.dtype),
        coefficient_shape=(_cell_count(discretization), discretization.component_count),
        mode="componentwise",
        value_shape=(discretization.component_count,),
    )
    route: AbstractSideRoute = (
        _TensorCellSideRoute(cells, space.shape)
        if isinstance(discretization, FiniteVolumeDiscretization)
        else cells
    )
    return PreparedTraceAction(
        _descriptor(
            discretization, domain, rule, frame, reconstruction_, representation, degree
        ),
        route,
        space,
        sites=frame.sites,
        weights=frame.weights,
        normals=frame.normals,
        support_rows=_support_rows(discretization, dofs),
    )


class _AbstractFaceStateEvaluator(StrictModule):
    """Pure-JAX face states of one nonlinear reconstruction at prepared sites."""

    @abc.abstractmethod
    def evaluate(self, coefficients: Array, /) -> Array:
        raise NotImplementedError


@final
class _MeshFaceStateEvaluator(_AbstractFaceStateEvaluator, NonTrainableState):
    reconstruction: (
        PreparedUnstructuredWENOZReconstruction | TriangleMUSCLReconstructionPlan
    )
    cells: Array
    sites: Array

    def __init__(
        self,
        reconstruction: PreparedUnstructuredWENOZReconstruction
        | TriangleMUSCLReconstructionPlan,
        cells: np.ndarray,
        sites: np.ndarray,
        /,
    ) -> None:
        self.reconstruction = reconstruction
        self.cells = jnp.asarray(cells, dtype=jnp.int32)
        self.sites = jnp.asarray(sites)

    def evaluate(self, coefficients: Array, /) -> Array:
        return self.reconstruction.evaluate_cells(
            coefficients, self.cells, self.sites.astype(coefficients.dtype)
        )


@final
class _StructuredFaceStateEvaluator(_AbstractFaceStateEvaluator):
    plan: MUSCLReconstruction | WENOReconstructionPlan
    stencils: Array = fixed_field()
    widths: Array = fixed_field()
    upper: Array = fixed_field()
    cell_count: int = eqx.field(static=True)
    sites_per_facet: int = eqx.field(static=True)

    def __init__(
        self,
        plan: MUSCLReconstruction | WENOReconstructionPlan,
        stencils: np.ndarray,
        widths: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        cell_count: int,
        sites_per_facet: int,
    ) -> None:
        self.plan = plan
        self.stencils = jnp.asarray(stencils, dtype=jnp.int32)
        self.widths = jnp.asarray(widths)
        self.upper = jnp.asarray(upper, dtype=jnp.bool_)
        self.cell_count = cell_count
        self.sites_per_facet = sites_per_facet

    def evaluate(self, coefficients: Array, /) -> Array:
        components = coefficients.shape[-1:]
        cells = coefficients.reshape((self.cell_count, *components))
        traces = jax.vmap(
            lambda values, widths, upper: _axis_trace(self.plan, values, widths, upper)
        )(cells[self.stencils], self.widths.astype(coefficients.dtype), self.upper)
        return jnp.broadcast_to(
            traces[:, None, :],
            (traces.shape[0], self.sites_per_facet, *components),
        )


@final
class PreparedNonlinearFaceTrace(StrictModule, NonTrainableState):
    """Face states of a nonlinear finite-volume reconstruction on selected facets.

    `apply` evaluates the reconstructed face states (WENO, limited MUSCL) at
    the prepared `sites` with shape `(facets, sites_per_facet, components)`.
    The states are nonlinear in the cell averages, so there is no coordinate
    transpose or dual pullback: `linearize(coefficients)` prepares the exact
    local derivative at one supplied state, whose `vjp` pulls trace covectors
    back to the owner's residual rows at that state only. `support_rows` are
    the cells the selected face states depend on.
    """

    descriptor: SideActionDescriptor
    coefficient_space: ArraySpace
    evaluator: _AbstractFaceStateEvaluator
    sites: Array
    weights: Array
    normals: Array
    support_rows: Array

    @checked
    def __init__(
        self,
        descriptor: SideActionDescriptor,
        coefficient_space: ArraySpace,
        evaluator: _AbstractFaceStateEvaluator,
        /,
        *,
        sites: ArrayLike,
        weights: ArrayLike,
        normals: ArrayLike,
        support_rows: ArrayLike,
    ) -> None:
        if descriptor.representation != "face-state" or descriptor.quantity != "value":
            raise ValueError("Nonlinear face traces publish value face states.")
        sites_ = np.asarray(sites)
        weights_ = np.asarray(weights)
        normals_ = np.asarray(normals)
        output = jax.eval_shape(evaluator.evaluate, coefficient_space.structure())
        if (
            sites_.ndim != 3
            or output.shape[:2] != sites_.shape[:2]
            or weights_.shape != sites_.shape[:2]
            or normals_.shape != sites_.shape
            or output.shape[0] != descriptor.facets.shape[0]
        ):
            raise ValueError("Face states, sites, weights, and normals do not align.")
        if not (np.all(np.isfinite(sites_)) and np.all(weights_ > 0.0)):
            raise ValueError("Side sites must be finite and weights positive.")
        rows = np.asarray(support_rows, dtype=np.int32)
        if rows.ndim != 1 or rows.size == 0 or np.any(np.diff(rows) <= 0):
            raise ValueError("support_rows must be sorted unique coefficient rows.")
        self.descriptor = descriptor
        self.coefficient_space = coefficient_space
        self.evaluator = evaluator
        self.sites = jnp.asarray(sites_)
        self.weights = jnp.asarray(weights_)
        self.normals = jnp.asarray(normals_)
        self.support_rows = jnp.asarray(rows)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return jax.eval_shape(
            self.evaluator.evaluate, self.coefficient_space.structure()
        ).shape

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self.output_shape[2:]

    @property
    def action_id(self) -> str:
        return self.descriptor.descriptor_id

    def require_revision(self, revision_id: str, /) -> None:
        """Refuse a geometry or reconstruction other than the prepared one."""
        if revision_id != self.descriptor.revision_id:
            raise ValueError(
                "The face trace was prepared on another revision; prepare it again "
                "on the refreshed owner."
            )

    def apply(self, coefficients: ArrayLike, /) -> Array:
        """Evaluate the face states of the owner's cell averages."""
        return self.evaluator.evaluate(self.coefficient_space.validate(coefficients))

    def trace_space(self) -> ArraySpace:
        """Face-state space paired by the facet measure."""
        output = jax.eval_shape(
            self.evaluator.evaluate, self.coefficient_space.structure()
        )
        measure = jnp.broadcast_to(
            self.weights.reshape(self.weights.shape + (1,) * (len(output.shape) - 2)),
            output.shape,
        ).astype(output.dtype)
        return ArraySpace(
            output.shape,
            dtype=output.dtype,
            pairing=DiagonalPairing(
                measure,
                pairing_id=canonical_fingerprint(
                    {"kind": "side-measure-pairing", "action": self.action_id}
                ),
            ),
        )

    def linearize(self, coefficients: ArrayLike, /) -> PreparedLinearization:
        """Prepare the exact face-state derivative at one supplied state."""
        values = self.coefficient_space.validate(coefficients)
        return prepare_linearization(
            self.evaluator.evaluate,
            values,
            source=self.coefficient_space,
            target=self.trace_space(),
            linearization_id=canonical_fingerprint(
                {
                    "kind": "prepared-nonlinear-face-trace-linearization",
                    "action": self.action_id,
                }
            ),
        )


def prepare_finite_volume_nonlinear_face_trace(
    discretization: FiniteVolumeFacetOwner,
    field_name: str,
    domain: IntegrationDomain,
    /,
    *,
    rule: FacetTraceRule,
    reconstruction: FiniteVolumeFaceReconstruction,
    quantity: SideTraceQuantity = "value",
    side: FieldTraceSide = "owner",
) -> PreparedNonlinearFaceTrace:
    """Prepare the face states of a nonlinear reconstruction on selected facets.

    Only the side cells' stencils are evaluated. Linear reconstructions are
    refused here; `prepare_side_trace` publishes them with an exact transpose.
    """
    frame = _prepared_frame(discretization, field_name, domain, rule, quantity, side)
    representation, degree, linear = _require_reconstruction(
        discretization, reconstruction
    )
    if linear:
        raise ValueError(
            "The reconstruction is linear in the cell averages; prepare_side_trace "
            "publishes it with an exact transpose."
        )
    evaluator: _AbstractFaceStateEvaluator
    match reconstruction:
        case PreparedUnstructuredWENOZReconstruction():
            evaluator = _MeshFaceStateEvaluator(reconstruction, frame.cells, frame.sites)
            optimal = reconstruction.optimal
            rows = np.concatenate(
                [frame.cells]
                + [
                    np.asarray(candidate.stencil_cells)[frame.cells][
                        np.asarray(candidate.stencil_valid)[frame.cells]
                    ]
                    for candidate in (optimal, *reconstruction.sectors)
                ]
            )
        case TriangleMUSCLReconstructionPlan():
            evaluator = _MeshFaceStateEvaluator(reconstruction, frame.cells, frame.sites)
            gradient = reconstruction.gradient
            rows = np.concatenate(
                (
                    frame.cells,
                    np.asarray(gradient.neighbor_cells)[frame.cells][
                        np.asarray(gradient.valid)[frame.cells]
                    ],
                )
            )
        case MUSCLReconstruction() | WENOReconstructionPlan():
            if not isinstance(discretization, FiniteVolumeDiscretization):
                raise TypeError("Directional plans act on structured grids.")
            stencils, widths = _axis_stencils(
                discretization,
                frame.cells,
                frame.local,
                _structured_radius(reconstruction),
            )
            if isinstance(reconstruction, WENOReconstructionPlan) and not np.allclose(
                widths, widths[:, :1], rtol=1.0e-12, atol=0.0
            ):
                raise ValueError(
                    "WENO face plans assume uniform cells across each facet stencil."
                )
            evaluator = _StructuredFaceStateEvaluator(
                reconstruction,
                stencils,
                widths,
                frame.local % 2 == 1,
                cell_count=_cell_count(discretization),
                sites_per_facet=frame.sites.shape[1],
            )
            rows = stencils.reshape((-1,))
        case (
            PiecewiseConstantReconstruction()
            | PreparedCellPolynomialReconstruction()
            | TriangleKExactReconstructionPlan()
        ):
            raise TypeError("Linear reconstructions use prepare_side_trace.")
        case _:
            assert_never(reconstruction)
    return PreparedNonlinearFaceTrace(
        _descriptor(
            discretization, domain, rule, frame, reconstruction, representation, degree
        ),
        _coefficient_space(discretization),
        evaluator,
        sites=frame.sites,
        weights=frame.weights,
        normals=frame.normals,
        support_rows=_support_rows(discretization, rows),
    )


__all__ = ["PreparedNonlinearFaceTrace"]
