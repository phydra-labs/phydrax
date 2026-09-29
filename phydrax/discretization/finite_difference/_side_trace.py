#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Boundary-node facets and SBP boundary traces of nodal finite-difference fields.

A nodal finite-difference field is single-valued at the grid nodes, so its
trace on a bounded box face is the exact nodal restriction `e_Gamma u`, measured
by the boundary quadrature of a declared diagonal SBP norm (the tangential
factors of `H = H_1 (x) ... (x) H_d`). The exterior facets are the (bounded
face, boundary node) incidences with one site each: faces in ascending axis
order, the lower face before the upper face, then the face nodes in row-major
order. A corner node appears once per face it bounds, each time with that
face's outward normal. Periodic axes have no faces, and there are no interior
facets. No flux is inferred from these traces.
"""

from __future__ import annotations

from typing import assert_never, Literal

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._validation import positive_integer
from ...linalg import ArraySpace
from ...typing import parse
from .._integration_domain import IntegrationDomain
from .._side_actions import (
    FacetTraceRule,
    PreparedTraceAction,
    SideActionDescriptor,
    SideGatherRoute,
    SideOrientation,
    SideTraceQuantity,
)
from .._tensor_entities import StructuredAxis
from .._tensor_support import PreparedTensorGrid
from .._topology import EntitySelection
from .._views import FieldTraceSide
from ._sbp import SBPGridNorm


def _require_boundary_nodes(axis: StructuredAxis, name: str, /) -> None:
    if axis.primary_entity != "point":
        raise ValueError(
            f"Axis {name!r} is cell-centered; boundary-node facets need "
            "point-primary bounded axes."
        )
    nodes = np.asarray(axis.point_coordinates, dtype=np.float64)
    bounds = np.asarray(axis.bounds, dtype=np.float64)
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(bounds))))
    if abs(nodes[0] - bounds[0]) > tolerance or abs(nodes[-1] - bounds[1]) > tolerance:
        raise ValueError(
            f"The end nodes of axis {name!r} do not lie on its bounds; the grid "
            "has no boundary nodes on that face."
        )


def boundary_node_facet_domain(
    grid: PreparedTensorGrid, support_id: str, /
) -> IntegrationDomain:
    """Exterior (bounded face, boundary node) facets in canonical order.

    `owner_cells` are the flat row-major node rows and `owner_local_entities`
    the local faces `2 * axis + side` (`side` 0 lower, 1 upper).
    """
    rows = np.arange(grid.size, dtype=np.int32).reshape(grid.shape)
    owners: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for axis, (name, structured) in enumerate(
        zip(grid.axis_names, grid.structured_axes, strict=True)
    ):
        if structured.periodic:
            continue
        _require_boundary_nodes(structured, name)
        for side, index in enumerate((0, grid.shape[axis] - 1)):
            face_rows = np.take(rows, index, axis=axis).reshape((-1,))
            owners.append(face_rows)
            faces.append(np.full(face_rows.shape, 2 * axis + side, dtype=np.int32))
    if not owners:
        raise ValueError("An all-periodic finite-difference grid has no boundary faces.")
    owner = np.concatenate(owners)
    return IntegrationDomain(
        "exterior_facet",
        np.arange(owner.size, dtype=np.int32),
        support_id,
        canonical_fingerprint(
            {
                "kind": "finite-difference-boundary-node-facets",
                "support": support_id,
                "shape": list(grid.shape),
                "periodic": [axis.periodic for axis in grid.structured_axes],
            }
        ),
        owner_cells=owner,
        owner_local_entities=np.concatenate(faces),
    )


def select_boundary_node_facets(
    base: IntegrationDomain, selection: EntitySelection, /
) -> IntegrationDomain:
    """Restrict the boundary-node facets to one entity selection."""
    if not isinstance(selection, EntitySelection):
        raise TypeError("selection must be EntitySelection or None.")
    if selection.entity_set_id != base.entity_set_id:
        raise ValueError("Entity selection does not match the boundary-node facets.")
    entities = np.asarray(base.entity_indices, dtype=np.int32)
    rows = np.flatnonzero(np.asarray(selection.mask, dtype=np.bool_)[entities])
    return IntegrationDomain(
        base.kind,
        entities[rows],
        base.support_id,
        base.entity_set_id,
        owner_cells=np.asarray(base.owner_cells)[rows],
        owner_local_entities=np.asarray(base.owner_local_entities)[rows],
        selection_id=selection.selection_id,
    )


def boundary_face_node_selection(
    grid: PreparedTensorGrid,
    base: IntegrationDomain,
    axis: str,
    side: Literal["lower", "upper"],
    /,
) -> EntitySelection:
    """Select every boundary-node facet of one bounded face."""
    name = str(axis)
    if name not in grid.axis_names:
        raise KeyError(f"Unknown finite-difference axis {name!r}.")
    index = grid.axis_names.index(name)
    if grid.structured_axes[index].periodic:
        raise ValueError(
            f"Axis {name!r} is periodic: it has no boundary face and no outward normal."
        )
    match side:
        case "lower":
            offset = 0
        case "upper":
            offset = 1
        case _:
            raise ValueError("side must be 'lower' or 'upper'.")
    faces = np.asarray(base.owner_local_entities, dtype=np.int32)
    return EntitySelection(
        base.entity_set_id,
        faces == 2 * index + offset,
        active_mask=np.ones(faces.shape, dtype=np.bool_),
    )


def _selected_facets(
    base: IntegrationDomain, domain: IntegrationDomain, side: FieldTraceSide, /
) -> tuple[np.ndarray, np.ndarray]:
    """Host `(node rows, local faces)` of `domain`, verified against `base`."""
    if not isinstance(domain, IntegrationDomain):
        raise TypeError("domain must be an IntegrationDomain.")
    match domain.kind:
        case "exterior_facet":
            pass
        case "interior_facet":
            raise ValueError(
                "Nodal finite-difference fields are single-valued at the grid "
                "nodes and have no interior facets."
            )
        case _:
            raise ValueError(
                "Finite-difference side traces act on exterior boundary-node facets."
            )
    if domain.support_id != base.support_id or domain.entity_set_id != base.entity_set_id:
        raise ValueError(
            "The facet domain belongs to another owner's support; prepare it from "
            "this discretization."
        )
    match side:
        case "owner":
            pass
        case "neighbor":
            raise ValueError(
                "Exterior facets have no neighbor side; prepare the owner trace."
            )
        case "average":
            raise ValueError(
                "Side actions are one-sided; compose the owner and neighbor actions "
                "instead of side='average'."
            )
        case _:
            assert_never(side)
    facets = np.asarray(domain.entity_indices, dtype=np.int32)
    if facets.size == 0 or np.any(facets >= base.entity_indices.shape[0]):
        raise ValueError("The facet domain names undeclared boundary-node facets.")
    rows = np.asarray(domain.owner_cells, dtype=np.int32)
    faces = np.asarray(domain.owner_local_entities, dtype=np.int32)
    if np.any(rows != np.asarray(base.owner_cells)[facets]) or np.any(
        faces != np.asarray(base.owner_local_entities)[facets]
    ):
        raise ValueError(
            "The facet domain routes do not match the grid's boundary-node incidence."
        )
    return rows, faces


def _tangential_measure(
    shape: tuple[int, ...], rows: np.ndarray, face_axes: np.ndarray, norm: SBPGridNorm, /
) -> np.ndarray:
    """Product of the tangential physical SBP norm weights at each boundary node."""
    indices = np.unravel_index(rows, shape)
    measure = np.ones(rows.shape, dtype=np.float64)
    for axis, (index, weights) in enumerate(zip(indices, norm.axis_weights, strict=True)):
        factor = np.asarray(weights, dtype=np.float64)[index]
        measure = measure * np.where(face_axes == axis, 1.0, factor)
    return measure


def _boundary_exact_degree(face_axes: np.ndarray, norm: SBPGridNorm, /) -> int | None:
    """Total polynomial degree the tangential SBP quadrature integrates exactly."""
    degrees = [evidence.norm_exact_degree for evidence in norm.evidence]
    if len(degrees) == 1:
        return None
    return min(
        min(degree for axis, degree in enumerate(degrees) if axis != face)
        for face in np.unique(face_axes)
    )


def _tangential_weights(normals: np.ndarray, /) -> tuple[np.ndarray, tuple[int, ...]]:
    """`u . tau` with `tau = (-n_y, n_x)` in 2-D; `u - (u . n) n` in 3-D."""
    dimension = normals.shape[-1]
    if dimension == 2:
        tangent = np.stack((-normals[:, 1], normals[:, 0]), axis=-1)
        return tangent[:, None, None, :], ()
    if dimension == 3:
        projector = np.eye(3) - normals[:, :, None] * normals[:, None, :]
        return projector[:, None, None], (3,)
    raise ValueError("One-dimensional boundary points have no tangential trace.")


def _trace_route(
    rows: np.ndarray,
    normals: np.ndarray,
    space: ArraySpace,
    quantity: SideTraceQuantity,
    /,
) -> SideGatherRoute:
    """One-node gather per facet; vector traces contract the axis normal."""
    dofs = rows[:, None]
    components = space.shape[1:]
    match quantity:
        case "value":
            return SideGatherRoute(
                dofs,
                np.ones((rows.size, 1, 1), dtype=space.dtype),
                coefficient_shape=space.shape,
                value_shape=components,
            )
        case "normal" | "tangential":
            if components != (normals.shape[-1],):
                raise ValueError(
                    f"{quantity!r} traces need a vector field with one component "
                    f"per grid axis (component_shape=({normals.shape[-1]},))."
                )
            if quantity == "normal":
                weights, value_shape = normals[:, None, None, :], ()
            else:
                weights, value_shape = _tangential_weights(normals)
            return SideGatherRoute(
                dofs,
                weights.astype(space.dtype),
                coefficient_shape=space.shape,
                mode="contracted",
                value_shape=value_shape,
            )
        case "conormal-flux":
            raise ValueError(
                "Conormal fluxes are published by compiled physics owners, not by "
                "discretization traces."
            )
        case _:
            assert_never(quantity)


def prepare_sbp_boundary_trace(
    grid: PreparedTensorGrid,
    base: IntegrationDomain,
    domain: IntegrationDomain,
    /,
    *,
    owner_id: str,
    field_space_id: str,
    dtype: np.dtype,
    rule: FacetTraceRule | None,
    quantity: SideTraceQuantity,
    side: FieldTraceSide,
    norm: SBPGridNorm,
    component_shape: tuple[int, ...],
) -> PreparedTraceAction:
    """Prepare the nodal boundary restriction measured by the SBP boundary norm."""
    if rule is not None:
        raise ValueError(
            "SBP boundary traces use the boundary quadrature fixed by the declared "
            "SBP norm; rule must be None."
        )
    if not isinstance(norm, SBPGridNorm):
        raise TypeError("norm must be an SBPGridNorm.")
    if norm.grid_id != grid.prepared_id:
        raise ValueError("The SBP norm was declared on another grid.")
    quantity = parse(quantity, SideTraceQuantity, "quantity")
    side = parse(side, FieldTraceSide, "side")
    components = tuple(
        positive_integer(size, "component_shape") for size in component_shape
    )
    rows, faces = _selected_facets(base, domain, side)
    face_axes = faces // 2
    normals = np.zeros((rows.size, len(grid.shape)), dtype=np.float64)
    normals[np.arange(rows.size), face_axes] = np.where(faces % 2 == 0, -1.0, 1.0)
    space = ArraySpace(
        (grid.size, *components),
        dtype=dtype,
        pairing=norm.pairing(components, layout="rows", dtype=dtype),
    )
    route = _trace_route(rows, normals, space, quantity)
    orientation: SideOrientation = "unoriented" if quantity == "value" else "outward"
    points = np.asarray(grid.points, dtype=np.float64)
    descriptor = SideActionDescriptor(
        owner_id=owner_id,
        field_space_id=canonical_fingerprint(
            {
                "kind": "finite-difference-node-row-space",
                "owner": owner_id,
                "space": field_space_id,
                "components": list(components),
            }
        ),
        quantity=quantity,
        representation="quadrature-values",
        orientation=orientation,
        approximation="exact",
        side=side,
        domain=domain,
        revision_id=canonical_fingerprint(
            {
                "kind": "finite-difference-side-revision",
                "grid": grid.prepared_id,
                "coordinates": array_tree_fingerprint(points),
                "norm": norm.norm_id,
            }
        ),
        rule=None,
        trace_degree=None,
        quadrature_exact_degree=_boundary_exact_degree(face_axes, norm),
    )
    return PreparedTraceAction(
        descriptor,
        route,
        space,
        sites=points[rows][:, None, :].astype(dtype),
        weights=_tangential_measure(grid.shape, rows, face_axes, norm)[:, None].astype(
            dtype
        ),
        normals=normals[:, None, :].astype(dtype),
        support_rows=np.unique(rows),
    )


__all__ = [
    "boundary_face_node_selection",
    "boundary_node_facet_domain",
    "prepare_sbp_boundary_trace",
    "select_boundary_node_facets",
]
