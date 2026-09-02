#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._conservation_boundary import AbstractConservationBoundary
from ...discretization.fem._mortar import FiniteElementMortarPlan


DGTraceRouteKind = Literal["conforming", "mortar", "boundary", "periodic"]


class PreparedDGTraceRoute(StrictModule, NonTrainableState):
    route_kind: DGTraceRouteKind = eqx.field(static=True)
    owner_dofs: Array
    neighbour_dofs: Array
    owner_basis: Array
    neighbour_basis: Array
    owner_gradients: Array
    physical_points: Array
    physical_weights: Array
    component_transform: Array
    coordinate_transform: Array
    normal: Array
    mortar: FiniteElementMortarPlan | None
    boundary: AbstractConservationBoundary | None
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        route_kind: DGTraceRouteKind,
        owner_dofs: ArrayLike,
        /,
        *,
        neighbour_dofs: ArrayLike = (),
        owner_basis: ArrayLike = (),
        neighbour_basis: ArrayLike = (),
        owner_gradients: ArrayLike = (),
        physical_points: ArrayLike = (),
        physical_weights: ArrayLike = (),
        normal: ArrayLike = (),
        mortar: FiniteElementMortarPlan | None = None,
        boundary: AbstractConservationBoundary | None = None,
        component_transform: ArrayLike = (),
        coordinate_transform: ArrayLike = (),
        route_id: str,
    ):
        if route_kind not in ("conforming", "mortar", "boundary", "periodic"):
            raise ValueError("Unknown DG trace route kind.")
        owner = jnp.asarray(owner_dofs, dtype=jnp.int32)
        neighbour = jnp.asarray(neighbour_dofs, dtype=jnp.int32)
        owner_basis_ = jnp.asarray(owner_basis)
        neighbour_basis_ = jnp.asarray(neighbour_basis)
        gradients = jnp.asarray(owner_gradients)
        points = jnp.asarray(physical_points)
        weights = jnp.asarray(physical_weights)
        normal_ = jnp.asarray(normal)
        identifier = str(route_id)
        component_transform_ = jnp.asarray(component_transform)
        coordinate_transform_ = jnp.asarray(coordinate_transform)
        if owner.ndim != 1 or not identifier:
            raise ValueError("DG trace route owner DOFs and ID are required.")
        if route_kind == "mortar":
            if not isinstance(mortar, FiniteElementMortarPlan) or neighbour.ndim != 1:
                raise ValueError("Mortar routes require mortar and neighbour DOFs.")
        elif route_kind == "boundary":
            if not isinstance(boundary, AbstractConservationBoundary):
                raise ValueError("Boundary routes require a conservation boundary.")
        else:
            if (
                neighbour.ndim != 1
                or owner_basis_.ndim != 2
                or neighbour_basis_.ndim != 2
                or weights.ndim != 1
                or normal_.ndim != 2
            ):
                raise ValueError("Conforming/periodic routes require two trace bases.")
        if route_kind == "periodic" and (
            component_transform_.ndim != 2 or coordinate_transform_.ndim != 2
        ):
            raise ValueError(
                "Periodic routes require component and coordinate transforms."
            )
        self.route_kind = route_kind
        self.owner_dofs = owner
        self.neighbour_dofs = neighbour
        self.owner_basis = owner_basis_
        self.neighbour_basis = neighbour_basis_
        self.owner_gradients = gradients
        self.physical_points = points
        self.physical_weights = weights
        self.normal = normal_
        self.mortar = mortar
        self.component_transform = component_transform_
        self.coordinate_transform = coordinate_transform_
        self.boundary = boundary
        self.route_id = canonical_fingerprint(
            {
                "kind": "prepared-dg-trace-route",
                "route_kind": route_kind,
                "source_route": identifier,
                "owner_width": int(owner.size),
                "neighbour_width": int(neighbour.size),
                "mortar": None if mortar is None else mortar.plan_id,
                "boundary": None if boundary is None else boundary.boundary_id,
                "component_transform_shape": tuple(component_transform_.shape),
                "coordinate_transform_shape": tuple(coordinate_transform_.shape),
            }
        )


class PreparedDGMortarBatch(StrictModule, NonTrainableState):
    owner_dofs: Array
    neighbour_dofs: Array
    left_interpolation: Array
    right_interpolation: Array
    left_dual_pullback: Array
    right_dual_pullback: Array
    physical_weights: Array
    normal: Array
    route_ids: tuple[str, ...] = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(self, routes, /):
        values = tuple(routes)
        if not values or any(
            not isinstance(route, PreparedDGTraceRoute)
            or route.route_kind != "mortar"
            or route.mortar is None
            for route in values
        ):
            raise ValueError("Mortar batches require prepared mortar routes.")
        shapes = {
            (
                route.owner_dofs.shape,
                route.neighbour_dofs.shape,
                route.mortar.left_interpolation.shape,
                route.mortar.right_interpolation.shape,
            )
            for route in values
        }
        if len(shapes) != 1:
            raise ValueError("Mortar batch route shapes differ.")
        self.owner_dofs = jnp.stack(tuple(route.owner_dofs for route in values))
        self.neighbour_dofs = jnp.stack(tuple(route.neighbour_dofs for route in values))
        self.left_interpolation = jnp.stack(
            tuple(route.mortar.left_interpolation for route in values)
        )
        self.right_interpolation = jnp.stack(
            tuple(route.mortar.right_interpolation for route in values)
        )
        self.left_dual_pullback = jnp.stack(
            tuple(route.mortar.left_raw_dual_pullback for route in values)
        )
        self.right_dual_pullback = jnp.stack(
            tuple(route.mortar.right_raw_dual_pullback for route in values)
        )
        self.physical_weights = jnp.stack(
            tuple(route.mortar.physical_weights for route in values)
        )
        self.normal = jnp.stack(tuple(route.normal for route in values))
        self.route_ids = tuple(route.route_id for route in values)
        self.batch_id = canonical_fingerprint(
            {
                "kind": "prepared-dg-mortar-batch",
                "routes": self.route_ids,
                "shape": tuple(next(iter(shapes))),
            }
        )


def batch_dg_mortar_routes(
    routes: tuple[PreparedDGTraceRoute, ...], /
) -> tuple[PreparedDGMortarBatch, ...]:
    groups = {}
    for route in routes:
        if route.mortar is None:
            raise ValueError("Mortar route has no mortar plan.")
        key = (
            route.owner_dofs.shape,
            route.neighbour_dofs.shape,
            route.mortar.left_interpolation.shape,
            route.mortar.right_interpolation.shape,
        )
        groups.setdefault(key, []).append(route)
    return tuple(PreparedDGMortarBatch(groups[key]) for key in sorted(groups, key=str))


class PreparedDGBoundaryBatch(StrictModule, NonTrainableState):
    owner_dofs: Array
    owner_basis: Array
    physical_points: Array
    physical_weights: Array
    normal: Array
    boundary: AbstractConservationBoundary
    route_ids: tuple[str, ...] = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(self, routes, /):
        values = tuple(routes)
        if (
            not values
            or any(
                not isinstance(route, PreparedDGTraceRoute)
                or route.route_kind != "boundary"
                or route.boundary is None
                for route in values
            )
            or len({route.boundary.boundary_id for route in values}) != 1
            or len(
                {
                    (
                        route.owner_dofs.shape,
                        route.owner_basis.shape,
                        route.physical_points.shape,
                    )
                    for route in values
                }
            )
            != 1
        ):
            raise ValueError("Boundary batches require equal-shape boundary routes.")
        self.owner_dofs = jnp.stack(tuple(route.owner_dofs for route in values))
        self.owner_basis = jnp.stack(tuple(route.owner_basis for route in values))
        self.physical_points = jnp.stack(tuple(route.physical_points for route in values))
        self.physical_weights = jnp.stack(
            tuple(route.physical_weights for route in values)
        )
        self.normal = jnp.stack(tuple(route.normal for route in values))
        self.boundary = values[0].boundary
        self.route_ids = tuple(route.route_id for route in values)
        self.batch_id = canonical_fingerprint(
            {
                "kind": "prepared-dg-boundary-batch",
                "routes": self.route_ids,
                "boundary": self.boundary.boundary_id,
            }
        )


def batch_dg_boundary_routes(
    routes: tuple[PreparedDGTraceRoute, ...], /
) -> tuple[PreparedDGBoundaryBatch, ...]:
    groups = {}
    for route in routes:
        if route.boundary is None:
            raise ValueError("Boundary route has no boundary policy.")
        key = (
            route.boundary.boundary_id,
            route.owner_dofs.shape,
            route.owner_basis.shape,
            route.physical_points.shape,
        )
        groups.setdefault(key, []).append(route)
    return tuple(PreparedDGBoundaryBatch(groups[key]) for key in sorted(groups, key=str))


__all__ = [
    "DGTraceRouteKind",
    "PreparedDGBoundaryBatch",
    "PreparedDGMortarBatch",
    "PreparedDGTraceRoute",
    "batch_dg_boundary_routes",
    "batch_dg_mortar_routes",
]
