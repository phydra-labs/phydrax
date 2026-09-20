#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._lattice_boundary import LatticeBoundaryPhasePlan
from ..discretization._oriented_path import CellBoundaryPathPlan
from ..metrix._complex_matrix_manifold import SpecialUnitaryGroup, UnitaryGroup
from ..metrix._gauge_representation import AbstractGaugeRepresentation
from ._matrix_gauge import MatrixGaugeLinkSpace


def _links(space: MatrixGaugeLinkSpace, value: ArrayLike, /) -> Array:
    links = jnp.asarray(value)
    if links.shape != space.configuration_shape:
        raise ValueError(
            f"links must have shape {space.configuration_shape}; got {links.shape}."
        )
    return links


class GaugeCovariantShiftPlan(StrictModule, NonTrainableState):
    """Prepared nearest-neighbor parallel transport on a C-order tensor lattice."""

    link_space: MatrixGaugeLinkSpace
    representation: AbstractGaugeRepresentation
    boundary_phases: LatticeBoundaryPhasePlan
    forward_sites: Array
    backward_sites: Array
    forward_edges: Array
    backward_edges: Array
    forward_orientations: Array
    backward_orientations: Array
    forward_phase_factors: Array
    backward_phase_factors: Array
    site_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    link_space_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        link_space: MatrixGaugeLinkSpace,
        representation: AbstractGaugeRepresentation,
        forward_sites: ArrayLike,
        forward_edges: ArrayLike,
        forward_orientations: ArrayLike,
        boundary_phases: LatticeBoundaryPhasePlan,
        /,
    ):
        if not isinstance(link_space, MatrixGaugeLinkSpace):
            raise TypeError("link_space must be MatrixGaugeLinkSpace.")
        if not isinstance(representation, AbstractGaugeRepresentation):
            raise TypeError("representation must implement AbstractGaugeRepresentation.")
        if representation.group.group_id != link_space.group.group_id:
            raise ValueError("Representation and link space must use the same group.")
        if not isinstance(boundary_phases, LatticeBoundaryPhasePlan):
            raise TypeError("boundary_phases must be LatticeBoundaryPhasePlan.")
        site_count = boundary_phases.site_count
        dimension = boundary_phases.dimension
        if link_space.num_vertices != site_count:
            raise ValueError(
                "Gauge-link vertices must equal the boundary-plan site count."
            )
        sites = np.asarray(forward_sites, dtype=np.int64)
        edges = np.asarray(forward_edges, dtype=np.int64)
        orientations = np.asarray(forward_orientations, dtype=np.int64)
        expected = (site_count, dimension)
        if (
            sites.shape != expected
            or edges.shape != expected
            or orientations.shape != expected
        ):
            raise ValueError(
                f"forward_sites, forward_edges, and forward_orientations must all have shape {expected}."
            )
        if np.any(sites < 0) or np.any(sites >= site_count):
            raise ValueError("Forward-site routes lie outside the lattice.")
        if np.any(edges < 0) or np.any(edges >= link_space.num_edges):
            raise ValueError("Forward-edge routes lie outside the link space.")
        if np.any(np.abs(orientations) != 1):
            raise ValueError("Forward-link orientations must be plus or minus one.")
        coordinates = np.stack(
            np.unravel_index(np.arange(site_count), boundary_phases.topology.axis_sizes),
            axis=-1,
        )
        canonical_sites = np.empty(expected, dtype=np.int64)
        for axis, size in enumerate(boundary_phases.topology.axis_sizes):
            neighbor = coordinates.copy()
            neighbor[:, axis] = (neighbor[:, axis] + 1) % size
            canonical_sites[:, axis] = np.ravel_multi_index(
                neighbor.T, boundary_phases.topology.axis_sizes
            )
        if not np.array_equal(sites, canonical_sites):
            raise ValueError(
                "forward_sites must be the canonical C-order positive-axis neighbors."
            )
        backward_sites = np.empty_like(sites)
        backward_edges = np.empty_like(edges)
        backward_orientations = np.empty_like(orientations)
        expected_vertices = np.arange(site_count)
        tails = np.asarray(link_space.tail_vertices)
        heads = np.asarray(link_space.head_vertices)
        for axis in range(dimension):
            if not np.array_equal(np.sort(sites[:, axis]), expected_vertices):
                raise ValueError("Every forward-site axis must be a bijection.")
            for site in range(site_count):
                edge = int(edges[site, axis])
                orientation = int(orientations[site, axis])
                route_tail = int(tails[edge]) if orientation > 0 else int(heads[edge])
                route_head = int(heads[edge]) if orientation > 0 else int(tails[edge])
                if route_tail != site or route_head != int(sites[site, axis]):
                    raise ValueError(
                        "Each oriented forward edge must join its site to its declared neighbor."
                    )
                target = int(sites[site, axis])
                backward_sites[target, axis] = site
                backward_edges[target, axis] = edge
                backward_orientations[target, axis] = -orientation
        forward_factors = np.stack(
            tuple(
                np.asarray(boundary_phases.phase_factors(axis, 1)).reshape((-1,))
                for axis in range(dimension)
            ),
            axis=1,
        )
        backward_factors = np.stack(
            tuple(
                np.asarray(boundary_phases.phase_factors(axis, -1)).reshape((-1,))
                for axis in range(dimension)
            ),
            axis=1,
        )
        self.link_space = link_space
        self.representation = representation
        self.boundary_phases = boundary_phases
        self.forward_sites = jnp.asarray(sites, dtype=jnp.int32)
        self.backward_sites = jnp.asarray(backward_sites, dtype=jnp.int32)
        self.forward_edges = jnp.asarray(edges, dtype=jnp.int32)
        self.backward_edges = jnp.asarray(backward_edges, dtype=jnp.int32)
        self.forward_orientations = jnp.asarray(orientations, dtype=jnp.int32)
        self.backward_orientations = jnp.asarray(backward_orientations, dtype=jnp.int32)
        self.forward_phase_factors = jnp.asarray(forward_factors)
        self.backward_phase_factors = jnp.asarray(backward_factors)
        self.site_count = site_count
        self.dimension = dimension
        self.link_space_id = link_space.link_space_id
        self.representation_id = representation.representation_id
        self.boundary_id = boundary_phases.plan_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gauge-covariant-shift-plan",
                "link_space": link_space.link_space_id,
                "representation": representation.representation_id,
                "boundary": boundary_phases.plan_id,
                "forward_sites": array_tree_fingerprint(sites),
                "forward_edges": array_tree_fingerprint(edges),
                "forward_orientations": array_tree_fingerprint(orientations),
                "site_order": "c-order",
            }
        )

    def _field(self, field: ArrayLike, /) -> Array:
        value = jnp.asarray(field)
        if value.ndim < 2 or value.shape[0] != self.site_count:
            raise ValueError("field must begin with site_count and contain a color axis.")
        axis = value.ndim + self.representation.color_axis
        if axis <= 0 or axis >= value.ndim:
            raise ValueError("The representation color axis must lie in field payload.")
        if value.shape[axis] != self.representation.dimension:
            raise ValueError("Field color extent disagrees with its representation.")
        return value

    def _shift(
        self,
        links: ArrayLike,
        field: ArrayLike,
        sites: Array,
        edges: Array,
        orientations: Array,
        phases: Array,
        /,
    ) -> Array:
        link_values = _links(self.link_space, links)
        matter = self._field(field)
        route_links = link_values[edges]
        inverse_links = self.link_space.group.inverse(route_links)
        oriented_links = jnp.where(
            (orientations > 0)[..., None, None], route_links, inverse_links
        )
        neighbor_values = matter[sites]
        transported = self.representation.apply(oriented_links, neighbor_values)
        phase_values = phases.astype(transported.dtype)
        return transported * phase_values.reshape(
            phases.shape + (1,) * (transported.ndim - phases.ndim)
        )

    def forward(self, links: ArrayLike, field: ArrayLike, /) -> Array:
        """Return all positive-direction covariant shifts as ``(site, axis, ...)``."""
        return self._shift(
            links,
            field,
            self.forward_sites,
            self.forward_edges,
            self.forward_orientations,
            self.forward_phase_factors,
        )

    def backward(self, links: ArrayLike, field: ArrayLike, /) -> Array:
        """Return all negative-direction covariant shifts as ``(site, axis, ...)``."""
        return self._shift(
            links,
            field,
            self.backward_sites,
            self.backward_edges,
            self.backward_orientations,
            self.backward_phase_factors,
        )


class GaugeStaplePlan(StrictModule, NonTrainableState):
    """Fixed-capacity complementary path products for matrix gauge links."""

    link_space: MatrixGaugeLinkSpace
    boundaries: CellBoundaryPathPlan
    complement_edges: Array
    complement_orientations: Array
    complement_valid: Array
    selected_orientations: Array
    route_valid: Array
    route_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        link_space: MatrixGaugeLinkSpace,
        boundaries: CellBoundaryPathPlan,
        /,
        *,
        maximum_staples_per_link: int = 64,
    ):
        if not isinstance(link_space, MatrixGaugeLinkSpace):
            raise TypeError("link_space must be MatrixGaugeLinkSpace.")
        if not isinstance(link_space.group, (UnitaryGroup, SpecialUnitaryGroup)):
            raise TypeError("Gauge staples require U(N) or SU(N) matrix links.")
        if not isinstance(boundaries, CellBoundaryPathPlan):
            raise TypeError("boundaries must be CellBoundaryPathPlan.")
        paths = boundaries.paths
        if paths.topology_id != link_space.topology.topology_id:
            raise ValueError("Staple boundaries and link space must share one topology.")
        resource_limit = int(maximum_staples_per_link)
        if resource_limit < 1 or resource_limit > 1024:
            raise ValueError("maximum_staples_per_link must lie in [1, 1024].")
        path_edges = np.asarray(paths.edge_indices, dtype=np.int32)
        path_orientations = np.asarray(paths.orientations, dtype=np.int32)
        path_valid = np.asarray(paths.valid, dtype=np.bool_)
        routes: list[list[tuple[int, int, list[int], list[int]]]] = [
            [] for _ in range(link_space.num_edges)
        ]
        for path in range(paths.num_paths):
            length = int(np.sum(path_valid[path]))
            for selected in range(length):
                edge = int(path_edges[path, selected])
                order = [int((selected + offset) % length) for offset in range(1, length)]
                routes[edge].append(
                    (
                        path,
                        int(path_orientations[path, selected]),
                        [int(path_edges[path, position]) for position in order],
                        [int(path_orientations[path, position]) for position in order],
                    )
                )
        capacity = max(1, max((len(values) for values in routes), default=0))
        if capacity > resource_limit:
            raise ValueError("Gauge-staple incidence exceeds maximum_staples_per_link.")
        complement_length = max(0, paths.max_length - 1)
        complement_edges = np.zeros(
            (link_space.num_edges, capacity, complement_length), dtype=np.int32
        )
        complement_orientations = np.ones_like(complement_edges)
        complement_valid = np.zeros_like(complement_edges, dtype=np.bool_)
        selected_orientations = np.ones((link_space.num_edges, capacity), dtype=np.int32)
        route_valid = np.zeros((link_space.num_edges, capacity), dtype=np.bool_)
        for edge, edge_routes in enumerate(routes):
            for slot, (_, selected_sign, edges_, signs_) in enumerate(edge_routes):
                length = len(edges_)
                complement_edges[edge, slot, :length] = edges_
                complement_orientations[edge, slot, :length] = signs_
                complement_valid[edge, slot, :length] = True
                selected_orientations[edge, slot] = selected_sign
                route_valid[edge, slot] = True
        self.link_space = link_space
        self.boundaries = boundaries
        self.complement_edges = jnp.asarray(complement_edges)
        self.complement_orientations = jnp.asarray(complement_orientations)
        self.complement_valid = jnp.asarray(complement_valid)
        self.selected_orientations = jnp.asarray(selected_orientations)
        self.route_valid = jnp.asarray(route_valid)
        self.route_capacity = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-gauge-staple-plan",
                "link_space": link_space.link_space_id,
                "boundaries": boundaries.boundary_plan_id,
                "complement_edges": array_tree_fingerprint(complement_edges),
                "complement_orientations": array_tree_fingerprint(
                    complement_orientations
                ),
                "complement_valid": array_tree_fingerprint(complement_valid),
                "selected_orientations": array_tree_fingerprint(selected_orientations),
                "route_capacity": capacity,
                "resource_limit": resource_limit,
            }
        )

    def staple(self, links: ArrayLike, edge: ArrayLike, /) -> Array:
        """Sum all oriented complementary products incident on one selected link."""
        values = _links(self.link_space, links)
        edge_ = jnp.asarray(edge, dtype=jnp.int32)
        identity = jnp.broadcast_to(
            self.link_space.group.identity(dtype=values.dtype),
            (self.route_capacity,) + self.link_space.point_shape,
        )
        route_edges = self.complement_edges[edge_]
        route_orientations = self.complement_orientations[edge_]
        route_active = self.complement_valid[edge_]

        def step(product: Array, position: Array) -> tuple[Array, None]:
            factors = values[route_edges[:, position]]
            inverses = self.link_space.group.inverse(factors)
            oriented = jnp.where(
                (route_orientations[:, position] > 0)[..., None, None],
                factors,
                inverses,
            )
            candidate = self.link_space.group.compose(product, oriented)
            active = route_active[:, position][..., None, None]
            return jnp.where(active, candidate, product), None

        products, _ = jax.lax.scan(
            step,
            identity,
            jnp.arange(self.complement_edges.shape[-1]),
        )
        selected_positive = self.selected_orientations[edge_] > 0
        contributions = jnp.where(
            selected_positive[..., None, None],
            products,
            self.link_space.group.inverse(products),
        )
        contributions = jnp.where(
            self.route_valid[edge_][..., None, None], contributions, 0.0
        )
        return jnp.sum(contributions, axis=0)

    def staples(self, links: ArrayLike, /) -> Array:
        """Return one complementary-product sum per link, with the link shape."""
        values = _links(self.link_space, links)
        return jax.vmap(lambda edge: self.staple(values, edge))(
            jnp.arange(self.link_space.num_edges, dtype=jnp.int32)
        )


__all__ = ["GaugeCovariantShiftPlan", "GaugeStaplePlan"]
