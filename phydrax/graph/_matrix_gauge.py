#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._oriented_path import (
    ordered_path_transport,
    oriented_edge_endpoints,
    OrientedEdgePathPlan,
)
from ..discretization._spaces import DiscreteFieldSpace, EntityDofLayout
from ..discretization._support import DiscreteSupport
from ..discretization._topology import CellComplexTopology
from ..linalg._spaces import ArraySpace
from ..metrix import (
    AbstractLieGroup,
    LieGroupStateGeometry,
    PointwiseStateGeometry,
    SpecialUnitaryGroup,
    UnitaryGroup,
)
from ..typing import checked


class MatrixGaugeLinkSpace(StrictModule, NonTrainableState):
    """Matrix Lie-group values bound to the oriented edges of one topology."""

    topology: CellComplexTopology
    support: DiscreteSupport
    field_space: DiscreteFieldSpace
    group: AbstractLieGroup
    geometry: PointwiseStateGeometry
    tail_vertices: Array
    head_vertices: Array
    num_vertices: int = eqx.field(static=True)
    num_edges: int = eqx.field(static=True)
    point_shape: tuple[int, int] = eqx.field(static=True)
    local_shape: tuple[int, ...] = eqx.field(static=True)
    link_space_id: str = eqx.field(static=True)

    @checked
    def __init__(self, topology: CellComplexTopology, group: AbstractLieGroup, /) -> None:
        if topology.dimension < 1:
            raise ValueError("Matrix gauge links require topology degree one.")
        tails, heads = oriented_edge_endpoints(topology)
        num_vertices = topology.entities(0).count
        num_edges = topology.entities(1).count
        identity = group.identity()
        support = DiscreteSupport(
            topology,
            max(1, topology.dimension),
            canonical_fingerprint(
                {
                    "kind": "matrix-gauge-link-embedding",
                    "topology": topology.topology_id,
                    "group": group.group_id,
                }
            ),
        )
        layout = EntityDofLayout(
            topology.entities(1).entity_set_id,
            num_edges,
            num_edges,
            component_shape=group.point_shape,
        )
        vector_space = ArraySpace(
            (num_edges,) + group.point_shape,
            dtype=identity.dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "matrix-gauge-link-ambient-space",
                    "topology": topology.topology_id,
                    "group": group.group_id,
                }
            ),
        )
        field_space = DiscreteFieldSpace(
            "matrix_gauge_links",
            support.support_id,
            layout,
            vector_space,
            representation="custom",
            conformity="unrestricted",
        )
        geometry = PointwiseStateGeometry(
            LieGroupStateGeometry(group),
            group.point_shape,
            local_shape=group.algebra_shape,
            tangent_shape=group.point_shape,
            geometry_id=canonical_fingerprint(
                {
                    "kind": "pointwise-matrix-gauge-geometry",
                    "topology": topology.topology_id,
                    "group": group.group_id,
                    "trivialization": "left-body",
                }
            ),
        )
        self.topology = topology
        self.support = support
        self.field_space = field_space
        self.group = group
        self.geometry = geometry
        self.tail_vertices = tails
        self.head_vertices = heads
        self.num_vertices = num_vertices
        self.num_edges = num_edges
        self.point_shape = group.point_shape
        self.local_shape = group.algebra_shape
        self.link_space_id = canonical_fingerprint(
            {
                "kind": "matrix-gauge-link-space",
                "topology": topology.topology_id,
                "field_space": field_space.field_space_id,
                "group": group.group_id,
                "geometry": geometry.geometry_id,
            }
        )

    @property
    def configuration_shape(self) -> tuple[int, ...]:
        return (self.num_edges,) + self.point_shape

    @property
    def local_coordinate_shape(self) -> tuple[int, ...]:
        return (self.num_edges,) + self.local_shape

    def contains(self, links: ArrayLike, /) -> Array:
        values = jnp.asarray(links)
        if values.shape != self.configuration_shape:
            return jnp.asarray(False)
        if isinstance(self.group, (UnitaryGroup, SpecialUnitaryGroup)):
            identity = jnp.eye(self.point_shape[0], dtype=values.dtype)
            gram = jnp.swapaxes(jnp.conj(values), -1, -2) @ values
            tolerance = jnp.maximum(
                jnp.asarray(float(self.group.tolerance), dtype=values.real.dtype),
                64.0 * jnp.finfo(values.real.dtype).eps,
            )
            finite = jnp.all(jnp.isfinite(values))
            unitary = jnp.max(jnp.abs(gram - identity)) <= tolerance
            special = (
                jnp.max(jnp.abs(jnp.linalg.det(values) - 1.0)) <= tolerance
                if isinstance(self.group, SpecialUnitaryGroup)
                else jnp.asarray(True)
            )
            return finite & unitary & special
        return self.geometry.contains(values)

    def identity(self, /) -> Array:
        return jnp.broadcast_to(
            self.group.identity(),
            self.configuration_shape,
        )


def _links(space: MatrixGaugeLinkSpace, links: ArrayLike, /) -> Array:
    if not isinstance(space, MatrixGaugeLinkSpace):
        raise TypeError("space must be MatrixGaugeLinkSpace.")
    values = jnp.asarray(links)
    if values.shape != space.configuration_shape:
        raise ValueError(
            f"links must have shape {space.configuration_shape}; got {values.shape}."
        )
    return values


def gauge_transform_links(
    space: MatrixGaugeLinkSpace,
    links: ArrayLike,
    vertex_elements: ArrayLike,
    /,
) -> Array:
    """Apply ``U_tail,head -> g_tail U_tail,head g_head^-1``."""
    values = _links(space, links)
    vertices = jnp.asarray(vertex_elements)
    expected = (space.num_vertices,) + space.point_shape
    if vertices.shape != expected:
        raise ValueError(f"vertex_elements must have shape {expected}.")
    tails = vertices[space.tail_vertices]
    heads = vertices[space.head_vertices]
    return space.group.compose(
        space.group.compose(tails, values),
        space.group.inverse(heads),
    )


def closed_path_trace(
    space: MatrixGaugeLinkSpace,
    links: ArrayLike,
    paths: OrientedEdgePathPlan,
    /,
    *,
    normalized: bool = True,
) -> Array:
    """Return complex traces of paths certified closed by their plan."""
    if not paths.require_closed:
        raise ValueError("closed_path_trace requires paths prepared as closed.")
    values = _links(space, links)
    if paths.topology_id != space.topology.topology_id:
        raise ValueError("Gauge-link space and path topology identities must agree.")
    holonomy = ordered_path_transport(paths, values)
    trace = jnp.trace(holonomy, axis1=-2, axis2=-1)
    return trace / space.point_shape[0] if normalized else trace


__all__ = [
    "MatrixGaugeLinkSpace",
    "closed_path_trace",
    "gauge_transform_links",
]
