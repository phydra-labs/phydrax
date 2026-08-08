#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._trainable import NonTrainableState
from ..graph._operator_topology import OperatorTopology
from ..nn.operator.data import FunctionSamples, OperatorAxis


BoundsPolicy = Literal["global", "case_bbox"]
GeometryComponent = Literal["interior", "boundary"]
MeshTopologyKind = Literal["graph", "simplicial"]
RegionalGeometryMode = Literal["fixed", "farthest_point"]


def _shape_tuple(shape: Sequence[int], /) -> tuple[int, ...]:
    result = tuple(int(size) for size in shape)
    if not result or any(size <= 1 for size in result):
        raise ValueError("Latent tensor dimensions must each be greater than one.")
    return result


def _bounds_array(bounds: Any, coord_dim: int, /) -> Array:
    result = jnp.asarray(bounds, dtype=float)
    if result.shape != (int(coord_dim), 2):
        raise ValueError(
            f"Latent bounds must have shape {(coord_dim, 2)}; got {result.shape}."
        )
    if not bool(jnp.all(jnp.isfinite(result))):
        raise ValueError("Latent bounds must be finite.")
    if not bool(jnp.all(result[:, 1] > result[:, 0])):
        raise ValueError("Every latent upper bound must exceed its lower bound.")
    return result


def _canonical_nodes(size: int, /) -> Array:
    return (jnp.arange(int(size), dtype=float) + 0.5) / float(size)


class TensorGridLatentGeometry(eqx.Module, NonTrainableState):
    """Persisted structured latent geometry for grid-based processors."""

    global_bounds: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    axis_names: tuple[str, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    bounds_policy: BoundsPolicy = eqx.field(static=True)
    margin: float = eqx.field(static=True)

    def __init__(
        self,
        shape: Sequence[int],
        /,
        *,
        bounds: Any | None = None,
        bounds_policy: BoundsPolicy = "global",
        axis_names: Sequence[str] | None = None,
        periodic: bool | Sequence[bool] = False,
        margin: float = 0.0,
    ):
        shape_ = _shape_tuple(shape)
        coord_dim = len(shape_)
        if bounds_policy not in ("global", "case_bbox"):
            raise ValueError("bounds_policy must be 'global' or 'case_bbox'.")
        if float(margin) < 0.0:
            raise ValueError("margin must be non-negative.")
        names = (
            tuple(f"latent_{index}" for index in range(coord_dim))
            if axis_names is None
            else tuple(str(name) for name in axis_names)
        )
        if len(names) != coord_dim or len(set(names)) != coord_dim:
            raise ValueError("axis_names must uniquely name every latent dimension.")
        if isinstance(periodic, bool):
            periodic_ = (bool(periodic),) * coord_dim
        else:
            periodic_ = tuple(bool(value) for value in periodic)
        if len(periodic_) != coord_dim:
            raise ValueError("periodic must give one value per latent dimension.")
        bounds_ = (
            jnp.stack((jnp.zeros((coord_dim,)), jnp.ones((coord_dim,))), axis=-1)
            if bounds is None
            else _bounds_array(bounds, coord_dim)
        )
        self.global_bounds = bounds_
        self.shape = shape_
        self.axis_names = names
        self.periodic = periodic_
        self.bounds_policy = bounds_policy
        self.margin = float(margin)

    @property
    def coord_dim(self) -> int:
        return len(self.shape)

    @property
    def point_count(self) -> int:
        return prod(self.shape)

    def axes(self) -> tuple[OperatorAxis, ...]:
        """Return canonical uniformly spaced axes used by latent processors."""
        return tuple(
            OperatorAxis(
                name,
                _canonical_nodes(size),
                quadrature_weights=jnp.full((size,), 1.0 / float(size)),
                periodic=periodic,
                basis="fourier" if periodic else "uniform",
            )
            for name, size, periodic in zip(
                self.axis_names, self.shape, self.periodic, strict=True
            )
        )

    def _case_bounds(
        self,
        case_shape: tuple[int, ...],
        /,
        *,
        source_coordinates: Array | None,
        source_mask: Array | None,
    ) -> Array:
        if self.bounds_policy == "global":
            return jnp.broadcast_to(
                self.global_bounds,
                case_shape + self.global_bounds.shape,
            )
        if source_coordinates is None:
            raise ValueError("case_bbox latent geometry requires source_coordinates.")
        coordinates = jnp.asarray(source_coordinates, dtype=float)
        expected_prefix = case_shape
        if tuple(int(size) for size in coordinates.shape[:-2]) != expected_prefix:
            raise ValueError(
                "source_coordinates case shape does not match the operator batch."
            )
        if int(coordinates.shape[-1]) != self.coord_dim:
            raise ValueError(
                f"Expected source coordinate dimension {self.coord_dim}; "
                f"got {coordinates.shape[-1]}."
            )
        mask = (
            jnp.ones(coordinates.shape[:-1], dtype=bool)
            if source_mask is None
            else jnp.asarray(source_mask, dtype=bool)
        )
        if mask.shape != coordinates.shape[:-1]:
            raise ValueError("source_mask must match the source point shape.")
        coordinates = eqx.error_if(
            coordinates,
            jnp.any(jnp.sum(mask, axis=-1) == 0),
            "case_bbox latent geometry requires one valid source point per case.",
        )
        lower = jnp.min(jnp.where(mask[..., None], coordinates, jnp.inf), axis=-2)
        upper = jnp.max(jnp.where(mask[..., None], coordinates, -jnp.inf), axis=-2)
        span = upper - lower
        span = eqx.error_if(
            span,
            jnp.any(span <= 0.0),
            "case_bbox latent geometry requires positive extent in every dimension.",
        )
        lower = lower - self.margin * span
        upper = upper + self.margin * span
        return jnp.stack((lower, upper), axis=-1)

    def coordinates(
        self,
        case_shape: Sequence[int] = (),
        /,
        *,
        source_coordinates: Array | None = None,
        source_mask: Array | None = None,
        flatten: bool = True,
    ) -> Array:
        """Materialize physical latent coordinates for every case."""
        cases = tuple(int(size) for size in case_shape)
        bounds = self._case_bounds(
            cases,
            source_coordinates=source_coordinates,
            source_mask=source_mask,
        )
        canonical_grids = jnp.meshgrid(
            *(_canonical_nodes(size) for size in self.shape),
            indexing="ij",
        )
        canonical = jnp.stack(canonical_grids, axis=-1)
        canonical = jnp.broadcast_to(canonical, cases + canonical.shape)
        lower = bounds[..., :, 0]
        upper = bounds[..., :, 1]
        for _ in self.shape:
            lower = jnp.expand_dims(lower, axis=-2)
            upper = jnp.expand_dims(upper, axis=-2)
        coordinates = lower + canonical * (upper - lower)
        if flatten:
            return coordinates.reshape(cases + (self.point_count, self.coord_dim))
        return coordinates

    def quadrature(
        self,
        case_shape: Sequence[int] = (),
        /,
        *,
        source_coordinates: Array | None = None,
        source_mask: Array | None = None,
        flatten: bool = True,
    ) -> Array:
        """Return uniform cell measures summing to each latent box volume."""
        cases = tuple(int(size) for size in case_shape)
        bounds = self._case_bounds(
            cases,
            source_coordinates=source_coordinates,
            source_mask=source_mask,
        )
        volume = jnp.prod(bounds[..., :, 1] - bounds[..., :, 0], axis=-1)
        weights = jnp.broadcast_to(
            volume[..., None] / float(self.point_count),
            cases + (self.point_count,),
        )
        if flatten:
            return weights
        return weights.reshape(cases + self.shape)


class RegionalPointLatentGeometry(eqx.Module, NonTrainableState):
    """Fixed-size regional point geometry for graph latent processors."""

    fixed_points: Array
    point_count: int = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    mode: RegionalGeometryMode = eqx.field(static=True)

    def __init__(
        self,
        point_count: int,
        coord_dim: int,
        /,
        *,
        mode: RegionalGeometryMode = "farthest_point",
        fixed_points: Any | None = None,
    ):
        count = int(point_count)
        dimension = int(coord_dim)
        if count <= 0 or dimension <= 0:
            raise ValueError("point_count and coord_dim must be positive.")
        if mode not in ("fixed", "farthest_point"):
            raise ValueError("mode must be 'fixed' or 'farthest_point'.")
        if mode == "fixed":
            if fixed_points is None:
                raise ValueError("fixed regional geometry requires fixed_points.")
            points = jnp.asarray(fixed_points, dtype=float)
            if points.shape != (count, dimension):
                raise ValueError(
                    f"fixed_points must have shape {(count, dimension)}; got {points.shape}."
                )
            if not bool(jnp.all(jnp.isfinite(points))):
                raise ValueError("fixed_points must be finite.")
        elif fixed_points is not None:
            raise ValueError("fixed_points are only used with mode='fixed'.")
        else:
            points = jnp.zeros((count, dimension), dtype=float)
        self.fixed_points = points
        self.point_count = count
        self.coord_dim = dimension
        self.mode = mode

    def coordinates(
        self,
        source_coordinates: Array,
        source_mask: Array | None = None,
        /,
    ) -> Array:
        """Return fixed points or deterministic farthest-point samples per case."""
        source = jnp.asarray(source_coordinates, dtype=float)
        if source.ndim < 2 or int(source.shape[-1]) != self.coord_dim:
            raise ValueError("source_coordinates must end in (num_points, coord_dim).")
        case_shape = tuple(int(size) for size in source.shape[:-2])
        if self.mode == "fixed":
            return jnp.broadcast_to(
                self.fixed_points,
                case_shape + self.fixed_points.shape,
            )
        mask = (
            jnp.ones(source.shape[:-1], dtype=bool)
            if source_mask is None
            else jnp.asarray(source_mask, dtype=bool)
        )
        if mask.shape != source.shape[:-1]:
            raise ValueError("source_mask must match the source point shape.")
        source = eqx.error_if(
            source,
            jnp.any(jnp.sum(mask, axis=-1) < self.point_count),
            "Farthest-point regional geometry has fewer valid sources than latent points.",
        )
        cases = prod(case_shape) if case_shape else 1
        point_count = int(source.shape[-2])
        flattened = source.reshape((cases, point_count, self.coord_dim))
        valid = mask.reshape((cases, point_count))
        mass = jnp.sum(valid, axis=-1, keepdims=True)
        centroid = (
            jnp.sum(flattened * valid[..., None].astype(flattened.dtype), axis=1) / mass
        )
        centroid_distance = jnp.sum((flattened - centroid[:, None, :]) ** 2, axis=-1)
        first = jnp.argmax(jnp.where(valid, centroid_distance, -jnp.inf), axis=-1)
        first_points = flattened[jnp.arange(cases), first]
        minimum_distance = jnp.sum((flattened - first_points[:, None, :]) ** 2, axis=-1)
        minimum_distance = jnp.where(valid, minimum_distance, -jnp.inf)

        def select_next(carry, _):
            distances, selected = carry
            index = jnp.argmax(distances, axis=-1)
            point = flattened[jnp.arange(cases), index]
            candidate_distance = jnp.sum(
                (flattened - point[:, None, :]) ** 2,
                axis=-1,
            )
            distances = jnp.minimum(distances, candidate_distance)
            distances = jnp.where(valid, distances, -jnp.inf)
            distances = distances.at[jnp.arange(cases), index].set(-jnp.inf)
            return (distances, index), index

        minimum_distance = minimum_distance.at[jnp.arange(cases), first].set(-jnp.inf)
        if self.point_count == 1:
            indices = first[:, None]
        else:
            (_, _), remaining = jax.lax.scan(
                select_next,
                (minimum_distance, first),
                xs=None,
                length=self.point_count - 1,
            )
            indices = jnp.concatenate(
                (first[:, None], jnp.swapaxes(remaining, 0, 1)), axis=1
            )
        selected = flattened[jnp.arange(cases)[:, None], indices]
        return selected.reshape(case_shape + (self.point_count, self.coord_dim))

    def quadrature(self, source_weights: Array, /) -> Array:
        """Distribute each case's source measure uniformly over regional nodes."""
        weights = jnp.asarray(source_weights, dtype=float)
        if weights.ndim < 1:
            raise ValueError("source_weights must have a source point axis.")
        total = jnp.sum(weights, axis=-1, keepdims=True)
        return jnp.broadcast_to(
            total / float(self.point_count),
            weights.shape[:-1] + (self.point_count,),
        )


def function_samples_from_geometry(
    geometry: Any,
    num_points: int,
    /,
    *,
    values: Any | None = None,
    component: GeometryComponent = "interior",
    sampler: str = "latin_hypercube",
    key: Key[Array, ""] = DOC_KEY0,
) -> FunctionSamples:
    """Sample a PhydraX geometry into measure-aware operator points."""
    from phydrax.domain import AbstractGeometry

    if not isinstance(geometry, AbstractGeometry):
        raise TypeError(
            "function_samples_from_geometry requires a PhydraX spatial geometry."
        )
    count = int(num_points)
    if count <= 0:
        raise ValueError("num_points must be positive.")
    if component == "interior":
        coordinates = geometry.sample_interior(count, sampler=sampler, key=key)
        measure = geometry.volume
    elif component == "boundary":
        coordinates = geometry.sample_boundary(count, sampler=sampler, key=key)
        measure = geometry.boundary_measure_value
    else:
        raise ValueError("component must be 'interior' or 'boundary'.")
    coordinates = jnp.asarray(coordinates, dtype=float)
    if coordinates.shape != (count, int(geometry.spatial_dim)):
        raise ValueError(
            "Geometry sampler returned an unexpected coordinate shape: "
            f"expected {(count, int(geometry.spatial_dim))}, got {coordinates.shape}."
        )
    weights = jnp.full((count,), jnp.asarray(measure, dtype=float) / float(count))
    return FunctionSamples(
        values=values,
        coordinates=coordinates,
        quadrature_weights=weights,
    )


def _point_cloud_topology(
    coordinates: Array,
    mask: Array | None,
    /,
    *,
    k: int | None,
    radius: float | None,
) -> OperatorTopology | None:
    if k is None and radius is None:
        return None
    from ..graph._geometry import point_cloud_to_graph
    from ..graph._ir import batch_graphs

    points = np.asarray(coordinates, dtype=float)
    case_shape = tuple(int(size) for size in points.shape[:-2])
    width = int(points.shape[-2])
    flattened = points.reshape((-1, width, int(points.shape[-1])))
    valid = (
        np.ones(flattened.shape[:-1], dtype=bool)
        if mask is None
        else np.broadcast_to(
            np.asarray(mask, dtype=bool),
            case_shape + (width,),
        ).reshape(flattened.shape[:-1])
    )
    graphs = []
    mappings = []
    for case_points, case_valid in zip(flattened, valid, strict=True):
        selected = case_points[case_valid]
        if selected.shape[0] == 0:
            raise ValueError("Every point-cloud geometry case requires one valid point.")
        graphs.append(
            point_cloud_to_graph(
                selected,
                k=k,
                radius=radius,
                node_features="geometry",
                edge_features="geometry",
            )
        )
        mapping = np.full((width,), -1, dtype=np.int32)
        mapping[case_valid] = np.arange(selected.shape[0], dtype=np.int32)
        mappings.append(mapping)
    sample_nodes = np.stack(mappings, axis=0)
    if not case_shape:
        return OperatorTopology.from_graph(graphs[0], sample_nodes[0], site="point")
    return OperatorTopology(
        batch_graphs(tuple(graphs)),
        sample_nodes.reshape(case_shape + (width,)),
        case_shape=case_shape,
        site="point",
    )


def function_samples_from_point_cloud(
    coordinates: Any,
    /,
    *,
    values: Any | None = None,
    quadrature_weights: Any | None = None,
    mask: Any | None = None,
    k: int | None = 8,
    radius: float | None = None,
) -> FunctionSamples:
    """Build operator samples and native topology from point-cloud data."""
    points = jnp.asarray(coordinates, dtype=float)
    if points.ndim < 2 or int(points.shape[-2]) <= 0 or int(points.shape[-1]) <= 0:
        raise ValueError(
            "coordinates must have shape case_shape + (num_points, coord_dim)."
        )
    mask_array = None if mask is None else jnp.asarray(mask, dtype=bool)
    topology = _point_cloud_topology(
        points,
        mask_array,
        k=k,
        radius=radius,
    )
    return FunctionSamples(
        values=values,
        coordinates=points,
        quadrature_weights=quadrature_weights,
        mask=mask_array,
        topology=topology,
    )


def _mesh_vertex_weights(vertices: Array, faces: Array, coord_dim: int, /) -> Array:
    coordinates = jnp.asarray(vertices, dtype=float)[:, :coord_dim]
    triangles = coordinates[jnp.asarray(faces, dtype=jnp.int32)]
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    if coord_dim == 2:
        measures = 0.5 * jnp.abs(
            first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        )
    else:
        measures = 0.5 * jnp.linalg.norm(jnp.cross(first, second), axis=-1)
    weights = jnp.zeros((coordinates.shape[0],), dtype=coordinates.dtype)
    return weights.at[jnp.asarray(faces, dtype=jnp.int32).reshape((-1,))].add(
        jnp.repeat(measures / 3.0, 3)
    )


def function_samples_from_mesh(
    mesh: Any,
    /,
    *,
    values: Any | None = None,
    topology_kind: MeshTopologyKind = "graph",
) -> FunctionSamples:
    """Use an explicit canonical simplicial mesh as operator sample sites."""

    from .brep import BRepModel
    from .simplicial import MeshRegion, TriangleMesh

    if isinstance(mesh, TriangleMesh):
        vertices, faces = mesh.vertices, mesh.faces
    elif isinstance(mesh, MeshRegion):
        vertices, faces = mesh.vertices, mesh.faces
    elif isinstance(mesh, BRepModel):
        vertices, faces = mesh.mesh_vertices, mesh.mesh_faces
    else:
        raise TypeError(
            "function_samples_from_mesh requires TriangleMesh, MeshRegion, or BRepModel."
        )
    vertices = jnp.asarray(vertices, dtype=float)
    faces = jnp.asarray(faces, dtype=jnp.int32)
    coord_dim = int(vertices.shape[1])
    coordinates = vertices
    weights = _mesh_vertex_weights(vertices, faces, coord_dim)
    if topology_kind == "graph":
        from ..graph._geometry import mesh_to_graph

        graph = mesh_to_graph(
            vertices,
            faces,
            node_features="geometry",
            edge_features="geometry",
        )
        topology = OperatorTopology.from_graph(
            graph,
            jnp.arange(vertices.shape[0], dtype=jnp.int32),
            site="vertex",
        )
    elif topology_kind == "simplicial":
        from ..graph._simplicial import triangle_mesh_to_simplicial_graph

        complex_graph = triangle_mesh_to_simplicial_graph(
            faces,
            num_vertices=int(vertices.shape[0]),
            vertex_features=coordinates,
        )
        topology = OperatorTopology.from_simplicial(complex_graph, site="vertex")
    else:
        raise ValueError("topology_kind must be 'graph' or 'simplicial'.")
    return FunctionSamples(
        values=values,
        coordinates=coordinates,
        quadrature_weights=weights,
        topology=topology,
    )


__all__ = [
    "BoundsPolicy",
    "GeometryComponent",
    "MeshTopologyKind",
    "RegionalGeometryMode",
    "RegionalPointLatentGeometry",
    "TensorGridLatentGeometry",
    "function_samples_from_geometry",
    "function_samples_from_mesh",
    "function_samples_from_point_cloud",
]
