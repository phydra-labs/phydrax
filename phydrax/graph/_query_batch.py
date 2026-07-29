#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, Sequence

import equinox as eqx
import jax.numpy as jnp

from ._geometry import mollified_kernel_weight, MollifierKind, QueryGraph
from ._ir import GraphIR


class QueryNeighborhood(eqx.Module):
    """Fixed-capacity, case-local source neighborhoods for target points."""

    indices: jnp.ndarray
    relative: jnp.ndarray
    distance: jnp.ndarray
    distance_squared: jnp.ndarray
    mask: jnp.ndarray
    count: jnp.ndarray


def _case_shape(
    source: jnp.ndarray,
    target: jnp.ndarray,
    /,
) -> tuple[int, ...]:
    source_cases = tuple(int(size) for size in source.shape[:-2])
    target_cases = tuple(int(size) for size in target.shape[:-2])
    if source_cases and target_cases and source_cases != target_cases:
        raise ValueError(
            "Source and target coordinates must have one shared case shape; "
            f"got {source_cases} and {target_cases}."
        )
    return source_cases or target_cases


def _point_array(
    name: str,
    value: Any,
    case_shape: tuple[int, ...],
    /,
    *,
    coord_dim: int | None = None,
) -> jnp.ndarray:
    array = jnp.asarray(value, dtype=float)
    if array.ndim < 2:
        raise ValueError(f"{name} must end in (num_points, coord_dim).")
    if int(array.shape[-2]) <= 0 or int(array.shape[-1]) <= 0:
        raise ValueError(f"{name} point and coordinate dimensions must be positive.")
    if coord_dim is not None and int(array.shape[-1]) != int(coord_dim):
        raise ValueError(
            f"{name} coordinate dimension must be {coord_dim}; got {array.shape[-1]}."
        )
    explicit = tuple(int(size) for size in array.shape[:-2])
    if explicit and explicit != case_shape:
        raise ValueError(f"{name} case shape must be {case_shape}; got {explicit}.")
    return jnp.broadcast_to(array, case_shape + array.shape[-2:])


def _point_mask(
    name: str,
    value: Any | None,
    case_shape: tuple[int, ...],
    point_count: int,
    /,
) -> jnp.ndarray:
    target = case_shape + (int(point_count),)
    if value is None:
        return jnp.ones(target, dtype=bool)
    array = jnp.asarray(value, dtype=bool)
    explicit = tuple(int(size) for size in array.shape)
    if explicit not in ((int(point_count),), target):
        raise ValueError(
            f"{name} must have shape {(point_count,)} or {target}; got {explicit}."
        )
    return jnp.broadcast_to(array, target)


def _point_scalar(
    name: str,
    value: Any | None,
    case_shape: tuple[int, ...],
    point_count: int,
    /,
) -> jnp.ndarray | None:
    if value is None:
        return None
    target = case_shape + (int(point_count),)
    array = jnp.asarray(value, dtype=float)
    explicit = tuple(int(size) for size in array.shape)
    if explicit not in ((int(point_count),), target):
        raise ValueError(
            f"{name} must have shape {(point_count,)} or {target}; got {explicit}."
        )
    return jnp.broadcast_to(array, target)


def _point_features(
    name: str,
    value: Any | None,
    case_shape: tuple[int, ...],
    point_count: int,
    /,
) -> jnp.ndarray | None:
    if value is None:
        return None
    array = jnp.asarray(value)
    case_ndim = len(case_shape)
    if array.ndim < 1:
        raise ValueError(f"{name} must have a leading point axis.")
    if tuple(int(size) for size in array.shape[:case_ndim]) == case_shape:
        point_axis = case_ndim
    elif int(array.shape[0]) == int(point_count):
        point_axis = 0
    else:
        raise ValueError(f"{name} does not contain the expected point axis.")
    if int(array.shape[point_axis]) != int(point_count):
        raise ValueError(f"{name} point axis must have size {point_count}.")
    trailing = tuple(int(size) for size in array.shape[point_axis + 1 :])
    return jnp.broadcast_to(array, case_shape + (int(point_count),) + trailing)


def _periodic_data(
    periodic_lengths: Sequence[float | None] | None,
    coord_dim: int,
    /,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if periodic_lengths is None:
        return jnp.ones((coord_dim,), dtype=float), jnp.zeros((coord_dim,), dtype=bool)
    lengths = tuple(periodic_lengths)
    if len(lengths) != int(coord_dim):
        raise ValueError(
            f"periodic_lengths must have {coord_dim} entries; got {len(lengths)}."
        )
    if any(value is not None and float(value) <= 0.0 for value in lengths):
        raise ValueError("Periodic lengths must be positive when supplied.")
    periodic = jnp.asarray([value is not None for value in lengths], dtype=bool)
    safe_lengths = jnp.asarray(
        [1.0 if value is None else float(value) for value in lengths], dtype=float
    )
    return safe_lengths, periodic


def _minimum_image(
    relative: jnp.ndarray,
    lengths: jnp.ndarray,
    periodic: jnp.ndarray,
    /,
) -> jnp.ndarray:
    wrapped = relative - jnp.round(relative / lengths) * lengths
    return jnp.where(periodic, wrapped, relative)


def _query_neighbors_block(
    source: jnp.ndarray,
    target: jnp.ndarray,
    source_mask: jnp.ndarray,
    target_mask: jnp.ndarray,
    /,
    *,
    neighbors: int,
    radius: float | None,
    periodic_lengths: Sequence[float | None] | None,
    exclude_self: bool,
) -> QueryNeighborhood:
    coord_dim = int(source.shape[-1])
    lengths, periodic = _periodic_data(periodic_lengths, coord_dim)
    relative = target[:, :, None, :] - source[:, None, :, :]
    relative = _minimum_image(relative, lengths, periodic)
    distance_squared = jnp.sum(relative * relative, axis=-1)
    valid = source_mask[:, None, :] & target_mask[:, :, None]
    if radius is not None:
        valid = valid & (distance_squared <= float(radius) ** 2)
    if exclude_self:
        if int(source.shape[1]) != int(target.shape[1]):
            raise ValueError(
                "exclude_self requires equal source and target point counts."
            )
        diagonal = jnp.eye(int(source.shape[1]), dtype=bool)[None, :, :]
        valid = valid & ~diagonal

    sortable = jnp.where(valid, distance_squared, jnp.inf)
    indices = jnp.argsort(sortable, axis=-1, stable=True)[..., :neighbors]
    selected_distance_squared = jnp.take_along_axis(distance_squared, indices, axis=2)
    selected_mask = jnp.take_along_axis(valid, indices, axis=2)
    selected_relative = jnp.take_along_axis(
        relative,
        indices[..., None],
        axis=2,
    )
    selected_distance_squared = jnp.where(
        selected_mask,
        selected_distance_squared,
        jnp.zeros_like(selected_distance_squared),
    )
    selected_relative = jnp.where(
        selected_mask[..., None], selected_relative, jnp.zeros_like(selected_relative)
    )
    return QueryNeighborhood(
        indices=indices.astype(jnp.int32),
        relative=selected_relative,
        distance=jnp.sqrt(jnp.maximum(selected_distance_squared, 0.0)),
        distance_squared=selected_distance_squared,
        mask=selected_mask,
        count=jnp.sum(selected_mask, axis=-1, dtype=jnp.int32),
    )


def query_neighbors(
    source_points: Any,
    target_points: Any,
    /,
    *,
    source_mask: Any | None = None,
    target_mask: Any | None = None,
    max_neighbors: int | None = None,
    radius: float | None = None,
    periodic_lengths: Sequence[float | None] | None = None,
    exclude_self: bool = False,
    target_chunk_size: int | None = None,
) -> QueryNeighborhood:
    """Return deterministic, fixed-capacity neighbors without leaving JAX."""
    source_raw = jnp.asarray(source_points, dtype=float)
    target_raw = jnp.asarray(target_points, dtype=float)
    case_shape = _case_shape(source_raw, target_raw)
    source = _point_array("source_points", source_raw, case_shape)
    target = _point_array(
        "target_points", target_raw, case_shape, coord_dim=int(source.shape[-1])
    )
    source_count = int(source.shape[-2])
    target_count = int(target.shape[-2])
    neighbor_count = source_count if max_neighbors is None else int(max_neighbors)
    if neighbor_count <= 0 or neighbor_count > source_count:
        raise ValueError(
            f"max_neighbors must be in [1, {source_count}]; got {neighbor_count}."
        )
    if radius is not None and float(radius) <= 0.0:
        raise ValueError("radius must be positive when supplied.")
    if target_chunk_size is not None and int(target_chunk_size) <= 0:
        raise ValueError("target_chunk_size must be positive when supplied.")

    cases = prod(case_shape) if case_shape else 1
    source = source.reshape((cases, source_count, int(source.shape[-1])))
    target = target.reshape((cases, target_count, int(target.shape[-1])))
    source_valid = _point_mask(
        "source_mask", source_mask, case_shape, source_count
    ).reshape((cases, source_count))
    target_valid = _point_mask(
        "target_mask", target_mask, case_shape, target_count
    ).reshape((cases, target_count))

    chunk_size = target_count if target_chunk_size is None else int(target_chunk_size)
    blocks = []
    for start in range(0, target_count, chunk_size):
        stop = min(start + chunk_size, target_count)
        blocks.append(
            _query_neighbors_block(
                source,
                target[:, start:stop],
                source_valid,
                target_valid[:, start:stop],
                neighbors=neighbor_count,
                radius=radius,
                periodic_lengths=periodic_lengths,
                exclude_self=exclude_self,
            )
        )
    return QueryNeighborhood(
        indices=jnp.concatenate([block.indices for block in blocks], axis=1),
        relative=jnp.concatenate([block.relative for block in blocks], axis=1),
        distance=jnp.concatenate([block.distance for block in blocks], axis=1),
        distance_squared=jnp.concatenate(
            [block.distance_squared for block in blocks], axis=1
        ),
        mask=jnp.concatenate([block.mask for block in blocks], axis=1),
        count=jnp.concatenate([block.count for block in blocks], axis=1),
    )


def _combined_features(
    source_features: jnp.ndarray | None,
    target_features: jnp.ndarray | None,
    case_count: int,
    source_count: int,
    target_count: int,
    /,
) -> jnp.ndarray | None:
    if source_features is None and target_features is None:
        return None
    template = source_features if source_features is not None else target_features
    if template is None:
        return None
    trailing = template.shape[2:]
    if source_features is None:
        source_features = jnp.zeros(
            (case_count, source_count) + trailing, dtype=template.dtype
        )
    if target_features is None:
        target_features = jnp.zeros(
            (case_count, target_count) + trailing, dtype=template.dtype
        )
    if source_features.shape[2:] != target_features.shape[2:]:
        raise ValueError("Source and target features must share trailing shape.")
    combined = jnp.concatenate((source_features, target_features), axis=1)
    return combined.reshape((case_count * (source_count + target_count),) + trailing)


def batched_knn_query_graph(
    source_points: Any,
    target_points: Any,
    /,
    *,
    k: int,
    source_mask: Any | None = None,
    target_mask: Any | None = None,
    source_features: Any | None = None,
    target_features: Any | None = None,
    source_measure: Any | None = None,
    source_measure_key: str = "quadrature_weight",
    radius: float | None = None,
    periodic_lengths: Sequence[float | None] | None = None,
    target_chunk_size: int | None = None,
    weight_kind: MollifierKind | None = None,
    weight_radius: float | None = None,
    source_type: int = 0,
    target_type: int = 1,
    query_edge_type: int = 0,
    validate: bool = True,
) -> QueryGraph:
    """Build a JIT-compatible counts-first batch of bipartite query graphs."""
    source_raw = jnp.asarray(source_points, dtype=float)
    target_raw = jnp.asarray(target_points, dtype=float)
    case_shape = _case_shape(source_raw, target_raw)
    source = _point_array("source_points", source_raw, case_shape)
    target = _point_array(
        "target_points", target_raw, case_shape, coord_dim=int(source.shape[-1])
    )
    source_count = int(source.shape[-2])
    target_count = int(target.shape[-2])
    coord_dim = int(source.shape[-1])
    case_count = prod(case_shape) if case_shape else 1
    source = source.reshape((case_count, source_count, coord_dim))
    target = target.reshape((case_count, target_count, coord_dim))
    source_valid = _point_mask(
        "source_mask", source_mask, case_shape, source_count
    ).reshape((case_count, source_count))
    target_valid = _point_mask(
        "target_mask", target_mask, case_shape, target_count
    ).reshape((case_count, target_count))
    measure = _point_scalar("source_measure", source_measure, case_shape, source_count)
    if measure is not None:
        measure = measure.reshape((case_count, source_count))
    source_payload = _point_features(
        "source_features", source_features, case_shape, source_count
    )
    target_payload = _point_features(
        "target_features", target_features, case_shape, target_count
    )
    if source_payload is not None:
        source_payload = source_payload.reshape(
            (case_count, source_count) + source_payload.shape[len(case_shape) + 1 :]
        )
    if target_payload is not None:
        target_payload = target_payload.reshape(
            (case_count, target_count) + target_payload.shape[len(case_shape) + 1 :]
        )

    neighborhood = query_neighbors(
        source,
        target,
        source_mask=source_valid,
        target_mask=target_valid,
        max_neighbors=int(k),
        radius=radius,
        periodic_lengths=periodic_lengths,
        target_chunk_size=target_chunk_size,
    )
    node_count = source_count + target_count
    edge_count = target_count * int(k)
    offsets = jnp.arange(case_count, dtype=jnp.int32)[:, None] * node_count
    source_nodes = offsets + jnp.arange(source_count, dtype=jnp.int32)[None, :]
    target_nodes = (
        offsets + source_count + jnp.arange(target_count, dtype=jnp.int32)[None, :]
    )
    senders = offsets[:, :, None] + neighborhood.indices
    receivers = jnp.broadcast_to(
        target_nodes[:, :, None], (case_count, target_count, int(k))
    )

    positions = jnp.concatenate((source, target), axis=1).reshape(
        (case_count * node_count, coord_dim)
    )
    node_mask = jnp.concatenate((source_valid, target_valid), axis=1).reshape((-1,))
    local_index = jnp.broadcast_to(
        jnp.concatenate(
            (
                jnp.arange(source_count, dtype=jnp.int32),
                jnp.arange(target_count, dtype=jnp.int32),
            )
        )[None, :],
        (case_count, node_count),
    ).reshape((-1,))
    is_source = jnp.broadcast_to(
        jnp.concatenate(
            (
                jnp.ones((source_count,), dtype=bool),
                jnp.zeros((target_count,), dtype=bool),
            )
        )[None, :],
        (case_count, node_count),
    ).reshape((-1,))
    nodes: dict[str, Any] = {
        "positions": positions,
        "type": jnp.where(is_source, int(source_type), int(target_type)).astype(
            jnp.int32
        ),
        "local_index": local_index,
        "is_source": is_source,
        "is_target": ~is_source,
    }
    features = _combined_features(
        source_payload,
        target_payload,
        case_count,
        source_count,
        target_count,
    )
    if features is not None:
        nodes["features"] = features
    if measure is not None:
        nodes[str(source_measure_key)] = jnp.concatenate(
            (measure, jnp.zeros((case_count, target_count), dtype=measure.dtype)), axis=1
        ).reshape((-1,))

    relative = neighborhood.relative.reshape((case_count * edge_count, coord_dim))
    distance = neighborhood.distance.reshape((case_count * edge_count, 1))
    scale = (
        float(radius)
        if radius is not None
        else (float(weight_radius) if weight_radius is not None else 1.0)
    )
    edges: dict[str, Any] = {
        "type": jnp.full(
            (case_count * edge_count,), int(query_edge_type), dtype=jnp.int32
        ),
        "source_index": neighborhood.indices.reshape((-1,)),
        "target_index": jnp.broadcast_to(
            jnp.arange(target_count, dtype=jnp.int32)[None, :, None],
            (case_count, target_count, int(k)),
        ).reshape((-1,)),
        "relative": relative,
        "scaled_relative": relative / scale,
        "distance": distance,
        "distance_squared": neighborhood.distance_squared.reshape(
            (case_count * edge_count, 1)
        ),
        "unit": jnp.where(distance > 0.0, relative / distance, jnp.zeros_like(relative)),
    }
    if weight_kind is not None:
        kernel_radius = scale if weight_radius is None else float(weight_radius)
        if kernel_radius <= 0.0:
            raise ValueError("weight_radius must be positive when supplied.")
        edges["kernel_weight"] = mollified_kernel_weight(
            distance, kernel_radius, kind=weight_kind
        )

    graph = GraphIR(
        nodes=nodes,
        edges=edges,
        senders=senders.reshape((-1,)),
        receivers=receivers.reshape((-1,)),
        n_node=jnp.full((case_count,), node_count, dtype=jnp.int32),
        n_edge=jnp.full((case_count,), edge_count, dtype=jnp.int32),
        node_mask=node_mask,
        edge_mask=neighborhood.mask.reshape((-1,)),
        graph_mask=jnp.ones((case_count,), dtype=bool),
        validate=validate,
    )
    return QueryGraph(
        graph,
        source_nodes=source_nodes.reshape((-1,)),
        target_nodes=target_nodes.reshape((-1,)),
        query_edges=jnp.arange(case_count * edge_count, dtype=jnp.int32),
        source_type=source_type,
        target_type=target_type,
        query_edge_type=query_edge_type,
    )


def batched_knn_graph(
    points: Any,
    /,
    *,
    k: int,
    node_mask: Any | None = None,
    node_features: Any | None = None,
    node_measure: Any | None = None,
    node_measure_key: str = "quadrature_weight",
    radius: float | None = None,
    periodic_lengths: Sequence[float | None] | None = None,
    include_self: bool = False,
    target_chunk_size: int | None = None,
    validate: bool = True,
) -> GraphIR:
    """Build a JIT-compatible counts-first batch of homogeneous KNN graphs."""
    raw = jnp.asarray(points, dtype=float)
    case_shape = tuple(int(size) for size in raw.shape[:-2])
    point_array = _point_array("points", raw, case_shape)
    point_count = int(point_array.shape[-2])
    coord_dim = int(point_array.shape[-1])
    case_count = prod(case_shape) if case_shape else 1
    point_array = point_array.reshape((case_count, point_count, coord_dim))
    valid = _point_mask("node_mask", node_mask, case_shape, point_count).reshape(
        (case_count, point_count)
    )
    payload = _point_features("node_features", node_features, case_shape, point_count)
    if payload is not None:
        payload = payload.reshape(
            (case_count, point_count) + payload.shape[len(case_shape) + 1 :]
        )
    measure = _point_scalar("node_measure", node_measure, case_shape, point_count)
    if measure is not None:
        measure = measure.reshape((case_count, point_count))
    neighborhood = query_neighbors(
        point_array,
        point_array,
        source_mask=valid,
        target_mask=valid,
        max_neighbors=int(k),
        radius=radius,
        periodic_lengths=periodic_lengths,
        exclude_self=not include_self,
        target_chunk_size=target_chunk_size,
    )
    offsets = jnp.arange(case_count, dtype=jnp.int32)[:, None, None] * point_count
    senders = offsets + neighborhood.indices
    receivers = jnp.broadcast_to(
        offsets + jnp.arange(point_count, dtype=jnp.int32)[None, :, None],
        (case_count, point_count, int(k)),
    )
    nodes: dict[str, Any] = {
        "positions": point_array.reshape((case_count * point_count, coord_dim)),
        "local_index": jnp.broadcast_to(
            jnp.arange(point_count, dtype=jnp.int32)[None, :],
            (case_count, point_count),
        ).reshape((-1,)),
    }
    if payload is not None:
        nodes["features"] = payload.reshape(
            (case_count * point_count,) + payload.shape[2:]
        )
    if measure is not None:
        nodes[str(node_measure_key)] = measure.reshape((-1,))
    edge_count = point_count * int(k)
    relative = neighborhood.relative.reshape((case_count * edge_count, coord_dim))
    distance = neighborhood.distance.reshape((case_count * edge_count, 1))
    scale = 1.0 if radius is None else float(radius)
    edges = {
        "relative": relative,
        "scaled_relative": relative / scale,
        "distance": distance,
        "distance_squared": neighborhood.distance_squared.reshape(
            (case_count * edge_count, 1)
        ),
        "unit": jnp.where(distance > 0.0, relative / distance, jnp.zeros_like(relative)),
    }
    return GraphIR(
        nodes=nodes,
        edges=edges,
        senders=senders.reshape((-1,)),
        receivers=receivers.reshape((-1,)),
        n_node=jnp.full((case_count,), point_count, dtype=jnp.int32),
        n_edge=jnp.full((case_count,), edge_count, dtype=jnp.int32),
        node_mask=valid.reshape((-1,)),
        edge_mask=neighborhood.mask.reshape((-1,)),
        graph_mask=jnp.ones((case_count,), dtype=bool),
        validate=validate,
    )


__all__ = [
    "QueryNeighborhood",
    "batched_knn_graph",
    "batched_knn_query_graph",
    "query_neighbors",
]
