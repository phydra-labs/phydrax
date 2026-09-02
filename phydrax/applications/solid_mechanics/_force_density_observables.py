#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ._force_density import ForceDensityState
from ._force_density_topology import ForceDensityStructure


def _positive_scale(name: str, value: Any, dtype: Any, /) -> Array:
    scale = jnp.asarray(value, dtype=dtype)
    if scale.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(
        scale,
        ~jnp.isfinite(scale) | (scale <= 0.0),
        f"{name} must be finite and positive.",
    )


def member_directions(state: ForceDensityState, /) -> Array:
    """Return unit member directions with zero rows for inactive members."""
    if not isinstance(state, ForceDensityState):
        raise TypeError("state must be a ForceDensityState.")
    safe = jnp.where(state.member_valid, state.member_lengths, 1.0)
    directions = state.member_vectors / safe[:, None]
    return jnp.where(state.member_valid[:, None], directions, 0.0)


def member_direction_residual(
    state: ForceDensityState,
    targets: ArrayLike,
    /,
) -> Array:
    """Return cross-product or projected direction mismatch per member."""
    target = jnp.asarray(targets, dtype=state.positions.dtype)
    if target.shape != state.member_vectors.shape:
        raise ValueError("targets must match member vector shape.")
    target_norm = jnp.sqrt(jnp.sum(target * target, axis=-1))
    target = eqx.error_if(
        target,
        jnp.any(state.member_valid & (target_norm <= 0.0)),
        "Active direction targets must be nonzero.",
    )
    unit_target = target / jnp.where(state.member_valid, target_norm, 1.0)[:, None]
    direction = member_directions(state)
    if state.positions.shape[1] == 3:
        return jnp.cross(direction, unit_target)
    alignment = jnp.sum(direction * unit_target, axis=-1)
    return direction - alignment[:, None] * unit_target


def member_angle_residual(
    state: ForceDensityState,
    targets: ArrayLike,
    target_angles: ArrayLike,
    /,
) -> Array:
    """Return cosine-angle mismatch without inverse trigonometric singularities."""
    target = jnp.asarray(targets, dtype=state.positions.dtype)
    angles = jnp.asarray(target_angles, dtype=state.positions.dtype)
    if (
        target.shape != state.member_vectors.shape
        or angles.shape != state.member_lengths.shape
    ):
        raise ValueError("direction and angle targets must match the member axis.")
    target_norm = jnp.sqrt(jnp.sum(target * target, axis=-1))
    target = target / jnp.where(state.member_valid, target_norm, 1.0)[:, None]
    cosine = jnp.sum(member_directions(state) * target, axis=-1)
    return jnp.where(state.member_valid, cosine - jnp.cos(angles), 0.0)


def scaled_target_residual(
    values: ArrayLike,
    targets: ArrayLike,
    scale: Any,
    /,
) -> Array:
    """Return one dimensionless target residual with an explicit physical scale."""
    value = jnp.asarray(values)
    target = jnp.asarray(targets, dtype=value.dtype)
    if target.shape != value.shape:
        target = jnp.broadcast_to(target, value.shape)
    return (value - target) / _positive_scale("scale", scale, value.dtype)


def scaled_uniformity_residual(
    values: ArrayLike,
    scale: Any,
    /,
    *,
    mask: ArrayLike | None = None,
) -> Array:
    """Return deviations from the selected mean divided by an explicit scale."""
    value = jnp.asarray(values)
    selected = jnp.ones(value.shape, dtype=bool) if mask is None else jnp.asarray(mask)
    if selected.shape != value.shape:
        raise ValueError("mask must match values.")
    count = jnp.maximum(jnp.sum(selected), 1)
    mean = jnp.sum(jnp.where(selected, value, 0.0)) / count
    residual = (value - mean) / _positive_scale("scale", scale, value.dtype)
    return jnp.where(selected, residual, 0.0)


def point_target_residual(
    state: ForceDensityState,
    targets: ArrayLike,
    length_scale: Any,
    /,
    *,
    node_mask: ArrayLike | None = None,
) -> Array:
    target = jnp.asarray(targets, dtype=state.positions.dtype)
    if target.shape != state.positions.shape:
        raise ValueError("targets must match node positions.")
    selected = state.node_valid if node_mask is None else jnp.asarray(node_mask)
    if selected.shape != state.node_valid.shape:
        raise ValueError("node_mask must match the node axis.")
    residual = scaled_target_residual(state.positions, target, length_scale)
    return jnp.where((selected & state.node_valid)[:, None], residual, 0.0)


def point_line_distance(
    points: ArrayLike,
    origins: ArrayLike,
    directions: ArrayLike,
    /,
) -> Array:
    point = jnp.asarray(points)
    origin = jnp.asarray(origins, dtype=point.dtype)
    direction = jnp.asarray(directions, dtype=point.dtype)
    origin = jnp.broadcast_to(origin, point.shape)
    direction = jnp.broadcast_to(direction, point.shape)
    norm = jnp.sqrt(jnp.sum(direction * direction, axis=-1))
    direction = eqx.error_if(
        direction,
        jnp.any(norm <= 0.0),
        "Line directions must be nonzero.",
    )
    unit = direction / norm[:, None]
    offset = point - origin
    normal = offset - jnp.sum(offset * unit, axis=-1)[:, None] * unit
    return jnp.sqrt(jnp.sum(normal * normal, axis=-1))


def point_plane_signed_distance(
    points: ArrayLike,
    origins: ArrayLike,
    normals: ArrayLike,
    /,
) -> Array:
    point = jnp.asarray(points)
    origin = jnp.broadcast_to(jnp.asarray(origins, dtype=point.dtype), point.shape)
    normal = jnp.broadcast_to(jnp.asarray(normals, dtype=point.dtype), point.shape)
    norm = jnp.sqrt(jnp.sum(normal * normal, axis=-1))
    normal = eqx.error_if(normal, jnp.any(norm <= 0.0), "Plane normals must be nonzero.")
    return jnp.sum((point - origin) * normal, axis=-1) / norm


def point_segment_distance(
    points: ArrayLike,
    starts: ArrayLike,
    stops: ArrayLike,
    /,
) -> Array:
    point = jnp.asarray(points)
    start = jnp.broadcast_to(jnp.asarray(starts, dtype=point.dtype), point.shape)
    stop = jnp.broadcast_to(jnp.asarray(stops, dtype=point.dtype), point.shape)
    vector = stop - start
    squared = jnp.sum(vector * vector, axis=-1)
    vector = eqx.error_if(
        vector, jnp.any(squared <= 0.0), "Segments must be nondegenerate."
    )
    coordinate = jnp.clip(jnp.sum((point - start) * vector, axis=-1) / squared, 0.0, 1.0)
    closest = start + coordinate[:, None] * vector
    return jnp.sqrt(jnp.sum((point - closest) ** 2, axis=-1))


def reaction_direction_residual(
    state: ForceDensityState,
    targets: ArrayLike,
    force_scale: Any,
    /,
) -> Array:
    target = jnp.asarray(targets, dtype=state.positions.dtype)
    if target.shape != state.support_reactions.shape:
        raise ValueError("targets must match support reactions.")
    reaction_norm = jnp.sqrt(jnp.sum(state.support_reactions**2, axis=-1))
    target_norm = jnp.sqrt(jnp.sum(target**2, axis=-1))
    direction = state.support_reactions / jnp.maximum(reaction_norm[:, None], 1.0)
    target_direction = target / jnp.maximum(target_norm[:, None], 1.0)
    magnitude = reaction_norm / _positive_scale(
        "force_scale", force_scale, reaction_norm.dtype
    )
    return magnitude[:, None] * (direction - target_direction)


def collinearity_residual(
    positions: ArrayLike,
    triples: ArrayLike,
    length_scale: Any,
    /,
) -> Array:
    xyz = jnp.asarray(positions)
    indices = jnp.asarray(triples)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("triples must have shape (count, 3).")
    first, middle, last = xyz[indices[:, 0]], xyz[indices[:, 1]], xyz[indices[:, 2]]
    chord = last - first
    squared = jnp.sum(chord * chord, axis=-1)
    chord = eqx.error_if(
        chord, jnp.any(squared <= 0.0), "Collinearity chords must be nonzero."
    )
    coordinate = jnp.sum((middle - first) * chord, axis=-1) / squared
    projected = first + coordinate[:, None] * chord
    return (middle - projected) / _positive_scale("length_scale", length_scale, xyz.dtype)


def graph_fairness_residual(
    structure: ForceDensityStructure,
    positions: ArrayLike,
    length_scale: Any,
    /,
) -> Array:
    xyz = jnp.asarray(positions)
    if xyz.shape != (structure.node_count, structure.dimension):
        raise ValueError("positions do not match the structure.")
    valid = structure.member_valid
    sums = jnp.zeros_like(xyz)
    degree = jnp.zeros((structure.node_count,), dtype=xyz.dtype)
    neighbor_sender = jnp.where(valid[:, None], xyz[structure.receivers], 0.0)
    neighbor_receiver = jnp.where(valid[:, None], xyz[structure.senders], 0.0)
    sums = sums.at[structure.senders].add(neighbor_sender)
    sums = sums.at[structure.receivers].add(neighbor_receiver)
    degree = degree.at[structure.senders].add(valid.astype(xyz.dtype))
    degree = degree.at[structure.receivers].add(valid.astype(xyz.dtype))
    average = sums / jnp.maximum(degree[:, None], 1.0)
    residual = (xyz - average) / _positive_scale("length_scale", length_scale, xyz.dtype)
    return jnp.where((structure.node_valid & (degree > 0.0))[:, None], residual, 0.0)


def surface_cell_areas(
    structure: ForceDensityStructure,
    positions: ArrayLike,
    /,
) -> Array:
    connectivity = structure.surface_connectivity
    if connectivity is None or structure.dimension != 3:
        raise ValueError("Surface areas require 3-D polygonal connectivity.")
    xyz = jnp.asarray(positions)
    indices = jnp.where(connectivity.cell_vertex_valid, connectivity.cell_vertices, 0)
    points = xyz[indices]
    first = 0.5 * jnp.sqrt(
        jnp.sum(
            jnp.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]) ** 2,
            axis=-1,
        )
    )
    second = 0.5 * jnp.sqrt(
        jnp.sum(
            jnp.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]) ** 2,
            axis=-1,
        )
    )
    return jnp.where(connectivity.cell_kinds == 3, first, first + second)


def surface_planarity_residual(
    structure: ForceDensityStructure,
    positions: ArrayLike,
    length_scale: Any,
    /,
) -> Array:
    connectivity = structure.surface_connectivity
    if connectivity is None or structure.dimension != 3:
        raise ValueError("Planarity requires 3-D polygonal connectivity.")
    xyz = jnp.asarray(positions)
    scale = _positive_scale("length_scale", length_scale, xyz.dtype)
    indices = jnp.where(connectivity.cell_vertex_valid, connectivity.cell_vertices, 0)
    points = xyz[indices]
    volume = ein.contract(
        "cd,cd->c",
        points[:, 3] - points[:, 0],
        jnp.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
    )
    return jnp.where(connectivity.cell_kinds == 4, volume / scale**3, 0.0)


def surface_rectangularity_residual(
    structure: ForceDensityStructure,
    positions: ArrayLike,
    length_scale: Any,
    /,
) -> Array:
    connectivity = structure.surface_connectivity
    if connectivity is None or structure.dimension != 3:
        raise ValueError("Rectangularity requires 3-D polygonal connectivity.")
    xyz = jnp.asarray(positions)
    scale = _positive_scale("length_scale", length_scale, xyz.dtype)
    indices = jnp.where(connectivity.cell_vertex_valid, connectivity.cell_vertices, 0)
    points = xyz[indices]
    edges = jnp.roll(points, -1, axis=1) - points
    adjacent = jnp.sum(edges * jnp.roll(edges, -1, axis=1), axis=-1)
    return jnp.where((connectivity.cell_kinds == 4)[:, None], adjacent / scale**2, 0.0)


def target_geometry_residual(
    state: ForceDensityState,
    signed_distance: Callable[[Array], Array],
    length_scale: Any,
    /,
    *,
    node_mask: ArrayLike | None = None,
) -> Array:
    if not callable(signed_distance):
        raise TypeError("signed_distance must be callable.")
    distance = jnp.asarray(signed_distance(state.positions))
    if distance.shape != state.node_valid.shape:
        raise ValueError("signed_distance must return one value per node.")
    selected = state.node_valid if node_mask is None else jnp.asarray(node_mask)
    return jnp.where(
        selected & state.node_valid,
        distance / _positive_scale("length_scale", length_scale, distance.dtype),
        0.0,
    )


__all__ = [
    "collinearity_residual",
    "graph_fairness_residual",
    "member_angle_residual",
    "member_direction_residual",
    "member_directions",
    "point_line_distance",
    "point_plane_signed_distance",
    "point_segment_distance",
    "point_target_residual",
    "reaction_direction_residual",
    "scaled_target_residual",
    "scaled_uniformity_residual",
    "surface_cell_areas",
    "surface_planarity_residual",
    "surface_rectangularity_residual",
    "target_geometry_residual",
]
