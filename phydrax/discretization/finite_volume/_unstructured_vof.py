#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_complex import PolygonalConnectivity
from ._cell_polynomial import PreparedCellPolynomialReconstruction
from ._geometry_protocol import FiniteVolumeStageMetrics
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_embedded_boundary import (
    _clip_positive_polygon,
    _polygon_measure_centroid,
    EmbeddedBoundaryMetrics,
)


class PLICInterfaceStatus(IntEnum):
    """Host-certified status of a per-cell PLIC interface."""

    EMPTY = 0
    FULL = 1
    INTERFACE = 2
    AMBIGUOUS = 3


class PLICReconstruction(StrictModule, NonTrainableState):
    normals: Array
    offsets: Array
    interface_endpoints: Array
    interface_centers: Array
    interface_measures: Array
    interface_active: Array
    owner_phase_aperture: Array
    receptor_phase_aperture: Array
    owner_phase_centroid: Array
    receptor_phase_centroid: Array
    interface_status: Array
    interface_evidence: Array
    geometry_id: str = eqx.field(static=True)
    volume_fraction_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)

    @property
    def owner_phase_apertures(self) -> Array:
        return self.owner_phase_aperture

    @property
    def receptor_phase_apertures(self) -> Array:
        return self.receptor_phase_aperture

    @property
    def owner_phase_centroids(self) -> Array:
        return self.owner_phase_centroid

    @property
    def receptor_phase_centroids(self) -> Array:
        return self.receptor_phase_centroid

    @property
    def owner_phase0_aperture(self) -> Array:
        return self.owner_phase_aperture[:, 0]

    @property
    def owner_phase1_aperture(self) -> Array:
        return self.owner_phase_aperture[:, 1]

    @property
    def receptor_phase0_aperture(self) -> Array:
        return self.receptor_phase_aperture[:, 0]

    @property
    def receptor_phase1_aperture(self) -> Array:
        return self.receptor_phase_aperture[:, 1]


class PLICFaceApertures(StrictModule, NonTrainableState):
    """Prepared owner/receptor phase geometry for one certified stage."""

    face_ids: Array
    owner_cells: Array
    receptor_cells: Array
    active_mask: Array
    owner_phase_apertures: Array
    receptor_phase_apertures: Array
    owner_phase_centroids: Array
    receptor_phase_centroids: Array
    plan_id: str = eqx.field(static=True)
    plic_reconstruction_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    volume_fraction_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    topology_epoch_id: str = eqx.field(static=True)
    apertures_id: str = eqx.field(static=True)

    @property
    def owner_phase_aperture(self) -> Array:
        return self.owner_phase_apertures

    @property
    def receptor_phase_aperture(self) -> Array:
        return self.receptor_phase_apertures

    @property
    def owner_phase_centroid(self) -> Array:
        return self.owner_phase_centroids

    @property
    def receptor_phase_centroid(self) -> Array:
        return self.receptor_phase_centroids

    @property
    def owner_phase0_aperture(self) -> Array:
        return self.owner_phase_apertures[:, 0]

    @property
    def owner_phase1_aperture(self) -> Array:
        return self.owner_phase_apertures[:, 1]

    @property
    def receptor_phase0_aperture(self) -> Array:
        return self.receptor_phase_apertures[:, 0]

    @property
    def receptor_phase1_aperture(self) -> Array:
        return self.receptor_phase_apertures[:, 1]


class JAXPLICStageReconstruction(StrictModule, NonTrainableState):
    """Trace-safe PLIC geometry reconstructed from one stage volume fraction."""

    volume_fraction: Array
    normals: Array
    offsets: Array
    reconstructed_volume_fraction: Array
    volume_residual: Array
    interface_endpoints: Array
    interface_centers: Array
    interface_measures: Array
    interface_active: Array
    interface_status: Array
    interface_evidence: Array
    face_ids: Array
    owner_cells: Array
    receptor_cells: Array
    open_face_active: Array
    owner_phase_apertures: Array
    receptor_phase_apertures: Array
    owner_phase_centroids: Array
    receptor_phase_centroids: Array
    aperture_ids: Array
    geometry_version: Array
    plan_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    physical_layout_id: str = eqx.field(static=True)
    effective_geometry_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)

    @property
    def volume_fraction_id(self) -> Array:
        """Lossless dynamic identity of the stage alpha data."""

        return self.volume_fraction

    @property
    def apertures_id(self) -> Array:
        """Lossless per-route identity of the alpha data used by each aperture."""

        return self.aperture_ids

    @property
    def owner_phase_aperture(self) -> Array:
        return self.owner_phase_apertures

    @property
    def receptor_phase_aperture(self) -> Array:
        return self.receptor_phase_apertures

    @property
    def owner_phase_centroid(self) -> Array:
        return self.owner_phase_centroids

    @property
    def receptor_phase_centroid(self) -> Array:
        return self.receptor_phase_centroids

    @property
    def owner_phase0_aperture(self) -> Array:
        return self.owner_phase_apertures[:, 0]

    @property
    def owner_phase1_aperture(self) -> Array:
        return self.owner_phase_apertures[:, 1]

    @property
    def receptor_phase0_aperture(self) -> Array:
        return self.receptor_phase_apertures[:, 0]

    @property
    def receptor_phase1_aperture(self) -> Array:
        return self.receptor_phase_apertures[:, 1]


_JAX_CLIPPED_VERTEX_CAPACITY = 8
_JAX_INTERSECTION_CAPACITY = 4


def _jax_polygon_measure_centroid(points: Array, count: Array, /) -> tuple[Array, Array]:
    """Signed-shoelace measure and centroid of one compact padded polygon."""

    dtype = points.dtype
    safe_count = jnp.maximum(count, jnp.asarray(1, dtype=jnp.int32))

    def accumulate(index, values):
        twice_measure, numerator = values
        active = index < count
        following = jnp.where(index + 1 < count, index + 1, 0)
        first = points[index]
        second = points[following]
        cross = first[0] * second[1] - first[1] * second[0]
        cross = jnp.where(active, cross, jnp.asarray(0.0, dtype=dtype))
        return (
            twice_measure + cross,
            numerator + cross * (first + second),
        )

    twice_measure, numerator = jax.lax.fori_loop(
        0,
        points.shape[0],
        accumulate,
        (
            jnp.asarray(0.0, dtype=dtype),
            jnp.zeros((2,), dtype=dtype),
        ),
    )
    measure = 0.5 * jnp.abs(twice_measure)
    vertex_mask = jnp.arange(points.shape[0]) < count
    mean = jnp.sum(jnp.where(vertex_mask[:, None], points, 0.0), axis=0) / (
        safe_count.astype(dtype)
    )
    safe_twice_measure = jnp.where(
        jnp.abs(twice_measure) > jnp.finfo(dtype).tiny,
        twice_measure,
        jnp.asarray(1.0, dtype=dtype),
    )
    centroid = jnp.where(
        (count >= 3) & (measure > jnp.finfo(dtype).tiny),
        numerator / (3.0 * safe_twice_measure),
        mean,
    )
    return measure, centroid


def _jax_clip_convex_polygon(
    vertices: Array,
    vertex_valid: Array,
    normal: Array,
    offset: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Clip one compact padded convex polygon by ``normal·x <= offset``."""

    dtype = vertices.dtype
    arity = jnp.sum(vertex_valid.astype(jnp.int32))
    clipped = jnp.zeros((_JAX_CLIPPED_VERTEX_CAPACITY, 2), dtype=dtype)
    intersections = jnp.zeros((_JAX_INTERSECTION_CAPACITY, 2), dtype=dtype)

    def edge(index, state):
        output, output_count, crossings, crossing_count = state

        def process(active_state):
            output_, output_count_, crossings_, crossing_count_ = active_state
            following = jnp.mod(index + 1, jnp.maximum(arity, 1))
            first = vertices[index]
            second = vertices[following]
            first_signed = offset - oe.contract("d,d->", normal, first)
            second_signed = offset - oe.contract("d,d->", normal, second)
            first_inside = first_signed >= 0.0
            second_inside = second_signed >= 0.0
            crosses = first_inside != second_inside
            denominator = first_signed - second_signed
            safe_denominator = jnp.where(
                jnp.abs(denominator) > jnp.finfo(dtype).tiny,
                denominator,
                jnp.asarray(1.0, dtype=dtype),
            )
            fraction = jnp.clip(first_signed / safe_denominator, 0.0, 1.0)
            intersection = first + fraction * (second - first)

            def append_crossing(carry):
                output__, output_count__, crossings__, crossing_count__ = carry
                output__ = output__.at[output_count__].set(intersection)
                crossings__ = crossings__.at[crossing_count__].set(intersection)
                return (
                    output__,
                    output_count__ + 1,
                    crossings__,
                    crossing_count__ + 1,
                )

            def append_second(carry):
                output__, output_count__, crossings__, crossing_count__ = carry
                output__ = output__.at[output_count__].set(second)
                return (
                    output__,
                    output_count__ + 1,
                    crossings__,
                    crossing_count__,
                )

            crossed_state = jax.lax.cond(
                crosses,
                append_crossing,
                lambda carry: carry,
                (output_, output_count_, crossings_, crossing_count_),
            )
            return jax.lax.cond(
                second_inside,
                append_second,
                lambda carry: carry,
                crossed_state,
            )

        return jax.lax.cond(
            (index < arity) & vertex_valid[index],
            process,
            lambda active_state: active_state,
            (output, output_count, crossings, crossing_count),
        )

    return jax.lax.fori_loop(
        0,
        vertices.shape[0],
        edge,
        (
            clipped,
            jnp.asarray(0, dtype=jnp.int32),
            intersections,
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )


def _jax_clipped_measure(
    vertices: Array,
    vertex_valid: Array,
    normal: Array,
    offset: Array,
    /,
) -> Array:
    clipped, count, _, _ = _jax_clip_convex_polygon(
        vertices, vertex_valid, normal, offset
    )
    measure, _ = _jax_polygon_measure_centroid(clipped, count)
    return measure


def _jax_convex_polygon_evidence(
    vertices: Array,
    vertex_valid: Array,
    expected_measure: Array,
    /,
) -> Array:
    """Certify one compact padded convex polygon or exact inactive zero."""

    dtype = vertices.dtype
    arity = jnp.sum(vertex_valid.astype(jnp.int32))
    compact_layout = jnp.all(vertex_valid == (jnp.arange(vertices.shape[0]) < arity))
    edge_lengths = jnp.zeros((vertices.shape[0],), dtype=dtype)
    turns = jnp.zeros((vertices.shape[0],), dtype=dtype)

    def inspect(index, values):
        lengths, local_turns = values
        first = vertices[index]
        second_index = jnp.mod(index + 1, jnp.maximum(arity, 1))
        third_index = jnp.mod(index + 2, jnp.maximum(arity, 1))
        second = vertices[second_index]
        third = vertices[third_index]
        first_edge = second - first
        second_edge = third - second
        active = index < arity
        length = jnp.linalg.norm(first_edge)
        turn = first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
        lengths = lengths.at[index].set(jnp.where(active, length, 0.0))
        local_turns = local_turns.at[index].set(jnp.where(active, turn, 0.0))
        return lengths, local_turns

    edge_lengths, turns = jax.lax.fori_loop(
        0, vertices.shape[0], inspect, (edge_lengths, turns)
    )
    active_turn = jnp.arange(vertices.shape[0]) < arity
    length_scale = jnp.maximum(jnp.max(edge_lengths), jnp.finfo(dtype).tiny)
    turn_tolerance = 256.0 * jnp.finfo(dtype).eps * length_scale**2
    positive = jnp.all(jnp.where(active_turn, turns > turn_tolerance, True))
    negative = jnp.all(jnp.where(active_turn, turns < -turn_tolerance, True))
    measure, _ = _jax_polygon_measure_centroid(vertices, arity)
    measure_tolerance = (
        2048.0
        * jnp.finfo(dtype).eps
        * jnp.maximum(
            jnp.maximum(jnp.abs(expected_measure), measure),
            jnp.finfo(dtype).tiny,
        )
    )
    finite = jnp.all(jnp.isfinite(vertices)) & jnp.isfinite(expected_measure)
    inactive = (
        (arity == 0)
        & (expected_measure == 0.0)
        & compact_layout
        & finite
        & jnp.all(vertices == 0.0)
    )
    active = (
        (arity >= 3)
        & (arity <= vertices.shape[0])
        & compact_layout
        & finite
        & (expected_measure > 0.0)
        & jnp.all(jnp.where(vertex_valid, edge_lengths > 0.0, True))
        & (positive | negative)
        & (jnp.abs(measure - expected_measure) <= measure_tolerance)
    )
    return inactive | active


def _jax_segment_phase_geometry(
    first: Array,
    second: Array,
    normal: Array,
    offset: Array,
    /,
) -> tuple[Array, Array]:
    """Return complementary phase apertures and centroids on one face."""

    dtype = first.dtype
    first_signed = offset - oe.contract("d,d->", normal, first)
    second_signed = offset - oe.contract("d,d->", normal, second)
    first_inside = first_signed >= 0.0
    second_inside = second_signed >= 0.0
    denominator = first_signed - second_signed
    safe_denominator = jnp.where(
        jnp.abs(denominator) > jnp.finfo(dtype).tiny,
        denominator,
        jnp.asarray(1.0, dtype=dtype),
    )
    fraction = jnp.clip(first_signed / safe_denominator, 0.0, 1.0)
    crossing = first + fraction * (second - first)
    center = 0.5 * (first + second)
    phase0_fraction = jnp.where(
        first_inside & second_inside,
        1.0,
        jnp.where(
            ~first_inside & ~second_inside,
            0.0,
            jnp.where(first_inside, fraction, 1.0 - fraction),
        ),
    )
    phase0_centroid = jnp.where(
        first_inside == second_inside,
        center,
        jnp.where(first_inside, 0.5 * (first + crossing), 0.5 * (crossing + second)),
    )
    phase1_centroid = jnp.where(
        first_inside == second_inside,
        center,
        jnp.where(first_inside, 0.5 * (crossing + second), 0.5 * (first + crossing)),
    )
    apertures = jnp.stack((phase0_fraction, 1.0 - phase0_fraction))
    centroids = jnp.stack((phase0_centroid, phase1_centroid))
    return apertures, centroids


def _dot(left: np.ndarray, right: np.ndarray, /) -> float:
    return float(oe.contract("d,d->", left, right))


def _segment_phase_geometry(
    first: np.ndarray,
    second: np.ndarray,
    normal: np.ndarray,
    offset: float,
    /,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return phase-0 fraction and both phase centroids on one segment."""

    tangent = second - first
    measure = float(np.linalg.norm(tangent))
    if not np.isfinite(measure) or measure <= 0.0:
        raise ValueError("PLIC aperture segment must have positive finite measure.")
    first_value = float(offset - _dot(normal, first))
    second_value = float(offset - _dot(normal, second))
    scale = max(measure, abs(float(offset)), 1.0)
    tolerance = 256.0 * np.finfo(np.result_type(first.dtype, normal.dtype)).eps * scale
    crossing_point = None
    if max(first_value, second_value) <= tolerance:
        phase0_fraction = 0.0
        phase0_first, phase0_second = first, first
    elif min(first_value, second_value) >= -tolerance:
        phase0_fraction = 1.0
        phase0_first, phase0_second = first, second
    else:
        crossing = first_value / (first_value - second_value)
        if not np.isfinite(crossing) or crossing <= 0.0 or crossing >= 1.0:
            raise ValueError("PLIC aperture has an ambiguous interface crossing.")
        crossing_point = first + crossing * tangent
        if first_value > 0.0:
            phase0_fraction = crossing
            phase0_first, phase0_second = first, crossing_point
        else:
            phase0_fraction = 1.0 - crossing
            phase0_first, phase0_second = crossing_point, second
    phase0_fraction = float(np.clip(phase0_fraction, 0.0, 1.0))
    phase1_fraction = 1.0 - phase0_fraction
    phase0_centroid = 0.5 * (phase0_first + phase0_second)
    if crossing_point is None:
        phase1_centroid = 0.5 * (first + second)
    elif first_value > 0.0:
        phase1_centroid = 0.5 * (crossing_point + second)
    else:
        phase1_centroid = 0.5 * (first + crossing_point)
    return (
        phase0_fraction,
        np.asarray((phase0_centroid, phase1_centroid)),
        np.asarray((phase0_fraction, phase1_fraction)),
    )


def _stage_segment(block, route: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Recover exact straight-face endpoints from stage quadrature geometry."""

    points = np.asarray(block.quadrature_points)[route]
    measure = float(np.asarray(block.face_measures)[route])
    center = np.asarray(block.face_centers)[route]
    if points.shape[0] == 2:
        difference = points[1] - points[0]
        half_tangent = 0.5 * np.sqrt(3.0) * difference
        first = center - half_tangent
        second = center + half_tangent
    elif points.shape[0] == 1:
        area_vector = np.asarray(block.area_vectors)[route]
        normal = np.asarray((area_vector[1], -area_vector[0]))
        tangent_norm = float(np.linalg.norm(normal))
        if tangent_norm <= 0.0 or not np.isfinite(tangent_norm):
            raise ValueError("PLIC cut face must have a finite nonzero tangent.")
        tangent = normal / tangent_norm
        first = center - 0.5 * measure * tangent
        second = center + 0.5 * measure * tangent
    else:
        raise ValueError(
            "PLIC apertures require two-point physical or one-point cut faces."
        )
    recovered_measure = float(np.linalg.norm(second - first))
    tolerance = 1024.0 * np.finfo(points.dtype).eps * max(measure, recovered_measure, 1.0)
    if abs(recovered_measure - measure) > tolerance:
        raise ValueError("Stage face quadrature does not reproduce its face measure.")
    return np.asarray(first), np.asarray(second)


class UnstructuredVOFPlan(StrictModule, NonTrainableState):
    """Bounded upwind VOF transport with host-prepared 2-D PLIC geometry."""

    discretization: UnstructuredFiniteVolumeDiscretization
    gradient: PreparedCellPolynomialReconstruction
    bisection_iterations: int = eqx.field(static=True)
    physical_layout_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        gradient: PreparedCellPolynomialReconstruction,
        /,
        *,
        bisection_iterations: int = 60,
    ):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("VOF requires unstructured FV geometry.")
        if discretization.cell_dimension != 2 or not isinstance(
            discretization.connectivity, PolygonalConnectivity
        ):
            raise ValueError("PLIC reconstruction currently supports 2-D polygons.")
        if not isinstance(gradient, PreparedCellPolynomialReconstruction):
            raise TypeError("gradient must be PreparedCellPolynomialReconstruction.")
        if gradient.basis.degree != 1:
            raise ValueError("VOF interface normals require a degree-one gradient.")
        if gradient.discretization.prepared_id != discretization.prepared_id:
            raise ValueError("VOF gradient belongs to a different geometry.")
        if (
            int(np.asarray(gradient.report.minimum_rank)) != gradient.basis.feature_count
            or not np.isfinite(float(np.asarray(gradient.report.minimum_singular_value)))
            or float(np.asarray(gradient.report.minimum_singular_value)) <= 0.0
            or not np.isfinite(
                float(np.asarray(gradient.report.maximum_condition_number))
            )
        ):
            raise ValueError("VOF gradient rank evidence is uncertain.")
        iterations = int(bisection_iterations)
        if iterations < 16:
            raise ValueError("bisection_iterations must be at least 16.")
        connectivity = discretization.connectivity
        face_count = int(discretization.face_measures.size)
        if (
            connectivity.cell_vertices.shape != (discretization.cell_count, 4)
            or connectivity.cell_vertex_valid.shape != (discretization.cell_count, 4)
            or connectivity.cell_kinds.shape != (discretization.cell_count,)
            or connectivity.edges.shape != (face_count, 2)
            or discretization.owner_cells.shape != (face_count,)
            or discretization.neighbour_cells.shape != (face_count,)
        ):
            raise ValueError("PLIC polygon or physical face layout is invalid.")
        physical_layout_id = canonical_fingerprint(
            {
                "kind": "jax-plic-physical-face-layout",
                "topology": discretization.topology_id,
                "face_ids": array_tree_fingerprint(np.arange(face_count, dtype=np.int32)),
                "owner_cells": array_tree_fingerprint(discretization.owner_cells),
                "receptor_cells": array_tree_fingerprint(discretization.neighbour_cells),
                "edges": array_tree_fingerprint(connectivity.edges),
            }
        )
        self.discretization = discretization
        self.gradient = gradient
        self.bisection_iterations = iterations
        self.physical_layout_id = physical_layout_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-vof-plan",
                "geometry": discretization.prepared_id,
                "gradient": gradient.prepared_id,
                "bisection_iterations": iterations,
                "physical_layout": physical_layout_id,
            }
        )

    def validate_volume_fraction(self, volume_fraction: ArrayLike, /) -> Array:
        alpha = jnp.asarray(volume_fraction)
        if alpha.shape != (self.discretization.cell_count,):
            raise ValueError("Volume fraction must contain one value per cell.")
        return eqx.error_if(
            alpha,
            jnp.any(~jnp.isfinite(alpha) | (alpha < 0.0) | (alpha > 1.0)),
            "Volume fraction must be finite and lie in [0, 1].",
        )

    def interface_normals(self, volume_fraction: ArrayLike, /) -> Array:
        alpha = self.validate_volume_fraction(volume_fraction)
        coefficients = self.gradient.coefficients(alpha)
        lengths = self.gradient.characteristic_lengths.astype(alpha.dtype)
        alpha_gradient = coefficients / lengths[:, None]
        magnitude = jnp.linalg.norm(alpha_gradient, axis=-1)
        fallback = jnp.broadcast_to(
            jnp.asarray((1.0, 0.0), dtype=alpha.dtype), alpha_gradient.shape
        )
        return jnp.where(
            (magnitude > 64.0 * jnp.finfo(alpha.dtype).eps)[:, None],
            -alpha_gradient / jnp.maximum(magnitude[:, None], 1e-30),
            fallback,
        )

    def reconstruct_stage(
        self,
        volume_fraction: ArrayLike,
        /,
        *,
        effective_geometry: EmbeddedBoundaryMetrics | None = None,
        geometry_layout_id: str | None = None,
        geometry_version: ArrayLike = 0,
        normal_override: ArrayLike | None = None,
        override_mask: ArrayLike | None = None,
    ) -> JAXPLICStageReconstruction:
        """Reconstruct fixed-capacity PLIC geometry from the current stage alpha."""

        alpha = self.validate_volume_fraction(volume_fraction)
        if not jnp.issubdtype(alpha.dtype, jnp.inexact):
            alpha = alpha.astype(self.discretization.cell_volumes.dtype)
        geometry = self.discretization
        connectivity = geometry.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise ValueError("Stage PLIC reconstruction requires 2-D polygons.")
        layout_id = (
            self.physical_layout_id
            if geometry_layout_id is None
            else str(geometry_layout_id)
        )
        if not layout_id:
            raise ValueError("geometry_layout_id must be non-empty.")
        version = jnp.asarray(geometry_version)
        if version.shape != () or version.dtype.kind not in "iu":
            raise ValueError("geometry_version must be a scalar integer.")
        version = eqx.error_if(
            version,
            (version < 0) | (version > jnp.iinfo(jnp.int32).max),
            "geometry_version must be nonnegative and representable as int32.",
        ).astype(jnp.int32)

        rank_evidence = (
            (self.gradient.report.minimum_rank == self.gradient.basis.feature_count)
            & jnp.isfinite(self.gradient.report.minimum_singular_value)
            & (self.gradient.report.minimum_singular_value > 0.0)
            & jnp.isfinite(self.gradient.report.maximum_condition_number)
        )
        alpha = eqx.error_if(
            alpha,
            ~rank_evidence,
            "Stage PLIC gradient rank evidence is uncertain.",
        )
        dtype = alpha.dtype
        vertices = geometry.vertices.astype(dtype)
        cell_indices = connectivity.cell_vertices.astype(jnp.int32)
        base_cell_vertex_valid = connectivity.cell_vertex_valid
        vertex_count = vertices.shape[0]
        cell_index_evidence = jnp.all(
            jnp.where(
                base_cell_vertex_valid,
                (cell_indices >= 0) & (cell_indices < vertex_count),
                True,
            )
        )
        safe_cell_indices = jnp.clip(cell_indices, 0, max(vertex_count - 1, 0))
        base_polygons = vertices[safe_cell_indices]

        face_edges = connectivity.edges.astype(jnp.int32)
        face_count = geometry.face_measures.size
        edge_index_evidence = jnp.all((face_edges >= 0) & (face_edges < vertex_count))
        safe_face_edges = jnp.clip(face_edges, 0, max(vertex_count - 1, 0))
        base_edge_points = vertices[safe_face_edges]
        owners = geometry.owner_cells.astype(jnp.int32)
        receptors = geometry.neighbour_cells.astype(jnp.int32)

        if effective_geometry is None:
            polygons = base_polygons
            cell_vertex_valid = base_cell_vertex_valid
            cell_volumes = geometry.cell_volumes.astype(dtype)
            cell_active = jnp.ones((geometry.cell_count,), dtype=jnp.bool_)
            edge_points = base_edge_points
            open_face_active = jnp.ones((face_count,), dtype=jnp.bool_)
            effective_geometry_id = geometry.prepared_id
            effective_evidence = jnp.asarray(True)
        else:
            if not isinstance(effective_geometry, EmbeddedBoundaryMetrics):
                raise TypeError(
                    "effective_geometry must be EmbeddedBoundaryMetrics or None."
                )
            if (
                effective_geometry.prepared_id != geometry.prepared_id
                or effective_geometry.topology_id != geometry.topology_id
                or effective_geometry.geometry_id != geometry.geometry_id
            ):
                raise ValueError(
                    "Effective PLIC geometry is stale for this prepared geometry."
                )
            if (
                effective_geometry.fluid_polygon_vertices.shape[0] != geometry.cell_count
                or effective_geometry.fluid_polygon_vertices.ndim != 3
                or effective_geometry.fluid_polygon_vertices.shape[-1] != 2
                or effective_geometry.fluid_polygon_valid.shape
                != effective_geometry.fluid_polygon_vertices.shape[:2]
                or effective_geometry.fluid_cell_volumes.shape != (geometry.cell_count,)
                or effective_geometry.active_fluid_cells.shape != (geometry.cell_count,)
                or effective_geometry.open_face_segment_endpoints.shape
                != (face_count, 2, 2)
                or effective_geometry.open_face_measures.shape != (face_count,)
            ):
                raise ValueError("Effective PLIC geometry has stale array shapes.")
            polygons = effective_geometry.fluid_polygon_vertices.astype(dtype)
            cell_vertex_valid = effective_geometry.fluid_polygon_valid
            cell_volumes = effective_geometry.fluid_cell_volumes.astype(dtype)
            cell_active = effective_geometry.active_fluid_cells
            edge_points = effective_geometry.open_face_segment_endpoints.astype(dtype)
            open_face_active = effective_geometry.open_face_measures > 0.0
            effective_geometry_id = effective_geometry.metrics_id
            effective_evidence = (
                effective_geometry.evidence.passed
                & (
                    effective_geometry.evidence.fluid_polygon_measure_defect
                    <= effective_geometry.evidence.fluid_polygon_measure_tolerance
                ).all()
                & (
                    effective_geometry.evidence.open_segment_measure_defect
                    <= effective_geometry.evidence.open_segment_measure_tolerance
                ).all()
            )

        polygon_evidence = jax.vmap(_jax_convex_polygon_evidence)(
            polygons,
            cell_vertex_valid,
            cell_volumes,
        )
        edge_tangents = edge_points[:, 1] - edge_points[:, 0]
        edge_lengths = jnp.linalg.norm(edge_tangents, axis=-1)
        route_evidence = (
            (face_edges.shape == (face_count, 2))
            & (owners.shape == (face_count,))
            & (receptors.shape == (face_count,))
            & (open_face_active.shape == (face_count,))
            & jnp.all((owners >= 0) & (owners < geometry.cell_count))
            & jnp.all((receptors >= -1) & (receptors < geometry.cell_count))
            & jnp.all(jnp.isfinite(edge_points))
            & jnp.all(jnp.isfinite(edge_lengths))
            & jnp.all(
                jnp.where(open_face_active, edge_lengths > 0.0, edge_lengths == 0.0)
            )
        )
        alpha = eqx.error_if(
            alpha,
            ~cell_index_evidence
            | ~edge_index_evidence
            | ~jnp.all(polygon_evidence)
            | ~route_evidence
            | ~effective_evidence,
            "Stage PLIC effective geometry is invalid or uncertain.",
        )

        normals = self.interface_normals(alpha)
        if (normal_override is None) != (override_mask is None):
            raise ValueError(
                "normal_override and override_mask must be supplied together."
            )
        if normal_override is not None and override_mask is not None:
            override = jnp.asarray(normal_override, dtype=dtype)
            mask = jnp.asarray(override_mask, dtype=jnp.bool_)
            if override.shape != normals.shape or mask.shape != alpha.shape:
                raise ValueError("PLIC normal override shapes are incompatible.")
            override_length = jnp.linalg.norm(override, axis=-1)
            override = override / jnp.maximum(override_length[:, None], 1e-30)
            normals = jnp.where(mask[:, None], override, normals)
        normal_lengths = jnp.linalg.norm(normals, axis=-1)
        normal_evidence = jnp.all(
            jnp.isfinite(normals) & jnp.isfinite(normal_lengths[:, None])
        ) & jnp.all(jnp.abs(normal_lengths - 1.0) <= 512.0 * jnp.finfo(dtype).eps)
        alpha = eqx.error_if(
            alpha,
            ~normal_evidence,
            "Stage PLIC normals are not finite unit vectors.",
        )

        projections = oe.contract("cvd,cd->cv", polygons, normals)
        lower = jnp.min(jnp.where(cell_vertex_valid, projections, jnp.inf), axis=1)
        upper = jnp.max(jnp.where(cell_vertex_valid, projections, -jnp.inf), axis=1)
        lower = jnp.where(cell_active, lower, 0.0)
        upper = jnp.where(cell_active, upper, 0.0)
        target_measure = jnp.where(cell_active, alpha * cell_volumes, 0.0)

        def bisect(_, bounds):
            lower_, upper_ = bounds
            midpoint = 0.5 * (lower_ + upper_)
            clipped_measure = jax.vmap(_jax_clipped_measure)(
                polygons,
                cell_vertex_valid,
                normals,
                midpoint,
            )
            below = clipped_measure < target_measure
            return (
                jnp.where(below, midpoint, lower_),
                jnp.where(below, upper_, midpoint),
            )

        bisected_lower, bisected_upper = jax.lax.fori_loop(
            0,
            self.bisection_iterations,
            bisect,
            (lower, upper),
        )
        mixed = cell_active & (alpha > 0.0) & (alpha < 1.0)
        offsets = 0.5 * (bisected_lower + bisected_upper)
        offsets = jnp.where(cell_active & (alpha == 0.0), lower, offsets)
        offsets = jnp.where(cell_active & (alpha == 1.0), upper, offsets)
        offsets = jnp.where(cell_active, offsets, 0.0)
        (
            clipped_polygons,
            clipped_counts,
            intersections,
            intersection_counts,
        ) = jax.vmap(_jax_clip_convex_polygon)(
            polygons,
            cell_vertex_valid,
            normals,
            offsets,
        )
        clipped_measures, _ = jax.vmap(_jax_polygon_measure_centroid)(
            clipped_polygons,
            clipped_counts,
        )
        volume_residual = jnp.abs(clipped_measures - target_measure)
        safe_cell_volumes = jnp.where(cell_active, cell_volumes, 1.0)
        reconstructed_volume_fraction = jnp.where(
            mixed,
            clipped_measures / safe_cell_volumes,
            jnp.where(cell_active, alpha, 0.0),
        )
        interface_endpoints = intersections[:, :2]
        interface_centers = 0.5 * (interface_endpoints[:, 0] + interface_endpoints[:, 1])
        interface_measures = jnp.linalg.norm(
            interface_endpoints[:, 1] - interface_endpoints[:, 0],
            axis=-1,
        )
        volume_tolerance = (
            2048.0
            * jnp.finfo(dtype).eps
            * jnp.maximum(cell_volumes, jnp.finfo(dtype).tiny)
        )
        interface_tolerance = (
            512.0
            * jnp.finfo(dtype).eps
            * jnp.maximum(jnp.sqrt(cell_volumes), jnp.finfo(dtype).tiny)
        )
        interface_evidence = (
            polygon_evidence
            & jnp.isfinite(offsets)
            & jnp.isfinite(volume_residual)
            & jnp.isfinite(reconstructed_volume_fraction)
            & (volume_residual <= volume_tolerance)
            & jnp.where(
                mixed,
                (intersection_counts == 2)
                & jnp.isfinite(interface_measures)
                & (interface_measures > interface_tolerance),
                True,
            )
        )
        interface_status = jnp.where(
            ~cell_active | (alpha == 0.0),
            int(PLICInterfaceStatus.EMPTY),
            jnp.where(
                alpha == 1.0,
                int(PLICInterfaceStatus.FULL),
                jnp.where(
                    interface_evidence,
                    int(PLICInterfaceStatus.INTERFACE),
                    int(PLICInterfaceStatus.AMBIGUOUS),
                ),
            ),
        ).astype(jnp.int32)
        interface_evidence = eqx.error_if(
            interface_evidence,
            ~jnp.all(interface_evidence),
            "Stage PLIC reconstruction has ambiguous or invalid geometry.",
        )
        interface_active = mixed & interface_evidence
        interface_endpoints = jnp.where(
            interface_active[:, None, None],
            interface_endpoints,
            jnp.zeros_like(interface_endpoints),
        )
        interface_centers = jnp.where(
            interface_active[:, None],
            interface_centers,
            jnp.zeros_like(interface_centers),
        )
        interface_measures = jnp.where(
            interface_active, interface_measures, jnp.zeros_like(interface_measures)
        )

        edge_centers = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
        owner_segment_apertures, owner_segment_centroids = jax.vmap(
            _jax_segment_phase_geometry
        )(
            edge_points[:, 0],
            edge_points[:, 1],
            normals[owners],
            offsets[owners],
        )
        owner_empty = alpha[owners] == 0.0
        owner_full = alpha[owners] == 1.0
        empty_aperture = jnp.asarray((0.0, 1.0), dtype=dtype)
        full_aperture = jnp.asarray((1.0, 0.0), dtype=dtype)
        edge_center_pairs = jnp.broadcast_to(
            edge_centers[:, None, :],
            owner_segment_centroids.shape,
        )
        owner_phase_apertures = jnp.where(
            owner_empty[:, None],
            empty_aperture,
            jnp.where(
                owner_full[:, None],
                full_aperture,
                owner_segment_apertures,
            ),
        )
        owner_phase_centroids = jnp.where(
            (owner_empty | owner_full)[:, None, None],
            edge_center_pairs,
            owner_segment_centroids,
        )

        safe_receptors = jnp.maximum(receptors, 0)
        receptor_segment_apertures, receptor_segment_centroids = jax.vmap(
            _jax_segment_phase_geometry
        )(
            edge_points[:, 0],
            edge_points[:, 1],
            normals[safe_receptors],
            offsets[safe_receptors],
        )
        receptor_empty = alpha[safe_receptors] == 0.0
        receptor_full = alpha[safe_receptors] == 1.0
        receptor_phase_apertures = jnp.where(
            receptor_empty[:, None],
            empty_aperture,
            jnp.where(
                receptor_full[:, None],
                full_aperture,
                receptor_segment_apertures,
            ),
        )
        receptor_phase_centroids = jnp.where(
            (receptor_empty | receptor_full)[:, None, None],
            edge_center_pairs,
            receptor_segment_centroids,
        )
        boundary = receptors < 0
        receptor_phase_apertures = jnp.where(
            boundary[:, None],
            owner_phase_apertures,
            receptor_phase_apertures,
        )
        receptor_phase_centroids = jnp.where(
            boundary[:, None, None],
            edge_center_pairs,
            receptor_phase_centroids,
        )
        owner_phase_apertures = jnp.where(
            open_face_active[:, None],
            owner_phase_apertures,
            jnp.zeros_like(owner_phase_apertures),
        )
        receptor_phase_apertures = jnp.where(
            open_face_active[:, None],
            receptor_phase_apertures,
            jnp.zeros_like(receptor_phase_apertures),
        )
        owner_phase_centroids = jnp.where(
            open_face_active[:, None, None],
            owner_phase_centroids,
            jnp.zeros_like(owner_phase_centroids),
        )
        receptor_phase_centroids = jnp.where(
            open_face_active[:, None, None],
            receptor_phase_centroids,
            jnp.zeros_like(receptor_phase_centroids),
        )
        aperture_evidence = (
            jnp.all(jnp.isfinite(owner_phase_apertures))
            & jnp.all(jnp.isfinite(receptor_phase_apertures))
            & jnp.all(jnp.isfinite(owner_phase_centroids))
            & jnp.all(jnp.isfinite(receptor_phase_centroids))
            & jnp.all((owner_phase_apertures >= 0.0) & (owner_phase_apertures <= 1.0))
            & jnp.all(
                (receptor_phase_apertures >= 0.0) & (receptor_phase_apertures <= 1.0)
            )
            & jnp.all(
                jnp.abs(
                    jnp.sum(owner_phase_apertures, axis=-1)
                    - open_face_active.astype(dtype)
                )
                <= 64.0 * jnp.finfo(dtype).eps
            )
            & jnp.all(
                jnp.abs(
                    jnp.sum(receptor_phase_apertures, axis=-1)
                    - open_face_active.astype(dtype)
                )
                <= 64.0 * jnp.finfo(dtype).eps
            )
        )
        owner_phase_apertures = eqx.error_if(
            owner_phase_apertures,
            ~aperture_evidence,
            "Stage PLIC face apertures are invalid.",
        )
        face_ids = jnp.arange(face_count, dtype=jnp.int32)
        return JAXPLICStageReconstruction(
            volume_fraction=alpha,
            normals=normals,
            offsets=offsets,
            reconstructed_volume_fraction=reconstructed_volume_fraction,
            volume_residual=volume_residual,
            interface_endpoints=interface_endpoints,
            interface_centers=interface_centers,
            interface_measures=interface_measures,
            interface_active=interface_active,
            interface_status=interface_status,
            interface_evidence=interface_evidence,
            face_ids=face_ids,
            owner_cells=owners,
            receptor_cells=receptors,
            open_face_active=open_face_active,
            owner_phase_apertures=owner_phase_apertures,
            receptor_phase_apertures=receptor_phase_apertures,
            owner_phase_centroids=owner_phase_centroids,
            receptor_phase_centroids=receptor_phase_centroids,
            aperture_ids=alpha,
            geometry_version=version,
            plan_id=self.plan_id,
            geometry_id=geometry.geometry_id,
            prepared_id=geometry.prepared_id,
            topology_id=geometry.topology_id,
            physical_layout_id=self.physical_layout_id,
            effective_geometry_id=effective_geometry_id,
            geometry_layout_id=layout_id,
        )

    def reconstruct(self, volume_fraction: ArrayLike, /) -> PLICReconstruction:
        """Construct exact polygon/half-plane PLIC segments on the host."""

        alpha = np.asarray(self.validate_volume_fraction(volume_fraction))
        normals = np.asarray(self.interface_normals(volume_fraction))
        geometry = self.discretization
        connectivity = geometry.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise TypeError("PLIC connectivity must be polygonal.")
        vertices = np.asarray(geometry.vertices)
        cell_vertices = np.asarray(connectivity.cell_vertices, dtype=np.int32)
        cell_kinds = np.asarray(connectivity.cell_kinds, dtype=np.int32)
        face_edges = np.asarray(connectivity.edges, dtype=np.int32)
        offsets = np.zeros((geometry.cell_count,))
        endpoints = np.zeros((geometry.cell_count, 2, 2))
        centers = np.zeros((geometry.cell_count, 2))
        measures = np.zeros((geometry.cell_count,))
        active = np.zeros((geometry.cell_count,), dtype=bool)
        status = np.full(
            (geometry.cell_count,), int(PLICInterfaceStatus.EMPTY), dtype=np.int32
        )
        evidence = np.zeros((geometry.cell_count,), dtype=bool)
        tolerance = 256.0 * np.finfo(float).eps
        for cell in range(geometry.cell_count):
            arity = int(cell_kinds[cell])
            polygon = vertices[cell_vertices[cell, :arity]]
            projection = oe.contract("vd,d->v", polygon, normals[cell])
            lower = float(np.min(projection))
            upper = float(np.max(projection))
            if alpha[cell] <= tolerance:
                offsets[cell] = lower
                status[cell] = int(PLICInterfaceStatus.EMPTY)
                evidence[cell] = True
                continue
            if alpha[cell] >= 1.0 - tolerance:
                offsets[cell] = upper
                status[cell] = int(PLICInterfaceStatus.FULL)
                evidence[cell] = True
                continue
            target_area = alpha[cell] * float(geometry.cell_volumes[cell])
            for _ in range(self.bisection_iterations):
                midpoint = 0.5 * (lower + upper)
                clipped, _ = _clip_positive_polygon(polygon, midpoint - projection)
                area, _ = _polygon_measure_centroid(clipped)
                if area < target_area:
                    lower = midpoint
                else:
                    upper = midpoint
            offset = 0.5 * (lower + upper)
            clipped, intersections = _clip_positive_polygon(polygon, offset - projection)
            reconstructed_area, _ = _polygon_measure_centroid(clipped)
            if len(intersections) != 2:
                status[cell] = int(PLICInterfaceStatus.AMBIGUOUS)
                raise ValueError(f"PLIC cell {cell} has ambiguous interface crossings.")
            first, second = intersections
            measure = np.linalg.norm(second - first)
            if measure <= 0.0:
                status[cell] = int(PLICInterfaceStatus.AMBIGUOUS)
                raise ValueError("PLIC interface segment has zero measure.")
            residual = abs(reconstructed_area - target_area)
            if residual > 1e-11 * max(float(geometry.cell_volumes[cell]), 1.0):
                status[cell] = int(PLICInterfaceStatus.AMBIGUOUS)
                raise ValueError("PLIC reconstruction failed its volume constraint.")
            offsets[cell] = offset
            endpoints[cell, 0] = first
            endpoints[cell, 1] = second
            centers[cell] = 0.5 * (first + second)
            measures[cell] = measure
            active[cell] = True
            status[cell] = int(PLICInterfaceStatus.INTERFACE)
            evidence[cell] = bool(np.isfinite(residual))

        owner_phase_aperture = np.zeros((geometry.face_measures.size, 2))
        receptor_phase_aperture = np.zeros_like(owner_phase_aperture)
        owner_phase_centroid = np.zeros((geometry.face_measures.size, 2, 2))
        receptor_phase_centroid = np.zeros_like(owner_phase_centroid)
        edge_points = vertices[face_edges]
        edge_centers = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
        owner = np.asarray(geometry.owner_cells, dtype=np.int32)
        receptor = np.asarray(geometry.neighbour_cells, dtype=np.int32)
        for face in range(edge_points.shape[0]):
            first, second = edge_points[face]
            edge_center = edge_centers[face]
            owner_cell = int(owner[face])
            owner_status = int(status[owner_cell])
            if owner_status == int(PLICInterfaceStatus.EMPTY):
                owner_centroid = np.broadcast_to(edge_center, (2, 2))
                owner_aperture = np.asarray((0.0, 1.0))
            elif owner_status == int(PLICInterfaceStatus.FULL):
                owner_centroid = np.broadcast_to(edge_center, (2, 2))
                owner_aperture = np.asarray((1.0, 0.0))
            else:
                _, owner_centroid, owner_aperture = _segment_phase_geometry(
                    first,
                    second,
                    normals[owner_cell],
                    offsets[owner_cell],
                )
            owner_phase_aperture[face] = owner_aperture
            owner_phase_centroid[face] = owner_centroid
            receptor_cell = int(receptor[face])
            if receptor_cell >= 0:
                receptor_status = int(status[receptor_cell])
                if receptor_status == int(PLICInterfaceStatus.EMPTY):
                    receptor_centroid = np.broadcast_to(edge_center, (2, 2))
                    receptor_aperture = np.asarray((0.0, 1.0))
                elif receptor_status == int(PLICInterfaceStatus.FULL):
                    receptor_centroid = np.broadcast_to(edge_center, (2, 2))
                    receptor_aperture = np.asarray((1.0, 0.0))
                else:
                    _, receptor_centroid, receptor_aperture = _segment_phase_geometry(
                        first,
                        second,
                        normals[receptor_cell],
                        offsets[receptor_cell],
                    )
                receptor_phase_aperture[face] = receptor_aperture
                receptor_phase_centroid[face] = receptor_centroid
            else:
                receptor_phase_aperture[face] = owner_aperture
                receptor_phase_centroid[face] = np.broadcast_to(edge_center, (2, 2))
        volume_fraction_id = canonical_fingerprint(
            {
                "kind": "unstructured-vof-volume-fraction",
                "plan": self.plan_id,
                "array": array_tree_fingerprint(alpha),
            }
        )
        reconstruction_id = canonical_fingerprint(
            {
                "kind": "plic-reconstruction",
                "plan": self.plan_id,
                "geometry": geometry.geometry_id,
                "volume_fraction": volume_fraction_id,
                "normals": array_tree_fingerprint(normals),
                "offsets": array_tree_fingerprint(offsets),
                "owner_phase_aperture": array_tree_fingerprint(owner_phase_aperture),
                "receptor_phase_aperture": array_tree_fingerprint(
                    receptor_phase_aperture
                ),
                "status": array_tree_fingerprint(status),
                "evidence": array_tree_fingerprint(evidence),
            }
        )
        return PLICReconstruction(
            normals=jnp.asarray(normals),
            offsets=jnp.asarray(offsets),
            interface_endpoints=jnp.asarray(endpoints),
            interface_centers=jnp.asarray(centers),
            interface_measures=jnp.asarray(measures),
            interface_active=jnp.asarray(active),
            owner_phase_aperture=jnp.asarray(owner_phase_aperture),
            receptor_phase_aperture=jnp.asarray(receptor_phase_aperture),
            owner_phase_centroid=jnp.asarray(owner_phase_centroid),
            receptor_phase_centroid=jnp.asarray(receptor_phase_centroid),
            interface_status=jnp.asarray(status),
            interface_evidence=jnp.asarray(evidence),
            geometry_id=geometry.geometry_id,
            volume_fraction_id=volume_fraction_id,
            prepared_id=geometry.prepared_id,
            topology_id=geometry.topology_id,
            reconstruction_id=reconstruction_id,
        )

    def face_phase_apertures(
        self,
        volume_fraction: ArrayLike,
        plic: PLICReconstruction,
        stage_metrics: FiniteVolumeStageMetrics,
        /,
    ) -> PLICFaceApertures:
        """Prepare exact phase apertures for certified stage face routes.

        This is intentionally host-side: stage geometry and route identities are
        checked once, while :meth:`phase_swept_flux` only consumes the resulting
        immutable arrays and remains JIT-safe.
        """

        alpha = np.asarray(self.validate_volume_fraction(volume_fraction))
        if not isinstance(plic, PLICReconstruction):
            raise TypeError("plic must be a PLICReconstruction.")
        if not isinstance(stage_metrics, FiniteVolumeStageMetrics):
            raise TypeError("stage_metrics must be FiniteVolumeStageMetrics.")
        geometry = self.discretization
        volume_fraction_id = canonical_fingerprint(
            {
                "kind": "unstructured-vof-volume-fraction",
                "plan": self.plan_id,
                "array": array_tree_fingerprint(alpha),
            }
        )
        if (
            plic.geometry_id != geometry.geometry_id
            or plic.prepared_id != geometry.prepared_id
            or plic.topology_id != geometry.topology_id
        ):
            raise ValueError("PLIC reconstruction is stale for this geometry.")
        if plic.volume_fraction_id != volume_fraction_id:
            raise ValueError("PLIC reconstruction is stale for this volume fraction.")
        plic_normals = np.asarray(plic.normals)
        plic_offsets = np.asarray(plic.offsets)
        if plic_normals.shape != (geometry.cell_count, 2):
            raise ValueError("PLIC normals have an incompatible geometry shape.")
        if plic_offsets.shape != (geometry.cell_count,):
            raise ValueError("PLIC offsets have an incompatible geometry shape.")
        if not np.all(np.isfinite(plic_normals)) or not np.all(np.isfinite(plic_offsets)):
            raise ValueError("PLIC geometry is not finite.")
        normal_lengths = np.linalg.norm(plic_normals, axis=-1)
        normal_tolerance = 1024.0 * np.finfo(plic_normals.dtype).eps
        if np.any(np.abs(normal_lengths - 1.0) > normal_tolerance):
            raise ValueError("PLIC normals are not unit vectors.")
        statuses = np.asarray(plic.interface_status)
        evidence = np.asarray(plic.interface_evidence, dtype=bool)
        if statuses.shape != (geometry.cell_count,) or evidence.shape != statuses.shape:
            raise ValueError("PLIC interface status/evidence shape is invalid.")
        valid_statuses = {
            int(PLICInterfaceStatus.EMPTY),
            int(PLICInterfaceStatus.FULL),
            int(PLICInterfaceStatus.INTERFACE),
        }
        if any(int(item) not in valid_statuses for item in statuses):
            raise ValueError("PLIC interface status is invalid.")
        if np.any(statuses == int(PLICInterfaceStatus.AMBIGUOUS)) or not np.all(evidence):
            raise ValueError("PLIC interface status/evidence is ambiguous.")
        if not bool(np.asarray(stage_metrics.evidence.passed)):
            raise ValueError("Stage geometry evidence is not certified.")
        if stage_metrics.cell_count != geometry.cell_count:
            raise ValueError("Stage geometry has an incompatible cell count.")

        vertices = np.asarray(geometry.vertices)
        face_edges = np.asarray(geometry.connectivity.edges, dtype=np.int32)
        base_owner = np.asarray(geometry.owner_cells, dtype=np.int32)
        base_receptor = np.asarray(geometry.neighbour_cells, dtype=np.int32)
        physical_face_count = int(face_edges.shape[0])
        face_ids: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        receptors: list[np.ndarray] = []
        active_masks: list[np.ndarray] = []
        owner_apertures: list[np.ndarray] = []
        receptor_apertures: list[np.ndarray] = []
        owner_centroids: list[np.ndarray] = []
        receptor_centroids: list[np.ndarray] = []
        for block in stage_metrics.face_blocks:
            layout = block.layout
            ids = np.asarray(layout.face_ids, dtype=np.int32)
            block_owners = np.asarray(layout.owner_cells, dtype=np.int32)
            block_receptors = np.asarray(layout.neighbour_cells, dtype=np.int32)
            active_mask = np.asarray(layout.active_mask, dtype=bool)
            if not np.all(active_mask):
                raise ValueError("PLIC phase apertures reject inactive stage routes.")
            if layout.block_kind == "physical":
                if np.any(ids < 0) or np.any(ids >= physical_face_count):
                    raise ValueError("Physical stage face IDs are stale.")
                if not np.array_equal(
                    block_owners, base_owner[ids]
                ) or not np.array_equal(block_receptors, base_receptor[ids]):
                    raise ValueError("Physical stage owner/receptor routes are stale.")
            elif layout.block_kind == "cut":
                if np.any(ids < physical_face_count) or np.any(
                    ids >= physical_face_count + geometry.cell_count
                ):
                    raise ValueError("Cut stage face IDs are stale.")
                expected_owners = ids - physical_face_count
                if not np.array_equal(block_owners, expected_owners) or np.any(
                    block_receptors != -1
                ):
                    raise ValueError("Cut stage owner/receptor routes are stale.")
            else:
                raise ValueError("Unsupported stage face block kind.")

            local_owner_aperture = np.zeros((ids.size, 2), dtype=float)
            local_receptor_aperture = np.zeros_like(local_owner_aperture)
            local_owner_centroid = np.zeros((ids.size, 2, 2), dtype=float)
            local_receptor_centroid = np.zeros_like(local_owner_centroid)
            for route, face_id in enumerate(ids):
                first, second = _stage_segment(block, route)
                if layout.block_kind == "physical":
                    base_first, base_second = vertices[face_edges[face_id]]
                    base_tangent = base_second - base_first
                    stage_tangent = second - first
                    base_length = float(np.linalg.norm(base_tangent))
                    stage_length = float(np.linalg.norm(stage_tangent))
                    collinearity = abs(
                        base_tangent[0] * stage_tangent[1]
                        - base_tangent[1] * stage_tangent[0]
                    )
                    line_distance = max(
                        abs(
                            base_tangent[0] * (first - base_first)[1]
                            - base_tangent[1] * (first - base_first)[0]
                        ),
                        abs(
                            base_tangent[0] * (second - base_first)[1]
                            - base_tangent[1] * (second - base_first)[0]
                        ),
                    )
                    tolerance = (
                        2048.0
                        * np.finfo(vertices.dtype).eps
                        * max(base_length, stage_length, 1.0)
                    )
                    if (
                        collinearity > tolerance
                        or line_distance > tolerance
                        or stage_length > base_length + tolerance
                    ):
                        raise ValueError("Stage physical face geometry is stale.")
                owner_cell = int(block_owners[route])
                center = np.asarray(block.face_centers)[route]
                owner_status = int(statuses[owner_cell])
                if owner_status == int(PLICInterfaceStatus.EMPTY):
                    owner_centroid = np.broadcast_to(center, (2, 2))
                    owner_aperture = np.asarray((0.0, 1.0))
                elif owner_status == int(PLICInterfaceStatus.FULL):
                    owner_centroid = np.broadcast_to(center, (2, 2))
                    owner_aperture = np.asarray((1.0, 0.0))
                else:
                    _, owner_centroid, owner_aperture = _segment_phase_geometry(
                        first,
                        second,
                        np.asarray(plic.normals)[owner_cell],
                        float(np.asarray(plic.offsets)[owner_cell]),
                    )
                local_owner_aperture[route] = owner_aperture
                local_owner_centroid[route] = owner_centroid
                receptor_cell = int(block_receptors[route])
                if receptor_cell >= 0:
                    receptor_status = int(statuses[receptor_cell])
                    if receptor_status == int(PLICInterfaceStatus.EMPTY):
                        receptor_centroid = np.broadcast_to(center, (2, 2))
                        receptor_aperture = np.asarray((0.0, 1.0))
                    elif receptor_status == int(PLICInterfaceStatus.FULL):
                        receptor_centroid = np.broadcast_to(center, (2, 2))
                        receptor_aperture = np.asarray((1.0, 0.0))
                    else:
                        _, receptor_centroid, receptor_aperture = _segment_phase_geometry(
                            first,
                            second,
                            np.asarray(plic.normals)[receptor_cell],
                            float(np.asarray(plic.offsets)[receptor_cell]),
                        )
                    local_receptor_aperture[route] = receptor_aperture
                    local_receptor_centroid[route] = receptor_centroid
                else:
                    local_receptor_aperture[route] = owner_aperture
                    local_receptor_centroid[route] = np.broadcast_to(center, (2, 2))
            face_ids.append(ids)
            owners.append(block_owners)
            receptors.append(block_receptors)
            active_masks.append(active_mask)
            owner_apertures.append(local_owner_aperture)
            receptor_apertures.append(local_receptor_aperture)
            owner_centroids.append(local_owner_centroid)
            receptor_centroids.append(local_receptor_centroid)

        route_count = sum(item.size for item in face_ids)
        if route_count:
            route_face_ids = np.concatenate(face_ids)
            route_owners = np.concatenate(owners)
            route_receptors = np.concatenate(receptors)
            route_active = np.concatenate(active_masks)
            route_owner_apertures = np.concatenate(owner_apertures)
            route_receptor_apertures = np.concatenate(receptor_apertures)
            route_owner_centroids = np.concatenate(owner_centroids)
            route_receptor_centroids = np.concatenate(receptor_centroids)
            sort_keys = tuple(
                (
                    (0, *sorted(np.asarray(face_edges[int(face_id)]).tolist()))
                    if int(face_id) < physical_face_count
                    else (1, int(face_id))
                )
                for face_id in route_face_ids
            )
            order = np.asarray(
                sorted(range(route_face_ids.size), key=lambda index: sort_keys[index]),
                dtype=np.int32,
            )
            route_face_ids = route_face_ids[order]
            route_owners = route_owners[order]
            route_receptors = route_receptors[order]
            route_active = route_active[order]
            route_owner_apertures = route_owner_apertures[order]
            route_receptor_apertures = route_receptor_apertures[order]
            route_owner_centroids = route_owner_centroids[order]
            route_receptor_centroids = route_receptor_centroids[order]
        apertures_id = canonical_fingerprint(
            {
                "kind": "plic-face-apertures",
                "plan": self.plan_id,
                "plic": plic.reconstruction_id,
                "geometry_layout": stage_metrics.geometry_layout_id,
                "topology_epoch": stage_metrics.topology_epoch_id,
                "face_ids": array_tree_fingerprint(route_face_ids),
                "owner_phase_apertures": array_tree_fingerprint(route_owner_apertures),
                "receptor_phase_apertures": array_tree_fingerprint(
                    route_receptor_apertures
                ),
            }
        )
        return PLICFaceApertures(
            face_ids=jnp.asarray(route_face_ids),
            owner_cells=jnp.asarray(route_owners),
            receptor_cells=jnp.asarray(route_receptors),
            active_mask=jnp.asarray(route_active),
            owner_phase_apertures=jnp.asarray(route_owner_apertures),
            receptor_phase_apertures=jnp.asarray(route_receptor_apertures),
            owner_phase_centroids=jnp.asarray(route_owner_centroids),
            receptor_phase_centroids=jnp.asarray(route_receptor_centroids),
            plan_id=self.plan_id,
            plic_reconstruction_id=plic.reconstruction_id,
            geometry_id=geometry.geometry_id,
            prepared_id=geometry.prepared_id,
            topology_id=geometry.topology_id,
            volume_fraction_id=volume_fraction_id,
            geometry_layout_id=stage_metrics.geometry_layout_id,
            topology_epoch_id=stage_metrics.topology_epoch_id,
            apertures_id=apertures_id,
        )

    def donor_phase_apertures(
        self,
        total_volume_flux: ArrayLike,
        reconstruction: JAXPLICStageReconstruction,
        /,
    ) -> Array:
        """Select both phase apertures from the upwind physical-face donor."""

        if not isinstance(reconstruction, JAXPLICStageReconstruction):
            raise TypeError("reconstruction must be a JAXPLICStageReconstruction.")
        if (
            reconstruction.plan_id != self.plan_id
            or reconstruction.geometry_id != self.discretization.geometry_id
            or reconstruction.prepared_id != self.discretization.prepared_id
            or reconstruction.topology_id != self.discretization.topology_id
            or reconstruction.physical_layout_id != self.physical_layout_id
            or not reconstruction.effective_geometry_id
            or not reconstruction.geometry_layout_id
        ):
            raise ValueError(
                "Stage PLIC reconstruction is stale for this topology or layout."
            )
        route_shape = (self.discretization.face_measures.size,)
        if (
            reconstruction.face_ids.shape != route_shape
            or reconstruction.owner_cells.shape != route_shape
            or reconstruction.receptor_cells.shape != route_shape
            or reconstruction.open_face_active.shape != route_shape
            or reconstruction.owner_phase_apertures.shape != (*route_shape, 2)
            or reconstruction.receptor_phase_apertures.shape != (*route_shape, 2)
            or reconstruction.volume_fraction.shape != (self.discretization.cell_count,)
            or reconstruction.aperture_ids.shape != (self.discretization.cell_count,)
        ):
            raise ValueError("Stage PLIC physical route shapes are stale.")
        total = jnp.asarray(total_volume_flux)
        if not jnp.issubdtype(total.dtype, jnp.inexact):
            total = total.astype(reconstruction.owner_phase_apertures.dtype)
        if total.shape != route_shape:
            raise ValueError("total_volume_flux must match physical face routes.")
        route_evidence = (
            jnp.all(
                reconstruction.face_ids == jnp.arange(route_shape[0], dtype=jnp.int32)
            )
            & jnp.all(reconstruction.owner_cells == self.discretization.owner_cells)
            & jnp.all(
                reconstruction.receptor_cells == self.discretization.neighbour_cells
            )
            & jnp.all(reconstruction.interface_evidence)
            & jnp.all(reconstruction.aperture_ids == reconstruction.volume_fraction)
            & (reconstruction.geometry_version >= 0)
            & jnp.all(
                jnp.where(
                    reconstruction.open_face_active,
                    True,
                    total == 0.0,
                )
            )
        )
        total = eqx.error_if(
            total,
            ~route_evidence,
            "Stage PLIC topology, layout, or alpha identity is stale.",
        )
        total = eqx.error_if(
            total,
            jnp.any(~jnp.isfinite(total)),
            "total_volume_flux must be finite.",
        )
        owner = reconstruction.owner_phase_apertures.astype(total.dtype)
        receptor = reconstruction.receptor_phase_apertures.astype(total.dtype)
        donor = jnp.where(
            (total >= 0.0)[:, None],
            owner,
            jnp.where(
                (reconstruction.receptor_cells >= 0)[:, None],
                receptor,
                owner,
            ),
        )
        tolerance = 64.0 * jnp.finfo(total.dtype).eps
        expected_sum = reconstruction.open_face_active.astype(total.dtype)
        donor = eqx.error_if(
            donor,
            jnp.any(~jnp.isfinite(donor) | (donor < 0.0) | (donor > 1.0))
            | jnp.any(jnp.abs(jnp.sum(donor, axis=-1) - expected_sum) > tolerance),
            "Stage PLIC donor apertures are invalid.",
        )
        return donor

    def donor_phase_aperture(
        self,
        total_volume_flux: ArrayLike,
        reconstruction: JAXPLICStageReconstruction,
        /,
    ) -> Array:
        """Select the phase-zero aperture from each upwind physical-face donor."""

        return self.donor_phase_apertures(total_volume_flux, reconstruction)[:, 0]

    def phase_swept_flux(
        self,
        total_volume_flux: ArrayLike,
        apertures: PLICFaceApertures | JAXPLICStageReconstruction,
        /,
    ) -> tuple[Array, Array, Array]:
        if isinstance(apertures, JAXPLICStageReconstruction):
            total = jnp.asarray(total_volume_flux)
            donor = self.donor_phase_apertures(total, apertures)[:, 0]
            phase0 = total * donor
            phase1 = total - phase0
            return phase0, phase1, phase0
        if not isinstance(apertures, PLICFaceApertures):
            raise TypeError(
                "apertures must be PLICFaceApertures or JAXPLICStageReconstruction."
            )
        if (
            apertures.plan_id != self.plan_id
            or apertures.geometry_id != self.discretization.geometry_id
            or apertures.prepared_id != self.discretization.prepared_id
            or apertures.topology_id != self.discretization.topology_id
        ):
            raise ValueError("Phase apertures are stale for this geometry.")
        route_shape = apertures.owner_cells.shape
        if (
            apertures.receptor_cells.shape != route_shape
            or apertures.active_mask.shape != route_shape
            or apertures.owner_phase_apertures.shape != (*route_shape, 2)
            or apertures.receptor_phase_apertures.shape != (*route_shape, 2)
        ):
            raise ValueError("Prepared phase aperture routes have incompatible shapes.")
        if isinstance(total_volume_flux, (tuple, list)):
            total = jnp.concatenate(
                tuple(jnp.asarray(item) for item in total_volume_flux)
            )
        else:
            total = jnp.asarray(total_volume_flux)
        expected_shape = apertures.owner_cells.shape
        if total.shape != expected_shape:
            raise ValueError("total_volume_flux must match prepared face routes.")
        total = eqx.error_if(
            total,
            jnp.any(~jnp.isfinite(total)),
            "total_volume_flux must be finite.",
        )
        total = eqx.error_if(
            total,
            jnp.any(~apertures.active_mask),
            "Phase swept flux received an inactive face route.",
        )
        owner = apertures.owner_phase_apertures[:, 0].astype(total.dtype)
        receptor = apertures.receptor_phase_apertures[:, 0].astype(total.dtype)
        donor = jnp.where(
            total >= 0.0,
            owner,
            jnp.where(apertures.receptor_cells >= 0, receptor, owner),
        )
        donor = eqx.error_if(
            donor,
            jnp.any(~jnp.isfinite(donor) | (donor < 0.0) | (donor > 1.0)),
            "Prepared phase aperture lies outside [0, 1].",
        )
        phase0 = total * donor
        phase1 = total - phase0
        return phase0, phase1, phase0

    def advective_face_flux(
        self, volume_fraction: ArrayLike, face_normal_volume_flux: ArrayLike, /
    ) -> Array:
        alpha = self.validate_volume_fraction(volume_fraction)
        volume_flux = jnp.asarray(face_normal_volume_flux, dtype=alpha.dtype)
        if volume_flux.shape != (self.discretization.face_measures.size,):
            raise ValueError("Face volume flux must contain one value per face.")
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        donor = jnp.where(
            volume_flux >= 0.0,
            alpha[owner],
            jnp.where(neighbour >= 0, alpha[safe_neighbour], alpha[owner]),
        )
        return donor * volume_flux

    def residual(
        self, volume_fraction: ArrayLike, face_normal_volume_flux: ArrayLike, /
    ) -> Array:
        alpha = self.validate_volume_fraction(volume_fraction)
        phase_flux = self.advective_face_flux(alpha, face_normal_volume_flux)
        integrated = phase_flux * self.discretization.face_measures.astype(alpha.dtype)
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        residual = jnp.zeros_like(alpha).at[owner].add(-integrated)
        residual = residual.at[safe_neighbour].add(
            jnp.where(neighbour >= 0, integrated, 0.0)
        )
        return residual / self.discretization.cell_volumes.astype(alpha.dtype)

    def stable_step(
        self,
        volume_fraction: ArrayLike,
        face_normal_volume_flux: ArrayLike,
        /,
        *,
        safety: float = 0.95,
    ) -> Array:
        alpha = self.validate_volume_fraction(volume_fraction)
        phase_flux = self.advective_face_flux(alpha, face_normal_volume_flux)
        integrated = phase_flux * self.discretization.face_measures.astype(alpha.dtype)
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        owner_outflow = jnp.maximum(integrated, 0.0)
        owner_inflow = jnp.maximum(-integrated, 0.0)
        neighbour_outflow = jnp.maximum(-integrated, 0.0)
        neighbour_inflow = jnp.maximum(integrated, 0.0)
        outflow = jnp.zeros_like(alpha).at[owner].add(owner_outflow)
        inflow = jnp.zeros_like(alpha).at[owner].add(owner_inflow)
        outflow = outflow.at[safe_neighbour].add(
            jnp.where(neighbour >= 0, neighbour_outflow, 0.0)
        )
        inflow = inflow.at[safe_neighbour].add(
            jnp.where(neighbour >= 0, neighbour_inflow, 0.0)
        )
        volume = self.discretization.cell_volumes.astype(alpha.dtype)
        liquid_step = jnp.where(outflow > 0.0, alpha * volume / outflow, jnp.inf)
        void_step = jnp.where(inflow > 0.0, (1.0 - alpha) * volume / inflow, jnp.inf)
        return jnp.asarray(safety, dtype=alpha.dtype) * jnp.min(
            jnp.minimum(liquid_step, void_step)
        )

    def advance(
        self,
        volume_fraction: ArrayLike,
        face_normal_volume_flux: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> Array:
        alpha = self.validate_volume_fraction(volume_fraction)
        step = jnp.asarray(step_size, dtype=alpha.dtype).reshape(())
        stable = self.stable_step(alpha, face_normal_volume_flux)
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step < 0.0) | (step > stable),
            "VOF step exceeds the conservative boundedness restriction.",
        )
        updated = alpha + step * self.residual(alpha, face_normal_volume_flux)
        return eqx.error_if(
            updated,
            jnp.any((updated < -1e-12) | (updated > 1.0 + 1e-12)),
            "VOF update violated [0, 1] bounds.",
        )

    def phase_volume(self, volume_fraction: ArrayLike, /) -> Array:
        alpha = self.validate_volume_fraction(volume_fraction)
        return jnp.sum(self.discretization.cell_volumes.astype(alpha.dtype) * alpha)


__all__ = [
    "JAXPLICStageReconstruction",
    "PLICFaceApertures",
    "PLICInterfaceStatus",
    "PLICReconstruction",
    "UnstructuredVOFPlan",
]
