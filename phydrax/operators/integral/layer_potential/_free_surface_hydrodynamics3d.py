#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....geometry import MeshRegion
from ....linalg import DenseLinearOperator, MaterializationPolicy
from ._free_surface_green3d import (
    FreeSurfaceGreenPolicy3D,
    FreeSurfaceGreenRepresentation3D,
    prepare_free_surface_green_3d,
)
from ._galerkin3d import (
    LaplaceSingleLayerDP0AssemblyReport3D,
    LaplaceSingleLayerDP0Galerkin3D,
    LaplaceSingleLayerDP0GalerkinPolicy3D,
    prepare_laplace_single_layer_dp0_3d,
)


_PDE_ID = "three-dimensional-inviscid-incompressible-irrotational-laplace"
_TIME_CONVENTION = "exp(-i*omega*t)"
_NORMAL_CONVENTION = "body-to-fluid"
_PRECISION_ID = "float64-complex128"
_NON_GOALS = (
    "continuum-discretization certification",
    "waterline-panel hydrodynamics",
    "forward speed",
    "viscous or nonlinear loads",
    "irregular-frequency removal",
)
_MODE_LABELS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _nonempty(value: str, name: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _mesh_arrays(region: MeshRegion, /) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(region, MeshRegion):
        raise TypeError("region must be a MeshRegion.")
    topology = region.triangle_mesh.topology
    if not topology.watertight:
        raise ValueError("Potential-flow body geometry must be watertight.")
    vertices = np.asarray(region.triangle_mesh.vertices, dtype=float)
    faces = np.asarray(region.triangle_mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or np.any(~np.isfinite(vertices)):
        raise ValueError("Potential-flow body vertices must be finite three-vectors.")
    triangles = vertices[faces]
    doubled_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    scale = max(float(np.max(np.ptp(vertices, axis=0))), 1.0)
    if np.any(doubled_areas <= 64.0 * np.finfo(float).eps * scale * scale):
        raise ValueError("Potential-flow body faces must be nondegenerate.")
    return vertices, faces


class HydrostaticProperties3D(StrictModule, NonTrainableState):
    """Exact polyhedral hydrostatics for the declared transversal waterline route."""

    displaced_volume: Array
    center_of_buoyancy: Array
    waterplane_area: Array
    waterplane_centroid: Array
    waterplane_second_moments: Array
    restoring_matrix: Array
    reference_point: Array
    fluid_density: Array
    gravity: Array
    waterline_loop_count: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    pde_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    time_convention: str = eqx.field(static=True)
    normal_convention: str = eqx.field(static=True)
    resource_evidence: tuple[int, int] = eqx.field(static=True)
    error_evidence: tuple[str, ...] = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    valid: Array
    result_id: str = eqx.field(static=True)


def _clip_triangle_below(
    triangle: np.ndarray, free_surface_z: float, /
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    polygon: list[np.ndarray] = []
    intersections: list[np.ndarray] = []
    for index in range(3):
        start = triangle[index]
        end = triangle[(index + 1) % 3]
        start_inside = bool(start[2] < free_surface_z)
        end_inside = bool(end[2] < free_surface_z)
        if start_inside and end_inside:
            polygon.append(end)
        elif start_inside != end_inside:
            fraction = (free_surface_z - start[2]) / (end[2] - start[2])
            point = start + fraction * (end - start)
            point[2] = free_surface_z
            intersections.append(point)
            if start_inside:
                polygon.append(point)
            else:
                polygon.extend((point, end))
    return polygon, intersections


def _waterline_loops(
    segments: Sequence[tuple[np.ndarray, np.ndarray]], tolerance: float, /
) -> list[np.ndarray]:
    points: dict[tuple[int, int], np.ndarray] = {}
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def key(point: np.ndarray) -> tuple[int, int]:
        return tuple(np.rint(point[:2] / tolerance).astype(np.int64).tolist())

    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for start, end in segments:
        start_key = key(start)
        end_key = key(end)
        if start_key == end_key:
            raise ValueError("Waterline contains a collapsed intersection segment.")
        points.setdefault(start_key, start[:2].copy())
        points.setdefault(end_key, end[:2].copy())
        adjacency.setdefault(start_key, []).append(end_key)
        adjacency.setdefault(end_key, []).append(start_key)
        edges.add(tuple(sorted((start_key, end_key))))
    if not edges or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError(
            "Waterline segments must form one or more closed degree-two loops."
        )

    remaining = set(edges)
    loops: list[np.ndarray] = []
    while remaining:
        first_edge = next(iter(remaining))
        start, current = first_edge
        loop_keys = [start]
        previous = start
        remaining.remove(first_edge)
        while current != start:
            loop_keys.append(current)
            neighbors = adjacency[current]
            following = neighbors[0] if neighbors[0] != previous else neighbors[1]
            edge = tuple(sorted((current, following)))
            if edge not in remaining:
                raise ValueError("Waterline loop repeats an edge before closing.")
            remaining.remove(edge)
            previous, current = current, following
        if len(loop_keys) < 3:
            raise ValueError("Every waterline loop must contain at least three vertices.")
        loops.append(np.stack([points[item] for item in loop_keys]))
    return loops


def _polygon_moments(
    loop: np.ndarray, /
) -> tuple[float, float, float, float, float, float]:
    following = np.roll(loop, -1, axis=0)
    cross = loop[:, 0] * following[:, 1] - following[:, 0] * loop[:, 1]
    twice_area = float(np.sum(cross))
    if abs(twice_area) <= 128.0 * np.finfo(float).eps:
        raise ValueError("Waterline loop has zero numerical area.")
    if twice_area < 0.0:
        loop = loop[::-1]
        following = np.roll(loop, -1, axis=0)
        cross = loop[:, 0] * following[:, 1] - following[:, 0] * loop[:, 1]
        twice_area = float(np.sum(cross))
    area = 0.5 * twice_area
    first_x = float(np.sum((loop[:, 0] + following[:, 0]) * cross) / 6.0)
    first_y = float(np.sum((loop[:, 1] + following[:, 1]) * cross) / 6.0)
    second_x = float(
        np.sum(
            (loop[:, 1] ** 2 + loop[:, 1] * following[:, 1] + following[:, 1] ** 2)
            * cross
        )
        / 12.0
    )
    second_y = float(
        np.sum(
            (loop[:, 0] ** 2 + loop[:, 0] * following[:, 0] + following[:, 0] ** 2)
            * cross
        )
        / 12.0
    )
    product = float(
        np.sum(
            (
                2.0 * loop[:, 0] * loop[:, 1]
                + loop[:, 0] * following[:, 1]
                + following[:, 0] * loop[:, 1]
                + 2.0 * following[:, 0] * following[:, 1]
            )
            * cross
        )
        / 24.0
    )
    return area, first_x, first_y, second_x, second_y, product


def prepare_hydrostatic_properties_3d(
    region: MeshRegion,
    /,
    *,
    fluid_density: float = 1025.0,
    gravity: float = 9.80665,
    free_surface_z: float = 0.0,
    reference_point: ArrayLike | None = None,
    frame_id: str = "z-up-cartesian",
    unit_system_id: str = "si",
) -> HydrostaticProperties3D:
    """Clip a watertight body at a transversal horizontal waterline.

    The bounded route rejects vertices on the waterplane, tangent contacts,
    open/non-manifold waterlines, and nested waterline loops. Restoring data use
    the supplied reference as the equilibrium center of gravity and the usual
    small-angle surge/sway/heave/roll/pitch/yaw ordering.
    """
    vertices, faces = _mesh_arrays(region)
    density = float(fluid_density)
    gravity_ = float(gravity)
    surface = float(free_surface_z)
    if any(not math.isfinite(value) or value <= 0.0 for value in (density, gravity_)):
        raise ValueError("fluid_density and gravity must be finite and positive.")
    if not math.isfinite(surface):
        raise ValueError("free_surface_z must be finite.")
    frame = _nonempty(frame_id, "frame_id")
    units = _nonempty(unit_system_id, "unit_system_id")
    scale = max(float(np.max(np.ptp(vertices, axis=0))), 1.0)
    tolerance = 256.0 * np.finfo(float).eps * scale
    signed_heights = vertices[:, 2] - surface
    if np.any(np.abs(signed_heights) <= tolerance):
        raise ValueError(
            "Waterline vertices are outside the qualified transversal route."
        )
    if not np.any(signed_heights < 0.0) or not np.any(signed_heights > 0.0):
        raise ValueError("Hydrostatics requires a body that crosses the free surface.")

    clipped_triangles: list[np.ndarray] = []
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for triangle in vertices[faces]:
        polygon, intersections = _clip_triangle_below(triangle.copy(), surface)
        if len(intersections) not in (0, 2):
            raise ValueError("Each cut face must intersect the waterline transversally.")
        if len(intersections) == 2:
            segments.append((intersections[0], intersections[1]))
        for index in range(1, len(polygon) - 1):
            clipped_triangles.append(
                np.stack((polygon[0], polygon[index], polygon[index + 1]))
            )
    loops = _waterline_loops(segments, max(tolerance, 1.0e-14))

    # Nested loops would require hole classification; reject instead of over-counting.
    for left in range(len(loops)):
        for right in range(left + 1, len(loops)):
            left_min, left_max = np.min(loops[left], axis=0), np.max(loops[left], axis=0)
            right_min, right_max = (
                np.min(loops[right], axis=0),
                np.max(loops[right], axis=0),
            )
            boxes_overlap = np.all(left_min <= right_max) and np.all(
                right_min <= left_max
            )
            if boxes_overlap:
                raise ValueError("Overlapping or nested waterline loops are unsupported.")

    relative_triangles = np.asarray(clipped_triangles, dtype=float)
    relative_triangles[:, :, 2] -= surface
    signed_volumes = (
        np.sum(
            relative_triangles[:, 0]
            * np.cross(relative_triangles[:, 1], relative_triangles[:, 2]),
            axis=1,
        )
        / 6.0
    )
    volume = float(np.sum(signed_volumes))
    if not math.isfinite(volume) or volume <= tolerance**3:
        raise ValueError("Clipped submerged volume must be finite and positive.")
    centroid_relative = (
        np.sum(
            signed_volumes[:, None] * np.sum(relative_triangles, axis=1) / 4.0,
            axis=0,
        )
        / volume
    )
    center_of_buoyancy = centroid_relative + np.asarray((0.0, 0.0, surface))

    moments = np.sum(np.asarray([_polygon_moments(loop) for loop in loops]), axis=0)
    area, first_x, first_y, second_x, second_y, product = moments.tolist()
    waterplane_centroid = np.asarray((first_x / area, first_y / area, surface))
    reference = (
        waterplane_centroid.copy()
        if reference_point is None
        else np.asarray(reference_point, dtype=float)
    )
    if reference.shape != (3,) or np.any(~np.isfinite(reference)):
        raise ValueError("reference_point must be one finite three-vector.")
    sx = first_x - reference[0] * area
    sy = first_y - reference[1] * area
    ix = second_x - 2.0 * reference[1] * first_y + reference[1] ** 2 * area
    iy = second_y - 2.0 * reference[0] * first_x + reference[0] ** 2 * area
    ixy = (
        product
        - reference[0] * first_y
        - reference[1] * first_x
        + reference[0] * reference[1] * area
    )
    waterplane_block = np.asarray(
        ((area, sy, -sx), (sy, ix, -ixy), (-sx, -ixy, iy)), dtype=float
    )
    metacentric_shift = volume * (center_of_buoyancy[2] - reference[2])
    waterplane_block[1, 1] += metacentric_shift
    waterplane_block[2, 2] += metacentric_shift
    restoring = np.zeros((6, 6), dtype=float)
    indices = np.asarray((2, 3, 4), dtype=np.int32)
    restoring[np.ix_(indices, indices)] = density * gravity_ * waterplane_block
    result_id = canonical_fingerprint(
        {
            "kind": "polyhedral-hydrostatics-3d",
            "region": region.feature_id,
            "surface": surface,
            "reference": array_tree_fingerprint(reference),
            "density": density,
            "gravity": gravity_,
            "frame": frame,
            "units": units,
        }
    )
    return HydrostaticProperties3D(
        displaced_volume=jnp.asarray(volume, dtype=jnp.float64),
        center_of_buoyancy=jnp.asarray(center_of_buoyancy, dtype=jnp.float64),
        waterplane_area=jnp.asarray(area, dtype=jnp.float64),
        waterplane_centroid=jnp.asarray(waterplane_centroid, dtype=jnp.float64),
        waterplane_second_moments=jnp.asarray((ix, iy, ixy), dtype=jnp.float64),
        restoring_matrix=jnp.asarray(restoring, dtype=jnp.float64),
        reference_point=jnp.asarray(reference, dtype=jnp.float64),
        fluid_density=jnp.asarray(density, dtype=jnp.float64),
        gravity=jnp.asarray(gravity_, dtype=jnp.float64),
        waterline_loop_count=len(loops),
        ambient_dimension=3,
        coordinate_convention="right-handed-cartesian-z-up",
        pde_id="hydrostatic-pressure-equilibrium",
        geometry_id=region.feature_id,
        formulation_id="transversal-polyhedral-clipping-and-waterplane-moments",
        provider_id="phydrax-host-polyhedral-hydrostatics",
        precision_id="float64",
        frame_id=frame,
        unit_system_id=units,
        time_convention="static",
        normal_convention=_NORMAL_CONVENTION,
        resource_evidence=(int(vertices.shape[0]), int(faces.shape[0])),
        error_evidence=(
            "exact for the input planar triangle geometry up to float64 roundoff",
            "no curved-surface or continuum geometry error estimate",
        ),
        non_goals=(
            "nested waterplane loops",
            "waterline vertices or tangent contact",
            "large-angle or nonlinear hydrostatics",
        ),
        valid=jnp.asarray(True),
        result_id=result_id,
    )


class FreeSurfaceHydrodynamicsPolicy3D(StrictModule, NonTrainableState):
    """Dense bounded preparation policy for the zero-speed DP0 body problem."""

    green: FreeSurfaceGreenPolicy3D
    galerkin: LaplaceSingleLayerDP0GalerkinPolicy3D
    max_faces: int = eqx.field(static=True)
    max_dense_entries: int = eqx.field(static=True)
    max_resident_bytes: int = eqx.field(static=True)
    max_preparation_workspace_bytes: int = eqx.field(static=True)
    minimum_geometric_clearance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        green: FreeSurfaceGreenPolicy3D | None = None,
        galerkin: LaplaceSingleLayerDP0GalerkinPolicy3D | None = None,
        max_faces: int = 256,
        max_dense_entries: int = 256 * 256 * 2,
        max_resident_bytes: int = 256 * 1024 * 1024,
        max_preparation_workspace_bytes: int = 512 * 1024 * 1024,
        minimum_geometric_clearance: float = 1.0e-8,
    ):
        green_ = FreeSurfaceGreenPolicy3D() if green is None else green
        galerkin_ = (
            LaplaceSingleLayerDP0GalerkinPolicy3D() if galerkin is None else galerkin
        )
        if not isinstance(green_, FreeSurfaceGreenPolicy3D):
            raise TypeError("green must be FreeSurfaceGreenPolicy3D or None.")
        if not isinstance(galerkin_, LaplaceSingleLayerDP0GalerkinPolicy3D):
            raise TypeError(
                "galerkin must be LaplaceSingleLayerDP0GalerkinPolicy3D or None."
            )
        faces = int(max_faces)
        entries = int(max_dense_entries)
        resident = int(max_resident_bytes)
        preparation = int(max_preparation_workspace_bytes)
        clearance = float(minimum_geometric_clearance)
        if min(faces, entries, resident, preparation) < 1:
            raise ValueError("Hydrodynamics resource limits must be positive.")
        if not math.isfinite(clearance) or clearance <= 0.0:
            raise ValueError("minimum_geometric_clearance must be finite and positive.")
        self.green = green_
        self.galerkin = galerkin_
        self.max_faces = faces
        self.max_dense_entries = entries
        self.max_resident_bytes = resident
        self.max_preparation_workspace_bytes = preparation
        self.minimum_geometric_clearance = clearance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "free-surface-hydrodynamics-policy-3d",
                "green": green_.policy_id,
                "galerkin": galerkin_.policy_id,
                "max_faces": faces,
                "max_dense_entries": entries,
                "max_resident_bytes": resident,
                "max_preparation_workspace_bytes": preparation,
                "minimum_geometric_clearance": clearance,
            }
        )


class FreeSurfaceHydrodynamicsAssemblyReport3D(StrictModule, NonTrainableState):
    """Resource and unresolved-error evidence for the prepared body operator."""

    laplace_report: LaplaceSingleLayerDP0AssemblyReport3D
    green_tail_bound: Array
    dispersion_residual: Array
    face_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    degree_of_freedom_count: int = eqx.field(static=True)
    boundary_operator_bytes: int = eqx.field(static=True)
    trace_operator_bytes: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    maximum_resident_bytes: int = eqx.field(static=True)
    maximum_preparation_workspace_bytes: int = eqx.field(static=True)
    continuum_discretization_error_estimated: bool = eqx.field(static=True)
    collocation_error_estimated: bool = eqx.field(static=True)
    finite: Array
    supported: Array
    report_id: str = eqx.field(static=True)


class PreparedFreeSurfaceHydrodynamics3D(StrictModule, NonTrainableState):
    """Prepared submerged-body zero-speed linear potential-flow problem."""

    galerkin: LaplaceSingleLayerDP0Galerkin3D
    green: FreeSurfaceGreenRepresentation3D
    boundary_operator: DenseLinearOperator
    trace_operator: DenseLinearOperator
    face_centroids: Array
    face_normals: Array
    face_areas: Array
    face_component_ids: Array
    rigid_mode_normal_velocity: Array
    reference_points: Array
    assembly_report: FreeSurfaceHydrodynamicsAssemblyReport3D
    angular_frequency: float = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    depth: float | None = eqx.field(static=True)
    free_surface_z: float = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    degree_of_freedom_count: int = eqx.field(static=True)
    mode_names: tuple[str, ...] = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    pde_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    formulation_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    time_convention: str = eqx.field(static=True)
    normal_convention: str = eqx.field(static=True)
    density_semantics: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def face_count(self) -> int:
        return int(self.face_areas.shape[0])

    def potential_trace(self, density: ArrayLike, /) -> Array:
        """Map unweighted DP0 source-strength coefficients to face potential."""
        values = jnp.asarray(density, dtype=jnp.complex128)
        if values.shape != (self.face_count,):
            raise ValueError("density must contain one unweighted value per body face.")
        return self.trace_operator.mv(values)

    def incident_wave(
        self,
        amplitude: ArrayLike,
        heading: float,
        /,
    ) -> tuple[Array, Array]:
        """Return incident potential trace and body-normal derivative.

        ``amplitude`` is the complex free-surface elevation amplitude and heading
        is measured counter-clockwise from +x in the declared z-up frame.
        """
        amplitude_ = jnp.asarray(amplitude, dtype=jnp.complex128)
        if amplitude_.shape != ():
            raise ValueError("amplitude must be one complex scalar.")
        heading_ = float(heading)
        if not math.isfinite(heading_):
            raise ValueError("heading must be finite.")
        direction = jnp.asarray((math.cos(heading_), math.sin(heading_)))
        k = self.green.wavenumber
        horizontal_phase = jnp.exp(1j * k * (self.face_centroids[:, :2] @ direction))
        z = self.face_centroids[:, 2] - self.free_surface_z
        if self.depth is None:
            vertical = jnp.exp(k * z)
            vertical_derivative = k * vertical
        else:
            vertical = jnp.cosh(k * (z + self.depth)) / jnp.cosh(k * self.depth)
            vertical_derivative = (
                k * jnp.sinh(k * (z + self.depth)) / jnp.cosh(k * self.depth)
            )
        scale = -1j * self.gravity * amplitude_ / self.angular_frequency
        potential = scale * vertical * horizontal_phase
        gradient = jnp.concatenate(
            (
                1j * k * direction[None, :] * potential[:, None],
                (scale * vertical_derivative * horizontal_phase)[:, None],
            ),
            axis=1,
        )
        normal_derivative = jnp.sum(gradient * self.face_normals, axis=1)
        return potential, normal_derivative


def _galerkin_policy_with_oracle(
    policy: LaplaceSingleLayerDP0GalerkinPolicy3D,
    entries: int,
    byte_limit: int,
    /,
) -> LaplaceSingleLayerDP0GalerkinPolicy3D:
    return LaplaceSingleLayerDP0GalerkinPolicy3D(
        regular_order=policy.regular_order,
        singular_order=policy.singular_order,
        near_order=policy.near_order,
        near_ratio=policy.near_ratio,
        near_max_depth=policy.near_max_depth,
        absolute_tolerance=policy.absolute_tolerance,
        relative_tolerance=policy.relative_tolerance,
        target_block_size=policy.target_block_size,
        source_block_size=policy.source_block_size,
        max_exception_pairs=policy.max_exception_pairs,
        max_preparation_workspace_bytes=policy.max_preparation_workspace_bytes,
        max_resident_bytes=policy.max_resident_bytes,
        precision=policy.precision,
        dense_oracle=MaterializationPolicy(max_entries=entries, max_bytes=byte_limit),
    )


def _blocked_wave_normal_matrix(
    green: FreeSurfaceGreenRepresentation3D,
    targets: Array,
    target_normals: Array,
    panel_points: Array,
    panel_weights: Array,
    nodes_per_panel: int,
    target_block_size: int,
    source_block_size: int,
    /,
) -> Array:
    face_count = int(targets.shape[0])
    rows = []
    for target_start in range(0, face_count, target_block_size):
        target_stop = min(target_start + target_block_size, face_count)
        target_block = targets[target_start:target_stop]
        normal_block = target_normals[target_start:target_stop]
        columns = []
        for source_start in range(0, face_count, source_block_size):
            source_stop = min(source_start + source_block_size, face_count)
            node_start = source_start * nodes_per_panel
            node_stop = source_stop * nodes_per_panel
            source_points = panel_points[node_start:node_stop]
            source_weights = panel_weights[node_start:node_stop]
            gradients = jax.vmap(
                lambda target: jax.vmap(
                    lambda source: green.wave_correction_target_gradient(target, source)
                )(source_points)
            )(target_block)
            normal_values = jnp.sum(gradients * normal_block[:, None, :], axis=2)
            integrated = jnp.sum(
                (normal_values * source_weights[None, :]).reshape(
                    (
                        target_stop - target_start,
                        source_stop - source_start,
                        nodes_per_panel,
                    )
                ),
                axis=2,
            )
            columns.append(jax.block_until_ready(integrated))
        rows.append(jnp.concatenate(tuple(columns), axis=1))
    return jnp.concatenate(tuple(rows), axis=0)


def _blocked_wave_value_matrix(
    green: FreeSurfaceGreenRepresentation3D,
    targets: Array,
    panel_points: Array,
    panel_weights: Array,
    nodes_per_panel: int,
    target_block_size: int,
    source_block_size: int,
    /,
) -> Array:
    face_count = int(targets.shape[0])
    rows = []
    for target_start in range(0, face_count, target_block_size):
        target_stop = min(target_start + target_block_size, face_count)
        target_block = targets[target_start:target_stop]
        columns = []
        for source_start in range(0, face_count, source_block_size):
            source_stop = min(source_start + source_block_size, face_count)
            node_start = source_start * nodes_per_panel
            node_stop = source_stop * nodes_per_panel
            source_points = panel_points[node_start:node_stop]
            source_weights = panel_weights[node_start:node_stop]
            values = jax.vmap(
                lambda target: jax.vmap(
                    lambda source: green.wave_correction(target, source)
                )(source_points)
            )(target_block)
            integrated = jnp.sum(
                (values * source_weights[None, :]).reshape(
                    (
                        target_stop - target_start,
                        source_stop - source_start,
                        nodes_per_panel,
                    )
                ),
                axis=2,
            )
            columns.append(jax.block_until_ready(integrated))
        rows.append(jnp.concatenate(tuple(columns), axis=1))
    return jnp.concatenate(tuple(rows), axis=0)


def prepare_free_surface_hydrodynamics_3d(
    region: MeshRegion,
    angular_frequency: float,
    /,
    *,
    gravity: float = 9.80665,
    depth: float | None = None,
    free_surface_z: float = 0.0,
    reference_points: ArrayLike | None = None,
    frame_id: str = "z-up-cartesian",
    unit_system_id: str = "si",
    policy: FreeSurfaceHydrodynamicsPolicy3D | None = None,
    numeric_version: str = "0",
) -> PreparedFreeSurfaceHydrodynamics3D:
    """Prepare a submerged-body DP0 source formulation with an outgoing Green kernel."""
    selected = FreeSurfaceHydrodynamicsPolicy3D() if policy is None else policy
    if not isinstance(selected, FreeSurfaceHydrodynamicsPolicy3D):
        raise TypeError("policy must be FreeSurfaceHydrodynamicsPolicy3D or None.")
    vertices, faces = _mesh_arrays(region)
    face_count = int(faces.shape[0])
    if face_count > selected.max_faces:
        raise ValueError("Body face count exceeds max_faces.")
    dense_entries = 2 * face_count * face_count
    if dense_entries > selected.max_dense_entries:
        raise ValueError("Hydrodynamics dense operators exceed max_dense_entries.")
    omega = float(angular_frequency)
    gravity_ = float(gravity)
    surface = float(free_surface_z)
    depth_ = None if depth is None else float(depth)
    if any(not math.isfinite(value) or value <= 0.0 for value in (omega, gravity_)):
        raise ValueError("angular_frequency and gravity must be finite and positive.")
    if not math.isfinite(surface):
        raise ValueError("free_surface_z must be finite.")
    if depth_ is not None and (not math.isfinite(depth_) or depth_ <= 0.0):
        raise ValueError("depth must be finite and positive when provided.")
    frame = _nonempty(frame_id, "frame_id")
    units = _nonempty(unit_system_id, "unit_system_id")
    scale = max(float(np.max(np.ptp(vertices, axis=0))), 1.0)
    required_clearance = max(
        selected.minimum_geometric_clearance,
        256.0 * np.finfo(float).eps * scale,
    )
    surface_clearance = surface - float(np.max(vertices[:, 2]))
    if surface_clearance <= required_clearance:
        raise ValueError(
            "Radiation/diffraction bodies must be strictly submerged; waterline "
            "or surface-piercing panels are outside this Green route."
        )
    bottom_clearance = math.inf
    if depth_ is not None:
        bottom_clearance = float(np.min(vertices[:, 2])) - (surface - depth_)
        if bottom_clearance <= required_clearance:
            raise ValueError("Finite-depth bodies must lie strictly above the bottom.")
    minimum_clearance = min(surface_clearance, bottom_clearance)

    component_ids_host = np.asarray(
        region.triangle_mesh.topology.face_component_ids, dtype=np.int32
    )
    component_count = int(region.triangle_mesh.topology.num_face_components)
    if reference_points is None:
        references = np.stack(
            [
                np.mean(
                    vertices[np.unique(faces[component_ids_host == component])], axis=0
                )
                for component in range(component_count)
            ]
        )
    else:
        references = np.asarray(reference_points, dtype=float)
        if component_count == 1 and references.shape == (3,):
            references = references[None, :]
    if references.shape != (component_count, 3) or np.any(~np.isfinite(references)):
        raise ValueError("reference_points must have shape (component_count, 3).")

    oracle_bytes = face_count * face_count * np.dtype(np.float64).itemsize
    galerkin_policy = _galerkin_policy_with_oracle(
        selected.galerkin,
        face_count * face_count,
        max(oracle_bytes, 1),
    )
    galerkin = prepare_laplace_single_layer_dp0_3d(
        region,
        policy=galerkin_policy,
        numeric_version=numeric_version,
    )
    if not bool(galerkin.assembly_report.accuracy_supported):
        raise ValueError("Direct Laplace DP0 Galerkin evidence is not supported.")
    green = prepare_free_surface_green_3d(
        omega,
        gravity_,
        minimum_clearance=minimum_clearance,
        depth=depth_,
        free_surface_z=surface,
        frame_id=frame,
        unit_system_id=units,
        policy=selected.green,
    )
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    doubled_areas = np.linalg.norm(cross, axis=1)
    centroids = np.mean(triangles, axis=1)
    normals = cross / doubled_areas[:, None]
    areas = 0.5 * doubled_areas

    centroids_jax = jnp.asarray(centroids, dtype=jnp.float64)
    normals_jax = jnp.asarray(normals, dtype=jnp.float64)
    panel_points = galerkin.panelization.points
    panel_weights = galerkin.panelization.weights
    nodes_per_panel = galerkin.panelization.nodes_per_panel
    source_face_ids = jnp.repeat(jnp.arange(face_count), nodes_per_panel)
    target_face_ids = jnp.arange(face_count)[:, None]
    differences = centroids_jax[:, None, :] - panel_points[None, :, :]
    radius_squared = jnp.sum(differences * differences, axis=2)
    off_diagonal = source_face_ids[None, :] != target_face_ids
    safe_radius_squared = jnp.where(off_diagonal, radius_squared, 1.0)
    direct_gradient = -differences / (
        4.0 * jnp.pi * safe_radius_squared[:, :, None] ** 1.5
    )
    direct_gradient = jnp.where(off_diagonal[:, :, None], direct_gradient, 0.0)
    direct_normal_kernel = jnp.sum(direct_gradient * normals_jax[:, None, :], axis=2)
    integrated_direct_normal = jnp.sum(
        (direct_normal_kernel * panel_weights[None, :]).reshape(
            (face_count, face_count, nodes_per_panel)
        ),
        axis=2,
    )
    target_block_size = min(selected.galerkin.target_block_size, face_count)
    source_block_size = min(selected.galerkin.source_block_size, face_count)
    block_pair_count = target_block_size * source_block_size * nodes_per_panel
    preparation_workspace_bytes = int(
        differences.nbytes
        + direct_gradient.nbytes
        + direct_normal_kernel.nbytes
        + 2 * face_count * face_count * np.dtype(np.complex128).itemsize
        + block_pair_count * green.resources.action_workspace_bytes
    )
    if preparation_workspace_bytes > selected.max_preparation_workspace_bytes:
        raise ValueError(
            "Blocked Green assembly exceeds max_preparation_workspace_bytes."
        )
    integrated_wave_normal = _blocked_wave_normal_matrix(
        green,
        centroids_jax,
        normals_jax,
        panel_points,
        panel_weights,
        nodes_per_panel,
        target_block_size,
        source_block_size,
    )
    boundary_matrix = (
        integrated_direct_normal
        + integrated_wave_normal
        - 0.5 * jnp.eye(face_count, dtype=jnp.complex128)
    )

    integrated_wave = _blocked_wave_value_matrix(
        green,
        centroids_jax,
        panel_points,
        panel_weights,
        nodes_per_panel,
        target_block_size,
        source_block_size,
    )
    direct_trace = jnp.asarray(galerkin.dense_oracle.matrix, dtype=jnp.complex128)
    trace_matrix = direct_trace + integrated_wave
    boundary_operator = DenseLinearOperator(
        boundary_matrix,
        operator_id=canonical_fingerprint(
            {
                "kind": "free-surface-neumann-source-boundary-operator-3d",
                "region": region.feature_id,
                "green": green.representation_id,
            }
        ),
    )
    trace_operator = DenseLinearOperator(
        trace_matrix,
        operator_id=canonical_fingerprint(
            {
                "kind": "free-surface-source-trace-operator-3d",
                "region": region.feature_id,
                "green": green.representation_id,
                "direct": galerkin.assembly_report.report_id,
            }
        ),
    )

    modes = np.zeros((face_count, 6 * component_count), dtype=float)
    mode_names: list[str] = []
    for component in range(component_count):
        selected_faces = component_ids_host == component
        relative = centroids[selected_faces] - references[component]
        modes[selected_faces, 6 * component : 6 * component + 3] = normals[selected_faces]
        modes[selected_faces, 6 * component + 3 : 6 * component + 6] = np.cross(
            relative, normals[selected_faces]
        )
        mode_names.extend(f"body-{component}:{label}" for label in _MODE_LABELS)

    resident_bytes = int(
        boundary_matrix.nbytes
        + trace_matrix.nbytes
        + centroids_jax.nbytes
        + normals_jax.nbytes
        + areas.nbytes
        + modes.nbytes
        + green.resources.resident_bytes
        + galerkin.assembly_report.resident_bytes
    )
    preparation_workspace_bytes = int(preparation_workspace_bytes)
    if resident_bytes > selected.max_resident_bytes:
        raise ValueError("Prepared hydrodynamics arrays exceed max_resident_bytes.")
    finite = (
        jnp.all(jnp.isfinite(boundary_matrix))
        & jnp.all(jnp.isfinite(trace_matrix))
        & jnp.all(jnp.isfinite(jnp.asarray(modes)))
    )
    supported = (
        finite & green.errors.supported & galerkin.assembly_report.accuracy_supported
    )
    report_id = canonical_fingerprint(
        {
            "kind": "free-surface-hydrodynamics-assembly-report-3d",
            "region": region.feature_id,
            "green": green.representation_id,
            "laplace": galerkin.assembly_report.report_id,
            "boundary": array_tree_fingerprint(boundary_matrix),
            "trace": array_tree_fingerprint(trace_matrix),
        }
    )
    report = FreeSurfaceHydrodynamicsAssemblyReport3D(
        laplace_report=galerkin.assembly_report,
        green_tail_bound=green.errors.spectral_tail_envelope_bound,
        dispersion_residual=green.dispersion.residual,
        face_count=face_count,
        component_count=component_count,
        degree_of_freedom_count=6 * component_count,
        boundary_operator_bytes=int(boundary_matrix.nbytes),
        trace_operator_bytes=int(trace_matrix.nbytes),
        resident_bytes=resident_bytes,
        preparation_workspace_bytes=preparation_workspace_bytes,
        maximum_resident_bytes=selected.max_resident_bytes,
        maximum_preparation_workspace_bytes=(selected.max_preparation_workspace_bytes),
        continuum_discretization_error_estimated=False,
        collocation_error_estimated=False,
        finite=finite,
        supported=supported,
        report_id=report_id,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-free-surface-hydrodynamics-3d",
            "report": report_id,
            "frame": frame,
            "units": units,
            "references": array_tree_fingerprint(references),
        }
    )
    return PreparedFreeSurfaceHydrodynamics3D(
        galerkin=galerkin,
        green=green,
        boundary_operator=boundary_operator,
        trace_operator=trace_operator,
        face_centroids=centroids_jax,
        face_normals=normals_jax,
        face_areas=jnp.asarray(areas, dtype=jnp.float64),
        face_component_ids=jnp.asarray(component_ids_host, dtype=jnp.int32),
        rigid_mode_normal_velocity=jnp.asarray(modes, dtype=jnp.float64),
        reference_points=jnp.asarray(references, dtype=jnp.float64),
        assembly_report=report,
        angular_frequency=omega,
        gravity=gravity_,
        depth=depth_,
        free_surface_z=surface,
        component_count=component_count,
        degree_of_freedom_count=6 * component_count,
        mode_names=tuple(mode_names),
        ambient_dimension=3,
        coordinate_convention="right-handed-cartesian-z-up",
        pde_id=_PDE_ID,
        geometry_id=region.feature_id,
        formulation_id=(
            "exterior-single-layer-dp0-neumann-centroid-collocation-with-"
            "galerkin-potential-trace"
        ),
        provider_id="phydrax-dense-dp0-free-surface-green",
        precision_id=_PRECISION_ID,
        frame_id=frame,
        unit_system_id=units,
        time_convention=_TIME_CONVENTION,
        normal_convention=_NORMAL_CONVENTION,
        density_semantics=(
            "one unweighted piecewise-constant single-layer source-strength "
            "coefficient per oriented body face; not body-normal velocity or "
            "area-integrated source flux"
        ),
        non_goals=_NON_GOALS,
        prepared_id=prepared_id,
    )


__all__ = [
    "FreeSurfaceHydrodynamicsAssemblyReport3D",
    "FreeSurfaceHydrodynamicsPolicy3D",
    "HydrostaticProperties3D",
    "PreparedFreeSurfaceHydrodynamics3D",
    "prepare_free_surface_hydrodynamics_3d",
    "prepare_hydrostatic_properties_3d",
]
