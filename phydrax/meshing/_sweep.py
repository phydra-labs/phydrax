#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native extrusion/revolution and explicitly scheduled rigid-frame sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import charge_native_geometry_queries, current_native_execution_budget
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_geometry import swept_coordinate_element
from ..discretization._cell_geometry_validity import CellValidityCertificate
from ..discretization.fem._reference import FiniteElementSpec
from ..geometry._mapped_reference_domain import mapped_source_corner_coordinates
from ..geometry._mesh_certificates import (
    GlobalEmbeddingCertificate,
    MeshCertificateLimits,
)
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._controls import LayerSchedule
from ._quad_generation import _family_host_array
from ._structured import certify_constructed_mesh


class SweepMapKind(StrEnum):
    EXTRUSION = "extrusion"
    REVOLUTION = "revolution"
    FRAMES = "frames"


@final
class SweepControl(StrictModule, NonTrainableState):
    """Physical layer stations and a declared map; no automatic cap matching.

    Extrusion thicknesses are distances along the unit axis. Revolution
    thicknesses are angles in radians, with positive direction around the unit
    axis. Frame schedules are absolute rigid transforms of the profile; the
    coordinate map interpolates transformed source controls affinely between
    stations, not by an inferred rigid motion or hidden curved reconstruction.
    ``closed=True`` declares the exact last-to-first source-vertex seam.
    """

    schedule: LayerSchedule
    origin: Array
    axis: Array
    frames: Array
    translations: Array
    kind: SweepMapKind = eqx.field(static=True)
    map_id: str = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: SweepMapKind,
        schedule: LayerSchedule,
        map_id: str,
        /,
        *,
        origin: ArrayLike | None = None,
        axis: ArrayLike | None = None,
        frames: ArrayLike | None = None,
        translations: ArrayLike | None = None,
        closed: bool = False,
    ) -> None:
        if not isinstance(kind, SweepMapKind) or not isinstance(schedule, LayerSchedule):
            raise TypeError("A sweep requires SweepMapKind and LayerSchedule.")
        if not isinstance(closed, bool):
            raise TypeError("closed must be bool.")
        if not map_id.strip():
            raise ValueError("A sweep requires explicit source-map identity.")
        center = (
            np.zeros((3,), dtype=np.float64)
            if origin is None
            else np.asarray(origin, dtype=np.float64)
        )
        direction = (
            np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
            if axis is None
            else np.asarray(axis, dtype=np.float64)
        )
        if (
            center.shape != (3,)
            or direction.shape != (3,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(direction))
        ):
            raise ValueError("Sweep origin and axis must be finite three-vectors.")
        if abs(np.linalg.norm(direction) - 1.0) > 1e-12:
            raise ValueError("Sweep axis must be explicitly normalized.")
        count = len(schedule.thicknesses) + 1
        match kind:
            case SweepMapKind.EXTRUSION | SweepMapKind.REVOLUTION:
                if frames is not None or translations is not None:
                    raise ValueError("Analytic sweep maps do not accept frame overrides.")
                rotations = np.zeros((0, 3, 3), dtype=np.float64)
                offsets = np.zeros((0, 3), dtype=np.float64)
                if kind is SweepMapKind.EXTRUSION and closed:
                    raise ValueError(
                        "A positive straight extrusion cannot have a closed seam."
                    )
                if kind is SweepMapKind.REVOLUTION:
                    angle = schedule.total_thickness
                    if closed and abs(angle - 2.0 * np.pi) > 1e-12:
                        raise ValueError(
                            "A closed revolution must declare exactly one full turn."
                        )
                    if not closed and angle >= 2.0 * np.pi:
                        raise ValueError(
                            "A closed revolution requires an explicitly glued seam with closed=True."
                        )
            case SweepMapKind.FRAMES:
                if frames is None or translations is None:
                    raise ValueError(
                        "A frame sweep requires all rotations and translations."
                    )
                rotations = np.asarray(frames, dtype=np.float64)
                offsets = np.asarray(translations, dtype=np.float64)
                if (
                    rotations.shape != (count, 3, 3)
                    or offsets.shape != (count, 3)
                    or not np.all(np.isfinite(rotations))
                    or not np.all(np.isfinite(offsets))
                ):
                    raise ValueError(
                        "Frame schedule shape must match all physical stations."
                    )
                gram = np.swapaxes(rotations, -1, -2) @ rotations
                if not np.allclose(
                    gram, np.eye(3, dtype=np.float64), atol=1e-12, rtol=0.0
                ) or np.any(np.linalg.det(rotations) <= 0.0):
                    raise ValueError("Sweep frames must be proper orthonormal rotations.")
                if closed and (
                    not np.array_equal(rotations[-1], rotations[0])
                    or not np.array_equal(offsets[-1], offsets[0])
                ):
                    raise ValueError(
                        "A closed frame sweep requires exact first/last source transforms."
                    )
            case _:
                raise ValueError("Unknown sweep map kind.")
        self.kind = kind
        self.schedule = schedule
        self.map_id = map_id
        self.closed = closed
        self.origin = jnp.asarray(center, dtype=jnp.float64)
        self.axis = jnp.asarray(direction, dtype=jnp.float64)
        self.frames = jnp.asarray(rotations, dtype=jnp.float64)
        self.translations = jnp.asarray(offsets, dtype=jnp.float64)
        self.control_id = canonical_fingerprint(
            {
                "kind": "sweep-control",
                "map": map_id,
                "map_kind": kind.value,
                "schedule": schedule.schedule_id,
                "origin": center,
                "axis": direction,
                "frames": array_tree_fingerprint(rotations),
                "translations": offsets,
                "closed": closed,
            }
        )


@dataclass(frozen=True, slots=True)
class SweepConstruction:
    mesh: CellMesh
    geometry: CellGeometrySpec
    source_vertex_ids: np.ndarray
    layer_vertex_ids: np.ndarray
    stations: np.ndarray
    measured_layer_thicknesses: np.ndarray
    validity: CellValidityCertificate
    embedding: GlobalEmbeddingCertificate
    control_id: str


def _mapped_stations(
    points: np.ndarray, control: SweepControl, /
) -> tuple[np.ndarray, np.ndarray]:
    stations = _family_host_array((control.schedule.layer_count + 1,), np.float64)
    stations[0] = 0.0
    np.cumsum(
        np.asarray(control.schedule.thicknesses, dtype=np.float64), out=stations[1:]
    )
    mapped = _family_host_array(
        (stations.size, points.shape[0], points.shape[1]), np.float64
    )
    origin = np.asarray(control.origin, dtype=np.float64)
    axis = np.asarray(control.axis, dtype=np.float64)
    map_count = stations.size * points.shape[0]
    charge_native_geometry_queries(map_count, work_units=map_count)
    match control.kind:
        case SweepMapKind.EXTRUSION:
            mapped[:] = points[None, :, :] + stations[:, None, None] * axis[None, None, :]
        case SweepMapKind.REVOLUTION:
            displacement = points - origin
            parallel = (displacement @ axis)[:, None] * axis
            radial = displacement - parallel
            if np.any(np.linalg.norm(radial, axis=-1) <= 1e-12):
                raise ValueError(
                    "Revolution profile touches the axis: prism/hex core correspondence collapses."
                )
            angles = stations[:, None, None]
            mapped[:] = (
                origin
                + parallel[None, :, :]
                + np.cos(angles) * radial[None, :, :]
                + np.sin(angles) * np.cross(axis, radial)[None, :, :]
            )
            mapped[0] = points
        case SweepMapKind.FRAMES:
            np.matmul(
                points[None, :, :],
                np.swapaxes(np.asarray(control.frames), -1, -2),
                out=mapped,
            )
            mapped += np.asarray(control.translations)[:, None, :]
        case _:
            raise ValueError("Unknown sweep map kind.")
    return mapped, stations


def _column_connectivity(
    rows: np.ndarray, layers: int, stride: int, closed: bool, /
) -> np.ndarray:
    node_layers = layers if closed else layers + 1
    if node_layers * stride - 1 > np.iinfo(np.int32).max:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Swept connectivity exceeds the canonical int32 node range.",
            stage="sweep_topology",
        )
    count, arity = rows.shape
    columns = _family_host_array((layers * count, 2 * arity), np.int32)
    budget = current_native_execution_budget()
    if budget is not None:
        budget.charge(work=columns.size)
    for layer in range(layers):
        lower = columns[layer * count : (layer + 1) * count, :arity]
        upper = columns[layer * count : (layer + 1) * count, arity:]
        np.add(rows, layer * stride, out=lower)
        np.add(
            rows, 0 if closed and layer == layers - 1 else (layer + 1) * stride, out=upper
        )
    return columns


def generate_sweep(
    profile: CellMesh,
    control: SweepControl,
    /,
    *,
    name: str = "sweep",
    certificate_limits: MeshCertificateLimits | None = None,
    profile_geometry: CellGeometrySpec | None = None,
) -> SweepConstruction:
    """Carry full polynomial source coordinates into hex/prism columns.

    Revolution evaluates the analytic map at stations; intervals retain the
    explicitly linear station coordinate space. Source fidelity to a curved
    analytic revolution must therefore be bounded, never claimed exact.
    """
    if not isinstance(profile, CellMesh) or not isinstance(control, SweepControl):
        raise TypeError("Sweeps require a CellMesh profile and SweepControl.")
    if profile.topological_dimension != 2 or profile.ambient_dimension not in (2, 3):
        raise ValueError("Sweep profiles must be two-dimensional cells in 2D or 3D.")
    if profile.periodic_topology is not None:
        raise ValueError(
            "Sweep profile periodic topology needs explicit lifted correspondence."
        )
    if any(
        block.cell_kind not in ("triangle", "quadrilateral") for block in profile.blocks
    ):
        raise ValueError("Prism/hex sweeps require triangular or quadrilateral profiles.")
    source_geometry = (
        CellGeometrySpec.affine(profile) if profile_geometry is None else profile_geometry
    )
    if not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("profile_geometry must be CellGeometrySpec or None.")
    source_elements, source_routes, source_coordinates = source_geometry.resolve(profile)
    corner_count = sum(block.vertices.size for block in profile.blocks)
    charge_native_geometry_queries(corner_count, work_units=corner_count)
    source_corners = mapped_source_corner_coordinates(profile, source_geometry)
    if not np.array_equal(source_corners, np.asarray(profile.coordinates)):
        raise ValueError(
            "The source coordinate maps must retain the exact profile topology corners."
        )
    if any(not isinstance(element, FiniteElementSpec) for element in source_elements):
        raise ValueError(
            "Sweeps require unrestricted canonical polynomial source elements."
        )
    points = np.asarray(profile.coordinates, dtype=np.float64)
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    mapped, stations = _mapped_stations(points, control)
    count = points.shape[0]
    layers = stations.size - 1
    if control.closed:
        if layers < 3:
            raise ValueError(
                "A closed sweep requires at least three noncollapsed layers."
            )
        seam_residual = np.max(np.linalg.norm(mapped[-1] - mapped[0], axis=-1))
        if seam_residual > 1e-12:
            raise ValueError(
                "The declared closed sweep has incompatible source/cap correspondence."
            )
        mapped[-1] = mapped[0]
    blocks = []
    cell_offset = 0
    for block in profile.blocks:
        cells = np.asarray(block.vertices, dtype=np.int32)
        columns = _column_connectivity(cells, layers, count, control.closed)
        kind = "prism" if block.cell_kind == "triangle" else "hexahedron"
        blocks.append(
            CellBlock(
                f"{name}:{block.name}",
                kind,
                columns,
                global_ids=np.arange(
                    cell_offset, cell_offset + columns.shape[0], dtype=np.int64
                ),
            )
        )
        cell_offset += columns.shape[0]
    mesh_points = mapped[:-1] if control.closed else mapped
    mesh = CellMesh(mesh_points.reshape(-1, 3), tuple(blocks))
    controls = np.asarray(source_coordinates, dtype=np.float64)
    if controls.shape[1] == 2:
        controls = np.pad(controls, ((0, 0), (0, 1)))
    mapped_controls, _ = _mapped_stations(controls, control)
    if control.closed and not np.allclose(
        mapped_controls[-1], mapped_controls[0], atol=1e-12, rtol=0.0
    ):
        raise ValueError(
            "The closed sweep seam must retain every source coordinate control."
        )
    control_count = controls.shape[0]
    elements = {}
    geometry_routes = {}
    for block, source_element, source_route in zip(
        mesh.blocks, source_elements, source_routes, strict=True
    ):
        if not isinstance(source_element, FiniteElementSpec):
            raise TypeError(
                "Swept source coordinate elements must support canonical tabulation."
            )
        elements[block.name] = swept_coordinate_element(source_element)
        geometry_routes[block.name] = _column_connectivity(
            np.asarray(source_route, dtype=np.int32),
            layers,
            control_count,
            control.closed,
        )
    geometry_points = mapped_controls[:-1] if control.closed else mapped_controls
    geometry = CellGeometrySpec(elements, geometry_routes, geometry_points.reshape(-1, 3))
    validity, embedding = certify_constructed_mesh(
        mesh, geometry=geometry, certificate_limits=certificate_limits
    )
    layer_ids = np.arange(mapped.shape[0] * count, dtype=np.int64).reshape(
        mapped.shape[:2]
    )
    if control.closed:
        layer_ids[-1] = layer_ids[0]
    differences = _family_host_array((layers, count, 3), np.float64)
    measured = _family_host_array((layers, count), np.float64)
    np.subtract(mapped[1:], mapped[:-1], out=differences)
    np.square(differences, out=differences)
    np.sum(differences, axis=-1, out=measured)
    np.sqrt(measured, out=measured)
    return SweepConstruction(
        mesh,
        geometry,
        np.asarray(profile.vertex_global_ids, dtype=np.int64),
        layer_ids,
        stations,
        measured,
        validity,
        embedding,
        control.control_id,
    )


__all__ = ["SweepMapKind", "SweepControl", "SweepConstruction", "generate_sweep"]
