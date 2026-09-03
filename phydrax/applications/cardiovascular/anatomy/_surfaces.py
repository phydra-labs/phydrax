#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import CellMesh
from ._roles import CardiacBoundaryRoles


def _nonempty(value: str, description: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{description} must be non-empty.")
    return result


def _edge_sign(first: int, second: int, /) -> int:
    return 1 if first < second else -1


def _oriented_closed_faces(
    triangles: np.ndarray,
    coordinates: np.ndarray,
    tolerance: float,
    /,
) -> tuple[np.ndarray, int, int, float, float]:
    keys = [tuple(sorted(int(value) for value in face)) for face in triangles]
    if any(len(set(key)) != 3 for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Chamber triangles must be nondegenerate and unique.")
    faces = np.asarray(sorted(keys), dtype=np.int32)
    edge_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, (first, second, third) in enumerate(faces):
        for start, stop in ((first, second), (second, third), (third, first)):
            edge = tuple(sorted((int(start), int(stop))))
            edge_owners.setdefault(edge, []).append(
                (face_index, _edge_sign(int(start), int(stop)))
            )
    counts = np.asarray(tuple(len(owners) for owners in edge_owners.values()))
    if np.any(counts != 2):
        raise ValueError("A closed chamber surface requires exactly two faces per edge.")
    neighbours: list[list[tuple[int, bool]]] = [[] for _ in range(faces.shape[0])]
    initial_mismatches = 0
    for owners in edge_owners.values():
        (first_face, first_sign), (second_face, second_sign) = owners
        same = first_sign == second_sign
        initial_mismatches += int(same)
        neighbours[first_face].append((second_face, same))
        neighbours[second_face].append((first_face, same))
    flips = np.full((faces.shape[0],), -1, dtype=np.int8)
    component_count = 0
    for seed in range(faces.shape[0]):
        if flips[seed] >= 0:
            continue
        component_count += 1
        flips[seed] = 0
        pending = [seed]
        while pending:
            current = pending.pop()
            for neighbour, relative_flip in neighbours[current]:
                expected = int(flips[current]) ^ int(relative_flip)
                if flips[neighbour] < 0:
                    flips[neighbour] = expected
                    pending.append(neighbour)
                elif int(flips[neighbour]) != expected:
                    raise ValueError("Chamber surface connectivity is not orientable.")
    if component_count != 1:
        raise ValueError(
            f"One chamber surface must be connected; found {component_count} components."
        )
    oriented = faces.copy()
    oriented[flips.astype(bool), 1:] = oriented[flips.astype(bool), 1:][:, ::-1]
    points = coordinates[oriented]
    double_areas = np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=-1
    )
    minimum_area = float(np.min(double_areas))
    if not np.all(np.isfinite(double_areas)) or np.any(double_areas <= tolerance):
        raise ValueError("Chamber surface contains a geometrically degenerate face.")
    used = np.unique(oriented)
    centered = coordinates - np.mean(coordinates[used], axis=0)
    centered_faces = centered[oriented]
    volume = float(
        np.sum(
            oe.contract(
                "fi,fi->f",
                centered_faces[:, 0],
                np.cross(centered_faces[:, 1], centered_faces[:, 2]),
            )
        )
        / 6.0
    )
    if not np.isfinite(volume) or abs(volume) <= tolerance:
        raise ValueError("Chamber reference surface must enclose nonzero finite volume.")
    global_flip = volume < 0.0
    if global_flip:
        oriented[:, 1:] = oriented[:, 1:][:, ::-1]
        flips = 1 - flips
        volume = -volume
    return (
        oriented,
        int(np.count_nonzero(flips)),
        initial_mismatches,
        volume,
        minimum_area,
    )


def _signed_volume(points: Array, triangles: Array, /) -> tuple[Array, Array]:
    # Any shared origin is valid for a closed surface; the full-vertex centroid
    # also keeps the operation fixed-shape when a mesh carries unused vertices.
    centered = points - jnp.mean(points, axis=0)
    face_points = centered[triangles]
    contributions = (
        oe.contract(
            "fi,fi->f",
            face_points[:, 0],
            jnp.cross(face_points[:, 1], face_points[:, 2]),
        )
        / 6.0
    )
    return jnp.sum(contributions), contributions


def _volume_derivative(points: Array, triangles: Array, /) -> Array:
    centered = points - jnp.mean(points, axis=0)
    face_points = centered[triangles]
    local_first = jnp.cross(face_points[:, 1], face_points[:, 2]) / 6.0
    local_second = jnp.cross(face_points[:, 2], face_points[:, 0]) / 6.0
    local_third = jnp.cross(face_points[:, 0], face_points[:, 1]) / 6.0
    derivative = jnp.zeros_like(points)
    derivative = derivative.at[triangles[:, 0]].add(local_first)
    derivative = derivative.at[triangles[:, 1]].add(local_second)
    derivative = derivative.at[triangles[:, 2]].add(local_third)
    return derivative


class ChamberSurfaceTopologyEvidence(StrictModule, NonTrainableState):
    """Reference closure, orientation, and component evidence."""

    edge_incidence_counts: Array
    component_count: Array
    initial_orientation_mismatch_count: Array
    reoriented_face_count: Array
    reference_signed_volume: Array
    reference_minimum_double_area: Array
    closed: Array
    orientable: Array
    outward: Array
    successful: Array

    def __init__(self, **values):
        self.edge_incidence_counts = jnp.asarray(
            values["edge_incidence_counts"], dtype=jnp.int32
        )
        self.component_count = jnp.asarray(values["component_count"], dtype=jnp.int32)
        self.initial_orientation_mismatch_count = jnp.asarray(
            values["initial_orientation_mismatch_count"], dtype=jnp.int32
        )
        self.reoriented_face_count = jnp.asarray(
            values["reoriented_face_count"], dtype=jnp.int32
        )
        self.reference_signed_volume = jnp.asarray(values["reference_signed_volume"])
        self.reference_minimum_double_area = jnp.asarray(
            values["reference_minimum_double_area"]
        )
        self.closed = jnp.asarray(values["closed"], dtype=bool)
        self.orientable = jnp.asarray(values["orientable"], dtype=bool)
        self.outward = jnp.asarray(values["outward"], dtype=bool)
        self.successful = jnp.asarray(values["successful"], dtype=bool)


class ChamberSurfacePlan(StrictModule, NonTrainableState):
    """Fixed-topology plan for one connected closed chamber surface."""

    chamber_name: str = eqx.field(static=True)
    reference_coordinates: Array
    triangles: Array
    vertex_global_ids: Array
    geometric_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        chamber_name: str,
        reference_coordinates: ArrayLike,
        triangles: ArrayLike,
        /,
        *,
        vertex_global_ids: ArrayLike | None = None,
        geometric_tolerance: float = 0.0,
    ):
        name = _nonempty(chamber_name, "Chamber name")
        coordinates = np.asarray(reference_coordinates, dtype=float)
        faces = np.asarray(triangles, dtype=np.int32)
        tolerance = float(geometric_tolerance)
        if coordinates.ndim != 2 or coordinates.shape[0] < 4 or coordinates.shape[1] != 3:
            raise ValueError(
                "Chamber coordinates must have shape (vertex_count >= 4, 3)."
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Chamber coordinates must be finite.")
        if faces.ndim != 2 or faces.shape[0] < 4 or faces.shape[1] != 3:
            raise ValueError("Chamber triangles must have shape (face_count >= 4, 3).")
        if np.any(faces < 0) or np.any(faces >= coordinates.shape[0]):
            raise ValueError("Chamber triangles index undeclared vertices.")
        keys = [tuple(sorted(int(value) for value in face)) for face in faces]
        if any(len(set(key)) != 3 for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Chamber triangles must be nondegenerate and unique.")
        faces = np.asarray(sorted(keys), dtype=np.int32)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("geometric_tolerance must be finite and non-negative.")
        global_ids = (
            np.arange(coordinates.shape[0], dtype=np.int64)
            if vertex_global_ids is None
            else np.asarray(vertex_global_ids, dtype=np.int64)
        )
        if global_ids.shape != (coordinates.shape[0],):
            raise ValueError("vertex_global_ids must have shape (vertex_count,).")
        if np.any(global_ids < 0) or np.unique(global_ids).size != global_ids.size:
            raise ValueError("vertex_global_ids must be unique and non-negative.")
        self.chamber_name = name
        self.reference_coordinates = jnp.asarray(coordinates)
        self.triangles = jnp.asarray(faces)
        self.vertex_global_ids = jnp.asarray(global_ids)
        self.geometric_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "chamber-surface-plan",
                "name": name,
                "coordinates": array_tree_fingerprint(coordinates),
                "triangles": array_tree_fingerprint(faces),
                "vertex_global_ids": array_tree_fingerprint(global_ids),
                "geometric_tolerance": tolerance,
            }
        )

    @classmethod
    def from_boundary_roles(
        cls,
        chamber_name: str,
        mesh: CellMesh,
        roles: CardiacBoundaryRoles,
        role_names: Sequence[str],
        /,
        *,
        geometric_tolerance: float = 0.0,
    ) -> ChamberSurfacePlan:
        """Create a plan from an explicitly selected, closed set of role facets."""
        if not isinstance(mesh, CellMesh) or not isinstance(roles, CardiacBoundaryRoles):
            raise TypeError("mesh and roles must be CellMesh/CardiacBoundaryRoles.")
        if roles.mesh.mesh_id != mesh.mesh_id:
            raise ValueError("Boundary roles and chamber surface must share one mesh.")
        names = tuple(str(name) for name in role_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("role_names must be non-empty and unique.")
        face_indices = np.concatenate(
            tuple(np.asarray(roles.face_indices(name), dtype=np.int32) for name in names)
        )
        triangles = np.asarray(mesh.connectivity.faces, dtype=np.int32)[face_indices]
        return cls(
            chamber_name,
            mesh.coordinates,
            triangles,
            vertex_global_ids=mesh.vertex_global_ids,
            geometric_tolerance=geometric_tolerance,
        )

    def prepare(self, /) -> OrientedChamberSurface:
        coordinates = np.asarray(self.reference_coordinates, dtype=float)
        triangles = np.asarray(self.triangles, dtype=np.int32)
        oriented, reoriented, mismatches, volume, minimum_area = _oriented_closed_faces(
            triangles, coordinates, self.geometric_tolerance
        )
        edge_counts: dict[tuple[int, int], int] = {}
        for first, second, third in oriented:
            for start, stop in ((first, second), (second, third), (third, first)):
                edge = tuple(sorted((int(start), int(stop))))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        evidence = ChamberSurfaceTopologyEvidence(
            edge_incidence_counts=np.asarray(tuple(edge_counts.values()), dtype=np.int32),
            component_count=1,
            initial_orientation_mismatch_count=mismatches,
            reoriented_face_count=reoriented,
            reference_signed_volume=volume,
            reference_minimum_double_area=minimum_area,
            closed=True,
            orientable=True,
            outward=True,
            successful=True,
        )
        return OrientedChamberSurface(
            self.chamber_name,
            self.reference_coordinates,
            oriented,
            self.vertex_global_ids,
            evidence,
            geometric_tolerance=self.geometric_tolerance,
            surface_id=canonical_fingerprint(
                {"kind": "oriented-chamber-surface", "plan": self.plan_id}
            ),
        )


class CavityVolumeEvidence(StrictModule, NonTrainableState):
    """Volume, closure, translation, derivative, and orientation evidence."""

    face_signed_contributions: Array
    signed_volume: Array
    minimum_double_area: Array
    closure_residual_norm: Array
    translation_volume_error: Array
    translation_derivative_norm: Array
    finite: Array
    positive_orientation: Array
    successful: Array

    def __init__(self, **values):
        self.face_signed_contributions = jnp.asarray(values["face_signed_contributions"])
        self.signed_volume = jnp.asarray(values["signed_volume"])
        self.minimum_double_area = jnp.asarray(values["minimum_double_area"])
        self.closure_residual_norm = jnp.asarray(values["closure_residual_norm"])
        self.translation_volume_error = jnp.asarray(values["translation_volume_error"])
        self.translation_derivative_norm = jnp.asarray(
            values["translation_derivative_norm"]
        )
        self.finite = jnp.asarray(values["finite"], dtype=bool)
        self.positive_orientation = jnp.asarray(
            values["positive_orientation"], dtype=bool
        )
        self.successful = jnp.asarray(values["successful"], dtype=bool)


class CavityVolumeResult(StrictModule):
    """Committed cavity volume and derivative with respect to all coordinates."""

    volume: Array
    coordinate_derivative: Array
    evidence: CavityVolumeEvidence
    result_id: str = eqx.field(static=True)


class CavityVolumeCandidate(StrictModule):
    """Candidate cavity volume that cannot commit after inversion/degeneracy."""

    volume: Array
    coordinate_derivative: Array
    evidence: CavityVolumeEvidence
    candidate_id: str = eqx.field(static=True)

    def commit(self, /) -> CavityVolumeResult:
        checked = eqx.error_if(
            self.volume,
            ~self.evidence.successful,
            "Cannot commit invalid or inverted chamber cavity volume.",
        )
        return CavityVolumeResult(
            checked,
            self.coordinate_derivative,
            self.evidence,
            canonical_fingerprint(
                {"kind": "committed-cavity-volume", "candidate": self.candidate_id}
            ),
        )


class OrientedChamberSurface(StrictModule, NonTrainableState):
    """Prepared closed outward-oriented chamber with fixed differentiation topology."""

    chamber_name: str = eqx.field(static=True)
    reference_coordinates: Array
    triangles: Array
    vertex_global_ids: Array
    topology_evidence: ChamberSurfaceTopologyEvidence
    geometric_tolerance: float = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)

    def __init__(
        self,
        chamber_name: str,
        reference_coordinates: ArrayLike,
        triangles: ArrayLike,
        vertex_global_ids: ArrayLike,
        topology_evidence: ChamberSurfaceTopologyEvidence,
        /,
        *,
        geometric_tolerance: float,
        surface_id: str,
    ):
        self.chamber_name = _nonempty(chamber_name, "Chamber name")
        self.reference_coordinates = jnp.asarray(reference_coordinates)
        self.triangles = jnp.asarray(triangles, dtype=jnp.int32)
        self.vertex_global_ids = jnp.asarray(vertex_global_ids, dtype=jnp.int64)
        if not isinstance(topology_evidence, ChamberSurfaceTopologyEvidence):
            raise TypeError("topology_evidence must be ChamberSurfaceTopologyEvidence.")
        self.topology_evidence = topology_evidence
        self.geometric_tolerance = float(geometric_tolerance)
        self.surface_id = _nonempty(surface_id, "surface_id")

    def evaluate(self, coordinates: ArrayLike | None = None, /) -> CavityVolumeCandidate:
        """Evaluate differentiable volume and exact analytic coordinate derivative."""
        points = jnp.asarray(
            self.reference_coordinates if coordinates is None else coordinates,
            dtype=self.reference_coordinates.dtype,
        )
        if points.shape != self.reference_coordinates.shape:
            raise ValueError(
                "Chamber coordinate updates must preserve fixed topology shape."
            )
        volume, contributions = _signed_volume(points, self.triangles)
        derivative = _volume_derivative(points, self.triangles)
        face_points = points[self.triangles]
        area_vectors = jnp.cross(
            face_points[:, 1] - face_points[:, 0],
            face_points[:, 2] - face_points[:, 0],
        )
        double_areas = jnp.sqrt(jnp.sum(area_vectors * area_vectors, axis=-1))
        closure_residual = jnp.sqrt(jnp.sum(jnp.sum(area_vectors, axis=0) ** 2))
        shift = jnp.asarray((0.731, -0.419, 0.263), dtype=points.dtype)
        translated_volume, _ = _signed_volume(points + shift, self.triangles)
        translation_error = jnp.abs(translated_volume - volume)
        translation_derivative = jnp.sqrt(jnp.sum(jnp.sum(derivative, axis=0) ** 2))
        finite = (
            jnp.isfinite(volume)
            & jnp.all(jnp.isfinite(contributions))
            & jnp.all(jnp.isfinite(derivative))
            & jnp.all(jnp.isfinite(double_areas))
        )
        scale = jnp.maximum(jnp.sum(double_areas), 1.0)
        roundoff = 256.0 * jnp.finfo(points.dtype).eps * scale
        nondegenerate = jnp.all(double_areas > self.geometric_tolerance)
        successful = (
            self.topology_evidence.successful
            & finite
            & nondegenerate
            & (volume > self.geometric_tolerance)
            & (closure_residual <= roundoff)
            & (translation_error <= roundoff)
            & (translation_derivative <= roundoff)
        )
        evidence = CavityVolumeEvidence(
            face_signed_contributions=contributions,
            signed_volume=volume,
            minimum_double_area=jnp.min(double_areas),
            closure_residual_norm=closure_residual,
            translation_volume_error=translation_error,
            translation_derivative_norm=translation_derivative,
            finite=finite,
            positive_orientation=volume > self.geometric_tolerance,
            successful=successful,
        )
        return CavityVolumeCandidate(
            volume,
            derivative,
            evidence,
            candidate_id=canonical_fingerprint(
                {"kind": "cavity-volume-candidate", "surface": self.surface_id}
            ),
        )


def prepare_chamber_surface(plan: ChamberSurfacePlan, /) -> OrientedChamberSurface:
    """Prepare deterministic closure and outward orientation for a chamber."""
    if not isinstance(plan, ChamberSurfacePlan):
        raise TypeError("plan must be a ChamberSurfacePlan.")
    return plan.prepare()


def evaluate_cavity_volume(
    surface: OrientedChamberSurface,
    coordinates: ArrayLike | None = None,
    /,
) -> CavityVolumeCandidate:
    """Evaluate a chamber volume candidate on fixed topology."""
    if not isinstance(surface, OrientedChamberSurface):
        raise TypeError("surface must be an OrientedChamberSurface.")
    return surface.evaluate(coordinates)


__all__ = [
    "CavityVolumeCandidate",
    "CavityVolumeEvidence",
    "CavityVolumeResult",
    "ChamberSurfacePlan",
    "ChamberSurfaceTopologyEvidence",
    "OrientedChamberSurface",
    "evaluate_cavity_volume",
    "prepare_chamber_surface",
]
