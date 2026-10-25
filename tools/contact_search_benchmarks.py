#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated canonical contact search through contact kinematics.

Host dense and sweep-and-prune authorities and the compiled and LBVH device
backends search one scene into the shared canonical candidate epoch, and every
epoch is passed through ``evaluate_contact_kinematics``. An independent NumPy
reference re-derives the documented candidate contract from the generated host
geometry arrays, never from a search backend: full-path primitive boxes are
hulls of per-vertex endpoint boxes, finite pairs activate within
``activation + 2 envelope + extents`` and static planes within the analytic
support bound ``min_box(n.x) - d <= activation + envelope + extents``; pairs
sharing a vertex or joining two static features are illegal. Planes are
analytic features and never finite boxes. Analytic sphere, capsule, segment,
triangle and plane distances check every valid kinematics gap.

Diagnostic rows (``--suite smoke`` or ``--suite scaling``; the default suite is
the legacy 256-segment self-contact curve) record geometry generation,
topology preparation, transfer, plan preparation, numeric refresh, and the
lowering, compilation, first and warmed calls of search, kinematics and the
fused search-to-kinematics pipeline, with compiler and logical retained bytes.
Refused, exhausted and failed statuses are reported beside their timings;
ratios are emitted only for matched successful work and are not total-cost
claims. Provider and allocator memory is not measured.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    CompilerEvidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


_D = phx.discretization
Kind = _D.ContactStencilKind
Profile = _D.ContactPrimitiveProfile
Status = _D.ContactSearchStatus

type SearchScene = _D.PreparedCollisionScene | _D.ContactParticipantScene
type DevicePlan = _D.CompiledContactSearchPlan | _D.LBVHContactSearchPlan
type HostPlan = _D.DenseContactSearchPlan | _D.SweepAndPruneContactSearchPlan
type Row = tuple[int, int, int, int]
type Rows = dict[int, set[Row]]
type Family = Literal["curve", "sheets", "spheres", "touch"]
type Distribution = Literal["uniform", "clustered", "tail"]
type CapacityPolicy = Literal["legacy", "slack", "shortfall"]

_KIND_ORDER = (
    Kind.VERTEX_VERTEX,
    Kind.EDGE_VERTEX,
    Kind.EDGE_EDGE,
    Kind.FACE_VERTEX,
    Kind.VERTEX_PLANE,
    Kind.EDGE_PLANE,
)
_SYMMETRIC = (Kind.VERTEX_VERTEX, Kind.EDGE_EDGE)
_PLANE_KINDS = (Kind.VERTEX_PLANE, Kind.EDGE_PLANE)
_HOST_BACKENDS = ("dense", "sweep-and-prune")
_DEVICE_BACKENDS = ("compiled", "lbvh")
_BACKENDS = (*_HOST_BACKENDS, *_DEVICE_BACKENDS)
_LEGACY_EDGE_VERTEX_CAPACITY = 4096
_STEP_SIZE = 0.01
_HLO_OPERATIONS = ("while", "conditional", "custom-call", "fusion", "sort", "scatter")
_HLO_PATTERN = re.compile(r"\b(while|conditional|custom-call|fusion|sort|scatter)\(")
_SUITE_DEFAULTS = {
    "default": (1, 50, 5),
    "smoke": (1, 3, 2),
    "scaling": (2, 10, 3),
}


def _kind_name(kind: int, /) -> str:
    return Kind(int(kind)).name.lower()


def _finite(value: Any, /) -> float | None:
    number = float(np.asarray(value))
    return number if math.isfinite(number) else None


def _status_name(status: Any, /) -> str:
    return Status(int(np.asarray(status))).name


# --------------------------------------------------------------------------
# Case specification and settings
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """One bounded diagnostic row; expectations name independent statuses."""

    name: str
    family: Family
    size: int
    capsules: int = 0
    distribution: Distribution = "uniform"
    moving: bool = False
    capacity: CapacityPolicy = "slack"
    visit_budget: int | None = None
    expected: tuple[tuple[str, str], ...] = ()
    expect_narrow_failure: bool = False
    seed: int = 0

    def expected_status(self, backend: str, /) -> str:
        return dict(self.expected).get(backend, Status.SUCCESS.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "size": self.size,
            "capsules": self.capsules,
            "distribution": self.distribution,
            "moving": self.moving,
            "capacity_policy": self.capacity,
            "visit_budget": self.visit_budget,
            "expected_search_status": {
                backend: self.expected_status(backend) for backend in _BACKENDS
            },
            "expect_narrow_failure": self.expect_narrow_failure,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class Settings:
    suite: str
    dtype: str
    warmup: int
    repeats: int
    host_repeats: int
    capacity_factor: float
    visit_budget: int
    compiled_row_limit: int
    dense_pair_limit: int
    oracle_pair_limit: int
    gap_tolerance_factor: float

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "dtype": self.dtype,
            "warmup": self.warmup,
            "device_repeats": self.repeats,
            "host_repeats": self.host_repeats,
            "capacity_factor": self.capacity_factor,
            "visit_budget": self.visit_budget,
            "compiled_row_limit": self.compiled_row_limit,
            "dense_pair_limit": self.dense_pair_limit,
            "oracle_pair_limit": self.oracle_pair_limit,
            "gap_tolerance_factor": self.gap_tolerance_factor,
            "step_size": _STEP_SIZE,
        }


def _suite(name: str, /) -> tuple[CaseSpec, ...]:
    overflow = tuple((backend, Status.CANDIDATE_OVERFLOW.name) for backend in _BACKENDS)
    if name == "default":
        return (CaseSpec("curve-2d-legacy", "curve", 256, capacity="legacy"),)
    if name == "smoke":
        return (
            CaseSpec("curve-2d-self", "curve", 32),
            CaseSpec("sheets-3d-static-plane", "sheets", 3, seed=1),
            CaseSpec("spheres-uniform-static", "spheres", 16, capsules=6, seed=2),
            CaseSpec(
                "spheres-clustered-moving",
                "spheres",
                16,
                capsules=6,
                distribution="clustered",
                moving=True,
                seed=3,
            ),
            CaseSpec(
                "spheres-tail-static",
                "spheres",
                16,
                capsules=6,
                distribution="tail",
                seed=4,
            ),
            CaseSpec(
                "capacity-shortfall",
                "spheres",
                16,
                capsules=6,
                capacity="shortfall",
                expected=overflow,
                seed=2,
            ),
            CaseSpec(
                "visit-exhaustion",
                "spheres",
                16,
                capsules=6,
                visit_budget=1,
                expected=(("lbvh", Status.TRAVERSAL_EXHAUSTED.name),),
                seed=2,
            ),
            CaseSpec("degenerate-touch-2d", "touch", 1, expect_narrow_failure=True),
        )
    if name == "scaling":
        return (
            CaseSpec("curve-2d-256", "curve", 256),
            CaseSpec("curve-2d-1024", "curve", 1024),
            CaseSpec("curve-2d-4096", "curve", 4096),
            CaseSpec("sheets-3d-4", "sheets", 4, seed=11),
            CaseSpec("sheets-3d-8-moving", "sheets", 8, moving=True, seed=12),
            CaseSpec("sheets-3d-16", "sheets", 16, seed=13),
            CaseSpec("spheres-uniform-64", "spheres", 64, capsules=16, seed=21),
            CaseSpec("spheres-uniform-256", "spheres", 256, capsules=64, seed=22),
            CaseSpec("spheres-uniform-1024", "spheres", 1024, capsules=256, seed=23),
            CaseSpec(
                "spheres-clustered-moving-256",
                "spheres",
                256,
                capsules=64,
                distribution="clustered",
                moving=True,
                seed=24,
            ),
            CaseSpec(
                "spheres-clustered-moving-1024",
                "spheres",
                1024,
                capsules=256,
                distribution="clustered",
                moving=True,
                seed=25,
            ),
            CaseSpec(
                "spheres-tail-256",
                "spheres",
                256,
                capsules=64,
                distribution="tail",
                seed=26,
            ),
        )
    raise ValueError(f"Unknown contact search suite {name!r}.")


# --------------------------------------------------------------------------
# Host geometry generation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HostParticipant:
    vertex_ids: np.ndarray
    start: np.ndarray
    end: np.ndarray
    edges: np.ndarray
    edge_ids: np.ndarray
    faces: np.ndarray
    face_ids: np.ndarray
    radius: np.ndarray
    static: bool
    participant_id: int


@dataclass(frozen=True, slots=True)
class HostPlane:
    normal: np.ndarray
    offset: float
    feature_id: int
    participant_id: int


@dataclass(frozen=True, slots=True)
class HostGeometry:
    dimension: int
    profile: Profile
    participants: tuple[HostParticipant, ...]
    planes: tuple[HostPlane, ...]
    collision_scene: bool
    activation: float
    envelope: float


def _participant(
    vertex_ids: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius: np.ndarray,
    dtype: np.dtype,
    /,
    *,
    edges: np.ndarray,
    edge_ids: np.ndarray,
    faces: np.ndarray | None = None,
    face_ids: np.ndarray | None = None,
    static: bool,
    participant_id: int,
) -> HostParticipant:
    # Round through the run dtype so the reference sees the searched values.
    def rounded(value: np.ndarray, /) -> np.ndarray:
        return np.asarray(value).astype(dtype).astype(np.float64)

    # Synthetic IDs are local to each kind. Declare disjoint namespaces before
    # either native preparation or the independent oracle consumes them.

    return HostParticipant(
        3 * np.asarray(vertex_ids, dtype=np.int64),
        rounded(start),
        rounded(end),
        np.asarray(edges, dtype=np.int64).reshape((-1, 2)),
        3 * np.asarray(edge_ids, dtype=np.int64) + 1,
        np.empty((0, 3), dtype=np.int64)
        if faces is None
        else np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
        np.empty((0,), dtype=np.int64)
        if face_ids is None
        else 3 * np.asarray(face_ids, dtype=np.int64) + 2,
        rounded(radius),
        static,
        participant_id,
    )


def _curve(spec: CaseSpec, dtype: np.dtype, /) -> HostGeometry:
    count = spec.size
    x = np.linspace(0.0, 1.0, count + 1)
    start = np.stack((x, 0.01 * np.sin(20.0 * x)), axis=-1)
    edges = np.stack((np.arange(count), np.arange(1, count + 1)), axis=-1)
    participant = _participant(
        np.arange(count + 1),
        start,
        start,
        np.zeros((count + 1,)),
        dtype,
        edges=edges,
        edge_ids=np.arange(count),
        static=False,
        participant_id=0,
    )
    # The legacy 256-segment row used 0.005; keep the activation per spacing.
    return HostGeometry(2, Profile.MESH, (participant,), (), True, 1.28 / count, 0.0)


def _touch(dtype: np.dtype, /) -> HostGeometry:
    upper = _participant(
        np.asarray((0, 1)),
        np.asarray(((-0.5, 0.0), (0.5, 0.0))),
        np.asarray(((-0.5, 0.0), (0.5, 0.0))),
        np.zeros((2,)),
        dtype,
        edges=np.asarray(((0, 1),)),
        edge_ids=np.asarray((0,)),
        static=False,
        participant_id=0,
    )
    lower = _participant(
        np.asarray((2, 3)),
        np.asarray(((-1.0, 0.0), (1.0, 0.0))),
        np.asarray(((-1.0, 0.0), (1.0, 0.0))),
        np.zeros((2,)),
        dtype,
        edges=np.asarray(((0, 1),)),
        edge_ids=np.asarray((1,)),
        static=True,
        participant_id=1,
    )
    return HostGeometry(2, Profile.MESH, (upper, lower), (), False, 0.1, 0.0)


def _grid_faces(cells: int, /) -> tuple[np.ndarray, np.ndarray]:
    stride = cells + 1
    row, column = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
    corner = (row * stride + column).reshape(-1)
    faces = np.concatenate(
        (
            np.stack((corner, corner + 1, corner + stride + 1), axis=-1),
            np.stack((corner, corner + stride + 1, corner + stride), axis=-1),
        )
    )
    local = faces[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2))
    edges = np.unique(np.sort(local, axis=1), axis=0)
    return faces, edges


def _ground(dimension: int, feature_id: int, participant_id: int, /) -> HostPlane:
    normal = np.zeros((dimension,))
    normal[-1] = 1.0
    return HostPlane(normal, 0.0, feature_id, participant_id)


def _sheets(spec: CaseSpec, dtype: np.dtype, /) -> HostGeometry:
    rng = np.random.default_rng(spec.seed)
    cells = spec.size
    spacing = 1.0 / cells
    faces, edges = _grid_faces(cells)
    coordinates = np.linspace(0.0, 1.0, cells + 1)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    planar = np.stack((x.reshape(-1), y.reshape(-1)), axis=-1)
    count = planar.shape[0]
    participants: list[HostParticipant] = []
    for index, (height, static) in enumerate(((0.5, False), (1.1, True))):
        jitter = rng.uniform(-0.1, 0.1, (count, 2)) * spacing
        z = (height + rng.uniform(-0.05, 0.05, (count, 1))) * spacing
        start = np.concatenate((planar + jitter, z), axis=-1)
        end = start.copy()
        if spec.moving and not static:
            end[:, 2] -= 0.2 * spacing
        participants.append(
            _participant(
                np.arange(count) + index * count,
                start,
                end,
                np.zeros((count,)),
                dtype,
                edges=edges,
                edge_ids=np.arange(edges.shape[0]) + index * edges.shape[0],
                faces=faces,
                face_ids=np.arange(faces.shape[0]) + index * faces.shape[0],
                static=static,
                participant_id=index,
            )
        )
    return HostGeometry(
        3,
        Profile.MESH,
        tuple(participants),
        (_ground(3, 0, 2),),
        False,
        0.75 * spacing,
        0.0,
    )


def _centres(
    rng: np.random.Generator,
    count: int,
    side: float,
    distribution: Distribution,
    clusters: np.ndarray,
    /,
) -> np.ndarray:
    if distribution == "clustered":
        members = clusters[rng.integers(clusters.shape[0], size=count)]
        values = members + rng.normal(0.0, 0.25, (count, 3))
    else:
        values = rng.uniform(0.0, side, (count, 3))
    values[:, 2] = np.clip(values[:, 2], 0.0, side)
    return values


def _radii(
    rng: np.random.Generator, count: int, distribution: Distribution, /
) -> np.ndarray:
    if distribution == "tail":
        return np.minimum(0.05 + 0.05 * rng.pareto(1.5, count), 0.75)
    return rng.uniform(0.05, 0.15, count)


def _capsule_axes(
    rng: np.random.Generator,
    centres: np.ndarray,
    side: float,
    distribution: Distribution,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    count = centres.shape[0]
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    if distribution == "tail":
        half = np.minimum(0.15 * (1.0 + rng.pareto(1.5, count)), 0.5 * side)
    else:
        half = rng.uniform(0.15, 0.4, count)
    offset = half[:, None] * direction
    return centres - offset, centres + offset


def _spheres(spec: CaseSpec, dtype: np.dtype, /) -> HostGeometry:
    rng = np.random.default_rng(spec.seed)
    spheres = spec.size
    capsules = spec.capsules
    obstacles = max(1, capsules // 4)
    total = spheres + capsules + obstacles
    side = float(max(1, round(total ** (1.0 / 3.0))))
    clusters = rng.uniform(0.0, side, (max(1, total // 32), 3))
    sphere_centres = _centres(rng, spheres, side, spec.distribution, clusters)
    sphere_radii = _radii(rng, spheres, spec.distribution)
    first, second = _capsule_axes(
        rng,
        _centres(rng, capsules, side, spec.distribution, clusters),
        side,
        spec.distribution,
    )
    capsule_radii = _radii(rng, capsules, spec.distribution)
    start = np.concatenate(
        (
            sphere_centres,
            np.stack((first, second), axis=1).reshape((-1, 3)),
        )
    )
    end = start.copy()
    if spec.moving:
        end = end + rng.normal(0.0, 0.05, start.shape)
    dynamic_count = start.shape[0]
    dynamic = _participant(
        np.arange(dynamic_count),
        start,
        end,
        np.concatenate((sphere_radii, np.repeat(capsule_radii, 2))),
        dtype,
        edges=spheres
        + np.stack((2 * np.arange(capsules), 2 * np.arange(capsules) + 1), axis=-1),
        edge_ids=np.arange(capsules),
        static=False,
        participant_id=0,
    )
    obstacle_first, obstacle_second = _capsule_axes(
        rng,
        _centres(rng, obstacles, side, spec.distribution, clusters),
        side,
        spec.distribution,
    )
    obstacle_start = np.stack((obstacle_first, obstacle_second), axis=1).reshape((-1, 3))
    obstacle = _participant(
        np.arange(2 * obstacles) + dynamic_count,
        obstacle_start,
        obstacle_start,
        np.repeat(_radii(rng, obstacles, spec.distribution), 2),
        dtype,
        edges=np.stack((2 * np.arange(obstacles), 2 * np.arange(obstacles) + 1), axis=-1),
        edge_ids=np.arange(obstacles) + capsules,
        static=True,
        participant_id=1,
    )
    return HostGeometry(
        3,
        Profile.SPHERE_CAPSULE,
        (dynamic, obstacle),
        (_ground(3, 0, 2),),
        False,
        0.1,
        0.0,
    )


def _geometry(spec: CaseSpec, dtype: np.dtype, /) -> HostGeometry:
    if spec.family == "curve":
        return _curve(spec, dtype)
    if spec.family == "sheets":
        return _sheets(spec, dtype)
    if spec.family == "spheres":
        return _spheres(spec, dtype)
    return _touch(dtype)


# --------------------------------------------------------------------------
# Phydrax scene preparation (topology) and numeric refresh variants
# --------------------------------------------------------------------------


def _precision(dtype: np.dtype, /) -> _D.ContactPrecisionPolicy:
    return _D.ContactPrecisionPolicy(
        geometry_dtype=dtype,
        accumulation_dtype=np.float64,
        certification_dtype=np.float64,
        output_dtype=dtype,
    )


def _surface_plan(
    participant: HostParticipant, dimension: int, /
) -> _D.CollisionSurfacePlan:
    has_edges = participant.edges.shape[0] > 0
    has_faces = participant.faces.shape[0] > 0
    used = np.zeros((participant.vertex_ids.size,), dtype=np.bool_)
    used[participant.edges.reshape(-1)] = True
    used[participant.faces.reshape(-1)] = True
    return _D.CollisionSurfacePlan(
        participant.vertex_ids,
        ambient_dimension=dimension,
        edges=participant.edges.astype(np.int32),
        faces=participant.faces.astype(np.int32) if has_faces else None,
        edge_ids=participant.edge_ids if has_edges else None,
        face_ids=participant.face_ids if has_faces else None,
        participant_ids=participant.participant_id,
        body_ids=participant.participant_id,
        static_mask=participant.static,
        physical_radius=participant.radius,
        allow_isolated_vertices=not bool(np.all(used)),
    )


def _plane(plane: HostPlane, dtype: np.dtype, /) -> _D.PlaneContactGeometry:
    return _D.PlaneContactGeometry(
        jnp.asarray(plane.normal, dtype=dtype),
        jnp.asarray(plane.offset, dtype=dtype),
        feature_policy=_D.CollisionFeaturePolicy(
            np.asarray((plane.feature_id,), dtype=np.int64),
            np.asarray((int(_D.CollisionFeatureKind.ANALYTIC),), dtype=np.int32),
            participant_ids=plane.participant_id,
            body_ids=plane.participant_id,
            material_ids=0,
            patch_ids=0,
            static_mask=True,
            provenance_id=f"contact-search-benchmark-plane-{plane.feature_id}",
        ),
    )


def _prepare_scene(geometry: HostGeometry, dtype: np.dtype, /) -> SearchScene:
    precision = _precision(dtype)
    if geometry.collision_scene:
        total = sum(value.vertex_ids.size for value in geometry.participants)
        source = phx.linalg.ArraySpace((total, geometry.dimension), dtype=dtype)
        surfaces: list[_D.PreparedCollisionSurface] = []
        offset = 0
        for participant in geometry.participants:
            count = participant.vertex_ids.size
            surfaces.append(
                _D.PreparedCollisionSurface(
                    _surface_plan(participant, geometry.dimension),
                    participant.start.astype(dtype),
                    _D.selection_collision_operator(
                        source, np.arange(offset, offset + count, dtype=np.int32)
                    ),
                    precision=precision,
                )
            )
            offset += count
        return _D.PreparedCollisionScene(tuple(surfaces))
    participants: list[_D.LinearContactParticipant] = []
    for participant in geometry.participants:
        count = participant.vertex_ids.size
        source = phx.linalg.ArraySpace((count, geometry.dimension), dtype=dtype)
        participants.append(
            _D.LinearContactParticipant(
                _D.PreparedCollisionSurface(
                    _surface_plan(participant, geometry.dimension),
                    participant.start.astype(dtype),
                    _D.selection_collision_operator(
                        source, np.arange(count, dtype=np.int32)
                    ),
                    precision=precision,
                )
            )
        )
    return _D.ContactParticipantScene(
        tuple(participants),
        planes=tuple(_plane(plane, dtype) for plane in geometry.planes),
    )


@dataclass(frozen=True, slots=True)
class RefreshVariant:
    """Numeric geometry change with unchanged topology, catalogue and routing."""

    name: str
    description: str
    scene: SearchScene
    geometry: HostGeometry
    bind_seconds: float


def _with_radius_delta(geometry: HostGeometry, delta: float, /) -> HostGeometry:
    first = geometry.participants[0]
    changed = replace(first, radius=first.radius + delta)
    return replace(geometry, participants=(changed, *geometry.participants[1:]))


def _with_plane_offset(geometry: HostGeometry, offset: float, /) -> HostGeometry:
    first = replace(geometry.planes[0], offset=offset)
    return replace(geometry, planes=(first, *geometry.planes[1:]))


def _radius_leaf(scene: SearchScene, /) -> Any:
    if isinstance(scene, _D.PreparedCollisionScene):
        return scene.surfaces[0].plan.feature_policy.physical_radius
    return scene.participants[0].surface_plan.feature_policy.physical_radius


def _refresh_variants(
    scene: SearchScene, geometry: HostGeometry, dtype: np.dtype, /
) -> tuple[RefreshVariant, ...]:
    delta = float(np.asarray(0.25 * geometry.activation, dtype=dtype))
    radius = _radius_leaf(scene)
    refreshed, bind_seconds = measure_synchronized(
        lambda: eqx.tree_at(
            _radius_leaf, scene, radius + jnp.asarray(delta, radius.dtype)
        )
    )
    variants = [
        RefreshVariant(
            "contact-extent",
            f"first participant physical radius increased by {delta!r}",
            refreshed,
            _with_radius_delta(geometry, delta),
            bind_seconds,
        )
    ]
    if geometry.planes and isinstance(scene, _D.ContactParticipantScene):
        offset = float(np.asarray(geometry.planes[0].offset + delta, dtype=dtype))
        raised, raise_seconds = measure_synchronized(
            lambda: eqx.tree_at(
                lambda value: value.planes[0].offset,
                scene,
                jnp.asarray(offset, dtype=dtype),
            )
        )
        variants.append(
            RefreshVariant(
                "plane-offset",
                f"ground plane offset raised to {offset!r}",
                raised,
                _with_plane_offset(geometry, offset),
                raise_seconds,
            )
        )
    return tuple(variants)


# --------------------------------------------------------------------------
# Independent reference
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OracleFeatures:
    ids: np.ndarray
    vertices: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    extent: np.ndarray
    static: np.ndarray

    def subset(self, mask: np.ndarray, /) -> OracleFeatures:
        return OracleFeatures(
            self.ids[mask],
            self.vertices[mask],
            self.lower[mask],
            self.upper[mask],
            self.extent[mask],
            self.static[mask],
        )


@dataclass(frozen=True, slots=True)
class OracleScene:
    start: np.ndarray
    vertices: OracleFeatures
    edges: OracleFeatures
    faces: OracleFeatures
    plane_ids: np.ndarray
    plane_normals: np.ndarray
    plane_offsets: np.ndarray
    plane_extent: np.ndarray
    isolated: np.ndarray
    scale: float


@dataclass(frozen=True, slots=True)
class OracleFamily:
    """Route family restated from the primitive-profile documentation."""

    kind: Kind
    left: OracleFeatures
    right: OracleFeatures | None
    slots: int
    plane_count: int

    @property
    def product(self) -> int:
        right = self.plane_count if self.right is None else self.right.ids.size
        return int(self.left.ids.size) * int(right)

    @property
    def row_bound(self) -> int:
        if self.kind in _SYMMETRIC:
            count = int(self.left.ids.size)
            return count * (count - 1) // 2 * self.slots
        return self.product * self.slots


def _oracle_features(
    ids: np.ndarray,
    table: np.ndarray,
    vertex_lower: np.ndarray,
    vertex_upper: np.ndarray,
    extent: np.ndarray,
    static: np.ndarray,
    /,
) -> OracleFeatures:
    padded = np.full((table.shape[0], 3), -1, dtype=np.int64)
    padded[:, : table.shape[1]] = table
    dimension = vertex_lower.shape[1]
    if table.shape[0] == 0:
        empty = np.zeros((0, dimension))
        return OracleFeatures(ids, padded, empty, empty, extent, static)
    return OracleFeatures(
        ids,
        padded,
        np.min(vertex_lower[table], axis=1),
        np.max(vertex_upper[table], axis=1),
        extent,
        static,
    )


def _oracle_scene(geometry: HostGeometry, /) -> OracleScene:
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    vertex_static: list[np.ndarray] = []
    vertex_ids: list[np.ndarray] = []
    edge_tables: list[np.ndarray] = []
    edge_ids: list[np.ndarray] = []
    edge_static: list[np.ndarray] = []
    face_tables: list[np.ndarray] = []
    face_ids: list[np.ndarray] = []
    face_static: list[np.ndarray] = []
    offset = 0
    for participant in geometry.participants:
        count = participant.vertex_ids.size
        starts.append(participant.start)
        ends.append(participant.end)
        radii.append(participant.radius)
        vertex_static.append(np.full((count,), participant.static))
        vertex_ids.append(participant.vertex_ids)
        edge_tables.append(participant.edges + offset)
        edge_ids.append(participant.edge_ids)
        edge_static.append(np.full((participant.edges.shape[0],), participant.static))
        face_tables.append(participant.faces + offset)
        face_ids.append(participant.face_ids)
        face_static.append(np.full((participant.faces.shape[0],), participant.static))
        offset += count
    start = np.concatenate(starts)
    end = np.concatenate(ends)
    radius = np.concatenate(radii)
    lower = np.minimum(start, end)
    upper = np.maximum(start, end)
    edges = np.concatenate(edge_tables).reshape((-1, 2))
    faces = np.concatenate(face_tables).reshape((-1, 3))
    used = np.zeros((start.shape[0],), dtype=np.bool_)
    used[edges.reshape(-1)] = True
    used[faces.reshape(-1)] = True
    # Surface feature extents are the maximum of their endpoint radii.
    edge_extent = np.max(radius[edges], axis=1) if edges.shape[0] else np.zeros((0,))
    face_extent = np.max(radius[faces], axis=1) if faces.shape[0] else np.zeros((0,))
    dimension = geometry.dimension
    planes = geometry.planes
    scale = max(
        1.0,
        float(np.max(np.abs(start), initial=0.0)),
        float(np.max(np.abs(end), initial=0.0)),
        float(np.max(radius, initial=0.0)),
        max((abs(plane.offset) for plane in planes), default=0.0),
    )
    return OracleScene(
        start,
        _oracle_features(
            np.concatenate(vertex_ids),
            np.arange(start.shape[0], dtype=np.int64)[:, None],
            lower,
            upper,
            radius,
            np.concatenate(vertex_static),
        ),
        _oracle_features(
            np.concatenate(edge_ids),
            edges,
            lower,
            upper,
            edge_extent,
            np.concatenate(edge_static),
        ),
        _oracle_features(
            np.concatenate(face_ids),
            faces,
            lower,
            upper,
            face_extent,
            np.concatenate(face_static),
        ),
        np.asarray([plane.feature_id for plane in planes], dtype=np.int64),
        np.asarray([plane.normal for plane in planes], dtype=np.float64).reshape(
            (-1, dimension)
        ),
        np.asarray([plane.offset for plane in planes], dtype=np.float64),
        np.zeros((len(planes),)),
        ~used,
        scale,
    )


def _oracle_families(
    scene: OracleScene, profile: Profile, dimension: int, /
) -> tuple[OracleFamily, ...]:
    planes = scene.plane_ids.size
    families: tuple[OracleFamily, ...]
    if profile == Profile.SPHERE_CAPSULE:
        spheres = scene.vertices.subset(scene.isolated)
        families = (
            OracleFamily(Kind.VERTEX_VERTEX, spheres, spheres, 1, planes),
            OracleFamily(Kind.EDGE_VERTEX, spheres, scene.edges, 1, planes),
            OracleFamily(Kind.EDGE_EDGE, scene.edges, scene.edges, 2, planes),
            OracleFamily(Kind.VERTEX_PLANE, spheres, None, 1, planes),
            OracleFamily(Kind.EDGE_PLANE, scene.edges, None, 2, planes),
        )
    elif dimension == 2 or scene.faces.ids.size == 0:
        families = (
            OracleFamily(Kind.EDGE_VERTEX, scene.vertices, scene.edges, 1, planes),
            OracleFamily(Kind.VERTEX_PLANE, scene.vertices, None, 1, planes),
        )
    else:
        families = (
            OracleFamily(Kind.FACE_VERTEX, scene.vertices, scene.faces, 1, planes),
            OracleFamily(Kind.EDGE_EDGE, scene.edges, scene.edges, 1, planes),
            OracleFamily(Kind.VERTEX_PLANE, scene.vertices, None, 1, planes),
        )
    return tuple(family for family in families if family.product > 0)


def _row(kind: int, left: int, right: int, slot: int, /) -> Row:
    if Kind(kind) in _SYMMETRIC and right < left:
        left, right = right, left
    return (int(kind), int(left), int(right), int(slot))


def _family_rows(
    family: OracleFamily, scene: OracleScene, activation: float, envelope: float, /
) -> set[Row]:
    rows: set[Row] = set()
    left = family.left
    kind = int(family.kind)
    if family.right is None:
        centre = 0.5 * (left.lower + left.upper)
        half = 0.5 * (left.upper - left.lower)
        clearance = (
            centre @ scene.plane_normals.T
            - half @ np.abs(scene.plane_normals).T
            - scene.plane_offsets[None, :]
        )
        radius = (
            activation + envelope + left.extent[:, None] + scene.plane_extent[None, :]
        )
        near = (clearance <= radius) & ~left.static[:, None]
        first, second = np.nonzero(near)
        for feature, plane in zip(left.ids[first], scene.plane_ids[second], strict=True):
            for slot in range(family.slots):
                rows.add(_row(kind, int(feature), int(plane), slot))
        return rows
    right = family.right
    symmetric = family.kind in _SYMMETRIC
    chunk = max(1, min(512, 4_000_000 // max(1, right.ids.size)))
    for begin in range(0, left.ids.size, chunk):
        stop = min(begin + chunk, left.ids.size)
        gap = np.maximum(
            0.0,
            np.maximum(
                left.lower[begin:stop, None, :] - right.upper[None, :, :],
                right.lower[None, :, :] - left.upper[begin:stop, None, :],
            ),
        )
        distance = np.sum(gap * gap, axis=-1)
        radius = (
            activation
            + 2.0 * envelope
            + left.extent[begin:stop, None]
            + right.extent[None, :]
        )
        near = distance <= radius * radius
        if symmetric:
            near &= np.arange(begin, stop)[:, None] < np.arange(right.ids.size)[None, :]
        first, second = np.nonzero(near)
        first = first + begin
        legal = ~(left.static[first] & right.static[second])
        left_vertices = left.vertices[first]
        right_vertices = right.vertices[second]
        shared = np.any(
            (left_vertices[:, :, None] >= 0)
            & (left_vertices[:, :, None] == right_vertices[:, None, :]),
            axis=(1, 2),
        )
        legal &= ~shared
        for feature, other in zip(
            left.ids[first[legal]], right.ids[second[legal]], strict=True
        ):
            for slot in range(family.slots):
                rows.add(_row(kind, int(feature), int(other), slot))
    return rows


def _oracle_rows(
    families: Sequence[OracleFamily],
    scene: OracleScene,
    activation: float,
    envelope: float,
    /,
) -> Rows:
    rows: Rows = {int(kind): set() for kind in _KIND_ORDER}
    for family in families:
        rows[int(family.kind)] |= _family_rows(family, scene, activation, envelope)
    return rows


# Analytic narrow-phase distances (independent of the stencil evaluator).


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, /) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(np.broadcast(numerator, denominator).shape),
        where=denominator != 0.0,
    )


def _dot(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    return np.sum(first * second, axis=-1)


def _point_segment_distance(
    point: np.ndarray, first: np.ndarray, second: np.ndarray, /
) -> np.ndarray:
    axis = second - first
    t = np.clip(_safe_divide(_dot(point - first, axis), _dot(axis, axis)), 0.0, 1.0)
    return np.linalg.norm(point - (first + t[..., None] * axis), axis=-1)


def _segment_segment_distance(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray, /
) -> np.ndarray:
    """Clamped closest points of two segments (Ericson, RTCD section 5.1.9)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = _dot(d1, d1)
    e = _dot(d2, d2)
    f = _dot(d2, r)
    c = _dot(d1, r)
    b = _dot(d1, d2)
    tiny = np.finfo(np.float64).tiny
    a_valid = a > tiny
    e_valid = e > tiny
    a_safe = np.where(a_valid, a, 1.0)
    e_safe = np.where(e_valid, e, 1.0)
    denominator = a * e - b * b
    parallel = denominator <= 1.0e-14 * a * e
    s = np.where(
        parallel,
        0.0,
        np.clip((b * f - c * e) / np.where(parallel, 1.0, denominator), 0.0, 1.0),
    )
    t = (b * s + f) / e_safe
    s = np.where(
        t < 0.0,
        np.clip(-c / a_safe, 0.0, 1.0),
        np.where(t > 1.0, np.clip((b - c) / a_safe, 0.0, 1.0), s),
    )
    t = np.clip(t, 0.0, 1.0)
    s = np.where(a_valid, s, 0.0)
    t = np.where(a_valid, t, np.clip(f / e_safe, 0.0, 1.0))
    s = np.where(a_valid & ~e_valid, np.clip(-c / a_safe, 0.0, 1.0), s)
    t = np.where(e_valid, t, 0.0)
    return np.linalg.norm((p1 + s[..., None] * d1) - (p2 + t[..., None] * d2), axis=-1)


def _point_triangle_distance(
    point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, /
) -> np.ndarray:
    """Closest point on a triangle by Voronoi regions (Ericson 5.1.5)."""
    ab = b - a
    ac = c - a
    d1 = _dot(ab, point - a)
    d2 = _dot(ac, point - a)
    d3 = _dot(ab, point - b)
    d4 = _dot(ac, point - b)
    d5 = _dot(ab, point - c)
    d6 = _dot(ac, point - c)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    total = va + vb + vc
    edge_ab = a + _safe_divide(d1, d1 - d3)[..., None] * ab
    edge_ac = a + _safe_divide(d2, d2 - d6)[..., None] * ac
    edge_bc = b + _safe_divide(d4 - d3, (d4 - d3) + (d5 - d6))[..., None] * (c - b)
    interior = (
        a
        + _safe_divide(vb, total)[..., None] * ab
        + _safe_divide(vc, total)[..., None] * ac
    )
    conditions = [
        ((d1 <= 0.0) & (d2 <= 0.0))[..., None],
        ((d3 >= 0.0) & (d4 <= d3))[..., None],
        ((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0))[..., None],
        ((d6 >= 0.0) & (d5 <= d6))[..., None],
        ((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0))[..., None],
        ((va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0))[..., None],
    ]
    closest = np.select(
        conditions, [a, b, edge_ab, c, edge_ac, edge_bc], default=interior
    )
    return np.linalg.norm(point - closest, axis=-1)


def _index(ids: np.ndarray, /) -> dict[int, int]:
    return {int(value): index for index, value in enumerate(ids)}


def _lookup(index: Mapping[int, int], values: np.ndarray, /) -> np.ndarray:
    return np.fromiter(
        (index[int(value)] for value in values), dtype=np.int64, count=values.size
    )


def _gap_reference(
    kinematics: _D.ContactKinematicsEpoch, oracle: OracleScene, tolerance: float, /
) -> dict[str, Any]:
    """Compare valid kinematics gaps with analytic distances minus extents."""
    vertex_index = _index(oracle.vertices.ids)
    edge_index = _index(oracle.edges.ids)
    face_index = _index(oracle.faces.ids)
    plane_index = _index(oracle.plane_ids)
    points = oracle.start
    vertex_extent = oracle.vertices.extent
    edge_extent = oracle.edges.extent
    edge_table = oracle.edges.vertices[:, :2]
    errors: dict[str, float] = {}
    compared: dict[str, int] = {}
    for batch in kinematics.batches:
        valid = np.asarray(batch.valid)
        if not np.any(valid):
            continue
        kind = Kind(int(batch.kind))
        left = np.asarray(batch.left_feature_ids)[valid]
        right = np.asarray(batch.right_feature_ids)[valid]
        gap = np.asarray(batch.gap, dtype=np.float64)[valid]
        if kind == Kind.VERTEX_VERTEX:
            i = _lookup(vertex_index, left)
            j = _lookup(vertex_index, right)
            reference = (
                np.linalg.norm(points[i] - points[j], axis=-1)
                - vertex_extent[i]
                - vertex_extent[j]
            )
            error = np.abs(gap - reference)
        elif kind == Kind.EDGE_VERTEX:
            i = _lookup(vertex_index, left)
            e = _lookup(edge_index, right)
            reference = (
                _point_segment_distance(
                    points[i], points[edge_table[e, 0]], points[edge_table[e, 1]]
                )
                - vertex_extent[i]
                - edge_extent[e]
            )
            error = np.abs(gap - reference)
        elif kind == Kind.FACE_VERTEX:
            i = _lookup(vertex_index, left)
            f = _lookup(face_index, right)
            corners = oracle.faces.vertices[f]
            reference = (
                _point_triangle_distance(
                    points[i],
                    points[corners[:, 0]],
                    points[corners[:, 1]],
                    points[corners[:, 2]],
                )
                - vertex_extent[i]
                - oracle.faces.extent[f]
            )
            error = np.abs(gap - reference)
        elif kind == Kind.VERTEX_PLANE:
            i = _lookup(vertex_index, left)
            k = _lookup(plane_index, right)
            reference = (
                _dot(points[i], oracle.plane_normals[k])
                - oracle.plane_offsets[k]
                - vertex_extent[i]
                - oracle.plane_extent[k]
            )
            error = np.abs(gap - reference)
        elif kind == Kind.EDGE_PLANE:
            e = _lookup(edge_index, left)
            k = _lookup(plane_index, right)
            # Each witness slot is one capsule endpoint support.
            supports = (
                np.sum(points[edge_table[e]] * oracle.plane_normals[k][:, None, :], -1)
                - oracle.plane_offsets[k][:, None]
                - edge_extent[e][:, None]
                - oracle.plane_extent[k][:, None]
            )
            error = np.min(np.abs(gap[:, None] - supports), axis=1)
        else:
            e = _lookup(edge_index, left)
            g = _lookup(edge_index, right)
            pairs = np.stack((np.minimum(e, g), np.maximum(e, g)), axis=1)
            unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
            closest = np.full((unique.shape[0],), np.inf)
            np.minimum.at(closest, inverse.reshape(-1), gap)
            reference = (
                _segment_segment_distance(
                    points[edge_table[unique[:, 0], 0]],
                    points[edge_table[unique[:, 0], 1]],
                    points[edge_table[unique[:, 1], 0]],
                    points[edge_table[unique[:, 1], 1]],
                )
                - edge_extent[unique[:, 0]]
                - edge_extent[unique[:, 1]]
            )
            error = np.abs(closest - reference)
        name = kind.name.lower()
        errors[name] = max(errors.get(name, 0.0), float(np.max(error)))
        compared[name] = compared.get(name, 0) + int(error.size)
    maximum = max(errors.values(), default=0.0)
    if not compared:
        status = "no-valid-rows"
    else:
        status = "passed" if maximum <= tolerance else "failed"
    return {
        "reference": (
            "analytic sphere, capsule, point-segment, segment-segment (minimum over "
            "witness slots per pair), point-triangle and plane distances minus "
            "contact extents at the evaluated positions"
        ),
        "status": status,
        "rows_compared": compared,
        "max_abs_error": errors,
        "overall_max_abs_error": maximum,
        "tolerance": tolerance,
    }


# --------------------------------------------------------------------------
# Epoch inspection
# --------------------------------------------------------------------------


def _epoch_rows(epoch: _D.ContactCandidateEpoch, /) -> tuple[Rows, int]:
    rows: Rows = {int(kind): set() for kind in _KIND_ORDER}
    duplicates = 0
    for batch in epoch.batches:
        valid = np.asarray(batch.valid)
        if not np.any(valid):
            continue
        kind = int(batch.kind)
        values = [
            _row(kind, int(left), int(right), int(slot))
            for left, right, slot in zip(
                np.asarray(batch.left_feature_ids)[valid],
                np.asarray(batch.right_feature_ids)[valid],
                np.asarray(batch.witness_slots)[valid],
                strict=True,
            )
        ]
        duplicates += len(values) - len(set(values))
        rows[kind] |= set(values)
    return rows, duplicates


def _valid_route_keys(epoch: _D.ContactCandidateEpoch, /) -> list[int]:
    keys: list[int] = []
    for batch in epoch.batches:
        valid = np.asarray(batch.valid)
        keys.extend(int(key) for key in np.asarray(batch.route_keys)[valid])
    return sorted(keys)


def _kind_counts(
    epoch: _D.ContactCandidateEpoch, reference: Rows, /
) -> dict[str, dict[str, int]]:
    return {
        _kind_name(int(batch.kind)): {
            "capacity": int(batch.capacity),
            "discovered": int(np.asarray(batch.actual_count)),
            "overflow": int(np.asarray(batch.overflow_count)),
            "valid": int(np.sum(np.asarray(batch.valid))),
            "reference": len(reference[int(batch.kind)]),
        }
        for batch in epoch.batches
    }


def _parity(epoch: _D.ContactCandidateEpoch, reference: Rows, /) -> dict[str, Any]:
    rows, duplicates = _epoch_rows(epoch)
    counts = _kind_counts(epoch, reference)
    discovered = all(
        value["discovered"] == value["reference"] for value in counts.values()
    )
    if not bool(np.asarray(epoch.successful)):
        return {
            "status": "not-applicable",
            "reason": "An unsuccessful epoch exposes no valid rows.",
            "missing": None,
            "extra": None,
            "duplicates": duplicates,
            "discovered_counts_match_reference": discovered,
        }
    missing = sum(len(reference[kind] - rows[kind]) for kind in reference)
    extra = sum(len(rows[kind] - reference[kind]) for kind in rows)
    matched = missing == 0 and extra == 0 and duplicates == 0
    return {
        "status": "matched" if matched else "mismatched",
        "reason": None,
        "missing": missing,
        "extra": extra,
        "duplicates": duplicates,
        "discovered_counts_match_reference": discovered,
    }


def _search_summary(epoch: _D.ContactCandidateEpoch, /) -> dict[str, Any]:
    return {
        "status": _status_name(epoch.status),
        "successful": bool(np.asarray(epoch.successful)),
        "complete": bool(np.asarray(epoch.complete)),
        "candidate_count": int(np.asarray(epoch.candidate_count)),
        "overflow_count": int(np.asarray(epoch.overflow_count)),
        "logical_epoch_bytes": logical_array_bytes(epoch),
    }


def _kinematics_summary(
    epoch: _D.ContactCandidateEpoch,
    kinematics: _D.ContactKinematicsEpoch,
    oracle: OracleScene,
    tolerance: float,
    /,
) -> dict[str, Any]:
    evidence = kinematics.evidence
    search_successful = bool(np.asarray(epoch.successful))
    successful = bool(np.asarray(evidence.successful))
    failed = int(np.asarray(evidence.failed_contacts))
    failed_rows_valid = any(
        bool(np.any(np.asarray(batch.failed) & np.asarray(batch.valid)))
        for batch in kinematics.batches
    )
    reported_complete = bool(np.asarray(evidence.search_complete))
    propagation = {
        "search_complete_matches_epoch": reported_complete == search_successful,
        "unsuccessful_search_refused": search_successful or not successful,
        "failed_witness_refused": failed == 0 or not successful,
        "failed_rows_never_valid": not failed_rows_valid,
    }
    propagation["consistent"] = all(propagation.values())
    return {
        "successful": successful,
        "active_contacts": int(np.asarray(evidence.active_contacts)),
        "failed_contacts": failed,
        "minimum_gap": _finite(evidence.minimum_gap),
        "minimum_feature_margin": _finite(evidence.minimum_feature_margin),
        "finite": bool(np.asarray(evidence.finite)),
        "failure_propagation": propagation,
        "gap_reference": _gap_reference(kinematics, oracle, tolerance),
        "logical_kinematics_bytes": logical_array_bytes(kinematics),
    }


def _outcome(search: Mapping[str, Any], kinematics: Mapping[str, Any], /) -> str:
    if not search["successful"]:
        return f"search-refused:{search['status']}"
    if not kinematics["successful"]:
        return "kinematics-failed"
    return "successful"


def _expectation(
    spec: CaseSpec,
    backend: str,
    search: Mapping[str, Any],
    kinematics: Mapping[str, Any],
    /,
) -> dict[str, Any]:
    expected = spec.expected_status(backend)
    status_met = search["status"] == expected
    if spec.expect_narrow_failure:
        narrow_met = kinematics["failed_contacts"] > 0 and not kinematics["successful"]
    elif search["successful"]:
        narrow_met = kinematics["successful"]
    else:
        narrow_met = not kinematics["successful"]
    return {
        "expected_search_status": expected,
        "observed_search_status": search["status"],
        "expect_narrow_failure": spec.expect_narrow_failure,
        "met": status_met and narrow_met,
    }


# --------------------------------------------------------------------------
# Measured phases
# --------------------------------------------------------------------------


def _search(
    plan: DevicePlan, scene: SearchScene, positions: jax.Array, end: jax.Array
) -> Any:
    return plan.evaluate(scene, positions, end_positions=end)


def _kinematics(
    scene: SearchScene,
    epoch: _D.ContactCandidateEpoch,
    positions: jax.Array,
    velocities: jax.Array,
    rest: jax.Array,
    step_size: jax.Array,
) -> _D.ContactKinematicsEpoch:
    return _D.evaluate_contact_kinematics(
        scene, epoch, positions, velocities, step_size, rest_positions=rest
    )


def _pipeline(
    plan: DevicePlan,
    scene: SearchScene,
    positions: jax.Array,
    end: jax.Array,
    velocities: jax.Array,
    rest: jax.Array,
    step_size: jax.Array,
) -> tuple[Any, _D.ContactKinematicsEpoch]:
    result = plan.evaluate(scene, positions, end_positions=end)
    return result, _D.evaluate_contact_kinematics(
        scene, result.epoch, positions, velocities, step_size, rest_positions=rest
    )


_JIT_KINEMATICS = eqx.filter_jit(_kinematics)


def _compiler_dict(value: CompilerEvidence, /) -> dict[str, Any]:
    return {
        "source": value.source,
        "flops": value.flops,
        "bytes_accessed": value.bytes_accessed,
        "argument_bytes": value.argument_bytes,
        "output_bytes": value.output_bytes,
        "temporary_bytes": value.temporary_bytes,
        "generated_code_bytes": value.generated_code_bytes,
        "estimated_device_memory_bytes": value.estimated_device_memory_bytes,
        "unavailable_reason": value.unavailable_reason,
    }


def _hlo_structure(compiled: Any, /) -> dict[str, Any]:
    text = compiled.as_text()
    reason = (
        "Kernel launch counts require a device profiler; these are instruction "
        "occurrences in optimized HLO text, including nested computations."
    )
    if not text:
        return {
            "source": "optimized-hlo-text",
            "instructions": None,
            "kernel_launches": None,
            "reason": "Optimized HLO text was unavailable. " + reason,
        }
    found = _HLO_PATTERN.findall(text)
    return {
        "source": "optimized-hlo-text",
        "instructions": {name: found.count(name) for name in _HLO_OPERATIONS},
        "kernel_launches": None,
        "reason": reason,
    }


@dataclass(frozen=True, slots=True)
class Phase:
    executable: Any
    value: Any
    record: dict[str, Any]


def _compiled_phase(
    function: Callable[..., Any], arguments: tuple[Any, ...], settings: Settings, /
) -> Phase:
    """Lower, compile, call once and warm one ahead-of-time executable."""
    jitted = eqx.filter_jit(function)
    executable, compilation = measure_lower_and_compile(
        # ty: ignore[unresolved-attribute]
        lambda: jitted.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    first, first_seconds = measure_synchronized(lambda: executable(*arguments))
    _, warm = measure_repeated(
        lambda: executable(*arguments),
        warmup=settings.warmup,
        repeats=settings.repeats,
    )
    compiled = executable.compiled
    cost = compiled.cost_analysis()
    memory = compiled.memory_analysis()
    evidence = compiler_evidence(
        cost,
        memory,
        source="jax-lowered-compiled-executable",
        unavailable_reason=(
            "Compiler cost and memory analysis were both unavailable on this backend."
            if not cost and memory is None
            else None
        ),
    )
    return Phase(
        executable,
        first,
        {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "first_call_seconds": first_seconds,
            "warm": warm.to_seconds_dict(),
            "warmup_calls": settings.warmup,
            "compiler": _compiler_dict(evidence),
            "hlo": _hlo_structure(compiled),
            "logical_argument_bytes": logical_array_bytes(arguments),
            "logical_result_bytes": logical_array_bytes(first),
        },
    )


@dataclass(frozen=True, slots=True)
class Capacities:
    vertex_vertex: int
    edge_vertex: int
    edge_edge: int
    face_vertex: int
    vertex_plane: int
    edge_plane: int
    policy: str

    def values(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.vertex_vertex,
            self.edge_vertex,
            self.edge_edge,
            self.face_vertex,
            self.vertex_plane,
            self.edge_plane,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "declared": {
                _kind_name(int(kind)): value
                for kind, value in zip(_KIND_ORDER, self.values(), strict=True)
            },
        }


def _capacities(
    spec: CaseSpec, references: Sequence[Rows], settings: Settings, /
) -> Capacities:
    if spec.capacity == "legacy":
        return Capacities(
            0,
            _LEGACY_EDGE_VERTEX_CAPACITY,
            0,
            0,
            0,
            0,
            f"legacy fixed: edge-vertex {_LEGACY_EDGE_VERTEX_CAPACITY}, other kinds zero",
        )
    base = {int(kind): len(references[0][int(kind)]) for kind in _KIND_ORDER}
    if spec.capacity == "shortfall":
        target = max(base, key=lambda kind: base[kind])
        values = dict(base)
        values[target] = max(base[target] - 1, 0)
        policy = (
            "shortfall: independent-reference row counts with "
            f"{_kind_name(target)} reduced by one"
        )
    else:
        values = {
            int(kind): math.ceil(
                settings.capacity_factor
                * max(len(reference[int(kind)]) for reference in references)
            )
            for kind in _KIND_ORDER
        }
        policy = (
            f"slack: ceil({settings.capacity_factor!r} x maximum independent-reference "
            "rows over base and refreshed geometry), declared before any search"
        )
    return Capacities(
        values[int(Kind.VERTEX_VERTEX)],
        values[int(Kind.EDGE_VERTEX)],
        values[int(Kind.EDGE_EDGE)],
        values[int(Kind.FACE_VERTEX)],
        values[int(Kind.VERTEX_PLANE)],
        values[int(Kind.EDGE_PLANE)],
        policy,
    )


@dataclass(frozen=True, slots=True)
class Prepared:
    spec: CaseSpec
    geometry: HostGeometry
    oracle: OracleScene
    families: tuple[OracleFamily, ...]
    reference: Rows
    scene: SearchScene
    variants: tuple[RefreshVariant, ...]
    variant_oracles: tuple[OracleScene, ...]
    variant_references: tuple[Rows, ...]
    start: jax.Array
    end: jax.Array
    velocities: jax.Array
    step_size: jax.Array
    capacities: Capacities
    tolerance: float


def _host_plan(name: str, prepared: Prepared, /) -> HostPlan:
    capacities = prepared.capacities
    geometry = prepared.geometry
    plan_type = (
        _D.DenseContactSearchPlan
        if name == "dense"
        else _D.SweepAndPruneContactSearchPlan
    )
    return plan_type(
        edge_vertex_capacity=capacities.edge_vertex,
        edge_edge_capacity=capacities.edge_edge,
        face_vertex_capacity=capacities.face_vertex,
        activation_distance=geometry.activation,
        envelope_radius=geometry.envelope,
        vertex_vertex_capacity=capacities.vertex_vertex,
        vertex_plane_capacity=capacities.vertex_plane,
        edge_plane_capacity=capacities.edge_plane,
        primitive_profile=geometry.profile,
    )


def _device_plan(name: str, prepared: Prepared, visit_budget: int, /) -> DevicePlan:
    capacities = prepared.capacities
    geometry = prepared.geometry
    if name == "compiled":
        return _D.CompiledContactSearchPlan(
            prepared.scene,
            edge_vertex_capacity=capacities.edge_vertex,
            edge_edge_capacity=capacities.edge_edge,
            face_vertex_capacity=capacities.face_vertex,
            activation_distance=geometry.activation,
            envelope_radius=geometry.envelope,
            vertex_vertex_capacity=capacities.vertex_vertex,
            vertex_plane_capacity=capacities.vertex_plane,
            edge_plane_capacity=capacities.edge_plane,
            primitive_profile=geometry.profile,
        )
    return _D.LBVHContactSearchPlan(
        prepared.scene,
        edge_vertex_capacity=capacities.edge_vertex,
        edge_edge_capacity=capacities.edge_edge,
        face_vertex_capacity=capacities.face_vertex,
        activation_distance=geometry.activation,
        maximum_traversal_visits=visit_budget,
        envelope_radius=geometry.envelope,
        vertex_vertex_capacity=capacities.vertex_vertex,
        vertex_plane_capacity=capacities.vertex_plane,
        edge_plane_capacity=capacities.edge_plane,
        primitive_profile=geometry.profile,
    )


def _refused(name: str, reason: str, /) -> dict[str, Any]:
    return {
        "backend": name,
        "outcome": "refused-resource",
        "reason": reason,
        "timed": False,
        "expectation": None,
    }


def _host_backend(name: str, prepared: Prepared, settings: Settings, /) -> dict[str, Any]:
    product = sum(
        family.product for family in prepared.families if family.right is not None
    )
    if name == "dense" and product > settings.dense_pair_limit:
        return _refused(
            name,
            f"Dense host search materializes {product} finite primitive pairs, above "
            f"--dense-pair-limit {settings.dense_pair_limit}.",
        )
    plan, preparation_seconds = measure_host(lambda: _host_plan(name, prepared))
    scene = prepared.scene

    def search(target: SearchScene, /) -> _D.HostContactSearchResult:
        return plan.search(target, prepared.start, end_positions=prepared.end)

    result, samples = measure_repeated(
        lambda: search(scene),
        warmup=settings.warmup,
        repeats=settings.host_repeats,
    )
    epoch = result.epoch
    summary = _search_summary(epoch)
    summary["host_synchronized_search"] = samples.to_seconds_dict()
    summary["host_evidence"] = {
        "elapsed_seconds_last_call": float(np.asarray(result.evidence.elapsed_seconds)),
        "estimated_route_bytes": int(np.asarray(result.evidence.estimated_memory_bytes)),
    }
    kinematics = _JIT_KINEMATICS(
        scene,
        epoch,
        prepared.start,
        prepared.velocities,
        prepared.start,
        prepared.step_size,
    )
    kinematics_summary = _kinematics_summary(
        epoch, kinematics, prepared.oracle, prepared.tolerance
    )
    kinematics_summary["timing"] = (
        "not measured for host epochs; kinematics phases are timed on device epochs"
    )
    refresh = [
        _host_refresh(search, epoch, variant, oracle, reference, prepared, settings)
        for variant, oracle, reference in zip(
            prepared.variants,
            prepared.variant_oracles,
            prepared.variant_references,
            strict=True,
        )
    ]
    return {
        "backend": name,
        "kind": "host",
        "outcome": _outcome(summary, kinematics_summary),
        "timed": True,
        "expectation": _expectation(prepared.spec, name, summary, kinematics_summary),
        "plan_preparation_seconds": preparation_seconds,
        "search": summary,
        "counts": _kind_counts(epoch, prepared.reference),
        "parity": _parity(epoch, prepared.reference),
        "kinematics": kinematics_summary,
        "refresh": refresh,
        "_epoch": epoch,
    }


def _host_refresh(
    search: Callable[[SearchScene], _D.HostContactSearchResult],
    stale: _D.ContactCandidateEpoch,
    variant: RefreshVariant,
    oracle: OracleScene,
    reference: Rows,
    prepared: Prepared,
    settings: Settings,
    /,
) -> dict[str, Any]:
    result, samples = measure_repeated(
        lambda: search(variant.scene),
        warmup=settings.warmup,
        repeats=settings.host_repeats,
    )
    epoch = result.epoch
    stale_kinematics = _JIT_KINEMATICS(
        variant.scene,
        stale,
        prepared.start,
        prepared.velocities,
        prepared.start,
        prepared.step_size,
    )
    kinematics = _JIT_KINEMATICS(
        variant.scene,
        epoch,
        prepared.start,
        prepared.velocities,
        prepared.start,
        prepared.step_size,
    )
    summary = _search_summary(epoch)
    summary["host_synchronized_search"] = samples.to_seconds_dict()
    return {
        "variant": variant.name,
        "stale_epoch_binds": bool(np.asarray(stale.binds(variant.scene))),
        "stale_kinematics_successful": bool(
            np.asarray(stale_kinematics.evidence.successful)
        ),
        "search": summary,
        "parity": _parity(epoch, reference),
        "kinematics": _kinematics_summary(epoch, kinematics, oracle, prepared.tolerance),
    }


def _compiled_evidence(
    result: _D.CompiledContactSearchResult, plan: _D.CompiledContactSearchPlan, /
) -> dict[str, Any]:
    evidence = result.evidence
    return {
        "candidate_count": int(np.asarray(evidence.candidate_count)),
        "overflow_count": int(np.asarray(evidence.overflow_count)),
        "duplicate_route_count": int(np.asarray(evidence.duplicate_route_count)),
        "finite": bool(np.asarray(evidence.finite)),
        "complete": bool(np.asarray(evidence.complete)),
        "prepared_legal_rows": sum(
            int(candidates.left_features.size) for candidates in plan.candidates
        ),
    }


def _lbvh_evidence(
    result: _D.LBVHContactSearchResult,
    plan: _D.LBVHContactSearchPlan,
    prepared: Prepared,
    /,
) -> dict[str, Any]:
    evidence = result.evidence
    oracle = prepared.oracle
    finite = oracle.vertices.ids.size + oracle.edges.ids.size + oracle.faces.ids.size
    expected_nodes = 2 * finite - 1
    expected_plane_work = sum(
        family.product for family in prepared.families if family.right is None
    )
    node_count = int(np.asarray(evidence.node_count))
    plane_work = int(np.asarray(evidence.plane_pair_work))
    return {
        "traversal_visits": int(np.asarray(evidence.traversal_visits)),
        "maximum_traversal_visits": plan.maximum_traversal_visits,
        "visit_overflow": bool(np.asarray(evidence.visit_overflow)),
        "stack_overflow": bool(np.asarray(evidence.stack_overflow)),
        "output_overflow": int(np.asarray(evidence.output_overflow)),
        "tree_depth": int(np.asarray(evidence.tree_depth)),
        "maximum_tree_depth": plan.maximum_tree_depth,
        "node_count": node_count,
        "expected_node_count_for_finite_primitives": expected_nodes,
        "planes_excluded_from_tree": node_count == expected_nodes,
        "plane_pair_work": plane_work,
        "expected_plane_pair_work": expected_plane_work,
        "plane_work_matches_analytic_route": plane_work == expected_plane_work,
        "duplicate_code_count": int(np.asarray(evidence.duplicate_code_count)),
        "duplicate_route_count": int(np.asarray(evidence.duplicate_route_count)),
        "finite_bounds": bool(np.asarray(evidence.finite_bounds)),
        "complete": bool(np.asarray(evidence.complete)),
        "discovery_note": (
            "Discovered counts are work actually found; after traversal exhaustion "
            "they are not the required capacity."
        ),
    }


def _device_backend(
    name: str, prepared: Prepared, settings: Settings, /
) -> dict[str, Any]:
    product = sum(family.product for family in prepared.families)
    if name == "compiled" and product > settings.compiled_row_limit:
        return _refused(
            name,
            f"Compiled preparation enumerates {product} oriented primitive pairs, "
            f"above --compiled-row-limit {settings.compiled_row_limit}.",
        )
    visit_budget = (
        settings.visit_budget
        if prepared.spec.visit_budget is None
        else prepared.spec.visit_budget
    )
    plan, preparation_seconds = measure_synchronized(
        lambda: _device_plan(name, prepared, visit_budget)
    )
    scene = prepared.scene
    start = prepared.start
    search = _compiled_phase(_search, (plan, scene, start, prepared.end), settings)
    result = search.value
    epoch = result.epoch
    kinematics = _compiled_phase(
        _kinematics,
        (scene, epoch, start, prepared.velocities, start, prepared.step_size),
        settings,
    )
    pipeline = _compiled_phase(
        _pipeline,
        (
            plan,
            scene,
            start,
            prepared.end,
            prepared.velocities,
            start,
            prepared.step_size,
        ),
        settings,
    )
    fused_result, fused_kinematics = pipeline.value
    summary = _search_summary(epoch)
    kinematics_summary = _kinematics_summary(
        epoch, kinematics.value, prepared.oracle, prepared.tolerance
    )
    fused_consistent = (
        _epoch_rows(fused_result.epoch) == _epoch_rows(epoch)
        and _status_name(fused_result.epoch.status) == summary["status"]
        and bool(np.asarray(fused_kinematics.evidence.successful))
        == kinematics_summary["successful"]
        and int(np.asarray(fused_kinematics.evidence.active_contacts))
        == kinematics_summary["active_contacts"]
    )
    backend_evidence = (
        _compiled_evidence(result, plan)
        if isinstance(plan, _D.CompiledContactSearchPlan)
        else _lbvh_evidence(result, plan, prepared)
    )
    pipeline_record = pipeline.record
    cold = (
        preparation_seconds
        + pipeline_record["lowering_seconds"]
        + pipeline_record["compilation_seconds"]
        + pipeline_record["first_call_seconds"]
    )
    refresh = [
        _device_refresh(
            plan,
            search,
            kinematics,
            epoch,
            variant,
            oracle,
            reference,
            prepared,
            settings,
        )
        for variant, oracle, reference in zip(
            prepared.variants,
            prepared.variant_oracles,
            prepared.variant_references,
            strict=True,
        )
    ]
    return {
        "backend": name,
        "kind": "device",
        "outcome": _outcome(summary, kinematics_summary),
        "timed": True,
        "expectation": _expectation(prepared.spec, name, summary, kinematics_summary),
        "plan_preparation_seconds": preparation_seconds,
        "visit_budget": visit_budget if name == "lbvh" else None,
        "logical_plan_bytes": logical_array_bytes(plan),
        "phases": {
            "search": search.record,
            "kinematics": kinematics.record,
            "end_to_end": pipeline_record,
        },
        "cold_pipeline_seconds": {
            "value": cold,
            "derivation": (
                "sum of measured plan preparation and fused pipeline lowering, "
                "compilation and first call"
            ),
        },
        "search": summary,
        "counts": _kind_counts(epoch, prepared.reference),
        "parity": _parity(epoch, prepared.reference),
        "kinematics": kinematics_summary,
        "fused_pipeline_consistent": fused_consistent,
        "backend_evidence": backend_evidence,
        "refresh": refresh,
        "_epoch": epoch,
    }


def _device_refresh(
    plan: DevicePlan,
    search: Phase,
    kinematics: Phase,
    stale: _D.ContactCandidateEpoch,
    variant: RefreshVariant,
    oracle: OracleScene,
    reference: Rows,
    prepared: Prepared,
    settings: Settings,
    /,
) -> dict[str, Any]:
    """Reuse the ahead-of-time executables; a signature change would raise."""
    start = prepared.start
    stale_kinematics = kinematics.executable(
        variant.scene, stale, start, prepared.velocities, start, prepared.step_size
    )
    result, samples = measure_repeated(
        lambda: search.executable(plan, variant.scene, start, prepared.end),
        warmup=settings.warmup,
        repeats=settings.repeats,
    )
    epoch = result.epoch
    refreshed = kinematics.executable(
        variant.scene, epoch, start, prepared.velocities, start, prepared.step_size
    )
    summary = _search_summary(epoch)
    summary["warm_search"] = samples.to_seconds_dict()
    return {
        "variant": variant.name,
        "executables_reused": True,
        "stale_epoch_binds": bool(np.asarray(stale.binds(variant.scene))),
        "stale_kinematics_successful": bool(
            np.asarray(stale_kinematics.evidence.successful)
        ),
        "search": summary,
        "parity": _parity(epoch, reference),
        "kinematics": _kinematics_summary(epoch, refreshed, oracle, prepared.tolerance),
    }


def _route_keys_agreement(backends: dict[str, dict[str, Any]], /) -> dict[str, Any]:
    dense = backends.get("dense", {})
    reference_epoch = dense.get("_epoch")
    agreement: dict[str, Any] = {}
    for name, record in backends.items():
        epoch = record.get("_epoch")
        if name == "dense":
            continue
        if reference_epoch is None or epoch is None:
            agreement[name] = None
        elif not (
            bool(np.asarray(reference_epoch.successful))
            and bool(np.asarray(epoch.successful))
        ):
            agreement[name] = None
        else:
            agreement[name] = _valid_route_keys(epoch) == _valid_route_keys(
                reference_epoch
            )
    return {
        "reference": "dense host canonical route keys (both epochs successful)",
        "equal": agreement,
    }


def _matched_ratios(backends: dict[str, dict[str, Any]], /) -> list[dict[str, Any]]:
    """Ratios only between successful, reference-matched, kinematics-successful work."""
    comparisons: list[dict[str, Any]] = []

    def matched(record: Mapping[str, Any], /) -> bool:
        return (
            record.get("outcome") == "successful"
            and record["parity"]["status"] == "matched"
            and record["kinematics"]["gap_reference"]["status"] != "failed"
        )

    for host in _HOST_BACKENDS:
        for device in _DEVICE_BACKENDS:
            host_record = backends[host]
            device_record = backends[device]
            entry: dict[str, Any] = {
                "numerator": f"{host} host synchronized search median",
                "denominator": f"{device} warm device search median",
                "ratio": None,
                "reason": None,
            }
            if not (host_record.get("timed") and device_record.get("timed")):
                entry["reason"] = "A backend was refused before timing."
            elif not (matched(host_record) and matched(device_record)):
                entry["reason"] = (
                    "Unmatched or unsuccessful work is never compared: outcomes "
                    f"{host_record['outcome']} and {device_record['outcome']}."
                )
            else:
                numerator = host_record["search"]["host_synchronized_search"][
                    "median_seconds"
                ]
                denominator = device_record["phases"]["search"]["warm"]["median_seconds"]
                if denominator and denominator > 0.0:
                    entry["ratio"] = numerator / denominator
                else:
                    entry["reason"] = "The device median was zero or unavailable."
            comparisons.append(entry)
    return comparisons


# --------------------------------------------------------------------------
# Case driver
# --------------------------------------------------------------------------


def _counts(geometry: HostGeometry, oracle: OracleScene, /) -> dict[str, Any]:
    static = int(
        np.sum(oracle.vertices.static)
        + np.sum(oracle.edges.static)
        + np.sum(oracle.faces.static)
    )
    sphere_profile = geometry.profile == Profile.SPHERE_CAPSULE
    return {
        "dimension": geometry.dimension,
        "primitive_profile": geometry.profile.name,
        "participants": len(geometry.participants),
        "vertices": int(oracle.vertices.ids.size),
        "edges": int(oracle.edges.ids.size),
        "triangles": int(oracle.faces.ids.size),
        "planes": int(oracle.plane_ids.size),
        "spheres": int(np.sum(oracle.isolated)) if sphere_profile else 0,
        "capsules": int(oracle.edges.ids.size) if sphere_profile else 0,
        "finite_primitives": int(
            oracle.vertices.ids.size + oracle.edges.ids.size + oracle.faces.ids.size
        ),
        "static_finite_features": static,
        "activation_distance": geometry.activation,
        "envelope_radius": geometry.envelope,
        "moving_vertices": int(
            sum(
                np.sum(np.any(value.start != value.end, axis=-1))
                for value in geometry.participants
            )
        ),
    }


def _not_run(spec: CaseSpec, counts: dict[str, Any], reason: str, /) -> dict[str, Any]:
    return {
        "case": spec.to_dict(),
        "counts": counts,
        "status": "not-run",
        "reason": reason,
    }


def _run_case(spec: CaseSpec, settings: Settings, /) -> dict[str, Any]:
    dtype = settings.numpy_dtype
    geometry, generation_seconds = measure_host(lambda: _geometry(spec, dtype))
    oracle = _oracle_scene(geometry)
    families = _oracle_families(oracle, geometry.profile, geometry.dimension)
    counts = _counts(geometry, oracle)
    work = sum(family.product for family in families)
    if work > settings.oracle_pair_limit:
        return _not_run(
            spec,
            counts,
            f"The independent reference would evaluate {work} pairs, above "
            f"--oracle-pair-limit {settings.oracle_pair_limit}.",
        )
    scene, topology_seconds = measure_synchronized(
        lambda: _prepare_scene(geometry, dtype)
    )
    variants: tuple[RefreshVariant, ...] = ()
    if spec.capacity != "shortfall":
        variants = _refresh_variants(scene, geometry, dtype)
    start_host = oracle.start.astype(dtype)
    end_host = np.concatenate([value.end for value in geometry.participants]).astype(
        dtype
    )
    velocities_host = ((end_host - start_host) / _STEP_SIZE).astype(dtype)
    (start, end, velocities, step_size), transfer_seconds = measure_synchronized(
        lambda: (
            jax.device_put(start_host),
            jax.device_put(end_host),
            jax.device_put(velocities_host),
            jax.device_put(np.asarray(_STEP_SIZE, dtype=dtype)),
        )
    )
    reference, reference_seconds = measure_host(
        lambda: _oracle_rows(families, oracle, geometry.activation, geometry.envelope)
    )
    variant_oracles = tuple(_oracle_scene(variant.geometry) for variant in variants)
    variant_references = tuple(
        _oracle_rows(
            _oracle_families(value, geometry.profile, geometry.dimension),
            value,
            geometry.activation,
            geometry.envelope,
        )
        for value in variant_oracles
    )
    capacities = _capacities(spec, (reference, *variant_references), settings)
    tolerance = (
        settings.gap_tolerance_factor
        * math.sqrt(float(np.finfo(dtype).eps))
        * oracle.scale
    )
    prepared = Prepared(
        spec,
        geometry,
        oracle,
        families,
        reference,
        scene,
        variants,
        variant_oracles,
        variant_references,
        start,
        end,
        velocities,
        step_size,
        capacities,
        tolerance,
    )
    backends: dict[str, dict[str, Any]] = {}
    for name in _HOST_BACKENDS:
        backends[name] = _host_backend(name, prepared, settings)
    for name in _DEVICE_BACKENDS:
        backends[name] = _device_backend(name, prepared, settings)
    keys = _route_keys_agreement(backends)
    ratios = _matched_ratios(backends)
    for record in backends.values():
        record.pop("_epoch", None)
    return {
        "case": spec.to_dict(),
        "status": "run",
        "counts": counts,
        "capacities": capacities.to_dict(),
        "host_preparation": {
            "geometry_generation_seconds": generation_seconds,
            "topology_preparation_seconds": topology_seconds,
            "topology_preparation_note": (
                "Collision plans, catalogues and the scene, including their small "
                "device constants; synchronized."
            ),
            "transfer_seconds": transfer_seconds,
            "transfer_bytes": logical_array_bytes((start, end, velocities, step_size)),
            "logical_scene_bytes": logical_array_bytes(scene),
        },
        "numeric_refresh": [
            {
                "variant": variant.name,
                "description": variant.description,
                "bind_seconds": variant.bind_seconds,
                "reference_rows": sum(len(value) for value in rows.values()),
            }
            for variant, rows in zip(variants, variant_references, strict=True)
        ],
        "reference": {
            "method": "independent-numpy-full-path-box-and-analytic-plane-contract",
            "seconds": reference_seconds,
            "evaluated_pairs": work,
            "rows": {_kind_name(kind): len(values) for kind, values in reference.items()},
            "total_rows": sum(len(values) for values in reference.values()),
            "family_row_bounds": {
                _kind_name(int(family.kind)): family.row_bound for family in families
            },
            "gap_tolerance": tolerance,
            "gap_tolerance_formula": (
                "factor * sqrt(eps(dtype)) * max(1, coordinate, radius and "
                "plane-offset magnitudes)"
            ),
        },
        "backends": backends,
        "route_keys": keys,
        "matched_ratios": ratios,
    }


def _summary(records: Sequence[Mapping[str, Any]], /) -> dict[str, Any]:
    outcomes: dict[str, dict[str, int]] = {name: {} for name in _BACKENDS}
    expectation_failures: list[dict[str, Any]] = []
    parity_failures: list[dict[str, Any]] = []
    propagation_failures: list[dict[str, Any]] = []
    gap_failures: list[dict[str, Any]] = []
    refresh_failures: list[dict[str, Any]] = []
    not_run: list[str] = []
    for record in records:
        case = record["case"]["name"]
        if record["status"] != "run":
            not_run.append(case)
            continue
        for name, backend in record["backends"].items():
            outcome = backend["outcome"]
            outcomes[name][outcome] = outcomes[name].get(outcome, 0) + 1
            if not backend["timed"]:
                continue
            expectation = backend["expectation"]
            if not expectation["met"]:
                expectation_failures.append(
                    {"case": case, "backend": name, **expectation}
                )
            if backend["parity"]["status"] == "mismatched":
                parity_failures.append(
                    {"case": case, "backend": name, **backend["parity"]}
                )
            propagation = backend["kinematics"]["failure_propagation"]
            if not propagation["consistent"]:
                propagation_failures.append(
                    {"case": case, "backend": name, **propagation}
                )
            gap = backend["kinematics"]["gap_reference"]
            if gap["status"] == "failed":
                gap_failures.append(
                    {
                        "case": case,
                        "backend": name,
                        "overall_max_abs_error": gap["overall_max_abs_error"],
                        "tolerance": gap["tolerance"],
                    }
                )
            for refresh in backend["refresh"]:
                problems: list[str] = []
                expected = record["case"]["expected_search_status"][name]
                refreshed_status = refresh["search"]["status"]
                if (
                    expected == Status.SUCCESS.name
                    and refreshed_status != Status.SUCCESS.name
                ):
                    problems.append(f"refreshed search refused: {refreshed_status}")
                if refresh["stale_epoch_binds"]:
                    problems.append("stale epoch still binds refreshed geometry")
                if refresh["stale_kinematics_successful"]:
                    problems.append("stale epoch kinematics reported success")
                if refresh["parity"]["status"] == "mismatched":
                    problems.append("refreshed epoch mismatched the reference")
                if refresh["kinematics"]["gap_reference"]["status"] == "failed":
                    problems.append("refreshed gaps failed the analytic reference")
                if problems:
                    refresh_failures.append(
                        {
                            "case": case,
                            "backend": name,
                            "variant": refresh["variant"],
                            "problems": problems,
                        }
                    )
    return {
        "cases": len(records),
        "not_run_cases": not_run,
        "backend_outcomes": outcomes,
        "expectation_failures": expectation_failures,
        "parity_failures": parity_failures,
        "failure_propagation_failures": propagation_failures,
        "gap_reference_failures": gap_failures,
        "refresh_failures": refresh_failures,
        "passed": not (
            expectation_failures
            or parity_failures
            or propagation_failures
            or gap_failures
            or refresh_failures
            or not_run
        ),
    }


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-separated canonical contact search through contact kinematics "
            "with an independent reference. Without arguments, runs the legacy "
            "256-segment self-contact curve."
        )
    )
    parser.add_argument(
        "--suite",
        choices=tuple(_SUITE_DEFAULTS),
        default="default",
        help=(
            "default: legacy curve; smoke: small diagnostic rows; "
            "scaling: bounded scaling rows."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Run only the named case(s) of the selected suite.",
    )
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None, help="Warm device samples.")
    parser.add_argument("--host-repeats", type=int, default=None)
    parser.add_argument("--capacity-factor", type=float, default=2.0)
    parser.add_argument("--visit-budget", type=int, default=1 << 20)
    parser.add_argument("--compiled-row-limit", type=int, default=4_000_000)
    parser.add_argument("--dense-pair-limit", type=int, default=4_000_000)
    parser.add_argument("--oracle-pair-limit", type=int, default=64_000_000)
    parser.add_argument("--gap-tolerance-factor", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _settings(arguments: argparse.Namespace, /) -> Settings:
    warmup, repeats, host_repeats = _SUITE_DEFAULTS[arguments.suite]
    settings = Settings(
        suite=arguments.suite,
        dtype=arguments.dtype,
        warmup=warmup if arguments.warmup is None else arguments.warmup,
        repeats=repeats if arguments.repeats is None else arguments.repeats,
        host_repeats=(
            host_repeats if arguments.host_repeats is None else arguments.host_repeats
        ),
        capacity_factor=arguments.capacity_factor,
        visit_budget=arguments.visit_budget,
        compiled_row_limit=arguments.compiled_row_limit,
        dense_pair_limit=arguments.dense_pair_limit,
        oracle_pair_limit=arguments.oracle_pair_limit,
        gap_tolerance_factor=arguments.gap_tolerance_factor,
    )
    if settings.warmup < 0 or settings.repeats < 1 or settings.host_repeats < 1:
        raise ValueError("warmup must be nonnegative and repeat counts positive.")
    if not math.isfinite(settings.capacity_factor) or settings.capacity_factor < 1.0:
        raise ValueError("--capacity-factor must be finite and at least one.")
    if settings.visit_budget <= 0:
        raise ValueError("--visit-budget must be positive.")
    return settings


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    settings = _settings(arguments)
    cache_directory = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    cache_enabled = os.environ.get("JAX_ENABLE_COMPILATION_CACHE")
    cases = _suite(settings.suite)
    if arguments.case:
        known = {case.name for case in cases}
        unknown = sorted(set(arguments.case) - known)
        if unknown:
            raise ValueError(f"Unknown case(s) for suite {settings.suite!r}: {unknown}.")
        cases = tuple(case for case in cases if case.name in arguments.case)
    records: list[dict[str, Any]] = []
    for case in cases:
        records.append(_run_case(case, settings))
        # Each case owns its cold compilations and warm samples.
        jax.clear_caches()
    payload: dict[str, Any] = {
        "benchmark": "contact-search",
        "command": [
            Path(sys.argv[0]).name,
            *(sys.argv[1:] if argv is None else argv),
        ],
        "settings": settings.to_dict(),
        "cache_policy": {
            "persistent_compilation_cache_dir": cache_directory,
            "jax_enable_compilation_cache_environment": cache_enabled,
            "persistent_compilation_cache_possible": cache_directory is not None
            and (cache_enabled or "true").lower() not in ("0", "false"),
            "persistent_policy": (
                "Inherited from the launch environment and never changed here; "
                "cold compile timings require no persistent cache directory or "
                "JAX_ENABLE_COMPILATION_CACHE=false."
            ),
            "in_process": (
                "Every device phase is lowered and compiled ahead of time; first "
                "calls exclude compilation; refreshes reuse the same executables; "
                "jax.clear_caches() runs after every case."
            ),
            "cold_process_case": records[0]["case"]["name"] if records else None,
        },
        "provider_memory": {
            "measured": False,
            "reason": (
                "No device allocator or external provider memory was queried; "
                "compiler buffer sizes are XLA estimates and logical bytes count "
                "reachable array payloads only."
            ),
        },
        "records": records,
        "summary": _summary(records),
    }
    driver = Path(__file__).resolve()
    payload["environment"] = capture_environment().to_dict()
    payload["identity"] = capture_benchmark_identity(
        driver.parent.parent, driver, tuple(payload)
    ).to_dict()
    if arguments.output is not None:
        print(write_json_atomic(arguments.output, payload))
    else:
        print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
