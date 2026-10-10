#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded, source-digested meshing corpus and frozen physical requests.

This module owns source/request consistency, not construction or acceptance
algorithms. Geometry coordinates are in meters and generated analytically.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, NoReturn

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint


@dataclass(frozen=True, slots=True)
class MeshingCase:
    name: str
    workflow: str
    dimension: int
    family: str
    expected_admission: str
    controls: tuple[str, ...]
    expected_regions: tuple[tuple[str, float], ...]
    provenance: str
    license_id: str = "LicenseRef-PHYDRA-Proprietary"
    source_author: str = "PHYDRA, Inc."

    def __post_init__(self) -> None:
        if self.expected_admission not in ("positive", "refusal"):
            raise ValueError("Meshing corpus admission must be positive or refusal.")
        if self.dimension not in (2, 3) or not self.name or not self.workflow:
            raise ValueError("Meshing corpus cases require a named 2D or 3D workflow.")
        if not self.controls or not self.expected_regions:
            raise ValueError(
                "Meshing corpus cases require controls and analytic references."
            )
        if not self.license_id or not self.source_author or not self.provenance:
            raise ValueError(
                "Meshing corpus source provenance and license are mandatory."
            )

    def to_record(self) -> dict[str, object]:
        reference = {
            "kind": "independent-analytic-region-measures",
            "expected_region_measures": [
                [name, measure] for name, measure in self.expected_regions
            ],
            "coordinate_unit": "meter",
        }
        return {
            "name": self.name,
            "workflow": self.workflow,
            "dimension": self.dimension,
            "family": self.family,
            "expected_admission": self.expected_admission,
            "controls": list(self.controls),
            "expected_region_measures": dict(self.expected_regions),
            "minimum_angle_radians": np.radians(25.0)
            if "minimum-angle" in self.controls
            else None,
            "minimum_mean_ratio": 0.05 if "minimum-quality" in self.controls else None,
            "coordinate_unit": "meter",
            "source_provenance": {
                "author": self.source_author,
                "license": self.license_id,
                "origin": self.provenance,
                "redistribution": "repository-internal",
            },
            "independent_reference": {
                **reference,
                "reference_digest": canonical_fingerprint(reference),
            },
        }


CORPUS = (
    MeshingCase(
        "planar-hole-feature",
        "planar-features",
        2,
        "triangle",
        "positive",
        ("hole", "embedded-segment", "hard-size", "minimum-angle"),
        (("plate", 3.84),),
        "Analytic square minus square; PHYDRA original corpus",
    ),
    MeshingCase(
        "plc-material-cavity",
        "constrained-solid",
        3,
        "tetrahedron",
        "positive",
        (
            "cavity",
            "material-interface",
            "embedded-segment",
            "hard-size",
            "minimum-quality",
        ),
        (("left", 1.0), ("right", 0.875)),
        "Analytic two-box material partition and cavity; PHYDRA original corpus",
    ),
    MeshingCase(
        "planar-thin-channel",
        "planar-features",
        2,
        "triangle",
        "positive",
        ("thin-channel", "hard-size", "minimum-angle"),
        (("plate", 0.324),),
        "Analytic dumbbell with width 0.02; PHYDRA original corpus",
    ),
    MeshingCase(
        "planar-capacity-refusal",
        "resource-boundary",
        2,
        "triangle",
        "refusal",
        ("hole", "embedded-segment", "hard-size", "minimum-angle", "maximum-vertices"),
        (("plate", 3.84),),
        "Same plate with deliberately impossible entity budget",
    ),
    MeshingCase(
        "planar-stale-revision",
        "source-identity",
        2,
        "triangle",
        "refusal",
        (
            "hole",
            "embedded-segment",
            "hard-size",
            "minimum-angle",
            "stale-source-revision",
        ),
        (("plate", 3.84),),
        "Same plate with mismatched request revision",
    ),
    *(
        MeshingCase(
            name,
            "constrained-solid",
            3,
            "tetrahedron",
            "positive",
            controls,
            regions,
            "Independent declared axis-aligned box union",
        )
        for name, controls, regions in (
            ("plc-nonconvex", ("nonconvex", "soft-size"), (("solid", 3.0),)),
            (
                "plc-disconnected",
                ("disconnected", "hard-size", "minimum-quality"),
                (("solid", 2.0),),
            ),
            (
                "plc-thin-channel",
                ("thin-channel", "hard-size", "minimum-quality"),
                (("solid", 0.324),),
            ),
            (
                "plc-triple-junction",
                ("triple-junction", "material-interface", "hard-size", "minimum-quality"),
                (("first", 1.0), ("second", 1.0), ("third", 2.0)),
            ),
        )
    ),
    *(
        MeshingCase(
            f"plc-{kind}-refusal",
            "resource-boundary",
            3,
            "tetrahedron",
            "refusal",
            (
                "cavity",
                "material-interface",
                "embedded-segment",
                "hard-size",
                "minimum-quality",
                f"maximum-{kind}",
            ),
            (("left", 1.0), ("right", 0.875)),
            "Original cavity source with one deliberately insufficient allowance",
        )
        for kind in ("work", "queries", "cavity", "entities", "storage")
    ),
    MeshingCase(
        "plc-small-angle",
        "constrained-solid",
        3,
        "tetrahedron",
        "positive",
        ("small-angle", "soft-size"),
        (("solid", 16.0),),
        "Original test_small_angle_wedge_is_protected_and_meshed",
    ),
    MeshingCase(
        "plc-fixed-boundary",
        "constrained-solid",
        3,
        "tetrahedron",
        "positive",
        ("fixed-boundary", "soft-size"),
        (("solid", 1.0),),
        "Original test_fixed_boundary_triangles_are_published_unchanged",
    ),
    MeshingCase(
        "plc-incompatible-constraints",
        "constrained-solid",
        3,
        "tetrahedron",
        "refusal",
        ("intersecting-constraints", "soft-size"),
        (("a", 1.0), ("b", 1.0)),
        "Original test_intersecting_facets_are_refused_with_the_polygons",
    ),
    MeshingCase(
        "plc-sliver-rich",
        "constrained-solid",
        3,
        "tetrahedron",
        "positive",
        ("sliver-rich", "soft-size", "dihedral-25"),
        (("solid", 1.0),),
        "Original test_native_volume_exudation source and four-pass schedule",
    ),
    MeshingCase(
        "plc-stale-revision",
        "source-identity",
        3,
        "tetrahedron",
        "refusal",
        (
            "cavity",
            "material-interface",
            "embedded-segment",
            "hard-size",
            "minimum-quality",
            "stale-source-revision",
        ),
        (("left", 1.0), ("right", 0.875)),
        "Original cavity with deliberately stale request revision",
    ),
)
CASE_NAMES = tuple(case.name for case in CORPUS)
SURFACE_CASE = MeshingCase(
    "sphere-feature-surface",
    "planar-features",
    2,
    "triangle",
    "positive",
    ("seam", "collapsed-poles", "hard-size", "hard-source-deviation"),
    (("surface", 4.0 * np.pi),),
    "Analytic unit sphere; PHYDRA original source fixture",
)
MANDATORY_WORKFLOWS = (
    "planar-features",
    "constrained-solid",
    "implicit-domain",
    "labeled-image",
    "imperfect-surface",
    "native-cad",
    "periodic-system",
    "hybrid-layers",
    "general-quads-hexes",
    "polyhedral-system",
    "distributed-lifecycle",
    "moving-overset",
    "design-learned-loop",
)


def meshing_case(name: str) -> MeshingCase:
    for case in (*CORPUS, SURFACE_CASE):
        if case.name == name:
            return case
    raise ValueError(f"Unknown meshing corpus case: {name}")


def _box(lower: tuple[float, ...], upper: tuple[float, ...]) -> np.ndarray:
    return np.asarray(
        [[(upper if i >> k & 1 else lower)[k] for k in range(3)] for i in range(8)],
        dtype=np.float64,
    )


def _solid_payload() -> dict[str, Any]:
    loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    left = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    right = _box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    cavity = _box((1.25, 0.25, 0.25), (1.75, 0.75, 0.75))
    vertices = np.concatenate((left, right[[1, 3, 5, 7]], cavity))
    right_index = {0: 1, 2: 3, 4: 5, 6: 7, 1: 8, 3: 9, 5: 10, 7: 11}
    facets: list[int] = []
    polygons: list[tuple[int, ...]] = []
    for loop in loops:
        if loop != (1, 3, 7, 5):
            polygons.append(loop)
            facets.append(0)
        if loop != (0, 4, 6, 2):
            polygons.append(tuple(right_index[v] for v in loop))
            facets.append(1)
    polygons.append((1, 3, 7, 5))
    facets.append(2)
    polygons.extend(tuple(v + 12 for v in loop) for loop in loops)
    facets.extend([3] * 6)
    vertices = np.concatenate(
        (vertices, np.asarray(((0.25, 0.25, 0.25), (0.75, 0.75, 0.75)), dtype=np.float64))
    )
    return {
        "vertices": vertices.tolist(),
        "polygons": polygons,
        "facets": facets,
        "facet_regions": [[-1, 0], [-1, 1], [1, 0], [1, -1]],
        "region_names": ("left", "right"),
        "segments": [[20, 21]],
    }


def declared_boxes(
    case: MeshingCase,
) -> tuple[tuple[int, tuple[float, ...], tuple[float, ...]], ...]:
    """Scientific reference, independent of PLC triangulation and publication."""
    references = {
        "plc-nonconvex": (
            (0, (0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (0, (0.0, 1.0, 0.0), (1.0, 2.0, 1.0)),
        ),
        "plc-disconnected": (
            (0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            (0, (2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
        ),
        "plc-thin-channel": (
            (0, (0.0, 0.0, 0.0), (0.4, 0.4, 1.0)),
            (0, (0.4, 0.19, 0.0), (0.6, 0.21, 1.0)),
            (0, (0.6, 0.0, 0.0), (1.0, 0.4, 1.0)),
        ),
        "plc-triple-junction": (
            (0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            (1, (1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (2, (0.0, 1.0, 0.0), (2.0, 2.0, 1.0)),
        ),
    }
    return references.get(
        case.name,
        ((0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (1, (1.0, 0.0, 0.0), (2.0, 1.0, 1.0))),
    )


def analytic_region_indices(case: MeshingCase, points: np.ndarray) -> np.ndarray:
    """Classify by declared solids; cavity is explicitly outside every region."""
    if case.name == "plc-small-angle":
        x, y, z = np.moveaxis(points, -1, 0)
        inside = (
            (x >= 0.0) & (x <= 16.0) & (np.abs(y) <= x / 16.0) & (z >= 0.0) & (z <= 1.0)
        )
        return np.where(inside, 0, -1).astype(np.int64)
    if case.name in ("plc-fixed-boundary", "plc-sliver-rich"):
        return np.where(np.all((points >= 0.0) & (points <= 1.0), axis=-1), 0, -1).astype(
            np.int64
        )
    result = np.full(points.shape[:-1], -1, dtype=np.int64)
    for region, lower, upper in declared_boxes(case):
        inside = np.all((points >= lower) & (points <= upper), axis=-1)
        result[inside] = region
    if "cavity" in case.controls:
        result[
            np.all((points > (1.25, 0.25, 0.25)) & (points < (1.75, 0.75, 0.75)), axis=-1)
        ] = -1
    return result


def _box_union_payload(case: MeshingCase) -> dict[str, Any]:
    """Tile declared boxes at all source planes, retaining shared interfaces."""
    boxes = declared_boxes(case)
    axes = [
        sorted({bound[k] for _, lower, upper in boxes for bound in (lower, upper)})
        for k in range(3)
    ]
    vertices: list[tuple[float, ...]] = []
    vertex_ids: dict[tuple[float, ...], int] = {}
    faces: dict[tuple[int, ...], tuple[tuple[int, ...], int, int]] = {}
    loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    for i in np.ndindex(*(len(axis) - 1 for axis in axes)):
        lower = tuple(axes[k][i[k]] for k in range(3))
        upper = tuple(axes[k][i[k] + 1] for k in range(3))
        region = int(
            analytic_region_indices(case, np.asarray([(np.asarray(lower) + upper) / 2]))[
                0
            ]
        )
        if region < 0:
            continue
        ids = []
        for point in _box(lower, upper):
            key = tuple(point)
            if key not in vertex_ids:
                vertex_ids[key] = len(vertices)
                vertices.append(key)
            ids.append(vertex_ids[key])
        for loop in loops:
            polygon = tuple(ids[v] for v in loop)
            key = tuple(sorted(polygon))
            if key in faces:
                old, previous, _ = faces.pop(key)
                if previous != region:
                    faces[key] = (old, previous, region)
            else:
                faces[key] = (polygon, region, -1)
    polygons, pairs = [], []
    for polygon, negative, positive in faces.values():
        polygons.append(polygon)
        pairs.append((positive, negative))
    return {
        "vertices": vertices,
        "polygons": polygons,
        "facets": list(range(len(polygons))),
        "facet_regions": pairs,
        "region_names": tuple(name for name, _ in case.expected_regions),
        "segments": [],
    }


def source_payload(case: MeshingCase) -> dict[str, Any]:
    if case.name in ("plc-nonconvex", "plc-thin-channel"):
        base = (
            ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0))
            if case.name == "plc-nonconvex"
            else source_payload(meshing_case("planar-thin-channel"))["vertices"]
        )
        count = len(base)
        return {
            "vertices": [(x, y, z) for z in (0.0, 1.0) for x, y in base],
            "polygons": [
                tuple(reversed(range(count))),
                tuple(range(count, 2 * count)),
                *(
                    (i, (i + 1) % count, (i + 1) % count + count, i + count)
                    for i in range(count)
                ),
            ],
            "facets": [0] * (count + 2),
            "facet_regions": [[-1, 0]],
            "region_names": ("solid",),
            "segments": [],
        }
    if case.name == SURFACE_CASE.name:
        return {
            "radius": 1.0,
            "center": [0.0, 0.0, 0.0],
            "chart_bounds": [[0.0, 2.0 * np.pi], [-0.5 * np.pi, 0.5 * np.pi]],
            "source_factory": "examples.native_surface_meshing.sphere",
        }
    if case.name == "plc-small-angle":
        base = ((0.0, 0.0), (16.0, -1.0), (16.0, 1.0))
        return {
            "vertices": [(x, y, z) for z in (0.0, 1.0) for x, y in base],
            "polygons": ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
            "facets": [0] * 5,
            "facet_regions": [[-1, 0]],
            "region_names": ("solid",),
            "segments": [],
        }
    if case.name in (
        "plc-fixed-boundary",
        "plc-incompatible-constraints",
        "plc-sliver-rich",
    ):
        loops = (
            (0, 2, 3, 1),
            (4, 5, 7, 6),
            (0, 1, 5, 4),
            (2, 6, 7, 3),
            (0, 4, 6, 2),
            (1, 3, 7, 5),
        )
        vertices = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        if case.name == "plc-sliver-rich":
            return {
                "vertices": np.concatenate(
                    (vertices, np.asarray(((0.5, 0.5, 0.001), (0.5, 0.5, 0.999))))
                ).tolist(),
                "polygons": loops,
                "facets": [0] * 6,
                "facet_regions": [[-1, 0]],
                "region_names": ("solid",),
                "segments": [],
            }
        if case.name == "plc-fixed-boundary":
            triangles = [
                tri
                for a, b, c, d in loops
                for tri in (
                    ((a, b, c), (a, c, d)) if a % 2 == 0 else ((a, b, d), (b, c, d))
                )
            ]
            return {
                "vertices": vertices.tolist(),
                "polygons": triangles,
                "facets": [0] * 12,
                "facet_regions": [[-1, 0]],
                "region_names": ("solid",),
                "segments": [],
                "boundary": "fixed",
            }
        return {
            "vertices": np.concatenate((vertices, vertices + 0.5)).tolist(),
            "polygons": [*loops, *(tuple(v + 8 for v in loop) for loop in loops)],
            "facets": [0] * 6 + [1] * 6,
            "facet_regions": [[-1, 0], [-1, 1]],
            "region_names": ("a", "b"),
            "segments": [],
        }
    if case.dimension == 3:
        if case.name in ("plc-disconnected", "plc-triple-junction"):
            return _box_union_payload(case)
        return _solid_payload()
    if case.name == "planar-thin-channel":
        vertices = [
            [0.0, 0.0],
            [0.4, 0.0],
            [0.4, 0.19],
            [0.6, 0.19],
            [0.6, 0.0],
            [1.0, 0.0],
            [1.0, 0.4],
            [0.6, 0.4],
            [0.6, 0.21],
            [0.4, 0.21],
            [0.4, 0.4],
            [0.0, 0.4],
        ]
        return {"vertices": vertices, "loops": [list(range(12))], "embedded": []}
    return {
        "vertices": [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [0.8, 0.8],
            [0.8, 1.2],
            [1.2, 1.2],
            [1.2, 0.8],
        ],
        "loops": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "embedded": [[0.2, 0.2], [0.5, 1.7]],
    }


def corpus_identity(case: MeshingCase) -> dict[str, object]:
    payload = json.dumps(
        source_payload(case), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    digest = sha256(payload).hexdigest()
    return {
        **case.to_record(),
        "source_artifact": {
            "kind": "analytic-generated-fixture",
            "locator": f"tools/_meshing_cases.py::{case.name}",
            "media_type": "application/json",
            "bytes": len(payload),
            "sha256": digest,
            "license": case.license_id,
        },
        "source_digest": digest,
        "source_revision": canonical_fingerprint(
            {
                "kind": "analytic-corpus-source",
                "digest": digest,
                "coordinate_unit": "meter",
                "regions": [name for name, _ in case.expected_regions],
            }
        ),
    }


def prepare_case(
    case: MeshingCase,
    resolution: int,
    capacity: int,
    timeout: float,
) -> tuple[Any, Any, Any]:
    """Freeze a source, physical request and native route without executing it."""
    if resolution < 1 or capacity < 1 or not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("Resolution, capacity and timeout must be positive.")
    identity = corpus_identity(case)
    revision = str(identity["source_revision"])
    payload = source_payload(case)
    limits = phx.meshing.MeshingLimits(
        maximum_vertices=1
        if "maximum-vertices" in case.controls or "maximum-entities" in case.controls
        else capacity,
        maximum_edges=8 * capacity,
        maximum_faces=8 * capacity,
        maximum_cells=8 * capacity,
        maximum_connectivity_entries=64 * capacity,
        maximum_data_bytes=1 if "maximum-storage" in case.controls else 1024 * capacity,
        maximum_work_units=1 if "maximum-work" in case.controls else 32 * capacity,
        maximum_cavity_cells=1 if "maximum-cavity" in case.controls else capacity,
        maximum_geometry_queries=1
        if "maximum-queries" in case.controls
        else 64 * capacity,
        maximum_scratch_bytes=4096 * capacity,
        maximum_wall_seconds=timeout,
    )
    if case.dimension == 2:
        region = phx.geometry.PlanarMeshRegion(
            np.asarray(payload["vertices"], dtype=np.float64),
            payload["loops"],
            feature_id="plate",
        )
        embedded = np.asarray(payload["embedded"], dtype=np.float64)
        crack = (
            None
            if embedded.size == 0
            else phx.geometry.SegmentMesh(
                jnp.asarray(embedded, dtype=jnp.float64),
                jnp.asarray(((0, 1),), dtype=jnp.int32),
                source_id="embedded-feature",
            )
        )
        source = phx.meshing.NativePlanarSource(region, revision, embedded=crack)
        scope = phx.meshing.MeshingScope(
            "plate",
            "stale" if case.name == "planar-stale-revision" else revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            "plate-region",
            np.asarray((0,), dtype=np.int64),
        )
        size = 0.8 / resolution
        specification = phx.meshing.SurfaceMeshingSpec(
            phx.meshing.CellMeshingTarget(
                2, 2, phx.meshing.CellFamilyPolicy(required=(case.family,))
            ),
            scope,
            planar_embedding=phx.geometry.PlanarEmbedding(
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
            ),
            size_controls=(
                phx.meshing.UniformSizeControl(
                    scope,
                    size,
                    maximum_size=1.5 * size,
                    strength=phx.meshing.SizeControlStrength.HARD,
                ),
            ),
            size_compliance=phx.meshing.SizeCompliancePolicy(
                relative_tolerance=0.5, target_statistics=("p50",)
            ),
            quality_target=phx.meshing.MeshQualityTarget(minimum_angle=np.radians(25.0)),
            limits=limits,
        )
        options = phx.meshing.NativeMeshingOptions("planar_constrained_delaunay")
    else:
        complex_ = phx.meshing.PiecewiseLinearComplex(
            payload["vertices"],
            payload["polygons"],
            payload["facets"],
            payload["facet_regions"],
            payload["region_names"],
            segments=payload["segments"] if payload["segments"] else None,
            boundary=payload.get("boundary", "conforming"),
        )
        source = phx.meshing.NativePlcSource(complex_, case.name, revision)
        scope = phx.meshing.MeshingScope(
            case.name,
            "stale" if "stale-source-revision" in case.controls else revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            "solid-facets",
            np.arange(complex_.facet_count, dtype=np.int64),
        )
        size = (
            0.25
            if case.name == "plc-sliver-rich"
            else 0.3
            if case.name == "plc-fixed-boundary"
            else 4.0
            if "soft-size" in case.controls
            else 0.8 / resolution
        )
        specification = phx.meshing.VolumeMeshingSpec(
            phx.meshing.CellMeshingTarget(
                3, 3, phx.meshing.CellFamilyPolicy(required=(case.family,))
            ),
            scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=(
                phx.meshing.UniformSizeControl(
                    scope,
                    size,
                    maximum_size=1.5 * size,
                    strength=phx.meshing.SizeControlStrength.SOFT
                    if "soft-size" in case.controls
                    else phx.meshing.SizeControlStrength.HARD,
                ),
            ),
            size_compliance=phx.meshing.SizeCompliancePolicy(
                relative_tolerance=0.5, target_statistics=("p50",)
            ),
            region_controls=tuple(
                phx.meshing.RegionControl(
                    phx.meshing.MeshingScope(
                        source.source_id,
                        source.source_revision,
                        phx.meshing.MeshingEntityKind.GEOMETRY,
                        3,
                        "solid-regions",
                        np.asarray((index,), dtype=np.int64),
                    ),
                    name,
                    f"{name}-material",
                    phx.meshing.RegionRole.SOLID,
                )
                for index, name in enumerate(complex_.region_ids)
            ),
            patch_controls=tuple(
                phx.meshing.PatchControl(
                    "left-right-interface"
                    if case.name == "plc-material-cavity"
                    else f"interface-{facet}",
                    phx.meshing.MeshingScope(
                        source.source_id,
                        source.source_revision,
                        phx.meshing.MeshingEntityKind.GEOMETRY,
                        2,
                        "solid-facets",
                        np.asarray((facet,), dtype=np.int64),
                    ),
                    tuple(complex_.region_ids[index] for index in sorted(pair)),
                )
                for facet, pair in enumerate(payload["facet_regions"])
                if min(pair) >= 0
            ),
            limits=limits,
        )
        options = phx.meshing.NativeMeshingOptions("plc_tetrahedral")
        if case.name == "plc-sliver-rich":
            from phydrax.meshing._volume_generation import NativeVolumeSchedule

            options = phx.meshing.NativeMeshingOptions(
                "plc_tetrahedral",
                volume_schedule=NativeVolumeSchedule(
                    minimum_dihedral_degrees=25.0,
                    improvement_passes=1,
                    exudation_passes=4,
                    exudation_weight_fraction=0.25,
                ),
            )
    return source, specification, options


def prepare_sphere_case(
    resolution: int,
    capacity: int,
    timeout: float,
) -> tuple[Any, Any, Any]:
    """Reuse the owning smooth seam/pole fixture with a frozen hard request."""
    from examples.native_surface_meshing import sphere

    if resolution < 1 or capacity < 1 or not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("Resolution, capacity and timeout must be positive.")
    domain = sphere()
    domain = phx.geometry.MeshingDomain(
        domain.patches,
        domain.curves,
        domain.corner_points.shape[0],
        source_id=domain.source_id,
        source_revision=str(corpus_identity(SURFACE_CASE)["source_revision"]),
        regions=domain.regions,
        tolerance=domain.tolerance,
        accuracy=domain.accuracy,
        source_indices=domain.source_indices,
        source_occurrences=domain.source_occurrences,
        source_kinds=domain.source_kinds,
        authority_id=domain.authority_id,
    )
    source = phx.meshing.NativeSurfaceSource(domain)
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.source_indices[2], dtype=np.int64),
    )
    size = 1.6 / resolution
    limits = phx.meshing.MeshingLimits(
        maximum_vertices=capacity,
        maximum_cells=8 * capacity,
        maximum_edges=8 * capacity,
        maximum_faces=8 * capacity,
        maximum_connectivity_entries=64 * capacity,
        maximum_data_bytes=1024 * capacity,
        maximum_work_units=32 * capacity,
        maximum_cavity_cells=capacity,
        maximum_geometry_queries=64 * capacity,
        maximum_scratch_bytes=4096 * capacity,
        maximum_wall_seconds=timeout,
    )
    request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                size,
                maximum_size=1.5 * size,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        protected_features=(
            phx.meshing.ProtectedFeature(
                scope,
                phx.meshing.FeatureKind.SURFACE,
                maximum_deviation=0.25 * size * size,
            ),
        ),
        limits=limits,
    )
    return source, request, phx.meshing.NativeMeshingOptions("parametric_surface")


def frozen_request_record(
    source: Any, specification: Any, options: Any
) -> dict[str, object]:
    """Serialize the exact immutable source/request controls used by a campaign."""
    quality = specification.quality_target
    family = specification.target.cell_families
    controls = [
        {
            "control_id": control.control_id,
            "target_size": control.target_size,
            "minimum_size": control.minimum_size,
            "maximum_size": control.maximum_size,
            "strength": control.strength.value,
            "scope_id": control.scope.scope_id,
        }
        for control in specification.size_controls
    ]
    record = {
        "source_binding_id": source.binding_id,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "specification_id": specification.specification_id,
        "options_id": options.options_id,
        "native_route": options.route,
        "target": {
            "topological_dimension": specification.target.topological_dimension,
            "ambient_dimension": specification.target.ambient_dimension,
            "geometry_order": specification.target.geometry_order,
            "require_conforming": specification.target.require_conforming,
            "required_families": list(family.required),
            "preferred_families": list(family.preferred),
            "allowed_transitions": list(family.allowed_transitions),
            "allow_mixed": family.allow_mixed,
            "family_policy_id": family.policy_id,
        },
        "size_controls": controls,
        "limits": {
            "maximum_vertices": specification.limits.maximum_vertices,
            "maximum_edges": specification.limits.maximum_edges,
            "maximum_faces": specification.limits.maximum_faces,
            "maximum_cells": specification.limits.maximum_cells,
            "maximum_connectivity_entries": specification.limits.maximum_connectivity_entries,
            "maximum_data_bytes": specification.limits.maximum_data_bytes,
            "maximum_work_units": specification.limits.maximum_work_units,
            "maximum_cavity_cells": specification.limits.maximum_cavity_cells,
            "maximum_geometry_queries": specification.limits.maximum_geometry_queries,
            "maximum_scratch_bytes": specification.limits.maximum_scratch_bytes,
            "maximum_wall_seconds": specification.limits.maximum_wall_seconds,
            "limits_id": specification.limits.limits_id,
        },
        "quality": None
        if quality is None
        else {
            "target_id": quality.target_id,
            "minimum_angle": quality.minimum_angle,
            "hard": quality.hard,
        },
    }
    return {**record, "frozen_request_id": canonical_fingerprint(record)}


def _source_files_digest(root: Path, paths: Iterable[Path]) -> str:
    digest = sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def python_source_revision(package_root: Path) -> str:
    """Digest the Python package actually imported, not an assumed checkout."""
    return _source_files_digest(package_root, package_root.rglob("*.py"))


@lru_cache(maxsize=1)
def source_tree_revision() -> str:
    """Digest actual first-party build inputs, including uncommitted edits."""
    root = Path(__file__).resolve().parents[1]
    native = root.joinpath("native/meshcore")
    paths = sorted(
        (
            *root.joinpath("phydrax").rglob("*.py"),
            *native.joinpath("src").rglob("*.cpp"),
            *native.joinpath("src").rglob("*.hpp"),
            *native.joinpath("include").rglob("*.h"),
            *native.joinpath("python").rglob("*.py"),
            native.joinpath("CMakeLists.txt"),
            native.joinpath("pyproject.toml"),
            root.joinpath("pyproject.toml"),
            Path(__file__),
            root.joinpath("tools/meshing_qualification.py"),
            root.joinpath("examples/native_surface_meshing.py"),
            root.joinpath("examples/native_polyhedral_meshing.py"),
            root.joinpath("tools/meshing_benchmarks.py"),
        )
    )
    package_root = Path(phx.__file__).resolve().parent
    if package_root != root.joinpath("phydrax").resolve() or not native.is_dir():
        available = [path for path in paths if path.is_file()]
        available.extend(root.joinpath("tools").rglob("*.py"))
        available.extend(root.joinpath("examples").rglob("*.py"))
        unavailable = [
            path.relative_to(root).as_posix() for path in paths if not path.is_file()
        ]
        return canonical_fingerprint(
            {
                "imported_python_source": python_source_revision(package_root),
                "driver_and_available_native_inputs": _source_files_digest(
                    root, available
                ),
                "unavailable_build_inputs": unavailable,
                "native_source_directory_present": native.is_dir(),
            }
        )
    return _source_files_digest(root, paths)


POLYHEDRAL_CASES = (
    MeshingCase(
        "polyhedral-material-L",
        "polyhedral-system",
        3,
        "polyhedron",
        "positive",
        ("nonconvex", "material-interface", "prescribed-sites", "soft-size"),
        (("left", 2.0), ("right", 1.0)),
        "Three analytic unit voxels; PHYDRA original fixture",
    ),
    MeshingCase(
        "polyhedral-stretched-L",
        "polyhedral-system",
        3,
        "polyhedron",
        "positive",
        ("nonconvex", "material-interface", "prescribed-sites", "soft-size"),
        (("left", 4.0), ("right", 2.0)),
        "Same three voxels stretched twofold along x",
    ),
    MeshingCase(
        "polyhedral-periodic-cube",
        "polyhedral-system",
        3,
        "polyhedron",
        "positive",
        ("periodic-quotient", "prescribed-sites", "soft-size", "zero-periodic-tolerance"),
        (("material-0", 1.0),),
        "Original three-axis periodic cube and two prescribed zero-weight sites",
    ),
)
POLYHEDRAL_CASE_NAMES = tuple(case.name for case in POLYHEDRAL_CASES)


def polyhedral_case(name: str) -> MeshingCase:
    for case in POLYHEDRAL_CASES:
        if case.name == name:
            return case
    raise ValueError(f"Unknown polyhedral corpus case: {name}")


def _polyhedral_complex(name: str) -> Any:
    from examples.native_polyhedral_meshing import material_domain

    case = polyhedral_case(name)
    if name == "polyhedral-periodic-cube":
        loops = (
            (0, 2, 3, 1),
            (4, 5, 7, 6),
            (0, 1, 5, 4),
            (2, 6, 7, 3),
            (0, 4, 6, 2),
            (1, 3, 7, 5),
        )
        return phx.meshing.PiecewiseLinearComplex(
            _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            loops,
            np.arange(6),
            [(-1, 0)] * 6,
            ("material-0",),
        )
    original = material_domain()
    scale = 2.0 if case.name == "polyhedral-stretched-L" else 1.0
    offsets = np.asarray(original.polygon_offsets)
    loops = [
        original.polygon_vertices[a:b]
        for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    return phx.meshing.PiecewiseLinearComplex(
        np.asarray(original.vertices) * np.asarray((scale, 1.0, 1.0), dtype=np.float64),
        loops,
        original.polygon_facets,
        original.facet_regions,
        original.region_ids,
        segments=original.segments,
        boundary=original.boundary,
    )


def polyhedral_identity(name: str) -> dict[str, object]:
    case = polyhedral_case(name)
    complex_ = _polyhedral_complex(name)
    artifact = {
        "kind": "analytic-generated-fixture",
        "locator": f"tools/_meshing_cases.py::{name}",
        "media_type": "application/vnd.phydrax.plc",
        "sha256": complex_.complex_id,
        "license": case.license_id,
    }
    if name == "polyhedral-periodic-cube":
        geometry, request, _ = _prepare_periodic_polyhedral_case(20000, 120.0)
        return {
            **case.to_record(),
            "source_id": geometry.source_id,
            "source_revision": geometry.source_revision,
            "source_digest": complex_.complex_id,
            "source_artifact": artifact,
            "input_id": geometry.binding_id,
            "sites": np.asarray(geometry.sites).tolist(),
            "weights": np.asarray(geometry.weights).tolist(),
            "periodic_constraints": [
                constraint.constraint_id for constraint in request.periodic_constraints
            ],
        }
    return {
        **case.to_record(),
        "source_revision": complex_.complex_id,
        "source_digest": complex_.complex_id,
        "source_artifact": artifact,
        "input_id": canonical_fingerprint(
            {
                "complex": complex_.complex_id,
                "coordinate_unit": "meter",
                "regions": case.expected_regions,
            }
        ),
    }


def prepare_polyhedral_case(
    name: str,
    site_count: int,
    capacity: int,
    timeout: float,
    *,
    source: bool = False,
) -> tuple[Any, Any, Any]:
    """Freeze exact material PLC, explicit dyadic sites and requested resources."""
    if site_count < 1 or capacity < 1 or not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("Site count, capacity and timeout must be positive.")
    if name == "polyhedral-periodic-cube":
        return _prepare_periodic_polyhedral_case(capacity, timeout)
    complex_ = _polyhedral_complex(name)
    scale = 2.0 if name == "polyhedral-stretched-L" else 1.0
    # Dyadic y slabs keep the fixture's bisectors exact under binary64;
    # nonbinary-site adversaries are a separate, still-unobserved corpus gate.
    denominator = 1 << (site_count - 1).bit_length()
    sites = np.column_stack(
        (
            np.full(site_count, 0.5 * scale, dtype=np.float64),
            (np.arange(site_count, dtype=np.float64) + 0.5) * (2.0 / denominator),
            np.full(site_count, 0.5, dtype=np.float64),
        )
    )
    revision = canonical_fingerprint(
        {
            "complex": complex_.complex_id,
            "sites": None if source else sites.tolist(),
            "coordinate_unit": "meter",
        }
    )
    geometry = (
        phx.meshing.NativePlcSource(complex_, name, revision)
        if source
        else phx.meshing.NativePolyhedralSource(complex_, name, revision, sites=sites)
    )
    scope = phx.meshing.MeshingScope(
        name,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "polyhedral-domain-facets",
        np.arange(complex_.facet_count, dtype=np.int64),
    )
    limits = phx.meshing.MeshingLimits(
        maximum_vertices=capacity,
        maximum_edges=8 * capacity,
        maximum_faces=8 * capacity,
        maximum_cells=8 * capacity,
        maximum_connectivity_entries=64 * capacity,
        maximum_data_bytes=1024 * capacity,
        maximum_work_units=32 * capacity,
        maximum_cavity_cells=capacity,
        maximum_geometry_queries=64 * capacity,
        maximum_scratch_bytes=4096 * capacity,
        maximum_wall_seconds=timeout,
    )
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("polyhedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 10.0 * scale, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        region_controls=tuple(
            phx.meshing.RegionControl(
                phx.meshing.MeshingScope(
                    name,
                    revision,
                    phx.meshing.MeshingEntityKind.GEOMETRY,
                    3,
                    "polyhedral-domain-regions",
                    np.asarray((index,), dtype=np.int64),
                ),
                region,
                f"{region}-material",
                phx.meshing.RegionRole.SOLID,
            )
            for index, region in enumerate(complex_.region_ids)
        ),
        region_seeds=tuple(
            phx.meshing.RegionSeed(
                np.asarray(
                    ((0.5 if index == 0 else 1.5) * scale, 0.5, 0.5), dtype=np.float64
                ),
                region,
                f"{region}-material",
                phx.meshing.RegionRole.SOLID,
            )
            for index, region in enumerate(complex_.region_ids)
        ),
        limits=limits,
    )
    return geometry, request, phx.meshing.NativeMeshingOptions("plc_restricted_power")


def _prepare_periodic_polyhedral_case(
    capacity: int, timeout: float
) -> tuple[Any, Any, Any]:
    """Retain the original cube source SCI, sites, weights and exact periods."""
    complex_ = _polyhedral_complex("polyhedral-periodic-cube")
    geometry = phx.meshing.NativePolyhedralSource(
        complex_,
        "polyhedral-domain",
        "r1",
        sites=np.asarray(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)), dtype=np.float64),
        weights=np.asarray((0.0, 0.0), dtype=np.float64),
    )
    constraints = []
    for axis, source_facet, target_facet in ((0, 4, 5), (1, 2, 3), (2, 0, 1)):
        source_scope = phx.meshing.MeshingScope(
            "polyhedral-domain",
            "r1",
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            f"source-{axis}",
            [source_facet],
        )
        target_scope = phx.meshing.MeshingScope(
            "polyhedral-domain",
            "r1",
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            f"target-{axis}",
            [target_facet],
        )
        matrix = np.eye(4)
        matrix[axis, 3] = 1.0
        constraints.append(
            phx.meshing.PeriodicConstraint(
                source_scope,
                target_scope,
                matrix,
                tolerance=0.0,
                source_entity_ids=np.asarray((source_facet,), dtype=np.int64),
                orientations=np.asarray((-1,), dtype=np.int64),
            )
        )
    scope = phx.meshing.MeshingScope(
        "polyhedral-domain",
        "r1",
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "polyhedral-facets",
        np.arange(6, dtype=np.int64),
    )
    limits = phx.meshing.MeshingLimits(
        maximum_vertices=capacity,
        maximum_edges=8 * capacity,
        maximum_faces=8 * capacity,
        maximum_cells=8 * capacity,
        maximum_connectivity_entries=64 * capacity,
        maximum_data_bytes=1024 * capacity,
        maximum_work_units=32 * capacity,
        maximum_cavity_cells=capacity,
        maximum_geometry_queries=64 * capacity,
        maximum_scratch_bytes=4096 * capacity,
        maximum_wall_seconds=timeout,
    )
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("polyhedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 10.0, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        periodic_constraints=tuple(constraints),
        limits=limits,
    )
    return geometry, request, phx.meshing.NativeMeshingOptions("plc_restricted_power")


def _reject_nonfinite_json(value: str, /) -> NoReturn:
    raise ValueError(f"Non-finite JSON value {value!r} is not permitted.")


def _unique_json_object(pairs: list[tuple[str, object]], /) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Corpus JSON contains duplicate field {name!r}.")
        result[name] = value
    return result


def _strict_json_object(raw: str | bytes, name: str, /) -> dict[str, object]:
    value = json.loads(
        raw,
        parse_constant=_reject_nonfinite_json,
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object.")
    return value


_ROOT = Path(__file__).resolve().parents[1]
CORPUS_REGISTRATION_PATH = _ROOT / "tests/data/meshing/corpus.json"
CAD_FIXTURE_PATH = _ROOT / "tests/data/cad/native_authored_boxes.json"
CURVED_PERIODIC_SOURCE_REGISTRATION_PATH = (
    _ROOT / "tests/data/meshing/curved_periodic_sources.json"
)


@lru_cache(maxsize=1)
def corpus_registration_record() -> dict[str, object]:
    """Canonical deterministic registration of every bounded source fixture."""
    cases = [
        corpus_identity(case)
        for case in sorted((*CORPUS, SURFACE_CASE), key=lambda item: item.name)
    ]
    cases.extend(polyhedral_identity(name) for name in sorted(POLYHEDRAL_CASE_NAMES))
    curved = load_curved_periodic_source_registration()
    body: dict[str, object] = {
        "kind": "native-meshing-corpus",
        "coordinate_unit": "meter",
        "mandatory_workflows": list(MANDATORY_WORKFLOWS),
        "hybrid_layer_sources": curved["sources"],
        "cases": cases,
    }
    return {**body, "registration_id": canonical_fingerprint(body)}


def load_corpus_registration(
    path: Path = CORPUS_REGISTRATION_PATH,
) -> dict[str, object]:
    """Load the reviewed corpus record and reject any drift from registration."""
    value = _strict_json_object(
        path.read_text(encoding="utf-8"), "Meshing corpus registration"
    )
    if value != corpus_registration_record():
        raise ValueError(f"Meshing corpus registration is stale or malformed: {path}")
    return value


@lru_cache(maxsize=1)
def load_curved_periodic_source_registration(
    path: Path = CURVED_PERIODIC_SOURCE_REGISTRATION_PATH,
) -> dict[str, object]:
    """Load the reviewed positive revision and immutable negative trace source."""
    raw = path.read_bytes()
    value = _strict_json_object(raw, "Curved periodic layer source registration")
    record_id = value.get("registration_id")
    body = {name: field for name, field in value.items() if name != "registration_id"}
    sources = body.get("sources")
    if (
        set(body) != {"kind", "coordinate_unit", "period", "sources"}
        or body.get("kind") != "curved-periodic-layer-sources"
        or body.get("coordinate_unit") != "meter"
        or body.get("period") != [1.0, 0.0, 0.0]
        or record_id != canonical_fingerprint(body)
        or not isinstance(sources, list)
        or len(sources) != 2
    ):
        raise ValueError(
            f"Curved periodic source registration is stale or malformed: {path}"
        )
    roles: set[str] = set()
    for source in sources:
        if (
            not isinstance(source, dict)
            or set(source)
            != {
                "role",
                "profile",
                "source_id",
                "source_revision",
                "source_digest",
                "source_artifact",
                "profile_control_points",
                "upper_construction",
                "orbit_ids",
                "periodic_trace",
            }
            or not isinstance(source["role"], str)
            or source["role"] in roles
            or not isinstance(source["source_id"], str)
            or not isinstance(source["source_revision"], str)
            or not isinstance(source["source_digest"], str)
            or len(source["source_revision"]) != 64
            or len(source["source_digest"]) != 64
            or not isinstance(source["source_artifact"], dict)
            or not isinstance(source["profile_control_points"], list)
            or not isinstance(source["orbit_ids"], list)
            or not isinstance(source["periodic_trace"], dict)
        ):
            raise ValueError(
                "Curved periodic source records require complete unique identities."
            )
        roles.add(source["role"])
    if roles != {"positive", "negative-refusal"}:
        raise ValueError(
            "Curved periodic source records require one positive and one refusal."
        )
    return {**value, "file_sha256": sha256(raw).hexdigest()}


def load_cad_fixture(path: Path = CAD_FIXTURE_PATH) -> dict[str, object]:
    """Load one licensed authored CAD source descriptor with content identity."""
    raw = path.read_bytes()
    value = _strict_json_object(raw, "CAD corpus fixture")
    record_id = value.get("record_id")
    body = {name: field for name, field in value.items() if name != "record_id"}
    if (
        set(body)
        != {
            "kind",
            "title",
            "author",
            "license",
            "provenance",
            "coordinate_unit",
            "operands",
            "independent_reference",
        }
        or record_id != canonical_fingerprint(body)
        or body.get("kind") != "native-authored-cad-boxes"
        or body.get("license") != "LicenseRef-PHYDRA-Proprietary"
        or body.get("coordinate_unit") != "meter"
    ):
        raise ValueError(
            f"CAD corpus fixture identity, license, or units are invalid: {path}"
        )
    provenance = body.get("provenance")
    reference = body.get("independent_reference")
    operands = body.get("operands")
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {"method", "origin", "redistribution"}
        or not all(isinstance(value, str) and value for value in provenance.values())
        or not isinstance(reference, dict)
        or set(reference)
        != {
            "intersection_bounds",
            "intersection_volume",
            "selected_top_area",
            "method",
        }
        or not isinstance(operands, list)
        or len(operands) != 2
    ):
        raise ValueError(
            "CAD corpus fixture provenance, reference, or operands are malformed."
        )
    for operand in operands:
        if (
            not isinstance(operand, dict)
            or set(operand) != {"name", "lower", "upper"}
            or not isinstance(operand["name"], str)
            or not isinstance(operand["lower"], list)
            or not isinstance(operand["upper"], list)
            or len(operand["lower"]) != 3
            or len(operand["upper"]) != 3
            or any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not np.isfinite(number)
                for number in (*operand["lower"], *operand["upper"])
            )
        ):
            raise ValueError(
                "CAD corpus fixture operands require finite named 3D bounds."
            )
    return {**value, "file_sha256": sha256(raw).hexdigest()}
