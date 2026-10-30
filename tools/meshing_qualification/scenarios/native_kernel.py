from __future__ import annotations


import argparse


import importlib.metadata


import json


import os


import platform


import shutil


import sys


from collections.abc import Callable


from pathlib import Path


from tempfile import TemporaryDirectory


from time import perf_counter


from typing import Any


import jax


import jax.numpy as jnp


import numpy as np


import phydrax as phx


from phydrax._fingerprint import canonical_fingerprint


from phydrax._meshcore import meshcore_available, meshcore_identity


from phydrax.geometry.brep import BRepTessellationPolicy
from phydrax.interchange import (
    CadImportPolicy,
    persist_occt_shape,
    read_brep_text,
)


def _contract() -> phx.SpatialCoordinateContract:
    return phx.SpatialCoordinateContract(phx.units.MILLIMETER)


def _measured(call: Callable[[], object], /) -> tuple[object, dict[str, float]]:
    """Run one stage; report wall seconds and the process peak resident bytes."""
    if os.name != "posix":
        raise RuntimeError("Meshing resource qualification requires a POSIX host.")
    import resource

    started = perf_counter()
    value = call()
    elapsed = perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports ru_maxrss in bytes, Linux in KiB.
    scale = 1 if sys.platform == "darwin" else 1024
    return value, {"wall_seconds": elapsed, "peak_resident_bytes": peak * scale}


def _quality_summary(report: phx.meshing.CellQualityReport, /) -> dict[str, object]:
    return {
        "minimum_mean_ratio": report.minimum_mean_ratio,
        "maximum_aspect_ratio": report.maximum_aspect_ratio,
        "minimum_scaled_jacobian": report.minimum_scaled_jacobian,
        "minimum_angle": report.minimum_angle,
        "maximum_angle": report.maximum_angle,
        "sampled_invalid_count": report.sampled_invalid_count,
    }


def _result_identity(result: phx.meshing.CellMeshingResult, /) -> dict[str, object]:
    return {
        "result_id": result.result_id,
        "topology_id": result.mesh.topology_id,
        "vertices": result.mesh.coordinates.shape[0],
        "cells": sum(block.cell_count for block in result.mesh.blocks),
        "audit_passed": result.audit.passed,
    }


def _missing_dependency(dependency: str, detail: str, /) -> dict[str, object]:
    """Explicit record of a scenario that cannot run in this environment."""
    return {
        "status": "missing-dependency",
        "dependency": dependency,
        "detail": detail,
        "runtime": _runtime_identity(),
    }


_PREDICATE_ARITY = {"orient2d": 3, "orient3d": 4, "incircle": 4, "insphere": 5}


_PREDICATE_OFFSET = 2.0**40


def _predicate_integers(arrays: tuple[np.ndarray, ...], /) -> tuple[np.ndarray, ...]:
    """Float64 coordinates as exact Python integers on one common binary scale."""
    mantissas, exponents = [], []
    for array in arrays:
        fraction, exponent = np.frexp(array)
        mantissas.append(np.ldexp(fraction, 53).astype(np.int64))
        exponents.append(exponent.astype(np.int64) - 53)
    scale = min(
        np.min(np.where(mantissa != 0, exponent, np.iinfo(np.int64).max))
        for mantissa, exponent in zip(mantissas, exponents, strict=True)
    )
    return tuple(
        np.left_shift(
            mantissa.astype(np.object_),
            np.where(mantissa != 0, exponent - scale, 0).astype(np.object_),
        )
        for mantissa, exponent in zip(mantissas, exponents, strict=True)
    )


def _integer_determinant(rows: list[list[np.ndarray]], /) -> np.ndarray:
    """Laplace expansion of a small matrix whose entries are integer object arrays."""
    if len(rows) == 1:
        return rows[0][0]
    total = np.zeros(rows[0][0].shape, dtype=np.object_)
    for column, entry in enumerate(rows[0]):
        minor = [row[:column] + row[column + 1 :] for row in rows[1:]]
        term = entry * _integer_determinant(minor)
        total = total + term if column % 2 == 0 else total - term
    return total


def _reference_predicate_signs(
    name: str, arrays: tuple[np.ndarray, ...], /
) -> np.ndarray:
    """Predicate signs in the phydrax convention from exact integer arithmetic."""
    points = _predicate_integers(arrays)
    width = arrays[0].shape[-1]
    match name:
        case "orient2d" | "orient3d":
            # det[b - a, c - a(, d - a)]
            determinant = _integer_determinant(
                [
                    [point[:, axis] - points[0][:, axis] for axis in range(width)]
                    for point in points[1:]
                ]
            )
        case "incircle" | "insphere":
            # Lifted rows (p - q, |p - q|^2) against the query q; insphere is
            # positive inside the sphere of a positively oriented tetrahedron.
            rows = [
                [point[:, axis] - points[-1][:, axis] for axis in range(width)]
                for point in points[:-1]
            ]
            determinant = _integer_determinant(
                [row + [sum(value * value for value in row)] for row in rows]
            )
            if name == "insphere":
                determinant = -determinant
        case _:
            raise ValueError(f"Unknown predicate {name!r}.")
    return (determinant > 0).astype(np.int8) - (determinant < 0).astype(np.int8)


def _ulp_steps(values: np.ndarray, steps: np.ndarray, /) -> np.ndarray:
    """Move each coordinate by ``steps`` in {-1, 0, 1} units in the last place."""
    return np.where(
        steps > 0,
        np.nextafter(values, np.inf),
        np.where(steps < 0, np.nextafter(values, -np.inf), values),
    )


def _integer_shell(radius: int, dimension: int, /) -> np.ndarray:
    """Integer points on the circle/sphere of an integer radius, by polar angle."""
    axis = np.arange(-radius, radius + 1)
    grid = np.stack(np.meshgrid(*(axis,) * dimension, indexing="ij"), axis=-1)
    grid = grid.reshape((-1, dimension))
    shell = grid[np.sum(grid * grid, axis=1) == radius * radius].astype(np.float64)
    return shell[np.argsort(np.arctan2(shell[:, 1], shell[:, 0]), kind="stable")]


def _shell_rows(
    shell: np.ndarray,
    simplices: np.ndarray,
    offset: float,
    rng: np.random.Generator,
    /,
) -> tuple[np.ndarray, ...]:
    """Every shell point queried against every simplex, queries moved by one ulp.

    The shell is translated by ``offset``; every coordinate is then nonzero, so a
    one-ulp move never creates a subnormal outside the meshcore exact domain.
    """
    count = shell.shape[0]
    corners = np.repeat(simplices, count, axis=0)
    query = shell[np.tile(np.arange(count), simplices.shape[0])] + offset
    query = _ulp_steps(query, rng.integers(-1, 2, size=query.shape))
    return tuple(
        shell[corners[:, index]] + offset for index in range(simplices.shape[1])
    ) + (query,)


def _predicate_families(
    rng: np.random.Generator, /
) -> dict[str, dict[str, tuple[np.ndarray, ...]]]:
    """Random rows plus adversarial near-degenerate rows for every predicate."""

    def random(arity: int, width: int) -> tuple[np.ndarray, ...]:
        return tuple(rng.standard_normal((2048, width)) for _ in range(arity))

    # Kettner et al.: points near the line y = x and the plane z = y, moved by
    # single ulps of 0.5.
    steps = np.arange(16) * 2.0**-53
    near_line = 0.5 + np.stack(np.meshgrid(steps, steps, indexing="ij"), axis=-1)
    near_line = near_line.reshape((-1, 2))
    steps = np.arange(8) * 2.0**-53
    near_plane = 0.5 + np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1)
    near_plane = near_plane.reshape((-1, 3))
    # Exactly collinear/coplanar integer points far from the origin; the last point
    # moves by at most one ulp of 2**40.
    line = rng.integers(-64, 65, size=(3, 512))
    far_line = _PREDICATE_OFFSET + np.stack((line, 2 * line + 1), axis=-1).astype(
        np.float64
    )
    far_line[2, :, 1] = _ulp_steps(far_line[2, :, 1], rng.integers(-1, 2, size=512))
    plane = rng.integers(-64, 65, size=(4, 512, 2))
    far_plane = _PREDICATE_OFFSET + np.concatenate(
        (plane, plane[..., :1] + 2 * plane[..., 1:] + 3), axis=-1
    ).astype(np.float64)
    far_plane[3, :, 2] = _ulp_steps(far_plane[3, :, 2], rng.integers(-1, 2, size=512))
    # Cocircular lattice: every counterclockwise triple of the 12 integer points of
    # the circle of radius 5. Cospherical lattice: positively oriented tetrahedra
    # of the 30 integer points of the sphere of radius 3.
    circle = _integer_shell(5, 2)
    index = np.arange(circle.shape[0])
    first, second, third = np.meshgrid(index, index, index, indexing="ij")
    ordered = (first < second) & (second < third)
    triples = np.stack((first[ordered], second[ordered], third[ordered]), axis=1)
    sphere = _integer_shell(3, 3)
    tetrahedra = np.argsort(rng.random((128, sphere.shape[0])), axis=1)[:, :4]
    orientation = _reference_predicate_signs(
        "orient3d", tuple(sphere[tetrahedra[:, index]] for index in range(4))
    )
    tetrahedra = np.where(
        (orientation < 0)[:, None], tetrahedra[:, (1, 0, 2, 3)], tetrahedra
    )[orientation != 0]
    return {
        "orient2d": {
            "random": random(3, 2),
            "collinear_ulp": (
                near_line,
                np.broadcast_to(np.asarray((12.0, 12.0)), near_line.shape),
                np.broadcast_to(np.asarray((24.0, 24.0)), near_line.shape),
            ),
            "large_offset": tuple(far_line),
        },
        "orient3d": {
            "random": random(4, 3),
            "coplanar_ulp": (
                np.broadcast_to(np.asarray((1.0, 12.0, 12.0)), near_plane.shape),
                np.broadcast_to(np.asarray((24.0, 24.0, 24.0)), near_plane.shape),
                np.broadcast_to(np.asarray((-7.0, 3.0, 3.0)), near_plane.shape),
                near_plane,
            ),
            "large_offset": tuple(far_plane),
        },
        "incircle": {
            "random": random(4, 2),
            "cocircular_lattice": _shell_rows(circle, triples, 0.5, rng),
            "large_offset": _shell_rows(circle, triples, _PREDICATE_OFFSET, rng),
        },
        "insphere": {
            "random": random(5, 3),
            "cospherical_lattice": _shell_rows(sphere, tetrahedra, 0.5, rng),
            "large_offset": _shell_rows(sphere, tetrahedra, _PREDICATE_OFFSET, rng),
        },
    }


def _predicate_family_record(
    routes: dict[str, tuple[np.ndarray, np.ndarray]],
    reference: np.ndarray,
    perturbed: np.ndarray,
    /,
) -> dict[str, object]:
    """Certainty, correctness, agreement, and degeneracy counts of one family."""
    rows = reference.size
    exact = routes["exact"][0]
    record: dict[str, object] = {"rows": rows}
    for route, (signs, certain) in routes.items():
        record[route] = {
            "certain": int(np.count_nonzero(certain)),
            "uncertain_fraction": 1.0 - int(np.count_nonzero(certain)) / rows,
            "certified_zeros": int(np.count_nonzero(certain & (signs == 0))),
            "wrong_certain_signs": int(np.count_nonzero(certain & (signs != reference))),
        }
    pairs = (("filtered", "device"), ("filtered", "exact"), ("device", "exact"))
    record["certain_disagreements"] = {
        f"{first}_{second}": int(
            np.count_nonzero(
                routes[first][1]
                & routes[second][1]
                & (routes[first][0] != routes[second][0])
            )
        )
        for first, second in pairs
    }
    record["exact_signs"] = {
        "negative": int(np.count_nonzero(exact == -1)),
        "zero": int(np.count_nonzero(exact == 0)),
        "positive": int(np.count_nonzero(exact == 1)),
    }
    record["reference_zeros"] = int(np.count_nonzero(reference == 0))
    record["symbolic"] = {
        "zeros": int(np.count_nonzero(perturbed == 0)),
        "changed_strict_signs": int(
            np.count_nonzero((exact != 0) & (perturbed != exact))
        ),
    }
    return record


def _qualify_predicate(
    name: str, families: dict[str, tuple[np.ndarray, ...]], /
) -> tuple[dict[str, object], dict[str, object], np.ndarray, int]:
    """Evaluate one predicate in every mode on all families against the reference."""
    from phydrax._meshcore import (
        symbolic_incircle,
        symbolic_insphere,
        symbolic_orient2d,
        symbolic_orient3d,
    )

    function, symbolic = {
        "orient2d": (phx.geometry.orient2d, symbolic_orient2d),
        "orient3d": (phx.geometry.orient3d, symbolic_orient3d),
        "incircle": (phx.geometry.incircle, symbolic_incircle),
        "insphere": (phx.geometry.insphere, symbolic_insphere),
    }[name]
    mode = phx.geometry.PredicateMode
    arity = _PREDICATE_ARITY[name]
    arrays = tuple(
        np.concatenate([rows[index] for rows in families.values()])
        for index in range(arity)
    )
    bounds = np.cumsum([0, *(rows[0].shape[0] for rows in families.values())])
    filtered, filtered_stage = _measured(lambda: function(*arrays, mode=mode.FILTERED))
    device_entry = jax.jit(lambda *values: function(*values, mode=mode.FILTERED_DEVICE))
    device_arrays = tuple(jnp.asarray(array) for array in arrays)
    _, device_cold = _measured(
        lambda: jax.block_until_ready(device_entry(*device_arrays))
    )
    device, device_warm = _measured(
        lambda: jax.block_until_ready(device_entry(*device_arrays))
    )
    exact, exact_stage = _measured(lambda: function(*arrays, mode=mode.EXACT))
    reference, reference_stage = _measured(
        lambda: np.concatenate(
            [_reference_predicate_signs(name, rows) for rows in families.values()]
        )
    )
    swapped = function(arrays[1], arrays[0], *arrays[2:], mode=mode.EXACT).signs
    # Distinct ids: the index-ordered perturbation used inside the triangulations.
    # ty: ignore[too-many-positional-arguments]
    perturbed = symbolic(*arrays, np.arange(arity, dtype=np.int64))
    routes = {
        # ty: ignore[unresolved-attribute]
        "filtered": (filtered.signs, filtered.certain),
        # ty: ignore[unresolved-attribute]
        "device": (np.asarray(device.signs), np.asarray(device.certain)),
        # ty: ignore[unresolved-attribute]
        "exact": (exact.signs, exact.certain),
    }
    records = {
        family: _predicate_family_record(
            {
                route: (signs[start:stop], certain[start:stop])
                for route, (signs, certain) in routes.items()
            },
            # ty: ignore[not-subscriptable]
            reference[start:stop],
            perturbed[start:stop],
        )
        for family, start, stop in zip(families, bounds[:-1], bounds[1:], strict=True)
    }
    resources = {
        "filtered": filtered_stage,
        "device_cold": device_cold,
        "device_warm": device_warm,
        "exact": exact_stage,
        # ty: ignore[unresolved-attribute]
        "exact_native_rows": int(np.count_nonzero(~filtered.certain)),
        "integer_reference": reference_stage,
    }
    # ty: ignore[invalid-return-type, unresolved-attribute]
    return records, resources, exact.signs, int(np.count_nonzero(swapped != -exact.signs))


def _predicate_failures(name: str, family: str, record: dict, /) -> list[str]:
    """Failed checks of one family record (empty when the family qualifies)."""
    routes = ("filtered", "device", "exact")
    failures = [
        f"{route} certified {record[route]['wrong_certain_signs']} wrong signs"
        for route in routes
        if record[route]["wrong_certain_signs"]
    ]
    failures += [
        f"{pair} disagree on {count} certain rows"
        for pair, count in record["certain_disagreements"].items()
        if count
    ]
    if record["exact"]["certain"] != record["rows"]:
        failures.append("EXACT left uncertain rows")
    if record["exact_signs"]["zero"] != record["reference_zeros"]:
        failures.append("EXACT zeros differ from the exact degeneracies")
    if record["symbolic"]["zeros"] or record["symbolic"]["changed_strict_signs"]:
        failures.append("symbolic perturbation left ties or changed a strict sign")
    if family == "random":
        if min(record["filtered"]["certain"], record["device"]["certain"]) < (
            0.9 * record["rows"]
        ):
            failures.append("filters left more than 10% of random rows uncertain")
    elif min(record["exact_signs"].values()) == 0:
        failures.append("adversarial rows miss an exact zero or a strict sign")
    return [f"{name}/{family}: {failure}" for failure in failures]


def qualify_predicates() -> dict[str, object]:
    if not meshcore_available():
        return _missing_dependency(
            "meshcore",
            "EXACT predicates and the symbolic tie-breaking require the native "
            "meshcore library; install phydrax[meshcore] or set "
            "PHYDRAX_MESHCORE_LIBRARY to the shared library path.",
        )
    from phydrax._meshcore import meshcore_exact_domain

    families = _predicate_families(np.random.default_rng(20260926))
    records, resources, exact_signs, antisymmetry = {}, {}, {}, {}
    for name, rows in families.items():
        records[name], resources[name], exact_signs[name], antisymmetry[name] = (
            _qualify_predicate(name, rows)
        )
    failures = [
        failure
        for name, per_family in records.items()
        for family, record in per_family.items()
        for failure in _predicate_failures(name, family, record)
    ]
    failures += [
        f"{name}: {count} exact signs are not antisymmetric under argument exchange"
        for name, count in antisymmetry.items()
        if count
    ]
    if failures:
        raise RuntimeError("Predicate qualification failed: " + "; ".join(failures))
    flat = [record for per_family in records.values() for record in per_family.values()]
    rows = sum(record["rows"] for record in flat)

    def total(route: str, key: str) -> int:
        return sum(record[route][key] for record in flat)

    return {
        "status": "passed",
        "source": {
            "inputs_id": canonical_fingerprint(families),
            "rows": {
                name: {family: record["rows"] for family, record in per_family.items()}
                for name, per_family in records.items()
            },
        },
        "target": {"exact_signs_id": canonical_fingerprint(exact_signs), "rows": rows},
        "runtime": {
            **_runtime_identity(),
            "meshcore_exact_exponent_range": list(meshcore_exact_domain()),
        },
        "quality": {
            "before": {
                "filtered_certain_fraction": total("filtered", "certain") / rows,
                "device_certain_fraction": total("device", "certain") / rows,
            },
            "after": {
                "exact_certain_fraction": total("exact", "certain") / rows,
                "exact_wrong_signs": total("exact", "wrong_certain_signs"),
            },
        },
        "conservation": {
            "antisymmetry_violations": antisymmetry,
            "wrong_certain_signs": {
                route: total(route, "wrong_certain_signs")
                for route in ("filtered", "device", "exact")
            },
            "certain_disagreements": sum(
                sum(record["certain_disagreements"].values()) for record in flat
            ),
        },
        "degeneracy_semantics": {
            # EXACT reports exact zeros; only the triangulations break ties, by
            # index-ordered symbolic perturbation.
            "exact_mode": "exact-zero",
            "triangulation_ties": "index-ordered-symbolic-perturbation",
            "exact_zero_rows": sum(record["exact_signs"]["zero"] for record in flat),
            "reference_zero_rows": sum(record["reference_zeros"] for record in flat),
            "symbolic_zero_rows": sum(record["symbolic"]["zeros"] for record in flat),
        },
        "resources": resources,
        "predicates": records,
    }


def _simplex_measures(points: np.ndarray, simplices: np.ndarray, /) -> np.ndarray:
    """Signed triangle areas or tetrahedron volumes."""
    edges = points[simplices[:, 1:]] - points[simplices[:, :1]]
    if points.shape[1] == 2:
        return 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    return np.sum(edges[:, 0] * np.cross(edges[:, 1], edges[:, 2]), axis=1) / 6.0


def _delaunay_record(points: np.ndarray, /) -> dict[str, object]:
    """Exact orientation, empty-ball, hull-coverage, and repeat checks of one set."""
    from scipy.spatial import ConvexHull

    exact = phx.geometry.PredicateMode.EXACT
    triangulation, stage = _measured(lambda: phx.geometry.DelaunayTriangulation(points))
    # ty: ignore[unresolved-attribute]
    simplices = triangulation.simplices
    corners = tuple(points[simplices[:, index]] for index in range(simplices.shape[1]))
    # Every (simplex, point) pair.
    balls = tuple(corner[:, None, :] for corner in corners)
    if points.shape[1] == 2:
        orientation = phx.geometry.orient2d(*corners, mode=exact)
        # ty: ignore[too-many-positional-arguments]
        inside = phx.geometry.incircle(*balls, points[None], mode=exact)
    else:
        orientation = phx.geometry.orient3d(*corners, mode=exact)
        # ty: ignore[too-many-positional-arguments]
        inside = phx.geometry.insphere(*balls, points[None], mode=exact)
    hull = ConvexHull(points).volume
    measure = float(np.sum(_simplex_measures(points, simplices)))
    repeated = phx.geometry.DelaunayTriangulation(points)
    # ty: ignore[unresolved-attribute]
    evidence = triangulation.evidence
    record = {
        "points": points.shape[0],
        "simplices": simplices.shape[0],
        "simplices_id": canonical_fingerprint(simplices),
        "evidence_id": evidence.evidence_id,
        "route": evidence.route,
        "status": evidence.status,
        "predicate_mode": evidence.predicate_mode.value,
        "positive_orientations": int(np.count_nonzero(orientation.signs == 1)),
        "in_ball_checks": inside.signs.size,
        "empty_ball_violations": int(np.count_nonzero(inside.signs > 0)),
        # Zero signs beyond each simplex's own vertices are cocircular/cospherical ties.
        "cospherical_ties": int(np.count_nonzero(inside.signs == 0)) - simplices.size,
        "hull_measure": hull,
        "measure": measure,
        "relative_hull_residual": abs(measure - hull) / hull,
        "repeat_equal": bool(
            np.array_equal(repeated.simplices, simplices)
            and repeated.evidence.evidence_id == evidence.evidence_id
        ),
        "resources": stage,
    }
    if (
        not (np.all(orientation.certain) and np.all(inside.certain))
        or record["positive_orientations"] != simplices.shape[0]
        or record["empty_ball_violations"]
        or record["relative_hull_residual"] > 1.0e-12
        or not record["repeat_equal"]
        or evidence.status != "ok"
    ):
        raise RuntimeError(f"{evidence.route} verification failed: {record}")
    return record


def _duplicate_record(rng: np.random.Generator, dimension: int, /) -> dict[str, object]:
    """Duplicated points map to their first copy and never become vertices or cells."""
    count, copies = 48, 8
    base = rng.random((count, dimension))
    chosen = np.sort(rng.choice(count, size=copies, replace=False))
    points = np.concatenate((base, base[chosen]))
    triangulation = phx.geometry.DelaunayTriangulation(points)
    unique = phx.geometry.DelaunayTriangulation(base)
    voronoi = phx.geometry.VoronoiDiagram(
        points,
        box_lower=np.zeros((dimension,), dtype=np.float64),
        box_upper=np.ones((dimension,), dtype=np.float64),
    )
    measures = voronoi.cells.measures
    record = {
        "input_points": points.shape[0],
        "duplicate_count": triangulation.evidence.duplicate_count,
        "vertex_count": triangulation.evidence.vertex_count,
        "vertex_map_to_first_copy": bool(
            np.array_equal(
                triangulation.vertex_map, np.concatenate((np.arange(count), chosen))
            )
        ),
        "duplicates_in_simplices": int(
            np.count_nonzero(triangulation.simplices >= count)
        ),
        "simplices_equal_unique_input": bool(
            np.array_equal(triangulation.simplices, unique.simplices)
        ),
        "voronoi_duplicate_count": voronoi.evidence.duplicate_count,
        "voronoi_duplicate_measure": float(np.sum(measures[count:])),
        "voronoi_measure_sum": float(np.sum(measures)),
    }
    if (
        record["duplicate_count"] != copies
        or record["voronoi_duplicate_count"] != copies
        or not record["vertex_map_to_first_copy"]
        or record["duplicates_in_simplices"]
        or not record["simplices_equal_unique_input"]
        or record["voronoi_duplicate_measure"] != 0.0
        or abs(record["voronoi_measure_sum"] - 1.0) > 1.0e-12
    ):
        raise RuntimeError(f"{dimension}D duplicate handling failed: {record}")
    return record


def _segment_conformity(
    triangulation: phx.geometry.ConstrainedDelaunayTriangulation,
    segments: np.ndarray,
    /,
) -> dict[str, object]:
    """Exact collinearity and length coverage of the edges carrying each segment."""
    points, triangles = triangulation.points, triangulation.triangles
    cells, corners = np.nonzero(triangulation.segment_ids >= 0)
    edges = np.sort(
        np.stack(
            (triangles[cells, (corners + 1) % 3], triangles[cells, (corners + 2) % 3]),
            axis=1,
        ),
        axis=1,
    )
    carried = np.unique(
        np.column_stack((triangulation.segment_ids[cells, corners], edges)), axis=0
    )
    owner, first, second = carried.T
    start, stop = points[segments[owner, 0]], points[segments[owner, 1]]
    on_line = phx.geometry.orient2d(
        np.concatenate((start, start)),
        np.concatenate((stop, stop)),
        np.concatenate((points[first], points[second])),
        mode=phx.geometry.PredicateMode.EXACT,
    )
    lengths = np.linalg.norm(points[second] - points[first], axis=1)
    covered = np.bincount(owner, weights=lengths, minlength=segments.shape[0])
    expected = np.linalg.norm(points[segments[:, 1]] - points[segments[:, 0]], axis=1)
    return {
        "segments": segments.shape[0],
        "carried_edges": carried.shape[0],
        "uncovered_segments": int(np.count_nonzero(covered == 0.0)),
        "off_segment_endpoints": int(np.count_nonzero(on_line.signs != 0)),
        "maximum_relative_length_defect": float(
            np.max(np.abs(covered - expected) / expected)
        ),
    }


def _constrained_record() -> tuple[
    dict[str, object], phx.meshing.CellMeshingResult, phx.meshing.CellMeshingResult
]:
    """Plate with a hole and an interior crack: CDT, refinement, and certification."""
    points = np.asarray(
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.75, 0.75),
            (1.25, 0.75),
            (1.25, 1.25),
            (0.75, 1.25),
            (0.25, 0.4),
            (1.75, 0.4),
        )
    )
    loop = np.arange(4, dtype=np.int32)
    ring = np.stack((loop, (loop + 1) % 4), axis=1)
    segments = np.concatenate((ring, 4 + ring, np.asarray(((8, 9),), dtype=np.int32)))
    holes = np.asarray(((1.0, 1.0),))
    min_angle, max_area = 28.0, 0.02

    def triangulate(refine: bool) -> Any:
        return phx.geometry.ConstrainedDelaunayTriangulation(
            points,
            segments,
            holes=holes,
            min_angle=min_angle if refine else 0.0,
            max_area=max_area if refine else np.inf,
        )

    unrefined, unrefined_stage = _measured(lambda: triangulate(False))
    refined, refined_stage = _measured(lambda: triangulate(True))
    repeated = triangulate(True)
    # ty: ignore[unresolved-attribute]
    areas = _simplex_measures(refined.points, refined.triangles)
    orientation = phx.geometry.orient2d(
        # ty: ignore[unresolved-attribute]
        *(refined.points[refined.triangles[:, index]] for index in range(3)),
        mode=phx.geometry.PredicateMode.EXACT,
    )

    def certify(triangulation: Any) -> Any:
        return phx.meshing.certify_cell_mesh(
            phx.discretization.CellMesh.from_triangles(
                triangulation.points, triangulation.triangles
            ),
            _contract(),
        )

    before, before_stage = _measured(lambda: certify(unrefined))
    after, after_stage = _measured(lambda: certify(refined))
    plate_area = 4.0 - 0.25
    record = {
        # ty: ignore[unresolved-attribute]
        "status": refined.evidence.status,
        # ty: ignore[unresolved-attribute]
        "route": refined.evidence.route,
        # ty: ignore[unresolved-attribute]
        "predicate_mode": refined.evidence.predicate_mode.value,
        "requested_minimum_angle_degrees": min_angle,
        # ty: ignore[unresolved-attribute]
        "minimum_angle_degrees": refined.evidence.minimum_angle_degrees,
        # ty: ignore[unresolved-attribute]
        "unrefined_minimum_angle_degrees": unrefined.evidence.minimum_angle_degrees,
        # ty: ignore[unresolved-attribute]
        "certified_minimum_angle_degrees": float(np.degrees(after.quality.minimum_angle)),
        "requested_maximum_area": max_area,
        "maximum_area": float(np.max(areas)),
        "relative_area_residual": abs(float(np.sum(areas)) - plate_area) / plate_area,
        "positive_orientations": int(np.count_nonzero(orientation.signs == 1)),
        # ty: ignore[unresolved-attribute]
        "triangles": refined.triangles.shape[0],
        # ty: ignore[unresolved-attribute]
        "unrefined_triangles": unrefined.triangles.shape[0],
        # ty: ignore[unresolved-attribute]
        "steiner_points": refined.evidence.steiner_count,
        # ty: ignore[invalid-argument-type]
        "conformity": _segment_conformity(refined, segments),
        # ty: ignore[invalid-argument-type]
        "unrefined_conformity": _segment_conformity(unrefined, segments),
        # ty: ignore[unresolved-attribute]
        "evidence_id": refined.evidence.evidence_id,
        "repeat_equal": bool(
            # ty: ignore[unresolved-attribute]
            repeated.evidence.evidence_id == refined.evidence.evidence_id
            # ty: ignore[unresolved-attribute]
            and np.array_equal(repeated.points, refined.points)
            # ty: ignore[unresolved-attribute]
            and np.array_equal(repeated.triangles, refined.triangles)
        ),
        "resources": {
            "unrefined": unrefined_stage,
            "refined": refined_stage,
            "certify_unrefined": before_stage,
            "certify_refined": after_stage,
        },
    }
    conformity = (record["conformity"], record["unrefined_conformity"])
    if (
        record["status"] != "ok"
        or record["minimum_angle_degrees"] < min_angle
        or record["maximum_area"] > max_area
        or record["relative_area_residual"] > 1.0e-12
        or record["positive_orientations"] != record["triangles"]
        or any(
            item["uncovered_segments"]
            or item["off_segment_endpoints"]
            # ty: ignore[unsupported-operator]
            or item["maximum_relative_length_defect"] > 1.0e-12
            for item in conformity
        )
        or not record["repeat_equal"]
        # ty: ignore[unresolved-attribute]
        or not (before.audit.passed and after.audit.passed)
        or record["certified_minimum_angle_degrees"] < min_angle * (1.0 - 1.0e-12)
    ):
        raise RuntimeError(f"Constrained Delaunay qualification failed: {record}")
    # ty: ignore[invalid-return-type]
    return record, before, after


def _reciprocal_faces(cells: phx.geometry.DiagramCells, /) -> bool:
    """Every interior face (owner, neighbor) has the opposite face (neighbor, owner)."""
    owners = np.repeat(np.arange(cells.cell_count), np.diff(cells.cell_face_offsets))
    interior = cells.face_labels >= 0
    forward = np.unique(
        np.stack((owners[interior], cells.face_labels[interior]), axis=1), axis=0
    )
    return bool(np.array_equal(forward, np.unique(forward[:, ::-1], axis=0)))


def _diagram_record(
    diagram: Any, repeated: Any, lower: np.ndarray, upper: np.ndarray, stage: dict, /
) -> dict[str, object]:
    """Box coverage, reciprocity, centroid, and repeat checks of one diagram."""
    cells, evidence = diagram.cells, diagram.evidence
    box = float(np.prod(upper - lower))
    total = float(np.sum(cells.measures))
    occupied = cells.measures > 0.0
    centroids = cells.centroids[occupied]
    record = {
        "route": evidence.route,
        "predicate_mode": evidence.predicate_mode.value,
        "evidence_id": evidence.evidence_id,
        "measures_id": canonical_fingerprint(cells.measures),
        "cells": cells.cell_count,
        "empty_cells": int(np.count_nonzero(~occupied)),
        "redundant_generators": evidence.redundant_count,
        "dual_simplices": evidence.simplex_count,
        "box_measure": box,
        "measure_sum": total,
        "relative_box_residual": abs(total - box) / box,
        "reciprocal_faces": _reciprocal_faces(cells),
        "centroids_inside_box": bool(np.all((centroids >= lower) & (centroids <= upper))),
        "repeat_equal": bool(
            repeated.evidence.evidence_id == evidence.evidence_id
            and np.array_equal(repeated.cells.vertices, cells.vertices)
        ),
        "resources": stage,
    }
    if (
        record["relative_box_residual"] > 1.0e-12
        or np.any(cells.measures < 0.0)
        or not record["reciprocal_faces"]
        or not record["centroids_inside_box"]
        or not record["repeat_equal"]
    ):
        raise RuntimeError(f"{evidence.route} qualification failed: {record}")
    return record


def _diagram_pair(
    rng: np.random.Generator, dimension: int, count: int, /
) -> dict[str, dict[str, object]]:
    """Voronoi and weighted power diagrams of one generator set in a box."""
    lower = np.zeros((dimension,), dtype=np.float64)
    upper = np.asarray((1.0, 2.0, 1.0)[:dimension], dtype=np.float64)
    generators = lower + (upper - lower) * rng.random((count, dimension))
    weights = 0.01 * rng.random(count)
    box = {"box_lower": lower, "box_upper": upper}
    voronoi, voronoi_stage = _measured(
        lambda: phx.geometry.VoronoiDiagram(generators, **box)
    )
    power, power_stage = _measured(
        lambda: phx.geometry.PowerDiagram(generators, weights, **box)
    )
    unweighted = phx.geometry.PowerDiagram(
        generators,
        np.zeros((count,), dtype=np.float64),
        **box,
    )
    voronoi_record = _diagram_record(
        voronoi,
        phx.geometry.VoronoiDiagram(generators, **box),
        lower,
        upper,
        voronoi_stage,
    )
    # ty: ignore[unresolved-attribute]
    deviation = float(np.max(np.abs(unweighted.cells.measures - voronoi.cells.measures)))
    voronoi_record["unweighted_power_maximum_deviation"] = deviation
    # ty: ignore[unresolved-attribute]
    if deviation > 1.0e-10 * float(np.max(voronoi.cells.measures)):
        raise RuntimeError("Unweighted power cells differ from the Voronoi cells.")
    return {
        f"voronoi_{dimension}d": voronoi_record,
        f"power_{dimension}d": _diagram_record(
            power,
            phx.geometry.PowerDiagram(generators, weights, **box),
            lower,
            upper,
            power_stage,
        ),
    }


def qualify_triangulation() -> dict[str, object]:
    if not meshcore_available():
        return _missing_dependency(
            "meshcore",
            "Exact triangulations and diagrams require the native meshcore library; "
            "install phydrax[meshcore] or set PHYDRAX_MESHCORE_LIBRARY to the shared "
            "library path.",
        )
    rng = np.random.default_rng(20260927)
    lattice_2d = np.stack(np.meshgrid(*(np.arange(6.0),) * 2, indexing="ij"), axis=-1)
    lattice_3d = np.stack(np.meshgrid(*(np.arange(4.0),) * 3, indexing="ij"), axis=-1)
    delaunay = {
        "random_2d": _delaunay_record(rng.random((200, 2))),
        "lattice_2d": _delaunay_record(lattice_2d.reshape((-1, 2))),
        "random_3d": _delaunay_record(rng.random((60, 3))),
        "lattice_3d": _delaunay_record(lattice_3d.reshape((-1, 3))),
    }
    duplicates = {
        f"{dimension}d": _duplicate_record(rng, dimension) for dimension in (2, 3)
    }
    constrained, unrefined, refined = _constrained_record()
    diagrams = {**_diagram_pair(rng, 2, 64), **_diagram_pair(rng, 3, 40)}
    # A zero-weight generator inside the heavy triangle is redundant: no cell.
    hidden = phx.geometry.PowerDiagram(
        np.asarray(((0.2, 0.2), (0.8, 0.2), (0.5, 0.8), (0.5, 0.4))),
        np.asarray((1.0, 1.0, 1.0, 0.0)),
        box_lower=(0.0, 0.0),
        box_upper=(1.0, 1.0),
    )
    redundant = {
        "redundant_count": hidden.evidence.redundant_count,
        # ty: ignore[unresolved-attribute]
        "dual_vertex_map": hidden.dual_vertex_map.tolist(),
        "redundant_measure": float(hidden.cells.measures[3]),
        "measure_sum": float(np.sum(hidden.cells.measures)),
    }
    if (
        redundant["redundant_count"] != 1
        or redundant["dual_vertex_map"][3] != -1
        or redundant["redundant_measure"] != 0.0
        or abs(redundant["measure_sum"] - 1.0) > 1.0e-12
    ):
        raise RuntimeError(f"Redundant power generator was not hidden: {redundant}")
    resources = {
        **{
            f"delaunay_{name}": record.pop("resources")
            for name, record in delaunay.items()
        },
        **{
            f"constrained_{name}": stage
            # ty: ignore[unresolved-attribute]
            for name, stage in constrained.pop("resources").items()
        },
        **{name: record.pop("resources") for name, record in diagrams.items()},
    }
    return {
        "status": "passed",
        "source": {
            "constrained_unrefined": _result_identity(unrefined),
            "delaunay_simplices": {
                name: record["simplices_id"] for name, record in delaunay.items()
            },
        },
        "target": {
            "constrained_refined": _result_identity(refined),
            "evidence": {
                **{name: record["evidence_id"] for name, record in delaunay.items()},
                "constrained": constrained["evidence_id"],
                **{name: record["evidence_id"] for name, record in diagrams.items()},
            },
        },
        "runtime": _runtime_identity(),
        "quality": {
            "before": _quality_summary(unrefined.quality),
            "after": _quality_summary(refined.quality),
        },
        "conservation": {
            "delaunay_relative_hull_residual": {
                name: record["relative_hull_residual"]
                for name, record in delaunay.items()
            },
            "constrained_relative_area_residual": constrained["relative_area_residual"],
            "diagram_relative_box_residual": {
                name: record["relative_box_residual"] for name, record in diagrams.items()
            },
        },
        "determinism": {
            **{name: record["repeat_equal"] for name, record in delaunay.items()},
            "constrained": constrained["repeat_equal"],
            **{name: record["repeat_equal"] for name, record in diagrams.items()},
        },
        "resources": resources,
        "delaunay": delaunay,
        "duplicates": duplicates,
        "constrained": constrained,
        "diagrams": diagrams,
        "redundant_power_generator": redundant,
    }



from .._runtime import runtime_identity as _runtime_identity
