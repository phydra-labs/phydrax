# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import numpy as np

from phydrax import canonical_fingerprint
from phydrax.meshing import meshcore_available
from phydrax.meshing.qualification import (
    qualify_predicate_families,
    reference_predicate_signs,
)
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import (
    _measured,
    _missing_dependency,
    _runtime_identity,
    qualification_failure,
)


_PREDICATE_ARITY = {"orient2d": 3, "orient3d": 4, "incircle": 4, "insphere": 5}


_PREDICATE_OFFSET = 2.0**40


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
    one-ulp move never creates a subnormal outside the meshcore exact arithmetic domain.
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
    orientation = reference_predicate_signs(
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

    families = _predicate_families(np.random.default_rng(20260926))
    records, resources, exact_signs, antisymmetry = {}, {}, {}, {}
    for name, rows in families.items():
        qualified = qualify_predicate_families(name, rows, measure=_measured)
        records[name] = qualified.records
        resources[name] = qualified.resources
        exact_signs[name] = qualified.exact_signs
        antisymmetry[name] = qualified.antisymmetry_violations
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
        raise qualification_failure(
            "qualify_predicates",
            ("Predicate qualification failed: " + "; ".join(failures)),
            {"failures": failures},
        )
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
            "meshcore_exact_exponent_range": list(qualified.exact_exponent_range),
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


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_predicates()
