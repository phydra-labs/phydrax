#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Capacity-scaled cost of the scalar/tiled image reference against the dense one.

For each particle count the driver prepares one ``TiledImageRouteSearch`` per
declared tile policy (``RxS`` receiver x source widths; ``1x1`` is the scalar
route) and, where its all-pairs candidate tensor is admitted by
``--dense-limit``, the ``DenseImageRouteSearch`` comparator.  Density is held
at the requested mean degree, so required routes grow linearly while the
all-pairs work grows quadratically.  Every record separates host preparation
(time and traced allocation peak), lowering, compilation, compiler argument/
output/temporary/code bytes, the first synchronized call and warm repeats,
and the logical tile schedule.  Routes and evidence of every tiled policy are
compared slot by slot with the dense comparator, or with the first tiled policy
where dense is not admitted.  Fitted log-log exponents of compiler temporary
bytes against particle count are reported per route.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    fitted_scaling_exponent,
    host_allocation_peak,
    process_peak_rss_bytes,
)
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.discretization import (
    AbstractImageRouteSearch,
    DenseImageRouteSearch,
    ImageRouteSearchResult,
    ImageTilePolicy,
    ParticleImageCapacity,
    TiledImageRouteSearch,
)


_SHEAR = np.asarray([[1.0, 0.0, 0.0], [0.18, 1.0, 0.0], [0.07, -0.11, 1.0]])


def _tiles(value: str, /) -> tuple[int, int]:
    receiver, _, source = value.partition("x")
    widths = (int(receiver), int(source))
    if min(widths) < 1:
        raise argparse.ArgumentTypeError("Tile widths must be positive RxS integers.")
    return widths


def _geometry(
    particles: int, cases: int, radius: float, degree: float, seed: int, dtype: str, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random triclinic cells at the mean density giving ``degree`` neighbors."""
    volume = particles * (4.0 / 3.0) * math.pi * radius**3 / degree
    vectors = np.cbrt(volume / np.linalg.det(_SHEAR)) * _SHEAR
    rng = np.random.default_rng(seed)
    fractional = rng.uniform(0.0, 1.0, size=(cases, particles, 3))
    positions = (fractional @ vectors).astype(dtype)
    ids = np.stack([rng.permutation(particles) for _ in range(cases)]) + 1
    return positions, np.broadcast_to(vectors, (cases, 3, 3)).copy(), ids


def _capacity(
    particles: int, cases: int, degree: float, images: int, /
) -> ParticleImageCapacity:
    edges = max(16, int(math.ceil(2.0 * degree * particles)))
    return ParticleImageCapacity(
        maximum_particles_per_cell=particles,
        maximum_edges=edges * cases,
        maximum_degree=max(16, int(math.ceil(4.0 * degree))),
        maximum_images=images,
        maximum_case_edges=edges,
    )


def _route(
    search: AbstractImageRouteSearch,
    positions: np.ndarray,
    vectors: np.ndarray,
    ids: np.ndarray,
    repeats: int,
    /,
) -> tuple[ImageRouteSearchResult, dict[str, Any]]:
    cases, particles = positions.shape[:2]
    arguments = (
        jnp.asarray(positions),
        jnp.ones((cases, particles), dtype=jnp.bool_),
        jnp.asarray(ids),
        jnp.asarray(vectors, dtype=positions.dtype),
        jnp.zeros((cases, 3), dtype=positions.dtype),
        jnp.ones((cases, 3), dtype=jnp.bool_),
    )

    def call(*values: jax.Array) -> ImageRouteSearchResult:
        return search.routes(
            *values, support_id="benchmark", relation_schema_id="benchmark"
        )

    function = jax.jit(call)
    compiled, phases = measure_lower_and_compile(
        lambda: function.lower(*arguments), lambda lowered: lowered.compile()
    )
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    result, first = measure_synchronized(lambda: compiled(*arguments))
    _, warm = measure_repeated(lambda: compiled(*arguments), warmup=0, repeats=repeats)
    evidence = result.evidence
    return result, {
        "search_id": search.search_id,
        "candidate_slot_count": search.candidate_slot_count,
        "route_capacity": search.route_capacity,
        "phases": asdict(phases),
        "compiler": asdict(
            compiler_evidence(cost, compiled.memory_analysis(), source="xla")
        ),
        "first_seconds": first,
        "warm": warm.to_dict(unit="seconds"),
        "required_edges": np.asarray(evidence.required_edges).tolist(),
        "required_degree": np.asarray(evidence.required_degree).tolist(),
        "successful": np.asarray(evidence.successful).tolist(),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }


def _identical(left: ImageRouteSearchResult, right: ImageRouteSearchResult, /) -> bool:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    return left_tree == right_tree and all(
        np.array_equal(np.asarray(first), np.asarray(second))
        for first, second in zip(left_leaves, right_leaves, strict=True)
    )


def _case(
    particles: int, args: argparse.Namespace, tiles: list[tuple[int, int]], /
) -> dict[str, Any]:
    positions, vectors, ids = _geometry(
        particles, args.cases, args.radius, args.degree, args.seed, args.dtype
    )
    periodic = np.ones((args.cases, 3), dtype=np.bool_)
    probe = TiledImageRouteSearch(
        args.radius,
        vectors,
        periodic,
        _capacity(particles, args.cases, args.degree, 1 << 20),
        particle_capacity=particles,
        tiles=ImageTilePolicy(
            receiver_width=1, source_width=1, maximum_tile_candidates=1 << 30
        ),
    )
    images = int(probe.stencil.shape[0])
    capacity = _capacity(particles, args.cases, args.degree, images)
    routes: dict[str, dict[str, Any]] = {}
    results: dict[str, ImageRouteSearchResult] = {}
    all_pairs = args.cases * particles * particles * images
    if all_pairs <= args.dense_limit:
        dense, preparation = _prepared(
            lambda: DenseImageRouteSearch(
                args.radius,
                vectors,
                periodic,
                capacity,
                particle_capacity=particles,
                maximum_dense_routes=args.dense_limit,
            )
        )
        results["dense"], routes["dense"] = _route(
            dense, positions, vectors, ids, args.repeats
        )
        routes["dense"]["preparation"] = preparation
    for receiver, source in tiles:
        name = f"tiled-{receiver}x{source}"
        policy = ImageTilePolicy(
            receiver_width=receiver,
            source_width=source,
            maximum_tile_candidates=args.maximum_tile_candidates,
        )
        search, preparation = _prepared(
            lambda policy=policy: TiledImageRouteSearch(
                args.radius,
                vectors,
                periodic,
                capacity,
                particle_capacity=particles,
                tiles=policy,
            )
        )
        results[name], routes[name] = _route(
            search, positions, vectors, ids, args.repeats
        )
        routes[name]["preparation"] = preparation
        schedule = search.tile_schedule()
        routes[name]["schedule"] = {
            "tile_steps": schedule.tile_steps,
            "workspace_candidates": schedule.workspace_candidates,
            "traversed_candidates": schedule.traversed_candidates,
            "receiver_width": schedule.receiver_width,
            "source_width": schedule.source_width,
        }
    reference_name = "dense" if "dense" in results else next(iter(results))
    for name, result in results.items():
        routes[name]["matches_reference"] = _identical(result, results[reference_name])
    return {
        "particles": particles,
        "cases": args.cases,
        "images": images,
        "all_pairs_candidates": all_pairs,
        "capacity_id": capacity.capacity_id,
        "reference": reference_name,
        "routes": routes,
    }


def _prepared[S: AbstractImageRouteSearch](
    operation: Callable[[], S], /
) -> tuple[S, dict[str, float | int]]:
    """Host preparation time and traced allocation peak (tracing overhead included)."""
    (search, peak), seconds = measure_host(lambda: host_allocation_peak(operation))
    return search, {"seconds": seconds, "host_allocation_peak_bytes": peak}


def _exponents(cases: list[dict[str, Any]], /) -> dict[str, float | None]:
    names = {name for case in cases for name in case["routes"]}
    exponents: dict[str, float | None] = {}
    for name in sorted(names):
        rows = [
            (case["particles"], case["routes"][name]["compiler"]["temporary_bytes"])
            for case in cases
            if name in case["routes"]
        ]
        known = [(count, value) for count, value in rows if value]
        exponents[name] = (
            fitted_scaling_exponent(
                [count for count, _ in known], [value for _, value in known]
            )
            if len(known) >= 2
            else None
        )
    return exponents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument(
        "--tiles", nargs="+", type=_tiles, default=["1x1", "8x16", "32x64"]
    )
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--degree", type=float, default=24.0)
    parser.add_argument("--dense-limit", type=int, default=2**22)
    parser.add_argument("--maximum-tile-candidates", type=int, default=2**24)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tiles = [value if isinstance(value, tuple) else _tiles(value) for value in args.tiles]
    if (
        any(count < 1 for count in args.particles)
        or sorted(set(args.particles)) != args.particles
    ):
        raise ValueError("Particle counts must be positive and strictly increasing.")
    if args.cases < 1 or args.repeats < 1 or args.degree <= 0.0 or args.radius <= 0.0:
        raise ValueError("cases, repeats, degree and radius must be positive.")
    root = Path(__file__).resolve().parents[1]
    owners = (
        "phydrax/discretization/particle/_tiled_image_search.py",
        "phydrax/discretization/particle/_image_neighborhood.py",
    )
    cases = [_case(count, args, tiles) for count in args.particles]
    record = {
        "benchmark": "tiled-image-route-search-capacity-scaling",
        "label": args.label,
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            root,
            Path(__file__),
            ("cases", "configuration", "environment", "exponents", "source_sha256"),
        ).to_dict(),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in owners
        },
        "configuration": {
            "tiles": [f"{receiver}x{source}" for receiver, source in tiles],
            "cases": args.cases,
            "radius": args.radius,
            "degree": args.degree,
            "dense_limit": args.dense_limit,
            "maximum_tile_candidates": args.maximum_tile_candidates,
            "dtype": args.dtype,
            "seed": args.seed,
            "repeats": args.repeats,
        },
        "cases": cases,
        "exponents": {"compiler_temporary_bytes_vs_particles": _exponents(cases)},
    }
    write_json_atomic(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
