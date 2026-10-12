"""Measure representative host-table hot sets and byte-bounded cache churn."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import math
import time
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from typing import Any

import jax
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    fitted_scaling_exponent,
    host_allocation_peak,
    process_peak_rss_bytes,
)
from benchmarks._runtime import capture_benchmark_identity, capture_environment


# These supported preparation workloads determine the ceilings, not guessed RSS.
HOT_SETS: tuple[tuple[str, str, tuple[tuple[Any, ...], ...]], ...] = (
    (
        "discretization.spectral._spherical_algebra",
        "_coupled_angular_momentum_block",
        tuple(
            (a, b, m)
            for a in range(5)
            for b in range(5)
            for m in range(-a - b, a + b + 1)
        ),
    ),
    (
        "discretization._cell_geometry_validity",
        "_bernstein_plan",
        (
            ("box", (2, 2)),
            ("box", (4, 4)),
            ("box", (2, 2, 2)),
            ("simplex", (4, 2)),
            ("simplex", (4, 3)),
            ("prism", (2, 2)),
        ),
    ),
    ("meshing._quality", "_corner_table", tuple(("polygon", n) for n in range(3, 33))),
    (
        "applications.conformal_bootstrap._virasoro_generic",
        "_partitions",
        tuple((n,) for n in range(13)),
    ),
    (
        "applications.conformal_bootstrap._virasoro_generic",
        "_canonical_negative_modes",
        tuple((tuple(range(1, n + 1)),) for n in range(1, 6)),
    ),
    (
        "_polynomial._orthogonal",
        "_derivative_operator",
        tuple(
            (family, n, order)
            for family in ("chebyshev", "legendre", "hermite_e")
            for n in (8, 16, 32, 64, 128)
            for order in (0, 1, 2)
        ),
    ),
    (
        "control.games._master_equation",
        "_weak_compositions",
        tuple((n, p) for n in (2, 4, 8, 16) for p in (2, 3, 4)),
    ),
    (
        "discretization.meshfree._stencils",
        "_basis_exponents",
        tuple((d, p) for d in (1, 2, 3) for p in range(1, 7)),
    ),
    (
        "exterior._basis",
        "wedge_table",
        tuple((n, 1, k) for n in range(2, 9) for k in range(1, n)),
    ),
    (
        "exterior._basis",
        "derivative_table",
        tuple((n, k) for n in range(2, 9) for k in range(n)),
    ),
    (
        "exterior._basis",
        "interior_table",
        tuple((n, k) for n in range(2, 9) for k in range(1, n + 1)),
    ),
    (
        "exterior._basis",
        "complement_table",
        tuple((n, k) for n in range(2, 9) for k in range(n + 1)),
    ),
    (
        "nn.operator.representations._irreps",
        "_prepared_real_coupling",
        tuple(
            (a, b, c)
            for a in range(4)
            for b in range(4)
            for c in range(abs(a - b), a + b + 1)
        ),
    ),
    (
        "special._real_harmonic",
        "_real_harmonic_basis_terms",
        tuple((n,) for n in range(33)),
    ),
    (
        "tensor_network._su2",
        "_su2_clebsch_gordan_entries",
        tuple(
            (a, b, c)
            for a in range(5)
            for b in range(5)
            for c in range(abs(a - b), a + b + 1, 2)
        ),
    ),
    (
        "_numerics._smolyak",
        "_weighted_total_degree_indices_cached",
        tuple((d, level, (1.0,) * d) for d in (2, 3, 5) for level in range(1, 6)),
    ),
    (
        "_numerics._smolyak",
        "_smolyak_terms_cached",
        tuple((d, level, (1.0,) * d) for d in (2, 3, 5) for level in range(1, 6)),
    ),
    ("_numerics._smolyak", "_leja_nodes", tuple((n,) for n in (4, 8, 16, 32, 64, 128))),
    (
        "_numerics._smolyak",
        "smolyak_axis_data",
        tuple(
            (rule, level)
            for rule in ("clenshaw-curtis", "leja", "gauss-hermite")
            for level in range(7)
        ),
    ),
    (
        "discretization._reference_cell",
        "reference_cell_topology",
        tuple(
            (f"{family}:{d}",) for family in ("simplex", "tensor") for d in range(1, 7)
        ),
    ),
    (
        "_numerics._adaptive_cubature_rules",
        "_genz_malik_host_table",
        tuple((d, 7) for d in range(2, 7)),
    ),
    (
        "nn.atomistic._symmetric_contraction",
        "_native_coupling_span",
        tuple(
            (((0, 1, 0), (1, -1, 1)), 4, out, parity, correlation)
            for correlation in (1, 2, 3)
            for out, parity in ((0, 1), (1, -1))
        ),
    ),
)


def fem_hot_sets() -> tuple[tuple[str, str, tuple[tuple[Any, ...], ...]], ...]:
    from phydrax.discretization.fem._form_elements import _cube_face, _tensor_vertices

    def labels(n: int, k: int, r: int) -> tuple[Any, ...]:
        rows = []
        vertices = _tensor_vertices(n)
        for blade in combinations(range(n), k):
            for index in product(
                *(range(r) if axis in blade else range(r + 1) for axis in range(n))
            ):
                states = tuple(
                    -1 if axis in blade or 0 < mode < r else (0 if mode == 0 else 1)
                    for axis, mode in enumerate(index)
                )
                rows.append((_cube_face(vertices, states), index, blade))
        return tuple(sorted(rows, key=lambda row: (len(row[0]), row)))

    module = "discretization.fem._form_elements"
    return (
        (module, "_tensor_interval", tuple((r,) for r in range(1, 6))),
        (
            module,
            "_tensor_derivative",
            tuple(
                (r, labels(n, k, r), labels(n, k + 1, r))
                for n in (1, 2, 3)
                for r in range(1, 5)
                for k in range(n)
            ),
        ),
    )


def payload_bytes(value: Any, /) -> int:
    """Explicit logical arrays and structural scalars, without copying payloads."""
    if isinstance(value, (np.ndarray, jax.Array)):
        return value.nbytes
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(
            payload_bytes(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, (tuple, list)):
        return sum(payload_bytes(item) for item in value)
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, complex):
        return 16
    if isinstance(value, Fraction):
        return payload_bytes((value.numerator, value.denominator))
    if isinstance(value, int):
        return max(8, (value.bit_length() + 7) // 8)
    if isinstance(value, (float, np.number)):
        return 8
    raise TypeError(f"Unsupported benchmark table value {type(value)!r}.")


def hot_sets(candidate: bool, /, *, fem_only: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    workloads = fem_hot_sets() if fem_only else (*HOT_SETS, *fem_hot_sets())
    for module, symbol, arguments in workloads:
        function = getattr(importlib.import_module(f"phydrax.{module}"), symbol)
        function.cache_clear()
        sizes: list[int] = []
        started = time.perf_counter()
        for args in arguments:
            sizes.append(payload_bytes(function(*args)))
        hot_bytes = sum(sizes)
        # Eight measured hot sets of headroom, with a one-MiB minimum.
        proposed_cap = 1 << max(20, math.ceil(math.log2(max(1, 8 * hot_bytes))))
        info = function.cache_info()
        record: dict[str, Any] = {
            "module": module,
            "symbol": symbol,
            "arguments": arguments,
            "requested_payload_bytes": hot_bytes,
            "largest_value_bytes": max(sizes),
            "measured_hot_set_cap_bytes": proposed_cap,
            "cap_rule": "max(1 MiB, next power of two >= 8 * measured hot-set bytes)",
            "cache_entries": info.currsize,
            "wall_seconds": time.perf_counter() - started,
        }
        if candidate:
            record["configured_cap_bytes"] = info.max_bytes
            record["configured_max_entries"] = info.max_entries
            record["retained_bytes"] = info.current_bytes
            record["within_cap"] = info.current_bytes <= 1.05 * info.max_bytes
        records.append(record)
    return records


def churn(candidate: bool, /) -> dict[str, Any]:
    cap = 1048576

    def compute(index: int) -> np.ndarray:
        count = cap // 8 + 1 if index == -1 else 64 + index % 128
        return np.arange(count, dtype=np.float64)

    if candidate:
        from phydrax._host_table_cache import host_table_cache

        function = host_table_cache(max_entries=2048, max_bytes=cap)(compute)
    else:
        function = lru_cache(maxsize=None)(compute)
    counts = (64, 256, 1024, 4096)
    records: list[dict[str, int | float | bool]] = []
    requested = 0
    cursor = 0
    retained_series: list[int] = []
    for count in counts:
        started = time.perf_counter()

        def sweep() -> None:
            nonlocal requested, cursor
            while cursor < count:
                requested += function(cursor).nbytes
                cursor += 1

        _, peak = host_allocation_peak(sweep)
        from phydrax._host_table_cache import HostTableCacheInfo

        info = function.cache_info()
        retained = (
            info.current_bytes if isinstance(info, HostTableCacheInfo) else requested
        )
        retained_series.append(retained)
        records.append(
            {
                "keys": count,
                "retained_bytes": retained,
                "traced_peak_bytes": peak,
                "wall_seconds": time.perf_counter() - started,
                "within_cap": retained <= 1.05 * cap,
            }
        )
    oversized: dict[str, int | bool] | None = None
    if candidate:
        before = function.cache_info()
        value = function(-1)
        function(-1)
        after = function.cache_info()
        if not isinstance(before, HostTableCacheInfo) or not isinstance(
            after, HostTableCacheInfo
        ):
            raise TypeError("Candidate cache must expose byte-retention evidence.")
        oversized = {
            "value_bytes": value.nbytes,
            "not_retained": before.current_bytes == after.current_bytes
            and before.currsize == after.currsize
            and after.misses == before.misses + 2,
        }
    return {
        "cap_bytes": cap,
        "retained_scaling_exponent": fitted_scaling_exponent(counts, retained_series),
        "oversized": oversized,
        "records": records,
        "plateau_pass": all(row["within_cap"] for row in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--fem-only",
        action="store_true",
        help="Capture newly scoped FE caches independently.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="The default workloads are already host-only smoke sized.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    root = Path(__file__).resolve().parents[1]
    identity = capture_benchmark_identity(
        root,
        Path(__file__),
        {
            "cases": "host-table-hot-sets-and-churn",
            "logical_weight": "array-nbytes-and-structural-scalars",
        },
    )
    result = {
        "identity": identity.to_dict(),
        "environment": capture_environment().to_dict(),
        "baseline": args.baseline,
        "hot_sets": hot_sets(not args.baseline, fem_only=args.fem_only),
        "churn": churn(not args.baseline),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "compiler_evidence": None,
    }
    destination = args.output or root / "benchmarks" / (
        "memory_host_caches_baseline.json" if args.baseline else "memory_host_caches.json"
    )
    write_json_atomic(destination, result)
    print(destination)


if __name__ == "__main__":
    main()
