"""Native meshing kernels against the measured best reference per row.

Each row times one native kernel on a fixed generated input, single threaded, with
host timing (the kernels return host arrays). The ceiling of a row is the best
reference time measured on the development machine for the same input
(`native_meshing_ceilings.json`, with its provenance); a row passes when the
kernel's minimum time is at most its ceiling times the row's allowed ratio.
Ratios above one are interim acceptance targets of the plan
`docs/plans/native-meshing-kernels.md`; the final target of every row is 1.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from _runtime import capture_environment, measure_host

from phydrax import _meshcore


_CEILINGS = Path(__file__).with_name("native_meshing_ceilings.json")


@dataclass(frozen=True, slots=True)
class _Row:
    name: str
    size: int
    run: Callable[[], object]
    count: Callable[[object], int]


def _uniform_points(count: int, dimension: int, seed: int, /) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.random((count, dimension), dtype=np.float64)


def _cells(result: object, /) -> int:
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("A Delaunay kernel returns (cells, regions).")
    cells = result[0]
    if not isinstance(cells, np.ndarray):
        raise TypeError("Delaunay cells must be a NumPy array.")
    return cells.shape[0]


def _rows(sizes: tuple[int, ...], /) -> tuple[_Row, ...]:
    rows: list[_Row] = []
    for size in sizes:
        planar = _uniform_points(size, 2, 20261119)
        spatial = _uniform_points(size, 3, 20261119)
        rows.append(
            _Row(
                "delaunay-2d",
                size,
                lambda points=planar: _meshcore.delaunay_2d(points),
                _cells,
            )
        )
        rows.append(
            _Row(
                "delaunay-3d",
                size,
                lambda points=spatial: _meshcore.delaunay_3d(points),
                _cells,
            )
        )
    return tuple(rows)


def _ceiling(
    ceilings: Mapping[str, Any], row: _Row, /
) -> tuple[float, float, str] | None:
    entry = ceilings["rows"].get(row.name, {}).get(str(row.size))
    if entry is None:
        return None
    return float(entry["seconds"]), float(entry["allowed_ratio"]), str(entry["tool"])


def _measure(row: _Row, repeats: int, /) -> dict[str, Any]:
    samples: list[float] = []
    count = 0
    for _ in range(repeats):
        result, seconds = measure_host(row.run)
        samples.append(seconds)
        count = row.count(result)
    return {"elements": count, "seconds": samples, "minimum_seconds": min(samples)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000, 1_000_000]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("Run with OMP_NUM_THREADS=1: rows are single-thread rows.")
    ceilings = json.loads(_CEILINGS.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    failed = False
    for row in _rows(tuple(arguments.sizes)):
        measured = _measure(row, arguments.repeats)
        record: dict[str, Any] = {"row": row.name, "size": row.size, **measured}
        ceiling = _ceiling(ceilings, row)
        if ceiling is not None:
            seconds, allowed, tool = ceiling
            ratio = measured["minimum_seconds"] / seconds
            record |= {
                "reference_tool": tool,
                "reference_seconds": seconds,
                "ratio": ratio,
                "allowed_ratio": allowed,
                "passed": ratio <= allowed,
            }
            failed = failed or ratio > allowed
        records.append(record)
        print(json.dumps(record), flush=True)
    if arguments.output is not None:
        arguments.output.write_text(
            json.dumps(
                {
                    "environment": capture_environment().to_dict(),
                    "meshcore": _meshcore.meshcore_identity(),
                    "ceilings_provenance": ceilings["provenance"],
                    "rows": records,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
