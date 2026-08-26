#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class AlgebraBenchmarkRecord:
    family: str
    coordinate_dimension: int
    product_terms: int
    plan_bytes: int
    associative_status: str
    alternative_status: str
    preparation_wall_ms: float
    first_jit_ms: float
    steady_wall_ms: float
    sparse_dense_defect: float
    finite: bool

    @property
    def passed(self) -> bool:
        return self.finite and self.sparse_dense_defect <= 1e-12


def _measure(function, left, right, repeats):
    compiled = eqx.filter_jit(function)
    started = time.perf_counter()
    value = compiled(left, right)
    jax.block_until_ready(value)
    first = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    for _ in range(repeats):
        value = compiled(left, right)
        jax.block_until_ready(value)
    steady = 1e3 * (time.perf_counter() - started) / repeats
    return value, first, steady


def run_algebra_benchmark(level: int, /, *, repeats: int = 5) -> AlgebraBenchmarkRecord:
    started = time.perf_counter()
    algebra = phx.metrix.algebra.CayleyDicksonAlgebraSpec(level)
    sparse = algebra.prepare_product(backend="sparse")
    dense = algebra.prepare_product(backend="dense")
    preparation = 1e3 * (time.perf_counter() - started)
    dimension = algebra.coordinate_dimension
    left = jnp.sin(jnp.arange(8 * dimension, dtype=float)).reshape((8, dimension))
    right = jnp.cos(jnp.arange(dimension, dtype=float))
    sparse_value, first, steady = _measure(sparse, left, right, repeats)
    dense_value = dense(left, right)
    defect = jnp.max(jnp.abs(sparse_value - dense_value))
    finite = bool(jnp.all(jnp.isfinite(sparse_value)) & jnp.isfinite(defect))
    return AlgebraBenchmarkRecord(
        family=algebra.family,
        coordinate_dimension=dimension,
        product_terms=algebra.structure.term_count,
        plan_bytes=sparse.evidence.resource_evidence.plan_bytes,
        associative_status=algebra.properties.claim("associative").status,
        alternative_status=algebra.properties.claim("alternative").status,
        preparation_wall_ms=float(preparation),
        first_jit_ms=float(first),
        steady_wall_ms=float(steady),
        sparse_dense_defect=float(defect),
        finite=finite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact finite real algebra products."
    )
    parser.add_argument("--maximum-level", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    maximum = 3 if arguments.smoke else arguments.maximum_level
    repeats = 1 if arguments.smoke else arguments.repeats
    records = [
        run_algebra_benchmark(level, repeats=repeats) for level in range(maximum + 1)
    ]
    print(
        json.dumps(
            [{**asdict(record), "passed": record.passed} for record in records], indent=2
        )
    )
    if not all(record.passed for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
