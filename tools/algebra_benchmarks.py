#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_repeated, measure_synchronized


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
    associator_first_jit_ms: float
    associator_steady_wall_ms: float
    associator_norm: float
    associator_classification_passed: bool
    regular_action_defect: float
    derivation_preparation_wall_ms: float | None
    derivation_dimension: int | None
    derivation_leibniz_residual: float | None
    derivation_converged: bool | None
    finite: bool

    @property
    def passed(self) -> bool:
        derivations_passed = (
            self.derivation_converged if self.derivation_converged is not None else True
        )
        return (
            self.finite
            and self.sparse_dense_defect <= 1e-12
            and self.regular_action_defect <= 1e-12
            and self.associator_classification_passed
            and derivations_passed
        )


def _measure(function, left, right, repeats):
    compiled = eqx.filter_jit(function)
    value, first_seconds = measure_synchronized(lambda: compiled(left, right))
    value, distribution = measure_repeated(
        lambda: compiled(left, right),
        warmup=0,
        repeats=repeats,
    )
    return (
        value,
        1_000.0 * first_seconds,
        1_000.0 * float(distribution.mean_seconds),
    )


def _measure_three(function, left, middle, right, repeats):
    compiled = eqx.filter_jit(function)
    value, first_seconds = measure_synchronized(lambda: compiled(left, middle, right))
    value, distribution = measure_repeated(
        lambda: compiled(left, middle, right),
        warmup=0,
        repeats=repeats,
    )
    return (
        value,
        1_000.0 * first_seconds,
        1_000.0 * float(distribution.mean_seconds),
    )


def run_algebra_benchmark(level: int, /, *, repeats: int = 5) -> AlgebraBenchmarkRecord:
    started = time.perf_counter()
    algebra = phx.metrix.algebra.CayleyDicksonAlgebraSpec(level)
    sparse = algebra.prepare_product(backend="sparse")
    dense = algebra.prepare_product(backend="dense")
    preparation = 1e3 * (time.perf_counter() - started)
    dimension = algebra.coordinate_dimension
    left = jnp.sin(jnp.arange(8 * dimension, dtype=float)).reshape((8, dimension))
    right = jnp.cos(jnp.arange(dimension, dtype=float))
    middle = jnp.sin(0.3 + jnp.arange(dimension, dtype=float))
    associator_left = jnp.sin(0.7 + jnp.arange(dimension, dtype=float))
    right_associator = right
    if algebra.properties.claim("associative").status == "disproven":
        witness = algebra.properties.claim("associative").witness
        positions = tuple(algebra.basis_index(label) for label in witness)
        basis = jnp.eye(dimension)
        associator_left, middle, right_associator = (
            basis[position] for position in positions
        )
    sparse_value, first, steady = _measure(sparse, left, right, repeats)
    dense_value = dense(left, right)
    defect = jnp.max(jnp.abs(sparse_value - dense_value))
    associator_value, associator_first, associator_steady = _measure_three(
        sparse.associator,
        associator_left,
        middle,
        right_associator,
        repeats,
    )
    associator_norm = jnp.linalg.norm(associator_value)
    associative_status = algebra.properties.claim("associative").status
    associator_classification_passed = (
        bool(associator_norm <= 1e-12)
        if associative_status == "proven"
        else bool(associator_norm > 1e-12)
        if associative_status == "disproven"
        else True
    )
    space = phx.linalg.AlgebraArraySpace((), algebra, dtype=jnp.float64)
    multiplier = jnp.sin(jnp.arange(dimension, dtype=float))
    action_value = jnp.cos(jnp.arange(dimension, dtype=float))
    action = phx.linalg.algebra_regular_action_operator(
        sparse,
        multiplier,
        space,
        side="left",
    )
    action_matrix = phx.linalg.materialize(
        action,
        phx.linalg.MaterializationPolicy(),
    )
    action_defect = jnp.max(
        jnp.abs(action.mv(action_value) - action_matrix @ action_value)
    )
    derivation_preparation = None
    derivation_dimension = None
    derivation_residual = None
    derivation_converged = None
    if level <= 3:
        derivation_started = time.perf_counter()
        derivations = phx.linalg.prepare_algebra_derivations(
            phx.linalg.plan_algebra_derivations(algebra)
        )
        derivation_preparation = 1e3 * (time.perf_counter() - derivation_started)
        derivation_dimension = int(derivations.dimension)
        derivation_residual = float(derivations.maximum_leibniz_residual)
        derivation_converged = bool(derivations.converged)
    finite = bool(
        jnp.all(jnp.isfinite(sparse_value))
        & jnp.all(jnp.isfinite(associator_value))
        & jnp.isfinite(defect)
        & jnp.isfinite(action_defect)
    )
    return AlgebraBenchmarkRecord(
        family=algebra.family,
        coordinate_dimension=dimension,
        product_terms=algebra.structure.term_count,
        plan_bytes=sparse.evidence.resource_evidence.plan_bytes,
        associative_status=associative_status,
        alternative_status=algebra.properties.claim("alternative").status,
        preparation_wall_ms=float(preparation),
        first_jit_ms=float(first),
        steady_wall_ms=float(steady),
        sparse_dense_defect=float(defect),
        associator_first_jit_ms=float(associator_first),
        associator_steady_wall_ms=float(associator_steady),
        associator_norm=float(associator_norm),
        associator_classification_passed=associator_classification_passed,
        regular_action_defect=float(action_defect),
        derivation_preparation_wall_ms=derivation_preparation,
        derivation_dimension=derivation_dimension,
        derivation_leibniz_residual=derivation_residual,
        derivation_converged=derivation_converged,
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
