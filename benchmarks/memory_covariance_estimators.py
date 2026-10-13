"""Covariance fitting workspace with requested dense model outputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._memory import fitted_scaling_exponent, process_peak_rss_bytes
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax.ml import MLBatch
from phydrax.ml.covariance import CovarianceModel, RobustCovariance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sizes = (4, 8, 16) if args.smoke else (16, 32, 64)
    records: list[dict[str, Any]] = []
    recipe = RobustCovariance(max_iterations=2, tolerance=0.0, regularization=0.03)

    def run(data: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        model = recipe.fit_batch(MLBatch(data)).as_trainable()
        if not isinstance(model, CovarianceModel):
            raise TypeError("Expected covariance geometry.")
        return model.covariance, model.precision, model.log_determinant

    for dimension in sizes:
        data = jnp.asarray(
            np.random.default_rng(818).normal(size=(2 * dimension, dimension)),
            dtype=jnp.float64,
        )
        compiled, timing = measure_lower_and_compile(
            lambda: jax.jit(run).lower(data), lambda lowered: lowered.compile()
        )
        result = compiled(data)
        jax.block_until_ready(result)
        _, samples = measure_repeated(lambda: compiled(data), warmup=1, repeats=3)
        records.append(
            {
                "dimension": dimension,
                "sample_count": 2 * dimension,
                "compiler": asdict(
                    compiler_evidence(
                        compiled.cost_analysis(),
                        compiled.memory_analysis(),
                        source="compiled",
                    )
                ),
                "compilation": asdict(timing),
                "execution": samples.to_dict(),
                "retained_bytes": logical_array_bytes(result),
                "precision_error": float(
                    np.max(np.abs(np.asarray(result[1]) - np.linalg.inv(result[0])))
                ),
                "logdet_error": float(
                    np.abs(np.asarray(result[2]) - np.linalg.slogdet(result[0])[1])
                ),
            }
        )
        jax.clear_caches()
    root = Path(__file__).resolve().parents[1]
    suffix = "_baseline" if args.baseline else ""
    write_json_atomic(
        root / f"benchmarks/memory_covariance_estimators{suffix}.json",
        {
            "scope": "robust covariance; baseline captured before inverse-to-action migration, after determinant factor reuse",
            "identity": capture_benchmark_identity(
                root, Path(__file__), ("compiler", "retained_bytes")
            ).to_dict(),
            "environment": capture_environment().to_dict(),
            "records": records,
            "retained_exponent": fitted_scaling_exponent(
                sizes, [record["retained_bytes"] for record in records]
            ),
            "peak_rss_bytes": process_peak_rss_bytes(),
        },
    )


if __name__ == "__main__":
    main()
