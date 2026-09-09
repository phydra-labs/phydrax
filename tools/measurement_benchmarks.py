#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Portable timing evidence for quantity-field preparation and comparison."""

from __future__ import annotations

import argparse
import json

import equinox as eqx
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_host,
    measure_repeated,
)


def benchmark(*, smoke: bool) -> dict[str, object]:
    size = 1_024 if smoke else 100_000
    repeats = 2 if smoke else 10
    support = phx.measurement.IndexSampleSupport((size,), ("sample",))
    quantity = phx.measurement.QuantitySpec(
        "benchmark", "signal", "signal", phx.units.ONE, "benchmark.signal"
    )
    values = np.linspace(0.0, 1.0, size)
    uncertainty = phx.measurement.IndependentStandardUncertainty(
        np.full((size,), 0.1), phx.units.ONE
    )
    field, construction_seconds = measure_host(
        lambda: phx.measurement.QuantityField(
            "benchmark-observation",
            quantity,
            phx.measurement.ValueLayout.scalar(),
            support,
            phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.POINT),
            values,
            uncertainty=uncertainty,
        )
    )
    observed = field.prepare()
    comparison = phx.observation.MeasurementComparisonPlan(observed)
    compiled = eqx.filter_jit(comparison.evaluate)
    result, distribution = measure_repeated(
        lambda: compiled(observed), warmup=1, repeats=repeats
    )
    if not bool(result.successful) or float(result.quadratic) != 0.0:
        raise RuntimeError("Measurement benchmark failed exact self-comparison.")
    return {
        "environment": capture_environment().to_dict(),
        "configuration": {"sample_count": size, "repeats": repeats},
        "host_construction_seconds": construction_seconds,
        "prepared_logical_bytes": logical_array_bytes(observed),
        "comparison": distribution.to_milliseconds_dict(),
        "successful": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(benchmark(smoke=arguments.smoke), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
