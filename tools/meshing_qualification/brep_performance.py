#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Isolated BRep containment, moments, and orientation-audit comparisons."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial
from time import perf_counter

import jax
import numpy as np

from phydrax.meshing.qualification import (
    benchmark_box,
    corpus_route_points,
    corpus_route_query,
    qualify_brep_containment,
    qualify_brep_moments,
    qualify_brep_orientation_audit,
    trimmed_polynomial_nurbs_box,
)
from tools.meshing_qualification._runtime import peak_resident_bytes, StageLedger


def _measure_samples[T](
    call: Callable[[], T], /, *, repeats: int
) -> tuple[T, dict[str, object]]:
    """Keep clean samples separate from profiled compilation/accounting processes."""

    def synchronized_call() -> T:
        return jax.block_until_ready(call())

    if os.environ.get("PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS") == "1":
        diagnostics: dict[str, object] = {}
        result = StageLedger(diagnostics).run("accounting_diagnostic", synchronized_call)
        return result, {
            "headline_timing_available": False,
            "diagnostic_measurements": diagnostics["stage_measurements"],
            "diagnostic_timing_is_headline": False,
            "peak_resident_bytes": peak_resident_bytes(),
        }
    started = perf_counter()
    result = synchronized_call()
    first_execution = perf_counter() - started
    warm: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        result = synchronized_call()
        warm.append(perf_counter() - started)
    return result, {
        "headline_timing_available": True,
        "first_execution_seconds": first_execution,
        "warm_samples_seconds": warm,
        "warm_median_seconds": float(np.median(np.asarray(warm, dtype=np.float64))),
        "peak_resident_bytes": peak_resident_bytes(),
        "diagnostic_measurements": None,
        "diagnostic_not_run_reason": "accounting diagnostics require a separate process",
    }


def benchmark_brep_containment(resolution: int, repeats: int) -> dict[str, object]:
    """Fixed corpus identities, with a cubic sweep of separated query points."""
    campaigns: dict[str, object] = {}
    for route in ("sphere", "revolution", "affine", "nurbs"):
        query = corpus_route_query(route)
        band = corpus_route_points(query, route)
        count = resolution**3
        outside = np.empty((count, 3), dtype=np.float64)
        outside[:, 0] = np.linspace(10, 11, count, dtype=np.float64)
        outside[:, 1:] = 10
        points = jax.device_put(np.concatenate((band, outside)))
        campaigns[route] = qualify_brep_containment(
            query, points, sampling=partial(_measure_samples, repeats=repeats)
        )
    return {"resolution": resolution, "repeats": repeats, "campaigns": campaigns}


def benchmark_brep_moments(resolution: int, repeats: int) -> dict[str, object]:
    result = qualify_brep_moments(
        trimmed_polynomial_nurbs_box(),
        resolution,
        sampling=partial(_measure_samples, repeats=repeats),
    )
    return {"resolution": resolution, "repeats": repeats, **result}


def benchmark_brep_orientation_audit(resolution: int, repeats: int) -> dict[str, object]:
    result = qualify_brep_orientation_audit(
        benchmark_box(),
        sampling=partial(_measure_samples, repeats=repeats),
    )
    return {"resolution": resolution, "repeats": repeats, **result}
