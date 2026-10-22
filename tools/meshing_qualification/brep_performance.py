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

from phydrax.geometry.brep import BRepQueryResourceError
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
    """Retain fixed-corpus measurements and declared-limit refusal evidence."""
    campaigns: dict[str, object] = {}
    refusal: dict[str, object] | None = None
    for route in ("sphere", "revolution", "affine", "nurbs"):
        if refusal is not None:
            campaigns[route] = {
                "status": "not-attempted",
                "reason": "containment-case-stopped-after-resource-refusal",
                "timing_is_headline": False,
                "seconds": None,
            }
            continue
        query = corpus_route_query(route)
        band = corpus_route_points(query, route)
        count = resolution**3
        outside = np.empty((count, 3), dtype=np.float64)
        outside[:, 0] = np.linspace(10, 11, count, dtype=np.float64)
        outside[:, 1:] = 10
        points = jax.device_put(np.concatenate((band, outside)))
        try:
            result = qualify_brep_containment(
                query, points, sampling=partial(_measure_samples, repeats=repeats)
            )
        except BRepQueryResourceError as error:
            faces = query.solid_faces[0]
            refusal = {
                "route": route,
                "resource": error.resource,
                "requested": error.requested,
                "remaining": error.remaining,
                "point_count": points.shape[0],
                "solid_count": 1,
                "face_count": len(faces),
                "coedge_count": sum(
                    len(query.faces[face].coedge_edges) for face in faces
                ),
                "model_id": query.model.model_id,
                "policy_id": query.policy.policy_id,
                "policy_limits": {
                    "maximum_operations": query.policy.maximum_operations,
                    "maximum_points": query.policy.maximum_points,
                    "maximum_scratch_bytes": query.policy.maximum_scratch_bytes,
                    "maximum_subdivisions": query.policy.maximum_subdivisions,
                    "maximum_depth": query.policy.maximum_depth,
                },
            }
            campaigns[route] = {
                "status": "resource-refused",
                "refusal": refusal,
                "seconds": None,
                "timing_is_headline": False,
                "headline_timing_available": False,
                "threshold_met": False,
                "decisions_identical": None,
                "baseline": None,
                "boundary_filter": None,
            }
        else:
            campaigns[route] = {"status": "completed", **result}
    return {
        "resolution": resolution,
        "repeats": repeats,
        "status": "completed" if refusal is None else "resource-refused",
        "seconds": None,
        "headline_timing_available": (
            refusal is None
            and os.environ.get("PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS") != "1"
        ),
        "threshold_met": False if refusal is not None else None,
        "timing_is_headline": (
            refusal is None
            and os.environ.get("PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS") != "1"
        ),
        "refusal": refusal,
        "campaigns": campaigns,
    }


def benchmark_brep_containment_campaign(
    resolutions: tuple[int, ...],
    repeats: int,
    /,
    *,
    run_case: Callable[[int], dict[str, object]],
) -> list[dict[str, object]]:
    """Preserve completed cells without retrying a refused or larger request."""
    results: list[dict[str, object]] = []
    refused_resolution: int | None = None
    for resolution in resolutions:
        if refused_resolution is not None and resolution >= refused_resolution:
            results.append(
                {
                    "resolution": resolution,
                    "repeats": repeats,
                    "status": "not-attempted",
                    "reason": "same-or-larger-containment-request-exceeds-declared-policy",
                    "refused_resolution": refused_resolution,
                    "seconds": None,
                    "timing_is_headline": False,
                    "headline_timing_available": False,
                    "threshold_met": False,
                }
            )
            continue
        result = run_case(resolution)
        results.append(result)
        if result["status"] == "resource-refused":
            refused_resolution = resolution
    return results


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
