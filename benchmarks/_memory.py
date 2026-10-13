#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Memory-scaling evidence and adoption gates for workspace-bounded routes.

Compiler temporary bytes from ``Compiled.memory_analysis()`` are fixed for one program
and environment, so they stay valid under CPU contention; wall-clock samples do not.
Host preparation is measured with ``tracemalloc``, to which NumPy reports its data
buffers. Process peak RSS is observed evidence only and never a bound.
"""

from __future__ import annotations

import math
import resource
import sys
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal


type MemoryGateDecision = Literal["adopt", "retain-baseline"]


def host_allocation_peak[T](operation: Callable[[], T], /) -> tuple[T, int]:
    """Run host work and return its peak traced allocation above the starting level."""
    if tracemalloc.is_tracing():
        raise RuntimeError("Host allocation measurement cannot nest inside tracing.")
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        value = operation()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, peak - baseline


def process_peak_rss_bytes() -> int:
    """Return the observed process high-water resident set size in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _validated_series(
    capacities: Sequence[int], values: Sequence[int], name: str, /
) -> None:
    if len(capacities) != len(values):
        raise ValueError(f"{name} must have one value per capacity.")
    if len(capacities) < 2:
        raise ValueError("Memory scaling requires at least two capacities.")
    if any(capacity < 1 for capacity in capacities) or any(
        later <= earlier for earlier, later in pairwise(capacities)
    ):
        raise ValueError("Capacities must be positive and strictly increasing.")
    if any(value < 0 for value in values):
        raise ValueError(f"{name} must be nonnegative byte counts.")


def fitted_scaling_exponent(capacities: Sequence[int], values: Sequence[int], /) -> float:
    """Least-squares log-log slope of bytes against the controlling capacity."""
    _validated_series(capacities, values, "Byte counts")
    if any(value == 0 for value in values):
        raise ValueError("A scaling exponent requires positive byte counts.")
    x = [math.log(capacity) for capacity in capacities]
    y = [math.log(value) for value in values]
    x_mean = math.fsum(x) / len(x)
    y_mean = math.fsum(y) / len(y)
    numerator = math.fsum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.fsum((a - x_mean) ** 2 for a in x)
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class MemoryGatePolicy:
    """Adoption rule for a rewrite whose benefit depends on compiler fusion."""

    maximum_candidate_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.maximum_candidate_fraction <= 1.0:
            raise ValueError("The candidate fraction must lie in (0, 1].")


@dataclass(frozen=True, slots=True)
class MemoryGateResult:
    """Largest-capacity comparison that decides whether a candidate route ships."""

    capacity: int
    baseline_bytes: int
    candidate_bytes: int
    maximum_candidate_fraction: float
    decision: MemoryGateDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "baseline_bytes": self.baseline_bytes,
            "candidate_bytes": self.candidate_bytes,
            "maximum_candidate_fraction": self.maximum_candidate_fraction,
            "decision": self.decision,
        }


def evaluate_memory_gate(
    capacities: Sequence[int],
    baseline_bytes: Sequence[int],
    candidate_bytes: Sequence[int],
    policy: MemoryGatePolicy,
    /,
) -> MemoryGateResult:
    """Adopt only when the candidate meets the fraction at the largest capacity.

    Smaller capacities are validated but do not decide: a fused baseline can look
    expensive at small sizes while matching the candidate where memory matters.
    """
    _validated_series(capacities, baseline_bytes, "Baseline bytes")
    _validated_series(capacities, candidate_bytes, "Candidate bytes")
    baseline = baseline_bytes[-1]
    candidate = candidate_bytes[-1]
    adopt = baseline > 0 and candidate <= policy.maximum_candidate_fraction * baseline
    return MemoryGateResult(
        capacity=capacities[-1],
        baseline_bytes=baseline,
        candidate_bytes=candidate,
        maximum_candidate_fraction=policy.maximum_candidate_fraction,
        decision="adopt" if adopt else "retain-baseline",
    )
