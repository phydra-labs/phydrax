#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import math
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, UTC
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import jax


MetricCategory = Literal[
    "accuracy",
    "calibration",
    "convergence",
    "performance",
    "diagnostic",
]


@dataclass(frozen=True)
class Metric:
    """One finite benchmark measurement and its optional inclusive release gate."""

    value: float
    category: MetricCategory
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        value = float(self.value)
        minimum = None if self.minimum is None else float(self.minimum)
        maximum = None if self.maximum is None else float(self.maximum)
        if not math.isfinite(value):
            raise ValueError("Benchmark metric values must be finite.")
        if minimum is not None and not math.isfinite(minimum):
            raise ValueError("Metric minimum must be finite or None.")
        if maximum is not None and not math.isfinite(maximum):
            raise ValueError("Metric maximum must be finite or None.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("Metric minimum cannot exceed maximum.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def gated(self) -> bool:
        return self.minimum is not None or self.maximum is not None

    @property
    def passed(self) -> bool:
        lower_ok = self.minimum is None or self.value >= self.minimum
        upper_ok = self.maximum is None or self.value <= self.maximum
        return lower_ok and upper_ok

    def as_dict(self) -> dict[str, Any]:
        gate = None
        if self.gated:
            gate = {
                "minimum": self.minimum,
                "maximum": self.maximum,
                "inclusive": True,
            }
        return {
            "value": self.value,
            "unit": self.unit,
            "category": self.category,
            "description": self.description,
            "gate": gate,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ScenarioResult:
    """Serializable outcome of one deterministic scientific benchmark scenario."""

    name: str
    description: str
    seed: int
    metrics: dict[str, Metric] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Scenario name must be non-empty.")
        if len(self.metrics) != len(set(self.metrics)):
            raise ValueError("Scenario metric names must be unique.")
        if (self.error_type is None) != (self.error_message is None):
            raise ValueError("Scenario error type and message must be set together.")
        _validate_json_value(self.metadata, path="metadata")

    @property
    def passed(self) -> bool:
        return self.error_type is None and all(
            metric.passed for metric in self.metrics.values()
        )

    @property
    def failures(self) -> tuple[str, ...]:
        failures = tuple(
            name for name, metric in self.metrics.items() if not metric.passed
        )
        if self.error_type is not None:
            return (*failures, "scenario_error")
        return failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "passed": self.passed,
            "failures": list(self.failures),
            "metrics": {
                name: metric.as_dict() for name, metric in sorted(self.metrics.items())
            },
            "metadata": self.metadata,
            "error": (
                None
                if self.error_type is None
                else {"type": self.error_type, "message": self.error_message}
            ),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Versioned aggregate report for the complete PhydraX UQ benchmark matrix."""

    profile: str
    root_seed: int
    started_at_utc: str
    duration_seconds: float
    configuration: dict[str, Any]
    environment: dict[str, Any]
    scenarios: tuple[ScenarioResult, ...]
    schema_version: str = "1.0"
    suite: str = "phydrax-uq-benchmark-matrix"

    def __post_init__(self) -> None:
        if not self.profile:
            raise ValueError("Benchmark profile must be non-empty.")
        if not math.isfinite(float(self.duration_seconds)) or self.duration_seconds < 0.0:
            raise ValueError("Benchmark duration must be finite and non-negative.")
        names = tuple(scenario.name for scenario in self.scenarios)
        if not names or len(names) != len(set(names)):
            raise ValueError("A benchmark report requires uniquely named scenarios.")
        _validate_json_value(self.configuration, path="configuration")
        _validate_json_value(self.environment, path="environment")

    @property
    def passed(self) -> bool:
        return all(scenario.passed for scenario in self.scenarios)

    @property
    def summary(self) -> dict[str, Any]:
        category_counts: dict[str, dict[str, int]] = {}
        metrics_by_name: dict[str, list[Metric]] = {}
        for scenario in self.scenarios:
            for name, metric in scenario.metrics.items():
                metrics_by_name.setdefault(name, []).append(metric)
                counts = category_counts.setdefault(
                    metric.category,
                    {"total": 0, "passed": 0, "failed": 0, "gated": 0},
                )
                counts["total"] += 1
                counts["passed" if metric.passed else "failed"] += 1
                counts["gated"] += int(metric.gated)
        return {
            "passed": self.passed,
            "scenario_count": len(self.scenarios),
            "scenarios_passed": sum(scenario.passed for scenario in self.scenarios),
            "scenarios_failed": sum(not scenario.passed for scenario in self.scenarios),
            "metric_categories": dict(sorted(category_counts.items())),
            "metric_aggregates": {
                name: {
                    "count": len(metrics),
                    "minimum": min(item.value for item in metrics),
                    "maximum": max(item.value for item in metrics),
                    "mean": sum(item.value for item in metrics) / len(metrics),
                    "failed": sum(not item.passed for item in metrics),
                    "categories": sorted({item.category for item in metrics}),
                    "units": sorted(
                        {item.unit for item in metrics if item.unit is not None}
                    ),
                }
                for name, metrics in sorted(metrics_by_name.items())
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "profile": self.profile,
            "root_seed": self.root_seed,
            "started_at_utc": self.started_at_utc,
            "duration_seconds": float(self.duration_seconds),
            "passed": self.passed,
            "summary": self.summary,
            "configuration": self.configuration,
            "environment": self.environment,
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return (
            json.dumps(
                self.as_dict(),
                indent=indent,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

    def write_json(self, path: str | os.PathLike[str], /) -> Path:
        """Atomically write the report so interrupted runs never leave partial JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(self.to_json(), encoding="utf-8")
        temporary.replace(destination)
        return destination


def metric(
    value: Any,
    category: MetricCategory,
    /,
    *,
    unit: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    description: str = "",
) -> Metric:
    """Convert a scalar JAX/Python measurement into a strict report metric."""
    return Metric(
        value=float(value),
        category=category,
        unit=unit,
        minimum=minimum,
        maximum=maximum,
        description=description,
    )


def collect_environment() -> dict[str, Any]:
    """Collect reproducibility metadata without invoking source-control commands."""

    def package_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    devices = [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in jax.devices()
    ]
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "devices": devices,
        "package_versions": {
            name: package_version(name)
            for name in ("phydrax", "jax", "equinox", "blackjax", "laplax")
        },
    }


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _validate_json_value(value: Any, *, path: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{path} must contain only finite JSON values.") from error


__all__ = [
    "BenchmarkReport",
    "Metric",
    "MetricCategory",
    "ScenarioResult",
    "collect_environment",
    "metric",
    "utc_now_iso",
]
