#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cold-import, configuration-resolution, and numerical-scope phases.

Each import scenario runs in fresh interpreters so every sample is a cold start;
`--package-root` selects the source tree a child imports, which measures another
checkout under the same environment. Resolution and scope costs are host-only
and measured in this process outside any numerical iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, TypeAlias

import jax.numpy as jnp

from benchmarks._io import write_json_atomic
from benchmarks._memory import process_peak_rss_bytes
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    DurationDistribution,
    measure_repeated,
)
from phydrax import config


ImportScenario: TypeAlias = Literal[
    "jax", "root", "configuration", "geometry", "lazy-geometry", "complete"
]

# Each child prints one JSON line: elapsed import seconds, peak RSS, and whether
# a JAX backend exists afterwards (`null` when Phydrax was not imported).
_CHILD_PRELUDE: Final = """
import json, time
started = time.perf_counter()
"""
_CHILD_SCENARIOS: Final[dict[ImportScenario, str]] = {
    "jax": "import jax\nready = None\n",
    "root": "import phydrax\nready = None\n",
    "configuration": (
        "from phydrax import config\nready = config.snapshot().backends_initialized\n"
    ),
    "geometry": (
        "import phydrax.geometry\n"
        "ready = phydrax.config.snapshot().backends_initialized\n"
    ),
    "lazy-geometry": (
        "import phydrax\n"
        "phydrax.geometry\n"
        "ready = phydrax.config.snapshot().backends_initialized\n"
    ),
    "complete": (
        "import importlib\n"
        "from phydrax._package_modules import PUBLIC_MODULES\n"
        "for name in PUBLIC_MODULES:\n"
        "    importlib.import_module(f'phydrax.{name}')\n"
        "ready = importlib.import_module('phydrax').config.snapshot().backends_initialized\n"
    ),
}
_CHILD_EPILOGUE: Final = """
elapsed = time.perf_counter() - started
import sys
from benchmarks._memory import process_peak_rss_bytes
package = sys.modules.get("phydrax")
print(json.dumps({"seconds": elapsed, "rss_bytes": process_peak_rss_bytes(), "backends_initialized": ready, "package_file": None if package is None else package.__file__}))
"""


@dataclass(frozen=True)
class _Solver:
    tolerance: float = 1.0e-6
    maximum_iterations: int = 100
    method: Literal["cg", "gmres"] = "cg"


@dataclass(frozen=True)
class _Study:
    __config_kind__: ClassVar[str] = "benchmark.configuration-study"

    label: str = "baseline"
    solver: _Solver = field(default_factory=_Solver)
    numerics: config.NumericsConfig = field(default_factory=config.NumericsConfig)
    weights: tuple[float, ...] = ()


def _cold_import(
    scenario: ImportScenario, package_root: Path, repo_root: Path, repeats: int, /
) -> dict[str, Any]:
    source = _CHILD_PRELUDE + _CHILD_SCENARIOS[scenario] + _CHILD_EPILOGUE
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(package_root), str(repo_root)))
    samples: list[float] = []
    peaks: list[int] = []
    ready: set[bool | None] = set()
    for _ in range(repeats):
        # `-P` keeps the working directory off `sys.path`, so `phydrax` resolves
        # from `package_root` rather than from the benchmark's own checkout.
        completed = subprocess.run(
            (sys.executable, "-P", "-c", source),
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        observation = json.loads(completed.stdout.strip().splitlines()[-1])
        imported = observation["package_file"]
        if imported is not None and not Path(imported).resolve().is_relative_to(
            package_root
        ):
            raise RuntimeError(
                f"{scenario}: imported phydrax from {imported}, not {package_root}."
            )
        samples.append(observation["seconds"])
        peaks.append(observation["rss_bytes"])
        ready.add(observation["backends_initialized"])
    if len(ready) != 1:
        raise RuntimeError(f"{scenario}: backend readiness differed between cold starts.")
    return {
        "scenario": scenario,
        "import": DurationDistribution(tuple(samples)).to_seconds_dict(),
        "peak_rss_bytes": max(peaks),
        "backends_initialized": ready.pop(),
    }


def _resolution(override_count: int, repeats: int, /) -> dict[str, Any]:
    overrides = tuple(
        ["solver.method=gmres", "numerics.enable_x64=True"]
        + [f"weights={[float(index) for index in range(override_count)]}"]
    )
    resolved, timing = measure_repeated(
        lambda: config.resolve(
            _Study, overrides=overrides, values={"label": "study"}, environment={}
        ),
        warmup=1,
        repeats=repeats,
    )
    return {
        "weight_count": override_count,
        "resolve": timing.to_seconds_dict(),
        "configuration_id": resolved.configuration_id,
    }


def _scope_entry(repeats: int, /) -> dict[str, Any]:
    settings = config.NumericsConfig(dtype_promotion="strict", rank_promotion="raise")

    def enter() -> bool:
        with config.context(settings):
            return config.x64_enabled()

    _, timing = measure_repeated(enter, warmup=1, repeats=repeats)
    values = jnp.ones((4,))
    with config.context(settings):
        scoped_dtype = str((values + values).dtype)
    return {"context": timing.to_seconds_dict(), "scoped_dtype": scoped_dtype}


def run(
    scenarios: Sequence[ImportScenario],
    package_root: Path,
    repeats: int,
    weight_counts: Sequence[int],
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    imports = [
        _cold_import(scenario, package_root, root, repeats) for scenario in scenarios
    ]
    return {
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            root,
            Path(__file__).resolve(),
            (
                "scenario",
                "import",
                "peak_rss_bytes",
                "backends_initialized",
                "weight_count",
                "resolve",
                "configuration_id",
                "context",
                "scoped_dtype",
            ),
        ).to_dict(),
        "package_root": str(package_root),
        "imports": imports,
        "resolution": [_resolution(count, repeats) for count in weight_counts],
        "scope": _scope_entry(repeats),
        "rss_bytes": process_peak_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/configuration.json")
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Source tree whose `phydrax` the cold-import children import.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(_CHILD_SCENARIOS),
        help="Import scenario; repeat to select several (default: all).",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    scenarios: tuple[ImportScenario, ...] = (
        tuple(arguments.scenario) if arguments.scenario else tuple(_CHILD_SCENARIOS)
    )
    repeats = 2 if arguments.smoke else arguments.repeats
    weight_counts = (1, 16) if arguments.smoke else (1, 16, 256)
    write_json_atomic(
        arguments.output,
        run(scenarios, arguments.package_root.resolve(), repeats, weight_counts),
    )


if __name__ == "__main__":
    main()
