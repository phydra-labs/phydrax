#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-side costs of Phydrax structural contracts.

Every measured operation is host-only metadata work: no JAX compiler metrics are
reported. Capacities vary the controlling sizes: contract fields per class
(plan compilation and warmed validation) and classes per cache (plan memory).
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import subprocess
import sys
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

import phydrax.typing as pt
from benchmarks._io import write_json_atomic
from benchmarks._runtime import capture_environment, DurationDistribution, measure_host
from phydrax import StrictModule


FIELD_COUNTS = (1, 4, 16, 64)
CLASS_COUNTS = (100, 1_000, 10_000)
_Basis: TypeAlias = Literal["nodal", "modal"]


class _NodeDim(pt.Dim, minimum=1):
    pass


def _host_samples(
    operation: Callable[[], object], repeats: int, /
) -> DurationDistribution:
    samples = []
    for _ in range(repeats):
        _, elapsed = measure_host(operation)
        samples.append(elapsed)
    return DurationDistribution(tuple(samples))


def _contract_class(field_count: int, /) -> type:
    namespace = {
        "__annotations__": {
            f"x{index}": pt.Float64[_NodeDim] for index in range(field_count)
        }
    }
    return dataclasses.dataclass(frozen=True)(
        type(f"Contract{field_count}", (), namespace)
    )


def _strict_class(field_count: int, /) -> type[StrictModule]:
    namespace = {
        "__annotations__": {f"x{index}": jax.Array for index in range(field_count)}
    }
    return type(f"Unopted{field_count}", (StrictModule,), namespace)


def _plans(field_counts: tuple[int, ...], repeats: int, /) -> list[dict[str, object]]:
    records = []
    for count in field_counts:
        values = tuple(jnp.zeros((8,)) for _ in range(count))
        cold = []
        for _ in range(repeats):
            instance = _contract_class(count)(*values)
            _, elapsed = measure_host(lambda instance=instance: pt.validate(instance))
            cold.append(elapsed)
        instance = _contract_class(count)(*values)
        pt.validate(instance)
        warm = _host_samples(lambda instance=instance: pt.validate(instance), repeats)
        strict = _strict_class(count)
        construction = _host_samples(
            lambda strict=strict, values=values: strict(*values), repeats
        )
        records.append(
            {
                "contract_fields": count,
                "cold_compile_and_validate": DurationDistribution(
                    tuple(cold)
                ).to_milliseconds_dict(),
                "warm_validate": warm.to_milliseconds_dict(),
                "unopted_strict_construction": construction.to_milliseconds_dict(),
            }
        )
    return records


def _parse_and_convert(repeats: int, /) -> dict[str, object]:
    device = jnp.zeros((64,))
    single = jnp.zeros((64,), dtype=jnp.float32)
    host_list = [float(index) for index in range(64)]
    operations: dict[str, Callable[[], object]] = {
        "parse_array": lambda: pt.parse(device, pt.Float64[_NodeDim], "values"),
        "parse_literal": lambda: pt.parse(np.str_("modal"), _Basis, "basis"),
        "parse_size": lambda: pt.parse(64, pt.Size[_NodeDim], "count"),
        "as_array_host_list": lambda: pt.as_array(
            host_list, pt.Float64[_NodeDim], "list"
        ),
        "as_array_device_exact": lambda: pt.as_array(
            device, pt.Float64[_NodeDim], "device"
        ),
        "as_array_device_cast": lambda: pt.as_array(single, pt.Float64[_NodeDim], "cast"),
        "as_host_array_device": lambda: pt.as_host_array(
            device, pt.HostFloat64[_NodeDim], "host"
        ),
    }
    records = {}
    for name, operation in operations.items():
        jax.block_until_ready(operation())
        records[name] = _host_samples(
            lambda operation=operation: jax.block_until_ready(operation()), repeats
        ).to_milliseconds_dict()
    return records


def _cache_memory(class_counts: tuple[int, ...], /) -> list[dict[str, object]]:
    records = []
    value = jnp.zeros((1,))
    for count in class_counts:
        classes = [_contract_class(1) for _ in range(count)]
        instances = [cls(value) for cls in classes]
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        for instance in instances:
            pt.validate(instance)
        after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        records.append(
            {
                "classes": count,
                "plan_bytes": after - before,
                "bytes_per_class": (after - before) / count,
            }
        )
    return records


_IMPORT_LINE = re.compile(r"import time:\s+\d+\s+\|\s+(\d+)\s+\|\s+(\S+)")


def _import_microseconds(module: str, /) -> dict[str, int]:
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=True,
    )
    cumulative = {}
    for line in result.stderr.splitlines():
        match = _IMPORT_LINE.match(line)
        if match is not None:
            # The first entry is the nested import; a later repeat is the outer request.
            cumulative.setdefault(match.group(2).strip(), int(match.group(1)))
    return {
        name: cumulative[name]
        for name in (
            "phydrax._dtype_names",
            "phydrax._typing_forms",
            "phydrax._typing_plan",
            "phydrax.typing",
        )
        if name in cumulative
    }


def run(*, quick: bool, repeats: int) -> dict[str, object]:
    field_counts = FIELD_COUNTS[:2] if quick else FIELD_COUNTS
    class_counts = CLASS_COUNTS[:1] if quick else CLASS_COUNTS
    return {
        "benchmark": "typing_structural_validation",
        "environment": capture_environment().to_dict(),
        "units": {
            "durations": "milliseconds",
            "memory": "bytes",
            "imports": "microseconds",
        },
        "plans": _plans(field_counts, repeats),
        "boundaries": _parse_and_convert(repeats),
        "plan_cache_memory": _cache_memory(class_counts),
        "import_cumulative_microseconds": _import_microseconds("phydrax.typing"),
        "compiler_metrics": "not applicable: structural checks are host-only metadata work",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/typing_structural_validation.json"),
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    write_json_atomic(
        arguments.output, run(quick=arguments.quick, repeats=arguments.repeats)
    )


if __name__ == "__main__":
    main()
