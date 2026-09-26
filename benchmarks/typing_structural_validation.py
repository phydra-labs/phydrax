#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-side costs of Phydrax structural contracts.

Structural checks are host-only metadata work. The benchmark reports cold first
construction (plan compilation), warmed construction of opted-in modules against
unchecked twins, explicit validation, tracing and lowering time with the number
of traced equations, boundary parsing and conversion, plan memory, and import
time. Capacities vary contract fields per class and classes per cache.
"""

from __future__ import annotations

import argparse
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
from phydrax._typing_plan import _CLASS_PLANS, class_plan


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


def _module_class(field_count: int, *, opted: bool) -> type[StrictModule]:
    annotations: dict[str, object] = {
        f"x{index}": pt.Float64[_NodeDim] if opted else jax.Array
        for index in range(field_count)
    }
    namespace: dict[str, object] = {"__annotations__": annotations}
    if opted:
        namespace["__strict_contract__"] = True
    name = f"{'Opted' if opted else 'Unopted'}{field_count}"
    return type(name, (StrictModule,), namespace)


def _plans(field_counts: tuple[int, ...], repeats: int, /) -> list[dict[str, object]]:
    records = []
    for count in field_counts:
        values = tuple(jnp.zeros((8,)) for _ in range(count))
        cold = []
        for _ in range(repeats):
            opted = _module_class(count, opted=True)
            _, elapsed = measure_host(lambda opted=opted: opted(*values))
            cold.append(elapsed)
        opted = _module_class(count, opted=True)
        unopted = _module_class(count, opted=False)
        instance = opted(*values)
        plans_before = len(_CLASS_PLANS)
        warm_opted = _host_samples(lambda opted=opted: opted(*values), repeats)
        warm_unopted = _host_samples(lambda unopted=unopted: unopted(*values), repeats)
        validate = _host_samples(lambda instance=instance: pt.validate(instance), repeats)
        records.append(
            {
                "contract_fields": count,
                "cold_first_construction": DurationDistribution(
                    tuple(cold)
                ).to_milliseconds_dict(),
                "warm_opted_construction": warm_opted.to_milliseconds_dict(),
                "warm_unopted_construction": warm_unopted.to_milliseconds_dict(),
                "warm_validate": validate.to_milliseconds_dict(),
                "plans_compiled_by_warm_constructions": len(_CLASS_PLANS) - plans_before,
                "trace": _trace_and_lower(opted, unopted, values, repeats),
            }
        )
    return records


def _trace_and_lower(
    opted: type[StrictModule],
    unopted: type[StrictModule],
    values: tuple[jax.Array, ...],
    repeats: int,
    /,
) -> dict[str, object]:
    record: dict[str, object] = {}
    for label, cls in (("opted", opted), ("unopted", unopted)):

        def body(*arrays: jax.Array, cls: type[StrictModule] = cls) -> object:
            return cls(*arrays)

        jaxpr = jax.make_jaxpr(body)(*values)
        record[f"{label}_equations"] = len(jaxpr.jaxpr.eqns)
        record[f"{label}_trace"] = _host_samples(
            lambda body=body: jax.make_jaxpr(body)(*values), repeats
        ).to_milliseconds_dict()
        record[f"{label}_lower"] = _host_samples(
            lambda body=body: jax.jit(body).lower(*values), repeats
        ).to_milliseconds_dict()
    return record


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
    for count in class_counts:
        classes = [_module_class(1, opted=True) for _ in range(count)]
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        for cls in classes:
            class_plan(cls)
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
