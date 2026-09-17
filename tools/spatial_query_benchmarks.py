#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

import equinox as eqx
import jax.numpy as jnp

from benchmarks._runtime import (
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax._bvh import build_packed_bvh, point_select_leaf_items


def _case(node_count: int, query_count: int, repeats: int) -> dict[str, object]:
    item_count = max(1, (node_count + 1) // 2)
    lower = jnp.stack(
        (
            jnp.arange(item_count, dtype=float),
            jnp.zeros(item_count),
            jnp.zeros(item_count),
        ),
        axis=-1,
    )
    upper = lower + jnp.asarray((0.75, 1.0, 1.0))
    bvh = build_packed_bvh(lower, upper, 0.5 * (lower + upper), leaf_size=2)
    points = jnp.stack(
        (
            jnp.linspace(0.25, item_count - 0.25, query_count),
            jnp.full((query_count,), 0.5),
            jnp.full((query_count,), 0.5),
        ),
        axis=-1,
    )

    def query(values):
        return point_select_leaf_items(
            values,
            bvh=bvh,
            maximum_candidates=4,
            query_batch_capacity=64,
        )

    compiled_query = eqx.filter_jit(query)
    executable, compilation = measure_lower_and_compile(
        lambda: compiled_query.lower(points),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: executable(points),
        warmup=1,
        repeats=repeats,
    )
    return {
        "item_count": item_count,
        "node_count": int(bvh.left.shape[0]),
        "query_count": query_count,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "compiler": asdict(
            compiler_evidence(
                executable.compiled.cost_analysis(),
                executable.compiled.memory_analysis(),
                source="jax-compiled-executable",
            )
        ),
        "output_bytes": logical_array_bytes(result),
        "complete": bool(jnp.all(result[2])),
    }


def run_spatial_query_scaling(
    node_counts: Sequence[int] = (127, 511, 2047),
    /,
    *,
    query_count: int = 512,
    repeats: int = 5,
) -> dict[str, object]:
    return {
        "campaign": "packed-bvh-query-scaling",
        "rows": [
            _case(int(node_count), int(query_count), int(repeats))
            for node_count in node_counts
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", nargs="+", type=int, default=(127, 511, 2047))
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_spatial_query_scaling(
                tuple(arguments.nodes),
                query_count=arguments.queries,
                repeats=arguments.repeats,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["run_spatial_query_scaling"]
