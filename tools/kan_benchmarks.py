#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax._trainable import partition_trainable


def _block(tree: Any) -> Any:
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready() if eqx.is_array(leaf) else leaf,
        tree,
    )


def _benchmark(
    function: Callable[..., Any],
    *arguments: Any,
    repeats: int,
) -> dict[str, float]:
    compiled = eqx.filter_jit(function)
    started = time.perf_counter()
    _block(compiled(*arguments))
    compile_and_first_ms = 1e3 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repeats):
        _block(compiled(*arguments))
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats
    return {
        "compile_and_first_ms": compile_and_first_ms,
        "steady_ms": steady_ms,
    }


def _parameter_count(model: phx.nn.KAN) -> int:
    trainable, _ = partition_trainable(model)
    return sum(int(leaf.size) for leaf in jax.tree.leaves(trainable))


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare equal-width orthogonal-polynomial and B-spline KANs."
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    return parser


def main() -> None:
    arguments = _parse().parse_args()
    coefficient_count = 11
    bases = {
        "orthogonal_degree_10": phx.nn.OrthogonalPolynomialEdgeBasis(
            degree=coefficient_count - 1
        ),
        "bspline_degree_3_intervals_8": phx.nn.BSplineEdgeBasis(
            degree=3,
            num_intervals=coefficient_count - 3,
        ),
    }
    inputs = jnp.linspace(-0.8, 0.8, 8)
    records: dict[str, Any] = {}

    for name, basis in bases.items():
        model = phx.nn.KAN(
            in_size=8,
            out_size=8,
            width_size=arguments.width,
            depth=arguments.depth,
            edge_basis=basis,
            scale_mode="none",
            skip_connection=False,
            scan=True,
            key=jr.key(0),
        )
        records[name] = {
            "coefficient_count_per_edge": basis.coefficient_count,
            "active_coefficients_per_query": (
                basis.degree + 1
                if isinstance(basis, phx.nn.BSplineEdgeBasis)
                else basis.coefficient_count
            ),
            "parameter_count": _parameter_count(model),
            "forward": _benchmark(
                lambda candidate, values: candidate(values),
                model,
                inputs,
                repeats=arguments.repeats,
            ),
            "input_jacobian": _benchmark(
                lambda candidate, values: jax.jacrev(candidate)(values),
                model,
                inputs,
                repeats=arguments.repeats,
            ),
            "input_hessian": _benchmark(
                lambda candidate, values: jax.hessian(
                    lambda argument: jnp.sum(candidate(argument))
                )(values),
                model,
                inputs,
                repeats=arguments.repeats,
            ),
            "parameter_gradient": _benchmark(
                eqx.filter_value_and_grad(
                    lambda candidate, values: jnp.sum(candidate(values) ** 2)
                ),
                model,
                inputs,
                repeats=arguments.repeats,
            ),
        }

    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "jax_version": jax.__version__,
                "repeats": arguments.repeats,
                "width": arguments.width,
                "depth": arguments.depth,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
