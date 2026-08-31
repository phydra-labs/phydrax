#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


def _properties():
    return phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
        },
    )


def _spectrum(kind: str, dimension: int) -> jax.Array:
    index = jnp.arange(dimension, dtype=float)
    if kind == "power":
        return 20.0 / (index + 1.0) ** 2
    if kind == "exponential":
        return 20.0 * jnp.exp(-index / max(dimension / 12.0, 1.0))
    if kind == "flat":
        return jnp.ones((dimension,))
    raise ValueError(f"Unknown spectrum kind {kind!r}.")


def _run(kind: str, dimension: int, rank: int, seed: int) -> dict[str, object]:
    eigenvalues = _spectrum(kind, dimension)
    base = phx.linalg.DiagonalLinearOperator(
        eigenvalues,
        properties=_properties(),
        operator_id=f"benchmark:{kind}:{dimension}",
    )
    shift = 1e-2
    shifted = phx.linalg.DiagonalLinearOperator(
        eigenvalues + shift,
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id=f"benchmark:{kind}:{dimension}:shifted",
    )
    rhs = jnp.sin(jnp.arange(dimension, dtype=float) + 1.0)
    builder = phx.linalg.RandomizedNystromPreconditionerBuilder(
        rank,
        oversampling=min(8, dimension - rank),
        shift=shift,
        seed=seed,
    )
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.PCG(),
        tolerance=phx.linalg.TolerancePolicy(relative=1e-6, absolute=1e-10),
        preconditioning=phx.linalg.PreconditioningPolicy(
            builder,
            setup_operator=base,
        ),
        differentiation=phx.linalg.DifferentiationPolicy("none"),
    )
    started = time.perf_counter()
    result = phx.linalg.solve(phx.linalg.LinearSystem(shifted), rhs, policy=policy)
    jax.block_until_ready(result.value)
    elapsed = time.perf_counter() - started
    return {
        "spectrum": kind,
        "dimension": dimension,
        "rank": rank,
        "seed": seed,
        "successful": bool(result.successful),
        "relative_residual": float(result.diagnostics.relative_residual),
        "iterations": int(result.diagnostics.iterations),
        "matvec_count": int(result.diagnostics.matvec_count),
        "setup_matvec_count": int(result.provenance.preconditioner_setup_matvec_count),
        "wall_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--spectra",
        nargs="+",
        choices=("power", "exponential", "flat"),
        default=("power", "exponential", "flat"),
    )
    args = parser.parse_args()
    rows = [
        _run(kind, args.dimension, args.rank, seed)
        for kind in args.spectra
        for seed in args.seeds
    ]
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
