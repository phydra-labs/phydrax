#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Increment phases: python -m benchmarks.periodic_increment_moments --size 4."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import prod
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization import (
    AxisDomain,
    FourierBasisPlan,
    PaddingDealiasingPlan,
    PeriodicIncrementMomentEstimatorPlan,
    PeriodicLatticeIncrementMomentPlan,
    PeriodicRadialIncrementMomentPlan,
    PeriodicThirdIncrementMomentPlan,
    TensorSpectralPlan,
)
from phydrax.discretization.spectral import PeriodicIncrementMomentResult

from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


@eqx.filter_jit
def _evaluate(
    plan: PeriodicLatticeIncrementMomentPlan
    | PeriodicRadialIncrementMomentPlan
    | PeriodicThirdIncrementMomentPlan
    | PeriodicIncrementMomentEstimatorPlan,
    coefficients: Array,
    /,
) -> PeriodicIncrementMomentResult:
    return plan(coefficients)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    space = TensorSpectralPlan(
        tuple(FourierBasisPlan(args.size) for _ in range(args.dimension)),
        axis_names=tuple("xyz"[: args.dimension]),
    ).prepare(tuple(AxisDomain.periodic(0, 2 * np.pi) for _ in range(args.dimension)))
    rng = np.random.default_rng(412)
    native = rng.normal(size=(args.members,) + space.modal_shape + (args.dimension,))
    coefficients = (
        np.fft.fftn(native, axes=tuple(range(1, args.dimension + 1)))
        * np.sqrt((2 * np.pi) ** args.dimension)
        / prod(space.modal_shape)
    )
    for index, axis in enumerate(space.axes, start=1):
        mask_shape = (
            (1,) * index + (axis.mode_count,) + (1,) * (coefficients.ndim - index - 1)
        )
        coefficients *= ~np.asarray(axis.modes.nyquist_mask).reshape(mask_shape)
    inputs = jnp.asarray(coefficients)

    def plans() -> tuple[
        PeriodicLatticeIncrementMomentPlan
        | PeriodicRadialIncrementMomentPlan
        | PeriodicThirdIncrementMomentPlan
        | PeriodicIncrementMomentEstimatorPlan,
        ...,
    ]:
        return (
            PeriodicLatticeIncrementMomentPlan(
                space, component_shape=(args.dimension,), member_chunk_size=args.chunk
            ),
            PeriodicRadialIncrementMomentPlan(
                space,
                np.linspace(0, 1, 8),
                averaging="ball",
                component_shape=(args.dimension,),
                member_chunk_size=args.chunk,
            ),
            PeriodicThirdIncrementMomentPlan(
                space,
                statistic="longitudinal_cube",
                dealiasing=PaddingDealiasingPlan(2),
                member_chunk_size=args.chunk,
                maximum_workspace_bytes=2**30,
            ),
            PeriodicThirdIncrementMomentPlan(
                space,
                statistic="energy_flux",
                dealiasing=PaddingDealiasingPlan(2),
                member_chunk_size=args.chunk,
                maximum_workspace_bytes=2**30,
            ),
            PeriodicIncrementMomentEstimatorPlan(
                space,
                np.eye(args.dimension, dtype=np.int32),
                order=3,
                projection="longitudinal",
                member_chunk_size=args.chunk,
                maximum_displacements=8,
            ),
        )

    prepared, preparation_seconds = measure_synchronized(plans)
    rows: list[dict[str, Any]] = []
    for plan in prepared:
        compiled, timing = measure_lower_and_compile(
            lambda: _evaluate.lower(plan, inputs),  # ty: ignore[unresolved-attribute]
            lambda lowered: lowered.compile(),
        )
        cold, cold_seconds = measure_synchronized(lambda: compiled(plan, inputs))
        warm, distribution = measure_repeated(
            lambda: compiled(plan, inputs), warmup=1, repeats=args.repeats
        )
        np.testing.assert_array_equal(cold.values, warm.values)
        if not np.all(np.asarray(warm.finite)):
            raise ValueError("Increment benchmark produced nonfinite statistics.")
        evidence = compiler_evidence(
            compiled.compiled.cost_analysis(),
            compiled.compiled.memory_analysis(),
            source="jax-compiled-periodic-increment-moments",
        )
        rows.append(
            {
                "plan": type(plan).__name__,
                "plan_id": plan.plan_id,
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "cold_seconds": cold_seconds,
                "warm": distribution.to_seconds_dict(),
                "compiler": asdict(evidence),
                "prepared_bytes": logical_array_bytes(plan),
                "output_shape": warm.values.shape,
            }
        )
    root = Path(__file__).resolve().parent.parent
    identity = capture_benchmark_identity(root, Path(__file__), tuple(rows[0]))
    print(
        json.dumps(
            {
                "benchmark_identity": identity.to_dict(),
                "environment": capture_environment().to_dict(),
                "size": args.size,
                "dimension": args.dimension,
                "members": args.members,
                "chunk": args.chunk,
                "preparation_seconds": preparation_seconds,
                "cases": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
