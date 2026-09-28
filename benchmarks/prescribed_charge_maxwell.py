from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx


jax.config.update("jax_enable_x64", True)

D = phx.discretization
PIC = phx.discretization.pic
mx = phx.solver.maxwell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--cpml", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    # The CPML power ledger is a Δt-convergent residual of absorbing the abrupt
    # stop's grid-scale radiation: on the default 24³ box it is 2.8e-2 at CFL
    # 0.45 (LEDGER_OPEN) and 1.4e-3 at 0.225, below the 1e-2 default tolerance.
    parser.add_argument("--cfl", type=float, default=0.225)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.size < 8
        or args.cpml < 0
        or 2 * args.cpml >= args.size
        or args.steps <= 0
        or not 0.0 < args.cfl <= 1.0
        or args.warmup < 0
        or args.repeats <= 0
    ):
        raise ValueError(
            "size >= 8, 0 <= 2·cpml < size, positive steps/repeats, 0 < cfl <= 1, "
            "and nonnegative warmup are required"
        )
    count = args.size
    options: dict[str, Any] = {
        "constitutive": mx.DiagonalMaxwellConstitutivePlan(permittivity=2.25),
        "pml": mx.MaxwellCPMLPlan(args.cpml) if args.cpml else None,
    }

    def prepare() -> Any:
        grid = D.TensorGridPlan(
            tuple(D.UniformCellAxisSpec(count) for _ in range(3)),
            axis_names=("x", "y", "z"),
        ).prepare(jnp.asarray([[0.0] * 3, [1.0] * 3]))
        bridge = D.StructuredCochainBridge(grid)
        stable = float(
            phx.solver.CompatibleMaxwellPlan(bridge, **options).prepare().stable_dt
        )
        times = args.cfl * stable * np.arange(args.steps + 1)
        # Uniform motion at β = 0.9 through n = 1.5, stopping at z = 0.7.
        positions = np.stack(
            [np.asarray([[0.5, 0.5, 0.3 + 0.9 * time]]) for time in times]
        )
        positions[..., 2] = np.minimum(positions[..., 2], 0.7)
        particles = D.ParticleSetPlan(
            jnp.arange(1), jnp.ones((1,)), ambient_dimension=3
        ).prepare()
        charged = D.ChargedParticlePlan(jnp.ones((1,)), "benchmark").prepare(particles)
        current = PIC.ChargeConservingCurrentPlan(
            PIC.PICParticleCochainTransferPlan(bridge).prepare(charged)
        )
        trajectory = mx.PrescribedChargeTrajectory(times, positions)
        runtime = phx.solver.CompatibleMaxwellPlan(
            bridge,
            sources=(mx.PrescribedChargeCurrentSourcePlan(trajectory, current),),
            **options,
        ).prepare()
        return mx.PrescribedChargeMaxwellPlan(
            runtime, current, trajectory, np.asarray([1.0])
        )

    plan, preparation_seconds = measure_host(prepare)
    solve = jax.jit(lambda value: value.solve())
    compiled, compilation = measure_lower_and_compile(
        lambda: solve.lower(plan), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(plan), warmup=args.warmup, repeats=args.repeats
    )
    evidence = result.evidence
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "shape": [count, count, count],
            "cpml": args.cpml,
            "steps": args.steps,
            "cfl": args.cfl,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "identity": plan.plan_id,
        "preparation": {"plan_seconds": preparation_seconds},
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "execution": {"solve": execution.to_seconds_dict()},
        "evidence": {
            "successful": bool(evidence.successful),
            "status": int(evidence.status),
            "maximum_continuity_defect": float(evidence.maximum_continuity_defect),
            "maximum_gauss_defect": float(evidence.maximum_gauss_defect),
            "relative_ledger_defect": float(evidence.relative_ledger_defect),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
