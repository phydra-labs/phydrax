#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import platform

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_repeated, measure_synchronized


def _bridge(shape):
    dimension = len(shape)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for count in shape),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    return phx.discretization.StructuredCochainBridge(grid)


def _measure_case(shape, polarization):
    bridge = _bridge(shape)
    source = phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
        jnp.asarray([0]),
        jnp.asarray([1.0]),
        angular_frequency=2.0,
    )
    runtime, prepare_seconds = measure_synchronized(
        lambda: phx.solver.CompatibleMaxwellPlan(
            bridge,
            polarization=polarization,
            sources=(source,),
        ).prepare()
    )
    state = runtime.initialize()
    step_size = 0.05 * runtime.stable_dt
    step = jax.jit(lambda time, value: runtime.leapfrog_step(time, value, step_size))
    state, first_step_seconds = measure_synchronized(
        lambda: step(jnp.asarray(0.0), state)
    )
    _, warm_distribution = measure_repeated(
        lambda: step(jnp.asarray(step_size), state),
        warmup=1,
        repeats=20,
    )
    report = runtime.diagnostics(step_size, state, step_size=step_size)
    resources = runtime.resource_estimate
    return {
        "shape": list(shape),
        "polarization": polarization,
        "degree_counts": list(bridge.cochain.cell_counts),
        "prepare_seconds": prepare_seconds,
        "first_step_seconds": first_step_seconds,
        "warm_step_seconds": warm_distribution.median_seconds,
        "electric_constraint_linf": float(report.electric_constraint_linf),
        "magnetic_constraint_linf": float(report.magnetic_constraint_linf),
        "magnetic_projection_elided": runtime.magnetic_projection_elided,
        "resource_bytes": {
            "logical_primary": resources.logical_primary_bytes,
            "material_auxiliary": resources.material_auxiliary_bytes,
            "cpml": resources.cpml_state_bytes,
            "projection_workspace": resources.projection_workspace_bytes,
            "observer": resources.observer_state_bytes,
            "total": resources.total_bytes,
        },
    }


def main():
    rows = (
        _measure_case((8, 8, 8), "full_3d"),
        _measure_case((64, 64), "tez"),
        _measure_case((64, 64), "tmz"),
    )
    print(
        json.dumps(
            {
                "kind": "compatible-maxwell-benchmark",
                "python": platform.python_version(),
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "dtype": "float64",
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
