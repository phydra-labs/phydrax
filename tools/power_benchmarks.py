#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Run with ``python tools/power_benchmarks.py``; timings retain solver evidence."""

import json
import platform

import equinox as eqx
import jax

from benchmarks._runtime import measure_repeated, measure_synchronized
from phydrax.applications import power


def _radial(size):
    """Replicate equal single-section feeders, preserving each operating point.

    A common source supplies size-1 independent feeder/load pairs. Electrical
    depth, branch impedance and per-feeder demand stay fixed as the sparse model
    grows; this measures problem-size scaling, not a long-feeder stress limit.
    """
    network = power.PowerNetwork(
        tuple(power.Bus(str(i)) for i in range(size)),
        tuple(power.Branch(str(i), "0", str(i), 0.02, 0.2) for i in range(1, size)),
        (power.Generator("source", "0"),),
        tuple(power.Load(str(i), str(i), 0.25, 0.05) for i in range(1, size)),
    )
    study = power.PowerStudy(
        (power.BusControl("0", "reference"),)
        + tuple(power.BusControl(str(i)) for i in range(1, size))
    )
    return network, study


def main():
    rows = []
    for size in (2, 8, 32):
        compiled, prepare_seconds = measure_synchronized(
            lambda: power.compile_network(*_radial(size))
        )
        solve = eqx.filter_jit(
            lambda injections: power.fixed_mode_power_flow(compiled, injections)
        )
        result, first_seconds = measure_synchronized(
            lambda: solve(compiled.specified_power)
        )
        _, warm = measure_repeated(
            lambda: solve(compiled.specified_power), warmup=1, repeats=8
        )
        rows.append(
            {
                "kind": "rectangular-fixed-mode-power-flow",
                "topology": "parallel-single-section-feeders",
                "buses": size,
                "aggregate_active_load": 0.25 * (size - 1),
                "aggregate_reactive_load": 0.05 * (size - 1),
                "minimum_voltage_magnitude": float(abs(result.voltage).min()),
                "maximum_voltage_deviation": float(abs(result.voltage - 1).max()),
                "admittance_entries": int(compiled.admittance.coefficients.size),
                "prepare_seconds": prepare_seconds,
                "first_solve_seconds": first_seconds,
                "warm_solve_seconds": warm.median_seconds,
                "residual_norm": float(result.residual_norm),
                "converged": bool(result.converged),
                "native_status": int(result.root.status),
                "native_nonlinear_iterations": int(result.root.diagnostics.iterations),
                "native_linear_iterations": int(
                    result.root.diagnostics.linear_iterations
                ),
                "dtype": str(result.voltage.dtype),
            }
        )
    network = power.PowerNetwork(
        (power.Bus("source"), power.Bus("demand")),
        (power.Branch("line", "source", "demand", 0.0, 0.1, rate=0.4),),
        (
            power.Generator(
                "cheap",
                "source",
                p=0.3,
                p_min=0,
                p_max=2,
                q_min=-1,
                q_max=1,
                cost=(0.1, 1, 0),
            ),
            power.Generator(
                "local",
                "demand",
                p=0.7,
                p_min=0,
                p_max=2,
                q_min=-1,
                q_max=1,
                cost=(0.1, 3, 0),
            ),
        ),
        (power.Load("load", "demand", 1, 0.1),),
    )
    study = power.PowerStudy(
        (power.BusControl("source", "reference"), power.BusControl("demand"))
    )
    dc, dc_prepare = measure_synchronized(
        lambda: power.compile_dc_opf(network, study=study)
    )
    dc_result, dc_seconds = measure_synchronized(lambda: power.solve_dc_opf(dc))
    rows.append(
        {
            "kind": "native-dc-opf",
            "prepare_seconds": dc_prepare,
            "solve_seconds": dc_seconds,
            "converged": bool(dc_result.converged),
            "original_feasibility": float(dc_result.original_feasibility),
            "objective": float(dc_result.objective),
        }
    )
    ac, ac_prepare = measure_synchronized(
        lambda: power.compile_ac_opf(network, study=study)
    )
    ac_result, ac_seconds = measure_synchronized(lambda: power.solve_ac_opf(ac))
    rows.append(
        {
            "kind": "native-structured-ac-opf",
            "prepare_seconds": ac_prepare,
            "solve_seconds": ac_seconds,
            "converged": bool(ac_result.converged),
            "original_feasibility": float(ac_result.original_feasibility),
            "objective": float(ac_result.objective),
            "jacobian_entries": ac.program.jacobian_plan.nnz,
            "hessian_entries": ac.program.hessian_plan.nnz,
        }
    )
    feasibility_tolerance = 1e-6
    passed = all(
        row["converged"]
        and row.get("original_feasibility", row.get("residual_norm", float("inf")))
        <= feasibility_tolerance
        for row in rows
    )
    print(
        json.dumps(
            {
                "kind": "balanced-power-benchmarks",
                "passed": passed,
                "feasibility_tolerance": feasibility_tolerance,
                "python": platform.python_version(),
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
