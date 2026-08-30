#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import cyipopt
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _problem():
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: control,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="ipopt-qualification-integrator",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, args: trajectory.final_state[0],
        lower=1.0,
        upper=1.0,
        constraint_id="ipopt-qualification-terminal",
    )
    return phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.asarray((0.0,)),
        running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
        trajectory_constraints=(terminal,),
        problem_id="ipopt-qualification-integrator",
    )


def _solve(intervals: int, hessian: str, *, warm_start=None):
    mesh = phx.discretization.TemporalMesh.uniform(
        0.0,
        1.0,
        intervals,
        role="collocation",
        mesh_id=f"ipopt-qualification:{intervals}:{hessian}:mesh",
    )
    plan = phx.control.DirectCollocationPlan(
        mesh,
        method=phx.solver.ThetaMethod(0.5, endpoint=False),
        derivatives=phx.control.DirectCollocationDerivativePolicy(
            hessian=hessian,
            verify=True,
            num_verification_probes=2,
        ),
        plan_id=f"ipopt-qualification:{intervals}:{hessian}",
    )
    states = 0.8 * mesh.nodes[:, None]
    controls = 0.8 * jnp.ones((intervals, 1))
    compilation = phx.control.compile_direct_collocation(
        _problem(),
        plan,
        states,
        controls,
    )
    prepared = phx.control.prepare_direct_collocation(
        compilation,
        method=phx.optim.IpoptMinimize(options={"print_level": 0}),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1.0e-8,
            relative_optimality=0.0,
            maximum_steps=200,
        ),
    )
    started = time.perf_counter()
    result = phx.control.solve_prepared_direct_collocation(
        prepared,
        warm_start=warm_start,
    )
    elapsed = time.perf_counter() - started
    evidence = result.optimization_result.method_evidence
    if not isinstance(evidence, phx.optim.StructuredIpoptEvidence):
        raise TypeError("structured Ipopt returned no typed evidence")
    record = {
        "intervals": intervals,
        "hessian": hessian,
        "warm_started": evidence.warm_started,
        "successful": bool(result.successful),
        "backend_status": evidence.status.status,
        "backend_status_name": evidence.status.status_name,
        "public_status": int(result.optimization_result.status),
        "objective": float(result.objective),
        "maximum_defect": float(result.diagnostics.maximum_defect),
        "maximum_constraint_violation": float(
            result.diagnostics.maximum_constraint_violation
        ),
        "stationarity": float(
            result.optimization_result.diagnostics.final_optimality_norm
        ),
        "primal_feasibility": float(
            result.optimization_result.diagnostics.primal_feasibility
        ),
        "dual_feasibility": float(
            result.optimization_result.diagnostics.dual_feasibility
        ),
        "complementarity": float(
            result.optimization_result.diagnostics.complementarity
        ),
        "variables": compilation.structured_program.num_variables,
        "constraints": compilation.structured_program.num_constraints,
        "jacobian_nonzeros": evidence.jacobian_nonzeros,
        "hessian_nonzeros": evidence.hessian_nonzeros,
        "counts": {
            "objective": evidence.counts.objective,
            "gradient": evidence.counts.gradient,
            "constraints": evidence.counts.constraints,
            "jacobian": evidence.counts.jacobian,
            "hessian": evidence.counts.hessian,
            "intermediate": evidence.counts.intermediate,
            "host_to_device": evidence.counts.host_to_device,
            "device_to_host": evidence.counts.device_to_host,
        },
        "elapsed_seconds": elapsed,
        "program_id": evidence.program_id,
        "structure_id": evidence.structure_id,
        "options_id": evidence.options_id,
    }
    return result, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", nargs="+", type=int, default=(32, 128))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/direct_collocation_ipopt.json"),
    )
    arguments = parser.parse_args()
    if any(intervals < 1 for intervals in arguments.intervals):
        raise ValueError("every interval count must be positive")
    records = []
    for intervals in arguments.intervals:
        for hessian in ("limited-memory", "exact-sparse"):
            cold, cold_record = _solve(intervals, hessian)
            records.append(cold_record)
            evidence = cold.optimization_result.method_evidence
            assert isinstance(evidence, phx.optim.StructuredIpoptEvidence)
            _, warm_record = _solve(
                intervals,
                hessian,
                warm_start=evidence.final_warm_start,
            )
            records.append(warm_record)
    artifact = {
        "qualification": "structured-ipopt-direct-collocation",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "cyipopt": cyipopt.__version__,
        "ipopt": ".".join(str(value) for value in cyipopt.IPOPT_VERSION),
        "dtype": str(jnp.asarray(0.0).dtype),
        "records": records,
        "passed": all(record["successful"] for record in records),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
