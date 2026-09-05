# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native economic dispatch -> AC operating point -> classical-machine fault DAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.energy._artifacts import (
    archive_metrics,
    archive_workflow,
    execution_identity,
)
from phydrax.applications import power
from phydrax.dynamics import TimeGrid
from phydrax.solver import solve_dae


def run_power_fault(output_dir, *, execution=None):
    """Audit native states/rates at initialization, every sample and both events.

    DC economic dispatch supplies only P. A separate original AC power flow sets
    Q/voltage before initialization. The infinite grid is explicitly declared;
    the machine is not turned into an infinite source by its PV study control.
    """
    if not jax.config.x64_enabled:
        raise ValueError("This qualification requires JAX_ENABLE_X64=1.")
    execution = execution_identity() if execution is None else execution
    network = power.PowerNetwork(
        (power.Bus("infinite", 110), power.Bus("machine", 110)),
        (power.Branch("tie", "infinite", "machine", 0.0, 0.2, rate=2.0),),
        (
            power.Generator(
                "grid",
                "infinite",
                p=0.4,
                p_min=0,
                p_max=2,
                q_min=-2,
                q_max=2,
                cost=(0.1, 3, 0),
            ),
            power.Generator(
                "unit",
                "machine",
                p=0.6,
                p_min=0,
                p_max=0.6,
                q_min=-2,
                q_max=2,
                cost=(0.1, 1, 0),
            ),
        ),
        (power.Load("demand", "infinite", 1.0, 0.1),),
        base_mva=100.0,
    )
    study = power.PowerStudy(
        (power.BusControl("infinite", "reference"), power.BusControl("machine", "pv"))
    )
    dispatch = power.solve_dc_opf(network, study=study)
    if not bool(dispatch.converged):
        raise RuntimeError("Native DC dispatch did not meet its original DC constraints.")
    operating_network = eqx.tree_at(
        lambda n: tuple(g.p for g in n.generators),
        network,
        tuple(float(value) for value in dispatch.generator_power),
    )
    compiled = power.compile_network(operating_network, study)
    operating_point = power.solve_power_flow(compiled)
    if not bool(operating_point.operationally_feasible):
        raise RuntimeError("DC dispatch does not admit the requested AC operating point.")
    machine = power.ClassicalMachine("unit", inertia=4.0, damping=1.0, xd_prime=0.3)
    initial = power.initialize_smib(
        compiled, operating_point, machine, infinite_bus="infinite"
    )
    if not bool(initial.valid):
        raise RuntimeError(
            f"Native DAE equilibrium initialization failed: {initial.status}"
        )
    requested = jnp.linspace(0.0, 0.2, 41)
    trajectory = power.simulate_power_dynamics(
        initial,
        requested,
        events=(
            power.PowerEvent(float(requested[10]), "fault", "machine", admittance=2 - 5j),
            power.PowerEvent(float(requested[20]), "clear", "machine"),
        ),
    )
    if not bool(trajectory.valid):
        raise RuntimeError(f"Native fault trajectory failed: {trajectory.status}")
    arrays, units = {}, {}
    initial_residual = initial.problem.system.residual(
        0.0, initial.problem.initial_state, initial.problem.initial_state_rate, None
    )
    sample_residual = 0.0
    differential = []
    for index, segment in enumerate(trajectory.segments):
        solution = segment.solution
        if solution is None:
            raise RuntimeError("A requested topology segment was not executed.")
        problem = initial.model.problem(solution.states[0], topology=segment.topology)
        residuals = jax.vmap(lambda t, x, dx: problem.system.residual(t, x, dx, None))(
            solution.times, solution.states, solution.state_rates
        )
        sample_residual = max(sample_residual, float(jnp.max(jnp.abs(residuals))))
        differential.append(
            np.asarray(solution.states[:, : initial.model.differential_size])
        )
        for name, value, unit in (
            ("time", solution.times, "s"),
            ("state", solution.states, "mixed:rad,pu-speed,rectangular-pu-voltage"),
            ("rate", solution.state_rates, "state/s"),
            ("residual", residuals, "mixed:state/s,pu-current"),
        ):
            arrays[f"segment-{index}/{name}"] = value
            units[f"segment-{index}/{name}"] = unit
    event_metrics = []
    for index, event in enumerate(trajectory.events):
        before_problem = initial.model.problem(
            event.before, topology=event.topology_before
        )
        after_problem = initial.model.problem(event.after, topology=event.topology_after)
        before = before_problem.system.residual(
            event.event.time, event.before, event.rate_before, None
        )
        after = after_problem.system.residual(
            event.event.time, event.after, event.rate_after, None
        )
        jump = (
            event.after[: initial.model.differential_size]
            - event.before[: initial.model.differential_size]
        )
        event_metrics.append(
            {
                "kind": event.event.kind,
                "time_s": event.event.time,
                "status": event.status,
                "applied": bool(event.applied),
                "restart_order": event.restart_order,
                "maximum_residual_before": float(jnp.max(jnp.abs(before))),
                "maximum_residual_after": float(jnp.max(jnp.abs(after))),
                "maximum_differential_jump": float(jnp.max(jnp.abs(jump))),
                "maximum_voltage_jump_pu": float(
                    jnp.max(
                        jnp.abs(
                            initial.model.voltage(event.after)
                            - initial.model.voltage(event.before)
                        )
                    )
                ),
                "faults_after_pu": [
                    [bus, float(z.real), float(z.imag)]
                    for bus, z in event.topology_after.faults
                ],
            }
        )
        for name, value, unit in (
            ("before", event.before, "mixed-state"),
            ("after", event.after, "mixed-state"),
            ("rate_before", event.rate_before, "mixed-state/s"),
            ("rate_after", event.rate_after, "mixed-state/s"),
            ("residual_before", before, "mixed-residual"),
            ("residual_after", after, "mixed-residual"),
        ):
            arrays[f"event-{index}/{name}"] = value
            units[f"event-{index}/{name}"] = unit
    states = np.concatenate(differential)
    metrics = {
        "scope": (
            "lossless DC economic dispatch; original AC operating point; one classical "
            "machine with explicit infinite bus; native segmented index-one BDF DAE"
        ),
        "power_base_total_three_phase_MVA": network.base_mva,
        "frequency_Hz": network.frequency,
        "machine": {
            "inertia_s": 4.0,
            "damping": 1.0,
            "xd_prime_pu": 0.3,
            "base": "network",
        },
        "load_model": trajectory.load_model,
        "differential_state_names": list(initial.model.differential_names),
        "dispatch_P_pu": np.asarray(dispatch.generator_power).tolist(),
        "operating_P_pu": np.asarray(operating_point.generator_power.real).tolist(),
        "operating_Q_pu": np.asarray(operating_point.generator_power.imag).tolist(),
        "native_dispatch_status": str(dispatch.native_result.status),
        "DC_original_feasibility": float(dispatch.original_feasibility),
        "AC_balance_pu": float(jnp.max(jnp.abs(operating_point.bus_balance))),
        "initial_equilibrium_norm": float(initial.equilibrium_norm),
        "initial_residual_norm": float(jnp.max(jnp.abs(initial_residual))),
        "maximum_sample_residual": sample_residual,
        "rotor_angle_excursion_rad": float(np.ptp(states[:, 0])),
        "speed_excursion_pu": float(np.ptp(states[:, 1])),
        "events": event_metrics,
        "final_time_s": float(trajectory.final_time),
        "criteria": {
            "initial_residual": 1e-7,
            "sample_event_residual": 1e-5,
            "differential_jump": 1e-10,
            "minimum_rotor_excursion_rad": 1e-4,
            "minimum_speed_excursion_pu": 1e-6,
            "minimum_fault_voltage_jump_pu": 0.01,
        },
    }
    metrics["passed"] = (
        metrics["initial_residual_norm"] <= 1e-7
        and metrics["initial_equilibrium_norm"] <= 1e-7
        and sample_residual <= 1e-5
        and metrics["AC_balance_pu"] <= 1e-6
        and metrics["rotor_angle_excursion_rad"] >= 1e-4
        and metrics["speed_excursion_pu"] >= 1e-6
        and len(event_metrics) == 2
        and all(
            event["applied"]
            and event["maximum_residual_before"] <= 1e-5
            and event["maximum_residual_after"] <= 1e-5
            and event["maximum_differential_jump"] <= 1e-10
            and event["restart_order"] == 1
            for event in event_metrics
        )
        and event_metrics[0]["maximum_voltage_jump_pu"] >= 0.01
        and not trajectory.events[-1].topology_after.faults
        and abs(float(trajectory.final_time) - 0.2) <= 1e-12
    )
    arrays.update(
        initial_state=initial.problem.initial_state,
        initial_rate=initial.problem.initial_state_rate,
        initial_residual=initial_residual,
        final_state=trajectory.final_state,
        final_rate=trajectory.final_state_rate,
    )
    units.update(
        initial_state="mixed-state",
        initial_rate="mixed-state/s",
        initial_residual="mixed-residual",
        final_state="mixed-state",
        final_rate="mixed-state/s",
    )
    archives = archive_workflow(
        output_dir,
        "power-fault",
        metrics,
        arrays,
        units,
        {
            "state": trajectory.final_state,
            "rate": trajectory.final_state_rate,
            "time": np.asarray([trajectory.final_time]),
            "branch_closed": trajectory.segments[-1].topology.branch_closed,
            "topology_admittance": trajectory.segments[-1].topology.admittance,
            "internal_emf": initial.model.internal_emf,
            "mechanical_reference": initial.model.mechanical_reference,
        },
        execution=execution,
    )
    if not metrics["passed"]:
        raise RuntimeError(
            f"Power fault acceptance failed: {json.dumps(metrics, allow_nan=False)}"
        )
    # Real continuation from reopened physical coordinates; fresh native BDF history.
    checkpoint = archives["checkpoint"].arrays
    if tuple(checkpoint["branch_closed"]) != initial.model.initial_topology.branch_closed:
        raise RuntimeError(
            "Reopened topology does not match the declared cleared-fault model."
        )
    topology = eqx.tree_at(
        lambda t: t.admittance,
        trajectory.segments[-1].topology,
        jnp.asarray(checkpoint["topology_admittance"]),
    )
    restored_model = eqx.tree_at(
        lambda m: (m.internal_emf, m.mechanical_reference),
        initial.model,
        (
            jnp.asarray(checkpoint["internal_emf"]),
            jnp.asarray(checkpoint["mechanical_reference"]),
        ),
    )
    resumed_problem = restored_model.problem(
        checkpoint["state"], state_rate=checkpoint["rate"], topology=topology
    )
    resumed = solve_dae(
        resumed_problem,
        TimeGrid(jnp.asarray([0.2, 0.205, 0.21]), time_id="power-fault:archive-resume"),
    )
    resumed_residuals = jax.vmap(
        lambda t, x, dx: resumed_problem.system.residual(t, x, dx, None)
    )(resumed.times, resumed.states, resumed.state_rates)
    if (
        not bool(jnp.all(resumed.valid))
        or float(jnp.max(jnp.abs(resumed_residuals))) > 1e-5
    ):
        raise RuntimeError(
            "Native DAE continuation from reopened checkpoint failed its residual audit."
        )
    continuation_metrics = {
        "passed": True,
        "checkpoint_archive_id": archives["checkpoint"].archive_id,
        "scope": (
            "fresh BDF continuation from reopened physical state/rate with restored "
            "topology; no solver-cache restoration"
        ),
        "maximum_residual": float(jnp.max(jnp.abs(resumed_residuals))),
        "final_time_s": float(resumed.times[-1]),
    }
    continuation = archive_workflow(
        output_dir,
        "power-fault-resumed",
        continuation_metrics,
        {
            "time": resumed.times,
            "state": resumed.states,
            "rate": resumed.state_rates,
            "residual": resumed_residuals,
        },
        {
            "time": "s",
            "state": "mixed-state",
            "rate": "mixed-state/s",
            "residual": "mixed-residual",
        },
        {
            "state": resumed.states[-1],
            "rate": resumed.state_rates[-1],
            "time": resumed.times[-1:],
        },
        execution=execution,
    )
    return {
        "metrics": metrics,
        "archives": archives,
        "execution": execution,
        "dispatch": dispatch,
        "operating_point": operating_point,
        "initialization": initial,
        "trajectory": trajectory,
        "resumed": resumed,
        "continuation_metrics": continuation_metrics,
        "continuation_archives": continuation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("energy-results/power"))
    args = parser.parse_args()
    result = run_power_fault(args.output)
    print(
        json.dumps(
            {
                **result["metrics"],
                **archive_metrics(result["archives"]),
                "continuation": {
                    **result["continuation_metrics"],
                    **archive_metrics(result["continuation_archives"]),
                },
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
