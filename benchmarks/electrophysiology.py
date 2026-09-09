from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

from phydrax.applications import electrophysiology as ep
from phydrax.linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    MaterializationPolicy,
    materialize,
    solve,
)


def _measure(function, arguments, warmup, repeats):
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(function).lower(*arguments), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(*arguments), warmup=warmup, repeats=repeats
    )
    return result, {
        "compilation": asdict(compilation),
        "execution": execution.to_seconds_dict(),
        "compiler": asdict(
            compiler_evidence(
                compiled.cost_analysis(),
                compiled.memory_analysis(),
                source="jax-compiled-executable",
                unavailable_reason="Backend does not report compiler cost or memory analysis.",
            )
        ),
    }


def _case(count, topology, scheme, args):
    compartments = tuple(
        ep.CompartmentSpec(
            f"c{index}",
            None
            if index == 0
            else f"c{index - 1 if topology == 'chain' else (index - 1) // 2}",
            20.0 if index == 0 else 50.0,
            12.0 if index == 0 else 2.0,
        )
        for index in range(count)
    )
    morphology = ep.CellMorphologyPlan(
        f"benchmark-{topology}-{count}", compartments
    ).prepare()
    membrane = ep.MembraneProgram((ep.PassiveLeak(0.3, -65.0), ep.HodgkinHuxleyNaK()))
    tolerance = 1e-9 if args.dtype == "float64" else 1e-5
    cable = ep.CableSolverPlan(
        0.025, scheme=scheme, residual_tolerance=tolerance
    ).prepare(morphology, membrane)
    state = ep.initialize_cable_state(cable, jnp.full((count,), -65.0))
    zeros = jnp.zeros((count,))
    injected = zeros.at[-1].set(0.02)
    mask = jnp.zeros((count,), dtype=bool).at[0].set(count > 1)
    inputs = ep.CableStepInputs(injected, zeros, zeros, mask, jnp.full((count,), -65.0))

    def run(initial):
        def step(carry, _):
            previous, successful, max_residual, max_kirchhoff = carry
            result = ep.step_cable(cable, previous, inputs)
            return (
                result.state,
                successful & result.evidence.successful,
                jnp.maximum(max_residual, result.evidence.relative_residual),
                jnp.maximum(
                    max_kirchhoff, jnp.max(jnp.abs(result.evidence.kirchhoff_residual_nA))
                ),
            ), None

        carry = (initial, jnp.asarray(True), jnp.asarray(0.0), jnp.asarray(0.0))
        return jax.lax.scan(step, carry, xs=None, length=args.steps)[0]

    result, primal = _measure(run, (state,), args.warmup, args.repeats)
    final_state, successful, max_residual, max_kirchhoff = result

    def one_step(current):
        varying_inputs = ep.CableStepInputs(
            current, zeros, zeros, mask, inputs.voltage_clamp_target_mV
        )
        return ep.step_cable(cable, state, varying_inputs)

    def objective(current):
        advanced = one_step(current)
        return jnp.mean(advanced.state.voltage_mV) + jnp.mean(
            advanced.state.membrane.gates[1]
        )

    direction = jnp.linspace(-0.01, 0.01, count)
    (objective_value, tangent), jvp = _measure(
        lambda current, vector: jax.jvp(objective, (current,), (vector,)),
        (injected, direction),
        args.warmup,
        args.repeats,
    )
    gradient, vjp = _measure(jax.grad(objective), (injected,), args.warmup, args.repeats)
    elapsed_epsilon = jnp.asarray(1e-3 if args.dtype == "float32" else 1e-5)
    finite_difference = (
        objective(injected + elapsed_epsilon * direction)
        - objective(injected - elapsed_epsilon * direction)
    ) / (2 * elapsed_epsilon)
    reverse_direction = jnp.vdot(gradient, direction)
    mean_seconds = primal["execution"]["mean_seconds"]
    case = {
        "configuration": {"compartments": count, "topology": topology, "scheme": scheme},
        "identities": {
            "morphology": morphology.plan.plan_id,
            "cable": cable.plan.plan_id,
        },
        "logical_storage_bytes": {
            "prepared_cable": logical_array_bytes(cable),
            "state": logical_array_bytes(state),
        },
        "primal": primal,
        "jvp": jvp,
        "vjp": vjp,
        "compartment_steps_per_second": None
        if mean_seconds in (None, 0.0)
        else count * args.steps / mean_seconds,
        "physics": {
            "successful": bool(successful),
            "finite_voltage": bool(jnp.all(jnp.isfinite(final_state.voltage_mV))),
            "maximum_relative_residual": float(max_residual),
            "maximum_kirchhoff_residual_nA": float(max_kirchhoff),
            "minimum_voltage_mV": float(jnp.min(final_state.voltage_mV)),
            "maximum_voltage_mV": float(jnp.max(final_state.voltage_mV)),
            "time_ms": float(final_state.time_ms),
        },
        "derivatives": {
            "objective": float(objective_value),
            "jvp": float(tangent),
            "vjp_direction": float(reverse_direction),
            "finite_difference": float(finite_difference),
            "forward_reverse_absolute_error": float(jnp.abs(tangent - reverse_direction)),
            "finite_difference_absolute_error": float(
                jnp.abs(tangent - finite_difference)
            ),
            "finite_gradient": bool(jnp.all(jnp.isfinite(gradient))),
        },
    }
    if count <= args.dense_limit:
        evaluation = ep.evaluate_membrane_program(
            membrane,
            state.membrane,
            morphology,
            state.voltage_mV,
            state.intracellular_mM,
            state.extracellular_mM,
        )
        operator, right, _, _ = ep.assemble_cable_system(cable, state, evaluation, inputs)
        matrix = materialize(operator, MaterializationPolicy(max_entries=count * count))

        def dense_reference(coefficients, rhs):
            return solve(
                LinearSystem(DenseLinearOperator(coefficients)),
                rhs,
                policy=LinearSolvePolicy(DenseLU()),
            ).value

        reference, timing = _measure(
            dense_reference, (matrix, right), args.warmup, args.repeats
        )
        tree_voltage = one_step(injected).candidate_voltage_mV
        case["dense_reference"] = {
            "timing": timing,
            "matrix_bytes": int(matrix.nbytes),
            "maximum_voltage_error_mV": float(jnp.max(jnp.abs(tree_voltage - reference))),
        }
    return case


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tree cable primal/implicit-derivative scaling and physical evidence."
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compartments", type=int, nargs="+", default=[4, 64, 256])
    parser.add_argument(
        "--topologies",
        choices=("chain", "branched"),
        nargs="+",
        default=["chain", "branched"],
    )
    parser.add_argument(
        "--schemes",
        choices=("backward-euler", "crank-nicolson"),
        nargs="+",
        default=["backward-euler", "crank-nicolson"],
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--dense-limit",
        type=int,
        default=32,
        help="Bounded optional dense reference; 0 disables it.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.warmup < 0
        or args.repeats <= 0
        or any(count <= 0 for count in args.compartments)
        or args.dense_limit < 0
    ):
        raise ValueError(
            "steps/repeats/compartments must be positive; warmup/dense-limit nonnegative"
        )
    with jax.enable_x64(args.dtype == "float64"):
        payload = {
            "environment": capture_environment().to_dict(),
            "configuration": {
                "steps": args.steps,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "dtype": args.dtype,
                "dense_limit": args.dense_limit,
            },
            "units": ep.ELECTROPHYSIOLOGY_UNITS.units_id,
            "cases": [
                _case(count, topology, scheme, args)
                for count in args.compartments
                for topology in args.topologies
                for scheme in args.schemes
            ],
        }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
