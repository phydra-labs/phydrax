"""Deterministic teacher/student spike-timing fit with the native Adam workflow.

Run from the repository root:
    python -m examples.artificial_lif_temporal_learning

This small full-batch smoke learns artificial recurrent weights from cumulative
spike trajectories. It is not a physical event-gradient or generalization claim.
"""

from __future__ import annotations

import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def run_example(*, steps: int = 120) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive.")
    length = 32
    nodes = jnp.arange(length, dtype=jnp.float32)
    phases = jnp.arange(4, dtype=jnp.float32)
    inputs = (0.35 + 0.9 * (((nodes[None, :] + phases[:, None]) % 8) < 3))[..., None]
    valid = jnp.ones((4, length), dtype=bool)
    time = jnp.broadcast_to(nodes, valid.shape)
    batch = phx.nn.layers.RecurrentBatch(inputs, valid, time=time)
    cell = phx.nn.layers.ArtificialLIFCell(
        1,
        1,
        time_constant_ms=2.0,
        surrogate_width=1.0,
        detach_reset=True,
        key=jr.key(2026),
    )
    teacher_cell = eqx.tree_at(
        lambda current: (current.weight_ih, current.weight_hh, current.bias),
        cell,
        (jnp.array([[1.8]]), jnp.array([[0.35]]), jnp.array([0.25])),
    )
    teacher = phx.nn.models.RecurrentSequenceModel(teacher_cell)
    targets = teacher(batch)
    student_cell = eqx.tree_at(
        lambda current: (current.weight_ih, current.weight_hh, current.bias),
        cell,
        (jnp.array([[1.2]]), jnp.array([[0.1]]), jnp.array([0.1])),
    )
    student = phx.nn.models.RecurrentSequenceModel(student_cell)
    parameters, configuration = eqx.partition(student, eqx.is_inexact_array)

    def scenario_loss(current, scenario, _args):
        model = eqx.combine(current, configuration)
        sequence = phx.nn.layers.RecurrentBatch(
            scenario["inputs"], jnp.ones((length,), dtype=bool), time=nodes
        )
        residual = jnp.cumsum(model(sequence) - scenario["target"], axis=0)
        return jnp.mean(jnp.square(residual))

    scenarios = {"inputs": inputs, "target": targets}
    sampling = phx.optim.FixedSampling(scenarios)
    problem = phx.optim.StochasticProblem(
        scenario_loss, sampling, problem_id="artificial-lif-temporal-fit"
    )
    fixed_batch = sampling.sample(jr.key(0), 0)
    initial_loss = problem.value(parameters, fixed_batch, None)
    result = phx.optim.minimize_stochastic(
        problem,
        parameters,
        method=phx.optim.StochasticAdam(0.02),
        termination=phx.optim.OptimizationTermination(
            maximum_steps=steps,
            absolute_optimality=0.0,
            relative_optimality=0.0,
        ),
        seed=2026,
    )
    fitted = eqx.combine(result.parameters, configuration)
    final_loss = problem.value(result.parameters, fixed_batch, None)
    # Native chunk continuation preserves both the membrane/spike pair and time.
    first = fitted.evaluate_with_state(
        phx.nn.layers.RecurrentBatch(inputs[:, :13], valid[:, :13], time=time[:, :13])
    )
    second = fitted.evaluate_with_state(
        phx.nn.layers.RecurrentBatch(inputs[:, 13:], valid[:, 13:], time=time[:, 13:]),
        initial_state=first.final_state,
        initial_context=first.final_context,
    )
    full_output = fitted(batch)
    streaming = jnp.concatenate((first.outputs, second.outputs), axis=1)
    streaming_error = jnp.max(jnp.abs(streaming - full_output))
    if not bool(jnp.isfinite(final_loss) & (final_loss < initial_loss)):
        raise RuntimeError(
            "Temporal surrogate training did not improve the spike-trajectory loss."
        )
    if not bool(streaming_error == 0.0):
        raise RuntimeError("Chunk continuation changed the fitted spike sequence.")
    return {
        "backend": jax.default_backend(),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "optimizer_status": int(result.status),
        "accepted_steps": int(result.diagnostics.accepted_steps),
        "teacher_spike_counts": jnp.sum(targets, axis=1).ravel().tolist(),
        "fitted_spike_counts": jnp.sum(full_output, axis=1).ravel().tolist(),
        "streaming_maximum_error": float(streaming_error),
        "input_weight": float(fitted.cell.weight_ih[0, 0]),
        "recurrent_weight": float(fitted.cell.weight_hh[0, 0]),
        "bias": float(fitted.cell.bias[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    arguments = parser.parse_args()
    print(json.dumps(run_example(steps=arguments.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
