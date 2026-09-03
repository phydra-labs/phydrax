"""Benchmark the native fixed-base reduced-articulation workflow."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.robotics import parse_urdf_text
from phydrax.control import (
    DiscreteControlDynamics,
    PiecewiseConstantControlParameterization,
)
from phydrax.discretization import (
    reduced_forward_dynamics,
    reduced_inverse_dynamics,
    reduced_symplectic_step,
    ReducedSymplecticStepPolicy,
)
from phydrax.dynamics import (
    DiscreteSystem,
    DiscreteTransitionResult,
    TimeGrid,
)


def _link_xml(index: int, /) -> str:
    mass = 1.0 + 0.05 * index
    inertia = 0.02 + 0.001 * index
    return f"""
  <link name="link_{index:02d}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.8f}"/>
      <inertia ixx="{inertia:.8f}" ixy="0" ixz="0"
               iyy="{inertia:.8f}" iyz="0" izz="{inertia:.8f}"/>
    </inertial>
  </link>"""


def _joint_xml(index: int, /) -> str:
    kind = "revolute" if index % 2 == 0 else "prismatic"
    axis = "0 0 1" if kind == "revolute" else "1 0 0"
    lower, upper = ("-2.5", "2.5") if kind == "revolute" else ("-0.2", "0.3")
    return f"""
  <joint name="joint_{index:02d}" type="{kind}">
    <parent link="link_{index:02d}"/>
    <child link="link_{index + 1:02d}"/>
    <origin xyz="0.25 0 0" rpy="0 0 0"/>
    <axis xyz="{axis}"/>
    <limit lower="{lower}" upper="{upper}" effort="50" velocity="5"/>
    <dynamics damping="0.01"/>
  </joint>"""


def _chain_urdf(joint_count: int, /) -> str:
    links = "".join(_link_xml(index) for index in range(joint_count + 1))
    joints = "".join(_joint_xml(index) for index in range(joint_count))
    return f'<robot name="benchmark-chain-{joint_count}">{links}{joints}\n</robot>'


def _prepare_chain(joint_count: int, /):
    adaptation = parse_urdf_text(_chain_urdf(joint_count))
    particles = adaptation.particles.prepare()
    bodies = adaptation.bodies.prepare(particles)
    graph = adaptation.joints.prepare(bodies, adaptation.reference)
    articulation = adaptation.articulation.prepare(graph, adaptation.reference)
    return adaptation, articulation


def _synchronize(value: Any, /) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(
    operation: Callable[[], Any],
    /,
    *,
    repeats: int,
) -> tuple[Any, dict[str, float]]:
    warmup = operation()
    _synchronize(warmup)
    samples = []
    result = warmup
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = operation()
        _synchronize(result)
        samples.append((time.perf_counter_ns() - started) * 1.0e-6)
    values = np.asarray(samples, dtype=float)
    return result, {
        "maximum_ms": round(float(np.max(values)), 6),
        "median_ms": round(float(np.median(values)), 6),
        "minimum_ms": round(float(np.min(values)), 6),
    }


def _scalar_bool(value: Any, /) -> bool:
    return bool(np.asarray(value))


def _scalar_float(value: Any, /) -> float:
    return float(np.asarray(value))


def _rollout_workload(
    articulation,
    configuration,
    /,
    *,
    batch_size: int,
    step_count: int,
):
    if articulation.state_layout is None or articulation.input_layout is None:
        raise ValueError("rollout benchmarking requires at least one moving joint")
    dtype = articulation.reference_position.dtype
    gravity = jnp.zeros((3,), dtype=dtype)
    policy = ReducedSymplecticStepPolicy(maximum_step_size=0.01)

    def transition(context, packed_state, generalized_effort, args):
        del args
        source = articulation.unpack_state(packed_state)
        result = reduced_symplectic_step(
            articulation,
            source,
            generalized_effort,
            gravity,
            context.duration,
            policy=policy,
        )
        return DiscreteTransitionResult(
            articulation.pack_state(result.candidate_state),
            articulation.pack_state(result.accepted_state),
            result.successful,
            result.status,
        )

    system = DiscreteSystem(
        transition,
        state_layout=articulation.state_layout,
        input_layout=articulation.input_layout,
        system_id=f"benchmark:robotics-articulation:{articulation.prepared_id}",
    )
    dynamics = DiscreteControlDynamics(
        system,
        method_id="benchmark:status-aware-symplectic-rollout",
    )
    step_size = 1.0e-3
    grid = TimeGrid(
        jnp.arange(step_count + 1, dtype=dtype) * step_size,
        time_id=f"benchmark:robotics-rollout:{step_count}",
    )
    parameterization = PiecewiseConstantControlParameterization(
        grid,
        articulation.input_layout.shape,
        parameterization_id="benchmark:zero-generalized-effort",
    )
    source = articulation.pack_state(
        configuration,
        jnp.zeros((articulation.nv,), dtype=dtype),
    )
    initial_states = jnp.broadcast_to(
        source,
        (batch_size, articulation.state_size),
    )
    controls = jnp.zeros(
        (batch_size, step_count, articulation.nv),
        dtype=dtype,
    )

    def rollout(initial, coefficients):
        return dynamics.rollout(
            grid,
            initial,
            parameterization,
            coefficients,
            problem_id=f"benchmark:robotics-rollout:{articulation.prepared_id}",
        )

    return jax.jit(rollout), initial_states, controls


def _run_case(
    joint_count: int,
    /,
    *,
    repeats: int,
    rollout_batch_size: int,
    rollout_step_count: int,
) -> dict[str, Any]:
    (adaptation, articulation), preparation_timing = _measure(
        lambda: _prepare_chain(joint_count),
        repeats=repeats,
    )
    dtype = articulation.reference_position.dtype
    configuration = jnp.linspace(0.025, 0.15, articulation.nq, dtype=dtype)
    velocity = jnp.linspace(-0.08, 0.12, articulation.nv, dtype=dtype)
    target_acceleration = jnp.linspace(0.03, -0.04, articulation.nv, dtype=dtype)
    gravity = jnp.asarray([0.0, -9.81, 0.0], dtype=dtype)
    tip_body_id = int(np.asarray(articulation.body_ids[-1]))
    frame_load = jnp.linspace(0.1, 0.6, 6, dtype=dtype)

    compiled_fk = jax.jit(
        lambda point, rate: articulation.forward_kinematics(point, rate)
    )

    def jacobian_actions(point, rate, load):
        operator = articulation.frame_jacobian_operator(point, tip_body_id)
        return operator.mv(rate), operator.transpose_mv(load)

    compiled_jacobian = jax.jit(jacobian_actions)
    compiled_inverse = jax.jit(
        lambda point, rate, acceleration: reduced_inverse_dynamics(
            articulation,
            point,
            rate,
            acceleration,
            gravity,
        )
    )
    compiled_forward = jax.jit(
        lambda point, rate, effort: reduced_forward_dynamics(
            articulation,
            point,
            rate,
            effort,
            gravity,
        )
    )

    kinematics, fk_timing = _measure(
        lambda: compiled_fk(configuration, velocity),
        repeats=repeats,
    )
    (frame_velocity, generalized_load), jacobian_timing = _measure(
        lambda: compiled_jacobian(configuration, velocity, frame_load),
        repeats=repeats,
    )
    inverse, inverse_timing = _measure(
        lambda: compiled_inverse(configuration, velocity, target_acceleration),
        repeats=repeats,
    )
    forward, forward_timing = _measure(
        lambda: compiled_forward(
            configuration,
            velocity,
            inverse.generalized_effort,
        ),
        repeats=repeats,
    )

    compiled_rollout, initial_states, controls = _rollout_workload(
        articulation,
        configuration,
        batch_size=rollout_batch_size,
        step_count=rollout_step_count,
    )
    trajectory, rollout_timing = _measure(
        lambda: compiled_rollout(initial_states, controls),
        repeats=repeats,
    )

    contact_power = jnp.vdot(frame_velocity, frame_load).real
    generalized_power = jnp.vdot(velocity, generalized_load).real
    jacobian_power_residual = jnp.abs(contact_power - generalized_power)
    jacobian_power_scale = jnp.maximum(
        1.0,
        jnp.maximum(jnp.abs(contact_power), jnp.abs(generalized_power)),
    )
    jacobian_relative_residual = jacobian_power_residual / jacobian_power_scale
    acceleration_error = jnp.max(
        jnp.abs(forward.acceleration - target_acceleration),
        initial=0.0,
    )
    rollout_valid = jnp.all(trajectory.valid)
    rollout_successful = jnp.all(trajectory.status == 0)
    finite_timings = all(
        math.isfinite(value)
        for timing in (
            preparation_timing,
            fk_timing,
            jacobian_timing,
            inverse_timing,
            forward_timing,
            rollout_timing,
        )
        for value in timing.values()
    )
    passed = (
        adaptation.negotiation.valid
        and not adaptation.report.losses
        and _scalar_bool(kinematics.successful)
        and _scalar_float(jacobian_relative_residual) <= 1.0e-6
        and _scalar_bool(inverse.successful)
        and _scalar_bool(forward.successful)
        and _scalar_float(acceleration_error) <= 5.0e-5
        and _scalar_bool(rollout_valid)
        and _scalar_bool(rollout_successful)
        and finite_timings
    )

    return {
        "evidence": {
            "forward_dynamics_status": int(np.asarray(forward.status)),
            "forward_dynamics_successful": _scalar_bool(forward.successful),
            "forward_inverse_relative_residual": _scalar_float(
                forward.relative_inverse_forward_residual
            ),
            "forward_kinematics_successful": _scalar_bool(kinematics.successful),
            "inverse_dynamics_decomposition_residual": _scalar_float(
                inverse.decomposition_residual
            ),
            "inverse_dynamics_status": int(np.asarray(inverse.status)),
            "inverse_dynamics_successful": _scalar_bool(inverse.successful),
            "jacobian_relative_power_residual": _scalar_float(
                jacobian_relative_residual
            ),
            "maximum_acceleration_reconstruction_error": _scalar_float(
                acceleration_error
            ),
            "minimum_articulated_inertia": _scalar_float(
                forward.minimum_articulated_inertia
            ),
            "rollout_all_nodes_valid": _scalar_bool(rollout_valid),
            "rollout_backend_status": np.asarray(trajectory.backend_status).tolist(),
            "rollout_status": np.asarray(trajectory.status).tolist(),
            "urdf_loss_count": len(adaptation.report.losses),
            "urdf_negotiation_valid": adaptation.negotiation.valid,
        },
        "passed": passed,
        "size": {
            "body_count": joint_count + 1,
            "generalized_coordinates": articulation.nq,
            "generalized_velocities": articulation.nv,
            "joint_count": joint_count,
        },
        "timings_ms": {
            "forward_dynamics": forward_timing,
            "forward_kinematics": fk_timing,
            "frame_jacobian_actions": jacobian_timing,
            "inverse_dynamics": inverse_timing,
            "preparation": preparation_timing,
            "status_aware_batched_rollout": rollout_timing,
        },
        "work": {
            "rollout_batch_size": rollout_batch_size,
            "rollout_step_count": rollout_step_count,
            "rollout_transition_count": rollout_batch_size * rollout_step_count,
            "timed_calls_per_operation": repeats,
            "warmup_calls_per_operation": 1,
        },
    }


def run_benchmarks(*, smoke: bool = False) -> dict[str, Any]:
    joint_counts = (2,) if smoke else (2, 4, 8)
    repeats = 1 if smoke else 5
    rollout_batch_size = 2 if smoke else 32
    rollout_step_count = 2 if smoke else 32
    results = [
        _run_case(
            joint_count,
            repeats=repeats,
            rollout_batch_size=rollout_batch_size,
            rollout_step_count=rollout_step_count,
        )
        for joint_count in joint_counts
    ]
    return {
        "benchmark": "native-fixed-base-hinge-prismatic-articulation",
        "execution": "jax-jit-with-one-untimed-warmup",
        "passed": all(result["passed"] for result in results),
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark native fixed-base reduced articulation."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one two-joint case with minimal repetitions and rollout work",
    )
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            run_benchmarks(smoke=arguments.smoke),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
