"""Benchmark provider-authoritative MJX built-in muscle projection workflow."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.applications.robotics import prepare_mjx_adapter
from phydrax.dynamics import PlantStepContext


def _xml(muscles: int) -> str:
    actuators = "".join(
        f'<muscle name="muscle-{index:03d}" tendon="path" force="{800 + 10 * index}"/>'
        for index in range(muscles)
    )
    return f"""
<mujoco>
  <compiler autolimits="true"/>
  <option timestep="0.001" solver="Newton"/>
  <worldbody>
    <site name="origin" pos="-0.4 0 0.15"/><site name="side" pos="0 0.3 0.15"/>
    <geom name="wrap" type="sphere" pos="0 0 0.15" size="0.1" contype="0" conaffinity="0"/>
    <body pos="0.4 0 0.15">
      <joint name="hinge" type="hinge" range="-0.6 0.6"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/>
      <site name="insertion" pos="0.3 0 0"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="path">
      <site site="origin"/><geom geom="wrap" sidesite="side"/><site site="insertion"/>
    </spatial>
  </tendon>
  <actuator>
    {actuators}<motor name="assist" joint="hinge"/>
  </actuator>
</mujoco>
"""


def benchmark(muscles: int, iterations: int) -> dict[str, object]:
    import mujoco

    model = mujoco.MjModel.from_xml_string(_xml(muscles))
    adapter = prepare_mjx_adapter(model, device=jax.devices("cpu")[0])
    projection = adapter.prepare_muscle_projection()
    excitation = jnp.linspace(0.05, 0.95, muscles)

    def rollout(initial):
        def step(state, _):
            control = projection.scatter_control(
                adapter.control(state),
                excitation,
            )
            context = PlantStepContext(
                state.time,
                state.time + jnp.asarray(model.opt.timestep, dtype=state.time.dtype),
                state.step_index,
            )
            stepped = adapter.step(
                context,
                state,
                control,
                adapter.parameters,
            )
            refreshed = adapter.refresh(stepped.accepted_state)
            snapshot = projection.snapshot(refreshed.accepted_state)
            return refreshed.accepted_state, (
                snapshot.raw_force_N.values,
                snapshot.successful,
            )

        return jax.lax.scan(step, initial, xs=None, length=iterations)

    initial = adapter.reset(jax.random.key(7), adapter.parameters).accepted_state
    action = eqx.filter_jit(rollout)
    start = time.perf_counter()
    first = action(initial)
    first[1][0].block_until_ready()
    compile_and_first_s = time.perf_counter() - start
    start = time.perf_counter()
    result = action(initial)
    result[1][0].block_until_ready()
    elapsed = time.perf_counter() - start
    return {
        "device": adapter.device,
        "muscle_count": muscles,
        "control_count": model.nu,
        "iterations": iterations,
        "compile_and_first_seconds": compile_and_first_s,
        "execution_seconds": elapsed,
        "step_forward_snapshot_per_second": iterations / elapsed,
        "successful": bool(jnp.all(result[1][1])),
        "raw_force_checksum_N": float(jnp.sum(result[1][0][-1])),
        "provider": adapter.provenance.provider,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscles", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/robotics_mjx_muscle_projection.json"),
    )
    arguments = parser.parse_args()
    if arguments.muscles <= 0 or arguments.iterations <= 0:
        raise ValueError("muscles and iterations must be positive.")
    payload = benchmark(arguments.muscles, arguments.iterations)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
