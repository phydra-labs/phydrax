#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualify the affine-P1 phenomenological monodomain foundation."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from phydrax.applications.cardiovascular.electrophysiology._activation import (
    activation_observation_result,
    ActivationObservationPlan,
    ChordConductionVelocityPlan,
    commit_activation_observation,
    evaluate_activation_observation,
    evaluate_chord_conduction_velocity,
    initialize_activation_observation,
)
from phydrax.applications.cardiovascular.electrophysiology._aliev_panfilov import (
    AlievPanfilovParameters,
)
from phydrax.applications.cardiovascular.electrophysiology._monodomain import (
    CellStimulusPulse,
    CellwiseDiffusivity,
    monodomain_state_identity,
    PhenomenologicalMonodomainPlan,
    read_monodomain_checkpoint,
    run_monodomain_steps,
    write_monodomain_checkpoint,
)
from phydrax.discretization import (
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)


def tetrahedral_slab(cube_count: int, /) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a conforming one-by-one-by-``cube_count`` Freudenthal slab."""

    coordinates = [
        (float(i), float(j), float(k))
        for i in range(cube_count + 1)
        for j in range(2)
        for k in range(2)
    ]

    def vertex(i: int, j: int, k: int) -> int:
        return 4 * i + 2 * j + k

    tetrahedra: list[tuple[int, int, int, int]] = []
    for i in range(cube_count):
        v000 = vertex(i, 0, 0)
        v001 = vertex(i, 0, 1)
        v010 = vertex(i, 1, 0)
        v011 = vertex(i, 1, 1)
        v100 = vertex(i + 1, 0, 0)
        v101 = vertex(i + 1, 0, 1)
        v110 = vertex(i + 1, 1, 0)
        v111 = vertex(i + 1, 1, 1)
        tetrahedra.extend(
            (
                (v000, v100, v110, v111),
                (v000, v110, v010, v111),
                (v000, v010, v011, v111),
                (v000, v011, v001, v111),
                (v000, v001, v101, v111),
                (v000, v101, v100, v111),
            )
        )
    return jnp.asarray(coordinates), jnp.asarray(tetrahedra, dtype=jnp.int32)


def prepared_slab(cube_count: int, dt_ms: float, pulse_ms: float, /):
    coordinates, tetrahedra = tetrahedral_slab(cube_count)
    mesh = CellMesh.from_tetrahedra(coordinates, tetrahedra)
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("activation", lagrange_element("tetrahedron", 1)),
    ).prepare()
    fibers = jnp.tile(jnp.asarray(((1.0, 0.0, 0.0),)), (tetrahedra.shape[0], 1))
    diffusivity = CellwiseDiffusivity.from_fibers(fibers, 0.5, 0.05)
    reaction = AlievPanfilovParameters(0.05, 0.15, 8.0, 0.002, 0.2, 0.3, 12.9)
    stimulus = CellStimulusPulse(tuple(range(6)), 0.0, pulse_ms, 3.0)
    plan = PhenomenologicalMonodomainPlan(
        discretization, diffusivity, reaction, pulses=(stimulus,)
    )
    return coordinates, plan.prepare(dt_ms)


def qualify(cube_count: int, dt_ms: float, step_count: int, pulse_ms: float, /) -> dict:
    coordinates, runtime = prepared_slab(cube_count, dt_ms, pulse_ms)
    initial = runtime.initialize(
        jnp.zeros(runtime.plan.node_count), jnp.zeros(runtime.plan.node_count)
    )
    selected = tuple(4 * index for index in range(cube_count + 1))
    activation_plan = ActivationObservationPlan(
        runtime.plan.node_count, selected, threshold=0.5
    )
    online = initialize_activation_observation(
        activation_plan, runtime.split(initial)[0], time_ms=0.0
    )
    state = initial
    accepted_steps = 0
    integration_success = True
    for _ in range(step_count):
        candidate = runtime.evaluate(state)
        if not bool(candidate.evidence.successful):
            integration_success = False
            break
        state = runtime.commit(candidate, state)
        activation, _ = runtime.split(state)
        observation = evaluate_activation_observation(
            activation_plan, online, activation, state.time_ms
        )
        if not bool(observation.evidence.successful):
            integration_success = False
            break
        online = commit_activation_observation(observation, online)
        accepted_steps += 1
    activation_result = activation_observation_result(activation_plan, online)
    chord_plan = ChordConductionVelocityPlan.from_coordinates(
        activation_plan, coordinates, selected[0], selected[-1]
    )
    chord_result = evaluate_chord_conduction_velocity(chord_plan, activation_result)

    split = step_count // 2
    prefix = run_monodomain_steps(runtime, initial, split)
    uninterrupted = run_monodomain_steps(runtime, initial, step_count)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "ep-foundation.phx"
        archive = write_monodomain_checkpoint(runtime, prefix.state, checkpoint_path)
        restored = read_monodomain_checkpoint(runtime, checkpoint_path)
        resumed = run_monodomain_steps(runtime, restored, step_count - split)
        checkpoint_id = archive.manifest.checkpoint_id
    replay_identical = (
        bool(uninterrupted.successful)
        and bool(resumed.successful)
        and resumed.state_id == uninterrupted.state_id
        and np.array_equal(
            np.asarray(resumed.state.values), np.asarray(uninterrupted.state.values)
        )
    )

    dense_stiffness = np.asarray(runtime.stiffness.as_dense())
    stiffness_symmetry_residual = float(
        np.max(np.abs(dense_stiffness - dense_stiffness.T))
    )
    mass = np.asarray(runtime.lumped_mass)
    activation_times = np.asarray(activation_result.activation_times_ms)
    all_activated = bool(np.all(np.asarray(activation_result.activated)))
    passed = (
        integration_success
        and accepted_steps == step_count
        and bool(activation_result.successful)
        and all_activated
        and bool(chord_result.successful)
        and replay_identical
        and bool(np.all(mass > 0.0))
        and stiffness_symmetry_residual <= 2.0e-6
        and dt_ms <= runtime.diffusion_step_limit_ms
    )
    return {
        "case": "affine-p1-tetrahedral-slab-propagation",
        "fidelity_route": "PhenomenologicalMonodomainPlan",
        "cube_count": cube_count,
        "node_count": runtime.plan.node_count,
        "cell_count": runtime.plan.cell_count,
        "dt_ms": dt_ms,
        "step_count": step_count,
        "accepted_steps": accepted_steps,
        "pulse_interval_ms": [0.0, pulse_ms],
        "pulse_interval_convention": "grid-aligned-half-open",
        "diffusion_step_limit_ms": runtime.diffusion_step_limit_ms,
        "minimum_lumped_mass_mm3": float(np.min(mass)),
        "stiffness_symmetry_residual": stiffness_symmetry_residual,
        "activation_times_ms": activation_times.tolist(),
        "all_selected_nodes_activated": all_activated,
        "chord_velocity_mm_per_ms": float(chord_result.velocity_mm_per_ms),
        "checkpoint_id": checkpoint_id,
        "replay_state_id": monodomain_state_identity(runtime, resumed.state),
        "replay_identical": replay_identical,
        "physical_ionic_current_claim": False,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubes", type=int, default=4)
    parser.add_argument("--dt-ms", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--pulse-ms", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.cubes < 2 or arguments.steps <= 0:
        raise ValueError("Qualification requires at least two cubes and one step.")
    report = qualify(
        arguments.cubes, arguments.dt_ms, arguments.steps, arguments.pulse_ms
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
