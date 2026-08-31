#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Small repeatable dense/cell spherical DEM throughput benchmark."""

import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def _configuration(count_per_axis=8):
    axis = jnp.linspace(-0.315, 0.315, count_per_axis)
    xx, yy = jnp.meshgrid(axis, axis, indexing="ij")
    position = jnp.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)
    count = position.shape[0]
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count, dtype=jnp.int64),
        jnp.ones((count,)),
        ambient_dimension=2,
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.full((count,), 0.05), jnp.zeros((count,), dtype=jnp.int32)
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([1.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4),
            tangential=phx.discretization.CundallStrackTangentialPlan(2.5e3),
        ),
        maximum_overlap_fraction=0.25,
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "dem-benchmark", materials, gravity=jnp.zeros((2,))
    )
    return particles, spheres, method, problem, position


def _compiled(backend, kernel_backend):
    particles, spheres, method, problem, position = _configuration()
    count = particles.capacity
    box = phx.discretization.ParticleBox(
        jnp.asarray([-0.5, -0.5]),
        jnp.asarray([0.5, 0.5]),
        periodic_axes=(False, False),
    )
    if backend == "dense":
        neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2
        )
        realization = "dense_pairs"
    elif backend == "cell":
        neighborhood = phx.discretization.CellListParticleNeighborhoodPlan(
            0.1, 8, 1024, box
        )
        realization = "cell_edge_list"
    else:
        base = phx.discretization.CellListParticleNeighborhoodPlan(0.121, 8, 1024, box)
        neighborhood = phx.discretization.VerletParticleNeighborhoodPlan(base, 0.1, 0.02)
        realization = "cell_edge_list"
    execution = phx.discretization.ParticleExecutionPolicy(
        realization=realization,
        accumulation="deterministic",
        kernel_backend=kernel_backend,
    )
    compiled = phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        spheres,
        method,
        neighborhood=neighborhood,
        execution=execution,
    )
    state = compiled.initialize_state(0.0, position, jnp.zeros_like(position))
    return compiled, state


def _tree_bytes(tree):
    return sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(tree))


def _measure(backend, kernel_backend, repeats=20):
    compiled, state = _compiled(backend, kernel_backend)
    step_size = jnp.asarray(1.0e-5)

    @jax.jit
    def step(current):
        return compiled.dynamics.step_detailed(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            current,
            step_size,
            None,
        ).accepted_state

    @jax.jit
    def evaluate(current):
        return compiled.dynamics.evaluate(
            jnp.asarray(0.0), current, step_size, None
        ).loads.total.force

    compile_start = time.perf_counter()
    state = step(state)
    jax.block_until_ready(state.kinematics.position)
    compile_seconds = time.perf_counter() - compile_start
    start = time.perf_counter()
    for _ in range(repeats):
        state = step(state)
    jax.block_until_ready(state.kinematics.position)
    elapsed = time.perf_counter() - start
    neighborhood = compiled.dynamics.neighborhood.build(state.kinematics.position)
    keys = compiled.dynamics.pair_key_space.keys(neighborhood.pair_relation)

    @jax.jit
    def remap(history):
        return phx.discretization.match_particle_pair_keys(
            history.pair_keys,
            history.valid,
            keys.keys,
            keys.valid,
        ).source_indices

    force = evaluate(state)
    jax.block_until_ready(force)
    evaluation_start = time.perf_counter()
    for _ in range(repeats):
        force = evaluate(state)
    jax.block_until_ready(force)
    evaluation_seconds = (time.perf_counter() - evaluation_start) / repeats
    indices = remap(state.particle_history)
    jax.block_until_ready(indices)
    remap_start = time.perf_counter()
    for _ in range(repeats):
        indices = remap(state.particle_history)
    jax.block_until_ready(indices)
    remap_seconds = (time.perf_counter() - remap_start) / repeats
    diagnostics = compiled.diagnostics(0.0, state)
    return {
        "backend": backend,
        "kernel_backend": kernel_backend,
        "particle_count": compiled.dynamics.bodies.capacity,
        "pair_capacity": compiled.dynamics.neighborhood.pair_capacity,
        "active_contacts": int(diagnostics.active_contacts),
        "candidate_pair_count": int(neighborhood.candidate_pair_count),
        "state_bytes": _tree_bytes(state),
        "candidate_relation_bytes": _tree_bytes(neighborhood.pair_relation),
        "contact_history_bytes": _tree_bytes(state.particle_history),
        "compile_seconds": compile_seconds,
        "steady_step_seconds": elapsed / repeats,
        "steps_per_second": repeats / elapsed,
        "evaluation_seconds": evaluation_seconds,
        "history_remap_seconds": remap_seconds,
        "candidate_pairs_per_second": (
            int(neighborhood.candidate_pair_count) * repeats / elapsed
        ),
        "active_contacts_per_second": (
            int(diagnostics.active_contacts) * repeats / elapsed
        ),
        "successful": bool(diagnostics.successful),
        "relative_energy_residual": float(
            diagnostics.energy.last_relative_energy_residual
        ),
        "contact_balance_loss": float(diagnostics.energy.cumulative_contact_balance_loss),
        "neighborhood_rebuild_count": int(diagnostics.neighborhood_rebuild_count),
        "method_id": compiled.dynamics.method.method_id,
        "precision_id": compiled.dynamics.precision.policy_id,
        "execution_id": compiled.dynamics.execution.policy_id,
    }


def main():
    cases = (
        ("dense", "reference"),
        ("dense", "dense_fused"),
        ("cell", "reference"),
        ("cell", "cell_fused"),
        ("verlet", "reference"),
        ("verlet", "verlet_fused"),
    )
    results = [_measure(backend, kernel) for backend, kernel in cases]
    print(json.dumps({"benchmark": "spherical-dem", "results": results}, indent=2))


if __name__ == "__main__":
    main()
