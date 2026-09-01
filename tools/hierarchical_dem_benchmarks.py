#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Repeatable broad-PSD DEM neighborhood and evaluation benchmark."""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _problem():
    side = 24
    axis = np.linspace(-0.5, 0.5, side, endpoint=False) + 0.5 / side
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    position = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)
    count = position.shape[0]
    radii = np.full((count,), 0.003)
    medium = np.linspace(8, count - 9, 12, dtype=int)
    large = np.asarray([0, 191, 383, 575])
    radii[medium] = 0.05
    radii[large] = 0.2
    particles = phx.discretization.ParticleSetPlan(
        np.arange(count), np.ones((count,)), ambient_dimension=2
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        radii, np.zeros((count,), dtype=np.int32)
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([1.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
        ),
        maximum_overlap_fraction=1.0,
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "broad-psd-dem-benchmark", materials, gravity=jnp.zeros((2,))
    )
    box = phx.discretization.ParticleBox(
        jnp.asarray([-0.5, -0.5]),
        jnp.asarray([0.5, 0.5]),
        periodic_axes=(True, True),
    )
    return (
        particles,
        spheres,
        method,
        problem,
        box,
        jnp.asarray(position),
        jnp.asarray(radii),
    )


def _compile(backend):
    particles, spheres, method, problem, box, position, radii = _problem()
    pair_capacity = particles.capacity * (particles.capacity - 1) // 2
    if backend == "single-cell-scale":
        neighborhood = phx.discretization.CellListParticleNeighborhoodPlan(
            0.4,
            180,
            pair_capacity,
            box,
            maximum_candidate_slots=2_000_000,
        )
    elif backend == "sparse-hierarchy":
        neighborhood = phx.discretization.HierarchicalRadiusParticleNeighborhoodPlan(
            radii,
            jnp.asarray([0.002, 0.005, 0.08, 0.25]),
            8,
            8192,
            box,
            maximum_candidate_slots=2_000_000,
        )
    else:
        raise ValueError("Unknown DEM neighborhood backend.")
    compiled = phx.equations.compile_discrete_element_problem(
        problem, particles, spheres, method, neighborhood=neighborhood
    )
    state = compiled.initialize_state(0.0, position, jnp.zeros_like(position))
    return compiled, state


def _measure(backend, repeats=10):
    compiled, state = _compile(backend)

    @jax.jit
    def evaluate(current):
        result = compiled.dynamics.evaluate(
            jnp.asarray(0.0), current, jnp.asarray(1.0e-5), None
        )
        return result.loads.total.force, result.neighborhood.candidate_pair_count

    compile_start = time.perf_counter()
    force, candidates = evaluate(state)
    jax.block_until_ready(force)
    compile_seconds = time.perf_counter() - compile_start
    start = time.perf_counter()
    for _ in range(repeats):
        force, candidates = evaluate(state)
    jax.block_until_ready(force)
    elapsed = time.perf_counter() - start
    neighborhood = compiled.dynamics.neighborhood.build(state.kinematics.position)
    return {
        "backend": backend,
        "particle_count": compiled.dynamics.bodies.capacity,
        "candidate_pair_count": int(candidates),
        "accepted_pair_count": int(neighborhood.pair_count),
        "compile_seconds": compile_seconds,
        "evaluation_seconds": elapsed / repeats,
        "successful": bool(neighborhood.successful),
    }


def main():
    results = [_measure("single-cell-scale"), _measure("sparse-hierarchy")]
    print(json.dumps({"benchmark": "broad-psd-dem", "results": results}, indent=2))


if __name__ == "__main__":
    main()
