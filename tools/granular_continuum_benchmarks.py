#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Constitutive and particle-to-continuum granular benchmarks."""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _timing(function, argument, *, repeats=20):
    started = time.perf_counter()
    result = function(argument)
    jax.block_until_ready(result)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        result = function(argument)
    jax.block_until_ready(result)
    return result, compile_seconds, (time.perf_counter() - started) / repeats


def _cohesion_benchmark():
    pair_count = 8192
    dtype = jnp.float64
    normal = jnp.broadcast_to(jnp.asarray([1.0, 0.0]), (pair_count, 2))
    vector = jnp.zeros((pair_count, 2), dtype=dtype)
    gap = jnp.linspace(0.0, 0.08, pair_count, dtype=dtype)
    valid = jnp.ones((pair_count,), dtype=bool)
    batch_template = phx.discretization.DEMContactBatch(
        normal,
        gap,
        jnp.maximum(-gap, 0.0),
        jnp.full((pair_count,), 0.25, dtype=dtype),
        vector,
        vector,
        jnp.zeros((pair_count,), dtype=dtype),
        vector,
        vector,
        vector,
        valid,
    )
    context = phx.discretization.DEMContactEvaluationContext(
        jnp.zeros((pair_count, 5), dtype=jnp.int64),
        valid,
        valid,
        jnp.ones((pair_count,), dtype=dtype),
        jnp.ones((pair_count,), dtype=dtype),
        jnp.full((pair_count,), 0.5, dtype=dtype),
        jnp.full((pair_count,), 0.5, dtype=dtype),
        jnp.zeros((pair_count,), dtype=jnp.int32),
        jnp.zeros((pair_count,), dtype=jnp.int32),
        jnp.asarray(1.0e-5, dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    plans = {
        "linear": phx.discretization.LinearCapillaryBridgePlan(0.072, 0.1, 1.0e-3, 0.2),
        "bagheri": phx.discretization.BagheriCapillaryBridgePlan(0.072, 0.1, 1.0e-3),
    }
    results = []
    for name, plan in plans.items():
        history = phx.discretization.DEMCohesionComponentHistory(
            valid,
            jnp.full((pair_count,), 1.0e-3, dtype=dtype),
            jnp.zeros((pair_count,), dtype=dtype),
            jnp.zeros((pair_count,), dtype=jnp.int32),
        )

        @jax.jit
        def evaluate(current_gap, plan=plan, history=history):
            batch = phx.discretization.DEMContactBatch(
                batch_template.normal,
                current_gap,
                jnp.maximum(-current_gap, 0.0),
                batch_template.effective_radius,
                batch_template.left_arm,
                batch_template.right_arm,
                batch_template.normal_velocity,
                batch_template.tangential_velocity,
                batch_template.left_angular_velocity,
                batch_template.right_angular_velocity,
                batch_template.valid,
            )
            response, _ = plan.evaluate(batch, None, history, context, materials)
            return response.force_magnitude

        force, compile_seconds, steady_seconds = _timing(evaluate, gap)
        results.append(
            {
                "law": name,
                "pair_count": pair_count,
                "compile_seconds": compile_seconds,
                "steady_seconds": steady_seconds,
                "pairs_per_second": pair_count / steady_seconds,
                "finite": bool(jnp.all(jnp.isfinite(force))),
            }
        )
    return results


def _coarse_graining_benchmark():
    side = 16
    count = side * side
    axis_values = np.linspace(0.05, 0.95, side)
    xx, yy = np.meshgrid(axis_values, axis_values, indexing="ij")
    position = jnp.asarray(np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.ones((count,)), ambient_dimension=2
    ).prepare()
    grid_axis = phx.discretization.UniformAxisSpec(32)
    grid = phx.discretization.TensorGridPlan(
        (grid_axis, grid_axis), axis_names=("x", "y")
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    left = np.arange(count - 1, dtype=np.int32)
    right = np.arange(1, count, dtype=np.int32)
    relation = phx.sparse.EdgeRelation(
        left,
        right,
        source_size=count,
        target_size=count,
    )
    pairs = phx.discretization.ParticlePairRelation(
        relation,
        left,
        right,
        source_support_id=particles.support.support_id,
        target_support_id=particles.support.support_id,
        same_set=True,
        unordered=True,
    )
    prepared = phx.discretization.ParticleCoarseGrainingPlan(
        phx.discretization.ParticleGridSplatPlan(grid), quadrature_order=4
    ).prepare(particles, count - 1)
    velocity = jnp.stack((position[:, 1], -position[:, 0]), axis=-1)
    displacement = position[left] - position[right]
    force = jnp.broadcast_to(jnp.asarray([1.0, -0.25]), displacement.shape)

    @jax.jit
    def evaluate(current_position):
        return prepared.evaluate(
            current_position,
            velocity,
            jnp.ones((count,)),
            jnp.full((count,), 1.0e-4),
            jnp.ones((count,), dtype=bool),
            pairs,
            displacement,
            force,
            jnp.ones((count - 1,), dtype=bool),
        )

    fields, compile_seconds, steady_seconds = _timing(evaluate, position, repeats=10)
    return {
        "particle_count": count,
        "interaction_count": count - 1,
        "target_count": 32 * 32,
        "compile_seconds": compile_seconds,
        "steady_seconds": steady_seconds,
        "successful": bool(fields.successful),
        "maximum_particle_balance_defect": float(fields.maximum_particle_balance_defect),
        "contact_stress_balance_defect": float(fields.contact_stress_balance_defect),
    }


def main():
    payload = {
        "benchmark": "granular-continuum",
        "cohesion": _cohesion_benchmark(),
        "coarse_graining": _coarse_graining_benchmark(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
