"""Compile and steady-state benchmarks for maximal cosmology profiles."""

from __future__ import annotations

import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _block(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(function, *args):
    start = time.perf_counter()
    value = function(*args)
    _block(value)
    return value, time.perf_counter() - start


def main() -> None:
    cosmo = phx.applications.cosmology
    count = 64
    coordinate = (jnp.arange(count, dtype=float) + 0.5) / count
    positions = jnp.stack(
        (
            coordinate,
            jnp.mod(13.0 * coordinate, 1.0),
            jnp.mod(29.0 * coordinate, 1.0),
        ),
        axis=-1,
    )
    masses = jnp.ones((count,)) / count
    octree = cosmo.ParticleOctreePlan3D((1.0, 1.0, 1.0), 3)
    prepare = eqx.filter_jit(octree.prepare)
    tree, tree_compile = _measure(prepare, positions, masses)
    _, tree_steady = _measure(prepare, positions, masses)

    bh = cosmo.BarnesHutGravityPlan(1.0, softening=0.01, opening_angle=0.5)
    bh_function = eqx.filter_jit(bh.evaluate)
    bh_result, bh_compile = _measure(bh_function, tree)
    _, bh_steady = _measure(bh_function, tree)

    fmm = cosmo.UniformFMMPlan(1.0, cosmo.CartesianExpansionSpace(1), softening=0.01)
    fmm_function = eqx.filter_jit(fmm.evaluate)
    fmm_result, fmm_compile = _measure(fmm_function, tree)
    _, fmm_steady = _measure(fmm_function, tree)

    fof = cosmo.PeriodicFoFFinderPlan((1.0, 1.0, 1.0), 0.05, 64)
    fof_function = eqx.filter_jit(
        lambda value: fof.find(
            jnp.arange(count),
            value,
            jnp.zeros_like(value),
            masses,
            jnp.ones((count,), dtype=bool),
        )
    )
    fof_result, fof_compile = _measure(fof_function, positions)
    _, fof_steady = _measure(fof_function, positions)

    report = {
        "particles": count,
        "octree_leaves": octree.leaf_count,
        "octree_compile_seconds": tree_compile,
        "octree_steady_seconds": tree_steady,
        "barnes_hut_compile_seconds": bh_compile,
        "barnes_hut_steady_seconds": bh_steady,
        "barnes_hut_finite": bool(bh_result.successful),
        "barnes_hut_interactions": int(
            bh_result.evidence.accepted_leaf_interactions
            + bh_result.evidence.direct_particle_interactions
        ),
        "fmm_compile_seconds": fmm_compile,
        "fmm_steady_seconds": fmm_steady,
        "fmm_finite": bool(fmm_result.successful),
        "fof_compile_seconds": fof_compile,
        "fof_steady_seconds": fof_steady,
        "fof_groups": int(jnp.sum(fof_result.group_active)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
