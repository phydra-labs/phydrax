"""Compile and steady-state benchmarks for spectral statistics and inverse realization."""

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
    results = {}
    for count in (16, 32):
        shape = (count, count, count)
        maximum = jnp.sqrt(3.0) * jnp.pi * count
        shells = phx.discretization.PeriodicFourierShellPlan(
            shape,
            (1.0, 1.0, 1.0),
            jnp.linspace(0.0, maximum, 32),
        )
        coordinate = (jnp.arange(count) + 0.5) / count
        x, y, z = jnp.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
        field = (
            jnp.cos(2.0 * jnp.pi * x)
            + 0.5 * jnp.cos(4.0 * jnp.pi * y)
            + 0.25 * jnp.cos(6.0 * jnp.pi * z)
        )
        transform = eqx.filter_jit(shells.transform)
        transformed, compile_seconds = _measure(transform, field)
        _, transform_seconds = _measure(transform, field)
        auto = eqx.filter_jit(shells.auto_power)
        _, auto_compile = _measure(auto, transformed)
        auto_result, auto_seconds = _measure(auto, transformed)
        discrepancy = eqx.filter_jit(shells.discrepancy)
        shifted = transform(jnp.roll(field, 1, axis=0))
        _, discrepancy_compile = _measure(discrepancy, transformed, shifted)
        discrepancy_result, discrepancy_seconds = _measure(
            discrepancy, transformed, shifted
        )
        results[f"fourier_{count}3"] = {
            "cells": count**3,
            "weighted_modes": int(jnp.sum(auto_result.weighted_mode_count)),
            "valid_shells": int(jnp.sum(auto_result.valid_shells)),
            "transform_compile_seconds": compile_seconds,
            "transform_steady_seconds": transform_seconds,
            "auto_compile_seconds": auto_compile,
            "auto_steady_seconds": auto_seconds,
            "discrepancy_compile_seconds": discrepancy_compile,
            "discrepancy_steady_seconds": discrepancy_seconds,
            "discrepancy": float(discrepancy_result.total_weighted_value),
        }

    grid_shape = (4, 4, 4)
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for count in grid_shape
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    capacity = 4**3
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(capacity),
        jnp.full((capacity,), 1.0 / capacity),
        ambient_dimension=3,
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    coordinate = tuple((jnp.arange(4) + 0.5) / 4.0 for _ in range(3))
    target_positions = jnp.stack(
        jnp.meshgrid(*coordinate, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    target = transfer.deposit_content(
        transfer.build(target_positions), particles.masses
    ).density
    layout = phx.observation.CoordinateLayout(
        tuple(f"density:{index}" for index in range(target.size))
    )
    observation = phx.solver.FieldObservationPlan(
        lambda value, args: value,
        target,
        phx.observation.CholeskyCovarianceAction(jnp.eye(target.size), layout),
        observation_id="inverse-benchmark",
    )
    inverse = phx.applications.cosmology.ParticleFieldRealizationPlan(
        transfer, observation, plan_id="inverse-benchmark"
    )
    initial = jnp.mod(target_positions + 0.01, 1.0)
    objective = eqx.filter_jit(lambda positions: inverse.objective(positions))
    _, inverse_compile = _measure(objective, initial)
    objective_value, inverse_seconds = _measure(objective, initial)
    gradient = eqx.filter_jit(lambda positions: inverse.value_and_gradient(positions))
    _, gradient_compile = _measure(gradient, initial)
    gradient_result, gradient_seconds = _measure(gradient, initial)
    results["inverse_4_3"] = {
        "particles": capacity,
        "grid_cells": target.size,
        "objective_compile_seconds": inverse_compile,
        "objective_steady_seconds": inverse_seconds,
        "gradient_compile_seconds": gradient_compile,
        "gradient_steady_seconds": gradient_seconds,
        "objective": float(objective_value),
        "gradient_finite": bool(jnp.all(jnp.isfinite(gradient_result[1]))),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
