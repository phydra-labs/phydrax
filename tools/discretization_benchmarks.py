#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _timed(function, repeats):
    started = time.perf_counter()
    value = None
    for _ in range(repeats):
        value = function()
        jax.block_until_ready(value)
    return value, (time.perf_counter() - started) / repeats


def _triangular_grid(width):
    axis = np.linspace(0.0, 1.0, width)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    vertices = np.stack((x.reshape((-1,)), y.reshape((-1,))), axis=1)
    faces = []
    for row in range(width - 1):
        for column in range(width - 1):
            lower_left = row * width + column
            lower_right = lower_left + 1
            upper_left = lower_left + width
            upper_right = upper_left + 1
            faces.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    return jnp.asarray(vertices), jnp.asarray(faces, dtype=jnp.int32)


def _tensor_case(size, repeats):
    axis = phx.discretization.FourierAxisSpec(size).materialize(0.0, 1.0)
    started = time.perf_counter()
    discretization = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    preparation = time.perf_counter() - started
    state = jnp.sin(2.0 * jnp.pi * axis.nodes)
    action = eqx.filter_jit(discretization.laplacian)
    action(state).block_until_ready()
    value, steady = _timed(lambda: action(state), repeats)
    return {
        "preparation_seconds": preparation,
        "steady_action_seconds": steady,
        "points": size,
        "maximum_absolute_value": float(jnp.max(jnp.abs(value))),
    }


def _fem_case(width, repeats):
    vertices, faces = _triangular_grid(width)
    started = time.perf_counter()
    mesh = phx.discretization.CellMesh.from_triangles(vertices, faces)
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", 1),
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.WeakForm(
        "benchmark-diffusion",
        "u",
        (phx.equations.DiffusionTerm("u"),),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free"
        ),
    )
    preparation = time.perf_counter() - started
    state = jnp.sin(jnp.pi * vertices[:, 0]) * jnp.sin(jnp.pi * vertices[:, 1])
    matrix_free = eqx.filter_jit(compiled.full_residual)
    sparse = eqx.filter_jit(discretization.stiffness.mv)
    matrix_free(state).block_until_ready()
    sparse(state).block_until_ready()
    value, matrix_free_steady = _timed(lambda: matrix_free(state), repeats)
    _, sparse_steady = _timed(lambda: sparse(state), repeats)
    return {
        "preparation_seconds": preparation,
        "matrix_free_steady_seconds": matrix_free_steady,
        "sparse_steady_seconds": sparse_steady,
        "vertices": int(vertices.shape[0]),
        "cells": int(faces.shape[0]),
        "dofs": int(discretization.dof_maps[0].global_dof_count),
        "routes": int(discretization.stiffness.relation.route_shape[0]),
        "maximum_absolute_value": float(jnp.max(jnp.abs(value))),
    }


def _smoothing_case(width, repeats):
    vertices, faces = _triangular_grid(width)
    mesh = phx.discretization.CellMesh.from_triangles(vertices, faces)
    smoothing = phx.discretization.fem.smoothing
    constitutive = smoothing.plane_stress_matrix(1.0, 0.3)
    started = time.perf_counter()
    edge = smoothing.SmoothedElasticityPlan("ES", mesh, constitutive)
    node = smoothing.SmoothedElasticityPlan("NS", mesh, constitutive)
    preparation = time.perf_counter() - started
    edge_action = eqx.filter_jit(edge.stiffness)
    node_action = eqx.filter_jit(node.stiffness)
    edge_action(vertices).block_until_ready()
    node_action(vertices).block_until_ready()
    edge_stiffness, edge_steady = _timed(lambda: edge_action(vertices), repeats)
    node_stiffness, node_steady = _timed(lambda: node_action(vertices), repeats)
    return {
        "preparation_seconds": preparation,
        "edge_steady_seconds": edge_steady,
        "node_steady_seconds": node_steady,
        "edge_patches": int(edge.layout.owner_entities.size),
        "node_patches": int(node.layout.owner_entities.size),
        "edge_matrix_norm": float(jnp.linalg.norm(edge_stiffness)),
        "node_matrix_norm": float(jnp.linalg.norm(node_stiffness)),
    }


def _finite_volume_case(width, repeats):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(width),
            phx.discretization.UniformCellAxisSpec(width),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    started = time.perf_counter()
    discretization = phx.discretization.FiniteVolumePlan(grid, field_name="u").prepare()
    velocity = (0.7, -0.2)
    system = phx.equations.ScalarConservationSystem(
        2,
        lambda state, axis, args: velocity[axis] * state,
        lambda left, right, axis, args: jnp.full(left.shape[:-1], abs(velocity[axis])),
        system_id="benchmark-transport",
    )
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    problem = phx.equations.ConservationProblemIR(
        "transport",
        "u",
        system,
        phx.discretization.FiniteVolumeBoundarySet(("x", "y"), (pair, pair)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.MUSCLReconstruction(),
        phx.discretization.HLLFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    preparation = time.perf_counter() - started
    state = jnp.sin(jnp.pi * discretization.cell_centers[..., :1])
    action = eqx.filter_jit(lambda value: compiled(jnp.asarray(0.0), value))
    action(state).block_until_ready()
    value, steady = _timed(lambda: action(state), repeats)
    return {
        "preparation_seconds": preparation,
        "steady_action_seconds": steady,
        "cells": int(np.prod(discretization.cell_shape)),
        "faces": int(
            sum(np.prod(layout.shape) for layout in discretization.face_layouts)
        ),
        "maximum_absolute_value": float(jnp.max(jnp.abs(value))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-size", type=int, default=256)
    parser.add_argument("--mesh-width", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.tensor_size < 4 or arguments.mesh_width < 2 or arguments.repeats < 1:
        raise ValueError("Benchmark sizes and repeats are below their valid minimum.")
    report = {
        "tensor": _tensor_case(arguments.tensor_size, arguments.repeats),
        "finite_element": _fem_case(arguments.mesh_width, arguments.repeats),
        "smoothed_finite_element": _smoothing_case(
            arguments.mesh_width,
            arguments.repeats,
        ),
        "structured_finite_volume": _finite_volume_case(
            arguments.mesh_width,
            arguments.repeats,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
