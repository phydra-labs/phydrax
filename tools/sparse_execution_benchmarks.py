#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _ready(value):
    return jax.block_until_ready(value)


def _time(function, *arguments, repeats=3):
    compiled = jax.jit(function)
    started = perf_counter()
    value = _ready(compiled(*arguments))
    first = (perf_counter() - started) * 1.0e3
    timings = []
    for _ in range(repeats):
        started = perf_counter()
        value = _ready(compiled(*arguments))
        timings.append((perf_counter() - started) * 1.0e3)
    return value, first, float(np.median(timings))


def _array_bytes(value):
    return int(
        sum(
            leaf.size * leaf.dtype.itemsize
            for leaf in jax.tree_util.tree_leaves(value)
            if isinstance(leaf, jax.Array)
        )
    )


def _group_case():
    count = 4096
    keys = (jnp.arange(count, dtype=jnp.int32) * 17) % 512
    plan = phx.sparse.KeyGroupPlan(
        count,
        512,
        511,
        maximum_group_size=8,
    )
    state, first, steady = _time(
        lambda value: plan.build(value, jnp.ones(value.shape, dtype=bool)), keys
    )
    return {
        "items": count,
        "logical_keys": 512,
        "active_groups": int(state.evidence.required_groups),
        "state_bytes": _array_bytes(state),
        "compile_and_first_ms": first,
        "steady_ms": steady,
        "successful": bool(state.evidence.successful),
    }


def _relation_case():
    count = 4096
    target_count = 512
    targets = (jnp.arange(count, dtype=jnp.int32) * 29) % target_count
    relation = phx.sparse.EdgeRelation(
        jnp.arange(count, dtype=jnp.int32),
        targets,
        source_size=count,
        target_size=target_count,
    )
    execution = phx.sparse.RelationExecutionPlan().prepare(relation)
    values = jnp.sin(jnp.arange(count, dtype=jnp.float64))
    fast, fast_first, fast_steady = _time(
        lambda payload: execution.reduce(payload, accumulation="fast")[0], values
    )
    deterministic, deterministic_first, deterministic_steady = _time(
        lambda payload: execution.reduce(payload, accumulation="deterministic")[0],
        values,
    )
    return {
        "routes": count,
        "targets": target_count,
        "maximum_contention": int(execution.groups.evidence.maximum_group_size),
        "fast_compile_and_first_ms": fast_first,
        "fast_steady_ms": fast_steady,
        "deterministic_compile_and_first_ms": deterministic_first,
        "deterministic_steady_ms": deterministic_steady,
        "fast_deterministic_defect": float(jnp.max(jnp.abs(fast - deterministic))),
    }


def _raster_case():
    image = phx.imaging.ImagePlaneSupport((128, 128))
    row, column = jnp.meshgrid(
        jnp.linspace(8.0, 120.0, 16),
        jnp.linspace(8.0, 120.0, 16),
        indexing="ij",
    )
    coordinates = jnp.stack((row, column), axis=-1).reshape((-1, 2))
    amplitudes = jnp.ones((coordinates.shape[0],))
    reference = phx.rendering.GaussianRasterizer(
        4,
        cutoff=3.0,
        execution=phx.rendering.GaussianRasterExecutionPlan("reference"),
    )
    tiled = phx.rendering.GaussianRasterizer(
        4,
        cutoff=3.0,
        execution=phx.rendering.GaussianRasterExecutionPlan(
            "tiled", tile_shape=(16, 16), accumulation="deterministic"
        ),
    )
    reference_result, reference_first, reference_steady = _time(
        lambda points: reference.render(image, points, amplitudes, 1.0), coordinates
    )
    tiled_result, tiled_first, tiled_steady = _time(
        lambda points: tiled.render(image, points, amplitudes, 1.0), coordinates
    )
    return {
        "particles": int(coordinates.shape[0]),
        "image_pixels": int(np.prod(image.image_shape)),
        "routes": int(tiled_result.evidence.route_count),
        "reference_compile_and_first_ms": reference_first,
        "reference_steady_ms": reference_steady,
        "tiled_compile_and_first_ms": tiled_first,
        "tiled_steady_ms": tiled_steady,
        "image_defect": float(
            jnp.max(jnp.abs(reference_result.image - tiled_result.image))
        ),
        "successful": bool(reference_result.successful & tiled_result.successful),
    }


def _periodic_index(count):
    plan = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y"),
    )
    return plan.prepare_index_space(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))


def _lbm_case():
    count = 32
    index = _periodic_index(count)
    rows, columns = np.meshgrid(np.arange(8, 24), np.arange(8, 24), indexing="ij")
    fluid = (rows * count + columns).reshape((-1,))
    prepared = phx.discretization.SparseLatticeBoltzmannPlan(
        index,
        phx.discretization.D2Q9(),
        (4, 4),
        16,
    ).prepare(fluid)
    state = prepared.initialize_state(1.0, jnp.asarray((0.01, 0.0)))
    result, first, steady = _time(lambda value: prepared.step(value, 1.0), state)
    return {
        "logical_cells": count * count,
        "fluid_cells": int(fluid.size),
        "storage_cells": prepared.storage_capacity,
        "state_bytes": _array_bytes(state),
        "compile_and_first_ms": first,
        "steady_ms": steady,
        "mass_defect": float(jnp.abs(result.diagnostics.mass_defect)),
        "successful": bool(result.successful),
    }


def _mpm_case():
    count = 32
    grid_plan = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(count, periodic=True, endpoint=False),
            phx.discretization.UniformAxisSpec(count, periodic=True, endpoint=False),
        ),
        axis_names=("x", "y"),
    )
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    index = grid_plan.prepare_index_space(bounds)
    row, column = jnp.meshgrid(
        jnp.linspace(0.2, 0.3, 4),
        jnp.linspace(0.2, 0.3, 4),
        indexing="ij",
    )
    position = jnp.stack((row, column), axis=-1).reshape((-1, 2))
    volume = jnp.full((position.shape[0],), 0.001)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0]), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        index,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(particles)
    topology = phx.discretization.SparseBlockTopologyPlan(
        index, (4, 4), 16, layout=index.vertices()
    )
    storage = phx.discretization.BlockSparseMPMNodalStoragePlan(topology)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "sparse-execution-benchmark",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            bounds, periodic=(True, True), support_margin=0.0
        ),
        nodal_storage=storage,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    state = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    result, first, steady = _time(
        lambda value: compiled.dynamics.step_detailed(value, 1.0e-4, arguments),
        state,
    )
    return {
        "logical_nodes": count * count,
        "storage_nodes": storage.storage_capacity,
        "active_blocks": int(state.storage_state.evidence.required_blocks),
        "state_bytes": _array_bytes(state),
        "compile_and_first_ms": first,
        "steady_ms": steady,
        "successful": bool(result.successful),
    }


def _flip_case():
    count = 32
    index = _periodic_index(count)
    row, column = jnp.meshgrid(
        jnp.linspace(0.2, 0.3, 4),
        jnp.linspace(0.2, 0.3, 4),
        indexing="ij",
    )
    position = jnp.stack((row, column), axis=-1).reshape((-1, 2))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0]),
        jnp.ones((position.shape[0],)),
        ambient_dimension=2,
    ).prepare()
    closure = tuple(
        (row_offset, column_offset)
        for row_offset in (-1, 0, 1)
        for column_offset in (-1, 0, 1)
    )
    cell = phx.discretization.SparseBlockTopologyPlan(
        index,
        (4, 4),
        16,
        layout=index.cells(),
        closure_offsets=closure,
    )
    faces = tuple(
        phx.discretization.SparseBlockTopologyPlan(
            index,
            (4, 4),
            16,
            layout=index.faces(axis),
            closure_offsets=closure,
        )
        for axis in index.axis_names
    )
    transfer = phx.discretization.SparseFLIPParticleTransferPlan(
        index, cell, faces
    ).prepare(particles)
    projection = phx.solver.SparseMACFreeSurfaceProjectionPlan(
        transfer, tolerance=1.0e-7, maximum_iterations=100
    )
    compiled = phx.equations.compile_sparse_flip_problem(
        phx.equations.FLIPProblemIR("sparse-execution-benchmark", 1.0, jnp.zeros((2,))),
        transfer,
        projection,
        phx.discretization.FLIPMethodPlan(
            1.0, liquid_fraction_threshold=0.01, cfl_fraction=1.0
        ),
    )
    velocity = jnp.broadcast_to(jnp.asarray((0.02, 0.0)), position.shape)
    state = compiled.initialize_state(position, velocity)
    result, first, steady = _time(
        lambda value: compiled.step_detailed(value, 1.0e-4), state
    )
    return {
        "logical_cells": count * count,
        "cell_storage": cell.storage_capacity,
        "face_storage": [value.storage_capacity for value in faces],
        "compile_and_first_ms": first,
        "steady_ms": steady,
        "mass_defect": float(jnp.abs(result.diagnostics.mass_balance_defect)),
        "projection_residual": float(result.diagnostics.projection_residual),
        "successful": bool(result.successful),
    }


def run(output: Path):
    cases = {
        "key_groups": _group_case(),
        "relation_execution": _relation_case(),
        "gaussian_raster": _raster_case(),
        "sparse_lbm": _lbm_case(),
        "sparse_mpm": _mpm_case(),
        "sparse_flip": _flip_case(),
    }
    passed = all(case.get("successful", True) for case in cases.values()) and all(
        np.isfinite(value)
        for case in cases.values()
        for key, value in case.items()
        if key.endswith("_ms") or key.endswith("_defect")
    )
    payload = {
        "device": str(jax.devices()[0]),
        "cases": cases,
        "passed": bool(passed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    run(Path("benchmarks/sparse_execution.json"))
