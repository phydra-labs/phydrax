#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.sparse import (
    KeyGroupAccumulation,
    KeyGroupLookup,
    KeyGroupPlan,
    KeyGroupReductionEvidence,
    KeyGroupState,
    reduce_key_groups,
)


@eqx.filter_jit
def _seeded_reduce(
    groups: KeyGroupState,
    values: jax.Array,
    initial: KeyGroupAccumulation,
    accumulation: Literal["deterministic", "compensated"],
) -> tuple[KeyGroupAccumulation, KeyGroupReductionEvidence]:
    return reduce_key_groups(groups, values, accumulation=accumulation, initial=initial)


@eqx.filter_jit
def _wordkey_build(
    plan: KeyGroupPlan, keys: jax.Array, stable_ids: jax.Array
) -> KeyGroupState:
    return plan.build(
        keys, jnp.ones((keys.shape[0],), dtype=jnp.bool_), stable_ids=stable_ids
    )


@eqx.filter_jit
def _wordkey_lookup(groups: KeyGroupState, keys: jax.Array) -> KeyGroupLookup:
    return groups.lookup(keys)


def _wordkey_skew_case() -> dict[str, Any]:
    count = 512
    tails = jnp.where(
        jnp.arange(count, dtype=jnp.uint32) < 256,
        jnp.uint32(0),
        jnp.arange(count, dtype=jnp.uint32) % jnp.uint32(64),
    )
    keys = jnp.full((count, 4), 2**32 - 1, dtype=jnp.uint32).at[:, -1].set(tails)
    plan = KeyGroupPlan(count, 64, (2**32 - 1,) * 4)
    ids = jnp.arange(count, dtype=jnp.int64) - jnp.int64(2**40)
    compiled, timing = measure_lower_and_compile(
        lambda: _wordkey_build.lower(plan, keys, ids),  # ty: ignore[unresolved-attribute]
        lambda lowered: lowered.compile(),
    )
    groups, first_seconds = measure_synchronized(lambda: compiled(plan, keys, ids))
    groups, warm = measure_repeated(
        lambda: compiled(plan, keys, ids), warmup=1, repeats=3
    )
    lookup, lookup_warm = measure_repeated(
        lambda: _wordkey_lookup(groups, keys), warmup=1, repeats=3
    )
    slots = np.asarray(lookup.group_slots)
    reconstructed = np.asarray(groups.group_keys)[np.maximum(slots, 0)]
    identity = bool(np.all(np.asarray(lookup.supported))) and bool(
        np.array_equal(reconstructed, np.asarray(keys))
    )
    executable = compiled.compiled
    return {
        "word_count": 4,
        "items": count,
        "groups": int(groups.evidence.required_groups),
        "maximum_contention": int(groups.evidence.maximum_group_size),
        "uneven_target_occupancy": True,
        "lookup_identity": identity,
        "successful": bool(groups.evidence.successful) and identity,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "first_synchronized_seconds": first_seconds,
        "warm": warm.to_seconds_dict(),
        "lookup_warm": lookup_warm.to_seconds_dict(),
        "compiler": compiler_evidence(
            executable.cost_analysis(),
            executable.memory_analysis(),
            source="jax-compiled-skew-wordkeys",
            unavailable_reason="backend compiler fields unavailable",
        ),
        "logical_retained_bytes": logical_array_bytes(groups),
    }


def _wordkey_seeded_case(
    *, words: int, accumulation: Literal["deterministic", "compensated"]
) -> dict[str, Any]:
    """Same-prefix exact identities and signed complex cancellation across seeds."""
    count = 128
    keys = jnp.full((count, words), 2**32 - 1, dtype=jnp.uint32)
    keys = keys.at[:, -1].set(jnp.arange(count, dtype=jnp.uint32))
    groups = KeyGroupPlan(count, count, (2**32 - 1,) * words).build(
        keys,
        jnp.ones((count,), dtype=jnp.bool_),
        stable_ids=jnp.arange(count, dtype=jnp.int64) - jnp.int64(2**40),
    )
    initial = KeyGroupAccumulation(
        jnp.full((count,), 1e16 + 1e16j, dtype=jnp.complex128),
        jnp.full((count,), 1 + 1j, dtype=jnp.complex128),
    )
    negative = jnp.full((count,), -1e16 - 1e16j, dtype=jnp.complex128)
    compiled, timing = measure_lower_and_compile(
        lambda: _seeded_reduce.lower(groups, negative, initial, accumulation),  # ty: ignore[unresolved-attribute]
        lambda lowered: lowered.compile(),
    )
    _, first_seconds = measure_synchronized(
        lambda: compiled(groups, negative, initial, accumulation)
    )
    (result, evidence), warm = measure_repeated(
        lambda: compiled(groups, negative, initial, accumulation), warmup=1, repeats=3
    )
    final, final_evidence = _seeded_reduce(
        groups, jnp.full((count,), 3 + 3j, dtype=jnp.complex128), result, accumulation
    )
    lookup = groups.lookup(keys)
    expected = 4 + 4j
    defect = float(jnp.max(jnp.abs(final.value - expected)))
    executable = compiled.compiled
    return {
        "word_count": words,
        "items": count,
        "same_prefix_different_tail": True,
        "accumulation": accumulation,
        "signed_stable_ids": True,
        "seeded_cancellation_defect": defect,
        "lookup_identity": bool(
            jnp.all(
                lookup.supported
                & (lookup.group_slots == jnp.arange(count, dtype=jnp.int32))
            )
        ),
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "first_synchronized_seconds": first_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": compiler_evidence(
            executable.cost_analysis(),
            executable.memory_analysis(),
            source="jax-compiled-seeded-wordkeys",
            unavailable_reason="backend compiler fields unavailable",
        ),
        "logical_retained_bytes": logical_array_bytes((groups, initial, result)),
        "successful": bool(evidence.successful & final_evidence.successful),
        "expected": {"real": 4.0, "imag": 4.0},
        "finite_control_passed": defect == 0.0,
    }


def projector_cases() -> dict[str, Any]:
    """Run only the new controls, without raster/fluid benchmark side effects."""
    cases = {
        f"wordkeys-{words}-{accumulation}": _wordkey_seeded_case(
            words=words, accumulation=accumulation
        )
        for words in (1, 4, 8)
        for accumulation in ("deterministic", "compensated")
    }
    cases["wordkeys-uneven-occupancy"] = _wordkey_skew_case()
    return cases


def _ready(value: Any) -> Any:
    return jax.block_until_ready(value)


def _time(function: Any, *arguments: Any, repeats: Any = 3) -> Any:
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


def _array_bytes(value: Any) -> Any:
    return int(
        sum(
            leaf.size * leaf.dtype.itemsize
            for leaf in jax.tree_util.tree_leaves(value)
            if isinstance(leaf, jax.Array)
        )
    )


def _group_case() -> Any:
    count = 4096
    keys = (jnp.arange(count, dtype=jnp.int32) * 17) % 512
    plan = phx.sparse.KeyGroupPlan(
        count,
        512,
        511,
        maximum_group_size=8,
    )
    state, first, steady = _time(
        lambda value: plan.build(value, jnp.ones(value.shape, dtype="bool")), keys
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


def _relation_case() -> Any:
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


def _streamed_relation_case() -> Any:
    """Shared bounded streamed runner against its materialized nonlinear reference."""
    count = 4096
    target_count = 512
    rng = np.random.default_rng(7)
    targets = np.sort(rng.integers(0, target_count, size=count)).astype(np.int32)
    sources = rng.integers(0, target_count, size=count).astype(np.int32)
    relation = phx.sparse.EdgeRelation(
        sources, targets, source_size=target_count, target_size=target_count
    )
    positions = jnp.asarray(rng.normal(size=(target_count, 3)))
    weights = jnp.asarray(rng.normal(size=(4, 8)))
    payload = phx.sparse.StreamedPayloadSpec(
        jax.ShapeDtypeStruct((8,), jnp.float64), jax.ShapeDtypeStruct((), jnp.float64)
    )

    def message(
        w: jax.Array, source: jax.Array, receiver: jax.Array, edge: jax.Array | None
    ) -> jax.Array:
        displacement = receiver - source
        distance = jnp.sqrt(jnp.sum(displacement**2) + 1.0)
        return jnp.tanh(jnp.concatenate((displacement, distance[None])) @ w)

    def readout(w: jax.Array, receiver: jax.Array, aggregate: jax.Array) -> jax.Array:
        return jnp.sum(jnp.sin(aggregate)) + 0.1 * jnp.sum(receiver**2)

    def reference(x: jax.Array) -> jax.Array:
        messages = jax.vmap(lambda s, r: message(weights, s, r, None))(
            x[sources], x[targets]
        )
        aggregate = jnp.zeros((target_count, 8)).at[targets].add(messages)
        return jax.vmap(lambda r, a: readout(weights, r, a))(x, aggregate)

    expected, reference_first, reference_steady = _time(reference, positions)
    case: dict[str, Any] = {
        "routes": count,
        "targets": target_count,
        "reference_compile_and_first_ms": reference_first,
        "reference_steady_ms": reference_steady,
    }
    successful = True
    for accumulation in ("fast", "deterministic", "compensated"):
        prepared = phx.sparse.StreamedRelationPlan(
            receiver_tile=32, edge_tile=256, channel_capacity=8, accumulation=accumulation
        ).prepare(relation, owner_id="sparse-execution-benchmark")
        result, first, steady = _time(
            lambda x: prepared.evaluate(
                payload, message, readout, weights, x, x, jnp.zeros((count,))
            ),
            positions,
        )
        successful = successful and bool(result.evidence.successful)
        case[f"{accumulation}_compile_and_first_ms"] = first
        case[f"{accumulation}_steady_ms"] = steady
        case[f"{accumulation}_reference_defect"] = float(
            jnp.max(jnp.abs(result.receiver_outputs - expected))
        )
        case["tile_count"] = prepared.schedule.tile_count
        case["schedule_bytes"] = _array_bytes(prepared.schedule)
    case["successful"] = successful
    return case


def _raster_case() -> Any:
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
        "particles": coordinates.shape[0],
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


def _periodic_index(count: Any) -> Any:
    plan = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y"),
    )
    return plan.prepare_index_space(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))


def _lbm_case() -> Any:
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
        "fluid_cells": fluid.size,
        "storage_cells": prepared.storage_capacity,
        "state_bytes": _array_bytes(state),
        "compile_and_first_ms": first,
        "steady_ms": steady,
        "mass_defect": float(jnp.abs(result.diagnostics.mass_defect)),
        "successful": bool(result.successful),
    }


def _mpm_case() -> Any:
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
        # ty: ignore[unresolved-attribute]
        "active_blocks": int(state.storage_state.evidence.required_blocks),
        "state_bytes": _array_bytes(state),
        "compile_and_first_ms": first,
        "steady_ms": steady,
        "successful": bool(result.successful),
    }


def _flip_case() -> Any:
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


def run(output: Path, *, projector_only: bool = False) -> None:
    cases = (
        {}
        if projector_only
        else {
            "key_groups": _group_case(),
            "relation_execution": _relation_case(),
            "streamed_relation": _streamed_relation_case(),
            "gaussian_raster": _raster_case(),
            "sparse_lbm": _lbm_case(),
            "sparse_mpm": _mpm_case(),
            "sparse_flip": _flip_case(),
        }
    )
    cases.update(projector_cases())
    passed = all(case.get("successful", True) for case in cases.values()) and all(
        np.isfinite(value)
        for case in cases.values()
        for key, value in case.items()
        if key.endswith("_ms") or key.endswith("_defect")
    )
    passed = passed and all(
        case.get("finite_control_passed", True) and case.get("lookup_identity", True)
        for case in cases.values()
    )
    payload = {
        "device": str(jax.devices()[0]),
        "cases": cases,
        "passed": bool(passed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    from tools.projector_monte_carlo_qualification import _json

    output.write_text(json.dumps(_json(payload), indent=2, allow_nan=False) + "\n")
    print(json.dumps(_json(payload), indent=2, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/sparse_execution.json")
    )
    parser.add_argument("--projector-only", action="store_true")
    arguments = parser.parse_args()
    run(arguments.output, projector_only=arguments.projector_only)
