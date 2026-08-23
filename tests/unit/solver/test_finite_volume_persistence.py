#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import zipfile
from importlib.util import find_spec
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _prepared_runtime(cells=16):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "persistent-euler",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(
        problem, discretization, method
    )
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics, phx.discretization.FluxPositivityPlan()
    )
    primitive = jnp.broadcast_to(jnp.asarray([1.0, 0.1, 1.0]), (cells, 3))
    state = phx.solver.FiniteVolumeRuntimeState(
        system.primitive_to_conserved(primitive),
        0.2,
        0.001,
        accepted_step=7,
        controller_state=jnp.asarray([0.3, 0.4]),
        integrator_state=jnp.asarray([1.2]),
        forcing_state=jnp.asarray([2.5]),
        random_state=jnp.asarray([11, 17], dtype=jnp.uint32),
        output_cursor=4,
    )
    return runtime, discretization, state


def test_case_schema_is_versioned_content_addressed_and_strict():
    runtime, _, _ = _prepared_runtime()
    execution = phx.solver.FiniteVolumeExecutionSpec(1.0, 1000)
    case = phx.solver.FiniteVolumeCaseSpec(
        "portable-euler",
        runtime,
        execution,
        precision=phx.solver.FiniteVolumePrecisionPolicy("float64"),
    )
    payload = case.to_dict()

    phx.solver.FiniteVolumeCaseSpec.validate_dict(payload)
    assert payload["schema_version"] == 1
    assert payload["case_id"] == case.case_id
    restored = phx.solver.FiniteVolumeCaseSpec.from_dict(
        payload, runtime, execution
    )
    assert restored.case_id == case.case_id
    with pytest.raises(ValueError, match="unknown"):
        phx.solver.FiniteVolumeCaseSpec.validate_dict(
            {**payload, "misspelled_flux": "HLLC"}
        )


def test_checkpoint_roundtrip_preserves_exact_runtime_state(tmp_path):
    runtime, _, state = _prepared_runtime()
    case = phx.solver.FiniteVolumeCaseSpec(
        "checkpoint-euler",
        runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    plan = phx.solver.FiniteVolumeCheckpointPlan(case)
    path = tmp_path / "restart.fvckpt"
    written = phx.solver.write_finite_volume_checkpoint(path, plan, state)
    loaded = phx.solver.read_finite_volume_checkpoint(path, plan)

    assert written.payload_id == loaded.payload_id
    np.testing.assert_array_equal(
        loaded.runtime_state.conservative_state, state.conservative_state
    )
    np.testing.assert_array_equal(loaded.runtime_state.time, state.time)
    np.testing.assert_array_equal(
        loaded.runtime_state.accepted_step, state.accepted_step
    )
    np.testing.assert_array_equal(
        loaded.runtime_state.controller_state, state.controller_state
    )
    np.testing.assert_array_equal(
        loaded.runtime_state.integrator_state, state.integrator_state
    )
    np.testing.assert_array_equal(
        loaded.runtime_state.forcing_state, state.forcing_state
    )
    np.testing.assert_array_equal(
        loaded.runtime_state.random_state, state.random_state
    )
    np.testing.assert_array_equal(
        loaded.runtime_state.output_cursor, state.output_cursor
    )
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_checkpoint_rejects_manifest_corruption(tmp_path):
    runtime, _, state = _prepared_runtime()
    case = phx.solver.FiniteVolumeCaseSpec(
        "corrupt-euler",
        runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    plan = phx.solver.FiniteVolumeCheckpointPlan(case)
    path = tmp_path / "restart.fvckpt"
    phx.solver.write_finite_volume_checkpoint(path, plan, state)
    with zipfile.ZipFile(path, "r") as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["case"]["method_id"] = "changed"
    entries["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)

    with pytest.raises(ValueError, match="case identity|corrupt"):
        phx.solver.read_finite_volume_checkpoint(path, plan)


def test_output_plan_is_explicitly_optional_when_h5py_is_unavailable(tmp_path):
    _, discretization, state = _prepared_runtime()
    plan = phx.solver.FiniteVolumeOutputPlan(
        tmp_path / "solution.h5", discretization
    )
    if find_spec("h5py") is None:
        with pytest.raises(ImportError, match="h5py"):
            plan.write_snapshot(discretization, state)
    else:
        index = plan.write_snapshot(discretization, state)
        assert index == 0
        assert Path(plan.hdf5_path).exists()
        assert Path(plan.xdmf_path).exists()


def test_allowlisted_case_loader_builds_portable_runtime():
    payload = {
        "schema_version": 1,
        "name": "loaded-euler",
        "grid": {
            "cells": 16,
            "lower": 0.0,
            "upper": 1.0,
            "periodic": True,
        },
        "equation": {
            "type": "ideal_gas_euler",
            "gamma": 1.4,
            "gas_constant": 1.0,
        },
        "method": {
            "reconstruction": "muscl",
            "flux": "hllc",
        },
        "boundary": {"type": "periodic"},
        "execution": {"end_time": 0.1, "maximum_steps": 100},
        "precision": {"dtype": "float64"},
    }
    prepared = phx.solver.load_finite_volume_case(payload)

    assert prepared.discretization.cell_shape == (16,)
    assert prepared.runtime.dynamics.system.component_names == (
        "density",
        "momentum_0",
        "total_energy",
    )
    with pytest.raises(ValueError, match="unknown"):
        phx.solver.load_finite_volume_case(
            {**payload, "misspelled_method": "hllc"}
        )


def test_interrupted_checkpoint_trajectory_matches_uninterrupted(tmp_path):
    runtime, _, initial = _prepared_runtime()

    def advance_many(state, count):
        current = state
        for _ in range(count):
            current = runtime.advance(current).runtime_state
        return current

    uninterrupted = advance_many(initial, 3)
    interrupted = advance_many(initial, 1)
    case = phx.solver.FiniteVolumeCaseSpec(
        "segmented-euler",
        runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    plan = phx.solver.FiniteVolumeCheckpointPlan(case)
    path = tmp_path / "segmented.fvckpt"
    phx.solver.write_finite_volume_checkpoint(path, plan, interrupted)
    restored = phx.solver.read_finite_volume_checkpoint(
        path, plan
    ).runtime_state
    resumed = advance_many(restored, 2)

    np.testing.assert_array_equal(
        resumed.conservative_state, uninterrupted.conservative_state
    )
    np.testing.assert_array_equal(resumed.time, uninterrupted.time)
    np.testing.assert_array_equal(
        resumed.accepted_step, uninterrupted.accepted_step
    )
    np.testing.assert_array_equal(
        resumed.controller_state, uninterrupted.controller_state
    )
    np.testing.assert_array_equal(
        resumed.output_cursor, uninterrupted.output_cursor
    )
