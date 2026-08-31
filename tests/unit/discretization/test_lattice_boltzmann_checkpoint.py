#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _compiled_periodic():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(6, periodic=True),
            phx.discretization.UniformCellAxisSpec(6, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    compiled = phx.equations.compile_lattice_boltzmann_problem(
        phx.equations.LatticeBoltzmannProblem("checkpoint", 2),
        discretization,
        phx.discretization.LatticeBoltzmannMethodPlan(
            phx.discretization.BGKCollisionPlan()
        ),
        phx.discretization.LatticeBoltzmannBoundaryPlan(),
        time_step=0.01,
    )
    parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(0.01)
    state = compiled.initialize_state(
        1.0,
        jnp.asarray((0.01, -0.005)),
        parameters,
    )
    return compiled, parameters, state


def test_kinetic_checkpoint_roundtrip_continues_exactly(tmp_path):
    compiled, parameters, initial = _compiled_periodic()
    first = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        initial,
        jnp.asarray(0.01),
        parameters,
    ).accepted_state
    plan = phx.discretization.KineticCheckpointPlan(
        compiled.dynamics.prepared_id,
        compiled.dynamics.program_manifest,
        topology_id=compiled.boundary.boundary_id,
    )
    path = tmp_path / "kinetic.phxcheckpoint"
    written = phx.discretization.write_kinetic_checkpoint(
        path,
        plan,
        jnp.asarray(0.01),
        jnp.asarray(1, dtype=jnp.int32),
        first,
        args=parameters,
    )
    restored = phx.discretization.read_kinetic_checkpoint(
        path,
        plan,
        first,
        args_template=parameters,
    )
    uninterrupted = compiled.dynamics.step_detailed(
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.01),
        first,
        jnp.asarray(0.01),
        parameters,
    ).accepted_state
    continued = compiled.dynamics.step_detailed(
        restored.step_index,
        restored.time,
        restored.state,
        jnp.asarray(0.01),
        restored.args,
    ).accepted_state

    assert written.payload_id == restored.payload_id
    np.testing.assert_array_equal(restored.state, first)
    np.testing.assert_array_equal(continued, uninterrupted)


def test_kinetic_checkpoint_rejects_runtime_and_template_mismatch(tmp_path):
    compiled, parameters, state = _compiled_periodic()
    plan = phx.discretization.KineticCheckpointPlan(
        compiled.dynamics.prepared_id,
        compiled.dynamics.program_manifest,
    )
    path = tmp_path / "kinetic.phxcheckpoint"
    phx.discretization.write_kinetic_checkpoint(
        path,
        plan,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        state,
        args=parameters,
    )
    incompatible = phx.discretization.KineticCheckpointPlan(
        "different-runtime",
        compiled.dynamics.program_manifest,
    )
    with pytest.raises(ValueError, match="plan_id"):
        phx.discretization.read_kinetic_checkpoint(
            path,
            incompatible,
            state,
            args_template=parameters,
        )
    with pytest.raises(ValueError, match="shape or dtype"):
        phx.discretization.read_kinetic_checkpoint(
            path,
            plan,
            state.astype(jnp.float32),
            args_template=parameters,
        )
    with pytest.raises(ValueError, match="finite inexact"):
        phx.discretization.write_kinetic_checkpoint(
            tmp_path / "invalid-time.phxcheckpoint",
            plan,
            jnp.asarray(jnp.nan),
            jnp.asarray(0, dtype=jnp.int32),
            state,
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        phx.discretization.write_kinetic_checkpoint(
            tmp_path / "invalid-step.phxcheckpoint",
            plan,
            jnp.asarray(0.0),
            jnp.asarray(-1, dtype=jnp.int32),
            state,
        )


def test_kinetic_checkpoint_preserves_raw_aa_parity(tmp_path):
    lattice = phx.discretization.D2Q9()
    precision = phx.discretization.LatticeBoltzmannPrecisionPolicy()
    manifest = phx.discretization.athermal_lattice_boltzmann_manifest(
        lattice.lattice_id,
        precision.policy_id,
        lattice.population_count,
        lattice.dimension,
    )
    aa = phx.discretization.AALatticeBoltzmannPlan(lattice)
    canonical = jnp.arange(4 * 5 * 9, dtype=jnp.float64).reshape((4, 5, 9))
    state = aa.encode(canonical, parity=1)
    plan = phx.discretization.KineticCheckpointPlan(
        aa.addressing_id,
        manifest,
        execution_id=aa.addressing_id,
    )
    path = tmp_path / "aa.phxcheckpoint"
    phx.discretization.write_kinetic_checkpoint(
        path,
        plan,
        jnp.asarray(2.0),
        jnp.asarray(2, dtype=jnp.int32),
        state,
    )
    restored = phx.discretization.read_kinetic_checkpoint(path, plan, state)

    np.testing.assert_array_equal(restored.state.storage, state.storage)
    np.testing.assert_array_equal(restored.state.parity, state.parity)
    np.testing.assert_array_equal(aa.canonical(restored.state), canonical)
