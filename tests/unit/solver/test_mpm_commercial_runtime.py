#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _case():
    claim = phx.discretization.MPMClaimTuple(
        equation_family="solid-mechanics",
        dimension=2,
        kinematics="plane-strain",
        grid_assignment="quadratic-bspline",
        source_domain="point",
        transfer="apic",
        schedule="usl-minus",
        material="neo-hookean",
        field_contact="single-field-none",
        fracture="none",
        integrator="explicit-fixed",
        storage_backend="dense-cpu-f64-deterministic",
        precision_accumulation="f64-deterministic",
        capacity_envelope="particles-3-grid-8x8",
        derivative_mode="branchwise",
    )
    intended = phx.discretization.MPMIntendedUse(
        "runtime checkpoint qualification",
        phenomena=("finite-strain elasticity",),
        target_observables=("particle position",),
        risk_class="commercial-low-consequence",
        geometry_loading_scope="periodic unit square",
        material_parameter_scope="positive shear and bulk modulus",
        accuracy_uq_goal="restart parity",
    )
    decision = phx.discretization.MPMSupportDecision(
        claim,
        phx.discretization.MPMClaimOutcome.SUPPORTED,
        reason="runtime qualification tuple",
        required_profile="commercial-runtime",
    )
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.27, 0.31], [0.43, 0.38], [0.36, 0.52]])
    volume = jnp.full((3,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    problem = phx.equations.MaterialPointProblemIR(
        "commercial-runtime",
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        intended_use=intended,
        claim=claim,
    )
    compiled = phx.equations.compile_material_point_problem(
        problem,
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
        support_decision=decision,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    state = compiled.initialize_state(
        position,
        jnp.broadcast_to(jnp.asarray((0.02, -0.01)), position.shape),
        volume,
        arguments,
    )
    return compiled, arguments, state


def test_checkpoint_generation_roundtrip_and_current_pointer(tmp_path):
    compiled, arguments, state = _case()
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)
    plan = phx.solver.MPMCheckpointPlan(compiled, state)
    manifest = plan.write_generation(
        tmp_path / "checkpoint",
        detail.accepted_state,
        generation=1,
    )
    restored, record = plan.read_current(tmp_path / "checkpoint")

    assert manifest.generation == 1
    assert record["payload_id"] == manifest.payload_id
    for expected, actual in zip(
        jax.tree.leaves(detail.accepted_state),
        jax.tree.leaves(restored),
        strict=True,
    ):
        np.testing.assert_array_equal(expected, actual)


def test_checkpoint_corruption_is_rejected(tmp_path):
    compiled, _, state = _case()
    plan = phx.solver.MPMCheckpointPlan(compiled, state)
    path = tmp_path / "state.mpmckpt"
    plan.write(path, state)
    corrupt = tmp_path / "corrupt.mpmckpt"
    with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(corrupt, "w") as target:
        for name in source.namelist():
            payload = source.read(name)
            if name.startswith("arrays/"):
                payload = payload[:-1] + bytes((payload[-1] ^ 1,))
            target.writestr(name, payload)
    with pytest.raises(ValueError, match="checksum"):
        plan.read(corrupt)


def test_hdf5_xdmf_vtk_output_and_backpressure(tmp_path):
    compiled, arguments, state = _case()
    output = phx.solver.MPMOutputPlan(compiled, tmp_path / "trajectory.h5")
    output.initialize()
    first = compiled.dynamics.step_detailed(state, 0.001, arguments).accepted_state
    output.append(first)
    manifest = output.manifest()
    vtk = output.write_vtk_snapshot(tmp_path / "particles.vtu", first)

    assert manifest.accepted_steps == 1
    assert (tmp_path / "trajectory.h5").exists()
    assert (tmp_path / "trajectory.xdmf").exists()
    assert vtk.exists()

    buffer = phx.solver.MPMBoundedOutputBuffer(1)
    buffer.push(first)
    with pytest.raises(BufferError):
        buffer.push(first)
    assert buffer.pop() is first


def test_run_supervisor_separates_numerical_rejection_and_operational_recovery(
    tmp_path,
):
    compiled, arguments, state = _case()
    checkpoint = phx.solver.MPMCheckpointPlan(compiled, state)
    output = phx.solver.MPMOutputPlan(compiled, tmp_path / "run.h5")
    supervisor = phx.solver.MPMRunSupervisor(
        compiled.dynamics,
        state,
        arguments,
        checkpoint_plan=checkpoint,
        checkpoint_directory=tmp_path / "checkpoints",
        output_plan=output,
    )
    accepted = supervisor.advance(0.001)
    assert bool(accepted.numerical_result.successful)
    assert accepted.output_complete
    assert supervisor.generation == 1

    rejected = supervisor.advance(10.0)
    assert not bool(rejected.numerical_result.successful)
    assert rejected.failure == phx.discretization.MPMCommercialFailure.NONE
    recovered = supervisor.recover()
    assert int(recovered.accepted_step) == 1
    snapshot = supervisor.snapshot()
    assert snapshot["metrics"]["rejected_steps"] == 1
