#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from phydrax._execution_plan import (
    ExecutionCandidate,
    ExecutionPlan,
    ExecutionRequirements,
    ProviderBinding,
    resolve_execution_plan,
)
from phydrax._execution_resources import (
    DeviceResource,
    ExecutionGroupSpec,
    ExecutionPolicy,
    ExecutionResourceEvidence,
    ResourceInventory,
    ResourceRequest,
)
from phydrax._external_resource import ResourceLimits
from phydrax._physical import RelativityScaleContract
from phydrax.applications.numerical_relativity._boundaries import PeriodicBoundary
from phydrax.applications.numerical_relativity._derivatives import FourthOrderDerivatives
from phydrax.applications.numerical_relativity._enforcement import (
    Z4cAlgebraicEnforcement,
)
from phydrax.applications.numerical_relativity._gauge import GeodesicGauge
from phydrax.applications.numerical_relativity._grid import FixedGridGeometry
from phydrax.applications.numerical_relativity._production import (
    compile_numerical_relativity_production,
    FixedGridZ4cProductionMethod,
    NumericalRelativityDomainBinding,
    NumericalRelativityProductionLimits,
    NumericalRelativitySupportBinding,
)
from phydrax.applications.numerical_relativity._state import flat_z4c_state
from phydrax.applications.numerical_relativity._temporal import FixedGridZ4cRuntime
from phydrax.applications.numerical_relativity._z4c import Z4cSystem
from phydrax.interchange._black_hole import (
    BlackHoleArtifactRights,
    BlackHoleArtifactUsePolicy,
    map_field_artifact,
    map_image_artifact,
)
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.qualification._registry import SupportTuple
from phydrax.solver._fixed_step import (
    CallableFixedStepMethod,
    FixedStepResult,
    RobustRetryPolicy,
)
from phydrax.solver._production_runtime import (
    CheckpointGenerationPolicy,
    DurableCheckpointStore,
    ProductionCaseManifest,
    ProductionFailureRecord,
    ProductionRunPlan,
    ProductionRunResult,
    ProductionRunState,
)
from phydrax.solver._runtime_lifecycle import ExactTimeSchedule
from phydrax.units import KILOGRAM


def _identity_step(step_index, time, state, step_size, args):
    del step_index, time, step_size, args
    dtype = jax.tree.leaves(state)[0].dtype
    return FixedStepResult(
        state,
        state,
        jnp.asarray(True),
        jnp.zeros((), dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.zeros((), dtype=dtype),
    )


def _neutral_artifact(tmp_path: Path, name: str, *, kind: str = "field"):
    payload = f"bounded-{name}".encode()
    path = tmp_path / name
    path.write_bytes(payload)
    rights = BlackHoleArtifactRights(
        kind,
        f"source:{name}",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        "CC0-1.0",
        f"generated:{name}",
        "attribution:generated",
        producer="producer:phydrax-test",
        producer_version="release:test",
        model_id=f"model:{kind}",
        coverage="coverage:bounded-test-output",
        commercial_use=True,
        training_use=False,
        redistribution=True,
        derivative_use=True,
        model_execution=False,
        export=True,
    )
    policy = BlackHoleArtifactUsePolicy(
        "numerical-relativity production",
        ("CC0-1.0",),
        commercial_use=True,
    )
    common = {
        "trusted_root": tmp_path,
        "limits": ResourceLimits(1024, 4, 32, 16, 2),
        "rights": rights,
        "use_policy": policy,
    }
    if kind == "field":
        return map_field_artifact(
            name,
            source_format="openpmd-hdf5",
            chart_id="chart:cartesian",
            coordinate_frame_id="frame:simulation",
            quantity_id="quantity:z4c-state",
            topology_id="topology:grid-8",
            unit_id="units:geometric",
            **common,
        )
    return map_image_artifact(
        name,
        source_format="fits",
        observable_id="observable:constraint-norm",
        screen_frame_id="frame:grid-slice",
        unit_id="units:dimensionless",
        **common,
    )


def _execution_policy_and_plan(
    resources, *, precision_id, solver_policy_id, dtype_name="float32"
):
    policy = ExecutionPolicy(resources=resources)
    requirements = ExecutionRequirements(
        "owner:numerical-relativity-test",
        dtypes=(dtype_name,),
    )
    group = ExecutionGroupSpec(
        "group:local-test",
        (0,),
        ((0, 0),),
        mesh_axes=(("device", 1),),
    )
    evidence = ExecutionResourceEvidence(
        per_device_peak_bytes=1024,
        per_device_reserve_bytes=256,
        per_host_peak_bytes=1024,
        per_host_reserve_bytes=256,
        checkpoint_staging_bytes=resources.maximum_checkpoint_staging_bytes,
        output_backlog_bytes=resources.maximum_output_backlog_bytes,
        dtypes=(dtype_name,),
        backends=(jax.default_backend(),),
    )
    candidate = ExecutionCandidate(
        "candidate:local-test",
        requirements.requirements_id,
        group,
        providers=(
            ProviderBinding(
                "backend", jax.default_backend(), "capability:numerical-relativity"
            ),
        ),
        estimated_memory_bytes=2048,
        resource_evidence=evidence,
    )
    inventory = ResourceInventory(
        1,
        0,
        (
            DeviceResource(
                0,
                0,
                0,
                jax.default_backend(),
                "test-device",
                memory_bytes=resources.memory_bytes,
            ),
        ),
    )
    plan = resolve_execution_plan(
        policy,
        inventory,
        (candidate,),
        precision_policy_id=precision_id,
        solver_policy_id=solver_policy_id,
    )
    return policy, plan


def _compiled(tmp_path: Path, *, artifacts=(), end_time=0.1, output_artifacts=1):
    grid = FixedGridGeometry(
        (5, 5, 5),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        periodic=True,
    )
    system = Z4cSystem(
        RelativityScaleContract.geometric(KILOGRAM),
        RelativityConvention(),
        chart_id="chart:cartesian",
    )
    gauge = GeodesicGauge()
    runtime = FixedGridZ4cRuntime(
        system,
        grid,
        FourthOrderDerivatives(grid.shape, grid.spacing),
        gauge,
        PeriodicBoundary(),
        Z4cAlgebraicEnforcement(),
        time_step=0.1,
        integrator="ssprk33",
    )
    method = FixedGridZ4cProductionMethod(runtime)
    initial = method.initialize(
        runtime.initialize(
            flat_z4c_state(grid.shape, grid_id=grid.grid_id, dtype=jnp.float32)
        )
    )
    domain = NumericalRelativityDomainBinding(
        formulation_id=system.system_id,
        chart_id=system.chart_id,
        gauge_id=gauge.gauge_id,
        eos_id="eos:vacuum",
        topology_id=grid.grid_id,
        precision_id="precision:float32",
        input_artifacts=artifacts,
    )
    support = SupportTuple(
        "numerical-relativity.production",
        {
            "formulation_id": domain.formulation_id,
            "chart_id": domain.chart_id,
            "gauge_id": domain.gauge_id,
            "eos_id": domain.eos_id,
            "topology_id": domain.topology_id,
            "precision_id": domain.precision_id,
        },
    )
    binding = NumericalRelativitySupportBinding(
        "scientific", "profile:numerical-relativity-z4c", support
    )
    resources = ResourceRequest(
        1,
        1_000_000,
        maximum_checkpoint_staging_bytes=500_000,
        maximum_output_backlog_bytes=500_000,
        required_dtypes=("float32",),
        required_backends=(jax.default_backend(),),
    )
    checkpoint_policy = CheckpointGenerationPolicy(2)
    run_plan = ProductionRunPlan(
        method,
        RobustRetryPolicy(maximum_retries=0),
        step_size=0.1,
        end_time=end_time,
        maximum_steps=4,
        checkpoint_interval=1,
        segment_steps=1,
    )
    execution_policy, execution = _execution_policy_and_plan(
        resources,
        precision_id=domain.precision_id,
        solver_policy_id=run_plan.plan_id,
    )
    limits = NumericalRelativityProductionLimits(
        execution_policy,
        maximum_input_artifacts=2,
        maximum_input_bytes=1024,
        maximum_output_manifests=2,
        maximum_output_artifacts=output_artifacts,
        maximum_output_bytes=1024,
        maximum_cancellation_detail_bytes=32,
    )
    case = ProductionCaseManifest(
        problem_id="problem:binary-black-hole",
        method_id=method.method_id,
        precision_id=domain.precision_id,
        topology_id=domain.topology_id,
        geometry_layout_id=domain.chart_id,
        dtype="float32",
    )
    resolved = ResolvedRunSpec(
        (binding.dependency,),
        (),
        release_index_id="release:numerical-relativity",
        profile_ids=(binding.profile_id,),
        trust_policy_id="trust:local-test",
        valid_at=10,
        valid_from=0,
        valid_until=20,
        prepared_configuration_id=domain.binding_id,
        precision_policy_id=domain.precision_id,
        resource_policy_id=limits.resource_policy_id,
        checkpoint_policy_id=checkpoint_policy.policy_id,
        output_policy_id=limits.output_policy_id,
        repository_id="repository:local-test",
        scheduler_id="scheduler:inline",
        auth_policy_id="auth:local-test",
    )
    production = compile_numerical_relativity_production(
        domain,
        (binding,),
        execution,
        resolved,
        case,
        run_plan,
        checkpoint_policy,
        limits,
    )
    store = DurableCheckpointStore(
        tmp_path / f"checkpoints-{end_time}", case, checkpoint_policy
    )
    return production, production.prepare(store), initial


def _replace_status(state, status, checkpoint):
    return ProductionRunState(
        state.step_index,
        state.time,
        state.accepted_state,
        state.controller_state,
        state.rng_state,
        state.schedule_cursor,
        state.moment_states,
        state.trigger_states,
        state.output_cursor,
        status,
        checkpoint,
    )


def _compiled_z4c_output(tmp_path: Path, artifact):
    grid = FixedGridGeometry(
        (5, 5, 5),
        (0.0, 0.0, 0.0),
        (0.1, 0.1, 0.1),
        periodic=True,
    )
    system = Z4cSystem(
        RelativityScaleContract.geometric(KILOGRAM),
        RelativityConvention(),
        chart_id="chart:cartesian",
    )
    gauge = GeodesicGauge()
    runtime = FixedGridZ4cRuntime(
        system,
        grid,
        FourthOrderDerivatives(grid.shape, grid.spacing),
        gauge,
        PeriodicBoundary(),
        Z4cAlgebraicEnforcement(),
        time_step=0.01,
        integrator="ssprk33",
    )
    method = FixedGridZ4cProductionMethod(runtime)
    initial = method.initialize(
        runtime.initialize(
            flat_z4c_state(grid.shape, grid_id=grid.grid_id, dtype=jnp.float64)
        )
    )
    domain = NumericalRelativityDomainBinding(
        formulation_id=system.system_id,
        chart_id=system.chart_id,
        gauge_id=gauge.gauge_id,
        eos_id="eos:vacuum",
        topology_id=grid.grid_id,
        precision_id="precision:float64",
    )
    support = SupportTuple(
        "numerical-relativity.production",
        {
            "formulation_id": domain.formulation_id,
            "chart_id": domain.chart_id,
            "gauge_id": domain.gauge_id,
            "eos_id": domain.eos_id,
            "topology_id": domain.topology_id,
            "precision_id": domain.precision_id,
        },
    )
    binding = NumericalRelativitySupportBinding(
        "scientific", "profile:z4c-output", support
    )
    resources = ResourceRequest(
        1,
        1_000_000,
        maximum_checkpoint_staging_bytes=500_000,
        maximum_output_backlog_bytes=500_000,
        required_dtypes=("float64",),
        required_backends=(jax.default_backend(),),
    )
    checkpoint_policy = CheckpointGenerationPolicy(1)
    run_plan = ProductionRunPlan(
        method,
        RobustRetryPolicy(maximum_retries=0),
        step_size=0.01,
        end_time=0.01,
        maximum_steps=1,
        checkpoint_interval=1,
        segment_steps=1,
        output_schedule=ExactTimeSchedule((0.01,)),
    )
    case = ProductionCaseManifest(
        problem_id="problem:z4c-output",
        method_id=method.method_id,
        precision_id=domain.precision_id,
        topology_id=domain.topology_id,
        geometry_layout_id=domain.chart_id,
        dtype="float64",
    )
    execution_policy, execution = _execution_policy_and_plan(
        resources,
        precision_id=domain.precision_id,
        solver_policy_id=run_plan.plan_id,
        dtype_name="float64",
    )
    limits = NumericalRelativityProductionLimits(
        execution_policy,
        maximum_input_artifacts=1,
        maximum_input_bytes=1024,
        maximum_output_manifests=1,
        maximum_output_artifacts=1,
        maximum_output_bytes=1024,
        maximum_cancellation_detail_bytes=64,
    )
    resolved = ResolvedRunSpec(
        (binding.dependency,),
        (),
        release_index_id="release:z4c-output",
        profile_ids=(binding.profile_id,),
        trust_policy_id="trust:test",
        valid_at=1,
        valid_from=0,
        valid_until=2,
        prepared_configuration_id=domain.binding_id,
        precision_policy_id=domain.precision_id,
        resource_policy_id=resources.resource_id,
        checkpoint_policy_id=checkpoint_policy.policy_id,
        output_policy_id=limits.output_policy_id,
        repository_id="repository:test",
        scheduler_id="scheduler:inline",
        auth_policy_id="auth:test",
    )
    production = compile_numerical_relativity_production(
        domain,
        (binding,),
        execution,
        resolved,
        case,
        run_plan,
        checkpoint_policy,
        limits,
    )
    store = DurableCheckpointStore(
        tmp_path / "z4c-output-checkpoints", case, checkpoint_policy
    )

    def writer(event_id, state):
        assert event_id
        assert state.runtime_state.step_index == 1
        return (artifact,)

    prepared, committer = production.prepare_with_output_committer(
        store, writer, "writer:test"
    )
    return production, prepared, committer, initial


def test_fixed_grid_adapter_preserves_runtime_acceptance_and_time_grid():
    grid = FixedGridGeometry(
        (5, 5, 5),
        (0.0, 0.0, 0.0),
        (0.1, 0.1, 0.1),
        periodic=True,
    )
    system = Z4cSystem(
        RelativityScaleContract.geometric(KILOGRAM),
        RelativityConvention(),
        chart_id="chart:cartesian",
    )
    runtime = FixedGridZ4cRuntime(
        system,
        grid,
        FourthOrderDerivatives(grid.shape, grid.spacing),
        GeodesicGauge(),
        PeriodicBoundary(),
        Z4cAlgebraicEnforcement(),
        time_step=0.01,
        integrator="ssprk33",
    )
    method = FixedGridZ4cProductionMethod(runtime)
    state = method.initialize(
        runtime.initialize(
            flat_z4c_state(grid.shape, grid_id=grid.grid_id, dtype=jnp.float32)
        )
    )

    accepted = method.step(0, 0.0, state, 0.01, None)
    assert bool(accepted.successful)
    assert int(accepted.accepted_state.runtime_state.step_index) == 1
    assert accepted.work == 3

    mismatched = method.step(1, 0.0, state, 0.01, None)
    assert not bool(mismatched.successful)
    assert int(mismatched.accepted_state.runtime_state.step_index) == 0


def test_compilation_binds_support_execution_resolved_run_and_artifacts(tmp_path: Path):
    initial_data = _neutral_artifact(tmp_path, "initial-data.bin")
    production, prepared, _initial = _compiled(tmp_path, artifacts=(initial_data,))

    assert production.domain.input_artifacts[0].artifact_id == initial_data.artifact_id
    retained = production.domain.input_artifacts[0]
    assert retained.rights.producer == "producer:phydrax-test"
    assert retained.producer_version == "release:test"
    assert retained.attribution_id == "attribution:generated"
    assert retained.redistribution
    assert retained.use_policy_id == initial_data.use_policy.policy_id
    assert production.execution_plan.solver_policy_id == production.run_plan.plan_id
    assert (
        production.resolved_run_spec.prepared_configuration_id
        == production.domain.binding_id
    )
    assert prepared.resolved_run_spec.spec_id == production.resolved_run_spec.spec_id

    substituted = _neutral_artifact(tmp_path, "substituted-data.bin")
    incompatible_domain = NumericalRelativityDomainBinding(
        formulation_id=production.domain.formulation_id,
        chart_id=production.domain.chart_id,
        gauge_id=production.domain.gauge_id,
        eos_id=production.domain.eos_id,
        topology_id=production.domain.topology_id,
        precision_id=production.domain.precision_id,
        input_artifacts=(substituted,),
    )
    with pytest.raises(ValueError, match="domain configuration"):
        compile_numerical_relativity_production(
            incompatible_domain,
            production.support_bindings,
            production.execution_plan,
            production.resolved_run_spec,
            production.case_manifest,
            production.run_plan,
            production.checkpoint_policy,
            production.limits,
        )


def test_production_rejects_callable_methods_and_unresolved_execution_plans(
    tmp_path: Path,
):
    production, _prepared, _initial = _compiled(tmp_path)
    direct = ExecutionPlan(
        "execution:direct-bypass",
        production.execution_plan.backend,
        production.domain.precision_id,
        production.run_plan.plan_id,
    )
    with pytest.raises(ValueError, match="resolver-produced"):
        compile_numerical_relativity_production(
            production.domain,
            production.support_bindings,
            direct,
            production.resolved_run_spec,
            production.case_manifest,
            production.run_plan,
            production.checkpoint_policy,
            production.limits,
        )

    callable_method = CallableFixedStepMethod(_identity_step, "method:callable-bypass")
    callable_plan = ProductionRunPlan(
        callable_method,
        RobustRetryPolicy(maximum_retries=0),
        step_size=0.1,
        end_time=0.1,
        maximum_steps=1,
        checkpoint_interval=1,
        segment_steps=1,
    )
    policy, execution = _execution_policy_and_plan(
        production.limits.resource_request,
        precision_id=production.domain.precision_id,
        solver_policy_id=callable_plan.plan_id,
    )
    limits = NumericalRelativityProductionLimits(
        policy,
        maximum_input_artifacts=2,
        maximum_input_bytes=1024,
        maximum_output_manifests=2,
        maximum_output_artifacts=1,
        maximum_output_bytes=1024,
        maximum_cancellation_detail_bytes=32,
    )
    resolved = ResolvedRunSpec(
        production.resolved_run_spec.scientific_dependencies,
        production.resolved_run_spec.deployment_dependencies,
        release_index_id=production.resolved_run_spec.release_index_id,
        profile_ids=production.resolved_run_spec.profile_ids,
        trust_policy_id=production.resolved_run_spec.trust_policy_id,
        valid_at=production.resolved_run_spec.valid_at,
        valid_from=production.resolved_run_spec.valid_from,
        valid_until=production.resolved_run_spec.valid_until,
        prepared_configuration_id=production.domain.binding_id,
        precision_policy_id=production.domain.precision_id,
        resource_policy_id=limits.resource_policy_id,
        checkpoint_policy_id=production.checkpoint_policy.policy_id,
        output_policy_id=limits.output_policy_id,
        repository_id=production.resolved_run_spec.repository_id,
        scheduler_id=production.resolved_run_spec.scheduler_id,
        auth_policy_id=production.resolved_run_spec.auth_policy_id,
    )
    case = ProductionCaseManifest(
        problem_id=production.case_manifest.problem_id,
        method_id=callable_method.method_id,
        precision_id=production.domain.precision_id,
        topology_id=production.domain.topology_id,
        geometry_layout_id=production.domain.chart_id,
        dtype="float32",
    )
    with pytest.raises(TypeError, match="typed supported method"):
        compile_numerical_relativity_production(
            production.domain,
            production.support_bindings,
            execution,
            resolved,
            case,
            callable_plan,
            production.checkpoint_policy,
            limits,
        )


def test_restart_admission_rejects_another_exact_runtime_plan(tmp_path: Path):
    production, prepared, initial = _compiled(tmp_path)
    state = prepared.initial_state(initial)
    checkpointed = prepared.checkpoint(state)
    restart = production.restart_manifest(prepared, checkpointed)
    restored = production.resume(prepared, state, restart)
    assert restored.last_checkpoint_id == restart.checkpoint_id

    other, other_prepared, _other_initial = _compiled(tmp_path, end_time=0.2)
    with pytest.raises(ValueError, match="Restart identity"):
        other.admit_restart(other_prepared, restart)


def test_output_manifest_derives_from_acknowledged_writer_receipt_and_run_lineage(
    tmp_path: Path,
):
    output = _neutral_artifact(tmp_path, "constraint-image.bin", kind="image")
    production, prepared, committer, initial = _compiled_z4c_output(tmp_path, output)
    result = prepared.run(prepared.initial_state(initial))
    (receipt,) = committer.committed_receipts()
    manifest = production.output_manifest(prepared, committer, receipt, result)

    assert manifest.receipt_id == receipt.receipt_id
    assert manifest.event_id == receipt.event_id
    assert manifest.step_index == int(result.state.step_index)
    assert manifest.time == float(result.state.time)
    assert manifest.artifacts[0].artifact_id == output.artifact_id
    assert (
        manifest.finite,
        manifest.converged,
        manifest.physically_valid,
        manifest.qualified,
        manifest.derivative_valid,
    ) == (
        receipt.finite,
        receipt.converged,
        receipt.physically_valid,
        receipt.qualified,
        receipt.derivative_valid,
    )
    prepared.publisher.close()


def test_bounded_terminal_manifests_preserve_checkpoint_and_distinct_statuses(
    tmp_path: Path,
):
    production, prepared, initial = _compiled(tmp_path, output_artifacts=1)
    state = prepared.initial_state(initial)

    canceled = _replace_status(state, "canceled", "")
    cancellation_result = ProductionRunResult(
        canceled, jnp.asarray(False), None, prepared.run_id, None
    )
    cancellation = production.cancellation_manifest(
        prepared, cancellation_result, "operator cancellation"
    )
    assert cancellation.preserved_checkpoint_id
    assert len(cancellation.checkpoint_receipt_id) == 64
    assert len(cancellation.checkpoint_content_digest) == 64
    assert cancellation.checkpoint_generation == 0
    assert cancellation.checkpoint_accepted_step == int(canceled.step_index)
    assert cancellation.checkpoint_durable_size_bytes > 0
    assert len(cancellation.checkpoint_durable_sha256) == 64
    restored_cancellation = prepared.resume(state)
    assert (
        restored_cancellation.last_checkpoint_id == cancellation.preserved_checkpoint_id
    )

    failed = _replace_status(state, "failed", "")
    failure = ProductionFailureRecord(
        0,
        0.0,
        "state-invalid",
        "PRODUCTION_STATE_INVALID",
        "",
    )
    failure_manifest = production.failure_manifest(
        prepared,
        ProductionRunResult(failed, jnp.asarray(False), failure, prepared.run_id, None),
    )
    assert failure_manifest.error_code == "PRODUCTION_STATE_INVALID"
    assert failure_manifest.finite
    assert not failure_manifest.qualified
