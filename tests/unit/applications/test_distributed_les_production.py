#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding

import phydrax as phx
from phydrax.applications.incompressible_flow._distributed_les import (
    compile_distributed_periodic_les,
    DistributedPeriodicLESMethodPlan,
    DistributedPeriodicLESProductionCase,
    DistributedPeriodicLESProductionPlan,
    DistributedPeriodicLESStatisticsPlan,
)
from phydrax.applications.incompressible_flow._forcing import (
    ConstantPowerFourierForcingPlan,
)
from phydrax.discretization.spectral._distributed import (
    SpectralMeshTopology,
    SpectralResourceError,
)
from phydrax.discretization.spectral._distributed_les import (
    DistributedPeriodicLESPlan,
    PreparedDistributedPeriodicLES,
)
from phydrax.equations._incompressible import _PeriodicRotationalDrift
from phydrax.equations._les_closures import (
    LESParameterProvenance,
    ResolvedLESFilter,
    SmagorinskyLESPlan,
)
from phydrax.equations._periodic_les import (
    PeriodicAlgebraicLESPlan,
    PeriodicFourierGridFilterPlan,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import SupportDependency
from phydrax.solver._production_runtime import ArtifactCheckpointStore


def _space(count=4):
    return phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(tuple(phx.discretization.AxisDomain.periodic(0.0, 1.0) for _ in range(3)))


def _algebraic_plan(space):
    resolved_filter = ResolvedLESFilter(
        "retained Fourier grid",
        family="sharp-fourier-projection",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    provenance = LESParameterProvenance(
        resolved_filter,
        space.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    model = SmagorinskyLESPlan(0.16).prepare(provenance)
    return PeriodicAlgebraicLESPlan(
        model,
        PeriodicFourierGridFilterPlan(resolved_filter),
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.OversamplingDealiasingPlan(1.5)
        ),
        energy_tolerance=3.0e-9,
    )


def _topology(schedule="slab", devices=None):
    available = (jax.devices("cpu")[0],) if devices is None else tuple(devices)
    if schedule == "pencil":
        if len(available) == 1:
            shape = (1, 1)
        else:
            shape = (2, len(available) // 2)
        return SpectralMeshTopology(
            shape,
            devices=available,
            axis_names=("px", "py"),
        )
    return SpectralMeshTopology(
        (len(available),),
        devices=available,
        axis_names=("spectral",),
    )


def _velocity(space):
    x, y, z = jnp.meshgrid(
        space.axes[0].nodes,
        space.axes[1].nodes,
        space.axes[2].nodes,
        indexing="ij",
    )
    physical = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * y),
            jnp.sin(2.0 * jnp.pi * z),
            jnp.sin(2.0 * jnp.pi * x),
        ),
        axis=-1,
    )
    return physical


def _compiled(
    *,
    count=4,
    schedule="slab",
    devices=None,
    checkpoint_count=1,
    maximum_bytes=2 * 1024**3,
    forcing=False,
):
    space = _space(count)
    algebraic = _algebraic_plan(space)
    spatial = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.OversamplingDealiasingPlan(1.5)
    )
    problem = phx.equations.IncompressibleFlowProblem(3, 0.01)
    projector = phx.discretization.PeriodicLerayProjector(space)
    scientific = algebraic.prepare(space, projector)
    prepared_spatial = spatial.prepare(
        space,
        required_polynomial_degree=2,
        nonlinear=True,
    )
    local = _PeriodicRotationalDrift(
        problem,
        space,
        prepared_spatial,
        projector,
        scientific,
        None,
    )
    source = DistributedPeriodicLESPlan(
        scientific,
        _topology(schedule, devices),
        schedule=schedule,
        checkpoint_count=checkpoint_count,
        maximum_bytes=maximum_bytes,
    )
    constant_power = (
        ConstantPowerFourierForcingPlan(
            projector,
            maximum_wavenumber=7.0,
            power_input=0.05,
            minimum_forced_energy=1.0e-12,
        )
        if forcing
        else None
    )
    distributed = compile_distributed_periodic_les(
        problem,
        source,
        constant_power_forcing=constant_power,
    )
    state = distributed.project_state(space.project(_velocity(space)))
    return space, local, source, distributed, constant_power, state


def test_distributed_full_flow_single_device_parity_forcing_jit_and_jvp():
    _, local, _, distributed, _, state = _compiled()

    local_stage = local.stage(jnp.asarray(0.0), state)
    stage = distributed.stage(jnp.asarray(0.0), state)
    expected = distributed.backend.execution.modal_layout.sharding(
        distributed.backend.execution.topology
    )

    np.testing.assert_allclose(
        stage.rates.advective_rate,
        local_stage.rates.advective_rate,
        rtol=3.0e-8,
        atol=3.0e-8,
    )
    np.testing.assert_allclose(
        stage.rates.algebraic_les_rate,
        local_stage.rates.algebraic_les_rate,
        rtol=3.0e-8,
        atol=3.0e-8,
    )
    np.testing.assert_allclose(
        stage.rates.molecular_rate,
        local_stage.rates.molecular_rate,
        rtol=3.0e-8,
        atol=3.0e-8,
    )
    np.testing.assert_allclose(
        stage.rates.total_rate,
        local_stage.rates.total_rate,
        rtol=3.0e-8,
        atol=3.0e-8,
    )
    assert stage.rates.total_rate.sharding == expected
    assert distributed.qualification_inherited is False

    eager = stage.rates.total_rate
    compiled = jax.jit(lambda value: distributed(0.0, value, None))(state)
    _, tangent = jax.jvp(
        lambda value: distributed(0.0, value, None),
        (state,),
        (jnp.ones_like(state),),
    )
    np.testing.assert_allclose(compiled, eager, rtol=3.0e-8, atol=3.0e-8)
    assert bool(jnp.all(jnp.isfinite(tangent)))
    assert compiled.sharding == expected

    _, _, _, forced, forcing, forced_state = _compiled(forcing=True)
    forcing_result = forcing.evaluate(forced_state)
    forced_stage = forced.stage(jnp.asarray(0.0), forced_state)
    np.testing.assert_allclose(
        forced_stage.rates.forcing_rate,
        forcing_result.forcing,
        rtol=3.0e-8,
        atol=3.0e-8,
    )
    assert bool(forcing_result.successful)
    assert bool(forced_stage.forcing_successful)
    assert float(jnp.max(jnp.abs(forced_stage.rates.forcing_rate))) > 0.0


@pytest.mark.parametrize("scheme", ("etdrk2", "etdrk4", "ssprk33", "ssprk54"))
def test_distributed_fixed_step_accepts_and_rejects_transactionally(scheme):
    space, _, _, dynamics, _, state = _compiled()
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        space, component_shape=(3,)
    )
    method = DistributedPeriodicLESMethodPlan(scheme, safety_factor=0.8).prepare(
        dynamics, coordinates
    )
    restriction = method.step_restriction(0.0, state)
    selected = (
        restriction.etdrk_selected
        if scheme.startswith("etdrk")
        else restriction.fully_explicit_selected
    )
    accepted_step = min(1.0e-4, 0.1 * float(selected))
    accepted = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(accepted_step),
        None,
    )
    rejected = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(0.81 * float(selected)),
        None,
    )

    assert bool(accepted.successful)
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.accepted_state, state)
    assert isinstance(accepted.accepted_state.sharding, NamedSharding)
    assert accepted.accepted_state.sharding == state.sharding
    assert method.method_id != method.plan.plan_id


def test_distributed_sharded_statistics_and_no_host_gather(monkeypatch):
    space, _, _, dynamics, _, state = _compiled()
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        space, component_shape=(3,)
    )
    statistics = DistributedPeriodicLESStatisticsPlan(dynamics, coordinates)

    with monkeypatch.context() as guard:
        guard.setattr(
            jax,
            "device_get",
            lambda *_args, **_kwargs: pytest.fail("host gather is forbidden"),
        )
        result = statistics.evaluate(0.0, state)
        restart = dynamics.backend.restart_evidence(state)
        restored = dynamics.backend.restore(restart)

    assert bool(result.finite)
    assert bool(result.successful)
    assert result.reduction_axes == ("spectral",)
    assert result.sharding_preserved
    assert result.kinetic_energy > 0.0
    assert result.molecular_dissipation >= 0.0
    assert restart.sharding_preserved
    np.testing.assert_array_equal(restored, state)
    source = inspect.getsource(type(dynamics))
    backend_source = inspect.getsource(PreparedDistributedPeriodicLES)
    assert "device_get" not in source
    assert "process_allgather" not in source
    assert "device_get" not in backend_source
    assert "process_allgather" not in backend_source


def _artifact_store(tmp_path, plan):
    profile = HPCFilesystemProfile(
        "distributed-les-posix",
        "test-filesystem",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    repository = POSIXArtifactRepository(
        tmp_path / "distributed-les-artifact",
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=256,
            maximum_metadata_bytes=1024 * 1024,
        ),
    )
    checkpoint_policy = phx.solver.CheckpointGenerationPolicy(plan.checkpoint_retention)
    dependency = SupportDependency(
        "repository-profile", repository.support_tuple.support_tuple_id
    )
    resolved = ResolvedRunSpec(
        (),
        (dependency,),
        release_index_id="release-index",
        profile_ids=(dependency.profile_id,),
        trust_policy_id="trust-policy",
        valid_at=10,
        valid_from=0,
        valid_until=20,
        prepared_configuration_id=plan.plan_id,
        precision_policy_id=plan.manifest.precision_id,
        resource_policy_id=plan.dynamics.backend.preparation.resource.report_id,
        checkpoint_policy_id=checkpoint_policy.policy_id,
        output_policy_id="distributed-output-policy",
        repository_id=repository.provider_id,
        scheduler_id="distributed-scheduler",
        auth_policy_id="distributed-auth-policy",
    )
    return ArtifactCheckpointStore(
        repository,
        plan.manifest,
        checkpoint_policy,
        resolved,
        writer_id="distributed-les-worker",
        encoding_plan=plan.checkpoint_encoding,
    )


def test_distributed_production_consumes_plan_and_artifact_restart_is_exact(
    tmp_path, monkeypatch
):
    _, _, source, dynamics, _, state = _compiled(checkpoint_count=1)
    problem = phx.equations.IncompressibleFlowProblem(3, 0.01)
    case = DistributedPeriodicLESProductionCase(
        dynamics,
        state,
        case_id="distributed-les-case",
    )
    plan = DistributedPeriodicLESProductionPlan(
        problem,
        source,
        DistributedPeriodicLESMethodPlan("etdrk2", safety_factor=0.8),
        case,
        start_time=0.0,
        end_time=2.0e-4,
        step_size=1.0e-4,
        checkpoint_interval=1,
        segment_steps=1,
        checkpoint_retention=2,
    )
    prepared = plan.prepare(_artifact_store(tmp_path, plan))
    initial = prepared.initialize(state)
    with pytest.raises(ValueError, match="bound production case"):
        prepared.initialize(2.0 * state)
    with monkeypatch.context() as guard:
        guard.setattr(
            jax,
            "device_get",
            lambda *_args, **_kwargs: pytest.fail(
                "distributed production gathered its accepted state"
            ),
        )
        following, transition = prepared.step(initial)
    checkpointed = prepared.checkpoint(following)
    resumed = prepared.resume(checkpointed)
    evidence = prepared.restart_evidence(resumed)
    expected = plan.dynamics.backend.execution.modal_layout.sharding(
        plan.dynamics.backend.execution.topology
    )

    assert source.plan_id == plan.source_plan.plan_id
    assert plan.dynamics.source_plan.plan_id == source.plan_id
    assert plan.qualification_inherited is False
    assert plan.runtime_plan.device_resident
    assert plan.dynamics.qualification_inherited is False
    assert bool(transition.successful)
    np.testing.assert_array_equal(resumed.accepted_state, following.accepted_state)
    assert following.accepted_state.sharding == expected
    assert transition.accepted_state.sharding == expected
    assert resumed.accepted_state.sharding == expected
    assert evidence.sharding_preserved
    assert resumed.last_checkpoint_id == checkpointed.last_checkpoint_id
    assert plan.manifest.topology_id == source.topology.topology_id
    assert (
        plan.manifest.geometry_layout_id
        == plan.dynamics.backend.execution.modal_layout.layout_id
    )

    changed_case = DistributedPeriodicLESProductionCase(
        dynamics,
        2.0 * state,
        case_id="distributed-les-case",
    )
    changed = DistributedPeriodicLESProductionPlan(
        problem,
        source,
        DistributedPeriodicLESMethodPlan("etdrk2", safety_factor=0.8),
        changed_case,
        start_time=0.0,
        end_time=2.0e-4,
        step_size=1.0e-4,
        checkpoint_interval=1,
        segment_steps=1,
        checkpoint_retention=2,
    )
    assert changed.plan_id != plan.plan_id
    with pytest.raises(ValueError, match="exactly bind"):
        changed.prepare(prepared.checkpoint_store)


def test_distributed_production_resource_refusal_precedes_runtime(tmp_path):
    _, local, zero_checkpoint_plan, dynamics, _, state = _compiled(checkpoint_count=0)
    case = DistributedPeriodicLESProductionCase(
        dynamics,
        state,
        case_id="resource-refusal",
    )
    with pytest.raises(ValueError, match="checkpoint_count>=1"):
        DistributedPeriodicLESProductionPlan(
            phx.equations.IncompressibleFlowProblem(3, 0.01),
            zero_checkpoint_plan,
            DistributedPeriodicLESMethodPlan(),
            case,
            start_time=0.0,
            end_time=0.01,
            step_size=0.001,
            checkpoint_interval=1,
        )

    with pytest.raises(SpectralResourceError) as caught:
        DistributedPeriodicLESPlan(
            local.algebraic_les,
            _topology(),
            checkpoint_count=1,
            maximum_bytes=128,
        ).prepare()
    assert caught.value.report.total_bytes > caught.value.report.maximum_bytes
    assert not (tmp_path / "unprepared-runtime").exists()


def test_distributed_full_flow_real_multi_device_slab_pencil_when_available():
    devices = tuple(jax.devices("cpu"))
    if len(devices) < 4:
        pytest.skip("Four forced CPU devices are required for slab/pencil execution.")
    _, _, _, slab, _, state = _compiled(
        count=8,
        schedule="slab",
        devices=devices[:4],
    )
    _, _, _, pencil, _, _ = _compiled(
        count=8,
        schedule="pencil",
        devices=devices[:4],
    )
    slab_stage = slab.stage(0.0, state)
    pencil_state = pencil.project_state(state)
    pencil_stage = pencil.stage(0.0, pencil_state)

    np.testing.assert_allclose(
        slab_stage.rates.total_rate,
        pencil_stage.rates.total_rate,
        rtol=4.0e-8,
        atol=4.0e-8,
    )
    assert len(slab_stage.rates.total_rate.addressable_shards) == 4
    assert len(pencil_stage.rates.total_rate.addressable_shards) == 4
    assert (
        slab_stage.rates.total_rate.sharding
        == slab.backend.execution.modal_layout.sharding(slab.backend.execution.topology)
    )
    assert (
        pencil_stage.rates.total_rate.sharding
        == pencil.backend.execution.modal_layout.sharding(
            pencil.backend.execution.topology
        )
    )
