#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.cosmology import _dark_matter_qualification as qualification
from phydrax.applications.cosmology._coupled import ComovingEulerState
from phydrax.applications.cosmology._mixed_matter import (
    MixedDensityAssembly,
    SharedPeriodicGravityResult,
    WaveParticleGasCosmologyResult,
    WaveParticleGasCosmologyState,
)
from phydrax.applications.cosmology._production_profiles import (
    PeriodicWaveProductionMethod,
    RareSIDMProductionMethod,
)
from phydrax.applications.cosmology._simulation_products import (
    CommonGravitySimulationSnapshot,
    CommonGravitySnapshotEvidence,
    CosmologyOutputBundle,
    DarkMatterCheckpointContract,
    DarkMatterCheckpointPayload,
    DarkMatterRestartSnapshot,
    GasSimulationSnapshot,
    GasSnapshotEvidence,
    ParticleSimulationSnapshot,
    ParticleSnapshotEvidence,
    WaveSimulationSnapshot,
    WaveSnapshotEvidence,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import (
    CampaignRole,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
)
from phydrax.solver._production_runtime import (
    CheckpointGenerationPolicy,
    DurableCheckpointStore,
    PreparedProductionRun,
    ProductionCaseManifest,
)
from phydrax.solver._runtime_lifecycle import (
    ByteBoundedAsyncPublisher,
    ExactTimeSchedule,
)


cosmology = phx.applications.cosmology


def _reference():
    return ReferenceArtifactManifest(
        "dark-matter-reference",
        checksum_algorithm="sha256",
        checksum="0" * 64,
        size_bytes=1,
        license_id="internal-test",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"code_length": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=("test-lineage",),
    )


def _campaign(criteria):
    calibration = ScientificCase(
        "calibration-case",
        "calibration-unit",
        "calibration-construct",
        "calibration-condition",
        "calibration-preparation",
        "calibration-batch",
        ("calibration-source",),
    )
    locked = ScientificCase(
        "locked-case",
        "locked-unit",
        "locked-construct",
        "locked-condition",
        "locked-preparation",
        "locked-batch",
        ("locked-source",),
    )
    return ScientificCampaign(
        (calibration, locked),
        (
            CampaignRole("calibration", (calibration.case_id,)),
            CampaignRole("locked_evaluation", (locked.case_id,)),
        ),
        criteria_ids=tuple(value.criterion_id for value in criteria),
    )


_CLAIM_FACTORIES = (
    ("periodic-wave", qualification.periodic_wave_claim_profile),
    ("rare-sidm-equal", qualification.rare_equal_sidm_claim_profile),
    ("mixed-root-wave-particle", qualification.mixed_wave_particle_claim_profile),
    (
        "mixed-root-wave-particle-gas",
        qualification.mixed_wave_particle_gas_claim_profile,
    ),
    ("wave-amr-periodic", qualification.periodic_wave_amr_claim_profile),
    ("rare-sidm-differential", qualification.differential_sidm_claim_profile),
    ("rare-sidm-weighted", qualification.weighted_sidm_claim_profile),
    ("frequent-sidm-angular", qualification.frequent_sidm_claim_profile),
    ("sidm-fluid-spherical", qualification.gravothermal_sidm_claim_profile),
    ("sidm-inelastic-2to2", qualification.inelastic_sidm_claim_profile),
)


@pytest.mark.parametrize(("name", "factory"), _CLAIM_FACTORIES)
def test_dark_matter_claim_profiles_are_independent_and_complete(name, factory):
    metric_ids = qualification.dark_matter_claim_metric_ids(name)
    criteria = qualification.dark_matter_claim_criteria(
        name,
        {metric_id: 1.0 for metric_id in metric_ids},
    )
    campaign = _campaign(criteria)
    claim = factory(
        campaign,
        criteria,
        (f"{name}-locked-domain",),
        {
            "backend": "cpu",
            "dtype": "float64",
            "maximum_steps": 16,
            "maximum_state_values": 4096,
        },
        reference_artifacts=(_reference(),),
        requested_use=qualification.DarkMatterReferenceUse(),
    )

    support = dict(claim.support.attributes)
    assert support["profile"] == name
    assert support["production_inheritance"] is False
    assert support["automatic_regime_switching"] is False
    assert support["external_products"] == "stop-gradient"
    assert support["checkpoint_contract"] == "runtime-checkpoint-envelope"
    assert support["analysis_output_contract"] == "typed-snapshot-not-restart"
    assert set(value.metric_id for value in claim.criteria) == set(metric_ids)
    assert "changed-physics-or-support-identity" in claim.invalidation_triggers
    assert "missing-or-inadmissible-source-rights" in claim.invalidation_triggers


def _wave_case(schedule=(1.0, 1.001, 1.002), *, dtype=jnp.float64):
    precision = phx.discretization.SpectralPrecisionPolicy(dtype)
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(6),),
        axis_names=("x",),
        field_name="psi",
        precision=precision,
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    background = cosmology.FLRWBackground(1.0, 1.0)
    prepared = cosmology.WaveDarkMatterPlan(
        1.0,
        jnp.asarray(schedule),
        gravitational_constant=0.05,
        reduced_planck_constant=0.03,
        step_policy=cosmology.WaveDarkMatterStepPolicy(
            maximum_phase_radians=2.0,
            minimum_de_broglie_cells=2.0,
            norm_relative_tolerance=1.0e-5,
            poisson_relative_tolerance=1.0e-5,
            zero_mode_absolute_tolerance=1.0e-6,
        ),
    ).prepare(space, background)
    complex_dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    return prepared, prepared.initialize(jnp.ones((6,), dtype=complex_dtype))


def test_periodic_wave_adapter_executes_one_bounded_transaction_and_rolls_back():
    prepared, state = _wave_case()
    method = PeriodicWaveProductionMethod(prepared)
    first = method.step(0, 1.0, state, 0.001, None)
    second = method.step(1, 1.001, first.accepted_state, 0.001, None)

    assert bool(first.successful)
    assert bool(second.successful)
    np.testing.assert_allclose(second.accepted_state.scale_factor, 1.002)
    run_plan = method.production_run_plan(segment_steps=1)
    assert run_plan.method.method_id == method.method_id
    assert run_plan.maximum_steps == 2
    assert not method.allows_step_reduction

    rejected = method.step(0, 0.9, state, 0.001, None)
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.accepted_state.psi, state.psi)
    np.testing.assert_array_equal(
        rejected.accepted_state.scale_factor, state.scale_factor
    )


def test_periodic_wave_decimal_knots_run_through_production_runtime(tmp_path):
    prepared, state = _wave_case(
        jnp.asarray((0.1, 0.2, 0.3), dtype=jnp.float32),
        dtype=jnp.float32,
    )
    method = PeriodicWaveProductionMethod(prepared)
    schedule = ExactTimeSchedule(
        jnp.asarray((0.2, 0.3), dtype=jnp.float32),
        tolerance=method.schedule_tolerance,
    )
    plan = method.production_run_plan(
        segment_steps=1,
        output_schedule=schedule,
    )
    support = prepared.discretization.support
    manifest = ProductionCaseManifest(
        problem_id=prepared.prepared_id,
        method_id=method.method_id,
        precision_id="complex64",
        topology_id=support.topology.topology_id,
        geometry_layout_id=support.embedding_id,
        dtype="complex64",
    )
    store = DurableCheckpointStore(
        tmp_path / "decimal-store",
        manifest,
        CheckpointGenerationPolicy(2),
    )
    published = []
    publisher = ByteBoundedAsyncPublisher(
        lambda event_id, snapshot: published.append((event_id, snapshot)),
        maximum_pending=2,
        maximum_pending_bytes=4096,
    )
    runtime = PreparedProductionRun(manifest, plan, store, publisher=publisher)
    result = runtime.run(method.initial_run_state(runtime, state))
    publisher.close()
    store.close()

    assert bool(result.successful)
    assert result.failure is None
    assert result.state.status == "completed"
    assert result.state.last_checkpoint_id
    assert len(published) == 2
    np.testing.assert_allclose(result.state.time, 0.3, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        result.state.accepted_state.scale_factor,
        0.3,
        rtol=0.0,
        atol=1.0e-6,
    )
    assert int(result.state.step_index) == 2
    assert int(result.state.schedule_cursor) == 2
    assert int(result.state.output_cursor) == 2


def _gravity(particles):
    axes = tuple(
        phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y", "z")).prepare(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    system = phx.equations.EulerSystem(3)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "dark-matter-production-test",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLCFluxPlan(),
        ),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    return phx.solver.ParticleMeshGravityPlan(
        phx.solver.NewtonianSelfGravityPlan(0.01).prepare(
            phx.solver.prepare_balance_law_transport(runtime)
        ),
        phx.discretization.ParticleGridSplatPlan(grid).prepare(particles),
    )


def _sidm_case():
    positions = jnp.asarray(
        (
            (0.25, 0.25, 0.25),
            (0.25, 0.75, 0.75),
            (0.75, 0.25, 0.75),
            (0.75, 0.75, 0.25),
        )
    )
    masses = jnp.full((4,), 0.25)
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((101, 7, 83, 19)), masses, ambient_dimension=3
    ).prepare()
    kdk = cosmology.CosmologicalKDKPlan(particles, (1.0, 1.0, 1.0))
    particle_mesh = cosmology.CosmologicalParticleMeshPlan(
        kdk,
        _gravity(particles),
        (0.5, 0.51, 0.52),
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        6,
        box=phx.discretization.ParticleBox(
            jnp.zeros((3,)), jnp.ones((3,)), periodic_axes=(True, True, True)
        ),
    ).prepare(particles)
    sidm = cosmology.CosmologicalSIDMPlan(
        particle_mesh,
        neighborhood,
        phx.discretization.CoupledSummationSmoothingLengthPlan(
            1.0,
            1.0e-3,
            2.0,
            maximum_iterations=80,
            tolerance=1.0e-6,
            relaxation=0.7,
        ),
        phx.discretization.WendlandC2SPHKernel(3),
        cosmology.SIDMCrossSectionPlan(0.0),
        cosmology.SIDMCollisionPolicy(
            maximum_pair_probability=0.9,
            maximum_particle_probability=0.9,
            minimum_knudsen_number=1.0e-3,
            maximum_events_per_half_step=2,
        ),
    )
    state = kdk.initialize(positions, jnp.zeros_like(positions), 0.5)
    return sidm, state, particles


def test_rare_sidm_adapter_preserves_stable_ids_root_key_and_event_epoch():
    sidm, particles, support = _sidm_case()
    method = RareSIDMProductionMethod(sidm, cosmology.FLRWBackground(1.0, 0.3))
    initial = method.initialize(particles, jr.key(17), event_epoch=8)
    first = method.step(0, 0.5, initial, 0.01, None)
    second = method.step(1, 0.51, first.accepted_state, 0.01, None)

    assert bool(first.successful)
    assert bool(second.successful)
    assert int(first.accepted_state.event_epoch) == 10
    assert int(second.accepted_state.event_epoch) == 12
    np.testing.assert_array_equal(first.accepted_state.prng_root, initial.prng_root)
    np.testing.assert_array_equal(method.particle_ids, support.particle_ids)
    assert method.production_run_plan(segment_steps=1).maximum_steps == 2


def _checkpoint_contract(restart_template, *, physics_id="physics", support_id="support"):
    return DarkMatterCheckpointContract(
        profile_name="rare-sidm-equal",
        physics_id=physics_id,
        support_ids=(support_id,),
        source_ids=("native-source",),
        interaction_ids=("constant-isotropic-elastic",),
        artifact_ids=("artifact-rights",),
        topology_id="particle-topology",
        method_id="rare-sidm-production",
        precision_id="float64",
        restart_template=restart_template,
        topology_epoch_id="particle-layout",
        scale_id="cosmology-scale",
    )


def _restart_snapshot(*, parent=None, epoch=12, output_cursor=4, value=3.0):
    return DarkMatterRestartSnapshot(
        cosmology.CosmologicalParticleState(
            jnp.asarray(((value, 0.0, 0.0), (0.0, value, 0.0))),
            jnp.zeros((2, 3)),
            jnp.asarray(0.52),
        ),
        stable_ids=jnp.asarray((101, 7)),
        active_mask=jnp.asarray((True, True)),
        incarnations=jnp.asarray((0, 0)),
        lineage_ids=jnp.asarray(((-1, -1), (-1, -1))),
        prng_root=jr.key(29),
        event_epoch=epoch,
        time=0.52,
        accepted_step=2,
        schedule_cursor=2,
        output_cursor=output_cursor,
        accepted_evidence={"accepted": jnp.asarray(True)},
        parent_checkpoint_id=parent,
    )


def test_checkpoint_payload_roundtrips_and_rejects_changed_physics_or_support(tmp_path):
    snapshot = _restart_snapshot()
    contract = _checkpoint_contract(snapshot)
    payload = contract.payload(snapshot)
    path = contract.write(tmp_path / "dark-matter.phx", payload)
    restored = contract.read(path, snapshot)

    assert bool(restored.successful)
    assert restored.envelope.checkpoint_id == payload.envelope.checkpoint_id
    np.testing.assert_array_equal(restored.snapshot.stable_ids, snapshot.stable_ids)
    np.testing.assert_array_equal(restored.snapshot.prng_root, snapshot.prng_root)
    assert int(restored.snapshot.event_epoch) == 12
    assert int(restored.snapshot.output_cursor) == 4
    assert int(restored.snapshot.schedule_cursor) == 2

    with pytest.raises(ValueError, match="compatibility identities"):
        _checkpoint_contract(snapshot, physics_id="changed-physics").read(path, snapshot)
    with pytest.raises(ValueError, match="compatibility identities"):
        _checkpoint_contract(snapshot, support_id="changed-support").read(path, snapshot)

    store = DurableCheckpointStore(
        tmp_path / "store",
        contract.case_manifest(),
        CheckpointGenerationPolicy(2),
        encoding_plan=contract.encoding_plan,
    )
    first_receipt = contract.commit(store, payload)
    second_snapshot = _restart_snapshot(
        parent=first_receipt.checkpoint_id,
        epoch=14,
        output_cursor=5,
        value=4.0,
    )
    second_payload = contract.payload(second_snapshot)
    second_receipt = contract.commit(store, second_payload)
    spliced = DarkMatterCheckpointPayload(
        snapshot,
        second_payload.envelope,
        second_payload.recovery,
        contract.contract_id,
    )
    with pytest.raises(ValueError, match="parent chain|spliced"):
        contract.require_payload(spliced)
    latest = contract.restore_latest(store, snapshot)
    store.close()

    assert latest.envelope.checkpoint_id == second_receipt.checkpoint_id
    assert latest.snapshot.parent_checkpoint_id == first_receipt.checkpoint_id
    assert int(latest.snapshot.event_epoch) == 14
    np.testing.assert_array_equal(latest.snapshot.stable_ids, snapshot.stable_ids)


def _artifact(kind, *, parents=()):
    return ScientificArtifactEnvelope(
        artifact_kind=kind,
        content_digest=f"{kind}-content",
        producer="phydrax-test",
        producer_version="current",
        build_id="test-build",
        license_id="internal-test",
        resource_id="cpu",
        status="complete",
        parent_artifact_ids=parents,
    )


def test_typed_analysis_products_compose_without_becoming_restart_state():
    wave_prepared, wave_initial = _wave_case((0.5, 0.5001))
    wave_result = wave_prepared.solve(wave_initial)
    poisson_result = wave_prepared.poisson(wave_result.state)
    wave_evidence = WaveSnapshotEvidence(wave_result)
    wave = WaveSimulationSnapshot(
        wave_result.state.psi,
        wave_result.state.scale_factor,
        wave_evidence,
        _artifact("wave-snapshot"),
        prepared=wave_prepared,
        grid_id=wave_prepared.discretization.prepared_id,
        physics_id=wave_prepared.plan.plan_id,
        solver_id=wave_prepared.prepared_id,
        scale_id=wave_prepared.scale_id,
        coordinate_time_level="accepted-end-scale-factor",
        density=poisson_result.density,
        potential=poisson_result.potential,
        poisson_result=poisson_result,
    )
    with pytest.raises(ValueError, match="solver-produced|contradicts"):
        WaveSimulationSnapshot(
            wave_result.state.psi.at[0].set(jnp.nan + 0.0j),
            wave_result.state.scale_factor,
            wave_evidence,
            _artifact("invalid-wave-snapshot"),
            prepared=wave_prepared,
            grid_id=wave_prepared.discretization.prepared_id,
            physics_id=wave_prepared.plan.plan_id,
            solver_id=wave_prepared.prepared_id,
            scale_id=wave_prepared.scale_id,
            coordinate_time_level="accepted-end-scale-factor",
        )
    with pytest.raises(TypeError, match="Analysis products"):
        DarkMatterRestartSnapshot(
            {"nested": wave},
            time=0.5,
            accepted_step=1,
            schedule_cursor=1,
            output_cursor=1,
        )
    particle_support = phx.discretization.ParticleSetPlan(
        jnp.asarray((101, 7)),
        jnp.ones((2,)),
        ambient_dimension=3,
    ).prepare()
    particle_kinematics = cosmology.CosmologicalKDKPlan(particle_support, (1.0, 1.0, 1.0))
    particle_plan = cosmology.CosmologicalParticleMeshPlan(
        particle_kinematics,
        _gravity(particle_support),
        (0.5, 0.5001),
    )
    particle_initial = particle_kinematics.initialize(
        jnp.asarray(((0.25, 0.25, 0.25), (0.75, 0.75, 0.75))),
        jnp.zeros((2, 3)),
        0.5,
    )
    particle_result = particle_plan.rollout(
        cosmology.FLRWBackground(1.0, 1.0),
        particle_initial,
    )
    particle_state = particle_result.state
    particle_evidence = ParticleSnapshotEvidence(
        particle_result,
        particle_support,
        producer_plan=particle_plan,
    )
    particles = ParticleSimulationSnapshot(
        particle_support.particle_ids,
        particle_state.positions,
        particle_state.canonical_momenta,
        particle_support.active_mask,
        particle_support.plan.masses,
        jnp.ones((2,)),
        jnp.zeros((2,), dtype=jnp.int64),
        jnp.asarray(((-1, -1), (-1, -1))),
        particle_state.scale_factor,
        particle_evidence,
        _artifact("particle-snapshot"),
        particle_result,
        particle_support,
        support_id=particle_support.prepared_id,
        interaction_id=particle_plan.plan_id,
        physics_id=particle_evidence.producer_result_id,
        scale_id=particle_kinematics.scale.scale_id,
        coordinate_time_level="accepted-end-scale-factor",
    )
    with pytest.raises(ValueError, match="typed result|contradicts"):
        ParticleSimulationSnapshot(
            jnp.asarray((101, 101)),
            jnp.zeros((2, 3)),
            jnp.zeros((2, 3)),
            jnp.asarray((True, True)),
            jnp.ones((2,)),
            jnp.ones((2,)),
            jnp.zeros((2,), dtype=jnp.int64),
            jnp.asarray(((-1, -1), (-1, -1))),
            particle_state.scale_factor,
            particle_evidence,
            _artifact("duplicate-particle-snapshot"),
            particle_result,
            particle_support,
            support_id=particle_support.prepared_id,
            interaction_id=particle_plan.plan_id,
            physics_id=particle_evidence.producer_result_id,
            scale_id=particle_kinematics.scale.scale_id,
            coordinate_time_level="accepted-end-scale-factor",
        )
    with pytest.raises(ValueError, match="lineage|typed result"):
        ParticleSimulationSnapshot(
            jnp.asarray((101, 7)),
            jnp.zeros((2, 3)),
            jnp.zeros((2, 3)),
            jnp.asarray((True, True)),
            jnp.ones((2,)),
            jnp.ones((2,)),
            jnp.ones((2,), dtype=jnp.int64),
            jnp.asarray(((7, 0), (101, 0))),
            particle_state.scale_factor,
            particle_evidence,
            _artifact("cyclic-particle-snapshot"),
            particle_result,
            particle_support,
            support_id=particle_support.prepared_id,
            interaction_id=particle_plan.plan_id,
            physics_id=particle_evidence.producer_result_id,
            scale_id=particle_kinematics.scale.scale_id,
            coordinate_time_level="accepted-end-scale-factor",
        )

    gas_fields = jnp.zeros((2, 5)).at[:, 0].set(1.0).at[:, -1].set(2.0)
    gravity_density = jnp.ones((2,))
    gravity_potential = jnp.zeros((2,))
    gravity_acceleration = jnp.zeros((2, 3))
    assembly = MixedDensityAssembly(
        wave_density=jnp.zeros((2,)),
        particle_density=gravity_density,
        gas_density=jnp.zeros((2,)),
        total_density=gravity_density,
        component_mass=jnp.asarray((0.0, 2.0, 0.0)),
        total_mass=jnp.asarray(2.0),
        particle_source_mass=jnp.asarray(2.0),
        particle_deposited_mass=jnp.asarray(2.0),
        particle_mass_balance_defect=jnp.asarray(0.0),
        particle_routes=None,
        scale_factor=wave_result.state.scale_factor,
        density_nonnegative=jnp.asarray(True),
        finite=jnp.asarray(True),
        successful=jnp.asarray(True),
        component_names=("wave", "particles", "gas"),
        scale_id=wave_prepared.scale_id,
        coordinate_convention="flat-periodic-comoving-cartesian",
        density_convention="comoving-mass-per-comoving-volume",
        assembler_id="test-assembler",
    )
    gravity_result = SharedPeriodicGravityResult(
        assembly=assembly,
        potential=gravity_potential,
        cell_acceleration=gravity_acceleration,
        particle_acceleration=jnp.zeros((2, 3)),
        mean_density=jnp.asarray(1.0),
        source_integral=jnp.asarray(0.0),
        poisson_relative_residual=jnp.asarray(0.0),
        gauge_defect=jnp.asarray(0.0),
        component_force=jnp.zeros((3, 3)),
        total_force=jnp.zeros((3,)),
        particle_force_adjoint_defect=jnp.asarray(0.0),
        particle_support_complete=jnp.asarray(True),
        finite=jnp.asarray(True),
        successful=jnp.asarray(True),
        mean_removal_count=1,
        gravitational_constant=1.0,
        potential_convention="phi=a*Phi",
        plan_id="test-shared-gravity",
    )
    mixed_state = WaveParticleGasCosmologyState(
        wave_result.state,
        particle_state,
        ComovingEulerState(gas_fields, wave_result.state.scale_factor),
    )
    mixed_result = WaveParticleGasCosmologyResult(
        mixed_state,
        gravity_result,
        None,
        jnp.asarray(True),
        "test-mixed-gas",
        wave_prepared.prepared_id,
        particle_plan.plan_id,
        "ideal-gas",
        particle_support.prepared_id,
        "test-shared-gravity",
        "test-assembler",
        wave_prepared.scale_id,
    )
    gas_evidence = GasSnapshotEvidence(mixed_result)
    gas = GasSimulationSnapshot(
        gas_fields,
        wave_result.state.scale_factor,
        gas_evidence,
        _artifact("gas-snapshot"),
        component_names=("mass", "momentum-x", "momentum-y", "momentum-z", "energy"),
        geometry_id="test-assembler",
        eos_id="ideal-gas",
        source_id="test-shared-gravity",
        physics_id="test-mixed-gas",
        scale_id=wave_prepared.scale_id,
        coordinate_time_level="accepted-end-scale-factor",
    )
    with pytest.raises(ValueError, match="minimum-density|typed producer"):
        GasSimulationSnapshot(
            gas_fields.at[0, 0].set(-1.0),
            wave_result.state.scale_factor,
            gas_evidence,
            _artifact("negative-gas-snapshot"),
            component_names=(
                "mass",
                "momentum-x",
                "momentum-y",
                "momentum-z",
                "energy",
            ),
            geometry_id="test-assembler",
            eos_id="ideal-gas",
            source_id="test-shared-gravity",
            physics_id="test-mixed-gas",
            scale_id=wave_prepared.scale_id,
            coordinate_time_level="accepted-end-scale-factor",
        )
    gravity_evidence = CommonGravitySnapshotEvidence(gravity_result)
    gravity = CommonGravitySimulationSnapshot(
        gravity_density,
        gravity_potential,
        gravity_acceleration,
        wave_result.state.scale_factor,
        gravity_evidence,
        _artifact("common-gravity-snapshot"),
        source_id="test-assembler",
        operator_id="test-shared-gravity",
        geometry_id="test-assembler",
        physics_id="test-shared-gravity",
        scale_id=wave_prepared.scale_id,
        potential_time_level="accepted-end-scale-factor",
        producer_result=gravity_result,
    )
    with pytest.raises(ValueError, match="typed producer"):
        CommonGravitySimulationSnapshot(
            gravity_density,
            gravity_potential + 1.0,
            gravity_acceleration,
            wave_result.state.scale_factor,
            gravity_evidence,
            _artifact("forged-common-gravity-snapshot"),
            source_id="test-assembler",
            operator_id="test-shared-gravity",
            geometry_id="test-assembler",
            physics_id="test-shared-gravity",
            scale_id=wave_prepared.scale_id,
            potential_time_level="accepted-end-scale-factor",
            producer_result=gravity_result,
        )
    child_artifacts = tuple(
        value.artifact.artifact_id for value in (particles, gas, gravity)
    )
    bundle = CosmologyOutputBundle(
        wave_result.state.scale_factor,
        _artifact("cosmology-output-bundle", parents=child_artifacts),
        execution_id="production-run",
        parent_checkpoint_id=None,
        output_cursor=1,
        cosmology_id="flat-flrw",
        scale_id=wave_prepared.scale_id,
        particles=particles,
        gas=gas,
        common_gravity=gravity,
        producer_result=mixed_result,
    )

    assert bool(bundle.status.successful)
    assert bundle.shared_gravity_result_id == gravity.producer_result_id
    assert bundle.child_artifact_ids == child_artifacts
    restart = _restart_snapshot()
    assert not isinstance(restart, (WaveSimulationSnapshot, ParticleSimulationSnapshot))
