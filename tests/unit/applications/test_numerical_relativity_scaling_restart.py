#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.numerical_relativity._amr import (
    NumericalRelativityAMRHaloPlan,
    NumericalRelativityAMRState,
    NumericalRelativityAMRTopologyEpoch,
    NumericalRelativityAMRTopologyTransition,
    reflux_relativistic_material,
    RelativisticMagneticAMRTransferPlan,
    RelativisticMagneticRefluxPlan,
    RelativisticMaterialTransferPlan,
    RelativisticRadiationTransferPlan,
    Z4cAMRTransferPlan,
)
from phydrax.applications.numerical_relativity._checkpoint import (
    assemble_distributed_numerical_relativity_checkpoint,
    evaluate_numerical_relativity_restart,
    NumericalRelativityCheckpointPlan,
    NumericalRelativityRestartPolicy,
    NumericalRelativityRestartState,
    publish_distributed_numerical_relativity_checkpoint,
    read_numerical_relativity_checkpoint,
    restore_distributed_numerical_relativity_checkpoint,
    write_numerical_relativity_checkpoint,
)
from phydrax.applications.numerical_relativity._coupled_runtime import (
    CoupledEvolutionState,
)
from phydrax.applications.numerical_relativity._distributed import (
    NumericalRelativityAMRDistributionPlan,
    NumericalRelativityDistributedPlan,
)
from phydrax.applications.numerical_relativity._matter_coupling import CoupledBudget
from phydrax.applications.numerical_relativity._state import flat_z4c_state
from phydrax.applications.numerical_relativity._temporal import Z4cRuntimeState
from phydrax.discretization import (
    BlockHierarchyPlan,
    BlockHierarchyState,
    BlockLevelPlan,
    BlockLevelState,
    BlockTopologyCompiler,
    FDAMRHierarchyPlan,
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.discretization.amr import FluxRegister
from phydrax.lifecycle import ProcessCheckpointPublication
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.solver._grmhd_ct import (
    GRMHDConstrainedTransportPlan,
    GRMHDCTState,
    GRMHDVectorPotentialGauge,
)
from phydrax.solver._grmhd_runtime import GRMHDState
from phydrax.solver._grrmhd_runtime import GRRMHDState


def _bridge(count, upper=1.0):
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(count, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (upper, upper, upper))))
    return StructuredCochainBridge(grid)


def _block_setup(*, fine_capacity=8, overflow=False):
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(4, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))))
    hierarchy = BlockHierarchyPlan(
        grid,
        (
            BlockLevelPlan(0, (2, 2, 2), 8, halo_width=1),
            BlockLevelPlan(1, (2, 2, 2), fine_capacity, halo_width=1),
        ),
    )
    compiler = BlockTopologyCompiler(hierarchy)
    initial = compiler.initialize()
    tags = jnp.zeros((8, 2, 2, 2), dtype=bool).at[0, 0, 0, 0].set(True)
    if overflow:
        tags = tags.at[7, 1, 1, 1].set(True)
    compiled = compiler.compile(initial.topology, (tags,))
    return hierarchy, initial, compiled, FDAMRHierarchyPlan(hierarchy).prepare()


def _hierarchy_state(topology, component_count=25, *, dtype):
    levels = tuple(
        BlockLevelState(
            plan,
            metadata,
            jnp.zeros(
                (plan.maximum_blocks, *plan.block_shape, component_count),
                dtype=dtype,
            ),
        )
        for plan, metadata in zip(topology.plan.levels, topology.levels, strict=True)
    )
    return BlockHierarchyState(topology, levels)


def _epoch(formulation, hierarchy, compilation, fd, *, magnetic_bridge=None):
    prepared = NumericalRelativityAMRDistributionPlan(formulation, hierarchy, 1).prepare(
        compilation, fd
    )
    state = _hierarchy_state(
        compilation.topology,
        dtype=fd.plan.precision.field_dtype,
    )
    return state, NumericalRelativityAMRTopologyEpoch(
        formulation,
        state,
        prepared,
        magnetic_bridge=magnetic_bridge,
    )


def test_single_device_named_fields_and_authoritative_block_ownership_fillpatch():
    fixed = NumericalRelativityDistributedPlan(
        "z4c",
        (4, 4, 4),
        (1, 1, 1),
        halo_width=1,
        periodic=(True, True, True),
        grid_id="grid",
    ).prepare(jax.devices()[:1])
    z4c = fixed.shard_z4c(jnp.zeros((25, 4, 4, 4)))
    material = fixed.shard_material(jnp.zeros((4, 4, 4, 5)))
    assert fixed.single_device_authority
    assert z4c.sharding == fixed.z4c_sharding
    assert material.sharding == fixed.material_sharding
    assert fixed.periodic_z4c_halo(z4c, 0).shape == (25, 6, 4, 4)
    assert fixed.periodic_material_halo(material, 1).shape == (4, 6, 4, 5)

    coupled = NumericalRelativityDistributedPlan(
        "z4c-grrmhd",
        (4, 4, 4),
        (1, 1, 1),
        halo_width=1,
        periodic=(True, True, True),
        grid_id="grrmhd-grid",
    ).prepare(jax.devices()[:1])
    radiation = coupled.shard_radiation(jnp.zeros((4, 4, 4, 4)))
    assert coupled.plan.formulation == "z4c-grrmhd"
    assert coupled.periodic_radiation_halo(radiation, 2).shape == (4, 4, 6, 4)

    hierarchy, _, compiled, fd = _block_setup()
    state, epoch = _epoch("z4c", hierarchy, compiled, fd)
    distribution = epoch.distribution
    packed = distribution.pack(state)
    restored = distribution.unpack(packed)
    for level, ownership in enumerate(distribution.ownership):
        active = np.asarray(compiled.topology.levels[level].active)
        np.testing.assert_array_equal(np.asarray(ownership.owner_indices)[active], 0)
    assert distribution.single_device_authority
    assert restored.topology.epoch.epoch_id == state.topology.epoch.epoch_id

    halo, request = NumericalRelativityAMRHaloPlan(compiled.topology, 0, "z4c").execute(
        state, state, state, 0.0, 0.0, 0.0
    )
    assert bool(jnp.all(halo.valid))
    assert bool(halo.finite)
    assert not request.required
    assert halo.source_class.shape == halo.valid.shape


def test_z4c_and_material_transfers_report_constraints_and_conservation():
    source = flat_z4c_state((2, 2, 2), grid_id="coarse")
    values = source.values.at[1].set(4.0).at[8].set(0.3)
    transferred, z4c_evidence = Z4cAMRTransferPlan(constraint_tolerance=2.0e-5).prolong(
        source.with_values(values), target_grid_id="fine"
    )
    assert transferred.values.shape == (25, 4, 4, 4)
    assert bool(z4c_evidence.finite)
    assert bool(z4c_evidence.physically_valid)
    assert bool(z4c_evidence.constraint_valid)
    assert bool(z4c_evidence.qualified)
    assert not bool(z4c_evidence.derivative_valid)

    material = jnp.zeros((2, 2, 2, 5)).at[..., 0].set(2.0).at[..., 4].set(3.0)
    transfer = RelativisticMaterialTransferPlan(conservation_tolerance=1.0e-6)
    fine, prolong_evidence = transfer.prolong(material, 1.0)
    restored, restrict_evidence = transfer.restrict(fine, 0.125)
    np.testing.assert_allclose(restored, material, rtol=0.0, atol=1.0e-6)
    assert bool(prolong_evidence.conservation_valid)
    assert bool(restrict_evidence.conservation_valid)

    radiation = jnp.zeros((2, 2, 2, 4)).at[..., 0].set(2.0).at[..., 1].set(0.5)
    radiation_transfer = RelativisticRadiationTransferPlan(conservation_tolerance=1.0e-6)
    fine_radiation, radiation_prolong = radiation_transfer.prolong(radiation, 1.0)
    restored_radiation, radiation_restrict = radiation_transfer.restrict(
        fine_radiation, 0.125
    )
    np.testing.assert_allclose(restored_radiation, radiation, atol=1.0e-6)
    assert bool(radiation_prolong.qualified)
    assert bool(radiation_restrict.qualified)

    coarse_flux = jnp.zeros_like(material)
    fine_flux = coarse_flux.at[0, 0, 0, 0].set(0.25)
    register = FluxRegister(
        coarse_flux,
        fine_flux,
        jnp.ones(material.shape[:3], dtype=bool),
        register_id="material-register",
    )
    reflux = reflux_relativistic_material(
        material, jnp.ones(material.shape[:3]), register
    )
    assert bool(reflux.qualified)
    np.testing.assert_allclose(reflux.conservation_residual, 0.0, atol=1.0e-7)


def test_magnetic_transfer_and_emf_curl_reflux_preserve_divergence():
    coarse = _bridge(2)
    fine = _bridge(4)
    magnetic = coarse.pack_normal_flux(
        tuple(jnp.ones(coarse.grid.shape) * value for value in (0.2, -0.1, 0.3))
    )
    transfer = RelativisticMagneticAMRTransferPlan(
        coarse,
        fine,
        direction="prolong",
        divergence_tolerance=1.0e-6,
    )
    reverse = RelativisticMagneticAMRTransferPlan(
        coarse,
        fine,
        direction="restrict",
        divergence_tolerance=1.0e-6,
    )
    fine_magnetic, prolong_evidence = transfer.prolong(magnetic)
    restored, restrict_evidence = reverse.restrict(fine_magnetic)
    assert bool(prolong_evidence.divergence_valid)
    assert bool(restrict_evidence.divergence_valid)
    np.testing.assert_allclose(restored, magnetic, rtol=0.0, atol=1.0e-6)

    edge_count = coarse.cochain.cell_counts[1]
    updated, reflux_evidence = RelativisticMagneticRefluxPlan(
        coarse, divergence_tolerance=1.0e-6
    ).reflux(
        magnetic,
        jnp.zeros((edge_count,)),
        jnp.linspace(0.0, 1.0e-6, edge_count),
    )
    assert updated.shape == magnetic.shape
    assert bool(reflux_evidence.divergence_valid)
    assert bool(reflux_evidence.qualified)
    hierarchy, initial, compiled, fd = _block_setup()
    _, source_epoch = _epoch("grmhd", hierarchy, initial, fd, magnetic_bridge=coarse)
    _, target_epoch = _epoch("grmhd", hierarchy, compiled, fd, magnetic_bridge=fine)
    unrelated = RelativisticMagneticAMRTransferPlan(
        _bridge(2, upper=2.0),
        _bridge(4, upper=2.0),
        direction="prolong",
        divergence_tolerance=1.0e-6,
    )
    with pytest.raises(ValueError, match="bridges"):
        NumericalRelativityAMRTopologyTransition(
            source_epoch,
            compiled,
            target_epoch,
            transfer_plans=(
                RelativisticMaterialTransferPlan(),
                unrelated,
            ),
        )


def test_compiled_topology_transition_commits_or_retains_predecessor_on_overflow():
    hierarchy, initial, compiled, fd = _block_setup()
    _, source_epoch = _epoch("z4c", hierarchy, initial, fd)
    _, target_epoch = _epoch("z4c", hierarchy, compiled, fd)
    capacity = sum(level.maximum_blocks for level in hierarchy.levels)
    flat = flat_z4c_state((2, 2, 2), grid_id="block").values
    source_values = jnp.broadcast_to(flat, (capacity,) + flat.shape)
    predecessor = NumericalRelativityAMRState(source_epoch, (source_values,))
    transfer = Z4cAMRTransferPlan(constraint_tolerance=1.0e-6)
    transition = NumericalRelativityAMRTopologyTransition(
        source_epoch,
        compiled,
        target_epoch,
        transfer_plans=(transfer,),
    )
    caller_candidate = NumericalRelativityAMRState(
        target_epoch, (jnp.ones_like(source_values),)
    )
    with pytest.raises(TypeError):
        transition.apply(predecessor, caller_candidate)
    committed = transition.apply(predecessor)
    assert bool(committed.committed)
    assert not jnp.array_equal(committed.candidate.fields[0], caller_candidate.fields[0])
    assert committed.compile_status == "success"
    assert committed.accepted_state.epoch.topology_id == target_epoch.topology_id
    assert committed.accepted_state.fields[0].shape == predecessor.fields[0].shape

    small, initial_small, overflow, fd_small = _block_setup(
        fine_capacity=1, overflow=True
    )
    source_state, overflow_source = _epoch("z4c", small, initial_small, fd_small)
    small_capacity = sum(level.maximum_blocks for level in small.levels)
    predecessor_overflow = NumericalRelativityAMRState(
        overflow_source, (jnp.zeros((small_capacity, 25, 2, 2, 2)),)
    )
    rejected = NumericalRelativityAMRTopologyTransition(overflow_source, overflow).apply(
        predecessor_overflow
    )
    assert overflow.topology.epoch.epoch_id == source_state.topology.epoch.epoch_id
    assert bool(rejected.overflow)
    assert not bool(rejected.committed)
    assert rejected.compile_status == "capacity_exceeded"
    assert rejected.accepted_state is predecessor_overflow


def _checkpoint_repository(tmp_path):
    profile = HPCFilesystemProfile(
        "posix.nr-restart",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    policy = POSIXRepositoryPolicy(
        profile,
        maximum_chunk_bytes=4096,
        maximum_metadata_bytes=64 * 1024,
    )
    return POSIXArtifactRepository(tmp_path / "repository", policy)


def test_checkpoint_binds_topology_and_separates_exact_from_tolerant_restart(tmp_path):
    z4c = flat_z4c_state((2, 2, 2), grid_id="grid")
    values = z4c.values
    runtime = Z4cRuntimeState(
        z4c,
        jnp.asarray(0.5),
        jnp.asarray(4, dtype=jnp.int32),
        "runtime",
    )
    state = NumericalRelativityRestartState.from_z4c(
        runtime,
        topology_id="topology-a",
        topology_epoch=2,
    )
    exact_policy = NumericalRelativityRestartPolicy("exact")
    plan = NumericalRelativityCheckpointPlan(
        "z4c",
        "runtime",
        "grid",
        "topology-a",
        analysis_plan_id="analysis",
        numeric_revision_id="revision",
        execution_plan_id="execution",
        topology_epoch=2,
        state_template=state,
        restart=exact_policy,
    )
    path = tmp_path / "nr.phxcheckpoint"
    written = write_numerical_relativity_checkpoint(path, plan, state)
    restored = read_numerical_relativity_checkpoint(path, plan, state)
    exact = evaluate_numerical_relativity_restart(state, restored.state, exact_policy)
    assert written.content_id == restored.content_id
    assert bool(exact.exact)
    assert bool(exact.within_tolerance)
    assert bool(exact.qualified)
    assert not bool(exact.derivative_valid)

    perturbed_runtime = Z4cRuntimeState(
        z4c.with_values(values.at[0, 0, 0, 0].add(1.0e-6)),
        runtime.time,
        runtime.step_index,
        runtime.runtime_id,
    )
    perturbed = NumericalRelativityRestartState.from_z4c(
        perturbed_runtime,
        topology_id="topology-a",
        topology_epoch=2,
    )
    tolerant = evaluate_numerical_relativity_restart(
        state,
        perturbed,
        NumericalRelativityRestartPolicy(
            "tolerance", absolute_tolerance=2.0e-6, relative_tolerance=0.0
        ),
    )
    assert not bool(tolerant.exact)
    assert bool(tolerant.within_tolerance)
    assert bool(tolerant.qualified)
    shifted_runtime = Z4cRuntimeState(
        z4c,
        runtime.time + 1.0e-7,
        runtime.step_index,
        runtime.runtime_id,
    )
    shifted = NumericalRelativityRestartState.from_z4c(
        shifted_runtime,
        topology_id="topology-a",
        topology_epoch=2,
    )
    shifted_evidence = evaluate_numerical_relativity_restart(
        state,
        shifted,
        NumericalRelativityRestartPolicy("tolerance", absolute_tolerance=2.0e-6),
    )
    assert not bool(shifted_evidence.within_tolerance)
    assert not bool(shifted_evidence.qualified)

    incompatible_state = NumericalRelativityRestartState.from_z4c(
        runtime,
        topology_id="topology-b",
        topology_epoch=2,
    )
    mismatch = evaluate_numerical_relativity_restart(
        state, incompatible_state, exact_policy
    )
    assert not bool(mismatch.exact)
    assert not bool(mismatch.topology_matches)
    assert not bool(mismatch.qualified)
    assert not bool(mismatch.restart_admitted)

    incompatible_plan = NumericalRelativityCheckpointPlan(
        "z4c",
        "runtime",
        "grid",
        "topology-b",
        analysis_plan_id="analysis",
        numeric_revision_id="revision",
        execution_plan_id="execution",
        topology_epoch=2,
        state_template=incompatible_state,
    )
    with pytest.raises(ValueError, match="plan_id"):
        read_numerical_relativity_checkpoint(path, incompatible_plan, incompatible_state)


def test_distributed_restart_requires_committed_rank_and_reconstructs_typed_state(
    tmp_path,
):
    z4c = flat_z4c_state((2, 2, 2), grid_id="grid")
    template_runtime = Z4cRuntimeState(
        z4c,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        "runtime",
    )
    template = NumericalRelativityRestartState.from_z4c(
        template_runtime,
        topology_id="topology",
        topology_epoch=3,
    )
    stored_z4c = z4c.with_values(z4c.values.at[19].set(0.1))
    runtime = Z4cRuntimeState(
        stored_z4c,
        jnp.asarray(0.5),
        jnp.asarray(4, dtype=jnp.int32),
        "runtime",
    )
    state = NumericalRelativityRestartState.from_z4c(
        runtime,
        topology_id="topology",
        topology_epoch=3,
    )
    plan = NumericalRelativityCheckpointPlan(
        "z4c",
        "runtime",
        "grid",
        "topology",
        analysis_plan_id="analysis",
        numeric_revision_id="revision",
        execution_plan_id="execution",
        topology_epoch=3,
        state_template=template,
    )
    repository = _checkpoint_repository(tmp_path)
    runtime_args = {"damping": jnp.asarray(0.2)}
    publication = publish_distributed_numerical_relativity_checkpoint(
        repository,
        plan,
        state,
        writer_id="writer",
        runtime_args=runtime_args,
    )
    forged = ProcessCheckpointPublication(
        publication.process_index,
        None,
        publication.shards,
    )
    with pytest.raises(ValueError, match="unauthenticated"):
        assemble_distributed_numerical_relativity_checkpoint(
            repository,
            plan,
            (forged,),
            expected_process_count=1,
        )
    manifest = assemble_distributed_numerical_relativity_checkpoint(
        repository,
        plan,
        (publication,),
        expected_process_count=1,
    )
    restarted = restore_distributed_numerical_relativity_checkpoint(
        repository,
        manifest,
        plan,
        template,
        runtime_args_template={"damping": jnp.asarray(0.0)},
    )
    assert bool(restarted.evidence.exact)
    assert bool(restarted.evidence.qualified)
    assert restarted.checkpoint.state.state_id == state.state_id
    assert restarted.checkpoint.state.state_id != template.state_id
    restored_runtime = restarted.checkpoint.state.field("z4c_runtime_state")
    assert isinstance(restored_runtime, Z4cRuntimeState)
    np.testing.assert_array_equal(restored_runtime.state.values, stored_z4c.values)
    np.testing.assert_array_equal(
        restarted.checkpoint.runtime_args["damping"],
        runtime_args["damping"],
    )

    substituted_runtime = Z4cRuntimeState(
        flat_z4c_state((2, 2, 2), grid_id="other-grid"),
        runtime.time,
        runtime.step_index,
        runtime.runtime_id,
    )
    substituted = NumericalRelativityRestartState(
        "z4c",
        "runtime",
        "grid",
        "topology",
        3,
        runtime.time,
        runtime.step_index,
        (substituted_runtime,),
    )
    with pytest.raises(ValueError, match="PyTree|grid"):
        plan.validate_state(substituted)


def test_grmhd_coupled_restart_retains_ct_budgets_and_failure_counters(tmp_path):
    bridge = _bridge(2)
    constrained_transport = GRMHDConstrainedTransportPlan(
        bridge,
        gauge=GRMHDVectorPotentialGauge("weyl"),
        divergence_tolerance=1.0e-6,
        compatibility_tolerance=1.0e-6,
    )
    ct_state = constrained_transport.initialize(
        vector_potential=jnp.zeros((constrained_transport.vector_potential_size,))
    )
    grmhd = GRMHDState(
        jnp.ones(constrained_transport.cell_shape + (5,)),
        ct_state,
        jnp.asarray(0.5),
        jnp.asarray(0.1),
        jnp.asarray(4, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    grrmhd = GRRMHDState(
        grmhd.material_state,
        grmhd.constrained_transport,
        jnp.ones(constrained_transport.cell_shape + (4,)),
        grmhd.time,
        grmhd.step_size,
        grmhd.accepted_step,
        grmhd.status,
    )
    grrmhd_restart = NumericalRelativityRestartState.from_grrmhd(
        grrmhd,
        runtime_id="grrmhd-runtime",
        geometry_id="grid",
        topology_id="topology",
        topology_epoch=0,
    )
    grrmhd_plan = NumericalRelativityCheckpointPlan(
        "grrmhd",
        "grrmhd-runtime",
        "grid",
        "topology",
        analysis_plan_id="analysis",
        numeric_revision_id="revision",
        execution_plan_id="execution",
        topology_epoch=0,
        state_template=grrmhd_restart,
        constrained_transport=constrained_transport,
    )
    grrmhd_path = tmp_path / "grrmhd.phxcheckpoint"
    write_numerical_relativity_checkpoint(grrmhd_path, grrmhd_plan, grrmhd_restart)
    restored_grrmhd = read_numerical_relativity_checkpoint(
        grrmhd_path, grrmhd_plan, grrmhd_restart
    )
    np.testing.assert_array_equal(
        restored_grrmhd.state.field("radiation"), grrmhd.radiation_state
    )
    coupled = CoupledEvolutionState(
        flat_z4c_state((2, 2, 2), grid_id="grid"),
        grmhd,
        CoupledBudget.zeros(dtype=jnp.float32),
        jnp.asarray(0.5),
        jnp.asarray(5, dtype=jnp.int32),
        jnp.asarray(4, dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
        topology_id="topology",
        runtime_id="coupled-runtime",
    )
    restart = NumericalRelativityRestartState.from_coupled(
        "z4c-grmhd",
        coupled,
        geometry_id="grid",
        topology_epoch=0,
    )
    plan = NumericalRelativityCheckpointPlan(
        "z4c-grmhd",
        "coupled-runtime",
        "grid",
        "topology",
        analysis_plan_id="analysis",
        numeric_revision_id="revision",
        execution_plan_id="execution",
        topology_epoch=0,
        state_template=restart,
        constrained_transport=constrained_transport,
    )
    path = tmp_path / "coupled.phxcheckpoint"
    write_numerical_relativity_checkpoint(path, plan, restart)
    restored = read_numerical_relativity_checkpoint(path, plan, restart)
    evidence = evaluate_numerical_relativity_restart(
        restart, restored.state, plan.restart
    )
    assert bool(evidence.exact)
    assert bool(evidence.qualified)
    assert int(restored.state.field("rejected_steps")) == 2
    assert int(restored.state.field("consecutive_failures")) == 1
    assert int(restored.state.field("next_step_id")) == 5
    assert int(restored.state.field("coupled_budget").floor_cell_count) == 0

    inconsistent_ct = GRMHDCTState(
        ct_state.magnetic_flux.at[0].set(1.0),
        ct_state.vector_potential,
        ct_state.gauge_scalar,
    )
    inconsistent_grmhd = GRMHDState(
        grmhd.material_state,
        inconsistent_ct,
        grmhd.time,
        grmhd.step_size,
        grmhd.accepted_step,
        grmhd.status,
    )
    inconsistent_coupled = CoupledEvolutionState(
        coupled.z4c,
        inconsistent_grmhd,
        coupled.budget,
        coupled.time,
        coupled.next_step_id,
        coupled.accepted_steps,
        coupled.rejected_steps,
        coupled.consecutive_failures,
        coupled.terminal,
        topology_id=coupled.topology_id,
        runtime_id=coupled.runtime_id,
    )
    with pytest.raises(ValueError, match="inconsistent with dA"):
        write_numerical_relativity_checkpoint(
            tmp_path / "inconsistent.phxcheckpoint",
            plan,
            NumericalRelativityRestartState.from_coupled(
                "z4c-grmhd",
                inconsistent_coupled,
                geometry_id="grid",
                topology_epoch=0,
            ),
        )
