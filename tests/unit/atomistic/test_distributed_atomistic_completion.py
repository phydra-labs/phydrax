import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._distributed import (
    certify_distributed_polarization,
    certify_distributed_reciprocal,
    checkpoint_distributed_atomistic,
    commit_distributed_migration,
    distributed_domain_evidence,
    distributed_particle_mesh_electrostatics,
    distributed_thermodynamic_reduction,
    DistributedAtomisticCheckpoint,
    DistributedAtomisticPlan,
    DistributedCollectiveOperations,
    DistributedOutputMask,
    DistributedPMEPlan,
    DistributedPolarizationPlan,
    DistributedReductionPolicy,
    evaluate_distributed_atomistic,
    exchange_distributed_halos,
    propose_distributed_migration,
    restore_distributed_atomistic_checkpoint,
    reverse_distributed_halo_force_return,
    reverse_halo_force_return,
)
from phydrax.atomistic._potential_program import AtomisticPotentialEvaluation


def _system():
    return phx.atomistic.AtomisticSystemPlan(
        [101, 102, 103, 104],
        [1, 1, 1, 1],
        [1.0, 2.0, 3.0, 4.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=[0, 0, 0, 0],
    ).prepare()


def _positions():
    return jnp.asarray(
        [
            [3.75, 1.0, 1.0],
            [0.25, 1.0, 1.0],
            [2.25, 1.0, 1.0],
            [1.75, 1.0, 1.0],
        ]
    )


def _plan(
    *,
    halo_capacity=2,
    migration_capacity=4,
    partition_capacity=2,
    output_mask=None,
    pme=None,
    polarization=None,
    execution_mode="local-reference",
    thermostat_capacity=0,
    barostat_capacity=0,
    bias_capacity=0,
):
    box = phx.discretization.ParticleBox(
        [0.0, 0.0, 0.0],
        [4.0, 4.0, 4.0],
        periodic_axes=(False, False, False),
    )
    decomposition = phx.discretization.ParticleDomainDecompositionPlan(2, 0.5, box)
    return DistributedAtomisticPlan(
        _system(),
        decomposition,
        partition_capacity=partition_capacity,
        halo_capacity=halo_capacity,
        migration_capacity=migration_capacity,
        output_mask=output_mask,
        pme=pme,
        polarization=polarization,
        execution_mode=execution_mode,
        thermostat_capacity=thermostat_capacity,
        barostat_capacity=barostat_capacity,
        bias_capacity=bias_capacity,
    )


def _evaluation(atom_energy, forces, virial, name):
    atom = jnp.asarray(atom_energy)
    return AtomisticPotentialEvaluation(
        jnp.sum(atom),
        jnp.asarray([jnp.sum(atom)]),
        atom,
        jnp.asarray(forces),
        jnp.asarray(virial),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(False),
        name,
    )


def test_stable_ownership_permutation_block_bounds_and_compiled_shape():
    runtime = _plan().prepare_runtime()
    positions = _positions()
    state = runtime.initialize(positions)
    np.testing.assert_array_equal(state.decomposition.owner, [1, 0, 1, 0])
    np.testing.assert_array_equal(state.decomposition.permutation, [1, 3, 0, 2])
    np.testing.assert_array_equal(state.decomposition.inverse_permutation, [2, 0, 3, 1])
    np.testing.assert_array_equal(state.decomposition.block_bounds, [0, 2, 4])
    np.testing.assert_array_equal(state.decomposition.owned_indices, [[1, 3], [0, 2]])
    assert bool(state.successful)

    compiled = jax.jit(
        lambda coordinate: runtime.initialize(coordinate).decomposition.local_indices
    )(positions)
    assert compiled.shape == (2, 6)


def test_halo_routes_are_padded_and_capacity_overflow_fails_closed():
    state = _plan(halo_capacity=2).prepare_runtime().initialize(_positions())
    decomposition = state.decomposition
    assert decomposition.halo_send_indices.shape == (2, 2, 2)
    assert decomposition.halo_receive_indices.shape == (2, 2, 2)
    np.testing.assert_array_equal(decomposition.route_counts, [[0, 1], [1, 0]])
    assert int(decomposition.halo_send_indices[0, 1, 0]) == 3
    assert int(decomposition.halo_send_indices[1, 0, 0]) == 2

    overflow = _plan(halo_capacity=0).prepare_runtime().initialize(_positions())
    assert overflow.decomposition.halo_send_indices.shape == (2, 2, 0)
    assert bool(overflow.decomposition.halo_overflow)
    assert not bool(overflow.status.halo_capacity_ok)
    assert not bool(overflow.successful)


def test_migration_candidate_commits_or_rolls_back_atomically():
    rollback_plan = _plan(migration_capacity=0, partition_capacity=3)
    rollback_state = rollback_plan.prepare_runtime().initialize(_positions())
    proposed = _positions().at[1, 0].set(2.75)
    candidate = propose_distributed_migration(rollback_plan, rollback_state, proposed)
    assert int(candidate.migration_count) == 1
    assert bool(candidate.overflow)
    committed = commit_distributed_migration(rollback_plan, rollback_state, candidate)
    np.testing.assert_array_equal(committed.positions, rollback_state.positions)
    np.testing.assert_array_equal(
        committed.decomposition.owner, rollback_state.decomposition.owner
    )
    assert int(committed.decomposition_epoch) == 0
    assert not bool(committed.successful)

    commit_plan = _plan(migration_capacity=1, partition_capacity=3)
    commit_runtime = commit_plan.prepare_runtime()
    commit_state = commit_runtime.initialize(_positions(), run_id="run-a")
    candidate = propose_distributed_migration(commit_plan, commit_state, proposed)
    committed = commit_distributed_migration(commit_plan, commit_state, candidate)
    np.testing.assert_array_equal(committed.positions, proposed)
    np.testing.assert_array_equal(committed.decomposition.owner, [1, 1, 1, 0])
    assert int(committed.decomposition_epoch) == 1
    assert bool(committed.successful)
    foreign_state = commit_runtime.initialize(
        _positions(),
        rng_key=jax.random.key(9),
        run_id="run-b",
    )
    rejected = commit_distributed_migration(commit_plan, foreign_state, candidate)
    np.testing.assert_array_equal(rejected.positions, foreign_state.positions)
    np.testing.assert_array_equal(rejected.rng_key, foreign_state.rng_key)
    assert not bool(rejected.successful)


def test_reverse_halo_force_return_is_conservative():
    state = _plan().prepare_runtime().initialize(_positions())
    receive = jnp.zeros(state.decomposition.halo_receive_indices.shape + (3,))
    receive = receive.at[0, 1, 0].set(jnp.asarray([1.0, 2.0, 3.0]))
    receive = receive.at[1, 0, 0].set(jnp.asarray([-0.5, 1.5, 2.0]))
    returned = reverse_halo_force_return(state.decomposition, receive)
    np.testing.assert_allclose(returned[2], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(returned[3], [-0.5, 1.5, 2.0])
    np.testing.assert_allclose(
        jnp.sum(returned, axis=0), jnp.sum(receive, axis=(0, 1, 2))
    )


def test_reverse_force_policy_orders_repeated_owner_contributions():
    box = phx.discretization.ParticleBox(
        [0.0, 0.0, 0.0],
        [6.0, 4.0, 4.0],
        periodic_axes=(False, False, False),
    )
    plan = DistributedAtomisticPlan(
        _system(),
        phx.discretization.ParticleDomainDecompositionPlan(3, 1.1, box),
        partition_capacity=4,
        halo_capacity=4,
        reduction=DistributedReductionPolicy("compensated"),
    )
    positions = jnp.asarray(
        [
            [3.0, 1.0, 1.0],
            [0.5, 1.0, 1.0],
            [4.5, 1.0, 1.0],
            [5.5, 1.0, 1.0],
        ]
    )
    state = plan.prepare_runtime().initialize(positions)
    assert int(state.decomposition.halo_receive_indices[0, 1, 0]) == 0
    assert int(state.decomposition.halo_receive_indices[2, 1, 0]) == 0
    received = jnp.zeros(state.decomposition.halo_receive_indices.shape + (3,))
    received = received.at[0, 1, 0].set(jnp.asarray([1.0e10, 1.0, 0.0]))
    received = received.at[2, 1, 0].set(jnp.asarray([-1.0e10, 2.0, 0.0]))
    returned = reverse_halo_force_return(
        state.decomposition,
        received,
        policy=plan.reduction,
    )
    np.testing.assert_allclose(returned[0], [0.0, 3.0, 0.0])


def test_local_shard_direct_sparse_reciprocal_energy_force_virial_parity():
    plan = _plan(
        pme=DistributedPMEPlan((8, 6, 4)),
        output_mask=DistributedOutputMask(atom_energy=True),
    )
    runtime = plan.prepare_runtime()
    state = runtime.initialize(_positions())
    assert runtime.pme_runtime().mesh_bounds.shape == (3,)
    direct = _evaluation(
        [1.0, 2.0, 3.0, 4.0],
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]],
        jnp.eye(3),
        "direct",
    )
    sparse = _evaluation(
        [0.1, 0.2, 0.3, 0.4],
        jnp.full((4, 3), 0.25),
        2.0 * jnp.eye(3),
        "sparse",
    )
    reciprocal = _evaluation(
        [-0.5, -0.5, -0.5, -0.5],
        jnp.full((4, 3), -0.125),
        -0.5 * jnp.eye(3),
        "reciprocal",
    )
    reciprocal_evidence = certify_distributed_reciprocal(
        runtime.pme_runtime(), state, reciprocal
    )
    result = evaluate_distributed_atomistic(
        runtime,
        state,
        direct,
        sparse_correction=sparse,
        reciprocal=reciprocal_evidence,
    )
    np.testing.assert_allclose(
        result.energy, direct.energy + sparse.energy + reciprocal.energy
    )
    pme_force, pme_energy = distributed_particle_mesh_electrostatics(
        runtime, state, reciprocal_evidence
    )
    np.testing.assert_allclose(pme_force, reciprocal.forces)
    np.testing.assert_allclose(pme_energy, reciprocal.energy)
    moved_state = runtime.initialize(_positions().at[0, 1].set(1.1))
    stale = evaluate_distributed_atomistic(
        runtime,
        moved_state,
        direct,
        sparse_correction=sparse,
        reciprocal=reciprocal_evidence,
    )
    assert not bool(stale.status.reciprocal_converged)
    assert not bool(stale.successful)
    np.testing.assert_allclose(
        result.forces, direct.forces + sparse.forces + reciprocal.forces
    )
    np.testing.assert_allclose(
        result.virial, direct.virial + sparse.virial + reciprocal.virial
    )
    np.testing.assert_allclose(
        result.atom_energy,
        direct.atom_energy + sparse.atom_energy + reciprocal.atom_energy,
    )
    np.testing.assert_allclose(jnp.sum(result.partition_energy), result.energy)
    np.testing.assert_allclose(result.phases.phase_energy, [10.0, 1.0, -2.0, 9.0])
    assert bool(result.successful)


def test_deterministic_reduction_and_load_evidence_are_reproducible():
    energy = jnp.asarray([1.0e10, 1.0, -1.0e10, 3.0], dtype=jnp.float32)
    momentum = jnp.arange(12, dtype=jnp.float32).reshape((4, 3))
    policy = DistributedReductionPolicy("deterministic")
    first = distributed_thermodynamic_reduction(energy, momentum, policy=policy)
    second = distributed_thermodynamic_reduction(energy, momentum, policy=policy)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])

    plan = _plan()
    state = plan.prepare_runtime().initialize(_positions())
    evidence = distributed_domain_evidence(
        plan,
        state,
        pair_work=jnp.asarray([4.0, 2.0]),
        iterative_work=jnp.asarray([1.0, 3.0]),
    )
    np.testing.assert_array_equal(evidence.owned_particles, [2, 2])
    np.testing.assert_array_equal(evidence.halo_particles, [1, 1])
    np.testing.assert_allclose(evidence.weighted_work, [8.0, 8.0])
    assert float(evidence.imbalance) == 1.0
    assert bool(evidence.successful)


def test_checkpoint_identity_covers_all_continuation_state():
    plan = _plan(
        thermostat_capacity=2,
        barostat_capacity=2,
        bias_capacity=2,
        polarization=DistributedPolarizationPlan(maximum_iterations=20, tolerance=1.0e-5),
    )
    runtime = plan.prepare_runtime()
    keyword = dict(
        momenta=jnp.arange(12, dtype=jnp.float32).reshape((4, 3)),
        thermostat_state=jnp.asarray([1.0, 2.0]),
        barostat_state=jnp.asarray([3.0, 4.0]),
        polarization_warm_start=jnp.full((4, 3), 0.25),
        bias_state=jnp.asarray([5.0, 6.0]),
        rng_key=jnp.asarray([7, 8], dtype=jnp.uint32),
        step_index=11,
        decomposition_epoch=3,
        run_id="run-a",
        replica_id="replica-b",
        epoch_id="epoch-c",
    )
    state = runtime.initialize(_positions(), **keyword)
    first = checkpoint_distributed_atomistic(runtime, state)
    second = checkpoint_distributed_atomistic(runtime, state)
    assert first.identity.checkpoint_id == second.identity.checkpoint_id
    assert first.units.unit_system_id == runtime.plan.system.plan.units.unit_system_id
    restored = restore_distributed_atomistic_checkpoint(runtime, first)
    np.testing.assert_array_equal(restored.rng_key, [7, 8])
    np.testing.assert_array_equal(restored.thermostat_state, [1.0, 2.0])
    changed = eqx.tree_at(
        lambda value: value.rng_key,
        state,
        jnp.asarray([7, 9], dtype=jnp.uint32),
    )
    assert (
        checkpoint_distributed_atomistic(runtime, changed).identity.checkpoint_id
        != first.identity.checkpoint_id
    )
    changed_halo = eqx.tree_at(
        lambda value: value.halos.local_mask,
        state,
        jnp.zeros_like(state.halos.local_mask),
    )
    changed_status = eqx.tree_at(
        lambda value: value.status.successful,
        state,
        jnp.asarray(False),
    )
    assert (
        checkpoint_distributed_atomistic(runtime, changed_halo).identity.checkpoint_id
        != first.identity.checkpoint_id
    )
    assert (
        checkpoint_distributed_atomistic(runtime, changed_status).identity.checkpoint_id
        != first.identity.checkpoint_id
    )
    prepared_polarization = runtime.polarization_runtime()
    evidence = certify_distributed_polarization(
        prepared_polarization,
        state,
        state.polarization_warm_start,
        jnp.asarray(1.0e-6),
        jnp.asarray(5),
    )
    assert bool(evidence.successful)
    polarization_stale_state = runtime.initialize(
        _positions().at[0, 1].set(1.1),
        step_index=11,
        decomposition_epoch=3,
        run_id="run-a",
        replica_id="replica-b",
        epoch_id="epoch-c",
    )
    polarization_direct = _evaluation(
        [1.0, 2.0, 3.0, 4.0],
        jnp.ones((4, 3)),
        jnp.eye(3),
        "direct",
    )
    polarization_stale_result = evaluate_distributed_atomistic(
        runtime,
        polarization_stale_state,
        polarization_direct,
        polarization=evidence,
    )
    assert not bool(polarization_stale_result.status.polarization_converged)
    assert not bool(polarization_stale_result.successful)
    negative_residual = certify_distributed_polarization(
        prepared_polarization,
        state,
        state.polarization_warm_start,
        jnp.asarray(-1.0e-6),
        jnp.asarray(5),
    )
    negative_iterations = certify_distributed_polarization(
        prepared_polarization,
        state,
        state.polarization_warm_start,
        jnp.asarray(1.0e-6),
        jnp.asarray(-1),
    )
    assert not bool(negative_residual.successful)
    assert not bool(negative_iterations.successful)
    with pytest.raises(TypeError, match="integral dtype"):
        certify_distributed_polarization(
            prepared_polarization,
            state,
            state.polarization_warm_start,
            jnp.asarray(1.0e-6),
            jnp.asarray(2.5),
        )
    forged = DistributedAtomisticCheckpoint(changed_halo, first.units, first.identity)
    with pytest.raises(ValueError, match="content identity"):
        restore_distributed_atomistic_checkpoint(runtime, forged)


def test_static_output_mask_preserves_shapes_and_zeros_unrequested_outputs():
    mask = DistributedOutputMask(
        energy=False,
        forces=True,
        virial=False,
        atom_energy=False,
        partition_energy=False,
    )
    runtime = _plan(output_mask=mask).prepare_runtime()
    state = runtime.initialize(_positions())
    direct = _evaluation(
        [1.0, 2.0, 3.0, 4.0],
        jnp.arange(12, dtype=jnp.float32).reshape((4, 3)),
        jnp.eye(3),
        "direct",
    )
    result = evaluate_distributed_atomistic(runtime, state, direct)
    assert result.energy.shape == () and float(result.energy) == 0.0
    assert result.forces.shape == (4, 3)
    np.testing.assert_array_equal(result.forces, direct.forces)
    np.testing.assert_array_equal(result.virial, jnp.zeros((3, 3)))
    np.testing.assert_array_equal(result.atom_energy, jnp.zeros((4,)))
    np.testing.assert_array_equal(result.partition_energy, jnp.zeros((2,)))
    np.testing.assert_array_equal(result.available, [False, True, False, False, False])


def test_collective_execution_requires_and_reduces_rank_local_contributions_once():
    plan = _plan(execution_mode="collective")
    with pytest.raises(ValueError, match="explicit JAX collectives"):
        plan.prepare_runtime()

    rank_one_force = jnp.zeros((4, 3)).at[jnp.asarray([0, 2])].set(1.0)
    rank_one_atom = jnp.asarray([1.0, 0.0, 3.0, 0.0])

    def global_sum(value):
        if jnp.issubdtype(value.dtype, jnp.integer):
            return value
        if value.shape == ():
            return value + 4.0
        if value.shape == (4, 3):
            return value + rank_one_force
        if value.shape == (3, 3):
            return value
        if value.shape == (4,):
            return value + rank_one_atom
        if value.shape == (2,):
            return value + jnp.asarray([0.0, 4.0])
        if value.shape == (3,):
            return value + jnp.asarray([4.0, 0.0, 0.0])
        raise AssertionError(f"Unexpected collective shape {value.shape}.")

    def exchange(send, mask):
        return jnp.swapaxes(jnp.where(mask[..., None], send, 0), 0, 1)

    def reverse_exchange(receive, mask):
        returned = jnp.zeros_like(receive)
        return returned.at[0, 1, 0].set(jnp.asarray([0.5, 1.0, 1.5]))

    operations = DistributedCollectiveOperations(
        exchange,
        reverse_exchange,
        global_sum,
        partition_index=0,
        collective_id="two-shard-test",
    )
    runtime = plan.prepare_runtime(operations)
    state = runtime.initialize(_positions())
    with pytest.raises(ValueError, match="Collective migration"):
        propose_distributed_migration(plan, state, _positions())
    received = exchange_distributed_halos(runtime, state, state.positions)
    assert received.shape == (2, 2, 2, 3)
    direct = _evaluation(
        [1.0, 2.0, 3.0, 4.0],
        jnp.ones((4, 3)),
        jnp.eye(3),
        "direct",
    )
    result = evaluate_distributed_atomistic(runtime, state, direct)
    np.testing.assert_allclose(result.energy, direct.energy)
    np.testing.assert_allclose(result.forces, direct.forces)
    np.testing.assert_allclose(result.virial, direct.virial)
    np.testing.assert_allclose(result.partition_energy, [6.0, 4.0])
    np.testing.assert_allclose(result.phases.phase_energy, [10.0, 0.0, 0.0, 10.0])
    assert bool(result.status.collective_supported)
    assert bool(result.successful)

    remote_force = reverse_distributed_halo_force_return(
        runtime,
        state,
        jnp.zeros(state.decomposition.halo_receive_indices.shape + (3,)),
    )
    np.testing.assert_allclose(remote_force[3], [0.5, 1.0, 1.5])

    def failed_global_sum(value):
        reduced = global_sum(value)
        return (
            reduced + jnp.ones_like(reduced)
            if jnp.issubdtype(value.dtype, jnp.integer)
            else reduced
        )

    failed_operations = DistributedCollectiveOperations(
        exchange,
        reverse_exchange,
        failed_global_sum,
        partition_index=0,
        collective_id="two-shard-failure-test",
    )
    failed_runtime = plan.prepare_runtime(failed_operations)
    failed_state = failed_runtime.initialize(_positions())
    failed_result = evaluate_distributed_atomistic(failed_runtime, failed_state, direct)
    assert not bool(failed_result.successful)


def test_plans_reject_ambiguous_or_impossible_static_contracts():
    with pytest.raises(TypeError, match="booleans"):
        DistributedOutputMask(energy=1)
    with pytest.raises(ValueError, match="partition_capacity"):
        _plan(partition_capacity=0)
    with pytest.raises(ValueError, match="grid_shape"):
        DistributedPMEPlan((8, 0, 8))
    with pytest.raises(ValueError, match="interpolation_order"):
        DistributedPMEPlan((8, 8, 8), interpolation_order=1)
    with pytest.raises(ValueError, match="cover every partition"):
        _plan(pme=DistributedPMEPlan((1, 8, 8))).prepare_runtime()
    with pytest.raises(TypeError, match="tolerance"):
        DistributedPolarizationPlan(tolerance=True)


def test_documented_distributed_surface_is_public():
    names = (
        "DistributedCollectiveOperations",
        "DistributedOutputMask",
        "DistributedPMEPlan",
        "DistributedPolarizationPlan",
        "DistributedReciprocalEvidence",
        "DistributedReductionPolicy",
        "certify_distributed_polarization",
        "certify_distributed_reciprocal",
        "commit_distributed_migration",
        "evaluate_distributed_atomistic",
        "exchange_distributed_halos",
        "propose_distributed_migration",
        "reverse_distributed_halo_force_return",
    )
    assert all(callable(getattr(phx.atomistic, name)) for name in names)
