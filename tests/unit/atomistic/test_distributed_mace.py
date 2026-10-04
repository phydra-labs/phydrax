"""Owner-local multilayer MACE execution against single-device global MACE.

Lane-reference ownership runs the owner regions as named vmap lanes on one
device with the same collectives as device-collective ownership; the
device-collective case runs only in a dedicated process with
``XLA_FLAGS=--xla_force_host_platform_device_count=2``. Forced CPU devices
prove functional collective parity only, not accelerator or multi-host
qualification. The oracle enumerates every periodic image route on the host
by brute force and evaluates the same native MACE layers on the global graph
(``R = C = N``), differentiated with ordinary JAX.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._execution_runtime import ExecutionRuntime
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import AtomisticScaleContract
from phydrax.atomistic._distributed import (
    _checkpoint_identity,
    checkpoint_owner_local_atomistic,
    evaluate_owner_local_atomistic,
    owner_local_loss_gradient,
    OwnerLocalAtomisticCheckpoint,
    OwnerLocalAtomisticPlan,
    OwnerLocalAtomisticState,
    prepare_owner_local_atomistic,
    rebuild_owner_local_atomistic,
    restore_owner_local_atomistic,
)
from phydrax.discretization._periodic_cell import PeriodicCell, PeriodicImageStencil
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.discretization.spatial._distributed_relations import (
    DistributedOwnershipPlan,
)
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.sparse import EdgeRelation
from phydrax.sparse._streamed import StreamedRelationPlan
from phydrax.units import ANGSTROM, ELECTRONVOLT


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
CUTOFF = 2.4
SKIN = 0.3
SPECIES = (1, 8)
# A triclinic cell whose shortest lattice translation is below the cutoff, so
# atoms see several images of the same source and of themselves.
CELL = np.array([[2.2, 0.0, 0.0], [0.6, 2.5, 0.0], [0.3, 0.4, 2.8]])


def _model(layers: int, key: int = 0) -> MACEPotential:
    kinds = ("real-agnostic",) + ("real-agnostic-residual",) * (layers - 1)
    architecture = MACEArchitecture(
        species=SPECIES,
        cutoff=CUTOFF,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=kinds,
        correlations=(2,) * layers,
        radial_widths=(8,),
        readout_width=4,
        average_neighbor_count=8.0,
    )
    return MACEPotential(
        SCALE,
        architecture,
        atomic_energies=np.array([-0.3, -1.1]),
        key=jax.random.key(key),
    )


def _streaming(model: MACEPotential) -> StreamedRelationPlan:
    return StreamedRelationPlan(
        receiver_tile=4,
        edge_tile=32,
        channel_capacity=model.configuration.message_payload_width(),
    )


def _ownership(owners: int, *, devices: bool = False) -> DistributedOwnershipPlan:
    address = MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8, periodic_axes=(True, True, True)
    )
    runtime = ExecutionRuntime.current()
    count = len(jax.devices())
    if devices:
        if count < owners or count % owners:
            pytest.skip(
                "Run with XLA_FLAGS=--xla_force_host_platform_device_count=2 to "
                "exercise device-collective owners."
            )
        return DistributedOwnershipPlan(
            address, runtime.child_groups(count // owners)[0], 8
        )
    group = runtime.child_groups(count)[0]
    return DistributedOwnershipPlan(address, group, 8, owner_lanes=owners)


def _plan(
    model: MACEPotential,
    *,
    owners: int = 2,
    devices: bool = False,
    cell: np.ndarray = CELL,
    edge_capacity: int = 512,
    migration_capacity: int = 8,
    message_capacity_bytes: int = 1 << 20,
) -> OwnerLocalAtomisticPlan:
    periodic = PeriodicCell(cell)
    return OwnerLocalAtomisticPlan(
        _ownership(owners, devices=devices),
        FractionalOwnerPartition(periodic, (owners, 1, 1)),
        periodic.image_stencil(CUTOFF + SKIN, maximum_image_count=4096),
        streaming=_streaming(model),
        cutoff=CUTOFF,
        skin=SKIN,
        alias_capacity=512,
        edge_capacity=edge_capacity,
        halo_capacity=8,
        migration_capacity=migration_capacity,
        message_capacity_bytes=message_capacity_bytes,
    )


def _structure(count: int = 6, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fractional = rng.uniform(0.0, 1.0, (count, 3))
    # Two atoms straddle the owner boundary at fractional x = 1/2.
    fractional[0] = (0.47, 0.31, 0.22)
    fractional[1] = (0.53, 0.33, 0.25)
    positions = fractional @ CELL
    # One stored coordinate starts outside the cell (an unwrapped atom).
    positions[2] += CELL[0]
    return positions, np.asarray(SPECIES, dtype=np.int32)[rng.integers(0, 2, count)]


def _image_routes(
    positions: np.ndarray, cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brute-force directed routes ``(source, receiver, n)`` with ``|d| < cutoff``."""
    count = positions.shape[0]
    reach = 3
    grid = np.stack(
        np.meshgrid(*(np.arange(-reach, reach + 1),) * 3, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    senders, receivers, shifts = [], [], []
    for receiver in range(count):
        for sender in range(count):
            displacement = positions[receiver] - positions[sender] + grid @ cell
            near = np.linalg.norm(displacement, axis=1) < CUTOFF
            if receiver == sender:
                near &= np.any(grid != 0, axis=1)
            for shift in grid[near]:
                senders.append(sender)
                receivers.append(receiver)
                shifts.append(shift)
    return (
        np.asarray(senders, dtype=np.int32),
        np.asarray(receivers, dtype=np.int32),
        np.asarray(shifts, dtype=np.int32),
    )


def _global_energy(
    model: MACEPotential,
    positions: jax.Array,
    species_ids: np.ndarray,
    strain: jax.Array,
    routes: tuple[np.ndarray, np.ndarray, np.ndarray],
    cell: np.ndarray = CELL,
) -> jax.Array:
    """Single-device global MACE energy at strain ``F = I + strain``."""
    senders, receivers, shifts = routes
    count = positions.shape[0]
    deformation = jnp.eye(3) + strain
    deformed = positions @ deformation.T
    vectors_cell = jnp.asarray(cell) @ deformation.T
    relation = _streaming(model).prepare(
        EdgeRelation(senders, receivers, source_size=count, target_size=count),
        owner_id="global-oracle",
    )
    vectors = (
        deformed[receivers] - deformed[senders] + shifts.astype(np.float64) @ vectors_cell
    )
    mask = jnp.ones((count,), dtype=jnp.bool_)
    species = model.species_indices(jnp.asarray(species_ids), mask)
    energies, successful = model.node_energies(species, mask, relation, vectors)
    return jnp.where(successful, jnp.sum(energies), jnp.nan)


def _prepare(
    plan: OwnerLocalAtomisticPlan,
    model: MACEPotential,
    positions: np.ndarray,
    species_ids: np.ndarray,
    **kwargs: Any,
) -> OwnerLocalAtomisticState:
    mask = jnp.ones(species_ids.shape, dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(species_ids), mask))
    return prepare_owner_local_atomistic(
        plan, model, positions, species, rng_key=jax.random.key(7), **kwargs
    )


def _assert_global_parity(plan: OwnerLocalAtomisticPlan, model: MACEPotential) -> None:
    positions, species_ids = _structure()
    state = _prepare(plan, model, positions, species_ids)
    evaluation = evaluate_owner_local_atomistic(plan, model, state)
    assert bool(evaluation.successful)
    routes = _image_routes(positions, CELL)
    zero = jnp.zeros((3, 3))
    reference = jnp.asarray(positions)
    energy = _global_energy(model, reference, species_ids, zero, routes)
    forces = -jax.grad(_global_energy, argnums=1)(
        model, reference, species_ids, zero, routes
    )
    strain = jax.grad(_global_energy, argnums=3)(
        model, reference, species_ids, zero, routes
    )
    np.testing.assert_allclose(float(evaluation.energy), float(energy), rtol=1e-11)
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)), forces, atol=1e-10
    )
    np.testing.assert_allclose(np.asarray(evaluation.strain_gradient), strain, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(evaluation.stress),
        0.5 * (strain + strain.T) / PeriodicCell(CELL).volume,
        atol=1e-10,
    )


def test_route_oracle_has_repeated_and_self_images() -> None:
    positions, _ = _structure()
    senders, receivers, _ = _image_routes(positions, CELL)
    pairs = np.stack((senders, receivers), axis=1)
    _, multiplicity = np.unique(pairs, axis=0, return_counts=True)
    assert np.max(multiplicity) > 1
    assert np.any(senders == receivers)


@pytest.mark.parametrize("layers", [2, 3], ids=["two-layer", "three-layer"])
def test_lane_owners_match_global_energy_forces_and_strain(layers: int) -> None:
    model = _model(layers)
    _assert_global_parity(_plan(model), model)


def test_device_collective_owners_match_global() -> None:
    model = _model(2)
    _assert_global_parity(_plan(model, devices=True), model)


def test_multi_hop_dependency_crosses_owners_at_successive_layers() -> None:
    """A chain 0-1-2-3 split between owners couples atom 0 to atom 3 only
    through three interactions, so a missing per-layer exchange or a doubled
    reverse return would change the force on atom 0."""
    cell = np.diag([30.0, 30.0, 30.0])
    spacing = 0.8 * CUTOFF
    base = np.array([[13.0 + spacing * index, 15.0, 15.0] for index in range(4)])
    species_ids = np.array([1, 8, 1, 8], dtype=np.int32)
    model = _model(3)
    plan = _plan(model, cell=cell)

    def owner_forces(positions: np.ndarray) -> np.ndarray:
        state = _prepare(plan, model, positions, species_ids)
        evaluation = evaluate_owner_local_atomistic(plan, model, state)
        assert bool(evaluation.successful)
        owners = np.asarray(plan.partition.owners(positions / 30.0))
        assert owners.tolist() == [0, 0, 1, 1]
        return np.asarray(state.layout.collect(evaluation.forces))

    moved = base.copy()
    moved[3, 1] += 0.2
    first, second = owner_forces(base), owner_forces(moved)
    assert np.max(np.abs(first[0] - second[0])) > 1e-8
    for positions, observed in ((base, first), (moved, second)):
        routes = _image_routes(positions, cell)
        expected = -jax.grad(_global_energy, argnums=1)(
            model, jnp.asarray(positions), species_ids, jnp.zeros((3, 3)), routes, cell
        )
        np.testing.assert_allclose(observed, expected, atol=1e-10)


def test_self_image_cell_has_cancelling_forces_and_nonzero_stress() -> None:
    model = _model(2)
    plan = _plan(model)
    positions = np.array([[0.4, 0.7, 1.1]])
    species_ids = np.array([8], dtype=np.int32)
    state = _prepare(plan, model, positions, species_ids)
    evaluation = evaluate_owner_local_atomistic(plan, model, state)
    assert bool(evaluation.successful)
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)), 0.0, atol=1e-11
    )
    routes = _image_routes(positions, CELL)
    strain = jax.grad(_global_energy, argnums=3)(
        model, jnp.asarray(positions), species_ids, jnp.zeros((3, 3)), routes
    )
    assert np.max(np.abs(strain)) > 1e-6
    np.testing.assert_allclose(np.asarray(evaluation.strain_gradient), strain, atol=1e-10)


def test_force_loss_parameter_gradient_matches_global() -> None:
    model = _model(2)
    plan = _plan(model)
    positions, species_ids = _structure()
    state = _prepare(plan, model, positions, species_ids)
    rng = np.random.default_rng(11)
    reference_forces = rng.normal(size=positions.shape)
    result = owner_local_loss_gradient(
        plan,
        model,
        state,
        -2.0,
        reference_forces,
        energy_weight=0.5,
        force_weight=2.0,
    )
    assert bool(result.successful)
    routes = _image_routes(positions, CELL)
    lanes = partition_parameters(model)

    def loss(parameters: Any) -> jax.Array:
        candidate = combine_parameters(parameters, lanes[1], lanes[2])
        reference = jnp.asarray(positions)
        zero = jnp.zeros((3, 3))
        energy = _global_energy(candidate, reference, species_ids, zero, routes)
        forces = -jax.grad(_global_energy, argnums=1)(
            candidate, reference, species_ids, zero, routes
        )
        return 0.5 * (energy + 2.0) ** 2 + 2.0 * jnp.sum((forces - reference_forces) ** 2)

    expected = jax.grad(loss)(lanes[0])
    np.testing.assert_allclose(float(result.loss), float(loss(lanes[0])), rtol=1e-11)
    for observed, oracle in zip(
        jax.tree.leaves(result.parameter_gradient),
        jax.tree.leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(oracle), atol=1e-9)


def test_stale_owner_epoch_and_expired_certificate_fail_closed() -> None:
    model = _model(2)
    plan = _plan(model)
    positions, species_ids = _structure()
    state = _prepare(plan, model, positions, species_ids)
    advanced = state.layout.migrate(
        state.layout.slot_owners, packet_capacity=plan.migration_capacity
    ).layout
    stale = eqx.tree_at(lambda value: value.layout, state, advanced)
    evaluation = evaluate_owner_local_atomistic(plan, model, stale)
    assert not bool(evaluation.status.owners_current)
    assert not bool(evaluation.successful)
    assert np.all(np.isnan(np.asarray(evaluation.forces)))

    shift = np.zeros_like(positions)
    shift[0, 1] = 0.6 * SKIN
    moved = state.with_dynamics(
        state.positions + state.layout.distribute(shift),
        state.velocities,
        step_index=1,
    )
    expired = evaluate_owner_local_atomistic(plan, model, moved)
    assert not bool(expired.status.displacement_certified)
    assert np.isnan(float(expired.energy))


def _stencil_like(
    stencil: PeriodicImageStencil, *, extents: tuple[int, ...] | None = None
) -> PeriodicImageStencil:
    return PeriodicImageStencil(
        extents=stencil.extents if extents is None else extents,
        radius=stencil.radius,
        fractional_excursion=stencil.fractional_excursion,
        axis_reach=stencil.axis_reach,
        minimum_singular_value=stencil.minimum_singular_value,
        condition_number=stencil.condition_number,
        maximum_image_count=stencil.maximum_image_count,
        cell_id=stencil.cell_id,
    )


def _replan(
    plan: OwnerLocalAtomisticPlan,
    stencil: PeriodicImageStencil,
    *,
    ownership: DistributedOwnershipPlan | None = None,
    partition: FractionalOwnerPartition | None = None,
) -> OwnerLocalAtomisticPlan:
    return OwnerLocalAtomisticPlan(
        plan.ownership if ownership is None else ownership,
        plan.partition if partition is None else partition,
        stencil,
        streaming=plan.streaming,
        cutoff=plan.cutoff,
        skin=plan.skin,
        alias_capacity=plan.alias_capacity,
        edge_capacity=plan.edge_capacity,
        halo_capacity=plan.halo_capacity,
        migration_capacity=plan.migration_capacity,
        message_capacity_bytes=plan.message_capacity_bytes,
    )


def test_plan_refuses_forged_incomplete_or_nonperiodic_image_stencils() -> None:
    # A one-atom cell needs every nonzero self image; a stencil that keeps the
    # identity but drops translations would silently remove all routes.
    model = _model(2)
    plan = _plan(model)
    stencil = plan.stencil
    rebuilt = _stencil_like(stencil)
    assert rebuilt.stencil_id == stencil.stencil_id
    np.testing.assert_array_equal(np.asarray(rebuilt.shifts), np.asarray(stencil.shifts))
    assert np.unique(np.asarray(stencil.shifts), axis=0).shape[0] == stencil.image_count
    forged = eqx.tree_at(
        lambda value: value.shifts, stencil, jnp.zeros_like(stencil.shifts)
    )
    with pytest.raises(ValueError, match="complete box"):
        _replan(plan, forged)
    with pytest.raises(ValueError, match="does not cover"):
        _replan(plan, _stencil_like(stencil, extents=(1, 1, 1)))

    open_cell = PeriodicCell(CELL, periodic_axes=(True, True, False))
    address = MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8, periodic_axes=(True, True, False)
    )
    ownership = DistributedOwnershipPlan(
        address,
        ExecutionRuntime.current().child_groups(len(jax.devices()))[0],
        8,
        owner_lanes=2,
    )
    open_stencil = open_cell.image_stencil(CUTOFF + SKIN, maximum_image_count=4096)
    partition = FractionalOwnerPartition(open_cell, (2, 1, 1))
    _replan(plan, open_stencil, ownership=ownership, partition=partition)
    translated = _stencil_like(open_stencil, extents=open_stencil.extents[:2] + (1,))
    with pytest.raises(ValueError, match="nonperiodic axis"):
        _replan(plan, translated, ownership=ownership, partition=partition)

    positions = np.array([[0.4, 0.7, 1.1]])
    state = _prepare(plan, model, positions, np.array([8], dtype=np.int32))
    senders, _, _ = _image_routes(positions, CELL)
    assert int(jnp.sum(state.topology.edge_valid)) >= senders.size > 0


def test_checkpoint_refuses_a_topology_of_another_layout_epoch() -> None:
    model = _model(2)
    plan = _plan(model)
    positions, species_ids = _structure()
    state = _prepare(plan, model, positions, species_ids)
    advanced = state.layout.migrate(
        state.layout.slot_owners, packet_capacity=plan.migration_capacity
    ).layout
    stale = eqx.tree_at(lambda value: value.layout, state, advanced)
    with pytest.raises(ValueError, match="another layout epoch"):
        checkpoint_owner_local_atomistic(plan, stale)

    # A self-consistent record whose layout epoch disagrees with its recorded
    # topology witness must not be restored as current.
    checkpoint = checkpoint_owner_local_atomistic(plan, state)
    arrays = {
        **checkpoint.arrays,
        "layout/owner_epochs": checkpoint.arrays["layout/owner_epochs"] + 1,
    }
    forged = OwnerLocalAtomisticCheckpoint(
        arrays=arrays,
        key_implementation=checkpoint.key_implementation,
        logical_count=checkpoint.logical_count,
        plan_id=checkpoint.plan_id,
        model_revision_id=checkpoint.model_revision_id,
        run_id=checkpoint.run_id,
        checkpoint_id=_checkpoint_identity(
            arrays,
            checkpoint.key_implementation,
            checkpoint.logical_count,
            checkpoint.plan_id,
            checkpoint.model_revision_id,
            checkpoint.run_id,
        ),
    )
    with pytest.raises(ValueError, match="another layout epoch"):
        restore_owner_local_atomistic(plan, model, forged)
    restored = restore_owner_local_atomistic(plan, model, checkpoint)
    assert bool(evaluate_owner_local_atomistic(plan, model, restored).successful)


def test_capacity_refusals_never_truncate() -> None:
    model = _model(2)
    positions, species_ids = _structure()
    with pytest.raises(ValueError, match="initial owner-local topology was refused"):
        _prepare(_plan(model, edge_capacity=4), model, positions, species_ids)
    plan = _plan(model, message_capacity_bytes=64)
    state = _prepare(plan, model, positions, species_ids)
    with pytest.raises(ValueError, match="message_capacity_bytes"):
        evaluate_owner_local_atomistic(plan, model, state)


def test_failed_migration_returns_the_accepted_state() -> None:
    model = _model(2)
    plan = _plan(model, migration_capacity=1)
    positions, species_ids = _structure()
    state = _prepare(plan, model, positions, species_ids)
    shift = np.zeros_like(positions)
    shift[:, 0] = 0.5 * CELL[0, 0]
    moved = state.with_dynamics(
        state.positions + state.layout.distribute(shift),
        state.velocities,
        step_index=1,
    )
    transition = rebuild_owner_local_atomistic(plan, model, moved)
    assert not bool(transition.committed)
    assert int(transition.migration.maximum_packet) > 1
    for kept, accepted in zip(
        jax.tree.leaves(transition.state), jax.tree.leaves(moved), strict=True
    ):
        if jax.dtypes.issubdtype(accepted.dtype, jax.dtypes.prng_key):
            kept, accepted = jax.random.key_data(kept), jax.random.key_data(accepted)
        np.testing.assert_array_equal(np.asarray(kept), np.asarray(accepted))


def _verlet(
    plan: OwnerLocalAtomisticPlan,
    model: MACEPotential,
    state: OwnerLocalAtomisticState,
    steps: int,
    rebuild_every: int,
) -> OwnerLocalAtomisticState:
    """Fixed-cell velocity Verlet with explicit owner epoch transactions."""
    timestep = 0.02
    for _ in range(steps):
        evaluation = evaluate_owner_local_atomistic(plan, model, state)
        assert bool(evaluation.successful)
        acceleration = evaluation.forces / state.masses[:, None]
        half = state.velocities + 0.5 * timestep * acceleration
        step = state.step_index + 1
        state = state.with_dynamics(
            state.positions + timestep * half, half, step_index=step
        )
        if int(step) % rebuild_every == 0:
            transition = rebuild_owner_local_atomistic(plan, model, state)
            assert bool(transition.committed)
            state = transition.state
        following = evaluate_owner_local_atomistic(plan, model, state)
        assert bool(following.successful)
        # The half-step velocity migrated with its atom; read it from the state.
        velocity = (
            state.velocities + 0.5 * timestep * following.forces / state.masses[:, None]
        )
        state = state.with_dynamics(
            state.positions,
            velocity,
            step_index=step,
            forces=following.forces,
            thermostat_state=state.thermostat_state + 1.0,
        )
    return state


def test_migration_and_restart_reproduce_uninterrupted_evolution() -> None:
    model = _model(2)
    plan = _plan(model)
    positions, species_ids = _structure()
    velocities = np.zeros_like(positions)
    velocities[:, 0] = 6.0
    count = positions.shape[0]
    initial = _prepare(
        plan,
        model,
        positions,
        species_ids,
        velocities=velocities,
        stable_ids=np.arange(count, dtype=np.int64) * 3 + 5,
        atom_payload={"constraint_group": np.arange(count, dtype=np.int32) % 2},
        thermostat_state=np.zeros((2,)),
    )
    uninterrupted = _verlet(plan, model, initial, 6, 1)
    first = _verlet(plan, model, initial, 3, 1)
    checkpoint = checkpoint_owner_local_atomistic(plan, first)
    restored = restore_owner_local_atomistic(plan, model, checkpoint)
    resumed = _verlet(plan, model, restored, 3, 1)
    assert int(uninterrupted.topology.epoch) == 6

    def owner_of(state: OwnerLocalAtomisticState) -> dict[int, int]:
        active = np.asarray(state.layout.active)
        return dict(
            zip(
                np.asarray(state.layout.stable_ids)[active].tolist(),
                np.asarray(state.layout.slot_owners)[active].tolist(),
                strict=True,
            )
        )

    assert owner_of(uninterrupted) != owner_of(initial)
    assert int(jnp.sum(uninterrupted.image_counts)) > 0
    for name in ("positions", "velocities", "image_counts", "force_cache", "species"):
        np.testing.assert_array_equal(
            np.asarray(resumed.layout.collect(getattr(resumed, name))),
            np.asarray(uninterrupted.layout.collect(getattr(uninterrupted, name))),
        )
    np.testing.assert_array_equal(
        np.asarray(resumed.layout.collect(resumed.atom_payload["constraint_group"])),
        np.arange(count) % 2,
    )
    np.testing.assert_array_equal(
        np.asarray(resumed.thermostat_state), np.full((2,), 6.0)
    )
    unwrapped = np.asarray(
        resumed.layout.collect(
            resumed.positions + resumed.image_counts.astype(np.float64) @ CELL
        )
    )
    assert np.all(unwrapped[:, 0] > positions[:, 0])

    tampered = eqx.tree_at(
        lambda value: value.arrays,
        checkpoint,
        {
            **checkpoint.arrays,
            "atoms/positions": checkpoint.arrays["atoms/positions"] + 1.0,
        },
    )
    with pytest.raises(ValueError, match="content identity"):
        restore_owner_local_atomistic(plan, model, tampered)
    with pytest.raises(ValueError, match="model revision"):
        restore_owner_local_atomistic(plan, _model(2, key=3), checkpoint)
