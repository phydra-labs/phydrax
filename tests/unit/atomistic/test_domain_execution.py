"""Distributed atomistic domain execution against single-device global MACE.

Lane-reference ownership runs the owner regions as named vmap lanes on one
device with the same collectives as device-collective ownership; the
device-collective cases use the available devices. Forced CPU devices prove
functional collective parity only, not accelerator or multi-host
qualification. The oracle (``tests._support.atomistic_domain``) enumerates
every periodic image route on the host by brute force and evaluates the same
native MACE layers on the global graph with ordinary JAX.

Migrated scenarios (old ``test_distributed_mace.py`` /
``test_distributed_atomistic_completion.py``) keep their names where the
contract is unchanged; checkpoint identity/restore scenarios moved to the
durability owner (``test_domain_checkpoint.py``).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic._constraints import DistanceConstraintPlan
from phydrax.atomistic._domain_dynamics import (
    AtomisticDomainConstraintPlan,
    domain_constraint_rows,
    export_domain_constraints,
)
from phydrax.atomistic._domain_plan import (
    AtomisticDomainPlan,
    AtomisticDomainReductionPolicy,
    AtomisticDomainRolePlan,
)
from phydrax.atomistic._domain_runtime import (
    _owner_selection,
    AtomisticDomainPacket,
    prepare_atomistic_domain,
    PreparedAtomisticDomainRuntime,
)
from phydrax.atomistic._domain_state import AtomisticDomainState
from phydrax.atomistic._dynamics import AtomisticStepRejectionReason
from phydrax.atomistic._execution_binding import atomistic_case_address_words
from phydrax.atomistic._feature_execution import AtomisticExchangeCapacityError
from phydrax.discretization._periodic_cell import PeriodicCell, PeriodicImageStencil
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.discretization.particle._image_neighborhood import (
    ImageNeighborSelectionPolicy,
)
from tests._support.atomistic_domain import (
    CELL,
    CUTOFF,
    domain_plan,
    global_energy,
    global_reference,
    image_routes,
    ingress,
    mace_model,
    ownership,
    SKIN,
    structure,
)


REASON = AtomisticStepRejectionReason


def _runtime(**kwargs: Any) -> tuple[Any, PreparedAtomisticDomainRuntime]:
    layers = kwargs.pop("layers", 2)
    model = mace_model(layers)
    return model, prepare_atomistic_domain(domain_plan(model, **kwargs), model)


def _assert_global_parity(
    runtime: PreparedAtomisticDomainRuntime,
    model: Any,
    state: AtomisticDomainState,
    species_ids: np.ndarray,
    *,
    cell: np.ndarray = CELL,
) -> None:
    evaluation = runtime.evaluate(state)
    assert bool(evaluation.successful)
    assert int(evaluation.reasons) == 0
    positions = np.asarray(state.layout.collect(state.positions))
    energy, forces, strain = global_reference(model, positions, species_ids, cell)
    np.testing.assert_allclose(float(evaluation.energy), float(energy), rtol=1e-11)
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)), forces, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(evaluation.strain_derivative), strain, atol=1e-10
    )
    volume = abs(np.linalg.det(cell))
    np.testing.assert_allclose(
        np.asarray(evaluation.stress), 0.5 * (strain + strain.T) / volume, atol=1e-10
    )
    np.testing.assert_allclose(
        float(jnp.sum(evaluation.owner_energies)), float(energy), rtol=1e-11
    )


def test_route_oracle_has_repeated_and_self_images() -> None:
    positions, _ = structure()
    senders, receivers, _ = image_routes(positions, CELL)
    pairs = np.stack((senders, receivers), axis=1)
    _, multiplicity = np.unique(pairs, axis=0, return_counts=True)
    assert np.max(multiplicity) > 1
    assert np.any(senders == receivers)


@pytest.mark.parametrize("layers", [2, 3], ids=["two-layer", "three-layer"])
def test_lane_owners_match_global_energy_forces_and_strain(layers: int) -> None:
    model, runtime = _runtime(layers=layers)
    positions, species_ids = structure()
    state = ingress(runtime, model, positions, species_ids)
    _assert_global_parity(runtime, model, state, species_ids)


def test_empty_owners_match_global() -> None:
    """Four owners along x with every atom in the lower half: two owners are empty."""
    model, runtime = _runtime(owners=4)
    positions, species_ids = structure()
    fractional = np.linalg.solve(CELL.T, positions.T).T % 1.0
    fractional[:, 0] = 0.05 + 0.4 * fractional[:, 0]
    positions = fractional @ CELL
    state = ingress(runtime, model, positions, species_ids)
    loads = np.asarray(state.layout.active).reshape((4, -1)).sum(axis=1)
    assert np.count_nonzero(loads == 0) == 2
    _assert_global_parity(runtime, model, state, species_ids)


@pytest.mark.parametrize("owners", [1, 2, 3, 4, 5, 8])
def test_device_collective_owners_match_global(owners: int) -> None:
    model, runtime = _runtime(owners=owners, devices=True)
    plan = runtime.plan
    positions, species_ids = structure()
    momenta = np.random.default_rng(9).normal(size=positions.shape)
    state = ingress(runtime, model, positions, species_ids, momenta=momenta)
    _assert_global_parity(runtime, model, state, species_ids)
    selected = plan.ownership.execution_group.devices
    assert len(selected) == owners
    assert not plan.ownership.lane_reference
    assert state.positions.sharding.device_set == set(selected)
    if owners > 1:
        halo = state.topologies[0].halo
        assert halo is not None  # the halo strategy routes through a halo plan
        assert int(halo.evidence.halo_columns) > 0

    # Translate every atom into its next owner; periodic wrapping and payload
    # migration must preserve the same global energy, forces, and strain oracle.
    displacement = np.broadcast_to(CELL[0] / owners, positions.shape)
    moved = runtime.with_dynamics(
        state,
        state.positions + state.layout.distribute(displacement),
        state.momenta,
        step_index=1,
    )
    transition = runtime.rebuild(moved)
    assert bool(transition.committed)
    assert int(transition.reasons) == 0
    assert int(transition.migration.migrated) == (positions.shape[0] if owners > 1 else 0)
    rebuilt = transition.state
    np.testing.assert_allclose(
        np.asarray(rebuilt.layout.collect(rebuilt.unwrapped_positions())),
        positions + displacement,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        np.asarray(rebuilt.layout.collect(rebuilt.momenta)), momenta
    )
    np.testing.assert_array_equal(
        np.asarray(rebuilt.layout.collect(rebuilt.species)),
        np.asarray(state.layout.collect(state.species)),
    )
    _assert_global_parity(runtime, model, rebuilt, species_ids)


def test_multi_hop_dependency_crosses_owners_at_successive_layers() -> None:
    """A chain 0-1-2-3 split between owners couples atom 0 to atom 3 only
    through three interactions, so a missing per-layer exchange or a doubled
    reverse return would change the force on atom 0."""
    cell = np.diag([30.0, 30.0, 30.0])
    spacing = 0.8 * CUTOFF
    base = np.array([[13.0 + spacing * index, 15.0, 15.0] for index in range(4)])
    species_ids = np.array([1, 8, 1, 8], dtype=np.int32)
    model, runtime = _runtime(layers=3, cell=cell)

    def owner_forces(positions: np.ndarray) -> np.ndarray:
        state = ingress(runtime, model, positions, species_ids)
        evaluation = runtime.evaluate(state)
        assert bool(evaluation.successful)
        owners = np.asarray(runtime.plan.partition.owners(positions / 30.0))
        assert owners.tolist() == [0, 0, 1, 1]
        return np.asarray(state.layout.collect(evaluation.forces))

    moved = base.copy()
    moved[3, 1] += 0.2
    first, second = owner_forces(base), owner_forces(moved)
    references = []
    for positions, observed in ((base, first), (moved, second)):
        routes = image_routes(positions, cell)
        expected = -jax.grad(global_energy, argnums=1)(
            model, jnp.asarray(positions), species_ids, jnp.zeros((3, 3)), routes, cell
        )
        np.testing.assert_allclose(observed, expected, atol=1e-10)
        references.append(np.asarray(expected))
    expected_response = references[1][0] - references[0][0]
    # A missing cross-owner dependency gives zero response. The independent
    # oracle must distinguish that fault at the existing force-parity tolerance.
    assert np.max(np.abs(expected_response)) > 1e-10
    np.testing.assert_allclose(
        second[0] - first[0], expected_response, rtol=0.0, atol=1e-10
    )


def test_self_image_cell_has_cancelling_forces_and_nonzero_stress() -> None:
    model, runtime = _runtime()
    positions = np.array([[0.4, 0.7, 1.1]])
    species_ids = np.array([8], dtype=np.int32)
    state = ingress(runtime, model, positions, species_ids)
    evaluation = runtime.evaluate(state)
    assert bool(evaluation.successful)
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)), 0.0, atol=1e-11
    )
    _, _, strain = global_reference(model, positions, species_ids)
    assert np.max(np.abs(strain)) > 1e-6
    np.testing.assert_allclose(
        np.asarray(evaluation.strain_derivative), strain, atol=1e-10
    )


@pytest.mark.parametrize("stress_weight", [0.0, 3.0], ids=["ef", "efs"])
def test_force_loss_parameter_gradient_matches_global(stress_weight: float) -> None:
    model, runtime = _runtime()
    positions, species_ids = structure()
    state = ingress(runtime, model, positions, species_ids)
    rng = np.random.default_rng(11)
    reference_forces = rng.normal(size=positions.shape)
    reference_stress = 0.01 * rng.normal(size=(3, 3))
    volume = abs(np.linalg.det(CELL))
    kwargs: dict[str, Any] = {}
    if stress_weight:
        kwargs = {"stress_reference": reference_stress, "stress_weight": stress_weight}
    result = runtime.loss_gradient(
        state,
        -2.0,
        state.layout.distribute(reference_forces),
        energy_weight=0.5,
        force_weight=2.0,
        **kwargs,
    )
    assert bool(result.successful)
    routes = image_routes(positions, CELL)
    lanes = partition_parameters(model)

    def loss(parameters: Any) -> jax.Array:
        candidate = combine_parameters(parameters, lanes[1], lanes[2])
        reference = jnp.asarray(positions)
        zero = jnp.zeros((3, 3))
        energy = global_energy(candidate, reference, species_ids, zero, routes)
        forces = -jax.grad(global_energy, argnums=1)(
            candidate, reference, species_ids, zero, routes
        )
        value = 0.5 * (energy + 2.0) ** 2 + 2.0 * jnp.sum(
            (forces - reference_forces) ** 2
        )
        if stress_weight:
            strain = jax.grad(global_energy, argnums=3)(
                candidate, reference, species_ids, zero, routes
            )
            stress = 0.5 * (strain + strain.T) / volume
            value = value + stress_weight * jnp.sum((stress - reference_stress) ** 2)
        return value

    expected = jax.grad(loss)(lanes[0])
    np.testing.assert_allclose(float(result.loss), float(loss(lanes[0])), rtol=1e-11)
    for observed, oracle in zip(
        jax.tree.leaves(result.parameter_gradient),
        jax.tree.leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(oracle), atol=1e-9)


def test_owner_parameter_partials_sum_exactly_once_to_the_energy_gradient() -> None:
    """Owner partials sum to the global gradient for any owner count (no scaling)."""
    positions, species_ids = structure()
    routes = image_routes(positions, CELL)
    gradients = []
    for owners in (1, 2, 3):
        model, runtime = _runtime(owners=owners)
        state = ingress(runtime, model, positions, species_ids)
        partials = runtime.owner_parameter_partials(state, energy_cotangent=1.0)
        gradients.append(jax.tree.map(runtime.plan.reduction.ordered_sum, partials))
    lanes = partition_parameters(model)
    expected = jax.grad(
        lambda lane: global_energy(
            combine_parameters(lane, lanes[1], lanes[2]),
            jnp.asarray(positions),
            species_ids,
            jnp.zeros((3, 3)),
            routes,
        )
    )(lanes[0])
    for gradient in gradients:
        for observed, oracle in zip(
            jax.tree.leaves(gradient), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(observed), np.asarray(oracle), atol=1e-10
            )


def test_stale_owner_epoch_and_expired_certificate_fail_closed() -> None:
    model, runtime = _runtime()
    positions, species_ids = structure()
    state = ingress(runtime, model, positions, species_ids)
    advanced = state.layout.migrate(
        state.layout.slot_owners, packet_capacity=runtime.plan.migration_capacity
    ).layout
    stale = eqx.tree_at(lambda value: value.layout, state, advanced)
    evaluation = runtime.evaluate(stale)
    assert not bool(evaluation.certificate.owners_current)
    assert not bool(evaluation.successful)
    assert int(evaluation.reasons) & int(REASON.DOMAIN)
    assert np.all(np.isnan(np.asarray(evaluation.forces)))

    shift = np.zeros_like(positions)
    shift[0, 1] = 0.6 * SKIN
    moved = runtime.with_dynamics(
        state,
        state.positions + state.layout.distribute(shift),
        state.momenta,
        step_index=1,
    )
    expired = runtime.evaluate(moved)
    assert not bool(expired.certificate.displacement_certified[0])
    assert int(expired.reasons) & int(REASON.DOMAIN)
    assert np.isnan(float(expired.energy))


def test_dynamic_cell_parity_under_strain_and_deformation_certificate() -> None:
    """Ownership, aliases and certificates follow the CURRENT cell."""
    model, runtime = _runtime(dynamic_cell=True)
    positions, species_ids = structure()
    state = ingress(runtime, model, positions, species_ids)

    def strained(scale: float) -> tuple[AtomisticDomainState, np.ndarray]:
        strain = scale * np.array([[1.0, 0.4, 0.0], [0.4, -0.5, 0.2], [0.0, 0.2, 0.7]])
        deformation = np.eye(3) + strain
        cell = CELL @ deformation.T
        moved = runtime.with_dynamics(
            state,
            state.positions @ jnp.asarray(deformation.T),
            state.momenta,
            step_index=1,
            cell_vectors=cell,
        )
        return moved, cell

    small, small_cell = strained(2.0e-3)
    certificate = runtime.certificate(small)
    assert bool(certificate.certified)
    assert float(certificate.cell_deformation[0]) > 0.0
    _assert_global_parity(runtime, model, small, species_ids, cell=small_cell)

    large, large_cell = strained(8.0e-2)
    expired = runtime.evaluate(large)
    assert not bool(expired.successful)
    assert not bool(expired.certificate.displacement_certified[0])
    assert int(expired.reasons) & int(REASON.DOMAIN)
    transition = runtime.rebuild(large)
    assert bool(transition.committed)
    rebuilt = transition.state
    np.testing.assert_array_equal(
        np.asarray(rebuilt.topologies[0].reference_cell_vectors), large_cell
    )
    fractional = np.linalg.solve(
        large_cell.T, np.asarray(rebuilt.layout.collect(rebuilt.positions)).T
    ).T
    assert np.all((fractional >= -1e-12) & (fractional < 1.0 + 1e-12))
    _assert_global_parity(runtime, model, rebuilt, species_ids, cell=large_cell)


def test_static_cell_plan_refuses_cell_changes() -> None:
    model, runtime = _runtime()
    positions, species_ids = structure()
    state = ingress(runtime, model, positions, species_ids)
    with pytest.raises(ValueError, match="fixed cell"):
        runtime.with_dynamics(
            state, state.positions, state.momenta, step_index=1, cell_vectors=1.01 * CELL
        )
    with pytest.raises(ValueError, match="reference cell"):
        ingress(runtime, model, positions, species_ids, cell_vectors=1.01 * CELL)


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
    plan: AtomisticDomainPlan,
    stencil: PeriodicImageStencil,
    *,
    owners: Any = None,
    partition: FractionalOwnerPartition | None = None,
) -> AtomisticDomainPlan:
    role = plan.roles[0]
    return AtomisticDomainPlan(
        plan.ownership if owners is None else owners,
        plan.partition if partition is None else partition,
        (
            AtomisticDomainRolePlan(
                role.requirement,
                stencil,
                streaming=role.streaming,
                skin=role.skin,
                edge_capacity=role.edge_capacity,
                alias_capacity=role.alias_capacity,
                halo_capacity=role.halo_capacity,
            ),
        ),
        migration_capacity=plan.migration_capacity,
        message_capacity_bytes=plan.message_capacity_bytes,
    )


def test_plan_refuses_forged_incomplete_or_nonperiodic_image_stencils() -> None:
    # A one-atom cell needs every nonzero self image; a stencil that keeps the
    # identity but drops translations would silently remove all routes.
    model, runtime = _runtime()
    plan = runtime.plan
    stencil = plan.roles[0].stencil
    rebuilt = _stencil_like(stencil)
    assert rebuilt.stencil_id == stencil.stencil_id
    forged = eqx.tree_at(
        lambda value: value.shifts, stencil, jnp.zeros_like(stencil.shifts)
    )
    with pytest.raises(ValueError, match="complete box"):
        _replan(plan, forged)
    with pytest.raises(ValueError, match="does not cover"):
        _replan(plan, _stencil_like(stencil, extents=(1, 1, 1)))

    open_cell = PeriodicCell(CELL, periodic_axes=(True, True, False))
    owners = ownership(2, periodic_axes=(True, True, False))
    open_stencil = open_cell.image_stencil(CUTOFF + SKIN, maximum_image_count=4096)
    partition = FractionalOwnerPartition(open_cell, (2, 1, 1))
    _replan(plan, open_stencil, owners=owners, partition=partition)
    translated = _stencil_like(open_stencil, extents=open_stencil.extents[:2] + (1,))
    with pytest.raises(ValueError, match="nonperiodic axis"):
        _replan(plan, translated, owners=owners, partition=partition)

    positions = np.array([[0.4, 0.7, 1.1]])
    state = ingress(runtime, model, positions, np.array([8], dtype=np.int32))
    senders, _, _ = image_routes(positions, CELL)
    assert int(jnp.sum(state.topologies[0].valid)) >= senders.size > 0


def test_model_role_must_carry_the_model_requirement() -> None:
    """The domain cutoff/selection is the model's own directed-graph requirement."""
    plan = domain_plan(mace_model(2))
    with pytest.raises(ValueError, match="directed-graph requirement"):
        prepare_atomistic_domain(plan, mace_model(2, cutoff=CUTOFF - 0.2))


@pytest.mark.parametrize(
    ("owners", "devices"),
    [(2, False), (3, True), (5, True)],
    ids=["lanes-2", "devices-3", "devices-5"],
)
def test_capacity_refusals_never_truncate(owners: int, devices: bool) -> None:
    positions, species_ids = structure()
    model, runtime = _runtime(owners=owners, devices=devices, edge_capacity=4)
    with pytest.raises(ValueError, match="domain topology was refused"):
        ingress(runtime, model, positions, species_ids)
    model, runtime = _runtime(owners=owners, devices=devices, message_capacity_bytes=64)
    state = ingress(runtime, model, positions, species_ids)
    with pytest.raises(AtomisticExchangeCapacityError, match="message_capacity_bytes"):
        runtime.evaluate(state)


@pytest.mark.parametrize(
    ("owners", "devices"),
    [(2, False), (2, True), (3, True), (4, True), (5, True)],
    ids=["lanes-2", "devices-2", "devices-3", "devices-4", "devices-5"],
)
def test_failed_migration_returns_the_accepted_state(owners: int, devices: bool) -> None:
    model, runtime = _runtime(owners=owners, devices=devices, migration_capacity=1)
    positions, species_ids = structure()
    fractional = np.linalg.solve(CELL.T, positions.T).T % 1.0
    # All atoms start on owner zero and move to owner one, guaranteeing an
    # overflowing pair packet even for odd owner counts with otherwise idle owners.
    fractional[:, 0] = 0.25 / owners
    positions = fractional @ CELL
    state = ingress(
        runtime,
        model,
        positions,
        species_ids,
        atom_payload={"history": np.arange(positions.shape[0], dtype=np.float64)},
    )
    assert bool(runtime.evaluate(state).successful)
    shift = np.broadcast_to(CELL[0] / owners, positions.shape)
    moved = runtime.with_dynamics(
        state,
        state.positions + state.layout.distribute(shift),
        state.momenta,
        step_index=1,
    )
    transition = runtime.rebuild(moved)
    assert not bool(transition.committed)
    assert int(transition.migration.maximum_packet) > 1
    assert int(transition.reasons) & int(REASON.ROUTE_CAPACITY)
    assert not int(transition.reasons) & int(REASON.DOMAIN)
    for kept, accepted in zip(
        jax.tree.leaves(transition.state), jax.tree.leaves(moved), strict=True
    ):
        if jax.dtypes.issubdtype(accepted.dtype, jax.dtypes.prng_key):
            kept, accepted = jax.random.key_data(kept), jax.random.key_data(accepted)
        np.testing.assert_array_equal(np.asarray(kept), np.asarray(accepted))


def _verlet(
    runtime: PreparedAtomisticDomainRuntime,
    state: AtomisticDomainState,
    steps: int,
    rebuild_every: int,
) -> AtomisticDomainState:
    """Fixed-cell velocity Verlet with explicit owner epoch transactions."""
    timestep = 0.02
    for _ in range(steps):
        evaluation = runtime.evaluate(state)
        assert bool(evaluation.successful)
        safe = jnp.where(state.layout.active, state.masses, 1.0)
        half = state.momenta + 0.5 * timestep * evaluation.forces
        step = state.step_index + 1
        state = runtime.with_dynamics(
            state,
            state.positions + timestep * half / safe[:, None],
            half,
            step_index=step,
            time=state.time + timestep,
        )
        if int(step) % rebuild_every == 0:
            transition = runtime.rebuild(state)
            assert bool(transition.committed)
            state = transition.state
        following = runtime.evaluate(state)
        assert bool(following.successful)
        # The half-step momentum migrated with its atom; read it from the state.
        momentum = state.momenta + 0.5 * timestep * following.forces
        state = runtime.with_dynamics(
            state,
            state.positions,
            momentum,
            step_index=step,
            force=following.force_cache(step),
            replicated_payload={
                "thermostat": state.replicated_payload["thermostat"] + 1.0
            },
        )
    return state


@pytest.mark.parametrize("accumulation", ["deterministic", "compensated"])
@pytest.mark.parametrize("devices", [False, True], ids=["lanes", "devices"])
def test_ordered_evaluation_replays_exactly(accumulation: Any, devices: bool) -> None:
    model, runtime = _runtime(
        devices=devices, streaming_accumulation=accumulation, reduction=accumulation
    )
    positions, species_ids = structure()
    state = ingress(runtime, model, positions, species_ids)
    expected = runtime.evaluate(state)
    assert bool(expected.successful)
    for _ in range(3):
        observed = runtime.evaluate(state)
        for actual, reference in zip(
            jax.tree.leaves(observed), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))


@pytest.mark.parametrize("devices", [False, True], ids=["lanes", "devices"])
def test_migration_and_packet_restart_continue_every_leaf(devices: bool) -> None:
    """Every continuation leaf migrates; a packet restart continues the trajectory.

    The packet path re-ingests through process-local ingress and fresh role
    epochs, so the continued trajectory is compared within roundoff (bitwise
    same-topology replay belongs to the durability owner).
    """
    model, runtime = _runtime(
        devices=devices,
        streaming_accumulation="deterministic",
        reduction="deterministic",
    )
    positions, species_ids = structure()
    count = positions.shape[0]
    momenta = np.zeros_like(positions)
    momenta[:, 0] = 6.0
    initial = ingress(
        runtime,
        model,
        positions,
        species_ids,
        momenta=momenta,
        masses=np.linspace(1.0, 2.0, count),
        stable_ids=np.arange(count, dtype=np.int64) * 3 + 5,
        atom_payload={"constraint_group": np.arange(count, dtype=np.int32) % 2},
        replicated_payload={"thermostat": np.zeros((2,))},
    )
    uninterrupted = _verlet(runtime, initial, 6, 1)
    first = _verlet(runtime, initial, 3, 1)
    packet = runtime.to_packet(first)
    assert isinstance(packet, AtomisticDomainPacket)
    restored = runtime.from_packet((packet,))
    np.testing.assert_array_equal(
        np.asarray(restored.layout.collect(restored.unwrapped_positions())),
        np.asarray(first.layout.collect(first.unwrapped_positions())),
    )
    assert bool(restored.force_cache_valid)
    resumed = _verlet(runtime, restored, 3, 1)
    assert int(uninterrupted.topologies[0].epoch) == 6

    def owner_of(state: AtomisticDomainState) -> dict[int, int]:
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
    for name in ("momenta", "masses", "image_counts", "species"):
        np.testing.assert_allclose(
            np.asarray(resumed.layout.collect(getattr(resumed, name))),
            np.asarray(uninterrupted.layout.collect(getattr(uninterrupted, name))),
            rtol=1e-12,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        np.asarray(resumed.layout.collect(resumed.unwrapped_positions())),
        np.asarray(uninterrupted.layout.collect(uninterrupted.unwrapped_positions())),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(resumed.layout.collect(resumed.force.forces)),
        np.asarray(uninterrupted.layout.collect(uninterrupted.force.forces)),
        atol=1e-10,
    )
    np.testing.assert_array_equal(
        np.asarray(resumed.layout.collect(resumed.atom_payload["constraint_group"])),
        np.arange(count) % 2,
    )
    np.testing.assert_array_equal(
        np.asarray(resumed.replicated_payload["thermostat"]), np.full((2,), 6.0)
    )
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(resumed.rng_key)),
        np.asarray(jax.random.key_data(initial.rng_key)),
    )
    assert np.all(
        np.asarray(resumed.layout.collect(resumed.unwrapped_positions()))[:, 0]
        > positions[:, 0]
    )
    with pytest.raises(ValueError, match="model revision"):
        prepare_atomistic_domain(runtime.plan, mace_model(2, key=3)).from_packet(
            (packet,)
        )


def _constraint_rows_by_id(rows: Any) -> dict[int, tuple[Any, ...]]:
    active = np.asarray(rows.active)
    return {
        int(row): (tuple(int(atom) for atom in atoms), float(target), float(multiplier))
        for row, atoms, target, multiplier in zip(
            np.asarray(rows.row_ids)[active],
            np.asarray(rows.atoms)[active],
            np.asarray(rows.targets)[active],
            np.asarray(rows.lagrange)[active],
            strict=True,
        )
    }


def test_packet_fragments_merge_constraint_rows() -> None:
    """Constraint rows of every fragment survive; a header-only fragment adds none."""
    model, runtime = _runtime(owners=2)
    positions, species_ids = structure()
    pairs = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    distances = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    plan = AtomisticDomainConstraintPlan(
        DistanceConstraintPlan(maximum_iterations=64, tolerance=1.0e-10),
        constraint_count=pairs.shape[0],
        row_capacity=pairs.shape[0],
        halo_capacity=8,
        directory_capacity=16,
        migration_capacity=pairs.shape[0],
        component_row_capacity=2,
    )
    state = ingress(
        runtime,
        model,
        positions,
        species_ids,
        stable_ids=np.arange(positions.shape[0], dtype=np.int64),
        constraints=domain_constraint_rows(
            plan, runtime.plan.ownership, pairs, distances
        ),
    )
    packet = runtime.to_packet(state)
    assert packet.constraints is not None
    header = packet.without_rows()
    assert header.constraints is not None
    assert not np.any(header.constraints.active)
    restored = runtime.from_packet((header, packet))
    assert restored.constraints is not None
    assert _constraint_rows_by_id(
        export_domain_constraints(restored.constraints)
    ) == _constraint_rows_by_id(packet.constraints)
    with pytest.raises(ValueError, match="repeat an active constraint row"):
        runtime.from_packet((packet, packet))


def test_process_local_ingress_migrates_rows_without_global_host_ingress() -> None:
    """Rows supplied in arbitrary owner blocks are migrated on device to owners."""
    model, runtime = _runtime(owners=2)
    positions, species_ids = structure()
    count = positions.shape[0]
    total = runtime.plan.ownership.total_capacity
    mask = jnp.ones(species_ids.shape, dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(species_ids), mask))

    def rows(values: np.ndarray) -> np.ndarray:
        # Every atom starts in owner zero's block regardless of its owner.
        output = np.zeros((total,) + values.shape[1:], dtype=values.dtype)
        output[:count] = values
        return output

    active = np.zeros((total,), dtype=np.bool_)
    active[:count] = True
    state = runtime.ingress_process_local(
        rows(positions),
        rows(species),
        rng_key=jax.random.key(7),
        active=active,
        stable_ids=rows(np.arange(count, dtype=np.int64) + 10),
        logical_indices=rows(np.arange(count, dtype=np.int32)),
        logical_count=count,
        atom_payload={"tag": rows(np.arange(count, dtype=np.int32))},
    )
    assert int(state.layout.owner_epochs[0]) == 1
    owners = np.asarray(state.layout.slot_owners)[np.asarray(state.layout.active)]
    assert set(owners.tolist()) == {0, 1}
    np.testing.assert_array_equal(
        np.asarray(state.layout.collect(state.atom_payload["tag"])), np.arange(count)
    )
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(state.unwrapped_positions())),
        positions,
        atol=1e-12,
    )
    _assert_global_parity(runtime, model, state, species_ids)
    reference = ingress(runtime, model, positions, species_ids)
    np.testing.assert_allclose(
        float(runtime.evaluate(state).energy),
        float(runtime.evaluate(reference).energy),
        rtol=1e-13,
    )
    duplicate = rows(np.arange(count, dtype=np.int64))
    duplicate[1] = duplicate[0]
    with pytest.raises(ValueError, match="unique"):
        runtime.ingress_process_local(
            rows(positions),
            rows(species),
            rng_key=jax.random.key(7),
            active=active,
            stable_ids=duplicate,
            logical_indices=rows(np.arange(count, dtype=np.int32)),
            logical_count=count,
        )


def test_route_identities_are_material_and_layout_independent() -> None:
    model, runtime = _runtime(owners=2)
    positions, species_ids = structure()
    state = ingress(runtime, model, positions, species_ids, case_identity="system-a")
    view = runtime.owner_view(state)
    topology = state.topologies[0]
    fields = [np.asarray(value) for value in view.owned.route_identity]
    valid = np.asarray(topology.valid)
    words = np.asarray(atomistic_case_address_words(("system-a",)))[0]
    for column in range(8):
        np.testing.assert_array_equal(fields[column][valid], words[column])
    local = runtime.plan.ownership.local_capacity
    edges = valid.shape[0] // runtime.plan.ownership.owner_count
    owner = np.repeat(np.arange(runtime.plan.ownership.owner_count), edges)
    receiver_rows = owner * local + np.asarray(topology.receivers)
    source_rows = np.asarray(topology.route_owners) * local + np.asarray(
        topology.route_slots
    )
    ids = np.asarray(state.layout.stable_ids)
    np.testing.assert_array_equal(fields[8][valid], ids[source_rows][valid])
    np.testing.assert_array_equal(fields[9][valid], ids[receiver_rows][valid])
    stored = np.asarray(state.positions)
    unwrapped = np.asarray(state.unwrapped_positions())
    material = np.stack(fields[10:13], axis=1).astype(np.float64)
    shifts = np.asarray(topology.shifts).astype(np.float64)
    represented = stored[receiver_rows] - stored[source_rows] + shifts @ CELL
    physical = unwrapped[receiver_rows] - unwrapped[source_rows] + material @ CELL
    np.testing.assert_allclose(physical[valid], represented[valid], atol=1e-12)


def _selection_reference(
    policy: ImageNeighborSelectionPolicy,
    receivers: np.ndarray,
    displacement: np.ndarray,
    valid: np.ndarray,
    source_ids: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    """Independent per-receiver rule of `ImageNeighborSelectionPolicy`."""
    squared = np.sum(displacement * displacement, axis=-1)
    bound = policy.radius**2
    eligible = valid & (
        squared <= bound if policy.radius_boundary == "closed" else squared < bound
    )
    measure = squared if policy.measure == "squared-distance" else np.sqrt(squared)
    selected = np.zeros_like(valid)
    for receiver in np.unique(receivers[eligible]):
        routes = np.flatnonzero(eligible & (receivers == receiver))
        if routes.size <= policy.maximum_neighbors:
            selected[routes] = True
            continue
        ordered = sorted(
            routes,
            key=lambda route: (
                measure[route],
                source_ids[route],
                tuple(shifts[route]),
            ),
        )
        if policy.strict:
            selected[ordered[: policy.maximum_neighbors]] = True
        else:
            anchor = measure[ordered[policy.maximum_neighbors]]
            selected[
                [r for r in ordered if measure[r] <= anchor + policy.degeneracy_tolerance]
            ] = True
    return selected


@pytest.mark.parametrize("strict", [True, False], ids=["strict", "degeneracy-band"])
def test_owner_selection_is_the_declared_single_device_rule(strict: bool) -> None:
    """Ties (equal distances, equal sources) resolve by stable ID then shift order."""
    rng = np.random.default_rng(3)
    routes, receivers_count = 64, 5
    receivers = rng.integers(0, receivers_count, routes).astype(np.int32)
    shifts = rng.integers(-1, 2, (routes, 3)).astype(np.int32)
    source_ids = rng.integers(0, 6, routes).astype(np.int64)
    lengths = rng.choice(np.array([0.8, 1.0, 1.0, 1.3, 1.7, 2.1, 2.6]), routes)
    directions = rng.normal(size=(routes, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    displacement = lengths[:, None] * directions
    valid = rng.uniform(size=routes) < 0.9
    policy = ImageNeighborSelectionPolicy(
        2.4,
        3,
        degeneracy_tolerance=1.0e-9,
        measure="squared-distance",
        strict=strict,
        radius_boundary="open",
    )
    expected = _selection_reference(
        policy, receivers, displacement, valid, source_ids, shifts
    )
    selected, capped, degree = _owner_selection(
        policy,
        jnp.asarray(receivers),
        jnp.asarray(displacement),
        jnp.asarray(valid),
        jnp.asarray(source_ids),
        jnp.asarray(shifts),
        receivers_count,
    )
    np.testing.assert_array_equal(np.asarray(selected), expected)
    assert int(capped) > 0
    assert int(degree) >= 3
    permutation = rng.permutation(routes)
    permuted, _, _ = _owner_selection(
        policy,
        jnp.asarray(receivers[permutation]),
        jnp.asarray(displacement[permutation]),
        jnp.asarray(valid[permutation]),
        jnp.asarray(source_ids[permutation]),
        jnp.asarray(shifts[permutation]),
        receivers_count,
    )
    np.testing.assert_array_equal(np.asarray(permuted), expected[permutation])


def test_rebuild_and_evaluation_are_traceable() -> None:
    model, runtime = _runtime(owners=2)
    positions, species_ids = structure()
    state = ingress(runtime, model, positions, species_ids)
    eager = runtime.evaluate(state)
    compiled = eqx.filter_jit(lambda owner, current: owner.evaluate(current).energy)(
        runtime, state
    )
    np.testing.assert_allclose(float(compiled), float(eager.energy), rtol=1e-13)
    transition = eqx.filter_jit(lambda owner, current: owner.rebuild(current))(
        runtime, state
    )
    assert bool(transition.committed)
    assert jax.tree.structure(transition.state) == jax.tree.structure(state)


def test_reduction_policy_orders_owner_sums() -> None:
    """Migrated from the slab reduction scenarios: declared order, compensation."""
    # Exact binary increments survive Kahan compensation but are individually
    # lost by an owner-ordered float32 sum after the leading 2**24.
    values = jnp.concatenate(
        (
            jnp.asarray([2**24], dtype=jnp.float32),
            jnp.ones((1024,), dtype=jnp.float32),
            jnp.asarray([-(2**24)], dtype=jnp.float32),
        )
    )
    deterministic = AtomisticDomainReductionPolicy("deterministic")
    compensated = AtomisticDomainReductionPolicy("compensated")
    assert float(deterministic.ordered_sum(values)) == 0.0
    assert float(compensated.ordered_sum(values)) == 1024.0
    plan = ownership(4)
    owner_values = jnp.asarray([2**24, 1.0, 1.0, -(2**24)], dtype=jnp.float32)
    axis = plan.axis_name
    gathered = plan.map(
        lambda value: compensated.owner_sum(value[0], axis)[None],
        (jax.sharding.PartitionSpec(axis),),
        jax.sharding.PartitionSpec(axis),
    )(owner_values)
    np.testing.assert_array_equal(np.asarray(gathered), np.full((4,), 2.0, np.float32))
    with pytest.raises(ValueError):
        AtomisticDomainReductionPolicy("unordered")  # ty: ignore[invalid-argument-type]
