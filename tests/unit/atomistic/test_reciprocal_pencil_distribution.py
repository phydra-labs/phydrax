"""Pencil particle-mesh reciprocal distribution against single-device terms.

The pencil schedule routes owner patches to the canonical pencil physical
layout (owner ``r`` holds x-block ``r % px`` and y-block ``r // px``), runs the
native distributed FFT on the owner-region spectral topology factoring the
ownership axis, evaluates the term-owned kernel on the modal z-slab ``r`` and
returns actions through the inverse transform and the exact reverse route.

Lane-reference cases are functional oracles on one device. ``devices`` cases
need ``px * py`` (four or six) physical or explicitly configured devices and
skip otherwise; forced host-CPU devices (``JAX_PLATFORMS=cpu
JAX_NUM_CPU_DEVICES=6``) are functional evidence only, never hardware
evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec

import phydrax as phx
from phydrax import config as pxconfig
from phydrax.atomistic._domain_plan import AtomisticDomainPlan, AtomisticDomainRolePlan
from phydrax.atomistic._domain_runtime import prepare_atomistic_domain
from phydrax.atomistic._domain_terms import AtomisticDomainEwaldTerm
from phydrax.atomistic._electrostatics import (
    EwaldReferencePotential,
    ParticleMeshEwaldPotential,
)
from phydrax.atomistic._neighborhood_bundle import AtomisticCellListImageSearch
from phydrax.atomistic._reciprocal_distribution import (
    _line_route_adjoint,
    _line_route_forward,
    _line_route_metadata,
    AtomisticReciprocalDistributionPlan,
    owner_reciprocal_actions,
    owner_reciprocal_energy,
    prepare_reciprocal_distribution,
    PreparedReciprocalDistribution,
    ReciprocalDistributionCapacityError,
    ReciprocalSitePayload,
)
from phydrax.discretization import PeriodicCell
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.discretization.spatial._distributed_relations import (
    DistributedOwnershipPlan,
)
from phydrax.sparse._streamed import StreamedRelationPlan
from tests._support import atomistic_domain as domain_support
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_program,
    role_capacity,
    search_capacity,
)
from tests.unit.atomistic.test_reciprocal_distribution import (
    _Blocks,
    _partition_owners,
    _pme_case,
    _pme_reference,
    LOCAL,
    TRICLINIC,
)


GRIDS = pytest.mark.parametrize(
    ("grid", "devices"),
    [((2, 3), False), ((3, 2), False), ((2, 2), True), ((2, 3), True)],
    ids=["lanes-2x3", "lanes-3x2", "devices-2x2", "devices-2x3"],
)


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


def _ownership(owners: int, *, devices: bool, local: int = LOCAL) -> Any:
    return domain_support.ownership(owners, devices=devices, local_capacity=local)


def _pencil(
    term: Any, grid: tuple[int, int], **options: Any
) -> AtomisticReciprocalDistributionPlan:
    options.setdefault("reuse_reach", 0.4)
    return AtomisticReciprocalDistributionPlan(
        term, schedule="pencil", pencil_grid=grid, **options
    )


def _shares(
    prepared: PreparedReciprocalDistribution, ownership: DistributedOwnershipPlan
) -> Callable[..., tuple[jax.Array, ...]]:
    """Owner-mapped ``(share, successful, capacity, scientific, fit, load)``."""
    axis = ownership.axis_name
    row = PartitionSpec(axis)

    def region(x: Any, q: Any, mask: Any, h: Any) -> tuple[Any, ...]:
        share, evidence = owner_reciprocal_energy(
            prepared, x, ReciprocalSitePayload(values=q), mask, h, axis
        )
        return (
            share[None],
            evidence.successful[None],
            evidence.capacity_successful[None],
            evidence.scientific_successful[None],
            evidence.packets_fit[None],
            evidence.maximum_packet_load[None],
        )

    return ownership.map(region, (row, row, row, PartitionSpec()), (row,) * 6)


# --- layout, identity and refusals ------------------------------------------------------


@GRIDS
def test_pencil_layout_binds_owner_coordinates_devices_and_collective_order(
    grid: tuple[int, int], devices: bool
) -> None:
    px, py = grid
    owners = px * py
    term, _, _, cell = _pme_case(owners)
    ownership = _ownership(owners, devices=devices)
    partition = FractionalOwnerPartition(cell, (px, py, 1))
    prepared = prepare_reciprocal_distribution(_pencil(term, grid), ownership, partition)
    report = prepared.report
    spectral = prepared.spectral
    assert spectral is not None and spectral.schedule == "pencil"
    assert report.schedule == "pencil" and report.spectral_grid == grid
    assert report.owner_coordinates == tuple(
        (owner % px, owner // px) for owner in range(owners)
    )
    assert report.physical_block_shape == (12 // px, 12 // py, 12, 1)
    assert report.modal_block_shape == (12, 12, 12 // owners, 1)
    assert report.physical_layout_id == spectral.physical_layout.layout_id
    assert report.modal_layout_id == spectral.modal_layout.layout_id
    patch = report.patch_shape
    assert patch is not None
    assert report.packet_planes is None
    assert report.packet_lines == min(patch[0], 12 // px) * min(patch[1], 12 // py)
    assert report.collective_order == (
        "patch-line-header:all_to_all",
        "patch-origin:all_gather",
        "patch-line-packets:all_to_all",
        f"spectral-forward:{spectral.report.forward_sequence_id}",
        f"spectral-inverse:{spectral.report.inverse_sequence_id}",
        "patch-line-adjoint:all_to_all",
    )
    assert spectral.report.collective_count == 1
    topology = spectral.topology
    assert topology.owner_axis == ownership.axis_name
    assert topology.mesh_shape == grid
    assert topology.execution_group_id == ownership.execution_group.spec.group_id
    names = topology.mesh_axis_names
    assert spectral.physical_layout.partition == (names[0], names[1], None, None)
    assert spectral.modal_layout.partition == (None, None, (names[1], names[0]), None)
    group_devices = tuple(ownership.execution_group.mesh.devices.flat)
    if devices:
        assert topology.lanes is None
        assert report.owner_devices == tuple(
            (int(device.process_index), int(device.id)) for device in group_devices
        )
        for owner, device in enumerate(group_devices):
            assert topology.mesh.devices[owner % px, owner // px] == device
    else:
        assert topology.lanes == owners and report.lane_reference
    # Schedule and grid are part of every identity.
    slab = prepare_reciprocal_distribution(
        AtomisticReciprocalDistributionPlan(term, reuse_reach=0.4),
        ownership,
        FractionalOwnerPartition(cell, (px, py, 1)),
    )
    assert slab.report.schedule == "slab" and slab.report.spectral_grid == (owners,)
    assert slab.plan.plan_id != prepared.plan.plan_id
    assert slab.report.report_id != report.report_id
    assert slab.prepared_id != prepared.prepared_id
    transposed = _pencil(term, (py, px))
    if px != py:
        assert transposed.plan_id != prepared.plan.plan_id
    # The scientific identity is the definition, never its placement.
    assert slab.plan.scientific_id == prepared.plan.scientific_id
    assert transposed.scientific_id == prepared.plan.scientific_id
    replicated = AtomisticReciprocalDistributionPlan(
        term, reuse_reach=0.8, strategy="replicated-mesh", maximum_packet_bytes=1 << 20
    )
    assert replicated.plan_id != slab.plan.plan_id
    assert replicated.scientific_id == prepared.plan.scientific_id


def test_pencil_plan_and_layout_refusals() -> None:
    term, _, _, cell = _pme_case(6)
    with pytest.raises(ValueError, match="explicit owner grid"):
        AtomisticReciprocalDistributionPlan(term, reuse_reach=0.4, schedule="pencil")
    with pytest.raises(ValueError, match="pencil_grid belongs"):
        AtomisticReciprocalDistributionPlan(term, reuse_reach=0.4, pencil_grid=(2, 3))
    with pytest.raises(ValueError, match="packet_lines"):
        AtomisticReciprocalDistributionPlan(term, reuse_reach=0.4, packet_lines=4)
    with pytest.raises(ValueError, match="packet_planes"):
        _pencil(term, (2, 3), packet_planes=4)
    with pytest.raises(ValueError, match="replicated"):
        _pencil(term, (2, 3), strategy="replicated-mesh")
    with pytest.raises(ValueError, match="pencil_grid"):
        _pencil(term, (0, 3))
    ewald = EwaldReferencePotential(1.1, 2.8, 3).prepare(term.system)
    with pytest.raises(ValueError, match="explicit"):
        _pencil(ewald, (2, 3))
    six = _ownership(6, devices=False)
    with pytest.raises(ValueError, match="does not factor"):
        prepare_reciprocal_distribution(
            _pencil(term, (2, 2)), six, FractionalOwnerPartition(cell, (6, 1, 1))
        )
    five = _ownership(5, devices=False)
    with pytest.raises(ValueError, match="not divisible"):
        prepare_reciprocal_distribution(
            _pencil(term, (5, 1)), five, FractionalOwnerPartition(cell, (5, 1, 1))
        )
    partition = FractionalOwnerPartition(cell, (6, 1, 1))
    with pytest.raises(ValueError, match="packet_lines"):
        prepare_reciprocal_distribution(
            _pencil(term, (2, 3), packet_lines=10_000), six, partition
        )
    with pytest.raises(ReciprocalDistributionCapacityError, match="maximum_packet_bytes"):
        prepare_reciprocal_distribution(
            _pencil(term, (2, 3), maximum_packet_bytes=64), six, partition
        )


# --- parity -------------------------------------------------------------------------------


@GRIDS
def test_pencil_pme_energy_forces_charges_and_cell_gradient_match_single_device(
    grid: tuple[int, int], devices: bool
) -> None:
    px, py = grid
    owners = px * py
    term, positions, charges, cell = _pme_case(owners)
    ownership = _ownership(owners, devices=devices)
    partition = FractionalOwnerPartition(cell, (px, py, 1))
    blocks = _Blocks(_partition_owners(partition, positions), owners, LOCAL)
    prepared = prepare_reciprocal_distribution(_pencil(term, grid), ownership, partition)
    mapped = _shares(prepared, ownership)
    mask = jnp.asarray(blocks.mask())
    vectors = jnp.asarray(TRICLINIC)
    x = jnp.asarray(blocks.block(positions))
    q = jnp.asarray(blocks.block(charges))
    shares, successful, capacity, scientific, fits, _ = mapped(x, q, mask, vectors)
    assert bool(jnp.all(successful & capacity & scientific & fits))

    def distributed(p: Any, c: Any, h: Any) -> jax.Array:
        return jnp.sum(mapped(p, c, mask, h)[0])

    def reference(p: Any, c: Any, h: Any) -> jax.Array:
        return _pme_reference(term, p, c, h)

    np.testing.assert_allclose(
        float(jnp.sum(shares)),
        float(reference(jnp.asarray(positions), jnp.asarray(charges), vectors)),
        rtol=1e-11,
        atol=1e-12,
    )
    gradient, charge_gradient, cell_gradient = jax.grad(distributed, argnums=(0, 1, 2))(
        x, q, vectors
    )
    expected, expected_charges, expected_cell = jax.grad(reference, argnums=(0, 1, 2))(
        jnp.asarray(positions), jnp.asarray(charges), vectors
    )
    np.testing.assert_allclose(blocks.unblock(gradient), expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        blocks.unblock(charge_gradient), expected_charges, rtol=1e-9, atol=1e-11
    )
    np.testing.assert_allclose(cell_gradient, expected_cell, rtol=1e-9, atol=1e-11)
    assert np.all(np.asarray(gradient)[~blocks.mask()] == 0.0)


def test_pencil_actions_are_exact_potentials_forces_and_hessian_actions() -> None:
    grid = (2, 3)
    owners = 6
    term, positions, charges, cell = _pme_case(owners)
    ownership = _ownership(owners, devices=False)
    partition = FractionalOwnerPartition(cell, (owners, 1, 1))
    blocks = _Blocks(_partition_owners(partition, positions), owners, LOCAL)
    prepared = prepare_reciprocal_distribution(_pencil(term, grid), ownership, partition)
    axis = ownership.axis_name
    row = PartitionSpec(axis)

    def region(position: Any, values: Any, mask: Any, lattice: Any) -> tuple[Any, Any]:
        actions = owner_reciprocal_actions(
            prepared, position, ReciprocalSitePayload(values=values), mask, lattice, axis
        )
        return actions.payload_gradient.values, actions.forces

    mapped = ownership.map(region, (row, row, row, PartitionSpec()), (row, row))
    shares = _shares(prepared, ownership)
    x = jnp.asarray(blocks.block(positions))
    q = jnp.asarray(blocks.block(charges))
    mask = jnp.asarray(blocks.mask())
    vectors = jnp.asarray(TRICLINIC)
    potentials, forces = mapped(x, q, mask, vectors)
    expected_forces, expected_potentials = jax.grad(
        lambda p, c: -_pme_reference(term, p, c, vectors), argnums=(0, 1)
    )(jnp.asarray(positions), jnp.asarray(charges))
    np.testing.assert_allclose(
        blocks.unblock(potentials), -expected_potentials, rtol=1e-9, atol=1e-11
    )
    np.testing.assert_allclose(
        blocks.unblock(forces), expected_forces, rtol=1e-9, atol=1e-11
    )

    def energy(p: Any, c: Any) -> jax.Array:
        return jnp.sum(shares(p, c, mask, vectors)[0])

    direction = jnp.asarray(
        blocks.block(np.random.default_rng(29).normal(size=charges.shape))
    )
    hessian_action = jax.grad(
        lambda c: jnp.vdot(direction, mapped(x, c, mask, vectors)[0])
    )(q)
    expected_action = jax.grad(
        lambda c: jnp.vdot(direction, jax.grad(energy, argnums=1)(x, c))
    )(q)
    assert float(jnp.max(jnp.abs(hessian_action))) > 0.0
    np.testing.assert_allclose(hessian_action, expected_action, rtol=1e-8, atol=1e-11)
    reference_action = jax.grad(
        lambda c: jnp.vdot(
            jnp.asarray(blocks.unblock(direction)),
            jax.grad(lambda v: _pme_reference(term, jnp.asarray(positions), v, vectors))(
                c
            ),
        )
    )(jnp.asarray(charges))
    np.testing.assert_allclose(
        blocks.unblock(hessian_action), reference_action, rtol=1e-8, atol=1e-11
    )


# --- transfer route -----------------------------------------------------------------------


@GRIDS
def test_pencil_line_route_is_the_canonical_layout_and_its_exact_transpose(
    grid: tuple[int, int], devices: bool
) -> None:
    """Forward lines land in the declared owner blocks; adjoint = linear transpose."""
    px, py = grid
    owners = px * py
    term, _, _, cell = _pme_case(owners)
    ownership = _ownership(owners, devices=devices)
    prepared = prepare_reciprocal_distribution(
        _pencil(term, grid), ownership, FractionalOwnerPartition(cell, (owners, 1, 1))
    )
    layout = prepared.mesh_layout
    assert layout is not None
    p0, p1, p2 = layout.patch_shape
    b0, b1, k2 = layout.block_shape
    k0, k1, _ = layout.grid_shape
    rng = np.random.default_rng(41)
    origins = rng.integers(0, 12, size=(owners, 3)).astype(np.int32)
    patches = rng.normal(size=(owners, p0, p1, p2, 1))
    fields = rng.normal(size=(owners, b0, b1, k2, 1))
    axis = ownership.axis_name
    row = PartitionSpec(axis)

    def region(origin: Any, patch: Any, block: Any) -> tuple[Any, ...]:
        route = _line_route_metadata(prepared, origin[0], axis)

        def forward(value: Any) -> Any:
            return _line_route_forward(prepared, route, value, axis)

        physical = forward(patch)
        adjoint = _line_route_adjoint(prepared, route, block, axis)
        (transposed,) = jax.linear_transpose(forward, patch)(block)
        return (
            physical,
            adjoint,
            transposed,
            jnp.vdot(physical, block)[None],
            jnp.vdot(patch, adjoint)[None],
            jnp.all(route.fits)[None],
        )

    physical, adjoint, transposed, left, right, fits = ownership.map(
        region, (row, row, row), (row,) * 6
    )(
        jnp.asarray(origins),
        jnp.asarray(patches.reshape((owners * p0, p1, p2, 1))),
        jnp.asarray(fields.reshape((owners * b0, b1, k2, 1))),
    )
    assert bool(jnp.all(fits))
    np.testing.assert_allclose(adjoint, transposed, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        float(jnp.sum(left)), float(jnp.sum(right)), rtol=1e-12, atol=1e-12
    )
    # Independent host assembly of the global mesh, cut into pencil blocks.
    mesh = np.zeros((k0, k1, k2, 1))
    for owner in range(owners):
        o0, o1, o2 = origins[owner]
        index = np.ix_(
            (o0 + np.arange(p0)) % k0,
            (o1 + np.arange(p1)) % k1,
            (o2 + np.arange(p2)) % k2,
        )
        np.add.at(mesh, index, patches[owner])
    blocks = np.asarray(physical).reshape((owners, b0, b1, k2, 1))
    for owner in range(owners):
        i0, i1 = owner % px, owner // px
        np.testing.assert_allclose(
            blocks[owner],
            mesh[i0 * b0 : (i0 + 1) * b0, i1 * b1 : (i1 + 1) * b1],
            rtol=1e-13,
            atol=1e-13,
        )


def test_pencil_packet_overflow_is_capacity_not_scientific_failure() -> None:
    grid = (2, 3)
    owners = 6
    term, positions, charges, cell = _pme_case(owners)
    ownership = _ownership(owners, devices=False)
    partition = FractionalOwnerPartition(cell, (owners, 1, 1))
    blocks = _Blocks(_partition_owners(partition, positions), owners, LOCAL)
    prepared = prepare_reciprocal_distribution(
        _pencil(term, grid, packet_lines=1), ownership, partition
    )
    assert prepared.report.packet_lines == 1
    _, successful, capacity, scientific, fits, load = _shares(prepared, ownership)(
        jnp.asarray(blocks.block(positions)),
        jnp.asarray(blocks.block(charges)),
        jnp.asarray(blocks.mask()),
        jnp.asarray(TRICLINIC),
    )
    assert not bool(jnp.any(fits)) and not bool(jnp.any(capacity))
    assert not bool(jnp.any(successful)) and bool(jnp.all(scientific))
    assert int(load[0]) > 1


# --- domain entrypoint --------------------------------------------------------------------


DOMAIN_LOCAL = 24
SKIN = 0.4


def _domain_case(
    grid: tuple[int, int], devices: bool
) -> tuple[Any, np.ndarray, ParticleMeshEwaldPotential, Any]:
    owners = grid[0] * grid[1]
    count = 12
    cell = PeriodicCell(TRICLINIC)
    rng = np.random.default_rng(3)
    inverse = np.linalg.inv(TRICLINIC)
    accepted: list[np.ndarray] = []
    while len(accepted) < count:
        candidate = rng.uniform(0.0, 1.0, 3) @ TRICLINIC
        separations = [
            np.linalg.norm(
                (
                    ((candidate - point) @ inverse)
                    - np.round((candidate - point) @ inverse)
                )
                @ TRICLINIC
            )
            for point in accepted
        ]
        if all(separation >= 1.0 for separation in separations):
            accepted.append(candidate)
    positions = np.asarray(accepted)
    charges = rng.normal(size=count)
    charges -= charges.mean()
    system = phx.atomistic.AtomisticSystemPlan(
        np.arange(10, 10 + count, dtype=np.int64),
        np.ones((count,), dtype=np.int64),
        np.ones((count,), dtype=np.float64),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros((count,), dtype=np.int64),
        charges=charges,
        cell=cell,
    ).prepare()
    pme = ParticleMeshEwaldPotential(1.1, 2.8, (12, 12, 12), spline_degree=4)
    prepared_pme = pme.prepare(system)
    term = AtomisticDomainEwaldTerm(
        prepared_pme, _pencil(prepared_pme, grid, reuse_reach=SKIN)
    )
    (requirement,) = pme.requirements.neighborhoods
    role = AtomisticDomainRolePlan(
        requirement,
        cell.image_stencil(requirement.bounded_cutoff() + SKIN, maximum_image_count=4096),
        streaming=StreamedRelationPlan(),
        skin=SKIN,
        edge_capacity=2048,
        alias_capacity=2048,
        halo_capacity=DOMAIN_LOCAL,
        term_key=pme.name,
    )
    plan = AtomisticDomainPlan(
        _ownership(owners, devices=devices, local=DOMAIN_LOCAL),
        FractionalOwnerPartition(cell, (owners, 1, 1)),
        (role,),
        migration_capacity=DOMAIN_LOCAL,
        message_capacity_bytes=1 << 24,
        terms=(term,),
    )
    return system, positions, pme, prepare_atomistic_domain(plan)


def _program_reference(pme: Any, system: Any, positions: np.ndarray) -> Any:
    prepared = prepare_program(
        phx.atomistic.AtomisticPotentialProgram([pme]),
        system,
        search=AtomisticCellListImageSearch(maximum_candidate_slots=1 << 16),
        skin=SKIN,
        search_capacities=(search_capacity(4096, particles_per_cell=32, images=343),),
        role_capacities=(role_capacity(4096, 256),),
    )
    state = bundle_state(prepared, positions)
    evaluation = prepared.evaluate(jnp.asarray(positions), state, compute_stress=True)
    assert bool(evaluation.successful)
    return evaluation


def _assert_domain_matches(evaluation: Any, reference: Any, state: Any) -> None:
    assert bool(evaluation.successful)
    assert bool(jnp.all(evaluation.status.term_capacity_successful))
    np.testing.assert_allclose(
        float(evaluation.energy), float(reference.energy), rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)),
        np.asarray(reference.forces),
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(evaluation.stress), np.asarray(reference.stress), rtol=1e-7, atol=1e-9
    )


@pytest.mark.parametrize(
    ("grid", "devices"),
    [((2, 3), False), ((2, 2), True)],
    ids=["lanes-2x3", "devices-2x2"],
)
def test_domain_pme_term_on_pencil_layout_matches_single_device_program(
    grid: tuple[int, int], devices: bool
) -> None:
    """The domain entrypoint: E/F/stress, then a migrating rebuild, on pencils."""
    system, positions, pme, runtime = _domain_case(grid, devices)
    (term,) = runtime.plan.terms
    assert isinstance(term, AtomisticDomainEwaldTerm)
    assert term.reciprocal.schedule == "pencil"
    state = runtime.ingress_global(
        positions,
        np.asarray(system.plan.atom_type_ids, dtype=np.int32),
        rng_key=jax.random.key(0),
    )
    _assert_domain_matches(
        runtime.evaluate(state), _program_reference(pme, system, positions), state
    )
    owners = grid[0] * grid[1]
    shift = np.broadcast_to(TRICLINIC[0] / owners, positions.shape)
    moved = runtime.with_dynamics(
        state,
        state.positions + state.layout.distribute(shift),
        state.momenta,
        step_index=1,
    )
    transition = runtime.rebuild(moved)
    assert bool(transition.committed)
    rebuilt = transition.state
    _assert_domain_matches(
        runtime.evaluate(rebuilt),
        _program_reference(pme, system, positions + shift),
        rebuilt,
    )
