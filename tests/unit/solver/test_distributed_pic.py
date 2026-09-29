#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Distributed PIC and PIC restart.

Multi-device tests need forced host devices and run in a dedicated invocation,
once with four and once with eight devices:
``XLA_FLAGS=--xla_force_host_platform_device_count=4 pytest
tests/unit/solver/test_distributed_pic.py`` (and ``=8``).
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import phydrax as phx
from phydrax import ElectromagneticScaleContract
from phydrax._strict import StrictModule
from phydrax.lifecycle import TopologyRestartPolicy
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.units import CHARGE, UnitDefinition


D = phx.discretization
PIC = D.pic
sp = phx.solver.maxwell.spectral


def _mesh(shape: tuple[int, ...], names: tuple[str, ...] | None = None) -> Mesh:
    count = math.prod(shape)
    if len(jax.devices()) < count:
        pytest.skip(
            f"Needs {count} devices: run with "
            f"XLA_FLAGS=--xla_force_host_platform_device_count={max(count, 4)}."
        )
    axes = names if names is not None else ("x", "y", "z")[: len(shape)]
    return Mesh(np.asarray(jax.devices()[:count], dtype=object).reshape(shape), axes)


def _repository(tmp_path: Path) -> POSIXArtifactRepository:
    profile = HPCFilesystemProfile(
        "posix.pic-restart-test",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    policy = POSIXRepositoryPolicy(
        profile, maximum_chunk_bytes=1 << 20, maximum_metadata_bytes=1 << 20
    )
    return POSIXArtifactRepository(tmp_path / "repository", policy)


# -- runs ------------------------------------------------------------------------------


def _species_plan(
    capacity: int, sign: float, name: str, dimension: int, offset: int
) -> Any:
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + capacity),
        jnp.ones((capacity,)),
        ambient_dimension=dimension,
    ).prepare()
    return PIC.PICSpeciesPlan(
        D.ParticlePopulationPlan(support),
        PIC.PICChargeModelPlan(
            sign,
            name,
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )


def _reduced_species(capacity: int, dimension: int = 1) -> tuple[Any, ...]:
    return (
        _species_plan(capacity, -1.0, "electrons", dimension, 0),
        _species_plan(capacity, 1.0, "ions", dimension, 10_000),
    )


def _reduced_base(count: int = 32, dimension: int = 1) -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(count, periodic=True) for _ in range(dimension)),
        axis_names=("x", "y")[:dimension],
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    field = (
        phx.solver.CompatibleMaxwell1DPlan(grid)
        if dimension == 1
        else phx.solver.CompatibleMaxwell2DPlan(grid)
    )
    return phx.solver.ReducedMaxwellPICFieldSolver(
        field, PIC.ReducedPICTransferPlan(grid)
    )


def _cochain_base(
    capacity: int, counts: tuple[int, int, int] = (8, 4, 4)
) -> tuple[Any, tuple[Any, ...]]:
    h = 0.125
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(n, periodic=True) for n in counts),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [n * h for n in counts]]))
    bridge = D.StructuredCochainBridge(grid)
    species, transfers = [], []
    for offset, sign, name in ((0, -1.0, "electrons"), (10_000, 1.0, "ions")):
        plan = _species_plan(capacity, sign, name, 3, offset)
        charged = D.ChargedParticlePlan(sign * jnp.ones((capacity,)), name).prepare(
            plan.population.particles
        )
        transfers.append(PIC.PICParticleCochainTransferPlan(bridge).prepare(charged))
        species.append(plan)
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        sources=(phx.solver.PICMaxwellCurrentSourcePlan(),),
        plan_id="distributed-pic-test",
    ).prepare()
    electrostatic = phx.solver.CochainElectrostaticPlan(
        bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
    )
    solver = phx.solver.CochainMaxwellPICFieldSolver(
        maxwell,
        electrostatic,
        tuple(transfers),
        tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers),
    )
    return solver, tuple(species)


def _spectral_base(
    capacity: int,
    mesh: Mesh | None,
    decomposition: str,
    *,
    subdomains: tuple[int, int, int] = (4, 2, 1),
    guards: tuple[int, int, int] = (3, 3, 3),
    order: int = 4,
    counts: tuple[int, int, int] = (8, 4, 4),
) -> tuple[Any, tuple[Any, ...]]:
    """Finite-order staggered PSATD with Vay deposition, cell width 1/8.

    Global FFTs run on the PIC mesh (``None``: one device); local-guarded
    subdomains tile the device blocks.
    """
    h = 0.125
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(n, periodic=True) for n in counts),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [n * h for n in counts]]))
    bridge = D.StructuredCochainBridge(grid)
    species = tuple(
        _species_plan(capacity, sign, name, 3, offset)
        for offset, sign, name in ((0, -1.0, "electrons"), (10_000, 1.0, "ions"))
    )
    charged = tuple(
        D.ChargedParticlePlan(
            plan.charge_model.base_specific_charge * jnp.ones((capacity,)),
            plan.species_id,
        ).prepare(plan.population.particles)
        for plan in species
    )
    transfer = PIC.PICParticleCochainTransferPlan(bridge, shape_order=2)
    transfers = tuple(transfer.prepare(value) for value in charged)
    currents = tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers)
    options: dict[str, Any] = {
        "charge_conservation": "vay-deposition",
        "stencil": "finite-order",
        "stencil_order": order,
        "decomposition": decomposition,
    }
    if mesh is not None:
        options["topology"] = D.spectral.SpectralMeshTopology(mesh)
    if decomposition == "local-guarded":
        options["subdomains"] = subdomains
        options["guard_cells"] = guards
    solver = sp.SpectralMaxwellPlan(bridge, grid="staggered", **options).prepare(
        transfers, currents
    )
    return solver, species


def _particles(
    capacity: int, dimension: int, lengths: tuple[float, ...], speed: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Half-filled slots: uniform positions, velocities of magnitude ``speed``."""
    rng = np.random.default_rng(seed)
    position = rng.uniform(0.0, 1.0, (capacity, dimension)) * np.asarray(lengths)
    velocity = np.zeros((capacity, 3))
    velocity[:, :dimension] = rng.uniform(-speed, speed, (capacity, dimension))
    active = np.arange(capacity) < capacity // 2
    return position, velocity, active, active.astype(np.float64)


def _initial(
    run: Any, capacity: int, dimension: int, lengths: tuple[float, ...], dt: float
) -> Any:
    position, velocity, active, mass = _particles(capacity, dimension, lengths, 0.3, 1)
    ions, _, _, _ = _particles(capacity, dimension, lengths, 0.3, 2)
    return run.initialize(
        (position, ions),
        (velocity, np.zeros((capacity, 3))),
        dt,
        active_masks=(active, active),
        masses=(mass, mass),
    )


def _identity(population: Any) -> np.ndarray:
    return (np.asarray(population.id_hi, dtype=np.uint64) << np.uint64(32)) | (
        np.asarray(population.id_lo, dtype=np.uint64)
    )


def _by_identity(population: Any, *arrays: Any) -> dict[str, np.ndarray]:
    active = np.asarray(population.active)
    identity = _identity(population)
    parent = (np.asarray(population.parent_hi, dtype=np.uint64) << np.uint64(32)) | (
        np.asarray(population.parent_lo, dtype=np.uint64)
    )
    order = np.argsort(identity[active])
    result = {
        "identity": identity[active][order],
        "parent": parent[active][order],
        "mass": np.asarray(population.mass)[active][order],
    }
    for index, value in enumerate(arrays):
        result[f"array{index}"] = np.asarray(value)[active][order]
    return result


def _assert_same_species(actual: Any, expected: Any, *, atol: float) -> None:
    found = _by_identity(
        actual.population, actual.particles.position, actual.particles.proper_velocity
    )
    wanted = _by_identity(
        expected.population,
        expected.particles.position,
        expected.particles.proper_velocity,
    )
    np.testing.assert_array_equal(found["identity"], wanted["identity"])
    np.testing.assert_array_equal(found["parent"], wanted["parent"])
    for name in ("mass", "array0", "array1"):
        np.testing.assert_allclose(found[name], wanted[name], rtol=0.0, atol=atol)


def _assert_same_run(actual: Any, expected: Any, *, atol: float) -> None:
    for left, right in zip(
        jax.tree.leaves(actual.field), jax.tree.leaves(expected.field), strict=True
    ):
        scale = max(1.0, float(np.max(np.abs(np.asarray(right)), initial=0.0)))
        np.testing.assert_allclose(
            np.asarray(left), np.asarray(right), rtol=0.0, atol=atol * scale
        )
    for left, right in zip(actual.species, expected.species, strict=True):
        _assert_same_species(left, right, atol=atol)


def _distributed(
    solver: Any,
    species: tuple[Any, ...],
    mesh: Mesh,
    *,
    packet_capacity: int = 32,
    processes: tuple[Any, ...] = (),
    **options: Any,
) -> Any:
    extra = {
        name: options.pop(name)
        for name in (
            "recorders",
            "ownership",
            "external_fields",
            "key",
            "constraint_tolerance",
        )
        if name in options
    }
    distributed = phx.solver.distribute_pic_field_solver(solver, mesh, **options)
    pic = phx.solver.ElectromagneticPICPlan(
        distributed, species=species, processes=processes, **extra
    )
    return phx.solver.DistributedElectromagneticPICPlan(
        pic, packet_capacity=packet_capacity
    )


def _reduced_runs(shape: tuple[int, ...]) -> tuple[Any, Any, float]:
    base, species = _reduced_base(), _reduced_species(128)
    reference = phx.solver.ElectromagneticPICPlan(base, species=species)
    run = _distributed(base, species, _mesh(shape))
    return reference, run, 0.5 * float(base.stable_step)


# -- decomposition substrate -----------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "periodic"),
    [((4,), True), ((4,), False), ((2, 2), True), ((2, 2), False)],
    ids=["slab-periodic", "slab-walled", "blocks-periodic", "blocks-walled"],
)
def test_halo_accumulation_and_guard_exchange_match_global_cells(
    shape: tuple[int, ...], periodic: bool
) -> None:
    mesh = _mesh(shape)
    axes = len(shape)
    counts = (16, 8)[:axes]
    decomposition = PIC.PICDomainDecomposition(
        shape,
        counts,
        (0.0,) * axes,
        (1.0,) * axes,
        periodic=(periodic,) * axes,
        guard_cells=2,
    )
    parts = decomposition.part_count
    windows = np.stack([decomposition.window_indicator(part) for part in range(parts)])
    rng = np.random.default_rng(3)
    # Each device's deposit: window-local content plus a uniform (mean) mode.
    uniform = (
        rng.normal(size=(parts,) + (1,) * axes + (3,))
        if periodic
        else np.zeros((parts,) + (1,) * axes + (3,))
    )
    local = rng.normal(size=(parts, *counts, 3)) * windows[..., None] + uniform
    owned = rng.normal(size=(*counts, 3))
    names = tuple(mesh.axis_names)
    slots = PartitionSpec(names)
    planes = PartitionSpec(*names)

    def device(values: Any, owned: Any) -> tuple[Any, Any, Any]:
        coordinates = tuple(jax.lax.axis_index(name) for name in names)
        return (
            decomposition.accumulate(values[0], coordinates, axis_names=names),
            decomposition.fill_window(owned, coordinates, axis_names=names)[None],
            decomposition.outside_window(values[0], coordinates)[None],
        )

    accumulated, filled, outside = jax.jit(
        jax.shard_map(
            device,
            mesh=mesh,
            in_specs=(slots, planes),
            out_specs=(planes, slots, slots),
            check_vma=False,
        )
    )(
        jnp.asarray(local),
        jax.device_put(jnp.asarray(owned), NamedSharding(mesh, planes)),
    )
    np.testing.assert_allclose(accumulated, local.sum(axis=0), rtol=0.0, atol=1e-13)
    # Guards (edges and corners) are filled from their owners.
    np.testing.assert_array_equal(filled, owned[None] * windows[..., None])
    assert not np.any(np.asarray(outside))
    for part in range(parts):
        coordinates = decomposition.coordinates(part)
        reach = np.ones(counts, dtype=bool)
        for axis, coordinate in enumerate(coordinates):
            cells = np.arange(counts[axis])
            width = counts[axis] // shape[axis]
            distance = np.abs(cells - (width * coordinate + (width - 1) / 2))
            if periodic:
                distance = np.minimum(distance, counts[axis] - distance)
            index = [None] * axes
            index[axis] = slice(None)
            reach = reach & (distance <= (width - 1) / 2 + 2)[tuple(index)]
        np.testing.assert_array_equal(windows[part], reach)


def test_identity_tiles_must_nest_in_device_blocks() -> None:
    with pytest.raises(ValueError, match="identity_tiles"):
        PIC.PICDomainDecomposition(
            (4,), (16,), (0.0,), (1.0,), periodic=(True,), identity_tiles=(2,)
        )


# -- distributed runs --------------------------------------------------------------------


_ROUTES = [
    ("reduced", (4,)),
    ("cochain", (4,)),
    ("cochain", (2, 2)),
    ("cochain", (2, 2, 2)),
    ("spectral-global-fft", (4,)),
    ("spectral-global-fft", (2, 2)),
    ("spectral-local-guarded", (4,)),
    ("spectral-local-guarded", (2, 2)),
]


@pytest.mark.parametrize(
    ("route", "shape"),
    _ROUTES,
    ids=[f"{route}-{'x'.join(map(str, shape))}" for route, shape in _ROUTES],
)
def test_distributed_run_matches_single_device_run(
    route: str, shape: tuple[int, ...]
) -> None:
    mesh = _mesh(shape)
    options: dict[str, Any] = {}
    match route:
        case "reduced":
            base, species = _reduced_base(), _reduced_species(128)
            capacity, dimension, lengths, steps = 128, 1, (1.0,), 12
        case "cochain":
            # Block windows are local except on the eight-device 2×2×2 mesh,
            # whose 4×2×2 blocks' windows cover the grid.
            counts = {1: (16, 4, 4), 2: (16, 12, 4), 3: (8, 4, 4)}[len(shape)]
            base, species = _cochain_base(64, counts)
            options["guard_cells"] = 2
            capacity, dimension, steps = 64, 3, 10
            lengths = tuple(0.125 * n for n in counts)
        case _:
            base, species = _spectral_base(
                32,
                mesh,
                route.removeprefix("spectral-"),
                subdomains=(4, 2, 1),
                counts=(16, 16, 4),
            )
            capacity, dimension, lengths, steps = 32, 3, (2.0, 2.0, 0.5), 6
    # Local-guarded PSATD keeps Gauss's law only to its stencil truncation; the
    # cochain Coulomb initialization is an iterative solve to ~1e-8.
    tolerance = (
        {"constraint_tolerance": 1e-2}
        if route.endswith("guarded")
        else {"constraint_tolerance": 1e-6}
        if route == "cochain"
        else {}
    )
    reference = phx.solver.ElectromagneticPICPlan(base, species=species, **tolerance)
    run = _distributed(base, species, mesh, **tolerance, **options)
    evidence = run.evidence
    if shape != (2, 2, 2):
        # Deposits and gathers are genuinely window-local.
        assert not np.all(run.solver.decomposition.window_indicator(0))
    assert evidence.distributed and evidence.mesh_shape == shape
    assert evidence.part_count == math.prod(shape)
    assert evidence.route == route
    if route == "cochain":
        capabilities = evidence.maxwell_capabilities
        assert capabilities is not None and capabilities.distributed
    if route == "spectral-local-guarded":
        assert evidence.spectral_guard_cells == (3, 3, 3)[: len(shape)]
    dt = 0.4 * float(base.stable_step)
    expected = _initial(reference, capacity, dimension, lengths, dt)
    actual = _initial(run, capacity, dimension, lengths, dt)
    _assert_same_run(actual, expected, atol=1e-12)
    advance = eqx.filter_jit(lambda state: reference.step_detailed(state, dt))
    distributed = eqx.filter_jit(lambda state: run.step_detailed(state, dt))
    migrated = 0
    for _ in range(steps):
        reference_step, step = advance(expected), distributed(actual)
        assert bool(reference_step.successful)
        assert bool(step.successful), int(step.rejection_reason)
        np.testing.assert_allclose(
            float(step.diagnostics.electric_constraint),
            float(reference_step.diagnostics.electric_constraint),
            rtol=1e-8,
            atol=1e-10,
        )
        expected, actual = reference_step.accepted_state, step.accepted_state
        migrated += int(step.migration.migrated)
    assert migrated > 0
    _assert_same_run(actual, expected, atol=1e-11)
    decomposition = run.solver.decomposition
    for value in actual.species:
        owner = np.asarray(decomposition.owner(value.particles.position))
        active = np.asarray(value.population.active)
        block = np.arange(capacity) // (capacity // decomposition.part_count)
        np.testing.assert_array_equal(owner[active], block[active])


def test_local_guarded_matches_global_fft_within_stencil_truncation() -> None:
    mesh = _mesh((4,))
    counts = (16, 4, 4)
    guarded, species = _spectral_base(
        32, mesh, "local-guarded", subdomains=(4, 1, 1), guards=(2, 2, 2), counts=counts
    )
    wide, _ = _spectral_base(
        32, mesh, "local-guarded", subdomains=(4, 1, 1), guards=(3, 3, 3), counts=counts
    )
    exact, _ = _spectral_base(32, None, "global-fft", counts=counts)
    dt = 0.4 * float(exact.stable_step)
    tail = guarded.guard_truncation(dt)
    # The truncation evidence decreases with the guard width.
    assert 0.0 < wide.guard_truncation(dt) < tail < 1.0
    reference = phx.solver.ElectromagneticPICPlan(exact, species=species)
    run = _distributed(guarded, species, mesh)
    expected = _initial(reference, 32, 3, (2.0, 0.5, 0.5), dt)
    actual = _initial(run, 32, 3, (2.0, 0.5, 0.5), dt)
    advance = eqx.filter_jit(lambda state: reference.step_detailed(state, dt))
    distributed = eqx.filter_jit(lambda state: run.step_detailed(state, dt))
    steps = 4
    for _ in range(steps):
        expected = advance(expected).accepted_state
        actual = distributed(actual).accepted_state
    scale = max(
        float(np.max(np.abs(np.asarray(value))))
        for value in (expected.field.electric, expected.field.magnetic)
    )
    difference = max(
        float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
        for left, right in (
            (actual.field.electric, expected.field.electric),
            (actual.field.magnetic, expected.field.magnetic),
        )
    )
    # Differ by the truncated kernel, not by roundoff; bounded by the evidence.
    assert 1e-12 * scale < difference <= 4.0 * steps * tail * scale


def test_diagonal_migration_crosses_a_block_corner_in_one_packet() -> None:
    mesh = _mesh((2, 2))
    base, species = _reduced_base(8, 2), _reduced_species(16, 2)
    run = _distributed(base, species, mesh, packet_capacity=4)
    dt = 0.4 * float(base.stable_step)
    position = np.full((16, 2), 0.1)
    position[0] = (0.499, 0.499)
    velocity = np.zeros((16, 3))
    velocity[0, :2] = 0.2
    active = np.arange(16) < 2
    ions = np.full((16, 2), 0.1)
    state = run.initialize(
        (position, ions),
        (velocity, np.zeros((16, 3))),
        dt,
        active_masks=(active, active),
        masses=(active * 1.0, active * 1.0),
    )
    identity = _identity(state.species[0].population)[
        np.asarray(state.species[0].population.active)
    ]
    result = eqx.filter_jit(lambda value: run.step_detailed(value, dt))(state)
    assert bool(result.successful), int(result.rejection_reason)
    migration = result.migration
    assert int(migration.migrated) == 1
    diagonal = migration.shifts.index((1, 1))
    assert int(migration.sent[diagonal]) == 1
    electrons = result.accepted_state.species[0]
    found = np.flatnonzero(
        np.asarray(electrons.population.active)
        & np.isin(_identity(electrons.population), identity)
    )
    moved = [
        slot
        for slot in found
        if np.all(np.asarray(electrons.particles.position)[slot] > 0.5)
    ]
    # Device (1, 1) owns the last slot block.
    assert len(moved) == 1 and moved[0] >= 12


def test_migration_overflow_rejects_the_whole_step() -> None:
    _, run, dt = _reduced_runs((4,))
    capacity = 128
    # Twelve electrons just below the slab-0/slab-1 face moving across it.
    position = np.full((capacity, 1), 0.1)
    position[:12, 0] = 0.2495
    velocity = np.zeros((capacity, 3))
    velocity[:12, 0] = 0.4
    active = np.arange(capacity) < 24
    ions = np.linspace(0.01, 0.99, capacity)[:, None]
    state = run.initialize(
        (position, ions),
        (velocity, np.zeros((capacity, 3))),
        dt,
        active_masks=(active, active),
        masses=(active * 1.0, active * 1.0),
    )
    narrow = phx.solver.DistributedElectromagneticPICPlan(run.pic, packet_capacity=4)
    result = eqx.filter_jit(lambda value: narrow.step_detailed(value, dt))(state)
    assert bool(result.diagnostics.successful)
    assert not bool(result.successful)
    assert bool(result.migration.packet_overflow)
    assert int(result.rejection_reason) & int(PIC.PICRejectionReason.MIGRATION)
    for left, right in zip(
        jax.tree.leaves(result.accepted_state), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    accepted = eqx.filter_jit(lambda value: run.step_detailed(value, dt))(state)
    assert bool(accepted.successful)
    assert int(accepted.migration.migrated) == 12


def test_spectral_transforms_off_the_pic_mesh_are_refused() -> None:
    mesh = _mesh((4,))
    base, _ = _spectral_base(8, None, "local-guarded", subdomains=(4, 1, 1))
    with pytest.raises(ValueError, match="FFT topology"):
        phx.solver.distribute_pic_field_solver(base, mesh)
    untiled, _ = _spectral_base(8, mesh, "local-guarded", subdomains=(2, 1, 1))
    with pytest.raises(ValueError, match="tile every device block"):
        phx.solver.distribute_pic_field_solver(untiled, mesh)


def test_deposits_that_are_not_window_local_are_refused() -> None:
    # The continuity-projected reduced 2-D current spreads over the whole grid.
    base, species = _reduced_base(16, 2), _reduced_species(64, 2)
    solver = phx.solver.distribute_pic_field_solver(base, _mesh((2, 2)))
    with pytest.raises(ValueError, match="not window-local"):
        phx.solver.ElectromagneticPICPlan(solver, species=species)


def test_size_one_mesh_axes_are_refused() -> None:
    base, _ = _cochain_base(8)
    with pytest.raises(ValueError, match="size-one"):
        phx.solver.distribute_pic_field_solver(base, _mesh((2, 1)))


def test_guard_narrower_than_the_transfer_footprint_is_refused() -> None:
    base, species = _reduced_base(), _reduced_species(128)
    solver = phx.solver.distribute_pic_field_solver(
        base, _mesh((4,)), guard_cells=1, particle_margin=0.9
    )
    with pytest.raises(ValueError, match="guard_cells=1"):
        phx.solver.ElectromagneticPICPlan(solver, species=species)


# -- distributed processes -----------------------------------------------------------------

_ALPHA = Fraction(1000, 137036)
_SCHWINGER = 410000
_HBAR = 4 * Fraction(math.pi) * _ALPHA / _SCHWINGER**2
_MASS = float(_HBAR) * _SCHWINGER
_SCALE = ElectromagneticScaleContract.code_units(
    PIC.PIC_CODE_RELATIVITY.dimensional_scale,
    UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
    gravitational_constant=1,
    speed_of_light=1,
    reduced_planck_constant=_HBAR,
    boltzmann_constant=1,
    elementary_charge=Fraction(_MASS),
    electron_mass=Fraction(_MASS),
    vacuum_permittivity=1,
    constant_set_id="distributed-qed-test",
)


class _UniformMagnetic(StrictModule):
    """Uniform ``B = b ẑ`` in code units (m ω₀/e for a 1 μm laser)."""

    strength: float = eqx.field(static=True)

    @property
    def source_id(self) -> str:
        return f"uniform-magnetic-{self.strength!r}"

    def external_fields(self, positions: Any, times: Any, /) -> Any:
        del positions
        magnetic = jnp.zeros((times.shape[0], 3)).at[:, 2].set(self.strength)
        return PIC.ExternalFieldSample(
            jnp.zeros_like(magnetic), magnetic, jnp.ones(times.shape, dtype=bool)
        )


@pytest.fixture(scope="module")
def tables() -> tuple[Any, Any]:
    return (
        PIC.QEDTable("nonlinear-compton", maximum_chi=100.0),
        PIC.QEDTable("nonlinear-breit-wheeler", maximum_chi=100.0),
    )


_QED_CAPACITY = 256


def _qed_run(tables: tuple[Any, Any], mesh: Mesh) -> Any:
    """γ = 4000 pairs in B = 2000 (χ ≈ 20): photon emission and pair creation."""
    compton = PIC.NonlinearComptonPlan(
        "lcfa",
        _SCALE,
        -_MASS,
        _MASS,
        tables[0],
        maximum_chi=100.0,
        minimum_gamma=1.0,
        maximum_event_probability=0.2,
    )
    pairs = PIC.NonlinearBreitWheelerPlan(
        _SCALE, _MASS, _MASS, tables[1], maximum_chi=100.0
    )
    photons = PIC.QEDPhotonSpeciesPlan(
        _QED_CAPACITY,
        1,
        escape_lower=(35.0,),
        escape_upper=(65.0,),
        energy_edges=tuple(np.geomspace(1.0, 1.0e5, 11) * _MASS),
    )
    process = PIC.QEDCascadeProcess(
        compton,
        photons,
        emitters=(0, 1),
        breit_wheeler=pairs,
        electron=0,
        positron=1,
        gather_species=0,
        minimum_photon_energy=2.0 * _MASS,
    )
    grid = D.TensorGridPlan(
        (D.UniformCellAxisSpec(64, periodic=True),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [100.0]]))
    base = phx.solver.ReducedMaxwellPICFieldSolver(
        phx.solver.CompatibleMaxwell1DPlan(grid), PIC.ReducedPICTransferPlan(grid)
    )
    species = (
        _species_plan(_QED_CAPACITY, -1.0, "electrons", 1, 0),
        _species_plan(_QED_CAPACITY, 1.0, "positrons", 1, 100_000),
    )
    recorder = PIC.PICTrackRecorder(
        species,
        np.zeros(4, dtype=np.int32),
        (np.zeros(4, dtype=np.uint32), np.arange(4, dtype=np.uint32)),
        relativity=PIC.PIC_CODE_RELATIVITY,
        sample_capacity=64,
    )
    return _distributed(
        base,
        species,
        mesh,
        packet_capacity=64,
        processes=(process,),
        recorders=(recorder,),
        ownership="subgrid-reaction",
        external_fields=(_UniformMagnetic(2000.0),),
        key=jax.random.key(1),
    )


def _qed_seed(run: Any, dt: float) -> Any:
    capacity, seeds = _QED_CAPACITY, 8
    position = np.zeros((capacity, 1))
    position[:seeds, 0] = np.linspace(30.0, 70.0, seeds)
    active = np.arange(capacity) < seeds
    velocity = np.zeros((capacity, 3))
    velocity[:seeds, 0] = math.sqrt(1.0 - 1.0 / 4000.0**2)
    mass = np.where(active, 1.0e-9 * _MASS, 0.0)
    return run.initialize(
        (position, position),
        (velocity, -velocity),
        dt,
        active_masks=(active, active),
        masses=(mass, mass),
    )


def _qed_advance(run: Any, state: Any, dt: float, steps: int) -> tuple[Any, list[Any]]:
    step = eqx.filter_jit(lambda value: run.step_detailed(value, dt))
    results = []
    for _ in range(steps):
        result = step(state)
        assert bool(result.successful), int(result.rejection_reason)
        results.append(result)
        state = result.accepted_state
    return state, results


def _assert_same_cascade(actual: Any, expected: Any) -> None:
    for left, right in zip(actual.species, expected.species, strict=True):
        _assert_same_species(left, right, atol=1e-9)
    bank, reference = actual.processes[0].photons, expected.processes[0].photons
    found = _by_identity(bank.population, bank.position, bank.momentum)
    wanted = _by_identity(reference.population, reference.position, reference.momentum)
    np.testing.assert_array_equal(found["identity"], wanted["identity"])
    np.testing.assert_array_equal(found["parent"], wanted["parent"])
    np.testing.assert_allclose(found["array0"], wanted["array0"], rtol=1e-12)
    np.testing.assert_allclose(found["array1"], wanted["array1"], rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(bank.escaped_number), np.asarray(reference.escaped_number), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(bank.escaped_energy), np.asarray(reference.escaped_energy), rtol=1e-12
    )


def test_qed_cascade_on_four_devices_equals_one_device(
    tables: tuple[Any, Any],
) -> None:
    dt, steps = 0.005, 150
    single = _qed_run(tables, _mesh((1,)))
    split = _qed_run(tables, _mesh((4,)))
    expected, reference = _qed_advance(single, _qed_seed(single, dt), dt, steps)
    actual, results = _qed_advance(split, _qed_seed(split, dt), dt, steps)
    _assert_same_cascade(actual, expected)
    decays = 0
    for left, right in zip(results, reference, strict=True):
        (ledger,), (wanted,) = left.diagnostics.processes, right.diagnostics.processes
        (evidence,), (wanted_evidence,) = (
            left.diagnostics.process_evidence,
            right.diagnostics.process_evidence,
        )
        assert int(ledger.event_count) == int(wanted.event_count)
        assert int(evidence.emissions) == int(wanted_evidence.emissions)
        assert int(evidence.decays) == int(wanted_evidence.decays)
        assert int(evidence.escaped) == int(wanted_evidence.escaped)
        radiation, wanted_radiation = ledger.radiation, wanted.radiation
        assert radiation is not None and wanted_radiation is not None
        np.testing.assert_allclose(
            float(radiation.radiated_energy),
            float(wanted_radiation.radiated_energy),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            float(left.diagnostics.energy.created_rest_energy),
            float(right.diagnostics.energy.created_rest_energy),
            rtol=1e-12,
        )
        decays += int(evidence.decays)
    assert decays > 0
    # Pairs and photons of every device carry distinct identities.
    for population in (
        *(value.population for value in actual.species),
        actual.processes[0].photons.population,
    ):
        identity = _identity(population)[np.asarray(population.active)]
        assert np.unique(identity).size == identity.size
    # The track recorder follows the seeds identically on both topologies.
    recorded, wanted_recorded = actual.recorders[0], expected.recorders[0]
    np.testing.assert_array_equal(
        np.asarray(recorded.buffer.active), np.asarray(wanted_recorded.buffer.active)
    )
    np.testing.assert_allclose(
        np.asarray(recorded.buffer.positions),
        np.asarray(wanted_recorded.buffer.positions),
        rtol=0.0,
        atol=1e-9,
    )


def test_identities_created_on_different_devices_never_collide() -> None:
    mesh = _mesh((4,))
    decomposition = PIC.PICDomainDecomposition(
        (4,), (16,), (0.0,), (1.0,), periodic=(True,), identity_tiles=(8,)
    )
    capacity = 8
    plan = _species_plan(capacity, -1.0, "electrons", 1, 0).population
    block = D.ParticlePopulationPlan(
        D.ParticleSetPlan(
            jnp.arange(capacity // 4), jnp.ones((capacity // 4,)), ambient_dimension=1
        ).prepare()
    )
    state = plan.initialize(
        active_mask=np.arange(capacity) < 2, masses=(np.arange(capacity) < 2) * 1.0
    )
    positions = np.asarray([[0.5], [4.5], [8.5], [12.5]])[:, None, :].repeat(2, axis=1)

    def device(active: Any, positions: Any) -> Any:
        coordinates = (jax.lax.axis_index("x"),)
        allocator = PIC.PICIdentityAllocator(decomposition, coordinates)
        local_state = D.ParticlePopulationState(
            active,
            jnp.where(active, 1.0, 0.0),
            jnp.where(active, 1, 0).astype(jnp.int32),
            active,
            jnp.zeros_like(active),
            jnp.zeros(active.shape, dtype=jnp.uint32),
            jnp.zeros(active.shape, dtype=jnp.uint32),
            jnp.full(active.shape, 0xFFFFFFFF, dtype=jnp.uint32),
            jnp.full(active.shape, 0xFFFFFFFF, dtype=jnp.uint32),
            state.next_id_hi,
            state.next_id_lo,
        )
        request = D.ParticleAllocationRequest(
            jnp.arange(2), jnp.ones((2,)), jnp.ones((2,), dtype=bool)
        )
        result = allocator.allocate(block, local_state, request, positions[0])
        return (
            result.accepted_state.id_hi,
            result.accepted_state.id_lo,
            result.successful[None],
            result.accepted_state.next_id_lo[None],
        )

    spec = PartitionSpec("x")
    hi, lo, successful, counters = jax.jit(
        jax.shard_map(
            device,
            mesh=mesh,
            in_specs=(spec, spec),
            out_specs=(spec, spec, spec, spec),
            check_vma=False,
        )
    )(jnp.zeros((capacity,), dtype=bool), jnp.asarray(positions))
    assert np.all(np.asarray(successful))
    identity = (np.asarray(hi, dtype=np.uint64) << np.uint64(32)) | np.asarray(
        lo, dtype=np.uint64
    )
    assert np.unique(identity).size == identity.size
    # counter + tile · R + rank with R = global capacity; every counter advances
    # by tiles · R without communication.
    base = int(state.next_id_lo)
    tiles = np.repeat([0, 2, 4, 6], 2)
    np.testing.assert_array_equal(identity, base + tiles * capacity + np.tile([0, 1], 4))
    np.testing.assert_array_equal(np.asarray(counters), base + 8 * capacity)


def test_resampling_on_four_devices_equals_one_device() -> None:
    capacity, steps = 64, 3
    base, species = _reduced_base(32), _reduced_species(capacity)
    merge = PIC.ParticleMergePlan(
        PIC.PICCellBinningPlan((0.0,), (1.0,), (8,), (True,)),
        PIC.PIC_CODE_RELATIVITY,
        species=(0,),
        maximum_per_cell=3,
        momentum_bins=(1, 1, 1),
        minimum_packet_size=3,
        maximum_packet_size=3,
    )
    split = PIC.ParticleSplitPlan(
        PIC.PICCellBinningPlan((0.0,), (1.0,), (8,), (True,)),
        PIC.PIC_CODE_RELATIVITY,
        species=(1,),
        minimum_per_cell=3,
        minimum_child_mass=0.05,
        maximum_splits=8,
    )
    runs = [
        _distributed(base, species, _mesh(shape), processes=(merge, split))
        for shape in ((1,), (4,))
    ]
    dt = 0.2 * float(base.stable_step)
    rng = np.random.default_rng(5)
    electrons = rng.uniform(0.0, 1.0, (capacity, 1))
    ions = rng.uniform(0.0, 1.0, (capacity, 1))
    velocity = np.zeros((capacity, 3))
    velocity[:, 0] = rng.normal(0.0, 0.05, capacity)
    active = np.arange(capacity) < 24
    states = [
        run.initialize(
            (electrons, ions),
            (velocity, np.zeros((capacity, 3))),
            dt,
            active_masks=(active, np.arange(capacity) < 12),
            masses=(active * 1.0, (np.arange(capacity) < 12) * 2.0),
        )
        for run in runs
    ]
    events = 0
    for _ in range(steps):
        results = [
            eqx.filter_jit(lambda value, run=run: run.step_detailed(value, dt))(state)
            for run, state in zip(runs, states, strict=True)
        ]
        for result in results:
            assert bool(result.successful), int(result.rejection_reason)
        single, split_result = results
        for left, right in zip(
            split_result.diagnostics.process_evidence,
            single.diagnostics.process_evidence,
            strict=True,
        ):
            (found,), (wanted,) = left, right
            assert int(found.events) == int(wanted.events)
            assert int(found.created) == int(wanted.created)
            assert int(found.removed) == int(wanted.removed)
            events += int(found.events)
        projection = split_result.diagnostics.gauss_projection
        assert projection is not None and bool(projection.successful)
        states = [result.accepted_state for result in results]
    assert events > 0
    _assert_same_run(states[1], states[0], atol=1e-11)


def test_processes_without_the_distributed_protocol_are_refused() -> None:
    class _Creation(PIC.AbstractPICProcess):
        process_id: str = eqx.field(static=True, default="creation-probe")
        stage: Any = eqx.field(static=True, default="creation")
        stochastic: bool = eqx.field(static=True, default=False)
        radiation_ownership: Any = eqx.field(static=True, default=None)
        species_indices: tuple[int, ...] = eqx.field(static=True, default=(0,))

        def apply(self, species: Any, context: Any, /) -> Any:
            raise NotImplementedError

    base, species = _reduced_base(), _reduced_species(128)
    solver = phx.solver.distribute_pic_field_solver(base, _mesh((4,)))
    pic = phx.solver.ElectromagneticPICPlan(
        solver, species=species, processes=(_Creation(),)
    )
    with pytest.raises(ValueError, match="PICDistributedProcess"):
        phx.solver.DistributedElectromagneticPICPlan(pic, packet_capacity=8)


# -- restart ---------------------------------------------------------------------------


def _publish(
    plan: Any, state: Any, repository: Any, checkpoint_id: str, **options: Any
) -> Any:
    plan.publish(
        repository, state, checkpoint_id=checkpoint_id, writer_id="writer", **options
    )
    return plan.assemble(repository, checkpoint_id, expected_process_count=1)


def _assert_bitwise(actual: Any, expected: Any) -> None:
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_same_topology_restart_continues_bitwise(tmp_path: Path) -> None:
    _, run, dt = _reduced_runs((4,))
    advance = eqx.filter_jit(lambda state: run.step_detailed(state, dt).accepted_state)
    state = _initial(run, 128, 1, (1.0,), dt)
    for _ in range(3):
        state = advance(state)
    plan = phx.solver.PICRestartPlan(run)
    repository = _repository(tmp_path)
    checkpoint = _publish(plan, state, repository, "pic-same")
    restored = plan.restore(repository, checkpoint)
    assert restored.restart_class == "bitwise"
    assert restored.source.topology_id == run.topology_id
    continued, resumed = state, restored.state
    for _ in range(3):
        continued, resumed = advance(continued), advance(resumed)
    _assert_bitwise(resumed, continued)


def test_repartition_restart_is_a_tolerance_restart(tmp_path: Path) -> None:
    reference, run, dt = _reduced_runs((4,))
    state = _initial(run, 128, 1, (1.0,), dt)
    state = eqx.filter_jit(lambda value: run.step_detailed(value, dt).accepted_state)(
        state
    )
    repository = _repository(tmp_path)
    checkpoint = _publish(phx.solver.PICRestartPlan(run), state, repository, "pic-4")
    smaller = _distributed(reference.solver, reference.species, _mesh((2,)))
    restored = phx.solver.PICRestartPlan(smaller).restore(repository, checkpoint)
    assert restored.restart_class == "tolerance"
    assert restored.source.part_count == 4
    # Repartition is a slot permutation: the restored state is exact.
    _assert_same_run(restored.state, state, atol=0.0)
    continued, resumed = state, restored.state
    for _ in range(4):
        continued = run.step_detailed(continued, dt).accepted_state
        resumed = smaller.step_detailed(resumed, dt).accepted_state
    _assert_same_run(resumed, continued, atol=1e-12)
    strict = phx.solver.PICRestartPlan(
        smaller, policy=TopologyRestartPolicy(allow_topology_change=False)
    )
    with pytest.raises(ValueError, match="not admitted"):
        strict.restore(repository, checkpoint)


def test_block_decomposition_restarts_on_slabs(tmp_path: Path) -> None:
    base, species = _reduced_base(8, 2), _reduced_species(64, 2)
    blocks = _distributed(base, species, _mesh((2, 2)))
    slabs = _distributed(base, species, _mesh((4,)))
    dt = 0.4 * float(base.stable_step)
    state = _initial(blocks, 64, 2, (1.0, 1.0), dt)
    advance = eqx.filter_jit(lambda value: blocks.step_detailed(value, dt))
    state = advance(state).accepted_state
    repository = _repository(tmp_path)
    plan = phx.solver.PICRestartPlan(blocks)
    checkpoint = _publish(plan, state, repository, "pic-blocks")
    same = plan.restore(repository, checkpoint)
    assert same.restart_class == "bitwise"
    _assert_bitwise(advance(same.state).accepted_state, advance(state).accepted_state)
    restored = phx.solver.PICRestartPlan(slabs).restore(repository, checkpoint)
    assert restored.restart_class == "tolerance"
    _assert_same_run(restored.state, state, atol=0.0)
    continued, resumed = state, restored.state
    for _ in range(3):
        continued = advance(continued).accepted_state
        resumed = slabs.step_detailed(resumed, dt).accepted_state
    _assert_same_run(resumed, continued, atol=1e-12)


def test_qed_cascade_restarts_bitwise_and_repartitions(
    tables: tuple[Any, Any], tmp_path: Path
) -> None:
    dt = 0.005
    run = _qed_run(tables, _mesh((4,)))
    state, _ = _qed_advance(run, _qed_seed(run, dt), dt, 20)
    repository = _repository(tmp_path)
    plan = phx.solver.PICRestartPlan(run)
    checkpoint = _publish(plan, state, repository, "pic-qed")
    same = plan.restore(repository, checkpoint)
    assert same.restart_class == "bitwise"
    continued, _ = _qed_advance(run, state, dt, 10)
    resumed, _ = _qed_advance(run, same.state, dt, 10)
    _assert_bitwise(resumed, continued)
    smaller = _qed_run(tables, _mesh((2,)))
    restored = phx.solver.PICRestartPlan(smaller).restore(repository, checkpoint)
    assert restored.restart_class == "tolerance"
    _assert_same_cascade(restored.state, state)
    resumed, _ = _qed_advance(smaller, restored.state, dt, 10)
    _assert_same_cascade(resumed, continued)


def test_moving_window_restart_restores_the_window_epoch(tmp_path: Path) -> None:
    base, species = _reduced_base(), _reduced_species(8)
    pic = phx.solver.ElectromagneticPICPlan(base, species=species)
    window = phx.solver.PICMovingWindowPlan(pic, 0)
    dt = 0.5 * float(base.stable_step)
    position = np.linspace(0.05, 0.95, 8)[:, None]
    velocity = np.zeros((8, 3))
    velocity[:, 0] = 0.1
    state = window.initialize(
        pic.initialize((position, position), (velocity, np.zeros((8, 3))), dt)
    )
    state = window.shift(state).accepted_state
    plan = phx.solver.PICRestartPlan(pic, window=window)
    repository = _repository(tmp_path)
    checkpoint = _publish(plan, state, repository, "pic-window")
    restored = plan.restore(repository, checkpoint)
    assert restored.source.window_component
    assert int(restored.state.cumulative_cells) == int(state.cumulative_cells) == 1
    _assert_bitwise(restored.state, state)
    unwindowed = phx.solver.PICRestartPlan(pic)
    with pytest.raises(ValueError, match="another PIC run or inventory"):
        unwindowed.restore(repository, checkpoint)
