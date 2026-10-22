#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Distributed PIC and PIC restart.

Multi-device tests admit physical accelerators or explicitly configured CPU
devices. Eight devices cover slabs, 2×4/4×2 spectral pencils, and the separate
2×2×2 cochain case. Forced-CPU runs are functional evidence only:
``JAX_NUM_CPU_DEVICES=8 pytest tests/unit/solver/test_distributed_pic.py``.
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
            f"Needs {count} devices: run with JAX_NUM_CPU_DEVICES={max(count, 4)}."
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


def _reduced_runs(
    shape: tuple[int, ...], *, count: int = 32, capacity: int = 128
) -> tuple[Any, Any, float]:
    base, species = _reduced_base(count), _reduced_species(capacity)
    reference = phx.solver.ElectromagneticPICPlan(base, species=species)
    run = _distributed(base, species, _mesh(shape))
    return reference, run, 0.5 * float(base.stable_step)


# -- decomposition substrate -----------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "periodic"),
    [
        ((4,), True),
        ((4,), False),
        ((2, 2), True),
        ((2, 2), False),
        ((8,), True),
        ((8,), False),
        ((2, 4), True),
        ((2, 4), False),
        ((4, 2), True),
        ((4, 2), False),
    ],
    ids=[
        "slab-periodic",
        "slab-walled",
        "blocks-periodic",
        "blocks-walled",
        "slab8-periodic",
        "slab8-walled",
        "2x4-periodic",
        "2x4-walled",
        "4x2-periodic",
        "4x2-walled",
    ],
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
    ("reduced", (3,)),
    ("reduced", (5,)),
    ("reduced", (8,)),
    ("cochain", (4,)),
    ("cochain", (3,)),
    ("cochain", (5,)),
    ("cochain", (2, 2)),
    ("cochain", (2, 2, 2)),
    ("spectral-global-fft", (4,)),
    ("spectral-global-fft", (3,)),
    ("spectral-global-fft", (5,)),
    ("spectral-global-fft", (2, 2)),
    ("spectral-global-fft", (8,)),
    ("spectral-global-fft", (2, 4)),
    ("spectral-global-fft", (4, 2)),
    ("spectral-local-guarded", (4,)),
    ("spectral-local-guarded", (3,)),
    ("spectral-local-guarded", (5,)),
    ("spectral-local-guarded", (2, 2)),
    ("spectral-local-guarded", (8,)),
    ("spectral-local-guarded", (2, 4)),
    ("spectral-local-guarded", (4, 2)),
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
    prime = shape in ((3,), (5,))
    parts = math.prod(shape)
    options: dict[str, Any] = {}
    match route:
        case "reduced":
            capacity = 32 * parts if prime else 128
            base = _reduced_base(8 * parts if prime or parts == 8 else 32)
            species = _reduced_species(capacity)
            dimension, lengths, steps = 1, (1.0,), 12
            reference_base = base
        case "cochain":
            # Block windows are local except on the eight-device 2×2×2 mesh,
            # whose 4×2×2 blocks' windows cover the grid.
            counts = {1: (16, 4, 4), 2: (16, 12, 4), 3: (8, 4, 4)}[len(shape)]
            if prime:
                counts = (4 * parts, 4, 4)
            capacity = 16 * parts if prime else 64
            base, species = _cochain_base(capacity, counts)
            options["guard_cells"] = 2
            dimension, steps = 3, 10
            lengths = tuple(0.125 * n for n in counts)
            reference_base = base
        case _:
            capacity = 16 * parts if prime or parts == 8 else 32
            counts = (
                (4 * parts, 4 * parts, 8 if parts == 8 else 4)
                if prime or parts == 8
                else (16, 16, 4)
            )
            decomposition = route.removeprefix("spectral-")
            if len(shape) not in (1, 2):
                raise ValueError(
                    "Spectral PIC qualification requires slab or pencil device axes."
                )
            subdomains = (
                (shape[0], shape[1] if len(shape) == 2 else 1, 1)
                if prime or parts == 8
                else (4, 2, 1)
            )
            base, species = _spectral_base(
                capacity, mesh, decomposition, subdomains=subdomains, counts=counts
            )
            # The oracle owns its FFTs on one device: same grid, transfers,
            # stencil, subdomains, and guards, so the same native numerics.
            reference_base, _ = _spectral_base(
                capacity, None, decomposition, subdomains=subdomains, counts=counts
            )
            assert reference_base.plan.topology.device_count == 1
            assert reference_base.numerical_id == base.numerical_id
            assert reference_base.solver_id != base.solver_id
            dimension, lengths, steps = 3, tuple(0.125 * n for n in counts), 6
    # Local-guarded PSATD keeps Gauss's law only to its stencil truncation; the
    # cochain Coulomb initialization is an iterative solve to ~1e-8.
    tolerance = (
        {"constraint_tolerance": 1e-2}
        if route.endswith("guarded")
        else {"constraint_tolerance": 1e-6}
        if route == "cochain"
        else {}
    )
    reference = phx.solver.ElectromagneticPICPlan(
        reference_base, species=species, **tolerance
    )
    run = _distributed(base, species, mesh, **tolerance, **options)
    assert run.solver.solver_id != base.solver_id
    assert run.pic.plan_id != reference.plan_id
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


def test_local_cubic_migration_science_and_restart(tmp_path: Path) -> None:
    if len(jax.local_devices()) < 8:
        pytest.skip("Needs eight devices on one host.")
    mesh = Mesh(
        np.asarray(jax.local_devices()[:8], dtype=object).reshape((2, 2, 2)),
        ("x", "y", "z"),
    )
    capacity = 128
    base, species = _cochain_base(capacity, (12, 12, 12))
    reference = phx.solver.ElectromagneticPICPlan(
        base, species=species, constraint_tolerance=1e-6
    )
    run = _distributed(
        base,
        species,
        mesh,
        guard_cells=2,
        packet_capacity=16,
        constraint_tolerance=1e-6,
    )
    decomposition = run.solver.decomposition
    for owner in range(8):
        window = decomposition.window_indicator(owner)
        assert np.any(window) and not np.all(window), ("locality", owner)
    # Each owner sends two particles across a face, an edge, and a corner.
    position = np.full((capacity, 3), 0.1, dtype=np.float64)
    velocity = np.zeros((capacity, 3), dtype=np.float64)
    for owner in range(8):
        coordinates = np.asarray(decomposition.coordinates(owner))
        for kind in range(3):
            for duplicate in range(2):
                slot = 6 * owner + 2 * kind + duplicate
                position[slot] = (coordinates + 0.5) * 0.75
                position[slot, : kind + 1] = (coordinates[: kind + 1] + 1) * 0.75 - 0.0001
                velocity[slot, : kind + 1] = 0.2
    active = np.arange(capacity) < 48
    mass = active.astype(np.float64) * 1e-4
    dt = 0.4 * float(base.stable_step)

    def initialize(plan: Any) -> Any:
        return plan.initialize(
            (position, position),
            (velocity, np.zeros_like(velocity)),
            dt,
            active_masks=(active, active),
            masses=(mass, mass),
        )

    state, expected = initialize(run), initialize(reference)
    for value in state.species:
        owners = np.asarray(decomposition.owner(value.particles.position))
        np.testing.assert_array_equal(
            np.bincount(owners[np.asarray(value.population.active)], minlength=8),
            np.full(8, 6),
        )
    _assert_same_run(state, expected, atol=1e-12)
    narrow = phx.solver.DistributedElectromagneticPICPlan(run.pic, packet_capacity=1)
    rejected = eqx.filter_jit(lambda value: narrow.step_detailed(value, dt))(state)
    assert bool(rejected.diagnostics.successful), ("physics", rejected.diagnostics)
    assert not bool(rejected.successful), "resource overflow must reject"
    assert bool(rejected.migration.packet_overflow), ("resource", rejected.migration)
    assert int(rejected.rejection_reason) & int(PIC.PICRejectionReason.MIGRATION)
    _assert_bitwise(rejected.accepted_state, state)
    advance = eqx.filter_jit(lambda value: run.step_detailed(value, dt))
    oracle = eqx.filter_jit(lambda value: reference.step_detailed(value, dt))
    result, single = advance(state), oracle(expected)
    assert bool(single.successful), (
        "reference physics",
        single.diagnostics.rejection_reason,
    )
    assert bool(result.successful), (
        "distributed physics/migration",
        result.rejection_reason,
    )
    assert int(result.migration.migrated) == 48, ("migration", result.migration)
    for shift in ((1, 0, 0), (1, 1, 0), (1, 1, 1)):
        index = result.migration.shifts.index(shift)
        assert int(result.migration.sent[index]) == 2, ("migration", shift)
    _assert_same_run(result.accepted_state, single.accepted_state, atol=1e-11)
    np.testing.assert_allclose(
        result.diagnostics.electric_constraint,
        single.diagnostics.electric_constraint,
        rtol=1e-8,
        atol=1e-10,
    )
    for before, after in zip(state.species, result.accepted_state.species, strict=True):
        original = _by_identity(before.population)
        moved = _by_identity(after.population)
        for name in ("identity", "parent", "mass"):
            np.testing.assert_array_equal(moved[name], original[name])
        owners = np.asarray(decomposition.owner(after.particles.position))
        occupied = np.asarray(after.population.active)
        blocks = np.arange(capacity) // (capacity // 8)
        np.testing.assert_array_equal(owners[occupied], blocks[occupied])
        np.testing.assert_array_equal(
            np.bincount(owners[occupied], minlength=8), np.full(8, 6)
        )
    plan = phx.solver.PICRestartPlan(run)
    repository = _repository(tmp_path)
    checkpoint = _publish(plan, result.accepted_state, repository, "pic-local-cube")
    restored = plan.restore(repository, checkpoint)
    assert restored.restart_class == "bitwise"
    _assert_bitwise(restored.state, result.accepted_state)
    continued, resumed = advance(result.accepted_state), advance(restored.state)
    assert bool(continued.successful), (
        "continuation physics",
        continued.rejection_reason,
    )
    assert bool(resumed.successful), ("restart physics", resumed.rejection_reason)
    _assert_bitwise(resumed.accepted_state, continued.accepted_state)
    reference_next = oracle(single.accepted_state)
    assert bool(reference_next.successful)
    _assert_same_run(continued.accepted_state, reference_next.accepted_state, atol=1e-11)


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


@pytest.mark.parametrize("parts", [3, 4, 5, 8])
def test_migration_overflow_rejects_the_whole_step(parts: int) -> None:
    capacity = 32 * parts
    _, run, dt = _reduced_runs((parts,), count=8 * parts, capacity=capacity)
    # Twelve electrons just below the slab-0/slab-1 face moving across it.
    position = np.full((capacity, 1), 0.1)
    position[:12, 0] = 1.0 / parts - 0.0005
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


@pytest.mark.parametrize("shape", [(2, 2), (3,), (5,)], ids=["2x2", "3", "5"])
def test_deposits_that_are_not_window_local_are_refused(
    shape: tuple[int, ...],
) -> None:
    # The continuity-projected reduced 2-D current spreads over the whole grid.
    parts = math.prod(shape)
    base, species = _reduced_base(4 * parts, 2), _reduced_species(16 * parts, 2)
    solver = phx.solver.distribute_pic_field_solver(base, _mesh(shape))
    with pytest.raises(ValueError, match="not window-local") as error:
        phx.solver.ElectromagneticPICPlan(solver, species=species)
    assert "transfer footprint of species 0" in str(error.value)
    assert f"block {(0,) * len(shape)}" in str(error.value)


@pytest.mark.parametrize("parts", [3, 5])
def test_prime_device_counts_refuse_unequal_species_capacity(parts: int) -> None:
    mesh = _mesh((parts,))
    base = _reduced_base(8 * parts)
    with pytest.raises(ValueError, match="capacities must divide"):
        _distributed(base, _reduced_species(32 * parts + 1), mesh)


def test_size_one_mesh_axes_are_refused() -> None:
    base, _ = _cochain_base(8)
    with pytest.raises(ValueError, match="size-one"):
        phx.solver.distribute_pic_field_solver(base, _mesh((2, 1)))


@pytest.mark.parametrize("parts", [3, 4, 5, 8])
def test_guard_narrower_than_the_transfer_footprint_is_refused(parts: int) -> None:
    base, species = _reduced_base(8 * parts), _reduced_species(32 * parts)
    solver = phx.solver.distribute_pic_field_solver(
        base, _mesh((parts,)), guard_cells=1, particle_margin=0.9
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
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        assert np.asarray(left).tobytes() == np.asarray(right).tobytes()


def test_reduced_deposit_has_deterministic_route_slot_order() -> None:
    transfer = _reduced_base().transfer
    # All slots collide at one cell center. Slot-order accumulation is one,
    # unlike reassociated/atomic cancellation of these exactly represented values.
    position = jnp.full((128, 1), 0.5 / 32, dtype=jnp.float64)
    content = jnp.tile(
        jnp.asarray([2.0**54, 1.0, -(2.0**54), 1.0], dtype=jnp.float64), 32
    )
    active = jnp.ones((128,), dtype=jnp.bool_)
    deposit = eqx.filter_jit(transfer.deposit)
    expected = np.zeros((32,), dtype=np.float64)
    expected[0] = 32.0
    np.testing.assert_array_equal(
        np.asarray(deposit(position, content, active)), expected
    )


@pytest.mark.parametrize("invalid_active", [True, False], ids=["active", "inactive"])
def test_reduced_deposit_preserves_failure_and_inactive_payload_semantics(
    invalid_active: bool,
) -> None:
    transfer = _reduced_base().transfer
    position = jnp.full((2, 1), 0.5 / 32, dtype=jnp.float64)
    content = jnp.asarray([jnp.inf, 1.0], dtype=jnp.float64)
    active = jnp.asarray([invalid_active, True], dtype=jnp.bool_)
    result = np.asarray(eqx.filter_jit(transfer.deposit)(position, content, active))
    if invalid_active:
        assert np.all(np.isnan(result))
    else:
        expected = np.zeros((32,), dtype=np.float64)
        expected[0] = 32.0
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("parts", [3, 4, 5, 8])
def test_same_topology_restart_continues_bitwise(tmp_path: Path, parts: int) -> None:
    capacity = 32 * parts
    _, run, dt = _reduced_runs((parts,), count=8 * parts, capacity=capacity)
    advance = eqx.filter_jit(lambda state: run.step_detailed(state, dt))
    state = _initial(run, capacity, 1, (1.0,), dt)
    for _ in range(3):
        result = advance(state)
        assert bool(result.successful), int(result.rejection_reason)
        state = result.accepted_state
    plan = phx.solver.PICRestartPlan(run)
    repository = _repository(tmp_path)
    checkpoint = _publish(plan, state, repository, "pic-same")
    restored = plan.restore(repository, checkpoint)
    assert restored.restart_class == "bitwise"
    assert restored.source.topology_id == run.topology_id
    assert restored.source.plan_id == run.plan_id
    assert restored.source.numerical_id == run.numerical_id
    # Checkpoint transport must preserve both bits and the numerical input type,
    # independently of the continuation reduction contract.
    _assert_bitwise(restored.state, state)
    for left, right in zip(
        jax.tree.leaves(restored.state), jax.tree.leaves(state), strict=True
    ):
        assert left.dtype == right.dtype
        assert left.weak_type == right.weak_type
        assert left.sharding.is_equivalent_to(right.sharding, left.ndim)
    continued, resumed = state, restored.state
    for _ in range(3):
        continued_step, resumed_step = advance(continued), advance(resumed)
        assert bool(continued_step.successful), int(continued_step.rejection_reason)
        assert bool(resumed_step.successful), int(resumed_step.rejection_reason)
        continued, resumed = continued_step.accepted_state, resumed_step.accepted_state
        _assert_bitwise(resumed, continued)


@pytest.mark.parametrize(
    ("source_parts", "target_parts"),
    [(4, 2), (4, 5), (5, 4), (4, 8), (8, 4)],
    ids=["4-to-2", "4-to-5", "5-to-4", "4-to-8", "8-to-4"],
)
def test_repartition_restart_is_a_tolerance_restart(
    tmp_path: Path, source_parts: int, target_parts: int
) -> None:
    common = math.lcm(source_parts, target_parts)
    capacity = max(128, 8 * common)
    count = max(32, 2 * common)
    reference, run, dt = _reduced_runs((source_parts,), count=count, capacity=capacity)
    _mesh((target_parts,))
    state = _initial(run, capacity, 1, (1.0,), dt)
    initial_step = eqx.filter_jit(lambda value: run.step_detailed(value, dt))(state)
    assert bool(initial_step.successful), int(initial_step.rejection_reason)
    state = initial_step.accepted_state
    repository = _repository(tmp_path)
    checkpoint = _publish(
        phx.solver.PICRestartPlan(run), state, repository, f"pic-{source_parts}"
    )
    smaller = _distributed(reference.solver, reference.species, _mesh((target_parts,)))
    with pytest.raises(ValueError, match="another plan"):
        smaller.pic.restore(run.pic.checkpoint(state))
    restored = phx.solver.PICRestartPlan(smaller).restore(repository, checkpoint)
    assert restored.restart_class == "tolerance"
    assert restored.source.plan_id == run.plan_id
    assert restored.source.numerical_id == smaller.numerical_id
    assert restored.source.plan_id != smaller.plan_id
    assert restored.source.part_count == source_parts
    # Repartition is a slot permutation: the restored state is exact.
    _assert_same_run(restored.state, state, atol=0.0)
    continued, resumed = state, restored.state
    for _ in range(4):
        source_step = run.step_detailed(continued, dt)
        target_step = smaller.step_detailed(resumed, dt)
        assert bool(source_step.successful), int(source_step.rejection_reason)
        assert bool(target_step.successful), int(target_step.rejection_reason)
        continued = source_step.accepted_state
        resumed = target_step.accepted_state
    _assert_same_run(resumed, continued, atol=1e-12)
    strict = phx.solver.PICRestartPlan(
        smaller, policy=TopologyRestartPolicy(allow_topology_change=False)
    )
    with pytest.raises(ValueError, match="not admitted"):
        strict.restore(repository, checkpoint)


@pytest.mark.parametrize(
    "shape", [(8,), (2, 4), (4, 2)], ids=["slab8", "pencil2x4", "pencil4x2"]
)
@pytest.mark.parametrize("decomposition", ["global-fft", "local-guarded"])
def test_eight_owner_spectral_restart_continues_bitwise(
    tmp_path: Path, shape: tuple[int, ...], decomposition: str
) -> None:
    mesh = _mesh(shape)
    counts = (32, 32, 8)
    capacity = 128
    if len(shape) not in (1, 2):
        raise ValueError(
            "Spectral PIC restart qualification requires slab or pencil device axes."
        )
    subdomains = (shape[0], shape[1] if len(shape) == 2 else 1, 1)
    base, species = _spectral_base(
        capacity, mesh, decomposition, subdomains=subdomains, counts=counts
    )
    tolerance = {"constraint_tolerance": 1e-2} if decomposition == "local-guarded" else {}
    run = _distributed(base, species, mesh, **tolerance)
    dt = 0.4 * float(base.stable_step)
    state = _initial(run, capacity, 3, tuple(0.125 * n for n in counts), dt)
    advance = eqx.filter_jit(lambda value: run.step_detailed(value, dt))
    initial = advance(state)
    assert bool(initial.successful), int(initial.rejection_reason)
    state = initial.accepted_state
    plan = phx.solver.PICRestartPlan(run)
    repository = _repository(tmp_path)
    checkpoint = _publish(plan, state, repository, "pic-spectral-eight")
    restored = plan.restore(repository, checkpoint)
    assert restored.restart_class == "bitwise"
    assert restored.source.plan_id == run.plan_id
    assert restored.source.numerical_id == run.numerical_id
    assert restored.source.topology_id == run.topology_id
    _assert_bitwise(restored.state, state)
    continued, resumed = state, restored.state
    for _ in range(3):
        continued_step, resumed_step = advance(continued), advance(resumed)
        assert bool(continued_step.successful), int(continued_step.rejection_reason)
        assert bool(resumed_step.successful), int(resumed_step.rejection_reason)
        continued, resumed = continued_step.accepted_state, resumed_step.accepted_state
        _assert_bitwise(resumed, continued)


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
