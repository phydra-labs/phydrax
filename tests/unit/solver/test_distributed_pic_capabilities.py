#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Capability preservation of PIC field solvers and their distributed wrapper.

Single-process tests check each configuration's declared optional-protocol
matrix against the structural protocols the PIC runtime checks, the declared
distribution refusals, and the consumers' refusals of withheld protocols (a
one-device mesh builds every distributed wrapper). Every protocol a distributed
wrapper publishes is exercised numerically on four forced host devices in a
subprocess (``XLA_FLAGS=--xla_force_host_platform_device_count=4``) against the
single-device base solver or an analytic reference.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import phydrax as phx


D = phx.discretization
PIC = D.pic
S = phx.solver
sp = phx.solver.maxwell.spectral
mx = phx.solver.maxwell
H = 0.125

# The protocol each capability name denotes, stated independently of the package.
_PROTOCOLS: dict[str, type[object]] = {
    "tensor-layout": S.PICTensorLayout,
    "spectral-symbol": S.PICSpectralSymbol,
    "huygens-sampling": S.PICHuygensSampling,
    "multi-deposit": S.PICMultiDeposit,
    "window-shift": S.PICWindowShift,
    "galilean-grid": S.PICGalileanGrid,
    "energy-accounting": S.PICEnergyAccounting,
    "open-domain": S.PICOpenDomain,
    "restart-state": S.PICRestartState,
    "gauss-projection": S.PICGaussProjection,
    "relativistic-self-fields": S.PICRelativisticSelfFields,
}


# -- construction -------------------------------------------------------------------


def _species(capacity: int, dimension: int) -> tuple[Any, ...]:
    plans = []
    for offset, sign, name in ((0, -1.0, "electrons"), (10_000, 1.0, "ions")):
        support = D.ParticleSetPlan(
            jnp.arange(offset, offset + capacity),
            jnp.ones((capacity,)),
            ambient_dimension=dimension,
        ).prepare()
        plans.append(
            PIC.PICSpeciesPlan(
                D.ParticlePopulationPlan(support),
                PIC.PICChargeModelPlan(
                    sign,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    return tuple(plans)


def _bridge(counts: tuple[int, int, int]) -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(n, periodic=True) for n in counts),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [n * H for n in counts]]))
    return D.StructuredCochainBridge(grid)


def _transfers(
    bridge: Any, species: tuple[Any, ...], shape_order: PIC.PICShapeOrder
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    plan = PIC.PICParticleCochainTransferPlan(bridge, shape_order=shape_order)
    transfers = tuple(
        plan.prepare(
            D.ChargedParticlePlan(
                value.charge_model.base_specific_charge * jnp.ones((value.capacity,)),
                value.species_id,
            ).prepare(value.population.particles)
        )
        for value in species
    )
    return transfers, tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers)


def _cochain(
    bridge: Any, species: tuple[Any, ...], observers: tuple[Any, ...] = ()
) -> Any:
    transfers, currents = _transfers(bridge, species, 1)
    maxwell = S.CompatibleMaxwellPlan(
        bridge,
        observers=observers,
        sources=(S.PICMaxwellCurrentSourcePlan(),),
        plan_id="capability-test",
    ).prepare()
    electrostatic = S.CochainElectrostaticPlan(
        bridge, S.CochainElectrostaticBoundaryPlan.periodic(bridge)
    )
    return S.CochainMaxwellPICFieldSolver(maxwell, electrostatic, transfers, currents)


def _spectral(
    bridge: Any,
    species: tuple[Any, ...],
    mesh: Mesh | None,
    decomposition: str,
    **options: Any,
) -> Any:
    transfers, currents = _transfers(bridge, species, 2)
    settings: dict[str, Any] = {"grid": "staggered", "decomposition": decomposition}
    if decomposition == "local-guarded":
        settings |= {
            "charge_conservation": "vay-deposition",
            "stencil": "finite-order",
            "stencil_order": 4,
            "subdomains": (4, 2, 1),
            "guard_cells": (3, 3, 3),
        }
    if mesh is not None:
        settings["topology"] = D.spectral.SpectralMeshTopology(mesh)
    return sp.SpectralMaxwellPlan(bridge, **(settings | options)).prepare(
        transfers, currents
    )


def _reduced(count: int, dimension: int) -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(count, periodic=True) for _ in range(dimension)),
        axis_names=("x", "y")[:dimension],
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    field = (
        S.CompatibleMaxwell1DPlan(grid)
        if dimension == 1
        else S.CompatibleMaxwell2DPlan(grid)
    )
    return S.ReducedMaxwellPICFieldSolver(field, PIC.ReducedPICTransferPlan(grid))


def _tetrahedral() -> Any:
    coordinates = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        + ((0.25, 0.25, 0.25),)
    )
    cells = jnp.asarray(((4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4)))
    mesh = D.CellMesh(coordinates, (D.CellBlock("tet", "tetrahedron", cells),))
    element = D.FiniteElementPlan(
        mesh, D.FiniteElementFieldSpec("u", D.lagrange_element("tetrahedron", 1))
    ).prepare()
    locator = D.PreparedSimplicialCellLocator(
        D.fem.prepare_finite_element_cell_map(element, 0),
        element.default_runtime.coordinates,
        D.SimplicialLocationPolicy(4, 8, 4),
    )
    complex_ = D.FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)
    maxwell = mx.UnstructuredMaxwellPlan(
        complex_,
        mx.DiagonalMaxwellConstitutivePlan(),
        spectral_upper_bound=100.0,
        courant_factor=0.9,
        boundary="relative",
    ).prepare()
    return S.UnstructuredMaxwellPICFieldSolver(
        maxwell, PIC.UnstructuredWhitneyCurrentPlan(locator, maximum_segments=4)
    )


def _quasi_cylindrical() -> Any:
    grid = D.pic.QuasiCylindricalGrid(2.0, 16, 0.0, 4.0, 16, 2)
    return sp.QuasiCylindricalMaxwellPlan(grid).prepare()


def _base(configuration: str) -> Any:
    match configuration:
        case "cochain-3d":
            return _cochain(_bridge((8, 4, 4)), _species(8, 3))
        case "reduced-1d":
            return _reduced(32, 1)
        case "reduced-2d":
            return _reduced(16, 2)
        case "unstructured-whitney":
            return _tetrahedral()
        case "psatd-global-fft":
            return _spectral(_bridge((8, 4, 4)), _species(8, 3), None, "global-fft")
        case "psatd-local-guarded":
            return _spectral(_bridge((8, 4, 4)), _species(8, 3), None, "local-guarded")
        case "quasi-cylindrical-psatd":
            return _quasi_cylindrical()
        case _:
            raise ValueError(configuration)


def _one_device_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1], dtype=object).reshape(1), ("x",))


# -- declared matrix ----------------------------------------------------------------

_SHARED = ("restart-state", "gauss-projection")
_BASE_ADMITTED = {
    "cochain-3d": {
        "tensor-layout",
        "spectral-symbol",
        "window-shift",
        "energy-accounting",
        "open-domain",
        *_SHARED,
    },
    "reduced-1d": {
        "tensor-layout",
        "spectral-symbol",
        "multi-deposit",
        "window-shift",
        *_SHARED,
    },
    "reduced-2d": {
        "tensor-layout",
        "spectral-symbol",
        "multi-deposit",
        "window-shift",
        *_SHARED,
    },
    "unstructured-whitney": set(_SHARED),
    "psatd-global-fft": {"spectral-symbol", "multi-deposit", *_SHARED},
    "psatd-local-guarded": {"spectral-symbol", "multi-deposit", *_SHARED},
    "quasi-cylindrical-psatd": {"spectral-symbol", "window-shift", *_SHARED},
}
# The periodic cochain publishes boosted-Coulomb self-fields but its gauged
# periodic electrostatic boundary refuses them; it publishes Huygens sampling,
# but Maxwell refuses Huygens boxes beside the dynamic PIC current. Standard
# PSATD without observers publishes Huygens sampling and the Galilean grid but
# has no phasors to sample and a lab-fixed grid.
_SPECTRAL_REFUSED = {"huygens-sampling", "galilean-grid"}
_BASE_PUBLISHED_ONLY = {
    "cochain-3d": {"relativistic-self-fields", "huygens-sampling"},
    "psatd-global-fft": _SPECTRAL_REFUSED,
    "psatd-local-guarded": _SPECTRAL_REFUSED,
    "quasi-cylindrical-psatd": _SPECTRAL_REFUSED,
}
_DISTRIBUTED_ADMITTED = {
    "cochain-3d": {
        "tensor-layout",
        "spectral-symbol",
        "multi-deposit",
        "energy-accounting",
        "open-domain",
        *_SHARED,
    },
    "reduced-1d": {"tensor-layout", "spectral-symbol", "multi-deposit", *_SHARED},
    "reduced-2d": {"tensor-layout", "spectral-symbol", "multi-deposit", *_SHARED},
    "psatd-global-fft": _BASE_ADMITTED["psatd-global-fft"],
    "psatd-local-guarded": _BASE_ADMITTED["psatd-local-guarded"],
}
_WITHHELD = {
    "cochain-3d": {"window-shift", "relativistic-self-fields", "huygens-sampling"},
    "reduced-1d": {"window-shift"},
    "reduced-2d": {"window-shift"},
}
# Forwarded base protocols keep the base refusal.
_DISTRIBUTED_PUBLISHED_ONLY = {
    "psatd-global-fft": _SPECTRAL_REFUSED,
    "psatd-local-guarded": _SPECTRAL_REFUSED,
}


def _structural(solver: Any) -> set[str]:
    return {name for name, protocol in _PROTOCOLS.items() if isinstance(solver, protocol)}


@pytest.mark.parametrize("configuration", sorted(_BASE_ADMITTED))
def test_base_matrix_is_the_structural_protocol_set(configuration: str) -> None:
    solver = _base(configuration)
    capabilities = solver.pic_capabilities
    assert capabilities.configuration == configuration
    assert tuple(record.capability for record in capabilities.records) == tuple(
        _PROTOCOLS
    )
    assert set(capabilities.published) == _structural(solver)
    assert set(capabilities.admitted) == _BASE_ADMITTED[configuration]
    assert set(capabilities.published) - set(capabilities.admitted) == (
        _BASE_PUBLISHED_ONLY.get(configuration, set())
    )
    assert all(record.basis for record in capabilities.records)


def test_published_but_refused_protocol_refuses_when_called() -> None:
    solver = _base("cochain-3d")
    record = solver.pic_capabilities.record("relativistic-self-fields")
    assert record.published and not record.admitted
    charge = jnp.zeros((solver.bridge.cochain.cell_counts[0],))
    with pytest.raises(ValueError, match="grounded"):
        solver.initialize_relativistic_field((charge,), ((0.0, 0.0, 0.5),), 1.0)
    # A grounded bounded box admits the same protocol.
    grounded = _bounded_cochain().pic_capabilities.record("relativistic-self-fields")
    assert grounded.published and grounded.admitted


def test_cochain_huygens_sampling_is_refused_beside_pic_current() -> None:
    solver = _base("cochain-3d")
    record = solver.pic_capabilities.record("huygens-sampling")
    assert record.published and not record.admitted
    charge = jnp.zeros((solver.bridge.cochain.cell_counts[0],))
    with pytest.raises(ValueError) as refusal:
        solver.huygens_phasors(solver.field_with_charge(charge))
    assert str(refusal.value) == record.basis
    bridge = _bridge((16, 16, 8))
    with pytest.raises(ValueError, match="Huygens surface"):
        _cochain(bridge, _species(8, 3), (_huygens_box((2, 2, 1), (14, 14, 6), bridge),))


def _field(solver: Any) -> Any:
    if isinstance(solver, sp.PreparedQuasiCylindricalMaxwell):
        grid = solver.grid
        return solver.field_with_charge(
            jnp.zeros((grid.mode_count, grid.radial_count, grid.axial_count))
        )
    return solver.field_with_charge(jnp.zeros(solver.plan.counts))


@pytest.mark.parametrize(
    "configuration",
    [
        "psatd-global-fft",
        "psatd-local-guarded",
        "quasi-cylindrical-psatd",
        "distributed-psatd-global-fft",
    ],
)
def test_observerless_standard_spectral_protocols_refuse_when_called(
    configuration: str,
) -> None:
    base = _base(configuration.removeprefix("distributed-"))
    solver: Any = (
        S.distribute_pic_field_solver(base, _one_device_mesh())
        if configuration.startswith("distributed-")
        else base
    )
    capabilities = solver.pic_capabilities
    huygens = capabilities.record("huygens-sampling")
    galilean = capabilities.record("galilean-grid")
    assert huygens.published and not huygens.admitted
    assert galilean.published and not galilean.admitted
    with pytest.raises(ValueError) as refusal:
        solver.huygens_phasors(_field(base))
    assert str(refusal.value) in huygens.basis
    with pytest.raises(ValueError) as refusal:
        solver.grid_velocity
    assert str(refusal.value) in galilean.basis


def test_spectral_observers_admit_huygens_sampling() -> None:
    bridge = _bridge((16, 16, 8))
    solver = _spectral(
        bridge,
        _species(8, 3),
        None,
        "global-fft",
        observers=(_huygens_box((2, 2, 1), (14, 14, 6), None),),
    )
    assert solver.pic_capabilities.record("huygens-sampling").admitted
    (phasors,) = solver.huygens_phasors(_field(solver))
    # One observer at the acquisition's frequencies, with no field sampled yet.
    np.testing.assert_array_equal(np.asarray(phasors.angular_frequencies), [4.0, 8.0])
    assert not np.any(np.asarray(phasors.electric))
    assert not np.any(np.asarray(phasors.magnetic))


@pytest.mark.parametrize(
    ("options", "grid_velocity"),
    [
        ({}, (0.0, 0.0, 0.0)),
        (
            {
                "variant": "galilean",
                "galilean_velocity": (0.5, 0.0, 0.0),
                "charge_conservation": "update-with-rho",
            },
            (0.5, 0.0, 0.0),
        ),
    ],
    ids=["standard", "galilean"],
)
def test_runtime_drifts_particles_by_the_admitted_grid_velocity(
    options: dict[str, Any], grid_velocity: tuple[float, ...]
) -> None:
    bridge = _bridge((8, 4, 4))
    species = _species(8, 3)
    solver = _spectral(bridge, species, None, "global-fft", **options)
    assert solver.pic_capabilities.record("galilean-grid").admitted == bool(options)
    plan = S.ElectromagneticPICPlan(solver, species=species)
    rng = np.random.default_rng(5)
    position = rng.uniform((0.3, 0.1, 0.1), (0.7, 0.4, 0.4), (8, 3))
    velocity = rng.uniform(-0.2, 0.2, (8, 3))
    # Co-located electrons and ions with equal velocities: no charge, no
    # current, so the fields stay zero and every particle coasts.
    dt = 0.4 * float(solver.stable_step)
    state = plan.initialize(
        (position, position),
        (velocity, velocity),
        dt,
        masses=(np.ones(8), np.ones(8)),
    )
    result = plan.step_detailed(state, dt)
    assert bool(result.successful)
    expected = position + (velocity - np.asarray(grid_velocity)) * dt
    for value in result.accepted_state.species:
        np.testing.assert_allclose(
            np.asarray(value.particles.position), expected, rtol=0.0, atol=1e-14
        )


@pytest.mark.parametrize("configuration", sorted(_DISTRIBUTED_ADMITTED))
def test_distributed_wrapper_publishes_exactly_its_executed_routes(
    configuration: str,
) -> None:
    base = _base(configuration)
    support = S.pic_distribution_support(base)
    assert support.configuration == configuration and support.route is not None
    solver = S.distribute_pic_field_solver(base, _one_device_mesh())
    capabilities = solver.pic_capabilities
    assert capabilities.configuration == f"distributed-{configuration}"
    assert set(capabilities.published) == _structural(solver)
    # A distributed route is published only if this configuration executes it
    # or forwards the base refusal of it.
    assert set(capabilities.published) - set(capabilities.admitted) == (
        _DISTRIBUTED_PUBLISHED_ONLY.get(configuration, set())
    )
    assert set(capabilities.admitted) == _DISTRIBUTED_ADMITTED[configuration]
    base_capabilities = base.pic_capabilities
    withheld = {
        record.capability
        for record in capabilities.records
        if record.basis.startswith("Withheld by distribution")
    }
    assert withheld == _WITHHELD.get(configuration, set())
    # No base protocol disappears without a stated distribution refusal.
    assert set(base_capabilities.published) <= set(capabilities.published) | withheld


def _bounded_cochain() -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(n) for n in (8, 4, 4)), axis_names=("x", "y", "z")
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.5, 0.5]]))
    bridge = D.StructuredCochainBridge(grid)
    transfers, currents = _transfers(bridge, _species(8, 3), 1)
    maxwell = S.CompatibleMaxwellPlan(
        bridge, sources=(S.PICMaxwellCurrentSourcePlan(),)
    ).prepare()
    electrostatic = S.CochainElectrostaticPlan(
        bridge, S.CochainElectrostaticBoundaryPlan.dirichlet(bridge)
    )
    return S.CochainMaxwellPICFieldSolver(maxwell, electrostatic, transfers, currents)


@pytest.mark.parametrize(
    ("builder", "reason"),
    [
        (_quasi_cylindrical, "Hankel"),
        (_tetrahedral, "tetrahedral"),
        (_bounded_cochain, "periodic axes"),
    ],
    ids=["quasi-cylindrical", "unstructured-whitney", "bounded-cochain"],
)
def test_distribution_refusals_are_declared_and_enforced(
    builder: Callable[[], Any], reason: str
) -> None:
    base = builder()
    support = S.pic_distribution_support(base)
    assert support.route is None and reason in support.basis
    with pytest.raises(ValueError) as refusal:
        S.distribute_pic_field_solver(base, _one_device_mesh())
    assert str(refusal.value) == support.basis


def test_withheld_protocols_are_refused_by_their_consumers() -> None:
    mesh = _one_device_mesh()
    reduced = S.distribute_pic_field_solver(_reduced(32, 1), mesh)
    pic = S.ElectromagneticPICPlan(reduced, species=_species(8, 1))
    with pytest.raises(TypeError, match="PICWindowShift"):
        S.PICMovingWindowPlan(pic, 0)
    bridge = _bridge((8, 4, 4))
    species = _species(8, 3)
    cochain = S.distribute_pic_field_solver(_cochain(bridge, species), mesh)
    run = S.DistributedElectromagneticPICPlan(
        S.ElectromagneticPICPlan(cochain, species=species), packet_capacity=8
    )
    position = np.full((8, 3), 0.3)
    with pytest.raises(TypeError, match="PICRelativisticSelfFields"):
        run.pic.initialize(
            (position, position),
            (np.zeros((8, 3)), np.zeros((8, 3))),
            0.01,
            self_fields="relativistic-per-species",
        )
    spectral = S.distribute_pic_field_solver(
        _spectral(bridge, species, None, "global-fft"), mesh
    )
    with pytest.raises(TypeError, match="PICTensorLayout"):
        S.ElectromagneticPICPlan(spectral, species=species, filters=(S.PICFilterPlan(),))


# -- multi-device scenarios -----------------------------------------------------------


def _four_device_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape(4), ("x",))


def _initial(
    run: Any,
    capacity: int,
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    dt: float,
    *,
    active_count: int | None = None,
    magnetic: Any = None,
) -> Any:
    """A quarter of the slots active (or ``active_count``), uniform in the box."""
    rng = np.random.default_rng(7)
    dimension = len(lower)
    count = capacity // 4 if active_count is None else active_count
    active = np.arange(capacity) < count
    velocity = np.zeros((capacity, 3))
    velocity[:, :dimension] = rng.uniform(-0.2, 0.2, (capacity, dimension))
    positions = tuple(rng.uniform(lower, upper, (capacity, dimension)) for _ in range(2))
    return run.initialize(
        positions,
        (velocity, np.zeros((capacity, 3))),
        dt,
        active_masks=(active, active),
        masses=(active * 1.0, active * 1.0),
        magnetic=magnetic,
    )


def _by_identity(species: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    population = species.population
    active = np.asarray(population.active)
    identity = (np.asarray(population.id_hi, dtype=np.uint64) << np.uint64(32)) | (
        np.asarray(population.id_lo, dtype=np.uint64)
    )
    order = np.argsort(identity[active])
    return (
        identity[active][order],
        np.asarray(species.particles.position)[active][order],
        np.asarray(species.particles.proper_velocity)[active][order],
    )


def _assert_same_state(actual: Any, expected: Any, atol: float) -> None:
    for left, right in zip(
        jax.tree.leaves(actual.field), jax.tree.leaves(expected.field), strict=True
    ):
        scale = max(1.0, float(np.max(np.abs(np.asarray(right)), initial=0.0)))
        np.testing.assert_allclose(
            np.asarray(left), np.asarray(right), rtol=0.0, atol=atol * scale
        )
    for left, right in zip(actual.species, expected.species, strict=True):
        found, wanted = _by_identity(left), _by_identity(right)
        np.testing.assert_array_equal(found[0], wanted[0])
        np.testing.assert_allclose(found[1], wanted[1], rtol=0.0, atol=atol)
        np.testing.assert_allclose(found[2], wanted[2], rtol=0.0, atol=atol)


def _paired_run(
    base: Any,
    species: tuple[Any, ...],
    box: tuple[tuple[float, ...], tuple[float, ...]],
    steps: int,
    *,
    initial: dict[str, Any] | None = None,
    **options: Any,
) -> tuple[Any, Any, list[tuple[Any, Any]]]:
    """Single-device and four-device runs of one declaration, compared per step."""
    reference = S.ElectromagneticPICPlan(base, species=species, **options)
    run = S.DistributedElectromagneticPICPlan(
        S.ElectromagneticPICPlan(
            S.distribute_pic_field_solver(base, _four_device_mesh()),
            species=species,
            **options,
        ),
        packet_capacity=32,
    )
    capacity = species[0].capacity
    dt = 0.4 * float(base.stable_step)
    settings = {} if initial is None else initial
    expected = _initial(reference, capacity, *box, dt, **settings)
    actual = _initial(run, capacity, *box, dt, **settings)
    advance = eqx.filter_jit(lambda state: reference.step_detailed(state, dt))
    distributed = eqx.filter_jit(lambda state: run.step_detailed(state, dt))
    results = []
    for _ in range(steps):
        single, split = advance(expected), distributed(actual)
        assert bool(single.successful)
        assert bool(split.successful), int(split.rejection_reason)
        results.append((single, split))
        expected, actual = single.accepted_state, split.accepted_state
    _assert_same_state(actual, expected, 1e-11)
    return reference, run, results


def _scenario_filters() -> None:
    """Tensor layouts: filtered distributed runs equal filtered single-device runs."""
    filters = (S.PICFilterPlan(),)
    reduced = _reduced(32, 1)
    _paired_run(reduced, _species(128, 1), ((0.0,), (1.0,)), 5, filters=filters)
    bridge = _bridge((16, 4, 4))
    species = _species(64, 3)
    cochain = _cochain(bridge, species)
    _, run, results = _paired_run(
        cochain,
        species,
        ((0.0, 0.0, 0.0), (2.0, 0.5, 0.5)),
        3,
        filters=filters,
        constraint_tolerance=1e-6,
    )
    # The filter is exercised: the accepted field differs from an unfiltered run's.
    unfiltered = S.DistributedElectromagneticPICPlan(
        S.ElectromagneticPICPlan(
            S.distribute_pic_field_solver(cochain, _four_device_mesh()),
            species=species,
            constraint_tolerance=1e-6,
        ),
        packet_capacity=32,
    )
    dt = 0.4 * float(cochain.stable_step)
    state = _initial(unfiltered, 64, (0.0, 0.0, 0.0), (2.0, 0.5, 0.5), dt)
    plain = unfiltered.step_detailed(state, dt).accepted_state
    filtered = run.step_detailed(
        _initial(run, 64, (0.0, 0.0, 0.0), (2.0, 0.5, 0.5), dt), dt
    ).accepted_state
    assert not np.allclose(
        np.asarray(plain.field.primary.electric_displacement),
        np.asarray(filtered.field.primary.electric_displacement),
        rtol=0.0,
        atol=1e-12,
    )
    del results


def _scenario_cochain_ledger() -> None:
    """Open domain, energy accounting, and the wrapper-fused multi-deposit."""
    bridge = _bridge((16, 4, 4))
    species = _species(64, 3)
    cochain = _cochain(bridge, species)
    lower, upper = (0.0, 0.0, 0.0), (2.0, 0.5, 0.5)
    periodic = PIC.PICOpenBoundaryPlan(
        jnp.asarray(lower), jnp.asarray(upper), kinds=(PIC.PICBoundaryKind.PERIODIC,) * 6
    )
    _, run, results = _paired_run(
        cochain,
        species,
        (lower, upper),
        4,
        boundaries=periodic,
        constraint_tolerance=1e-6,
    )
    for single, split in results:
        wanted, found = single.diagnostics.energy, split.diagnostics.energy
        assert found.material is not None and found.dissipated is not None
        for name in ("electric_field", "magnetic_field", "material", "total", "defect"):
            np.testing.assert_allclose(
                float(getattr(found, name)),
                float(getattr(wanted, name)),
                rtol=1e-9,
                atol=1e-13,
            )
        # Leapfrog-corrected split, not the total field energy in one slot.
        assert float(found.magnetic_field) != 0.0
    solver = run.solver
    assert solver.domain_periodic == (True, True, True)
    # Fused deposit equals the base solver's per-species deposits.
    rng = np.random.default_rng(3)
    # Slot block p (16 slots per device) lies in device p's 4-cell x slab.
    offset = np.zeros((64, 3))
    offset[:, 0] = 0.5 * (np.arange(64) // 16)
    starts = tuple(
        jnp.asarray(offset + rng.uniform((0.05, 0.0, 0.0), (0.45, 0.5, 0.5), (64, 3)))
        for _ in species
    )
    ends = tuple(value + 0.01 for value in starts)
    velocity = tuple(jnp.full((64, 3), 0.01) / 0.02 for _ in species)
    charge = tuple(jnp.full((64,), sign) for sign in (-1.0, 1.0))
    active = tuple(jnp.ones((64,), dtype=jnp.bool_) for _ in species)
    step = jnp.asarray(0.02)
    fused = eqx.filter_jit(solver.deposit_all)(
        starts, ends, velocity, charge, active, step
    )
    separate = [
        cochain.deposit(
            index,
            starts[index],
            ends[index],
            velocity[index],
            charge[index],
            active[index],
            step,
        )
        for index in range(2)
    ]
    assert bool(fused.successful)
    np.testing.assert_allclose(
        np.asarray(fused.current),
        np.asarray(separate[0].current + separate[1].current),
        rtol=0.0,
        atol=1e-12,
    )


def _huygens_box(lower: tuple[int, ...], upper: tuple[int, ...], bridge: Any) -> Any:
    acquisition = mx.MaxwellSpectralAcquisition(
        jnp.asarray([4.0, 8.0]), sign="positive", measure="time-integral", stop_time=1.0
    )
    exterior = mx.HomogeneousMaxwellExterior()
    if bridge is None:
        return sp.SpectralHuygensBoxPlan(lower, upper, acquisition, exterior)
    return mx.MaxwellHuygensBoxPlan(bridge, lower, upper, acquisition, exterior)


def _assert_same_phasors(found: Any, wanted: Any) -> None:
    peak = max(
        float(np.max(np.abs(np.asarray(leaf)))) for leaf in jax.tree.leaves(wanted)
    )
    assert peak > 0.0
    for left, right in zip(jax.tree.leaves(found), jax.tree.leaves(wanted), strict=True):
        np.testing.assert_allclose(
            np.asarray(left), np.asarray(right), rtol=0.0, atol=1e-10 * peak
        )


def _scenario_huygens() -> None:
    """Huygens phasors and restart on the distributed PSATD routes."""
    mesh = _four_device_mesh()
    counts = (16, 16, 8)
    bridge = _bridge(counts)
    species = _species(64, 3)
    # Node planes 1 <= lower < upper <= points - 2 keep both staggered faces.
    lower_node, upper_node = (2, 2, 1), (14, 14, 6)
    region = ((6 * H, 6 * H, 3.5 * H), (10 * H, 10 * H, 4.5 * H))
    bases = tuple(
        _spectral(
            bridge,
            species,
            mesh,
            decomposition,
            observers=(_huygens_box(lower_node, upper_node, None),),
        )
        for decomposition in ("global-fft", "local-guarded")
    )
    for base in bases:
        options = {
            "constraint_tolerance": 1e-2
            if base.pic_configuration.endswith("guarded")
            else 1e-6
        }
        # PSATD Huygens surfaces must stay current-free, and the nonlocal
        # charge-conservation modes spread particle current over the grid, so a
        # vacuum pulse (inactive particles) crosses the box.
        z = (np.arange(8) + 0.5) * H
        magnetic = np.zeros((16, 16, 8, 3))
        magnetic[..., 0] = np.exp(-(((z - 0.5) / 0.15) ** 2))[None, None, :]
        _, run, results = _paired_run(
            base,
            species,
            region,
            4,
            initial={"active_count": 0, "magnetic": magnetic},
            **options,
        )
        single, split = results[-1]
        _assert_same_phasors(
            run.solver.huygens_phasors(split.accepted_state.field),
            base.huygens_phasors(single.accepted_state.field),
        )
        restored = run.pic.restore(run.pic.checkpoint(split.accepted_state))
        _assert_same_state(restored, split.accepted_state, 0.0)


def _scenario_galilean() -> None:
    """Galilean PSATD: distributed particles drift and migrate in grid coordinates."""
    mesh = _four_device_mesh()
    bridge = _bridge((16, 16, 4))
    species = _species(32, 3)
    base = _spectral(
        bridge,
        species,
        mesh,
        "global-fft",
        variant="galilean",
        galilean_velocity=(0.9, 0.0, 0.0),
        charge_conservation="update-with-rho",
    )
    # Particles drift −0.9 relative to the grid: ≈ 0.2 cells per step across
    # 4-cell device blocks, so some cross block faces within six steps.
    _, run, results = _paired_run(base, species, ((0.0, 0.0, 0.0), (2.0, 2.0, 0.5)), 6)
    # Particles move in grid coordinates at v − v_grid (compare
    # test_runtime_drifts_particles_by_the_admitted_grid_velocity), so they
    # migrate between device blocks.
    assert sum(int(split.migration.migrated) for _, split in results) > 0


def _shard(value: Any, mesh: Mesh, count: int) -> Any:
    sharding = NamedSharding(mesh, PartitionSpec("x"))
    return jax.tree.map(
        lambda leaf: (
            jax.device_put(leaf, sharding)
            if leaf.ndim and leaf.shape[0] == count
            else leaf
        ),
        value,
    )


def _scenario_symbol_and_projection() -> None:
    """Advertised dispersion predicts the sharded vacuum update; sharded projection."""
    mesh = _four_device_mesh()
    reduced = _reduced(32, 1)
    solver = S.distribute_pic_field_solver(reduced, mesh)
    k = 2.0 * np.pi * 3
    x = (np.arange(32) + 0.5) / 32
    zero = jnp.zeros((32,))
    state = _shard(
        reduced.field.initialize(electric=(zero, jnp.cos(k * x), zero)), mesh, 32
    )
    dt = 0.9 * float(reduced.field.stable_dt)
    advance = eqx.filter_jit(
        lambda value: solver.advance(
            jnp.asarray(0.0), value, (zero, zero, zero), jnp.asarray(dt)
        )
    )
    for _ in range(40):
        result = advance(state)
        assert bool(result.successful)
        state = result.field
    omega = float(solver.dispersion_frequency(jnp.asarray([[k]]), dt)[0])
    # Independent Yee dispersion sin(ωΔt/2) = cΔt sin(kh/2)/h, c = 1.
    np.testing.assert_allclose(
        omega, 2.0 / dt * np.arcsin(dt * np.sin(k / 64) * 32), rtol=1e-13
    )
    # A Störmer–Verlet eigenmode started with B = 0 evolves as cos(nωΔt).
    np.testing.assert_allclose(
        np.asarray(state.electric[1]), np.cos(40 * omega * dt) * np.cos(k * x), atol=1e-10
    )
    counts = (16, 8, 4)
    bridge = _bridge(counts)
    species = _species(8, 3)
    base = _spectral(bridge, species, mesh, "global-fft")
    spectral = S.distribute_pic_field_solver(base, mesh)
    wave = 2.0 * np.pi * 2 / (counts[0] * H)
    magnetic = np.zeros((*counts, 3))
    magnetic[..., 2] = np.cos(wave * (np.arange(counts[0]) + 0.5) * H)[:, None, None]
    field = eqx.tree_at(
        lambda value: value.magnetic,
        base.field_with_charge(jnp.zeros(counts)),
        jnp.asarray(magnetic),
    )
    field = _shard(field, mesh, counts[0])
    source = sp.SpectralMaxwellSource(jnp.zeros((1, *counts, 3)), jnp.zeros((1, *counts)))
    step = jnp.asarray(0.7 * float(base.stable_step))
    march = eqx.filter_jit(
        lambda value: spectral.advance(jnp.asarray(0.0), value, source, step).field
    )
    for _ in range(12):
        field = march(field)
    omega = float(
        spectral.dispersion_frequency(jnp.asarray([[wave, 0.0, 0.0]]), float(step))[0]
    )
    np.testing.assert_allclose(omega, wave, rtol=1e-13)
    np.testing.assert_allclose(
        np.asarray(field.magnetic[..., 2]),
        np.cos(12 * omega * float(step)) * magnetic[..., 2],
        atol=1e-10,
    )
    rng = np.random.default_rng(11)
    charge = rng.normal(size=counts)
    charge -= charge.mean()
    empty = base.field_with_charge(jnp.zeros(counts))
    projected = spectral.project_gauss(
        _shard(empty, mesh, counts[0]), _shard(jnp.asarray(charge), mesh, counts[0])
    )
    reference = base.project_gauss(empty, jnp.asarray(charge))
    assert bool(projected.successful)
    assert float(projected.divergence_after) <= 1e-10 * float(np.max(np.abs(charge)))
    np.testing.assert_allclose(
        np.asarray(projected.field.electric),
        np.asarray(reference.field.electric),
        rtol=0.0,
        atol=1e-12,
    )


_SCENARIOS: dict[str, Callable[[], None]] = {
    "filters": _scenario_filters,
    "cochain-ledger": _scenario_cochain_ledger,
    "huygens": _scenario_huygens,
    "galilean": _scenario_galilean,
    "symbol-projection": _scenario_symbol_and_projection,
}


def _run_scenario(name: str) -> None:
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["PYTHONPATH"] = str(Path(phx.__file__).resolve().parents[1])
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
        "import test_distributed_pic_capabilities as module; "
        f"module._SCENARIOS[{name!r}]()"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-6000:]


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_distributed_protocol_routes_on_four_devices(name: str) -> None:
    _run_scenario(name)
