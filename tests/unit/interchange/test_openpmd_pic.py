#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import DimensionalScaleContract, ElectromagneticScaleContract
from phydrax._external_resource import (
    BoundedResource,
    read_bounded_resource,
    ResourceLimits,
    ResourceReadError,
)
from phydrax._external_runtime import pin_executable
from phydrax.interchange import (
    OpenPMDADIOS2Provider,
    OpenPMDMeshError,
    OpenPMDMeshImportPolicy,
    OpenPMDPICLayout,
    OpenPMDPICStreamWriter,
    read_openpmd_adios2,
    read_openpmd_meshes_hdf5,
    read_openpmd_pic_state,
    write_openpmd_pic_state,
)
from phydrax.interchange._report import AdapterStatus
from phydrax.units import CHARGE, KILOGRAM, LENGTH, TIME, UnitDefinition


D = phx.discretization
PIC = phx.discretization.pic
_SI = ElectromagneticScaleContract.si()
# Single-particle rest masses in code mass units; macroparticle mass is one.
_MASSES = (1.0e-3, 2.0e-3)


def _limits(max_bytes: int = 4_000_000) -> ResourceLimits:
    return ResourceLimits(max_bytes, 12, 400_000, 16_384, 1)


def _resource(path: Path) -> BoundedResource:
    return read_bounded_resource(path.name, trusted_root=path.parent, limits=_limits())


def _code_scale() -> ElectromagneticScaleContract:
    """Electron-normalized code units with c = 1 and a 1 µm length unit."""
    length = Fraction(1, 10**6)
    time = length / _SI.speed_of_light
    mass, charge, relativity = _SI.electron_mass, _SI.elementary_charge, _SI.relativity
    return ElectromagneticScaleContract.code_units(
        DimensionalScaleContract(
            UnitDefinition("L0", LENGTH, "si", length),
            UnitDefinition("m_e", KILOGRAM.dimension, "si", mass),
            UnitDefinition("T0", TIME, "si", time),
        ),
        UnitDefinition("q_e", CHARGE, "si", charge),
        gravitational_constant=relativity.gravitational_constant
        * mass
        * time**2
        / length**3,
        speed_of_light=1,
        reduced_planck_constant=_SI.reduced_planck_constant * time / (mass * length**2),
        boltzmann_constant=relativity.boltzmann_constant * time**2 / (mass * length**2),
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=_SI.vacuum_permittivity
        * length**3
        * mass
        / (charge**2 * time**2),
        constant_set_id="codata-2022",
    )


def _species(capacity: int, sign: float, name: str, dimension: int, offset: int) -> Any:
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
            maximum_charge_number=2,
            initial_charge_number=1,
        ),
    )


def _reduced_plan(
    count: int = 16, *, periodic: bool = True, capacity: int = 4, absorbing: bool = False
) -> Any:
    grid = D.TensorGridPlan(
        (D.UniformCellAxisSpec(count, periodic=periodic),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    pec = phx.solver.maxwell.MaxwellBoundaryPlan("pec")
    field = phx.solver.CompatibleMaxwell1DPlan(
        grid, boundaries=None if periodic else ((pec, pec),)
    )
    boundary = (
        PIC.PICOpenBoundaryPlan(
            jnp.asarray([0.0]),
            jnp.asarray([1.0]),
            kinds=(PIC.PICBoundaryKind.ABSORB, PIC.PICBoundaryKind.ABSORB),
        )
        if absorbing
        else None
    )
    return phx.solver.ElectromagneticPICPlan(
        phx.solver.ReducedMaxwellPICFieldSolver(field, PIC.ReducedPICTransferPlan(grid)),
        species=(
            _species(capacity, -1.0, "electrons", 1, 0),
            _species(capacity, 1.0, "ions", 1, 10),
        ),
        boundaries=boundary,
    )


def _reduced_state(plan: Any, step_size: float) -> Any:
    position = jnp.asarray([[0.1], [0.3], [0.55], [0.8]])
    velocity = jnp.zeros((4, 3)).at[:, 0].set(jnp.asarray([0.05, -0.1, 0.2, 0.02]))
    velocity = velocity.at[:, 1].set(0.03)
    return plan.initialize(
        (position + 0.01, position), (velocity, jnp.zeros((4, 3))), step_size
    )


def _particles_by_identity(species: Any) -> dict[str, np.ndarray]:
    """Active particles of one species in ascending (id_hi, id_lo) order."""
    population = species.population
    active = np.asarray(population.active)
    identity = (np.asarray(population.id_hi, dtype=np.uint64) << np.uint64(32)) | (
        np.asarray(population.id_lo, dtype=np.uint64)
    )
    order = np.argsort(identity[active])
    return {
        name: np.asarray(value)[active][order]
        for name, value in (
            ("identity", identity),
            ("parent_hi", population.parent_hi),
            ("parent_lo", population.parent_lo),
            ("mass", population.mass),
            ("charge_number", species.charge.charge_number),
            ("position", species.particles.position),
            ("proper_velocity", species.particles.proper_velocity),
        )
    }


def _assert_states_match(actual: Any, expected: Any, *, atol: float = 1e-12) -> None:
    for left, right in zip(
        jax.tree.leaves(actual.field), jax.tree.leaves(expected.field), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=atol)
    np.testing.assert_allclose(actual.wall_charge, expected.wall_charge, atol=atol)
    np.testing.assert_allclose(actual.time, expected.time, rtol=1e-15)
    assert int(actual.accepted_step) == int(expected.accepted_step)
    for left, right in zip(actual.species, expected.species, strict=True):
        found, wanted = _particles_by_identity(left), _particles_by_identity(right)
        for name in ("identity", "parent_hi", "parent_lo", "charge_number"):
            np.testing.assert_array_equal(found[name], wanted[name])
        np.testing.assert_allclose(found["mass"], wanted["mass"], rtol=1e-14)
        for name in ("position", "proper_velocity"):
            np.testing.assert_allclose(found[name], wanted[name], rtol=0.0, atol=atol)


def _cochain_plan() -> tuple[Any, Any]:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(4, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = D.StructuredCochainBridge(grid)
    species, charged = [], []
    for offset, sign, name in ((0, -1.0, "electrons"), (100, 1.0, "ions")):
        support = D.ParticleSetPlan(
            jnp.arange(offset, offset + 4), jnp.ones((4,)), ambient_dimension=3
        ).prepare()
        charged.append(
            D.ChargedParticlePlan(sign * jnp.ones((4,)), name).prepare(support)
        )
        species.append(
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
    transfers = tuple(
        PIC.PICParticleCochainTransferPlan(bridge).prepare(value) for value in charged
    )
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge, sources=(phx.solver.PICMaxwellCurrentSourcePlan(),), plan_id="pic"
    ).prepare()
    solver = phx.solver.CochainMaxwellPICFieldSolver(
        maxwell,
        phx.solver.CochainElectrostaticPlan(
            bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        transfers,
        tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers),
    )
    return phx.solver.ElectromagneticPICPlan(solver, species=species), maxwell


def test_cochain_state_round_trips_with_consistent_staggered_meshes(
    tmp_path: Path,
) -> None:
    plan, maxwell = _cochain_plan()
    ions = jnp.asarray(
        [[0.2, 0.2, 0.2], [0.35, 0.45, 0.55], [0.6, 0.3, 0.7], [0.8, 0.75, 0.4]]
    )
    electrons = ions + jnp.asarray([0.07, -0.04, 0.05])
    velocity = jnp.asarray(
        [[0.3, 0.1, -0.2], [-0.1, 0.25, 0.05], [0.2, -0.3, 0.1], [0.0, 0.1, 0.35]]
    )
    dt = 0.5 * float(maxwell.stable_dt)
    initial = plan.initialize((electrons, ions), (velocity, jnp.zeros((4, 3))), dt)
    result = plan.step_detailed(initial, dt)
    assert result.successful
    layout = OpenPMDPICLayout(plan, _code_scale(), _MASSES)
    exported = write_openpmd_pic_state(
        tmp_path,
        layout,
        result.accepted_state,
        step_size=dt,
        limits=_limits(),
        current=result.current,
    )
    assert exported.path == tmp_path / "pic_1.h5"
    assert exported.report.status == AdapterStatus.DECLARED_LOSS

    meshes = read_openpmd_meshes_hdf5(
        _resource(exported.path), OpenPMDMeshImportPolicy(1), scale=layout.scale
    ).iteration
    electric, rho = meshes.record("E"), meshes.record("rho")
    assert electric.positions == ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
    assert meshes.record("B").positions == (
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    )
    np.testing.assert_allclose(meshes.record("J").time_offset, -0.5 * dt, rtol=1e-15)
    # Gauss at the vertices from the declared staggering: E_i sits half a cell
    # after vertex i, so div E is the backward difference (vacuum eps = 1).
    spacing = electric.grid_spacing
    divergence = sum(
        (value - np.roll(value, 1, axis=axis)) / spacing[axis]
        for axis, value in enumerate(electric.components)
    )
    assert np.max(np.abs(rho.components[0])) > 1.0
    np.testing.assert_allclose(divergence, rho.components[0], rtol=0.0, atol=1e-9)
    # The step current integrates to the particles' charge displacement rate.
    cell = float(np.prod(spacing))
    displacement = sum(
        sign * (after.particles.position - before.particles.position)
        for sign, after, before in zip(
            (-1.0, 1.0), result.accepted_state.species, initial.species, strict=True
        )
    )
    np.testing.assert_allclose(
        [np.sum(value) * cell for value in meshes.record("J").components],
        np.sum(displacement, axis=0) / dt,
        rtol=1e-10,
        atol=1e-12,
    )

    imported = read_openpmd_pic_state(
        _resource(exported.path), layout, OpenPMDMeshImportPolicy(1)
    )
    assert imported.step_size == dt
    _assert_states_match(imported.state, result.accepted_state)
    continued = plan.step_detailed(imported.state, dt)
    reference = plan.step_detailed(result.accepted_state, dt)
    assert continued.successful
    _assert_states_match(continued.accepted_state, reference.accepted_state)


def test_reduced_state_restores_wall_charge_and_continues_the_run(
    tmp_path: Path,
) -> None:
    plan = _reduced_plan(periodic=False, absorbing=True)
    dt = 0.9 * plan.solver.field.stable_dt
    position = jnp.asarray([[0.1], [0.3], [0.55], [0.99]])
    velocity = jnp.zeros((4, 3)).at[3, 0].set(0.5)
    state = plan.initialize((position, position), (velocity, jnp.zeros((4, 3))), dt)
    result = plan.step_detailed(state, dt)
    assert result.successful
    assert float(jnp.sum(jnp.abs(result.accepted_state.wall_charge))) > 0.0
    layout = OpenPMDPICLayout(plan, _code_scale(), _MASSES)
    exported = write_openpmd_pic_state(
        tmp_path, layout, result.accepted_state, step_size=dt, limits=_limits()
    )
    assert "boundaries" in {value.path for value in exported.report.losses}
    with h5py.File(exported.path, "r") as handle:
        electrons = handle["data/1/particles/electrons"]
        # The absorbed electron is not exported; weighting = macro mass / m.
        assert electrons["id"].shape == (3,)
        np.testing.assert_allclose(electrons["weighting"].attrs["value"], 1.0e3)

    imported = read_openpmd_pic_state(
        _resource(exported.path),
        layout,
        OpenPMDMeshImportPolicy(1, records=("E", "B", "rho")),
    )
    _assert_states_match(imported.state, result.accepted_state)
    continued = plan.step_detailed(imported.state, dt)
    reference = plan.step_detailed(result.accepted_state, dt)
    assert continued.successful
    _assert_states_match(continued.accepted_state, reference.accepted_state)


def _exported_reduced(tmp_path: Path) -> tuple[Any, float, Path]:
    plan = _reduced_plan()
    dt = 0.5 * plan.solver.field.stable_dt
    result = plan.step_detailed(_reduced_state(plan, dt), dt)
    path = write_openpmd_pic_state(
        tmp_path,
        OpenPMDPICLayout(plan, _code_scale(), _MASSES),
        result.accepted_state,
        step_size=dt,
        limits=_limits(),
        current=result.current,
    ).path
    return plan, dt, path


def _other_mass(plan: Any, path: Path) -> OpenPMDPICLayout:
    del path
    return OpenPMDPICLayout(plan, _code_scale(), (_MASSES[0], 3.0e-3))


def _other_grid(plan: Any, path: Path) -> OpenPMDPICLayout:
    del plan, path
    return OpenPMDPICLayout(_reduced_plan(32), _code_scale(), _MASSES)


def _small_capacity(plan: Any, path: Path) -> OpenPMDPICLayout:
    del plan, path
    return OpenPMDPICLayout(_reduced_plan(capacity=2), _code_scale(), _MASSES)


def _unstaggered_momenta(plan: Any, path: Path) -> OpenPMDPICLayout:
    with h5py.File(path, "r+") as handle:
        handle["data/1/particles/ions/momentum"].attrs["timeOffset"] = 0.0
    return OpenPMDPICLayout(plan, _code_scale(), _MASSES)


def _truncated_particles(plan: Any, path: Path) -> OpenPMDPICLayout:
    with h5py.File(path, "r+") as handle:
        name = "data/1/particles/electrons/position/x"
        values = handle[name][()]
        attributes = dict(handle[name].attrs)
        del handle[name]
        handle.create_dataset(name, data=values[:-1]).attrs.update(attributes)
    return OpenPMDPICLayout(plan, _code_scale(), _MASSES)


@pytest.mark.parametrize(
    ("layout", "status"),
    [
        (_other_mass, AdapterStatus.INCONSISTENT_SOURCE),
        (_other_grid, AdapterStatus.INCONSISTENT_SOURCE),
        (_small_capacity, AdapterStatus.INCONSISTENT_SOURCE),
        (_unstaggered_momenta, AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC),
        (_truncated_particles, AdapterStatus.MALFORMED_SOURCE),
    ],
    ids=["particle-mass", "grid", "capacity", "unstaggered-momenta", "truncated"],
)
def test_import_refuses_records_that_do_not_rebuild_the_plan_state(
    tmp_path: Path,
    layout: Callable[[Any, Path], OpenPMDPICLayout],
    status: AdapterStatus,
) -> None:
    plan, _, path = _exported_reduced(tmp_path)
    bound = layout(plan, path)
    with pytest.raises(OpenPMDMeshError) as caught:
        read_openpmd_pic_state(_resource(path), bound, OpenPMDMeshImportPolicy(1))
    assert caught.value.status == status
    assert not caught.value.report.valid


def test_layout_requires_the_run_code_units() -> None:
    with pytest.raises(ValueError, match="speed of light"):
        OpenPMDPICLayout(_reduced_plan(), _SI, _MASSES)


def test_stream_writer_publishes_due_iterations_atomically_within_bounds(
    tmp_path: Path,
) -> None:
    plan = _reduced_plan()
    dt = 0.5 * plan.solver.field.stable_dt
    layout = OpenPMDPICLayout(plan, _code_scale(), _MASSES)
    writer = OpenPMDPICStreamWriter(
        tmp_path,
        layout,
        limits=_limits(),
        maximum_iterations=3,
        maximum_total_bytes=10_000_000,
        interval=2,
    )
    state = _reduced_state(plan, dt)
    assert writer.write(state, step_size=dt) is not None
    results = []
    for _ in range(4):
        result = plan.step_detailed(state, dt)
        assert result.successful
        results.append(result)
        receipt = writer.write(result, step_size=dt)
        assert (receipt is None) == (int(result.accepted_state.accepted_step) % 2 == 1)
        state = result.accepted_state
    assert writer.write(results[-1], step_size=dt) is None
    rejected = eqx.tree_at(
        lambda value: value.successful, results[-1], jnp.asarray(False)
    )
    assert writer.write(rejected, step_size=dt) is None
    with pytest.raises(ValueError, match="back in time"):
        writer.write(results[1].accepted_state, step_size=dt)

    assert [value.iteration for value in writer.receipts] == [0, 2, 4]
    assert sorted(value.name for value in tmp_path.iterdir()) == [
        "pic_0.h5",
        "pic_2.h5",
        "pic_4.h5",
    ]
    assert writer.total_bytes == sum(value.stat().st_size for value in tmp_path.iterdir())
    with h5py.File(tmp_path / "pic_0.h5", "r") as initial:
        assert "J" not in initial["data/0/meshes"]
    imported = read_openpmd_pic_state(
        _resource(tmp_path / "pic_2.h5"),
        layout,
        OpenPMDMeshImportPolicy(2),
    )
    _assert_states_match(imported.state, results[1].accepted_state)

    for _ in range(2):
        result = plan.step_detailed(state, dt)
        state = result.accepted_state
    with pytest.raises(ResourceReadError, match="maximum_iterations"):
        writer.write(result, step_size=dt)
    assert not (tmp_path / "pic_6.h5").exists()


def test_stream_writer_refuses_the_total_byte_budget_before_publishing(
    tmp_path: Path,
) -> None:
    plan = _reduced_plan()
    dt = 0.5 * plan.solver.field.stable_dt
    writer = OpenPMDPICStreamWriter(
        tmp_path,
        OpenPMDPICLayout(plan, _code_scale(), _MASSES),
        limits=_limits(),
        maximum_iterations=10,
        maximum_total_bytes=1_024,
    )
    with pytest.raises(ResourceReadError, match="maximum_total_bytes"):
        writer.write(_reduced_state(plan, dt), step_size=dt)
    assert list(tmp_path.iterdir()) == []


def test_stream_writer_adios2_provider_route(tmp_path: Path) -> None:
    executable = shutil.which(os.environ.get("PHYDRAX_OPENPMD_PIPE", "openpmd-pipe"))
    version = os.environ.get("PHYDRAX_OPENPMD_API_VERSION")
    if executable is None or version is None:
        pytest.skip(
            "requires openPMD-api openpmd-pipe (set PHYDRAX_OPENPMD_PIPE and "
            "PHYDRAX_OPENPMD_API_VERSION)"
        )
    provider = OpenPMDADIOS2Provider(
        pin_executable(executable, version=version, license_id="LGPL-3.0-or-later")
    )
    plan = _reduced_plan()
    dt = 0.5 * plan.solver.field.stable_dt
    layout = OpenPMDPICLayout(plan, _code_scale(), _MASSES)
    writer = OpenPMDPICStreamWriter(
        tmp_path,
        layout,
        limits=_limits(),
        maximum_iterations=4,
        maximum_total_bytes=10_000_000,
        provider=provider,
    )
    result = plan.step_detailed(_reduced_state(plan, dt), dt)
    receipt = writer.write(result, step_size=dt)
    assert receipt is not None
    assert receipt.path == tmp_path / "pic_1.bp"
    assert len(receipt.report.stages) == 2
    image = read_openpmd_adios2(provider, receipt.path, limits=_limits())
    imported = read_openpmd_pic_state(image, layout, OpenPMDMeshImportPolicy(1))
    _assert_states_match(imported.state, result.accepted_state)
