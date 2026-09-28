#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Contracts of the pinned WarpX, Smilei, and PIConGPU PIC oracles.

Decks are checked against SI values computed here from CODATA 2022 literals.
The reference outputs under ``tests/data/providers/{warpx,smilei}`` were
produced by the real providers from `_tiny_yee_case` (provenance, command, and
date in each directory's ``provenance.json``). Live comparisons skip only when
their ``PHYDRAX_*`` variables are absent.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax import DimensionalScaleContract, ElectromagneticScaleContract
from phydrax._external_resource import read_bounded_resource, ResourceLimits
from phydrax._external_runtime import pin_executable
from phydrax.discretization.pic import ExternalFieldSample
from phydrax.electromagnetics import (
    ChargedTrajectory,
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
)
from phydrax.interchange import format_capabilities
from phydrax.units import CHARGE, KILOGRAM, LENGTH, TIME, UnitDefinition


D = phx.discretization
PIC = D.pic
S = phx.solver
SPECTRAL = phx.solver.maxwell.spectral
_DATA = Path(__file__).parents[2] / "data" / "providers"
# CODATA 2022 exact or recommended values, independent of the scale contract.
_C = 299_792_458.0
_E = 1.602176634e-19
_ME = 9.1093837139e-31
_EPS0 = 8.8541878188e-12
_LENGTH = 1.0e-6
_ION_RATIO = 1836.0


def _code_scale(length: Fraction = Fraction(1, 10**6)) -> ElectromagneticScaleContract:
    """Code units with c = ε₀ = 1, electron q/m = −1, and length unit ``length`` (m).

    With ``T = L/c``, the charge unit ``Q = ε₀ L c² m_e / e`` and mass unit
    ``M = m_e Q / e`` make ``ε₀`` and ``e/m_e`` unity; the electron density
    ``ε₀ m_e c²/(e² L²)`` then has plasma frequency ``c/L`` (plasma units).
    """
    si = ElectromagneticScaleContract.si()
    time = length / si.speed_of_light
    charge = (
        si.vacuum_permittivity
        * length
        * si.speed_of_light**2
        * si.electron_mass
        / si.elementary_charge
    )
    mass = si.electron_mass * charge / si.elementary_charge
    relativity = si.relativity
    return ElectromagneticScaleContract.code_units(
        DimensionalScaleContract(
            UnitDefinition("L0", LENGTH, "si", length),
            UnitDefinition("M0", KILOGRAM.dimension, "si", mass),
            UnitDefinition("T0", TIME, "si", time),
        ),
        UnitDefinition("Q0", CHARGE, "si", charge),
        gravitational_constant=relativity.gravitational_constant
        * mass
        * time**2
        / length**3,
        speed_of_light=1,
        reduced_planck_constant=si.reduced_planck_constant * time / (mass * length**2),
        boltzmann_constant=relativity.boltzmann_constant * time**2 / (mass * length**2),
        elementary_charge=si.elementary_charge / charge,
        electron_mass=si.electron_mass / mass,
        vacuum_permittivity=1,
        constant_set_id="codata-2022",
    )


_SCALE = _code_scale()
_ELECTRON_MASS = float(_SCALE.electron_mass)
_PLASMA_LENGTH = 5.0e-6
_PLASMA_SCALE = _code_scale(Fraction(5, 10**6))
_PLASMA_ELECTRON_MASS = float(_PLASMA_SCALE.electron_mass)


def _bridge(counts: tuple[int, int, int], spacing: float) -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(n, periodic=True) for n in counts),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [n * spacing for n in counts]]))
    return D.StructuredCochainBridge(grid)


def _species(
    bridge: Any, count: int, weight: float, shape_order: PIC.PICShapeOrder
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Electrons (q/m = −1) and ions (q/m = 1/1836) of equal macrocharge."""
    species, charged = [], []
    for offset, specific, name, mass in (
        (0, -1.0, "electrons", weight),
        (10**6, 1.0 / _ION_RATIO, "ions", _ION_RATIO * weight),
    ):
        support = D.ParticleSetPlan(
            jnp.arange(offset, offset + count),
            mass * jnp.ones((count,)),
            ambient_dimension=3,
        ).prepare()
        charged.append(
            D.ChargedParticlePlan(specific * mass * jnp.ones((count,)), name).prepare(
                support
            )
        )
        species.append(
            PIC.PICSpeciesPlan(
                D.ParticlePopulationPlan(support),
                PIC.PICChargeModelPlan(
                    specific,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    transfer = PIC.PICParticleCochainTransferPlan(bridge, shape_order=shape_order)
    return tuple(species), tuple(transfer.prepare(value) for value in charged)


def _lattice(counts: tuple[int, int, int], spacing: float) -> np.ndarray:
    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in counts), indexing="ij")
    return np.stack(
        ((ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing), axis=-1
    ).reshape(-1, 3)


def _weight(spacing: float, plasma2: float) -> float:
    """Macroparticle mass giving ``ω_pe² = plasma2`` with one particle per cell."""
    return plasma2 * spacing**3


def _yee_plan(
    counts: tuple[int, int, int], spacing: float, shape_order: PIC.PICShapeOrder
) -> Any:
    bridge = _bridge(counts, spacing)
    species, transfers = _species(
        bridge, math.prod(counts), _weight(spacing, 1.0), shape_order
    )
    maxwell = S.CompatibleMaxwellPlan(
        bridge, sources=(S.PICMaxwellCurrentSourcePlan(),), plan_id="pic"
    ).prepare()
    solver = S.CochainMaxwellPICFieldSolver(
        maxwell,
        S.CochainElectrostaticPlan(
            bridge, S.CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        transfers,
        tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers),
    )
    return S.ElectromagneticPICPlan(solver, species=species)


def _spectral_plan(
    counts: tuple[int, int, int], spacing: float, plasma2: float, **options: Any
) -> tuple[Any, Any]:
    bridge = _bridge(counts, spacing)
    species, transfers = _species(bridge, math.prod(counts), _weight(spacing, plasma2), 2)
    solver = SPECTRAL.SpectralMaxwellPlan(bridge, **options).prepare(
        transfers, tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers)
    )
    return S.ElectromagneticPICPlan(solver, species=species), solver


def _drift(count: int, speed: float) -> np.ndarray:
    velocity = np.zeros((count, 3))
    velocity[:, 0] = speed
    return velocity


def _oscillation_case(
    counts: tuple[int, int, int],
    shape_order: PIC.PICShapeOrder,
    *,
    steps: int,
    interval: int,
    speed: float = 0.01,
) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    """Electrons drifting through co-located ions: a uniform ``ω_pe = 1`` oscillation."""
    spacing, step = 0.2, 0.1
    plan = _yee_plan(counts, spacing, shape_order)
    lattice = _lattice(counts, spacing)
    electrons = _drift(lattice.shape[0], speed)
    ions = np.zeros_like(electrons)
    case = S.pic_oracle_case(
        plan,
        _SCALE,
        (lattice, lattice),
        (electrons, ions),
        step,
        particle_masses=(_ELECTRON_MASS, _ION_RATIO * _ELECTRON_MASS),
        steps=steps,
        output_interval=interval,
    )
    return case, plan, lattice, electrons, ions


def _tiny_yee_case() -> Any:
    """The case the stored WarpX and Smilei reference outputs were produced from."""
    return _oscillation_case((6, 6, 6), 2, steps=2, interval=1)[0]


def _nci_case(
    counts: tuple[int, int, int] = (8, 8, 8), gamma: float = 10.0, **options: Any
) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, float]:
    spacing = 0.3868
    plan, solver = _spectral_plan(counts, spacing, 4.0 * gamma, **options)
    ions = _lattice(counts, spacing)
    electrons = ions.copy()
    electrons[:, (0, 2)] += np.random.default_rng(1).uniform(
        -0.05 * spacing, 0.05 * spacing, (ions.shape[0], 2)
    )
    speed = math.sqrt(1.0 - 1.0 / gamma**2)
    return plan, solver, electrons, ions, _drift(ions.shape[0], speed), speed


def _deck(text: bytes) -> dict[str, str]:
    lines = [line.split(" = ", 1) for line in text.decode().splitlines() if line]
    return {key: value for key, value in lines}


def _floats(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split()])


# One electron gyrating in a uniform external field (the A1 ↔ P scenario).
_B0 = 10.0
_START = np.asarray([0.5, 0.5, 0.3], dtype=np.float64)
_VELOCITY = np.asarray([0.0, 0.4, 0.1], dtype=np.float64)
_ORBIT_GAMMA = 1.0 / math.sqrt(1.0 - float(_VELOCITY @ _VELOCITY))
# Ω = |q|B/(γm) with the unit electron specific charge of `_code_scale`.
_OMEGA = _B0 / _ORBIT_GAMMA
_WINDOW = 4.0 * 2.0 * math.pi / _OMEGA
_TRACK_WEIGHT = 1.0e-9


class _ExternalField(phx.StrictModule):
    """``B = magnetic (1 + gradient x)(1 + rate t)`` and ``E = 0``."""

    magnetic: Array
    gradient: float = eqx.field(static=True)
    rate: float = eqx.field(static=True)

    @property
    def source_id(self) -> str:
        return "pic-oracle-test-field"

    def external_fields(self, positions: Array, times: Array, /) -> ExternalFieldSample:
        factor = (1.0 + self.gradient * positions[:, :1]) * (
            1.0 + self.rate * times[:, None]
        )
        return ExternalFieldSample(
            jnp.zeros_like(positions),
            factor * self.magnetic,
            jnp.ones(positions.shape[:1], dtype=jnp.bool_),
        )


def _field(gradient: float = 0.0, rate: float = 0.0) -> _ExternalField:
    return _ExternalField(
        magnetic=jnp.asarray([0.0, 0.0, _B0], dtype=jnp.float64),
        gradient=gradient,
        rate=rate,
    )


def _track_plan(
    *,
    counts: tuple[int, int] = (1, 1),
    lanes: int = 1,
    fields: tuple[Any, ...] | None = None,
    capacity: int = 512,
) -> tuple[Any, Any]:
    """Electron and resting ion of weight 1e-9 on a periodic 8³ Yee cochain box."""
    bridge = _bridge((8, 8, 8), 0.125)
    species, charged = [], []
    for offset, specific, name, count in (
        (0, -1.0, "electrons", counts[0]),
        (100, 1.0, "ions", counts[1]),
    ):
        support = D.ParticleSetPlan(
            jnp.arange(offset, offset + count),
            jnp.full((count,), _TRACK_WEIGHT),
            ambient_dimension=3,
        ).prepare()
        charged.append(
            D.ChargedParticlePlan(
                specific * _TRACK_WEIGHT * jnp.ones((count,)), name
            ).prepare(support)
        )
        species.append(
            PIC.PICSpeciesPlan(
                D.ParticlePopulationPlan(support),
                PIC.PICChargeModelPlan(
                    specific,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    transfer = PIC.PICParticleCochainTransferPlan(bridge)
    transfers = tuple(transfer.prepare(value) for value in charged)
    maxwell = S.CompatibleMaxwellPlan(
        bridge, sources=(S.PICMaxwellCurrentSourcePlan(),), plan_id="track"
    ).prepare()
    solver = S.CochainMaxwellPICFieldSolver(
        maxwell,
        S.CochainElectrostaticPlan(
            bridge, S.CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        transfers,
        tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers),
    )
    recorder = PIC.PICTrackRecorder(
        tuple(species),
        [0] * max(lanes, 1),
        (
            np.zeros(max(lanes, 1), dtype=np.uint32),
            np.arange(max(lanes, 1), dtype=np.uint32),
        ),
        relativity=_SCALE.relativity,
        sample_capacity=capacity,
    )
    plan = S.ElectromagneticPICPlan(
        solver,
        species=tuple(species),
        recorders=(recorder,) if lanes else (),
        external_fields=(_field(),) if fields is None else fields,
        pusher=PIC.RelativisticPushPlan(_SCALE.relativity, method="boris"),
    )
    return plan, recorder


def _track_inputs(plan: Any) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    electrons, ions = (value.capacity for value in plan.species)
    return (
        (np.tile(_START, (electrons, 1)), np.tile(_START, (ions, 1))),
        (np.tile(_VELOCITY, (electrons, 1)), np.zeros((ions, 3))),
    )


def _track_case(steps: int, plan: Any | None = None, *, step: float | None = None) -> Any:
    """``steps`` steps of the gyration (four periods unless ``step`` is given)."""
    plan = _track_plan()[0] if plan is None else plan
    positions, velocities = _track_inputs(plan)
    return S.pic_oracle_case(
        plan,
        _SCALE,
        positions,
        velocities,
        _WINDOW / steps if step is None else step,
        particle_masses=(_ELECTRON_MASS, _ELECTRON_MASS),
        steps=steps,
        output_interval=1,
    )


def _tiny_track_case() -> Any:
    """The case the stored WarpX single-particle-radiation output came from."""
    return _track_case(3, step=_WINDOW / 96)


# Laser wakefield in plasma units (the P3 ↔ FBPIC configuration).
_IMMOBILE = 1.0e9


def _cylinder_plasma(
    grid: Any,
    lower: float,
    upper: float,
    radius: float,
    per_cell: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced cold plasma of unit density with irrational angle offsets."""
    dz = (upper - lower) / (round((upper - lower) / grid.axial_spacing) * per_cell[0])
    dr = radius / (round(radius / grid.radial_spacing) * per_cell[1])
    z = np.arange(lower + 0.5 * dz, upper, dz)
    r = np.arange(0.5 * dr, radius, dr)
    theta = 2.0 * np.pi / per_cell[2] * np.arange(per_cell[2])
    zz, rr, tt = np.meshgrid(z, r, theta, indexing="ij")
    tt = tt + 2.0 * np.pi / per_cell[2] * (
        (
            np.arange(z.size)[:, None, None] * 0.618
            + np.arange(r.size)[None, :, None] * 0.414
        )
        % 1.0
    )
    positions = np.stack((rr * np.cos(tt), rr * np.sin(tt), zz), axis=-1).reshape(-1, 3)
    return positions, (rr * 2.0 * np.pi / per_cell[2] * dr * dz).reshape(-1)


def _wakefield(
    grid: Any,
    plasma: tuple[float, float, float],
    per_cell: tuple[int, int, int],
    **options: Any,
) -> tuple[Any, Any, np.ndarray]:
    """P3 plan of plasma electrons and immobile (m/q = 10⁹) neutralizing ions."""
    positions, weights = _cylinder_plasma(grid, *plasma, per_cell)
    transfer = PIC.AzimuthalTransferPlan(grid, shape_order=1)
    species, transfers = [], []
    for offset, specific, name, mass in (
        (0, -1.0, "electrons", 1.0),
        (10**7, 1.0 / _IMMOBILE, "ions", _IMMOBILE),
    ):
        support = D.ParticleSetPlan(
            jnp.arange(offset, offset + weights.size),
            jnp.asarray(mass * weights),
            ambient_dimension=3,
        ).prepare()
        species.append(
            PIC.PICSpeciesPlan(
                D.ParticlePopulationPlan(support),
                PIC.PICChargeModelPlan(
                    specific,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
        transfers.append(transfer.prepare(support))
    solver = SPECTRAL.QuasiCylindricalMaxwellPlan(grid, **options).prepare(
        tuple(transfers)
    )
    return S.ElectromagneticPICPlan(solver, species=tuple(species)), solver, positions


def _wakefield_case(
    plan: Any, positions: np.ndarray, laser: Any, step: float, steps: int, interval: int
) -> Any:
    zero = np.zeros_like(positions)
    return S.pic_oracle_wakefield_case(
        plan,
        _PLASMA_SCALE,
        (positions, positions),
        (zero, zero),
        step,
        laser=laser,
        particle_masses=(_PLASMA_ELECTRON_MASS, _IMMOBILE * _PLASMA_ELECTRON_MASS),
        steps=steps,
        output_interval=interval,
    )


def _tiny_wakefield() -> tuple[Any, Any, np.ndarray]:
    """17 × 34 cells (Δr = 0.125, Δz = 0.1): just above WarpX's guard cells."""
    grid = PIC.QuasiCylindricalGrid(2.125, 17, 0.0, 3.4, 34, 2)
    return _wakefield(grid, (2.3, 3.2, 1.5), (1, 1, 2))


_TINY_LASER = S.PICOracleLaser(1.0, 1.0, 1.0, 0.4, 1.0)


def _tiny_wakefield_case() -> Any:
    """The case the stored WarpX laser-wakefield-stage output came from."""
    plan, solver, positions = _tiny_wakefield()
    step = 0.9 * float(solver.stable_step)
    return _wakefield_case(plan, positions, _TINY_LASER, step, 4, 4)


# Deck generation -----------------------------------------------------------------


def test_warpx_psatd_deck_carries_the_plan_in_si() -> None:
    plan, _, electrons, ions, velocity, speed = _nci_case()
    case = S.pic_oracle_case(
        plan,
        _SCALE,
        (electrons, ions),
        (velocity, velocity),
        0.45 * 0.3868,
        particle_masses=(_ELECTRON_MASS, _ION_RATIO * _ELECTRON_MASS),
        steps=4,
        output_interval=2,
    )
    deck = _deck(S.warpx_input(case)["inputs"])
    spacing = 0.3868 * _LENGTH
    np.testing.assert_allclose(_floats(deck["geometry.prob_hi"]), 8 * spacing, rtol=1e-12)
    assert float(deck["warpx.const_dt"]) == pytest.approx(0.45 * spacing / _C, rel=1e-12)
    assert float(deck["electrons.charge"]) == pytest.approx(-_E, rel=1e-12)
    assert float(deck["electrons.mass"]) == pytest.approx(_ME, rel=1e-12)
    assert float(deck["ions.mass"]) == pytest.approx(_ION_RATIO * _ME, rel=1e-12)
    # ω_pe² = 4γ in units of c/L; one macroparticle per cell carries n Δ³ electrons.
    density = _EPS0 * _ME * 40.0 * (_C / _LENGTH) ** 2 / _E**2
    np.testing.assert_allclose(
        _floats(deck["electrons.multiple_particles_weight"]),
        density * spacing**3,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        _floats(deck["electrons.multiple_particles_ux"]),
        speed / math.sqrt(1.0 - speed**2),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        _floats(deck["electrons.multiple_particles_pos_x"]),
        electrons[:, 0] * _LENGTH,
        rtol=1e-12,
    )
    assert deck["algo.maxwell_solver"] == "psatd"
    assert deck["warpx.grid_type"] == "collocated"
    assert deck["psatd.nox"] == "inf"
    assert deck["psatd.current_correction"] == "1"
    assert deck["algo.current_deposition"] == "direct"
    assert deck["algo.particle_shape"] == "2"
    assert deck["fields.intervals"] == "2"


def test_warpx_yee_deck_uses_staggered_esirkepov_galerkin_transfer() -> None:
    case = _oscillation_case((8, 8, 8), 3, steps=4, interval=2)[0]
    deck = _deck(S.warpx_input(case)["inputs"])
    assert deck["algo.maxwell_solver"] == "yee"
    assert deck["warpx.grid_type"] == "staggered"
    assert deck["algo.current_deposition"] == "esirkepov"
    assert deck["interpolation.galerkin_scheme"] == "1"
    assert deck["algo.particle_shape"] == "3"
    assert float(deck["warpx.const_dt"]) == pytest.approx(0.1 * _LENGTH / _C, rel=1e-12)


def _execute_namelist(inputs: dict[str, bytes]) -> dict[str, Any]:
    """Run a generated Smilei namelist against recording stand-ins."""
    recorded: dict[str, Any] = {"Species": []}

    def main(**options: Any) -> None:
        recorded["Main"] = options

    def species(**options: Any) -> None:
        recorded["Species"].append(options)

    def diagnostic(**options: Any) -> None:
        recorded["DiagFields"] = options

    def load(name: str) -> np.ndarray:
        return np.load(BytesIO(inputs[name]), allow_pickle=False)

    numpy = type("numpy", (), {"load": staticmethod(load)})
    source = inputs["smilei.py"].decode().replace("import numpy as np\n", "")
    # The namelist is the generated deck under test.
    exec(
        compile(source, "smilei.py", "exec"),
        {"Main": main, "Species": species, "DiagFields": diagnostic, "np": numpy},
    )
    return recorded


def test_smilei_namelist_normalizes_to_the_scale_length() -> None:
    case, _, lattice, electrons, _ = _oscillation_case((8, 8, 8), 2, steps=4, interval=2)
    recorded = _execute_namelist(S.smilei_input(case))
    main = recorded["Main"]
    # ω_r = c/L makes Smilei's length unit c/ω_r the 1 µm scale length.
    omega = _C / _LENGTH
    assert main["reference_angular_frequency_SI"] == pytest.approx(omega, rel=1e-12)
    np.testing.assert_allclose(main["cell_length"], 0.2, rtol=1e-12)
    assert main["timestep"] == pytest.approx(0.1, rel=1e-12)
    assert main["maxwell_solver"] == "Yee"
    assert int(main["simulation_time"] / main["timestep"]) == 4
    electron, ion = recorded["Species"]
    assert (electron["mass"], electron["charge"]) == pytest.approx((1.0, -1.0))
    assert (ion["mass"], ion["charge"]) == pytest.approx((_ION_RATIO, 1.0))
    positions = electron["position_initialization"]
    np.testing.assert_allclose(positions[:3].T, lattice, rtol=1e-12)
    # Weight unit N_r L_r³ with N_r = ε₀ m_e ω_r²/e² = the ω_pe = ω_r density.
    np.testing.assert_allclose(positions[3], 0.2**3, rtol=1e-9)
    np.testing.assert_allclose(
        electron["momentum_initialization"].T,
        electrons / np.sqrt(1.0 - 0.01**2),
        rtol=1e-12,
    )
    assert recorded["DiagFields"]["every"] == 2


def _param_value(text: str, name: str) -> float:
    for line in text.splitlines():
        if f" {name} = " in line:
            return float(line.split(" = ")[1].rstrip(";"))
    raise AssertionError(f"{name} is absent")


def test_picongpu_params_declare_lattice_density_drift_and_shape() -> None:
    case = _oscillation_case((24, 24, 12), 2, steps=4, interval=2, speed=0.6)[0]
    params = {
        Path(key).name: value.decode() for key, value in S.picongpu_input(case).items()
    }
    simulation = params["simulation.param"]
    assert _param_value(simulation, "DELTA_T_SI") == pytest.approx(
        0.1 * _LENGTH / _C, rel=1e-12
    )
    assert _param_value(simulation, "CELL_WIDTH_SI") == pytest.approx(
        0.2 * _LENGTH, rel=1e-12
    )
    # ω_pe = c/L: n = ε₀ m_e ω²/e².
    density = _EPS0 * _ME * (_C / _LENGTH) ** 2 / _E**2
    assert _param_value(simulation, "BASE_DENSITY_SI") == pytest.approx(density, rel=1e-9)
    particle = params["particle.param"]
    gammas = [
        float(line.split(" = ")[1].rstrip(";"))
        for line in particle.splitlines()
        if "float_64 gamma" in line
    ]
    assert gammas == pytest.approx([1.25, 1.0], rel=1e-12)
    offsets = [
        [float(value) for value in line.split("float3_X(")[1].rstrip(");").split(",")]
        for line in particle.splitlines()
        if "inCellOffset" in line
    ]
    np.testing.assert_allclose(offsets, 0.5, rtol=1e-12)
    species = params["speciesDefinition.param"]
    assert "ChargeRatioSpecies0, 1.0" in species
    assert "ChargeRatioSpecies1, -1.0" in species
    assert "MassRatioSpecies1, 1836.0" in species
    assert "particles::shapes::TSC" in params["species.param"]
    assert "precision64Bit" in params["precision.param"]


def test_warpx_track_deck_applies_the_uniform_field_and_writes_the_tracked_species() -> (
    None
):
    case = _tiny_track_case()
    assert case.scenario == "single-particle-radiation"
    assert case.track.identity == (0, 0)
    deck = _deck(S.warpx_input(case)["inputs"])
    # B unit m_e c/(e L) of `_code_scale`.
    np.testing.assert_allclose(
        _floats(deck["particles.B_external_particle"]),
        [0.0, 0.0, _B0 * _ME * _C / (_E * _LENGTH)],
        rtol=1e-12,
    )
    np.testing.assert_array_equal(_floats(deck["particles.E_external_particle"]), 0.0)
    # Macro mass 1e-9 over the electron mass e²/(ε₀ L c² m_e) in these units.
    assert float(deck["electrons.multiple_particles_weight"]) == pytest.approx(
        _TRACK_WEIGHT * _EPS0 * _LENGTH * _C**2 * _ME / _E**2, rel=1e-9
    )
    assert float(deck["electrons.multiple_particles_uy"]) == pytest.approx(
        _ORBIT_GAMMA * 0.4, rel=1e-12
    )
    assert float(deck["warpx.const_dt"]) == pytest.approx(
        _WINDOW / 96 * _LENGTH / _C, rel=1e-12
    )
    assert deck["diagnostics.diags_names"] == "track"
    assert deck["track.species"] == "electrons"
    assert deck["track.openpmd_encoding"] == "g"
    assert deck["track.intervals"] == "1"
    assert deck["algo.maxwell_solver"] == "yee"
    assert deck["algo.particle_pusher"] == "boris"


def test_warpx_rz_deck_launches_the_laser_antenna_on_the_case_grid() -> None:
    case = _tiny_wakefield_case()
    deck = _deck(S.warpx_input(case)["inputs"])
    length = _PLASMA_LENGTH
    assert deck["geometry.dims"] == "RZ"
    assert deck["amr.n_cell"] == "17 34"
    assert deck["warpx.n_rz_azimuthal_modes"] == "2"
    # The box starts half a cell below the first axial node, so WarpX's
    # cell-centered output lands on the case's nodes 0.1 k.
    np.testing.assert_allclose(
        _floats(deck["geometry.prob_lo"]), [0.0, -0.05 * length], rtol=1e-12
    )
    np.testing.assert_allclose(
        _floats(deck["geometry.prob_hi"]), [2.125 * length, 3.35 * length], rtol=1e-12
    )
    # Field unit m_e c²/(e L); the pulse center 1.0 is the antenna plane.
    assert float(deck["laser.e_max"]) == pytest.approx(
        _ME * _C**2 / (_E * length), rel=1e-12
    )
    np.testing.assert_allclose(
        _floats(deck["laser.position"]), [0.0, 0.0, length], rtol=1e-12
    )
    assert float(deck["laser.wavelength"]) == pytest.approx(length, rel=1e-12)
    assert float(deck["laser.profile_waist"]) == pytest.approx(length, rel=1e-12)
    assert float(deck["laser.profile_duration"]) == pytest.approx(
        0.4 * length / _C, rel=1e-12
    )
    assert float(deck["laser.profile_focal_distance"]) == 0.0
    # Four durations of pre-roll: iteration n + k is the case's step k.
    preroll = math.ceil(4.0 * 0.4 / case.step_size)
    assert S.warpx_preroll_steps(case) == preroll
    assert float(deck["laser.profile_t_peak"]) == pytest.approx(
        preroll * case.step_size * length / _C, rel=1e-12
    )
    assert deck["max_step"] == str(preroll + 4)
    assert deck["fields.intervals"] == f"{preroll}:{preroll + 4}:4"
    # Unit density: macro mass w carries w ε₀ L c² m_e/e² electrons.
    grid = PIC.QuasiCylindricalGrid(2.125, 17, 0.0, 3.4, 34, 2)
    positions, weights = _cylinder_plasma(grid, 2.3, 3.2, 1.5, (1, 1, 2))
    np.testing.assert_allclose(
        _floats(deck["electrons.multiple_particles_weight"]),
        weights * _EPS0 * length * _C**2 * _ME / _E**2,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        _floats(deck["ions.multiple_particles_pos_z"]),
        positions[:, 2] * length,
        rtol=1e-12,
    )
    assert float(deck["ions.mass"]) == pytest.approx(_IMMOBILE * _ME, rel=1e-12)
    assert deck["algo.maxwell_solver"] == "psatd"
    assert deck["psatd.update_with_rho"] == "1"
    assert deck["warpx.use_filter"] == "0"
    assert deck["boundary.field_hi"] == "none damped"


# Refusals -----------------------------------------------------------------------


def _nci_oracle_case() -> Any:
    plan, _, electrons, ions, velocity, _ = _nci_case()
    return S.pic_oracle_case(
        plan,
        _SCALE,
        (electrons, ions),
        (velocity, velocity),
        0.1,
        particle_masses=(_ELECTRON_MASS, _ION_RATIO * _ELECTRON_MASS),
        steps=2,
        output_interval=1,
    )


@pytest.mark.parametrize(
    ("build", "translate", "message"),
    [
        pytest.param(_nci_oracle_case, S.smilei_input, "PSATD", id="smilei-psatd"),
        pytest.param(
            lambda: _oscillation_case((8, 8, 8), 1, steps=2, interval=1)[0],
            S.smilei_input,
            "shape_order",
            id="smilei-linear-shape",
        ),
        pytest.param(_nci_oracle_case, S.picongpu_input, "Yee", id="picongpu-psatd"),
        pytest.param(
            lambda: _oscillation_case((8, 8, 6), 2, steps=2, interval=1)[0],
            S.picongpu_input,
            "multiples",
            id="picongpu-supercell",
        ),
        pytest.param(
            lambda: _oscillation_case((4, 8, 8), 2, steps=2, interval=1)[0],
            S.warpx_input,
            "guard",
            id="warpx-guard-cells",
        ),
        pytest.param(
            _tiny_track_case, S.smilei_input, "openPMD 1.0.0", id="smilei-track"
        ),
        pytest.param(
            _tiny_track_case,
            S.picongpu_input,
            "tracked macroparticle",
            id="picongpu-track",
        ),
        pytest.param(_tiny_wakefield_case, S.smilei_input, "FDTD", id="smilei-rz"),
        pytest.param(
            _tiny_wakefield_case,
            S.picongpu_input,
            "quasi-cylindrical",
            id="picongpu-rz",
        ),
        pytest.param(
            lambda: _wakefield_case(
                *_wakefield(
                    PIC.QuasiCylindricalGrid(2.0, 16, 0.0, 3.4, 34, 2),
                    (2.3, 3.2, 1.5),
                    (1, 1, 2),
                )[::2],
                _TINY_LASER,
                0.05,
                4,
                4,
            ),
            S.warpx_input,
            "guard cells",
            id="warpx-rz-guard-cells",
        ),
    ],
)
def test_providers_refuse_cases_outside_their_subset(
    build: Callable[[], Any], translate: Callable[[Any], Any], message: str
) -> None:
    case = build()
    with pytest.raises(ValueError, match=message):
        translate(case)


def test_picongpu_refuses_species_that_are_not_one_per_cell_lattices() -> None:
    _, _, electrons, ions, velocity, _ = _nci_case((24, 24, 12), 10.0)
    case = S.pic_oracle_case(
        _yee_plan((24, 24, 12), 0.3868, 2),
        _SCALE,
        (electrons, ions),
        (velocity, velocity),
        0.1,
        particle_masses=(_ELECTRON_MASS, _ION_RATIO * _ELECTRON_MASS),
        steps=2,
        output_interval=1,
    )
    with pytest.raises(ValueError, match="in-cell offset"):
        S.picongpu_input(case)


@pytest.mark.parametrize(
    ("scale", "positions_shift", "steps", "message"),
    [
        pytest.param(
            ElectromagneticScaleContract.si(), 0.0, 4, "speed of light", id="scale"
        ),
        pytest.param(_SCALE, 1.6, 4, "inside the periodic box", id="outside-box"),
        pytest.param(_SCALE, 0.0, 3, "multiple of output_interval", id="interval"),
    ],
)
def test_case_refuses_inconsistent_plan_bindings(
    scale: ElectromagneticScaleContract,
    positions_shift: float,
    steps: int,
    message: str,
) -> None:
    plan = _yee_plan((8, 8, 8), 0.2, 2)
    lattice = _lattice((8, 8, 8), 0.2) + positions_shift
    zero = np.zeros_like(lattice)
    with pytest.raises(ValueError, match=message):
        S.pic_oracle_case(
            plan,
            scale,
            (lattice, lattice),
            (zero, zero),
            0.1,
            particle_masses=(_ELECTRON_MASS, _ION_RATIO * _ELECTRON_MASS),
            steps=steps,
            output_interval=2,
        )


def _spectral_tracked_plan() -> Any:
    plan, solver = _spectral_plan((8, 8, 8), 0.3868, 1.0)
    return S.ElectromagneticPICPlan(
        solver, species=plan.species, external_fields=(_field(),)
    )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(
            lambda: _track_plan(fields=(_field(gradient=0.1),))[0],
            "uniform, static",
            id="gradient",
        ),
        pytest.param(
            lambda: _track_plan(fields=(_field(rate=0.01),))[0],
            "uniform, static",
            id="time-dependent",
        ),
        pytest.param(
            lambda: _track_plan(counts=(2, 1), lanes=2)[0],
            "exactly one identity",
            id="two-lanes",
        ),
        pytest.param(
            lambda: _track_plan(counts=(2, 1))[0],
            "only the tracked macroparticle",
            id="shared-species",
        ),
        pytest.param(
            lambda: _track_plan(lanes=0)[0],
            "exactly one PICTrackRecorder",
            id="field-without-recorder",
        ),
        pytest.param(_spectral_tracked_plan, "cochain Yee", id="psatd"),
    ],
)
def test_single_particle_case_refuses_plans_outside_its_scenario(
    build: Callable[[], Any], message: str
) -> None:
    plan = build()
    with pytest.raises(ValueError, match=message):
        _track_case(96, plan)


def _refused_wakefield(
    grid: tuple[float, int, float, float, int, int] = (2.125, 17, 0.0, 3.4, 34, 2),
    *,
    laser: Any = _TINY_LASER,
    radial_shift: float = 0.0,
    **options: Any,
) -> Any:
    plan, _, positions = _wakefield(
        PIC.QuasiCylindricalGrid(*grid), (2.3, 3.2, 1.5), (1, 1, 2), **options
    )
    moved = positions.copy()
    moved[:, 0] += radial_shift
    return _wakefield_case(plan, moved, laser, 0.05, 4, 4)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(
            lambda: _refused_wakefield(laser=S.PICOracleLaser(1.0, 1.0, 1.0, 0.4, 5.0)),
            "laser center",
            id="laser-outside-box",
        ),
        pytest.param(
            lambda: _refused_wakefield((2.125, 17, 0.0, 3.4, 34, 1)),
            "modes 0 and 1",
            id="one-mode",
        ),
        pytest.param(
            lambda: _refused_wakefield(
                absorber="radial-damping", damping=SPECTRAL.RadialDampingPlan(4)
            ),
            "radial damping",
            id="radial-damping",
        ),
        pytest.param(
            lambda: _refused_wakefield(radial_shift=1.0),
            "cylinder",
            id="outside-cylinder",
        ),
        pytest.param(
            lambda: _wakefield_case(
                _yee_plan((8, 8, 8), 0.2, 2),
                _lattice((8, 8, 8), 0.2),
                _TINY_LASER,
                0.05,
                4,
                4,
            ),
            "quasi-cylindrical PSATD solver",
            id="cartesian-solver",
        ),
    ],
)
def test_wakefield_case_refuses_plans_outside_its_scenario(
    build: Callable[[], Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build()


def test_warpx_rz_refuses_particles_beyond_its_shifted_box() -> None:
    plan, _, positions = _tiny_wakefield()
    moved = positions.copy()
    moved[0, 2] = 3.37
    case = _wakefield_case(plan, moved, _TINY_LASER, 0.05, 4, 4)
    with pytest.raises(ValueError, match="half a cell"):
        S.warpx_input(case)


def test_warpx_runners_refuse_the_other_scenarios(tmp_path: Path) -> None:
    provider = S.WarpXProvider(
        pin_executable("/usr/bin/true", version="0", license_id="BSD-3-Clause-LBNL")
    )
    with pytest.raises(ValueError, match="run_warpx_track"):
        S.run_warpx(provider, _tiny_track_case(), tmp_path)
    with pytest.raises(ValueError, match="single-particle-radiation"):
        S.run_warpx_track(provider, _tiny_yee_case(), tmp_path)
    with pytest.raises(TypeError, match="PICOracleWakefieldCase"):
        S.run_warpx_wakefield(provider, _tiny_yee_case(), tmp_path)


def test_smilei_field_format_is_catalogued_read_only() -> None:
    (smilei,) = [
        value
        for value in format_capabilities()
        if value.profile.format == "smilei-fields"
    ]
    assert smilei.directions == ("read",)
    assert smilei.domain == "plasma"


# Parsers on real provider output ---------------------------------------------------


def _resource(path: Path) -> Any:
    return read_bounded_resource(
        path.name,
        trusted_root=path.parent,
        limits=ResourceLimits(4_000_000, 16, 200_000, 65_536, 1),
    )


def _field_units() -> tuple[float, float]:
    """``E`` and ``B`` units of `_code_scale`: ``m_e c²/(e L)`` and ``m_e c/(e L)``."""
    return _ME * _C**2 / (_E * _LENGTH), _ME * _C / (_E * _LENGTH)


def test_warpx_reference_snapshots_import_in_scale_units() -> None:
    import h5py

    case = _tiny_yee_case()
    paths = sorted((_DATA / "warpx").glob("openpmd_*.h5"))
    fields = S.read_pic_oracle_openpmd(case, tuple(_resource(path) for path in paths))
    electric_unit, magnetic_unit = _field_units()
    assert fields.electric.shape == (3, 6, 6, 6, 3)
    np.testing.assert_allclose(fields.times, [0.0, 0.1, 0.2], rtol=1e-12, atol=1e-15)
    assert fields.electric_positions == ((0.5, 0.5, 0.5),) * 3
    with h5py.File(paths[-1], "r") as handle:
        record = handle["data/2/fields/E/x"]
        # WarpX stores C-ordered z, y, x arrays in SI.
        stored = np.transpose(record[()], (2, 1, 0)) * record.attrs["unitSI"]
    np.testing.assert_allclose(
        fields.electric[-1, ..., 0], stored / electric_unit, rtol=1e-12
    )
    # Uniform E_x of the drifting electrons: E ≈ v₀ ω t in the first steps.
    np.testing.assert_allclose(fields.electric[-1, ..., 0], 0.01 * 0.2, rtol=2e-2)
    assert np.all(np.abs(fields.magnetic) < 1e-9 * magnetic_unit)


def test_smilei_reference_snapshots_import_in_scale_units() -> None:
    import h5py

    case = _tiny_yee_case()
    path = _DATA / "smilei" / "Fields0.h5"
    fields = S.read_smilei_fields(case, _resource(path))
    electric_unit, _ = _field_units()
    assert fields.electric.shape == (3, 6, 6, 6, 3)
    np.testing.assert_allclose(fields.times, [0.0, 0.1, 0.2], rtol=1e-12, atol=1e-15)
    # Yee staggering: E_x on x-edges, B_x on x-faces.
    assert fields.electric_positions[0] == (0.5, 0.0, 0.0)
    assert fields.magnetic_positions[0] == (0.0, 0.5, 0.5)
    with h5py.File(path, "r") as handle:
        record = handle["data/0000000002/Ex"]
        stored = record[:6, :6, :6] * record.attrs["unitSI"]
    np.testing.assert_allclose(
        fields.electric[-1, ..., 0], stored / electric_unit, rtol=1e-12
    )
    np.testing.assert_allclose(fields.electric[-1, ..., 0], 0.01 * 0.2, rtol=2e-2)


def test_warpx_reference_track_imports_the_boris_gyration_under_the_recorder_id() -> None:
    case = _tiny_track_case()
    track = S.read_pic_oracle_track(
        case, _resource(_DATA / "warpx" / "single-particle-radiation" / "openpmd.h5")
    )
    step = _WINDOW / 96
    np.testing.assert_allclose(
        np.asarray(track.times[:, 0]), step * np.arange(4), rtol=0.0, atol=1e-15
    )
    assert (int(track.id_hi[0]), int(track.id_lo[0])) == (0, 0)
    np.testing.assert_allclose(
        np.asarray(track.charges * track.multiplicities), [-_TRACK_WEIGHT], rtol=1e-9
    )
    positions = np.asarray(track.positions[:, 0])
    proper = np.asarray(track.proper_velocities[:, 0])
    np.testing.assert_allclose(positions[0], _START, rtol=1e-12)
    np.testing.assert_allclose(proper[0], _ORBIT_GAMMA * _VELOCITY, rtol=1e-12)
    # A Boris step in a pure magnetic field rotates u⊥ counterclockwise (electron,
    # +z field) by 2 arctan(ΩΔt/2) and keeps |u| and u∥ (up to its 1e-9 self field).
    transverse = proper[:, 0] + 1j * proper[:, 1]
    np.testing.assert_allclose(
        np.angle(transverse[1:] / transverse[:-1]),
        2.0 * math.atan(0.5 * _OMEGA * step),
        rtol=1e-9,
    )
    np.testing.assert_allclose(proper[:, 2], _ORBIT_GAMMA * _VELOCITY[2], rtol=1e-8)
    np.testing.assert_allclose(np.abs(transverse), _ORBIT_GAMMA * _VELOCITY[1], rtol=1e-8)


def test_warpx_reference_wakefield_imports_modes_on_the_case_nodes() -> None:
    import h5py

    case = _tiny_wakefield_case()
    paths = sorted((_DATA / "warpx" / "laser-wakefield-stage").glob("openpmd_*.h5"))
    fields = S.read_warpx_wakefield(case, tuple(_resource(path) for path in paths))
    assert fields.electric.shape == (2, 3, 17, 34, 3)
    np.testing.assert_allclose(
        fields.radial_coordinates, 0.125 * (np.arange(17) + 0.5), rtol=1e-12
    )
    np.testing.assert_allclose(fields.axial_coordinates, 0.1 * np.arange(34), atol=1e-12)
    np.testing.assert_allclose(
        fields.times, [0.0, 4.0 * case.step_size], rtol=0.0, atol=1e-12
    )
    iteration = S.warpx_preroll_steps(case) + 4
    with h5py.File(paths[-1], "r") as handle:
        record = handle[f"data/{iteration}/fields/E/z"]
        # WarpX stores C-ordered [mode, z, r] planes in SI.
        stored = np.transpose(record[()], (0, 2, 1)) * record.attrs["unitSI"]
    np.testing.assert_allclose(
        fields.electric[-1, ..., 2],
        stored / (_ME * _C**2 / (_E * _PLASMA_LENGTH)),
        rtol=1e-12,
    )
    # At t = 0 the antenna sheet at the pulse center (node 10) has emitted equal
    # pulses toward ±z: mode-1 E_r (E_x at θ = 0) is even about it, B_θ odd.
    electric = fields.electric[0, 1, :, :, 0]
    magnetic = fields.magnetic[0, 1, :, :, 1]
    size = np.max(np.abs(electric))
    assert size > 0.5
    for offset in range(1, 10):
        np.testing.assert_allclose(
            electric[:, 10 + offset], electric[:, 10 - offset], rtol=0, atol=1e-5 * size
        )
        np.testing.assert_allclose(
            magnetic[:, 10 + offset], -magnetic[:, 10 - offset], rtol=0, atol=1e-5 * size
        )


def test_reference_outputs_record_their_provenance() -> None:
    import hashlib

    for provider in ("warpx", "smilei"):
        provenance = json.loads((_DATA / provider / "provenance.json").read_text())
        assert provenance["provider"] == provider
        assert provenance["version"]
        assert provenance["command"]
    for scenario in ("single-particle-radiation", "laser-wakefield-stage"):
        directory = _DATA / "warpx" / scenario
        provenance = json.loads((directory / "provenance.json").read_text())
        assert (provenance["provider"], provenance["scenario"]) == ("warpx", scenario)
        assert provenance["version"] == "26.01"
        assert provenance["command"] and provenance["date"]
        assert provenance["files"] == {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.h5"))
        }


# Live oracles --------------------------------------------------------------------


def _pinned(variable: str, version: str, license_id: str) -> Any:
    path, release = os.environ.get(variable), os.environ.get(version)
    if path is None or release is None:
        pytest.skip(f"set {variable} and {version} to a pinned provider")
    return pin_executable(path, version=release, license_id=license_id)


def _phydrax_electric_energy(
    plan: Any,
    positions: tuple[np.ndarray, ...],
    velocities: tuple[np.ndarray, ...],
    step: float,
    steps: int,
) -> np.ndarray:
    """Phydrax's electric field energy after every step."""
    state = eqx.filter_jit(lambda: plan.initialize(positions, velocities, step))()

    def body(value: Any, _: None) -> tuple[Any, Any]:
        result = plan.step_detailed(value, step)
        accepted = result.accepted_state
        return accepted, plan.synchronized_energy(accepted, step).electric_field

    _, energy = eqx.filter_jit(
        lambda value: jax.lax.scan(body, value, None, length=steps)
    )(state)
    return np.asarray(energy)


def _oracle_electric_energy(fields: Any, spacing: float) -> np.ndarray:
    return 0.5 * np.sum(fields.electric**2, axis=(1, 2, 3, 4)) * spacing**3


@pytest.mark.parametrize(
    ("variable", "license_id", "provider", "run", "counts"),
    [
        pytest.param(
            "PHYDRAX_WARPX",
            "BSD-3-Clause-LBNL",
            S.WarpXProvider,
            S.run_warpx,
            (16, 8, 8),
            id="warpx",
        ),
        pytest.param(
            "PHYDRAX_SMILEI",
            "CECILL-B",
            S.SmileiProvider,
            S.run_smilei,
            (16, 8, 8),
            id="smilei",
        ),
        pytest.param(
            "PHYDRAX_PICONGPU",
            "GPL-3.0-or-later",
            S.PIConGPUProvider,
            S.run_picongpu,
            (24, 24, 12),
            id="picongpu",
        ),
    ],
)
def test_yee_plasma_oscillation_matches_phydrax_and_the_leapfrog_frequency(
    tmp_path: Path,
    variable: str,
    license_id: str,
    provider: Any,
    run: Any,
    counts: tuple[int, int, int],
) -> None:
    executable = _pinned(variable, f"{variable}_VERSION", license_id)
    steps, interval = 126, 2
    case = _oscillation_case(counts, 2, steps=steps, interval=interval)[0]
    result = run(provider(executable), case, tmp_path, timeout=3600)
    assert result.report.valid
    assert result.provider_version == executable.version
    # The uniform mode does not depend on the box, so Phydrax runs the (16, 8, 8)
    # box and energies are compared per unit volume.
    oracle = _oracle_electric_energy(result.fields, 0.2) / math.prod(counts)
    _, plan, lattice, electrons, ions = _oscillation_case(
        (16, 8, 8), 2, steps=steps, interval=interval
    )
    reference = _phydrax_electric_energy(
        plan, (lattice, lattice), (electrons, ions), 0.1, steps
    ) / math.prod((16, 8, 8))
    assert reference.max() > 0.0
    # Same Yee/Esirkepov discretization of a uniform mode: only unit-conversion
    # round-off (provider constants vs CODATA 2022, ~1e-9) separates the codes.
    np.testing.assert_allclose(
        oracle[1:],
        reference[interval - 1 :: interval],
        rtol=0.0,
        atol=1e-6 * oracle.max(),
    )
    # Leapfrog plasma frequency sin(ωΔt/2) = ω_p Δt/2 with ω_p² = 1 + m_e/m_i.
    mean = result.fields.electric[..., 0].mean(axis=(1, 2, 3))
    omega_p = math.sqrt(1.0 + 1.0 / _ION_RATIO)
    omega = 2.0 / 0.1 * math.asin(0.05 * omega_p)
    basis = np.stack(
        (np.sin(omega * result.fields.times), np.cos(omega * result.fields.times)), -1
    )
    coefficients, *_ = np.linalg.lstsq(basis, mean, rcond=None)
    np.testing.assert_allclose(basis @ coefficients, mean, atol=1e-3 * 0.01)
    assert coefficients[0] == pytest.approx(0.01 * omega_p**2 / omega, rel=2e-3)


_STAGGERED_E = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
_STAGGERED_B = ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))


def _cell_centered(
    values: np.ndarray, positions: tuple[tuple[float, ...], ...]
) -> np.ndarray:
    """Average each staggered component to cell centers along its nodal axes."""
    result = values.copy()
    for component, position in enumerate(positions):
        value = values[..., component]
        for axis, offset in enumerate(position):
            if offset == 0.0:
                value = 0.5 * (value + np.roll(value, -1, axis=axis + 1))
        result[..., component] = value
    return result


def test_warpx_psatd_drifting_plasma_grows_numerical_cherenkov_like_phydrax(
    tmp_path: Path,
) -> None:
    executable = _pinned("PHYDRAX_WARPX", "PHYDRAX_WARPX_VERSION", "BSD-3-Clause-LBNL")
    counts, spacing, steps = (24, 8, 12), 0.3868, 80
    step = 0.45 * spacing
    plan, solver, electrons, ions, velocity, _ = _nci_case(counts, 10.0, grid="staggered")
    case = S.pic_oracle_case(
        plan,
        _SCALE,
        (electrons, ions),
        (velocity, velocity),
        step,
        particle_masses=(_ELECTRON_MASS, _ION_RATIO * _ELECTRON_MASS),
        steps=steps,
        output_interval=2,
    )
    result = S.run_warpx(S.WarpXProvider(executable), case, tmp_path, timeout=1800)
    assert result.fields.electric_positions == ((0.5, 0.5, 0.5),) * 3
    state = eqx.filter_jit(
        lambda: plan.initialize((electrons, ions), (velocity, velocity), step)
    )()

    def body(value: Any, _: None) -> tuple[Any, tuple[Any, Any]]:
        field = plan.step_detailed(value, step).accepted_state
        return field, (field.field.electric, field.field.magnetic)

    _, (electric, magnetic) = eqx.filter_jit(
        lambda value: jax.lax.scan(body, value, None, length=steps)
    )(state)
    monitor = SPECTRAL.SpectralNCIMonitorPlan(solver, high_fraction=0.3)

    def high_energy(e: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                float(
                    monitor.sample(
                        SPECTRAL.SpectralMaxwellState(
                            jnp.asarray(x),
                            jnp.asarray(y),
                            jnp.zeros(counts),
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            (),
                        )
                    ).high_energy
                )
                for x, y in zip(e, b, strict=True)
            ]
        )

    # WarpX writes cell-centered averages (declared loss); Phydrax's staggered
    # fields receive the same averaging so both sample one observable.
    phydrax = high_energy(
        _cell_centered(np.asarray(electric), _STAGGERED_E),
        _cell_centered(np.asarray(magnetic), _STAGGERED_B),
    )
    warpx = high_energy(result.fields.electric, result.fields.magnetic)
    times = step * (np.arange(steps) + 1.0)
    window = {"start": 7.0, "stop": 13.9}
    fit = SPECTRAL.SpectralNCIMonitorPlan.fit
    grown = float(fit(times, phydrax, **window).rate)
    oracle = float(fit(result.fields.times, warpx, **window).rate)
    # Same staggered PSATD/Esirkepov scheme; the codes differ only in the initial
    # Gauss field WarpX omits, whose transient has decayed by t = 7.
    assert grown > 0.1
    assert oracle == pytest.approx(grown, rel=0.25)
    assert warpx[-1] == pytest.approx(phydrax[-1], rel=0.5)


def _window(track: Any, steps: int) -> ChargedTrajectory:
    """The first ``steps`` samples of a one-lane track."""
    return ChargedTrajectory(
        track.times[:steps],
        track.positions[:steps],
        track.proper_velocities[:steps],
        track.charges,
        track.multiplicities,
        track.active[:steps],
        (track.id_hi, track.id_lo),
    )


def _recorded_track(plan: Any, recorder: Any, steps: int) -> ChargedTrajectory:
    step = _WINDOW / steps
    positions, velocities = _track_inputs(plan)
    state = plan.initialize(positions, velocities, step)
    advance = eqx.filter_jit(plan.step_detailed)
    for _ in range(steps):
        result = advance(state, step)
        assert bool(result.successful)
        state = result.accepted_state
    return _window(recorder.to_charged_trajectory(state.recorders[0], _SCALE), steps)


def _helix(times: np.ndarray, track: ChargedTrajectory) -> ChargedTrajectory:
    """Exact relativistic helix (Jackson §12.2); the electron turns counterclockwise."""
    rotation = np.exp(1j * _OMEGA * times)
    transverse = complex(_VELOCITY[0], _VELOCITY[1])
    plane = complex(_START[0], _START[1]) + transverse * (rotation - 1.0) / (1j * _OMEGA)
    swept = transverse * rotation
    positions = np.stack(
        [plane.real, plane.imag, _START[2] + _VELOCITY[2] * times], axis=-1
    )
    proper = _ORBIT_GAMMA * np.stack(
        [swept.real, swept.imag, np.full_like(times, _VELOCITY[2])], axis=-1
    )
    return ChargedTrajectory(
        times,
        positions[:, None],
        proper[:, None],
        np.asarray(track.charges),
        np.asarray(track.multiplicities),
        np.ones((times.shape[0], 1), dtype=np.bool_),
        (np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint32)),
    )


def _band_errors(value: Any, reference: Any) -> tuple[float, float]:
    """Fundamental- and second-harmonic-band relative field-spectrum errors."""
    difference = np.asarray(value.field_spectrum - reference.field_spectrum)
    exact = np.asarray(reference.field_spectrum)
    return (
        float(np.linalg.norm(difference[:3]) / np.linalg.norm(exact[:3])),
        float(np.linalg.norm(difference[3:]) / np.linalg.norm(exact[3:])),
    )


def test_warpx_tracked_cyclotron_spectrum_matches_the_phydrax_recorder(
    tmp_path: Path,
) -> None:
    """A1 spectra of the WarpX track against the PICTrackRecorder track and the helix.

    Both codes push the electron with Boris in the same uniform field; positions
    agree to round-off. Velocity samples differ: WarpX half-pushes u^{k−1/2} to
    t_k (|u| kept), the recorder averages (u^{k−1/2} + u^{k+1/2})/2, whose modulus
    is |u| cos(θ/2) with θ = 2 arctan(ΩΔt/2), i.e. short by (ΩΔt)²/8. Harmonic m
    of the radiated field scales as β⊥^m, so the WarpX–Phydrax band error is at
    most m(ΩΔt)²/8 and second order in Δt. Against the exact helix both carry
    the Boris phase lag m(ΩT)³/(24N²) (A1 ↔ P row; 1.25 × its measured
    coefficients 736 and 1621).
    """
    executable = _pinned("PHYDRAX_WARPX", "PHYDRAX_WARPX_VERSION", "BSD-3-Clause-LBNL")
    frequencies = _OMEGA * np.asarray([0.9, 1.0, 1.1, 1.9, 2.0, 2.1])
    radiation = TrajectoryRadiationPlan(
        _SCALE,
        RadiationObserverPlan(
            np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.6, 0.0, 0.8]]),
            np.asarray([0.0, 1.0, 0.0]),
        ),
        frequencies,
        coherence="coherent",
        route="segment-exact",
        emission="truncated",
    ).prepare()
    discrepancies = []
    for steps in (192, 384):
        plan, recorder = _track_plan(capacity=steps)
        case = _track_case(steps, plan)
        destination = tmp_path / str(steps)
        destination.mkdir()
        result = S.run_warpx_track(
            S.WarpXProvider(executable), case, destination, timeout=600
        )
        assert result.report.valid
        assert result.provider_version == executable.version
        warpx = _window(result.track, steps)
        phydrax = _recorded_track(plan, recorder, steps)
        times = np.asarray(phydrax.times[:, 0])
        np.testing.assert_allclose(np.asarray(warpx.times[:, 0]), times, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(warpx.positions), np.asarray(phydrax.positions), atol=1e-10
        )
        exact = radiation.evaluate(_helix(times, phydrax))
        oracle = radiation.evaluate(warpx)
        fundamental, second = _band_errors(oracle, radiation.evaluate(phydrax))
        lag = (_OMEGA * _WINDOW / steps) ** 2
        assert fundamental < lag / 8.0
        assert second < 2.0 * lag / 8.0
        helix = _band_errors(oracle, exact)
        assert helix[0] < 1.25 * 736.0 / steps**2
        assert helix[1] < 1.25 * 1621.0 / steps**2
        discrepancies.append((fundamental, second))
    orders = np.log2(np.asarray(discrepancies[0]) / np.asarray(discrepancies[1]))
    assert np.all((orders > 1.8) & (orders < 2.2)), orders


def _phydrax_wake(
    plan: Any, solver: Any, positions: np.ndarray, laser: Any, step: float, steps: int
) -> np.ndarray:
    """On-axis mode-0 E_z of Phydrax P3 after ``steps`` steps from the focused pulse."""
    grid = solver.grid
    zero = np.zeros_like(positions)
    state = plan.initialize((positions, positions), (zero, zero), step)
    radius = grid.radial_coordinates[:, None]
    axial = grid.axial_coordinates[None, :] - laser.center
    pulse = np.zeros((grid.mode_count, grid.radial_count, grid.axial_count, 3), complex)
    pulse[1, :, :, 0] = (
        laser.amplitude
        * np.exp(-(radius**2) / laser.waist**2 - axial**2 / laser.length**2)
        * np.cos(2.0 * np.pi / laser.wavelength * axial)
    )
    state = eqx.tree_at(
        lambda value: value.field, state, solver.add_propagating_field(state.field, pulse)
    )
    advance = eqx.filter_jit(lambda value: plan.step_detailed(value, step))
    for _ in range(steps):
        result = advance(state)
        assert bool(result.successful)
        state = result.accepted_state
    return np.real(np.asarray(state.field.electric[0, 0, :, 2]))


def test_warpx_rz_laser_wakefield_converges_to_phydrax_p3(tmp_path: Path) -> None:
    """Linear LWFA (a₀ = 0.5, k₀ = 5k_p; the FBPIC test's P3 configuration) in WarpX RZ.

    Phydrax runs once at Δt (step-converged: halving Δt moves its wake by 0.7 %).
    WarpX runs at Δt, Δt/2, Δt/4: its antenna, damped axial boundaries, local
    FFTs, and direct deposition with rho update (declared losses) leave a step
    error that decays with a measured order p ≈ 0.7. Error model: the WarpX–P3
    discrepancy at Δt/4 stays within the P3 ↔ FBPIC cross-code level of this
    configuration (5.6 %, bound 6 %) plus WarpX's Richardson step error
    ‖w_{Δt/2} − w_{Δt/4}‖/(2^p − 1), and shrinks under every refinement.
    """
    executable = _pinned("PHYDRAX_WARPX_RZ", "PHYDRAX_WARPX_VERSION", "BSD-3-Clause-LBNL")
    k0 = 5.0
    length = 120 * (2.0 * np.pi / k0) / 12.0
    grid = PIC.QuasiCylindricalGrid(6.0, 48, 0.0, length, 120, 2)
    plan, solver, positions = _wakefield(
        grid,
        (5.0, length - 0.2, 4.5),
        (2, 2, 4),
        charge_conservation="spectral-correction",
    )
    laser = S.PICOracleLaser(0.5 * k0, 2.0 * np.pi / k0, 2.5, 1.0, 3.0)
    step = 0.9 * float(solver.stable_step)
    steps = round(5.5 / step)
    ours = _phydrax_wake(plan, solver, positions, laser, step, steps)
    wake = (grid.axial_coordinates > 5.4) & (grid.axial_coordinates < 9.5)
    wakes = []
    for refine in (1, 2, 4):
        case = _wakefield_case(
            plan, positions, laser, step / refine, steps * refine, steps * refine
        )
        destination = tmp_path / str(refine)
        destination.mkdir()
        result = S.run_warpx_wakefield(
            S.WarpXProvider(executable), case, destination, timeout=1800
        )
        assert result.report.valid
        np.testing.assert_allclose(result.fields.times[-1], steps * step, rtol=1e-9)
        wakes.append(result.fields.electric[-1, 0, 0, :, 2][wake])
    reference = ours[wake]
    assert np.max(np.abs(reference)) > 0.03

    def relative(value: np.ndarray, target: np.ndarray) -> float:
        return float(np.linalg.norm(value - target) / np.linalg.norm(target))

    errors = [relative(value, reference) for value in wakes]
    assert errors[0] > errors[1] > errors[2], errors
    first, second = relative(wakes[0], wakes[1]), relative(wakes[1], wakes[2])
    order = math.log2(first / second)
    assert 0.3 < order < 1.2, order
    assert errors[2] < 0.06 + second / (2.0**order - 1.0), errors
    assert np.max(np.abs(wakes[2])) == pytest.approx(np.max(np.abs(reference)), rel=0.15)
