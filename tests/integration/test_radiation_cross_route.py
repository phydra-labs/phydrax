#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cross-route consistency release matrix of the charged-particle radiation stack.

Each test drives two or more independently implemented routes through one
physical scenario and asserts that they agree on one observable. The routes
are compared at two or more resolutions. The tolerance at the finest one is
set from the measured coefficient of the stated error model, so every bound is
justified by measured convergence. Rows asserted in this module:

- A1 ↔ P: an electron tracked by electromagnetic PIC and recorded by
  ``PICTrackRecorder``, radiated through ``TrajectoryRadiationPlan``, against
  the same plan on the exact helix.
- P1 ↔ P2 ↔ P3: one axisymmetric TM pulse on the cochain, Cartesian-PSATD and
  quasi-cylindrical PIC field solvers, against the closed-form solution.
- Q1 ↔ Q2: the χ → 0 limit of the nonlinear-Compton Monte Carlo energy loss
  against classical Landau–Lifshitz radiation reaction.
- C2 ↔ A1: thermal cyclotron harmonic emissivity against trajectory radiation
  of helices averaged over the same Jüttner distribution.
- M2 ↔ B2: Cherenkov photon yield of the optical source against the Poynting
  flux of the moving-charge field divided by ``ħω``.

The A1 ↔ B1, B2 ↔ B4, X4 steady ↔ retarded-mesh and X5 ↔ X6 rows are owned by
their milestone tests. ``radiation.cross-route-release-matrix`` in the
qualification catalog names every row's test node IDs as its required gates.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from scipy import special
from scipy.special import erf

import phydrax as phx
from phydrax import ElectromagneticScaleContract
from phydrax.discretization.pic import (
    ExternalFieldSample,
    NonlinearComptonPlan,
    PIC_CODE_RELATIVITY,
    PICSpeciesPlan,
    PICTrackRecorder,
    QEDEventFlag,
    QEDTable,
    RadiationReactionFlag,
    RadiationReactionPlan,
)
from phydrax.electromagnetics import (
    ChargedTrajectory,
    ColdPlasmaDielectric,
    MagnetobremsstrahlungPlan,
    MagnetobremsstrahlungResult,
    PreparedTrajectoryRadiation,
    RadiationObserverPlan,
    ThermalJuttnerDistribution,
    TrajectoryRadiationPlan,
    TrajectoryRadiationResult,
    TrajectoryRadiationStatus,
    UniformMotionFieldPlan,
    UniformMotionMedium,
)
from phydrax.optics.transport import (
    ChargedOpticalSteps,
    CherenkovEmission,
    emit_optical_photons,
    OpticalEmissionStatus,
    OpticalPhotonSourcePlan,
    SpectralOpticalMedium,
)
from phydrax.qualification._radiation_release_matrix import (
    radiation_release_matrix_candidate_profiles,
)
from phydrax.typing import PRNGKey
from phydrax.units import CHARGE, UnitDefinition


_SI_SCALE = ElectromagneticScaleContract.si()
_ELEMENTARY_CHARGE = float(_SI_SCALE.elementary_charge)


# A1 ↔ P


_MAGNETIC = 10.0


_START = np.asarray([0.5, 0.5, 0.3], dtype=np.float64)


_VELOCITY = np.asarray([0.0, 0.4, 0.1], dtype=np.float64)


_ORBIT_GAMMA = 1.0 / np.sqrt(1.0 - float(_VELOCITY @ _VELOCITY))


_OMEGA = _MAGNETIC / _ORBIT_GAMMA


_WINDOW = 4.0 * 2.0 * np.pi / _OMEGA


_WEIGHT = 1.0e-9


_STEPS = (96, 192, 384)


_FUNDAMENTAL = _OMEGA * np.asarray([0.9, 1.0, 1.1], dtype=np.float64)


_SECOND_HARMONIC = _OMEGA * np.asarray([1.9, 2.0, 2.1], dtype=np.float64)


# Measured relative discrepancies at N = 96/192/384:
#   fundamental field ‖ΔF‖/‖F‖  7.814e-2, 1.985e-2, 4.988e-3 (N²·e = 720, 731, 736)
#   second-harmonic field       1.698e-1, 4.366e-2, 1.099e-2 (N²·e = 1565, 1609, 1621)
#   spectral energy max|ΔW|/max W  1.358e-1, 3.640e-2, 9.302e-3 (N²·e = 1252, 1342, 1372)
# The phase-lag estimate m(ΩT)³/(24N²) gives 661 m. Tolerances are 1.25× the
# measured finest coefficient; observed orders are 1.98–1.99 (energy 1.90–1.97).
_FUNDAMENTAL_COEFFICIENT = 1.25 * 736.0


_SECOND_COEFFICIENT = 1.25 * 1621.0


_ENERGY_COEFFICIENT = 1.25 * 1372.0


_ORDER_RANGE = (1.8, 2.2)


class _UniformMagneticField(phx.StrictModule):
    magnetic: Array

    @property
    def source_id(self) -> str:
        return "a1-pic-uniform-axial-field"

    def external_fields(self, positions: Array, times: Array, /) -> ExternalFieldSample:
        del times
        return ExternalFieldSample(
            jnp.zeros_like(positions),
            jnp.broadcast_to(self.magnetic, positions.shape),
            jnp.ones(positions.shape[:1], dtype=jnp.bool_),
        )


def _code_scale() -> ElectromagneticScaleContract:
    return ElectromagneticScaleContract.code_units(
        PIC_CODE_RELATIVITY.dimensional_scale,
        UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
        gravitational_constant=1,
        speed_of_light=1,
        reduced_planck_constant=1,
        boltzmann_constant=1,
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=1,
        constant_set_id="a1-pic-cross-route",
    )


def _species() -> tuple[PICSpeciesPlan, ...]:
    species = []
    for identity, specific, name in ((0, -1.0, "electrons"), (100, 1.0, "ions")):
        support = phx.discretization.ParticleSetPlan(
            jnp.asarray([identity]),
            jnp.full((1,), _WEIGHT, dtype=jnp.float64),
            ambient_dimension=3,
        ).prepare()
        species.append(
            PICSpeciesPlan(
                phx.discretization.ParticlePopulationPlan(support),
                phx.discretization.pic.PICChargeModelPlan(
                    specific,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    return tuple(species)


def _field_solver(species: tuple[PICSpeciesPlan, ...]) -> Any:
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float64))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    transfer_plan = phx.discretization.pic.PICParticleCochainTransferPlan(bridge)
    transfers = tuple(
        transfer_plan.prepare(
            phx.discretization.ChargedParticlePlan(
                plan.charge_model.base_specific_charge * jnp.ones((1,), jnp.float64),
                plan.species_id,
            ).prepare(plan.population.particles)
        )
        for plan in species
    )
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        sources=(phx.solver.PICMaxwellCurrentSourcePlan(),),
        plan_id="a1-pic-cross-route-maxwell",
    ).prepare()
    return phx.solver.CochainMaxwellPICFieldSolver(
        maxwell,
        phx.solver.CochainElectrostaticPlan(
            bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        transfers,
        tuple(phx.discretization.pic.ChargeConservingCurrentPlan(v) for v in transfers),
    )


def _pic_plan(
    species: tuple[PICSpeciesPlan, ...],
) -> tuple[phx.solver.ElectromagneticPICPlan, PICTrackRecorder]:
    """One run plan for every resolution, so a single step trace serves all."""
    recorder = PICTrackRecorder(
        species,
        [0],
        (np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint32)),
        relativity=PIC_CODE_RELATIVITY,
        sample_capacity=max(_STEPS),
    )
    pic = phx.solver.ElectromagneticPICPlan(
        _field_solver(species),
        species=species,
        recorders=(recorder,),
        external_fields=(
            _UniformMagneticField(jnp.asarray([0.0, 0.0, _MAGNETIC], dtype=jnp.float64)),
        ),
    )
    return pic, recorder


def _pic_track(
    pic: phx.solver.ElectromagneticPICPlan,
    recorder: PICTrackRecorder,
    advance: Callable[..., Any],
    steps: int,
) -> ChargedTrajectory:
    """Run ``steps`` PIC steps over the window; return the recorded electron.

    Rows past ``steps`` were never written and are inactive; they are dropped
    so the lane ends at the window edge rather than at an activity boundary.
    """
    dt = _WINDOW / steps
    assert dt < float(pic.solver.stable_step)
    start = jnp.asarray(_START[None], dtype=jnp.float64)
    state = pic.initialize(
        (start, start),
        (jnp.asarray(_VELOCITY[None], dtype=jnp.float64), jnp.zeros((1, 3), jnp.float64)),
        dt,
    )
    successful = True
    for _ in range(steps):
        result = advance(state, dt)
        successful = successful and bool(result.successful)
        state = result.accepted_state
    assert successful
    track = recorder.to_charged_trajectory(state.recorders[0], _code_scale())
    assert not bool(jnp.any(track.active[steps:]))
    return ChargedTrajectory(
        track.times[:steps],
        track.positions[:steps],
        track.proper_velocities[:steps],
        track.charges,
        track.multiplicities,
        track.active[:steps],
        (track.id_hi, track.id_lo),
    )


def _exact_helix(times: np.ndarray) -> ChargedTrajectory:
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
        np.asarray([-1.0], dtype=np.float64),
        np.asarray([_WEIGHT], dtype=np.float64),
        np.ones((times.shape[0], 1), dtype=np.bool_),
        (np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint32)),
    )


def _radiation() -> PreparedTrajectoryRadiation:
    observers = RadiationObserverPlan(
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.6, 0.0, 0.8]], np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    )
    return TrajectoryRadiationPlan(
        _code_scale(),
        observers,
        np.concatenate([_FUNDAMENTAL, _SECOND_HARMONIC]),
        coherence="coherent",
        route="segment-exact",
        emission="truncated",
    ).prepare()


def _assert_trajectory_radiation_clean(result: TrajectoryRadiationResult) -> None:
    # The gyration never stops, so the truncated window's edge acceleration is
    # the only reported condition.
    assert int(result.evidence.status) == int(
        TrajectoryRadiationStatus.WINDOW_EDGE_ACCELERATION
    )
    assert bool(result.evidence.resolved)
    assert float(result.evidence.maximum_phase_increment) < 1.0


def _relative_errors(
    pic: TrajectoryRadiationResult, exact: TrajectoryRadiationResult
) -> tuple[float, float, float]:
    """Fundamental- and second-harmonic-band field errors and the spectral-energy error.

    Field errors are Frobenius norms over each band's frequencies, observers, and
    basis components; the energy error is normalized per frequency by its
    brightest observer, so the weaker second harmonic is not masked.
    """
    difference = np.asarray(pic.field_spectrum - exact.field_spectrum)
    reference = np.asarray(exact.field_spectrum)
    bands = len(_FUNDAMENTAL)
    energy = np.abs(np.asarray(pic.spectral_energy - exact.spectral_energy))
    brightest = np.max(np.asarray(exact.spectral_energy), axis=1, keepdims=True)
    return (
        float(np.linalg.norm(difference[:bands]) / np.linalg.norm(reference[:bands])),
        float(np.linalg.norm(difference[bands:]) / np.linalg.norm(reference[bands:])),
        float(np.max(energy / brightest)),
    )


def test_pic_tracked_cyclotron_spectrum_converges_to_a1_on_exact_helix() -> None:
    """PIC-tracked cyclotron electron radiated through A1 against the exact helix.

    Scenario (code units, ``c = ε₀ = e = m_e = 1``): one electron macroparticle
    (specific charge ``−1``) starts at ``(0.5, 0.5, 0.3)`` with velocity
    ``(0, 0.4, 0.1)`` (``γ ≈ 1.0976``) in a uniform external field ``B = 10 ẑ``
    inside `ElectromagneticPICPlan` over a periodic ``4³`` cochain Maxwell grid of
    the unit box. A resting ion of the same weight neutralizes the periodic
    Poisson start. The run covers four relativistic cyclotron periods at
    ``N ∈ {96, 192, 384}`` steps.

    Route P: `PICTrackRecorder` samples the electron identity in the run (sample
    ``k`` holds ``x^k`` and ``(u^{k−1/2} + u^{k+1/2})/2``), `to_charged_trajectory`
    converts it with the code-unit `ElectromagneticScaleContract`, and A1
    (`TrajectoryRadiationPlan`, ``segment-exact``, coherent, truncated window)
    radiates it.

    Reference: the same A1 plan on the exact relativistic helix,
    ``u⊥(t) = γ v⊥ e^{iΩt}`` with ``Ω = |q|B/(γm)`` (an electron in ``+ẑ``
    rotates counterclockwise), ``x⊥(t) = x⊥(0) + v⊥ (e^{iΩt} − 1)/(iΩ)`` and
    ``z(t) = z(0) + v∥ t`` (Jackson, *Classical Electrodynamics*, 3rd ed., §12.2),
    sampled at the recorded times with the recorded charge weight. Both
    evaluations share frequencies, observers, window, and segment rule, so the
    spectral difference is owned by the PIC trajectory alone.

    Error model: the default Boris pusher rotates ``u⊥`` by
    ``2 arctan(ΩΔt/2) = ΩΔt − (ΩΔt)³/12 + …`` per step, so the phase lag grows
    linearly to ``φ_T = (ΩT)(ΩΔt)²/12`` over the window ``T``. For harmonic
    ``m`` the windowed spectrum then misses by ``≈ m φ_T / 2 = m (ΩT)³/(24 N²)``
    (``≈ 661 m / N²`` here); the time-centred recorder velocity and the Boris
    orbit radius add further ``O((ΩΔt)²)`` amplitude terms. The discrepancy is
    therefore second order in ``Δt`` with a measured coefficient slightly above
    the phase-lag estimate.

    Self-field: the macro weight ``10⁻⁹`` makes the grid self-force ``~10⁻⁹``
    of the magnetic force. Raising the weight to ``10⁻⁶`` moved the finest
    fundamental-band discrepancy by ``3 × 10⁻⁷`` absolute (from
    ``4.98767 × 10⁻³``), so at ``10⁻⁹`` it is ``~3 × 10⁻¹⁰``, far below the
    pusher error.
    """
    pic_plan, recorder = _pic_plan(_species())
    advance = eqx.filter_jit(pic_plan.step_detailed)
    radiation = _radiation()
    errors = []
    for steps in _STEPS:
        track = _pic_track(pic_plan, recorder, advance, steps)
        times = np.asarray(track.times[:, 0])
        np.testing.assert_allclose(
            times, np.arange(steps) * (_WINDOW / steps), rtol=0.0, atol=1e-13
        )
        assert bool(np.all(np.asarray(track.active)))
        np.testing.assert_allclose(
            np.asarray(track.charges * track.multiplicities), [-_WEIGHT], rtol=1e-15
        )
        pic = radiation.evaluate(track)
        exact = radiation.evaluate(_exact_helix(times))
        _assert_trajectory_radiation_clean(pic)
        _assert_trajectory_radiation_clean(exact)
        errors.append(_relative_errors(pic, exact))

    measured = np.asarray(errors)
    orders = np.log2(measured[:-1] / measured[1:])
    assert np.all((orders > _ORDER_RANGE[0]) & (orders < _ORDER_RANGE[1])), orders
    finest = _STEPS[-1] ** 2
    fundamental, second, energy = measured[-1]
    assert fundamental < _FUNDAMENTAL_COEFFICIENT / finest
    assert second < _SECOND_COEFFICIENT / finest
    assert energy < _ENERGY_COEFFICIENT / finest


# P1 ↔ P2 ↔ P3


D = phx.discretization


sp = phx.solver.maxwell.spectral


_RADIUS = 0.35  # Gaussian radius ``a`` of ψ₀.


_DURATION = 0.6


_BOX = 3.0


_SPECTRAL_SPACING = 0.1


_PROBE_RADII = (0.25, 0.45, 0.65)


_PROBE_AXIAL = (-0.4, -0.2, 0.0, 0.2, 0.4)


def _probes() -> np.ndarray:
    """Points ``(r, 0, z)``: nodes of both PSATD grids, off the pulse center."""
    r, z = np.meshgrid(
        np.asarray(_PROBE_RADII, dtype=np.float64),
        np.asarray(_PROBE_AXIAL, dtype=np.float64),
        indexing="ij",
    )
    return np.stack((r.ravel(), np.zeros(r.size), z.ravel()), axis=-1)


def _exact(points: np.ndarray, time: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ``E`` and ``B`` of the spherically symmetric wave solution."""
    a2 = _RADIUS**2

    def g(s: Array) -> Array:
        return jnp.exp(-(s**2) / a2)

    def psi(point: Array, t: Array) -> Array:
        rho = jnp.linalg.norm(point)
        return ((rho - t) * g(rho - t) + (rho + t) * g(rho + t)) / (2.0 * rho)

    def phi(point: Array, t: Array) -> Array:
        rho = jnp.linalg.norm(point)
        return a2 * (g(rho - t) - g(rho + t)) / (4.0 * rho)

    def fields(point: Array) -> tuple[Array, Array]:
        t = jnp.asarray(time, dtype=jnp.float64)
        gradient = jax.grad(psi)(point, t)
        magnetic = jnp.stack((gradient[1], -gradient[0], jnp.zeros(())))
        electric = jax.hessian(phi)(point, t)[:, 2] - jnp.asarray(
            [0.0, 0.0, 1.0]
        ) * jax.grad(psi, argnums=1)(point, t)
        return electric, magnetic

    electric, magnetic = jax.vmap(fields)(jnp.asarray(points, dtype=jnp.float64))
    return np.asarray(electric), np.asarray(magnetic)


def _initial_magnetic(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``B₀ = ∇ × (ψ₀ẑ) = (∂_yψ₀, −∂_xψ₀, 0)`` on broadcast coordinates."""
    psi = np.exp(-(x**2 + y**2 + z**2) / _RADIUS**2)
    scale = 2.0 / _RADIUS**2
    return np.stack((-scale * y * psi, scale * x * psi, np.zeros_like(psi)), axis=-1)


def _bridge(spacing: float) -> D.StructuredCochainBridge:
    """Periodic box of side ``_BOX``; ``x`` nodes at half, ``y``/``z`` at whole cells."""
    count = round(_BOX / spacing)
    half = 0.5 * count * spacing
    lower = np.asarray([-half - 0.5 * spacing, -half, -half], dtype=np.float64)
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(count, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(np.stack((lower, lower + count * spacing))))
    return D.StructuredCochainBridge(grid)


def _node_axes(bridge: D.StructuredCochainBridge) -> tuple[np.ndarray, ...]:
    return tuple(
        float(axis.bounds[0])
        + float(axis.interval_widths[0]) * np.arange(axis.interval_centers.size)
        for axis in bridge.grid.structured_axes
    )


def _particles() -> D.ParticleDiscretization:
    """One probe particle per probe point; only their positions are used."""
    capacity = _probes().shape[0]
    return D.ParticleSetPlan(
        jnp.arange(capacity),
        jnp.ones((capacity,), dtype=jnp.float64),
        ambient_dimension=3,
    ).prepare()


def _cochain_transfer(
    bridge: D.StructuredCochainBridge,
) -> D.pic.PreparedPICParticleCochainTransfer:
    particles = _particles()
    charged = D.ChargedParticlePlan(
        -jnp.ones((_probes().shape[0],), dtype=jnp.float64), "probes"
    ).prepare(particles)
    return D.pic.PICParticleCochainTransferPlan(bridge, shape_order=1).prepare(charged)


class _RouteResult(eqx.Module):
    """Probe fields of one route at ``T`` and its solver evidence."""

    electric: Array
    magnetic: Array
    support: Array
    successful: Array
    constraint: Array


def _run(
    solver: phx.solver.AbstractPreparedPICFieldSolver,
    magnetic: Array,
    charge: Array,
    source: Any,
) -> _RouteResult:
    """Initialize ``E = 0``, ``B = B₀``; advance to ``T``; gather at the probes.

    One compiled program per route with the solver as a traced argument (closing
    over it would make XLA constant-fold its prepared tables). Field states and
    sources are each solver's own layout, typed ``Any`` by the PIC protocol.
    """
    steps = int(np.ceil(_DURATION / (0.9 * float(solver.stable_step))))
    step = jnp.asarray(_DURATION / steps, dtype=jnp.float64)
    probes = jnp.asarray(_probes())

    def program(
        solver: phx.solver.AbstractPreparedPICFieldSolver,
        magnetic: Array,
        charge: Array,
        source: Any,
    ) -> _RouteResult:
        field, initialized = solver.initialize_field(charge, magnetic=magnetic)

        def body(value: Any, index: Array) -> tuple[Any, tuple[Array, Array]]:
            result = solver.advance(index * step, value, source, step)
            constraint = jnp.maximum(
                result.electric_constraint, result.magnetic_constraint
            )
            return result.field, (result.successful, constraint)

        final, (successful, constraint) = jax.lax.scan(
            body, field, jnp.arange(steps, dtype=jnp.float64)
        )
        electric, magnetic_, support = solver.gather_fields(
            0, probes, jnp.ones((probes.shape[0],), dtype=jnp.bool_), final
        )
        return _RouteResult(
            electric,
            magnetic_,
            support,
            initialized & jnp.all(successful),
            jnp.max(constraint),
        )

    return eqx.filter_jit(program)(solver, magnetic, charge, source)


def _cochain_route(bridge: D.StructuredCochainBridge) -> _RouteResult:
    """P1: compatible-cochain Maxwell PIC field solver on ``bridge``."""
    transfer = _cochain_transfer(bridge)
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge, sources=(phx.solver.PICMaxwellCurrentSourcePlan(),)
    ).prepare()
    solver = phx.solver.CochainMaxwellPICFieldSolver(
        maxwell,
        phx.solver.CochainElectrostaticPlan(
            bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        (transfer,),
        (D.pic.ChargeConservingCurrentPlan(transfer),),
    )
    x, y, z = _node_axes(bridge)
    spacing = z[1] - z[0]
    # Exact z-edge integrals of A = ψ₀ẑ; B₀'s face fluxes are their discrete curl.
    along = (
        0.5
        * np.sqrt(np.pi)
        * _RADIUS
        * (erf((z + spacing) / _RADIUS) - erf(z / _RADIUS))[None, None, :]
        * np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / _RADIUS**2)[:, :, None]
    )
    zero = np.zeros_like(along)
    flux = bridge.exterior_derivative(1, bridge.pack(1, (zero, zero, along)))
    return _run(
        solver,
        flux,
        jnp.zeros((bridge.cochain.cell_counts[0],), dtype=jnp.float64),
        jnp.zeros((maxwell.primary_counts[0],), dtype=jnp.float64),
    )


def _cartesian_route(bridge: D.StructuredCochainBridge) -> _RouteResult:
    """P2: collocated infinite-order Cartesian PSATD on ``bridge``."""
    transfer = _cochain_transfer(bridge)
    solver = sp.SpectralMaxwellPlan(bridge).prepare(
        (transfer,), (D.pic.ChargeConservingCurrentPlan(transfer),)
    )
    x, y, z = _node_axes(bridge)
    shape = (x.size, y.size, z.size)
    magnetic = _initial_magnetic(x[:, None, None], y[None, :, None], z[None, None, :])
    source = sp.SpectralMaxwellSource(
        jnp.zeros((1, *shape, 3), dtype=jnp.float64),
        jnp.zeros((1, *shape), dtype=jnp.float64),
    )
    return _run(
        solver, jnp.asarray(magnetic), jnp.zeros(shape, dtype=jnp.float64), source
    )


def _quasi_cylindrical_route() -> _RouteResult:
    """P3: m = 0 quasi-cylindrical PSATD; the wall at r = 2.4 sees g(1.8) ≈ 3e-12."""
    radial = 24
    axial = round(_BOX / _SPECTRAL_SPACING)
    grid = D.pic.QuasiCylindricalGrid(
        radial * _SPECTRAL_SPACING, radial, -0.5 * _BOX, 0.5 * _BOX, axial, 1
    )
    transfer = D.pic.AzimuthalTransferPlan(grid, shape_order=1).prepare(_particles())
    solver = sp.QuasiCylindricalMaxwellPlan(grid).prepare((transfer,))
    r = grid.radial_coordinates[:, None]
    z = grid.axial_coordinates[None, :]
    azimuthal = _initial_magnetic(r, np.zeros_like(r), z)[..., 1]
    # Circular components B± = (B_r ∓ iB_θ)/2 of the m = 0 mode.
    circular = np.zeros((1, radial, axial, 3), dtype=np.complex128)
    circular[0, :, :, 0] = -0.5j * azimuthal
    circular[0, :, :, 1] = 0.5j * azimuthal
    source = sp.QuasiCylindricalSource(
        jnp.zeros((1, radial, axial, 3), dtype=jnp.complex128),
        jnp.zeros((1, radial, axial), dtype=jnp.complex128),
    )
    return _run(
        solver,
        jnp.asarray(circular),
        jnp.zeros((1, radial, axial), dtype=jnp.complex128),
        source,
    )


def _fields(route: _RouteResult) -> np.ndarray:
    """Probe ``(E, B)`` as ``[probe, 6]``."""
    return np.concatenate(
        (np.asarray(route.electric), np.asarray(route.magnetic)), axis=-1
    )


def _error(value: np.ndarray, reference: np.ndarray) -> float:
    """Max-norm difference relative to the reference field's max norm."""
    return float(np.max(np.abs(value - reference)) / np.max(np.abs(reference)))


def test_axisymmetric_tm_pulse_agrees_across_cochain_and_psatd_solvers() -> None:
    """Cross-route consistency of the three electromagnetic PIC field solvers.

    Scenario: a free-space, axisymmetric TM pulse (``B_θ``, ``E_r``, ``E_z``).
    At ``t = 0`` the field is ``E = 0``, ``B = ∇ × (ψ₀ẑ)`` with the isotropic
    Gaussian ``ψ₀ = exp(−ρ²/a²)`` (``c = ε₀ = μ₀ = 1``); it is solenoidal and
    charge-free, so each route propagates pure vacuum Maxwell for a time ``T``.

    Routes, driven through the one consumer interface every PIC field solver
    shares (``initialize_field(charge, magnetic=…)``, ``advance``, and
    ``gather_fields`` at the same physical probe points):

    - P1 ``CochainMaxwellPICFieldSolver``: compatible-cochain (Yee) leapfrog with
      spline-Whitney gathering; the initial face fluxes are the exact discrete curl
      ``d₁A`` of the exact edge integrals of ``A = ψ₀ẑ``.
    - P2 ``PreparedSpectralMaxwell``: collocated infinite-order Cartesian PSATD.
    - P3 ``PreparedQuasiCylindricalMaxwell``: m = 0 Hankel–Fourier PSATD.

    Independent reference: the closed-form d'Alembert solution. ``ψ`` solves the
    scalar wave equation with ``ψ(0) = ψ₀``, ``∂ₜψ(0) = 0``; spherical symmetry
    gives ``ρψ = ½[(ρ − t)g(ρ − t) + (ρ + t)g(ρ + t)]``, ``g(s) = exp(−s²/a²)``.
    Then ``B = ∇ × (ψẑ)`` and ``E = ∇∂_zΦ − ẑ∂ₜψ`` with ``Φ = ∫₀ᵗψ dt' =
    a²[g(ρ − t) − g(ρ + t)]/(4ρ)``, since ``∂ₜE = ∇ × B = ∇∂_zψ − ẑ∂ₜ²ψ``.
    Derivatives of the closed form are taken with automatic differentiation.

    Error model: the pulse spectrum ``exp(−k²a²/4)`` is below 1e-10 at the
    Nyquist wavenumber of the 0.1 grids; the radial wall (r = 2.4) and the
    periodic images of the 3.0 box sit ≥ 4.7 Gaussian radii beyond the probes'
    light cone at ``T``, and ``B₀`` is ≈ 1e-7 of its peak at the box seam. Both
    PSATD routes therefore reproduce the reference to that truncation at their
    shared grid nodes (probes are chosen on those nodes, where linear gathering is
    exact). The Yee route carries ``O(h²)`` dispersion and interpolation error;
    halving ``h`` must cut its error by ≈ 4, and all three routes agree to within
    that error.

    Complements the pairwise checks in ``tests/unit/solver``: the P2↔P3 TE pulse
    (``E_θ``, ``B_z``) compared on solver nodes, and the P1↔P2 Hertzian-dipole
    far field through Huygens boxes. Here all three solvers share one TM scenario,
    one consumer observable (the PIC gather), and one closed-form reference.
    """
    exact = np.concatenate(_exact(_probes(), _DURATION), axis=-1)
    spectral_bridge = _bridge(_SPECTRAL_SPACING)
    cartesian = _cartesian_route(spectral_bridge)
    cylindrical = _quasi_cylindrical_route()
    coarse = _cochain_route(spectral_bridge)
    fine = _cochain_route(_bridge(0.5 * _SPECTRAL_SPACING))
    for route in (cartesian, cylindrical, coarse, fine):
        assert bool(route.successful)
        assert bool(np.all(np.asarray(route.support)))
    # Gauss/∇·B residuals: roundoff for the exact-curl cochain and Hankel
    # routes (measured ≤ 1.6e-17 and 8.8e-16); the Cartesian nodal B₀ is not
    # periodic at the box seam, where |B₀| ≈ 2.5e-7 (measured 2.0e-6).
    assert float(coarse.constraint) < 1e-12
    assert float(fine.constraint) < 1e-12
    assert float(cylindrical.constraint) < 1e-12
    assert float(cartesian.constraint) < 1e-5

    # PSATD is exact in vacuum up to the box-seam truncation of B₀ (≈ 1e-7 of
    # its peak). Measured 3.1e-8 (Cartesian) and 1.3e-9 (quasi-cylindrical).
    spectral = _fields(cartesian)
    assert _error(spectral, exact) < 1e-6
    assert _error(_fields(cylindrical), exact) < 1e-6
    assert _error(_fields(cylindrical), spectral) < 1e-6

    # Yee dispersion and Whitney gathering are O(h²). Measured max-norm errors
    # 0.253 (h = 0.1) and 0.0654 (h = 0.05): ratio 3.87, order 1.95; on a
    # 3.6 box with more probes 0.327/0.189/0.0743 at h = 0.1/0.075/0.05.
    errors = [_error(_fields(route), exact) for route in (coarse, fine)]
    assert 1.8 < np.log2(errors[0] / errors[1]) < 2.3
    # C = e/h² = 26 at h = 0.05; tolerance 0.08 is a 1.22× safety factor.
    assert errors[1] < 0.08
    # The routes agree with each other within the Yee error, independently of
    # the reference (the PSATD routes differ from it by < 1e-6).
    assert _error(_fields(fine), spectral) < 0.08
    assert _error(_fields(fine), _fields(cylindrical)) < 0.08


# Q1 ↔ Q2


# Code units c = ε₀ = 1 with α = 1/137.036: q = m = 1 gives E_S = 1/ħ.
_QED_HBAR = Fraction(1, 1) / (4 * Fraction(math.pi) * Fraction(1, 137036) * 1000)


_QED_SCALE = ElectromagneticScaleContract.code_units(
    PIC_CODE_RELATIVITY.dimensional_scale,
    UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
    gravitational_constant=1,
    speed_of_light=1,
    reduced_planck_constant=_QED_HBAR,
    boltzmann_constant=1,
    elementary_charge=1,
    electron_mass=1,
    vacuum_permittivity=1,
    constant_set_id="qed-cross-route",
)


_CRITICAL = 1.0 / float(_QED_HBAR)


_QED_GAMMA = 2000.0


_CHIS = (0.02, 0.01, 0.005)


_COUNT = 200_000


# Event probability of one Q2 step Δt = p/W₀.
_EVENT_PROBABILITY = 0.1


_REACTION_SUBSTEPS = 8


_CLASSICAL_LOSS = 0.01


_LEADING = 55.0 * math.sqrt(3.0) / 16.0


_QUADRATIC = 48.0


_CUBIC = -479.3


_FOURTH_BOUND = 6000.0


_DIFFUSION = 55.0 / (16.0 * math.sqrt(3.0))


_SIGMAS = 4.0


@dataclass(frozen=True)
class _PairRun:
    """Both routes over one window at one ``χ₀`` and one Q2 subcycling."""

    ratio: float
    standard_error: float
    classical_loss: float
    subcycle_probability: float
    subcycles: int
    compton_flags: int
    compton_successful: bool
    largest_event_probability: float
    reaction_flags: int
    reaction_successful: bool


def _flag_union(flags: Array, /) -> Array:
    return jax.lax.reduce(flags, jnp.int32(0), jax.lax.bitwise_or, (0,))


@eqx.filter_jit
def _evolve(
    compton: NonlinearComptonPlan,
    reaction: RadiationReactionPlan,
    magnetic_z: Array,
    step: Array,
    steps: Array,
    key: PRNGKey,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Advance ``_COUNT`` Q2 electrons and one (deterministic) Q1 electron."""
    speed = math.sqrt(_QED_GAMMA**2 - 1.0)
    u = jnp.zeros((_COUNT, 3), dtype=jnp.float64).at[:, 0].set(speed)
    electric = jnp.zeros_like(u)
    magnetic = jnp.zeros_like(u).at[:, 2].set(magnetic_z)
    active = jnp.ones((_COUNT,), dtype=jnp.bool_)
    shape = (_COUNT, compton.maximum_subcycles, compton.uniform_count)
    depth_key, event_key = jr.split(key)
    depth = -jnp.log1p(-jr.uniform(depth_key, (_COUNT,), dtype=jnp.float64))
    substep = step / _REACTION_SUBSTEPS

    def react(_: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        u1, flags1, ok1 = carry
        drift = reaction.apply(u1, electric[:1], magnetic[:1], substep, active[:1])
        return (
            drift.proper_velocity,
            flags1 | _flag_union(drift.flags),
            ok1 & drift.successful,
        )

    def advance(
        index: Array, carry: tuple[Array, Array, Array, Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        u2, depth2, flags2, ok2, largest, u1, flags1, ok1 = carry
        uniforms = jr.uniform(jr.fold_in(event_key, index), shape, dtype=jnp.float64)
        emission = compton.apply(u2, electric, magnetic, step, active, depth2, uniforms)
        u1, flags1, ok1 = jax.lax.fori_loop(
            0, _REACTION_SUBSTEPS, react, (u1, flags1, ok1)
        )
        return (
            emission.proper_velocity,
            emission.optical_depth,
            flags2 | _flag_union(emission.flags),
            ok2 & emission.successful,
            jnp.maximum(largest, jnp.max(emission.event_probability)),
            u1,
            flags1,
            ok1,
        )

    zero, ok = jnp.int32(0), jnp.bool_(True)
    return jax.lax.fori_loop(
        0,
        steps,
        advance,
        (u, depth, zero, ok, jnp.float64(0.0), u[:1], zero, ok),
    )


def _run_pair(
    compton: NonlinearComptonPlan, reaction: RadiationReactionPlan, chi: float, seed: int
) -> _PairRun:
    speed = math.sqrt(_QED_GAMMA**2 - 1.0)
    magnetic_z = chi * _CRITICAL / speed
    u = np.zeros((1, 3), dtype=np.float64)
    u[0, 0] = speed
    field = np.zeros((1, 3), dtype=np.float64)
    field[0, 2] = magnetic_z
    rate = float(compton.rate(u, np.zeros_like(u), field)[0])
    step = _EVENT_PROBABILITY / rate
    initial = reaction.apply(
        u, np.zeros_like(u), field, step, np.ones((1,), dtype=np.bool_)
    )
    steps = round(_CLASSICAL_LOSS * _QED_GAMMA / (-float(initial.drift_rate[0]) * step))
    u2, _, flags2, ok2, largest, u1, flags1, ok1 = _evolve(
        compton,
        reaction,
        jnp.float64(magnetic_z),
        jnp.float64(step),
        jnp.int32(steps),
        jr.key(seed),
    )
    loss2 = _QED_GAMMA - np.sqrt(1.0 + np.sum(np.asarray(u2) ** 2, axis=-1))
    loss1 = _QED_GAMMA - float(np.sqrt(1.0 + np.sum(np.asarray(u1) ** 2)))
    return _PairRun(
        ratio=float(np.mean(loss2)) / loss1,
        standard_error=float(np.std(loss2, ddof=1)) / (math.sqrt(_COUNT) * loss1),
        classical_loss=loss1 / _QED_GAMMA,
        subcycle_probability=_EVENT_PROBABILITY / compton.maximum_subcycles,
        subcycles=steps * compton.maximum_subcycles,
        compton_flags=int(flags2),
        compton_successful=bool(ok2),
        largest_event_probability=float(largest),
        reaction_flags=int(flags1),
        reaction_successful=bool(ok1),
    )


def _compton(table: QEDTable, subcycles: int) -> NonlinearComptonPlan:
    """Q2 plan that splits a step of event probability ``p`` into ``subcycles``.

    With ``W₀Δt = p`` and a limit of ``1.5 p/subcycles`` the plan's own
    subcycle count ``ceil(W Δt/limit)`` is exactly ``subcycles``.
    """
    return NonlinearComptonPlan(
        "lcfa",
        _QED_SCALE,
        -1.0,
        1.0,
        table,
        maximum_chi=0.05,
        minimum_gamma=1.0,
        maximum_event_probability=1.5 * _EVENT_PROBABILITY / subcycles,
        maximum_subcycles=subcycles,
    )


def _window_corrected(
    runs: list[_PairRun], chi: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """``r/F`` and its standard error, ``F`` the finite-window and backlog factor."""
    ratio = np.asarray([run.ratio for run in runs], dtype=np.float64)
    error = np.asarray([run.standard_error for run in runs], dtype=np.float64)
    loss = np.asarray([run.classical_loss for run in runs], dtype=np.float64)
    backlog = np.asarray(
        [run.subcycle_probability / (2.0 * run.subcycles) for run in runs],
        dtype=np.float64,
    )
    # G ≈ r/(1 − backlog) is the ratio of the Q2 and Q1 initial powers.
    power = ratio / (1.0 - backlog)
    window = (1.0 + (1.0 - power) * loss + _DIFFUSION * chi * loss) * (1.0 - backlog)
    return ratio / window, error / window


def _assert_pair_run_clean(run: _PairRun, probability: float, label: str, /) -> None:
    assert run.compton_successful, label
    assert run.compton_flags == QEDEventFlag.NONE, label
    assert run.largest_event_probability <= 1.01 * probability, label
    assert run.reaction_successful, label
    assert run.reaction_flags == RadiationReactionFlag.NONE, label


def test_compton_energy_loss_approaches_classical_landau_lifshitz_as_chi() -> None:
    """Cross-route χ → 0 limit: Monte Carlo photon emission against classical reaction.

    Scenario: an ensemble of electrons (``γ₀ = 2000``) moving along ``x`` in a
    constant magnetic field ``B ẑ`` loses about 1 % of its energy to radiation,
    for a ladder of quantum parameters ``χ₀ = 0.02, 0.01, 0.005`` set by ``B`` at
    fixed ``γ₀``. No Lorentz push is applied, so ``u ⊥ B`` and the geometry stay
    fixed while the energy evolves over tens of steps.

    Routes, each advancing its own particles for the same window ``T``:

    - Q1, `RadiationReactionPlan` with the classical ``"landau-lifshitz-reduced"``
      model: deterministic Euler steps of the reduced Landau–Lifshitz force, whose
      power in this geometry is exactly the classical synchrotron power
      ``P_cl = (2/3) α m²c⁴ χ²/ħ``;
    - Q2, `NonlinearComptonPlan` (``"lcfa"``, `QEDTable` spectra): stochastic
      emission of discrete photons by the optical-depth method with independent
      random streams per electron.

    Physics: the LCFA mean emitted power is ``g(χ) P_cl`` with the Baier–Katkov
    (Erber) expansion ``g(χ) = 1 − (55√3/16) χ + 48 χ² + c₃ χ³ + O(χ⁴)``, so the
    ensemble energy-loss ratio ``ΔE_Q2/ΔE_Q1 − 1`` must vanish linearly in ``χ``
    with the leading coefficient ``−55√3/16``. This is the classical limit across
    routes over a finite, energy-evolving window; the instantaneous comparison with
    the *quantum-corrected* Q1 model is owned by
    `tests/unit/discretization/test_pic_qed.py`.

    Error model of the ratio ``r = ΔE_Q2/ΔE_Q1`` (all corrections independent of
    both implementations):

    - statistical: per-electron losses are independent, so ``σ_r`` is their sample
      standard deviation over ``√N ΔE_Q1`` (a compound-Poisson spread);
    - Q2 step: the optical depth is a renewal process (each fresh target is reduced
      by the previous overshoot), so the mean crossing rate is unbiased at any
      subcycle event probability ``p = W h``. At most one photon is emitted per
      subcycle and a second crossing stays pending, so at the end of the window a
      stationary backlog of ``p²/2`` crossings per electron is not yet emitted:
      against the ``p M`` expected emissions of ``M`` subcycles the loss is short by
      the factor ``1 − p/(2M)`` (0.45 % for ``p = 0.1``, ``M = 11``). One row is
      rerun at ``p/2`` (the same ``Δt`` with two subcycles, chosen by the plan from
      ``maximum_event_probability``) to show the corrected ratio is step
      independent;
    - finite window: with classical fractional loss ``s`` and ``G`` the ratio of the
      Q2 and Q1 initial powers (``r`` without the backlog), the ``γ²``-scaling of
      the power, the ``χ(γ)``-dependence of ``g`` and the Fokker–Planck spread
      ``Var γ ≈ (55/(16√3)) χ s γ₀²`` shift ``r`` by the factor
      ``F = 1 + (1 − G) s + (55/(16√3)) χ s + O(s², χ² s)``;
    - expansion remainder: SciPy quadrature of the Baier–Katkov spectrum gives
      ``(g − 1 + (55√3/16)χ − 48χ²)/χ³ = −478.76, −477.62, −473.76`` at
      ``χ = 10⁻⁴, 3·10⁻⁴, 10⁻³``, so ``c₃ = −479.3``; the three-term expansion is
      then within ``6000 χ⁴`` of ``g`` for ``χ ≤ 0.02`` (quadrature: ``(g − g₃)/χ⁴ =
      4448, 4953, 5256`` at ``χ = 0.02, 0.01, 0.005``);
    - Q1 Euler: each Q2 step takes 8 Q1 substeps, a relative ``O(s/(8K))`` error
      below ``1.2·10⁻⁴`` for ``K ≥ 11`` steps.
    """
    table = QEDTable("nonlinear-compton", maximum_chi=0.1)
    assert table.rate_interpolation_error <= 1e-6
    assert table.cdf_row_error + table.cdf_node_error <= 1e-3
    reaction = RadiationReactionPlan(
        "landau-lifshitz-reduced",
        _QED_SCALE,
        -1.0,
        1.0,
        maximum_chi=0.05,
        minimum_gamma=1.0,
    )
    compton = _compton(table, 1)
    runs = [_run_pair(compton, reaction, chi, 10 + i) for i, chi in enumerate(_CHIS)]
    halved = _run_pair(_compton(table, 2), reaction, _CHIS[1], 20)
    for chi, run in zip(_CHIS, runs, strict=True):
        _assert_pair_run_clean(run, _EVENT_PROBABILITY, f"chi={chi}")
    _assert_pair_run_clean(halved, _EVENT_PROBABILITY / 2.0, "halved step")

    chi = np.asarray(_CHIS, dtype=np.float64)
    estimate, spread = _window_corrected(runs, chi)
    step_estimate, step_spread = _window_corrected([halved], chi[1:2])
    # Step independence: halving the subcycle event probability at χ = 0.01
    # moves ρ from 0.94147 to 0.94480 (σ = 2.9e-3 each, 0.8σ combined). Six
    # more seed pairs per row gave ρ/g − 1 = −0.50 %/−0.31 % raw and
    # −0.05 %/−0.07 % backlog-corrected at p = 0.1 (χ = 0.02/0.01, ±0.17 %), and
    # −0.01 %/−0.06 % corrected at p = 0.05.
    assert abs(estimate[1] - step_estimate[0]) <= _SIGMAS * math.hypot(
        spread[1], step_spread[0]
    )

    expansion = 1.0 - _LEADING * chi + _QUADRATIC * chi**2 + _CUBIC * chi**3
    remainder = _FOURTH_BOUND * chi**4
    # Measured (seeds 10-12, N = 2·10⁵, p = 0.1, s ≈ 0.01, K = 11/21/43):
    # ρ − 1 = −0.10660/−0.05853/−0.02609, σ_ρ = 3.84e-3/2.95e-3/2.14e-3, i.e.
    # (ρ − g₃)/σ_ρ = −0.75/−1.12/+1.19. Tolerance: 4σ_ρ (two-sided 6e-5) +
    # 6000χ⁴.
    np.testing.assert_array_less(
        np.abs(estimate - expansion), _SIGMAS * spread + remainder
    )

    # The discrepancy vanishes linearly: its observed order over the ladder
    # matches the expansion's (0.929 between χ = 0.02 and 0.005, lowered by
    # the 48χ² term). Measured 1.015 ± 0.065.
    discrepancy = 1.0 - estimate
    assert np.all(np.diff(discrepancy) < 0.0)
    span = math.log(chi[0] / chi[-1])
    order = math.log(discrepancy[0] / discrepancy[-1]) / span
    expected_order = math.log((1.0 - expansion[0]) / (1.0 - expansion[-1])) / span
    order_error = (
        math.hypot(spread[0] / discrepancy[0], spread[-1] / discrepancy[-1]) / span
    )
    assert abs(order - expected_order) <= _SIGMAS * order_error, (order, order_error)

    # Leading coefficient: inverse-variance mean of (1 − ρ + 48χ² + c₃χ³)/χ,
    # biased by at most the weighted 6000χ³ (0.03). Measured rows 6.10/6.29/
    # 5.45, mean 6.07 ± 0.15 against 55√3/16 = 5.954 (0.75σ).
    coefficient = (discrepancy + _QUADRATIC * chi**2 + _CUBIC * chi**3) / chi
    weight = (chi / spread) ** 2
    mean = float(np.sum(weight * coefficient) / np.sum(weight))
    mean_error = float(1.0 / math.sqrt(np.sum(weight)))
    bias = float(np.sum(weight * remainder / chi) / np.sum(weight))
    assert abs(mean - _LEADING) <= _SIGMAS * mean_error + bias, (mean, mean_error)
    # The quantum deficit itself is resolved, not merely tolerated.
    assert mean > 10.0 * mean_error


# C2 ↔ A1


_LIGHT = float(_SI_SCALE.speed_of_light)


_FIELD = 1.0


_GYRO = _ELEMENTARY_CHARGE * _FIELD / float(_SI_SCALE.electron_mass)


_TEMPERATURE = 0.02


_EMITTERS = 1.0e15


_BACKGROUND = 1.0e10


_ANGLES = np.asarray([np.pi / 3.0, 5.0 * np.pi / 12.0], dtype=np.float64)


_HARMONICS = np.asarray([1.0, 2.0], dtype=np.float64)


# Filter width and truncation in units of Ω; W < 10⁻⁹ beyond the span.
_FILTER_WIDTH = 0.2


_FILTER_SPAN = 6.5


_TOP = float(_HARMONICS[-1]) + _FILTER_SPAN * _FILTER_WIDTH


# C2 resonance needs sΩ/ω = γ − N∥u∥ ≤ γ + |u cos θ| ≤ 2.6 on the Jüttner
# support ((γ − 1)/θ ≤ 41), so its emission vanishes below 0.39 Ω.
_C2_BOTTOM = 0.35


_C2_PANEL = 0.4


_C2_GRADING = (0.02, 0.08, 0.25)


_NODES = 8


_SAMPLES_PER_PERIOD = 32


_WINDOWS = (4, 8)


_QUADRATURE_ORDERS = (6, 8)


# Measured relative A1 − C2 discrepancies, rows (s, θ) = (1, π/3), (2, π/3),
# (1, 5π/12), (2, 5π/12):
#   raw, order 8, N = 4:   −1.417161e-1, −8.137022e-2, −1.454087e-1, −9.384133e-2
#   raw, order 8, N = 8:   −7.085828e-2, −4.068544e-2, −7.270461e-2, −4.692117e-2
#   (ratio 2.0000 to 5 digits: the window error is exactly first order)
#   Richardson, order 6:   −5.0e-7, −1.128e-4, −5.2e-7, −9.6e-7
#   Richardson, order 8:   −5.0e-7, −6.7e-7, −5.2e-7, −1.0e-6
# The window ratio band is ±0.5 % around 2. The tolerance is 10× the largest
# order-8 Richardson discrepancy; the received-power fault above is ≥ 4·10⁻³.
_WINDOW_RATIO_RANGE = (1.99, 2.01)


_TOLERANCE = 1.0e-5


def _gauss_panels(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(_NODES)
    lower, upper = edges[:-1, None], edges[1:, None]
    half = 0.5 * (upper - lower)
    return (half * nodes + lower + half).ravel(), (half * weights).ravel()


def _filters(omega: np.ndarray) -> np.ndarray:
    """``W_s(ω)`` for both harmonics, ``[2, F]``, with ``ω`` in units of ``Ω``."""
    offset = (omega[None, :] - _HARMONICS[:, None]) / _FILTER_WIDTH
    return np.exp(-0.5 * offset * offset)


def _c2_rule(angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights in ``ω/Ω`` on ``[0.35, 3.3]``, graded to every ``kΩ/sin θ``."""
    edges = [k / np.sin(angle) for k in (1, 2, 3) if k / np.sin(angle) < _TOP]
    cuts = {_C2_BOTTOM, _TOP, *edges}
    for edge in edges:
        cuts.update(edge + sign * step for step in _C2_GRADING for sign in (-1.0, 1.0))
    coarse = np.asarray(sorted(c for c in cuts if _C2_BOTTOM <= c <= _TOP))
    pieces = [
        np.linspace(a, b, int(np.ceil((b - a) / _C2_PANEL)) + 1)[:-1]
        for a, b in zip(coarse[:-1], coarse[1:], strict=True)
    ]
    panels = np.append(np.concatenate(pieces), _TOP)
    nodes, weights = _gauss_panels(panels)
    # Square-root edges: map the panel ending at each edge to ω = edge − t².
    reference, reference_weights = np.polynomial.legendre.leggauss(_NODES)
    for edge in edges:
        index = int(np.flatnonzero(np.isclose(panels, edge))[0])
        span = np.sqrt(edge - panels[index - 1])
        t = 0.5 * span * (reference + 1.0)
        rows = slice((index - 1) * _NODES, index * _NODES)
        nodes[rows] = edge - t * t
        weights[rows] = span * t * reference_weights
    return nodes, weights


def _c2_observable() -> tuple[np.ndarray, MagnetobremsstrahlungResult]:
    plasma = ColdPlasmaDielectric(
        _SI_SCALE,
        densities=[_BACKGROUND],
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        magnetic_field=[0.0, 0.0, _FIELD],
    )
    plan = MagnetobremsstrahlungPlan(
        plasma,
        ThermalJuttnerDistribution(_TEMPERATURE),
        emitter_density=_EMITTERS,
        maximum_harmonics=8,
        batch_size=512,
        quadrature_order=31,
    )
    rules = [_c2_rule(float(angle)) for angle in _ANGLES]
    omega = np.concatenate([nodes for nodes, _ in rules])
    angle = np.concatenate(
        [np.full(nodes.shape, a) for (nodes, _), a in zip(rules, _ANGLES, strict=True)]
    )
    result = eqx.filter_jit(plan.evaluate)(omega * _GYRO, angle)
    emission = np.asarray(result.emission).sum(axis=-1)
    observable = np.empty((_ANGLES.size, _HARMONICS.size), dtype=np.float64)
    start = 0
    for row, (nodes, weights) in enumerate(rules):
        chunk = emission[start : start + nodes.size]
        observable[row] = _filters(nodes) @ (weights * chunk) * _GYRO
        start += nodes.size
    return observable, result


def _juttner_nodes(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(u⊥, u∥, w)`` with ``Σ w F ≈ ∫ f F d³u`` for the normalized Jüttner ``f``."""
    x, x_weights = special.roots_genlaguerre(order, 0.5)
    theta = _TEMPERATURE
    gamma = 1.0 + theta * x
    speed = np.sqrt(theta * x * (2.0 + theta * x))
    radial = (
        np.sqrt(theta)
        / (2.0 * special.kve(np.float64(2.0), np.float64(1.0 / theta)))
        * x_weights
        * gamma
        * np.sqrt(2.0 + theta * x)
    )
    cosine, cosine_weights = np.polynomial.legendre.leggauss(order)
    u_perp = speed[:, None] * np.sqrt(1.0 - cosine * cosine)[None, :]
    u_par = speed[:, None] * cosine[None, :]
    return u_perp.ravel(), u_par.ravel(), (radial[:, None] * cosine_weights).ravel()


def _helices(order: int, periods: int) -> ChargedTrajectory:
    """Exact electron helices about ``+ẑ`` (counterclockwise), ``N`` whole periods."""
    u_perp, u_par, weights = _juttner_nodes(order)
    gamma = np.sqrt(1.0 + u_perp * u_perp + u_par * u_par)
    rotation = _GYRO / gamma
    period = 2.0 * np.pi / rotation
    steps = np.arange(periods * _SAMPLES_PER_PERIOD + 1, dtype=np.float64)
    times = steps[:, None] * (period / _SAMPLES_PER_PERIOD)[None, :]
    phase = rotation * times
    cosine, sine = np.cos(phase), np.sin(phase)
    radius = u_perp * _LIGHT / _GYRO
    zeros = np.zeros_like(times)
    positions = np.stack(
        (radius * cosine, radius * sine, (u_par * _LIGHT / gamma) * times), axis=-1
    )
    proper = _LIGHT * np.stack((-u_perp * sine, u_perp * cosine, zeros + u_par), axis=-1)
    acceleration = (-_LIGHT * u_perp * rotation)[None, :, None] * np.stack(
        (cosine, sine, zeros), axis=-1
    )
    count = weights.size
    return ChargedTrajectory(
        times,
        positions,
        proper,
        np.full(count, -_ELEMENTARY_CHARGE, dtype=np.float64),
        _EMITTERS * weights / (periods * period),
        np.ones(times.shape, dtype=np.bool_),
        (np.arange(count, dtype=np.uint32), np.zeros(count, dtype=np.uint32)),
        proper_accelerations=acceleration,
    )


def _a1_observable(
    order: int, periods: int
) -> tuple[np.ndarray, TrajectoryRadiationResult]:
    omega, weights = _gauss_panels(
        np.linspace(0.0, _TOP, int(np.ceil(_TOP * periods)) + 1)
    )
    directions = np.stack(
        (np.sin(_ANGLES), np.zeros_like(_ANGLES), np.cos(_ANGLES)), axis=-1
    )
    plan = TrajectoryRadiationPlan(
        _SI_SCALE,
        RadiationObserverPlan(directions, np.asarray([0.0, 0.0, 1.0])),
        omega * _GYRO,
        coherence="incoherent",
        route="segment-hermite",
        emission="truncated",
        quadrature_order=4,
    )
    result = eqx.filter_jit(plan.prepare().evaluate)(_helices(order, periods))
    spectrum = np.asarray(result.spectral_energy)
    return (_filters(omega) @ (weights[:, None] * spectrum)).T * _GYRO, result


def test_thermal_cyclotron_harmonics_match_trajectory_radiation_of_helices() -> None:
    """Thermal cyclotron harmonics: C2 emissivity against A1 on exact helices.

    Scenario (SI): electrons (``n_e = 10¹⁵ m⁻³``) with a Maxwell–Jüttner
    distribution at ``kT/(m c²) = 0.02`` gyrate in ``B = 1 T ẑ`` inside a tenuous
    cold electron background (``10¹⁰ m⁻³``, ``ω_p²/Ω² ≈ 10⁻⁹``). The observable is
    the emissivity of harmonic ``s ∈ {1, 2}`` seen through a Gaussian spectral
    filter ``W_s(ω) = exp(−(ω − sΩ)²/(2 (0.2 Ω)²))`` at ``θ ∈ {π/3, 5π/12}`` to
    ``B``, ``O_s(θ) = ∫ W_s(ω) j(ω, θ) dω`` (W m⁻³ sr⁻¹), with ``Ω = eB/m_e``.

    Route C2: `MagnetobremsstrahlungPlan` (``harmonic-sum``) emission summed over
    both cold-plasma modes, integrated over ``ω`` with composite Gauss–Legendre
    panels graded towards the resonance-ellipse edges ``ω = kΩ/sin θ``, where
    each harmonic ends in a square root (the last panel below an edge uses
    ``ω = edge − t²``). In the tenuous limit ``n_σ → 1``, the two modes'
    polarizations are an orthonormal basis of the transverse plane
    (Hermitian dielectric), so ``Σ_σ |e_σ*·V|²/|e_σT|² = |κ̂ × V|²`` and the
    two-mode sum is the vacuum emissivity. C2's delta function
    ``δ(ω − sΩ/γ − k∥v∥)`` makes ``j = n_e ⟨Σ_s P_s⟩`` with ``P_s`` the power
    emitted per unit (lab) emission time — Bekefi's single ``1/(1 − β∥ cos θ)``.

    Route A1: `TrajectoryRadiationPlan` (``segment-hermite``, incoherent) on
    exact relativistic helices, one lane per node of an independent tensor
    quadrature of the Jüttner density: generalized Gauss–Laguerre (``α = ½``) in
    ``x = (γ − 1)/θ`` times Gauss–Legendre in the pitch cosine, using
    ``d³u f = √θ/(2 K₂(1/θ) e^{1/θ}) · x^{½} e^{−x} γ √(2 + θx) dx dμ`` (SciPy
    ``kve``). Each lane covers exactly ``N`` gyroperiods ``T_g = 2πγ/Ω`` from
    ``t = 0`` and carries multiplicity ``n_e w / (N T_g)``: A1's spectral energy
    integrates the whole received pulse, so dividing by the *emission*-time window
    gives emitted power, the quantity C2 represents. Dividing by the observer-time
    window ``N T_g (1 − β∥ cos θ)`` instead (received power) moves the extrapolated
    A1 values by +1.6 %, +3.2 % (``θ = π/3``) and +0.41 %, +0.99 % (``θ = 5π/12``)
    (measured), far outside the tolerance below. The filtered spectrum is
    integrated on a shared frequency grid with panels of width ``Ω/N``.

    Error model: over exactly ``N`` periods the received field repeats with the
    observer period ``T_o``, so each lane's spectrum is ``|F₁(ω)|² |D_N(ωT_o)|²``
    with ``F₁`` the one-period spectrum and ``D_N`` the Dirichlet kernel. For
    ``g = W_s |F₁|²``, ``∫ g |D_N|² dω ∝ Σ_{|k|<N} (N − |k|) ĝ(kT_o)`` while the
    steady harmonic power is ``∝ N Σ_k ĝ(kT_o)``; the smooth filter makes ``ĝ``
    negligible for ``|k| ≥ 4`` (Gaussian decay ``exp(−(0.2 Ω · 4T_o)²/2) < 10⁻⁴``),
    so the A1 value is exactly ``O (1 + a/N)`` (Fejér saturation) for ``N ≥ 4`` and
    the Richardson value ``2 A_{2N} − A_N`` removes the window error. A sharp band
    would instead cut individual Doppler-shifted lines (at ``θ = π/3`` backward
    electrons put their third harmonic below ``2.5 Ω``), making the per-electron
    observable discontinuous in momentum. What remains is the Jüttner quadrature
    (spectrally convergent: order 6 → 8 cuts the worst Richardson discrepancy
    from ``1.1·10⁻⁴`` to ``10⁻⁶``), Hermite sampling (``24``, ``32`` and ``48``
    samples per period agree to ``10⁻⁶``), and C2's frequency quadrature
    (``≤ 2·10⁻⁷`` against finer panel rules).
    """
    reference, c2 = _c2_observable()
    assert np.all(np.asarray(c2.status) == 0)
    assert float(c2.tail_mass_bound) <= 1.0e-15
    index = np.asarray(c2.faraday.wave.refractive_index)
    assert np.max(np.abs(index - 1.0)) < 1.0e-6

    discrepancy: dict[tuple[int, int], np.ndarray] = {}
    for order in _QUADRATURE_ORDERS:
        for periods in _WINDOWS:
            value, a1 = _a1_observable(order, periods)
            assert bool(a1.evidence.finite)
            assert bool(a1.evidence.resolved)
            # A steady emitter is truncated by the window by construction.
            assert int(a1.evidence.status) == int(
                TrajectoryRadiationStatus.WINDOW_EDGE_ACCELERATION
            )
            discrepancy[order, periods] = value / reference - 1.0

    coarse, fine = _WINDOWS
    finest = _QUADRATURE_ORDERS[-1]
    ratio = discrepancy[finest, coarse] / discrepancy[finest, fine]
    assert np.all(ratio > _WINDOW_RATIO_RANGE[0]), ratio
    assert np.all(ratio < _WINDOW_RATIO_RANGE[1]), ratio

    richardson = {
        order: 2.0 * discrepancy[order, fine] - discrepancy[order, coarse]
        for order in _QUADRATURE_ORDERS
    }
    assert np.max(np.abs(richardson[finest])) < np.max(
        np.abs(richardson[_QUADRATURE_ORDERS[0]])
    )
    assert np.max(np.abs(richardson[finest])) < _TOLERANCE, richardson[finest]


# M2 ↔ B2


mx = phx.solver.maxwell


_C = float(_SI_SCALE.speed_of_light)


_HBAR = float(_SI_SCALE.reduced_planck_constant)


_EPSILON_0 = float(_SI_SCALE.vacuum_permittivity)


_MU_0 = float(_SI_SCALE.vacuum_permeability)


_ALPHA = float(_SI_SCALE.fine_structure)


_BETA = 0.8


_BAND = (300e-9, 600e-9)


# Single-term Sellmeier law of water (resonance 100 nm, strength 0.758).
_RESONANCE = 2.0 * np.pi * _C / 100e-9


_STRENGTH = 0.758


_STEP_LENGTH = 1e-3


# Near field (ρ ≪ λ), a few wavelengths, and 4·10³ wavelengths out.
_RADII = (0.2e-6, 20e-6, 2e-3)


_AZIMUTHS = 8


_GAUSS_ORDERS = (8, 16)


_NODE_COUNTS = (17, 33, 65)


def _material() -> mx.PreparedLorentzDrudeMaxwellConstitutive:
    """Undamped Lorentz pole ``ε_r = 1 + f/(ω₀² − ω²)`` with ``f = B ω₀²``."""
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2),
            phx.discretization.UniformCellAxisSpec(2),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float64))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    return mx.LorentzDrudeMaxwellConstitutivePlan(
        mx.MaxwellLorentzPoles([_RESONANCE], [0.0], [_STRENGTH * _RESONANCE**2])
    ).prepare(bridge.cochain, mx.MaxwellCochainLayout(bridge, "tez"))


def _relative_permittivity(
    material: mx.PreparedLorentzDrudeMaxwellConstitutive, omega: np.ndarray
) -> np.ndarray:
    """Real ``ε_r(ω)`` from the constitutive owner, with its lossless evidence."""
    values = np.empty(omega.shape, dtype=np.float64)
    for index, frequency in enumerate(omega):
        response = material.frequency_response(frequency)
        assert response.lossless and response.dispersive
        permittivity = np.asarray(response.permittivity, dtype=np.complex128)
        np.testing.assert_array_equal(permittivity.imag, 0.0)
        np.testing.assert_array_equal(permittivity, permittivity[0])
        np.testing.assert_array_equal(np.asarray(response.permeability), 1.0)
        values[index] = permittivity[0].real
    return values


def _field_spectral_energy(
    material: mx.PreparedLorentzDrudeMaxwellConstitutive, omega: np.ndarray
) -> np.ndarray:
    """``d²W/(dx dω)`` from the field Poynting flux, shape ``(radii, frequencies)``."""
    permittivity = _relative_permittivity(material, omega)
    plan = UniformMotionFieldPlan(
        "point",
        UniformMotionMedium(omega, _EPSILON_0 * permittivity, _MU_0),
        charge=_ELEMENTARY_CHARGE,
        speed=_BETA * _C,
        origin=np.asarray([0.1e-6, -0.2e-6, 0.3e-6], dtype=np.float64),
        direction=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )
    evidence = plan.evidence()
    assert bool(jnp.all(evidence.radiating))
    np.testing.assert_allclose(
        evidence.cherenkov_cosine, 1.0 / (_BETA * np.sqrt(permittivity)), rtol=1e-13
    )
    azimuth = np.linspace(0.0, 2.0 * np.pi, _AZIMUTHS, endpoint=False)
    radial = np.stack((np.cos(azimuth), np.sin(azimuth), np.zeros_like(azimuth)), axis=1)
    radius = np.repeat(np.asarray(_RADII, dtype=np.float64), _AZIMUTHS)
    # The ring sits at an arbitrary height along the path: only the phase moves.
    points = np.asarray(plan.origin) + radius[:, None] * np.tile(radial, (3, 1))
    points[:, 2] += 0.7e-6
    field = plan.evaluate(points)
    assert bool(jnp.all(field.supported))
    poynting = np.real(
        np.cross(np.asarray(field.electric), np.conj(np.asarray(field.magnetic)))
    )
    outward = np.sum(poynting * np.tile(radial, (3, 1))[None], axis=-1)
    rings = outward.reshape(omega.size, len(_RADII), _AZIMUTHS)
    mean = rings.mean(axis=-1)
    # Azimuthal symmetry of the field about the path (rounding level).
    np.testing.assert_allclose(
        rings, np.broadcast_to(mean[..., None], rings.shape), rtol=1e-12
    )
    # (1/π) ρ ∮ dφ with the exact uniform-azimuth rule for a constant integrand.
    return (2.0 * np.asarray(_RADII)[None, :] * mean).T


def _field_photon_yields(
    material: mx.PreparedLorentzDrudeMaxwellConstitutive,
) -> np.ndarray:
    """``dN/dx = ∫ d²W/(dx dω) / (ħω) dω``, shape ``(Gauss orders, radii)``.

    Gauss–Legendre in ``ω`` over the band; the nodes of every order share one
    field evaluation.
    """
    lower, upper = 2.0 * np.pi * _C / _BAND[1], 2.0 * np.pi * _C / _BAND[0]
    rules = [np.polynomial.legendre.leggauss(order) for order in _GAUSS_ORDERS]
    omega = np.concatenate(
        [0.5 * (upper - lower) * nodes + 0.5 * (upper + lower) for nodes, _ in rules]
    )
    photons = _field_spectral_energy(material, omega) / (_HBAR * omega)
    bounds = np.cumsum([0, *_GAUSS_ORDERS])
    return np.stack(
        [
            photons[:, start:stop] @ (0.5 * (upper - lower) * weights)
            for (_, weights), start, stop in zip(
                rules, bounds[:-1], bounds[1:], strict=True
            )
        ]
    )


def _source_photon_yield(
    material: mx.PreparedLorentzDrudeMaxwellConstitutive, node_count: int
) -> float:
    """Expected Cherenkov photons per unit length of one constant-speed step."""
    wavelengths = np.linspace(*_BAND, node_count, dtype=np.float64)
    index = np.sqrt(_relative_permittivity(material, 2.0 * np.pi * _C / wavelengths))
    medium = SpectralOpticalMedium(
        wavelengths, index[None], np.full((1, node_count), np.inf, dtype=np.float64)
    )
    plan = OpticalPhotonSourcePlan(
        relativity=_SI_SCALE.relativity,
        photon_capacity=1024,
        cherenkov=CherenkovEmission(medium, wavelengths),
    )
    steps = ChargedOpticalSteps(
        np.zeros((1, 1, 3), dtype=np.float64),
        np.asarray([[[0.0, 0.0, _STEP_LENGTH]]], dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
        np.full((1, 1), _BETA, dtype=np.float64),
        np.full((1, 1), _BETA, dtype=np.float64),
        np.zeros((1, 1), dtype=np.int32),
        np.ones((1, 1), dtype=np.bool_),
        (np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint32)),
        speed_of_light=_C,
    )
    emission = emit_optical_photons(plan, steps, jr.key(0))
    assert int(emission.status) == int(OpticalEmissionStatus.SUCCESS)
    assert bool(emission.successful)
    assert int(np.asarray(emission.nonoptical_steps)[0]) == 0
    return float(np.asarray(emission.cherenkov_expected)[0]) / _STEP_LENGTH


def _trapezoid_error_coefficient(
    material: mx.PreparedLorentzDrudeMaxwellConstitutive, reference: float
) -> float:
    """Euler–Maclaurin ``[f'(λ_b) − f'(λ_a)] / (12 dN/dx)``: relative error per ``h²``."""
    offset = 1e-12
    wavelengths = np.asarray(
        [
            _BAND[0] - offset,
            _BAND[0] + offset,
            _BAND[1] - offset,
            _BAND[1] + offset,
        ],
        dtype=np.float64,
    )
    permittivity = _relative_permittivity(material, 2.0 * np.pi * _C / wavelengths)
    density = (
        2.0 * np.pi * _ALPHA / wavelengths**2 * (1.0 - 1.0 / (_BETA**2 * permittivity))
    )
    slope_lower = (density[1] - density[0]) / (2.0 * offset)
    slope_upper = (density[3] - density[2]) / (2.0 * offset)
    return float((slope_upper - slope_lower) / (12.0 * reference))


def test_source_cherenkov_yield_equals_field_poynting_photon_flux() -> None:
    """Cherenkov photon yield: charged-step optical source against the moving-charge field.

    Scenario. A singly charged particle crosses a lossless water-like dielectric at
    ``β = 0.8``. The medium is one undamped Lorentz pole, the single-term Sellmeier
    law of water, ``ε_r(ω) = 1 + B ω₀²/(ω₀² − ω²)`` with ``B = 0.758`` and a
    resonance at ``λ₀ = 100 nm`` (``n = 1.358 → 1.331`` over the declared band
    ``300–600 nm``, ``n(589 nm) = 1.334``). The solver's Lorentz/Drude constitutive
    law owns ``ε_r(ω)`` through its ``frequency_response``; both routes read the
    medium from it. ``βn`` runs from ``1.065`` to ``1.086``, so the whole band is
    above threshold, and the Frank–Tamm factor ``1 − 1/(β²n²)`` changes by 28%
    across the band. The dispersion therefore matters at leading order.

    Route M2 (optics transport). ``emit_optical_photons`` with a
    ``CherenkovEmission`` on ``N`` uniform wavelength nodes, fed the index
    ``n(λ) = √ε_r(2πc/λ)`` tabulated on exactly those nodes, reports the expected
    photon count of one straight constant-speed step. The source represents the
    Frank–Tamm density ``d²N/(dx dλ) = 2πα/λ² (1 − 1/(β²n²))`` piecewise linearly
    between nodes. Its band integral is therefore the trapezoid rule, and its error
    is the Euler–Maclaurin term ``(h²/12)[f'(λ_b) − f'(λ_a)] + O(h⁴)``.

    Route B2 (electromagnetics). ``UniformMotionFieldPlan`` gives the SI field
    phasors ``Ẽ, H̃`` of the point charge ``e`` in the medium
    ``(ε₀ ε_r(ω), μ₀)``. The one-sided spectral energy per unit path is the radial
    Poynting flux through a cylinder of radius ``ρ``:
    ``d²W/(dx dω) = (1/π) ρ ∮ Re(Ẽ × H̃*)·ρ̂ dφ`` (plan §1: ``exp(−iωt)`` phasors
    and ``F(ω) = ∫ f e^{+iωt} dt``). It is evaluated on a ring of azimuths, divided
    by ``ħω``, and integrated over the band with Gauss–Legendre in ``ω``. The
    constants ``e``, ``ε₀``, ``μ₀``, ``c``, and ``ħ`` come from the shared
    ``ElectromagneticScaleContract.si()``, the same contract whose ``α`` the
    source uses.

    Independent physics. Both routes compute ``dN/dx``. M2 uses the Frank–Tamm
    photon density in wavelength. B2 derives it from Maxwell's fields, with no
    Frank–Tamm formula and no wavelength variable: ``ω = 2πc/λ`` and
    ``q²μ₀/(4πħ) = α/c`` are the only links. In a lossless medium the energy
    leaving the cylinder is independent of ``ρ``: near field, the Cherenkov-cone
    wave zone, and ``ρ`` of millimeters must agree. They are checked as
    conservation evidence, together with azimuthal uniformity and the B2 branch
    evidence (radiating everywhere, cone cosine ``1/(βn)``).

    Error model. B2 is Gauss–Legendre on an analytic integrand; its self-change
    between orders 8 and 16 bounds its error. The remaining cross-route discrepancy
    is M2's trapezoid error, ``O(h²)`` in the node spacing ``h = Δλ``. It is
    checked three ways:

    - The discrepancy halves twice per node doubling.
    - It matches the Euler–Maclaurin prediction computed from the medium.
    - The Richardson extrapolation of M2 equals B2 to ``O(h⁴)``.
    """
    material = _material()
    yields = _field_photon_yields(material)
    field = float(yields[-1, -1])
    # Lossless medium: the photon flux leaving the cylinder does not depend on ρ,
    # from the near field out to 4·10³ wavelengths. Measured spread 4e-16.
    np.testing.assert_allclose(yields[-1], field, rtol=1e-12)
    # Gauss–Legendre self-convergence of B2 in ω (orders 8 → 16). Measured 3e-16.
    np.testing.assert_allclose(yields[0], yields[-1], rtol=1e-12)

    source = {count: _source_photon_yield(material, count) for count in _NODE_COUNTS}
    discrepancy = {count: source[count] / field - 1.0 for count in _NODE_COUNTS}
    # Measured discrepancy (M2/B2 − 1): 17 → 1.7876e-3, 33 → 4.4739e-4,
    # 65 → 1.1188e-4, i.e. 0.4576, 0.4581, 0.4583 × (N − 1)⁻².
    coarse, middle, fine = (discrepancy[count] for count in _NODE_COUNTS)
    assert 3.9 < coarse / middle < 4.1
    assert 3.9 < middle / fine < 4.1
    # Euler–Maclaurin prediction for a trapezoid rule on f = d²N/(dx dλ). Its
    # coefficient comes from the medium alone, not from either route. The O(h²)
    # correction to it is measured at 1.5e-3, 3.7e-4, 9.4e-5 relative; the
    # tolerance is 3× the coarsest.
    coefficient = _trapezoid_error_coefficient(material, field)
    for count in _NODE_COUNTS:
        spacing = (_BAND[1] - _BAND[0]) / (count - 1)
        np.testing.assert_allclose(
            discrepancy[count], coefficient * spacing**2, rtol=5e-3
        )
    # Finest resolution: 1.5 × the measured 0.4583 (N − 1)⁻² coefficient.
    assert 0.0 < fine < 1.5 * 0.4583 / (_NODE_COUNTS[-1] - 1) ** 2
    # Richardson limit of M2 equals B2 to O(h⁴). Measured 4.2e-8 at 33/65;
    # tolerance 10× that.
    extrapolated = (4.0 * source[_NODE_COUNTS[2]] - source[_NODE_COUNTS[1]]) / 3.0
    assert abs(extrapolated / field - 1.0) < 4e-7


# Release-matrix gate


def test_release_matrix_gates_name_existing_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every required gate of the release-matrix profile names a collectable test.

    The gates are pytest node IDs consumed by release tooling, so a renamed or
    removed matrix test must fail here rather than leave an unsatisfiable gate.
    """
    (matrix,) = radiation_release_matrix_candidate_profiles()
    assert matrix.capability == "radiation.cross-route-release-matrix"
    assert not matrix.released
    assert matrix.dependencies
    root = Path(__file__).resolve().parents[2]
    for gate in matrix.required_gates:
        path, name = gate.split("::")
        spec = importlib.util.spec_from_file_location(
            f"_matrix_{Path(path).stem}", root / path
        )
        assert spec is not None and spec.loader is not None, gate
        module = importlib.util.module_from_spec(spec)
        # Dataclasses resolve their defining module through ``sys.modules``.
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        assert name.startswith("test_") and callable(vars(module).get(name)), gate
