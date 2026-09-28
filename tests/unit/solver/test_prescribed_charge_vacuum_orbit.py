#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Vacuum circular orbit: prescribed-charge Maxwell far field against A1.

A unit charge (code units, ε₀ = μ₀ = c = 1) circles at radius ``R = 0.1`` with
peak speed ``β = 0.4`` in the ``z = 0.5`` plane of a unit box. The angular
speed ramps as ``sin²`` from rest to ``ω₀ = β/R`` over one period, holds for
one period, and ramps back down over one period. The accumulated phase is
exactly ``4π``, so the charge ends at rest on its starting point, on top of
the static compensator of the coincident-neutral start. The emission then
fits inside the window and no static dipole is left behind. The box is
nonperiodic, with a CPML of physical thickness ``0.2``. A Huygens box on
the node planes ``0.3``/``0.7`` encloses the orbit and the deposit support,
and ``MaxwellFarFieldPlan`` turns its tangential phasors into
``r Ẽ e^{−ikr}``.

Reference (independent of the Maxwell runtime): the A1 trajectory far field
(``TrajectoryRadiationPlan``, ``route="segment-exact"``, coherent)

    r Ẽ(ω) = q/(4π ε₀ c) ∫ d/dt[n × (n × β)/(1 − n·β)] e^{iω(t − n·r/c)} dt

(Jackson, *Classical Electrodynamics*, 3rd ed., eq. 14.67, with the
``exp(−iωt)`` phasor convention shared by both routes). It is evaluated on
exactly the path the Maxwell run deposits: the Maxwell-step samples joined by
straight chords, with rest samples before the start and after the freeze.
The static compensator does not radiate. A1 gives each segment the mean of
its end-node proper velocities, so node velocities solving
``(u_j + u_{j+1})/2 = γ_j β_j`` from ``u_0 = 0`` give every A1 segment
exactly its Maxwell chord velocity ``β_j = (r_{j+1} − r_j)/(c Δt)``.

Both routes are compared as Cartesian vectors, which removes any dependence on
the polarization basis. What remains is the second-order discretization
error: Yee dispersion ``O((kh)²)``, the width of the Whitney (CIC) point-charge
form factor ``O((kh)²)``, the midpoint Huygens quadrature and the
staggered-field average ``O((kh)²)``, and the trapezoid time integral and
leapfrog ``O((ωΔt)²)``, where ``Δt ∝ h`` at a fixed CFL fraction.
Tolerances are therefore stated as multiples of ``(kh)²``.

Runtime evidence: every constraint status bit must be clear. At CFL 0.9 the
CPML power ledger keeps an ``O(Δt²)`` residual of about 2%, which is reported as
``LEDGER_OPEN``; at fixed ``h`` it converges at second order in Δt and closes
below the default tolerance from CFL 0.45 on.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import ElectromagneticScaleContract
from phydrax.discretization.pic import PIC_CODE_RELATIVITY
from phydrax.electromagnetics import (
    ChargedTrajectory,
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
)
from phydrax.units import CHARGE, UnitDefinition


D = phx.discretization
PIC = phx.discretization.pic
mx = phx.solver.maxwell

_CENTER = 0.5
_RADIUS = 0.1
_BETA = 0.4
_OMEGA0 = _BETA / _RADIUS
_PERIOD = 2.0 * np.pi / _OMEGA0
_START = 0.05
_RAMP = _PERIOD
_PLATEAU = _PERIOD
_STOP = _START + 2.0 * _RAMP + _PLATEAU
# After the stop the last emission crosses the Huygens surface within
# ``√3·0.2 + R ≈ 0.45``. The rest of the tail lets the quasi-static fields
# that the source left in the CPML decay before the DFT window closes.
_TAIL = 2.0
_CFL = 0.9
_OMEGAS = _OMEGA0 * np.asarray([0.9, 1.0, 1.1, 1.9, 2.0])
_COUNTS = (20, 30, 40)
_AXIS = np.asarray([0.0, 0.0, 1.0])


def _directions() -> np.ndarray:
    """Near-axis (circular), oblique, and in-plane (linear) observers."""
    polar = np.radians([20.0, 45.0, 60.0, 90.0, 90.0, 120.0, 150.0])
    azimuth = np.radians([0.0, 30.0, 200.0, 0.0, 45.0, 300.0, 100.0])
    return np.stack(
        (
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ),
        axis=-1,
    )


def _phase(time: np.ndarray) -> np.ndarray:
    """Orbit phase ``∫ ω(t) dt`` of the ``sin²`` ramp-up, plateau, ramp-down."""
    elapsed = np.clip(time - _START, 0.0, _STOP - _START)

    def ramp(duration: np.ndarray) -> np.ndarray:
        # ∫₀^s sin²(πs'/(2T)) ds' = s/2 − T sin(πs/T)/(2π)
        return duration / 2.0 - _RAMP * np.sin(np.pi * duration / _RAMP) / (2.0 * np.pi)

    rising = np.minimum(elapsed, _RAMP)
    holding = np.clip(elapsed - _RAMP, 0.0, _PLATEAU)
    falling = np.clip(elapsed - _RAMP - _PLATEAU, 0.0, _RAMP)
    return _OMEGA0 * (ramp(rising) + holding + falling - ramp(falling))


def _orbit(time: np.ndarray) -> np.ndarray:
    phase = _phase(time)
    return np.stack(
        (
            _CENTER + _RADIUS * np.cos(phase),
            _CENTER + _RADIUS * np.sin(phase),
            np.full_like(phase, _CENTER),
        ),
        axis=-1,
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
        constant_set_id="prescribed-charge-vacuum-orbit-test",
    )


class _OrbitRun(NamedTuple):
    spacing: float
    maxwell_field: np.ndarray
    maxwell_energy: np.ndarray
    reference_field: np.ndarray
    reference_energy: np.ndarray
    reference_status: int
    evidence: Any


def _bridge(count: int) -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(count) for _ in range(3)), axis_names=("x", "y", "z")
    ).prepare(jnp.asarray([[0.0] * 3, [1.0] * 3]))
    return D.StructuredCochainBridge(grid)


def _maxwell(
    count: int, times: np.ndarray, positions: np.ndarray, step_count: int
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Cartesian ``r Ẽ`` and ``d²W/dωdΩ`` of the Huygens far field, plus evidence."""
    bridge = _bridge(count)
    particles = D.ParticleSetPlan(
        jnp.arange(1), jnp.ones((1,)), ambient_dimension=3
    ).prepare()
    charged = D.ChargedParticlePlan(jnp.ones((1,)), "orbit").prepare(particles)
    current = PIC.ChargeConservingCurrentPlan(
        PIC.PICParticleCochainTransferPlan(bridge).prepare(charged)
    )
    trajectory = mx.PrescribedChargeTrajectory(times, positions[:, None, :])
    exterior = mx.HomogeneousMaxwellExterior()
    acquisition = mx.MaxwellSpectralAcquisition(
        jnp.asarray(_OMEGAS), sign="positive", measure="time-integral"
    )
    huygens = mx.MaxwellHuygensBoxPlan(
        bridge,
        (round(0.3 * count),) * 3,
        (round(0.7 * count),) * 3,
        acquisition,
        exterior,
    )
    prepared = phx.solver.CompatibleMaxwellPlan(
        bridge,
        pml=mx.MaxwellCPMLPlan(round(0.2 * count)),
        observers=(huygens,),
        sources=(
            mx.PrescribedChargeCurrentSourcePlan(
                trajectory, current, step_count=step_count
            ),
        ),
    ).prepare()
    result = mx.solve_prescribed_charge_maxwell(
        mx.PrescribedChargeMaxwellPlan(
            prepared, current, trajectory, np.asarray([1.0]), step_count=step_count
        )
    )
    sampler = prepared.observers[0]
    assert isinstance(sampler, mx.PreparedMaxwellHuygensBox)
    phasors = sampler.surface_phasors(result.final_state.observations[0])
    far = mx.MaxwellFarFieldPlan(_directions(), _AXIS, exterior).evaluate(phasors)
    spectrum = np.asarray(far.field_spectrum)
    field = (
        spectrum[..., 0, None] * np.asarray(far.theta_basis)[None]
        + spectrum[..., 1, None] * np.asarray(far.phi_basis)[None]
    )
    return field, np.asarray(far.spectral_energy), result.evidence


def _trajectory_radiation(
    times: np.ndarray, positions: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """A1 far field of the sampled chord path with one frozen rest segment."""
    nodes = np.append(times, times[-1] + step)
    path = np.concatenate((positions, positions[-1:]))
    beta = np.diff(path, axis=0) / step
    chord = beta / np.sqrt(1.0 - np.sum(beta**2, axis=-1))[:, None]
    proper = np.zeros_like(path)
    for index in range(chord.shape[0]):
        proper[index + 1] = 2.0 * chord[index] - proper[index]
    samples = nodes.shape[0]
    trajectory = ChargedTrajectory(
        nodes,
        path[:, None, :],
        proper[:, None, :],
        np.ones((1,)),
        np.ones((1,)),
        np.ones((samples, 1), dtype=np.bool_),
        (np.zeros((1,), dtype=np.uint32), np.zeros((1,), dtype=np.uint32)),
    )
    observers = RadiationObserverPlan(_directions(), _AXIS)
    result = (
        TrajectoryRadiationPlan(
            _code_scale(),
            observers,
            _OMEGAS,
            coherence="coherent",
            route="segment-exact",
        )
        .prepare()
        .evaluate(trajectory)
    )
    spectrum = np.asarray(result.field_spectrum)
    field = (
        spectrum[..., 0, None] * np.asarray(observers.basis_first)[None]
        + spectrum[..., 1, None] * np.asarray(observers.basis_second)[None]
    )
    return field, np.asarray(result.spectral_energy), int(result.evidence.status)


def _orbit_run(count: int, cfl: float = _CFL) -> _OrbitRun:
    stable = phx.solver.CompatibleMaxwellPlan(_bridge(count)).prepare().stable_dt
    step = cfl * float(stable)
    # Samples cover the motion plus at least one rest chord on either side.
    samples = int(np.ceil((_STOP + 2.0 * step) / step)) + 1
    times = step * np.arange(samples)
    positions = _orbit(times)
    step_count = int(np.ceil((_STOP + _TAIL) / step))
    field, energy, evidence = _maxwell(count, times, positions, step_count)
    reference, reference_energy, status = _trajectory_radiation(times, positions, step)
    return _OrbitRun(
        1.0 / count, field, energy, reference, reference_energy, status, evidence
    )


@pytest.fixture(scope="module")
def orbit_runs() -> dict[int, _OrbitRun]:
    return {count: _orbit_run(count) for count in _COUNTS}


def _field_errors(run: _OrbitRun) -> np.ndarray:
    """Relative vector error over all directions, per frequency."""
    difference = np.linalg.norm(run.maxwell_field - run.reference_field, axis=(1, 2))
    return difference / np.linalg.norm(run.reference_field, axis=(1, 2))


def _resolution(run: _OrbitRun) -> np.ndarray:
    """``(kh)²`` of every acquired frequency."""
    return (_OMEGAS * run.spacing) ** 2


def test_finest_grid_far_field_matches_trajectory_radiation(
    orbit_runs: dict[int, _OrbitRun],
) -> None:
    run = orbit_runs[_COUNTS[-1]]
    # The A1 reference resolves the sampled path and declares complete emission.
    assert run.reference_status == 0
    resolution = _resolution(run)
    # kh = 0.09-0.2 at N = 40 (31-70 cells per wavelength); the measured
    # coefficient of (kh)² is ≈ 0.04 near ω₀ and ≈ 0.1 near 2ω₀.
    assert np.all(_field_errors(run) < 0.25 * resolution)
    energy_error = np.linalg.norm(
        run.maxwell_energy - run.reference_energy, axis=1
    ) / np.linalg.norm(run.reference_energy, axis=1)
    # |F|² doubles the relative field error to first order.
    assert np.all(energy_error < 0.4 * resolution)


def test_far_field_error_converges_at_second_order(
    orbit_runs: dict[int, _OrbitRun],
) -> None:
    errors = np.stack([_field_errors(orbit_runs[count]) for count in _COUNTS])
    assert np.all(np.diff(errors, axis=0) < 0.0)
    spacing = np.log(1.0 / np.asarray(_COUNTS))
    # Least-squares slope of log error against log h, one per frequency.
    order = np.polyfit(spacing, np.log(errors), 1)[0]
    assert np.all(order > 1.7), order


def test_orbit_runs_hold_every_runtime_constraint(
    orbit_runs: dict[int, _OrbitRun],
) -> None:
    # The CPML power ledger is an O(Δt²) residual at this CFL (its convergence is
    # tested below); every other status bit is clear.
    ledger = int(mx.PrescribedChargeStatus.LEDGER_OPEN)
    for count, run in orbit_runs.items():
        evidence = run.evidence
        assert int(evidence.status) & ~ledger == 0, (count, int(evidence.status))
        assert not bool(np.any(np.asarray(evidence.exited))), count
        assert float(evidence.maximum_support_leak) == 0.0, count


def test_cpml_power_ledger_converges_at_second_order_in_the_step() -> None:
    # At N = 20 the ledger defect is 2.0% at CFL 0.9, 3.6e-3 at 0.45 and
    # 7.9e-4 at 0.225 (measured): once asymptotic, halving Δt at fixed h cuts it
    # by ≈ 4, so the CPML loss the runtime reports is a consistent second-order
    # account of what the update absorbs, not a missing loss channel.
    coarse, fine = (_orbit_run(_COUNTS[0], cfl).evidence for cfl in (0.45, 0.225))
    for evidence in (coarse, fine):
        assert int(evidence.status) == 0, int(evidence.status)
    defects = [float(value.relative_ledger_defect) for value in (coarse, fine)]
    assert defects[1] < 1.5e-3, defects
    order = np.log2(defects[0] / defects[1])
    assert 1.7 < order < 2.5, order
