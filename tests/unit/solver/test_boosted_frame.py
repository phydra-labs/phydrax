#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Contracts of Lorentz-boosted-frame PIC over Galilean PSATD.

References are independent of the implementation: relativistic velocity
addition and length contraction, the Lorentz invariants of the field tensor,
the analytic vacuum laser, the undulator resonance ``2γ²ω_u/(1 + K²/2)``, and
lab-frame PIC runs of the same stage.
"""

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax import ElectromagneticScaleContract
from phydrax.discretization.pic import (
    ExternalFieldSample,
    PIC_CODE_RELATIVITY,
    PICTrackRecorder,
)
from phydrax.units import CHARGE, UnitDefinition


D = phx.discretization
sp = phx.solver.maxwell.spectral
S = phx.solver


def _scale() -> ElectromagneticScaleContract:
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
        constant_set_id="boosted-frame-test",
    )


def _frame(gamma: float, lower: float, upper: float, *, center: float = 1.5) -> Any:
    """Boost along z; the lab domain spans ``[lower, upper]`` and a 3×3 cross-section."""
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    return S.BoostedFramePlan(
        phx.LorentzFrame(phx.boost_matrix(jnp.asarray([0.0, 0.0, beta]))),
        phx.geometry.Box(
            [center, center, 0.5 * (lower + upper)], [3.0, 3.0, upper - lower]
        ),
    )


def _bridge(count: int, lower: float, upper: float, *, center: float = 1.5) -> Any:
    grid = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(3, periodic=True),
            D.UniformCellAxisSpec(3, periodic=True),
            D.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(
        jnp.asarray(
            [[center - 1.5, center - 1.5, lower], [center + 1.5, center + 1.5, upper]]
        )
    )
    return D.StructuredCochainBridge(grid)


def _species(offset: int, count: int, specific: float, name: str) -> tuple[Any, Any]:
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + count), jnp.ones((count,)), ambient_dimension=3
    ).prepare()
    charged = D.ChargedParticlePlan(specific * jnp.ones((count,)), name).prepare(support)
    plan = D.pic.PICSpeciesPlan(
        D.ParticlePopulationPlan(support),
        D.pic.PICChargeModelPlan(
            specific,
            name,
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )
    return plan, charged


def _solver(bridge: Any, charged: tuple[Any, ...], **options: Any) -> Any:
    transfer = D.pic.PICParticleCochainTransferPlan(bridge, shape_order=1)
    transfers = tuple(transfer.prepare(value) for value in charged)
    currents = tuple(D.pic.ChargeConservingCurrentPlan(value) for value in transfers)
    return sp.SpectralMaxwellPlan(bridge, **options).prepare(transfers, currents)


def _galilean(frame: Any) -> dict[str, Any]:
    return {
        "variant": "galilean",
        "galilean_velocity": frame.galilean_velocity,
        "charge_conservation": "update-with-rho",
    }


def _recorder(plans: tuple[Any, ...], lanes: int, capacity: int) -> PICTrackRecorder:
    return PICTrackRecorder(
        plans,
        np.zeros(lanes, dtype=np.int32),
        (np.zeros(lanes, dtype=np.uint32), np.arange(lanes, dtype=np.uint32)),
        relativity=PIC_CODE_RELATIVITY,
        sample_capacity=capacity,
    )


def _scan(step: Any, state: Any, dt: float, steps: int) -> tuple[Any, Array]:
    def body(value: Any, _: None) -> tuple[Any, Array]:
        result = step(value, dt)
        return result.accepted_state, result.successful

    return eqx.filter_jit(lambda value: jax.lax.scan(body, value, None, length=steps))(
        state
    )


class _PlaneLaser(phx.StrictModule):
    """Lab vacuum pulse ``E_x = c B_y = E₀ cos²(πξ/2L) cos(k ξ)``, ``ξ = z − ct − z₀``.

    The envelope has compact support ``|ξ| < L``; any ``f(z − ct)`` is an exact
    one-dimensional vacuum solution.
    """

    amplitude: float
    center: float
    half_length: float
    wavenumber: float

    @property
    def source_id(self) -> str:
        return f"plane-laser-{self.amplitude}-{self.center}-{self.half_length}"

    def external_fields(self, positions: Array, times: Array, /) -> ExternalFieldSample:
        return _laser_sample(self, positions[:, 2], times)


def _laser_sample(laser: _PlaneLaser, z: Array, t: Array) -> ExternalFieldSample:
    xi = z - t - laser.center
    envelope = jnp.where(
        jnp.abs(xi) < laser.half_length,
        jnp.cos(0.5 * jnp.pi * xi / laser.half_length) ** 2,
        0.0,
    )
    value = laser.amplitude * envelope * jnp.cos(laser.wavenumber * xi)
    zero = jnp.zeros_like(value)
    return ExternalFieldSample(
        jnp.stack((value, zero, zero), axis=-1),
        jnp.stack((zero, value, zero), axis=-1),
        jnp.ones(value.shape, dtype=jnp.bool_),
    )


# -- frame and transforms ------------------------------------------------------------------


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param(np.eye(4), id="identity"),
        pytest.param(
            np.asarray(phx.boost_matrix(jnp.asarray([0.3, 0.0, 0.4]))), id="oblique"
        ),
        pytest.param(
            np.block(
                [
                    [np.ones((1, 1)), np.zeros((1, 3))],
                    [np.zeros((3, 1)), np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])],
                ]
            ).astype(np.float64)
            @ np.asarray(phx.boost_matrix(jnp.asarray([0.0, 0.0, 0.6]))),
            id="rotated",
        ),
    ],
)
def test_frame_must_be_a_pure_boost_along_one_axis(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match="boost"):
        S.BoostedFramePlan(
            phx.LorentzFrame(matrix), phx.geometry.Box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        )


def test_boosted_particles_contract_resting_plasma_and_add_beam_velocities() -> None:
    gamma = 3.0
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    frame = _frame(gamma, 0.0, 30.0)
    lab = np.stack((np.full(5, 1.0), np.full(5, 2.0), np.linspace(4.0, 12.0, 5)), -1)
    plasma = frame.boost_particles(lab, np.zeros_like(lab), boosted_time=-7.0)
    grid = np.asarray(plasma.positions) - (-7.0) * np.asarray(frame.galilean_velocity)
    # Length contraction at unchanged weights: density γn, drift −v_b.
    np.testing.assert_allclose(grid[:, 2], lab[:, 2] / gamma, rtol=1e-14)
    np.testing.assert_allclose(grid[:, :2], lab[:, :2], rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(plasma.velocities), [[0.0, 0.0, -beta]] * 5, rtol=1e-14, atol=1e-15
    )
    beam_beta = np.asarray([0.2, 0.0, 0.95])
    beam = frame.boost_particles(
        lab[:1], beam_beta[None], boosted_time=3.0, lab_times=np.asarray([1.5])
    )
    # Relativistic velocity addition along and across the boost.
    denominator = 1.0 - beam_beta[2] * beta
    expected = np.asarray(
        [beam_beta[0] / (gamma * denominator), 0.0, (beam_beta[2] - beta) / denominator]
    )
    np.testing.assert_allclose(np.asarray(beam.velocities[0]), expected, rtol=1e-13)
    # The drifted particle lies on the lab world line x(t) = x₀ + v (t − 1.5).
    lab_time, lab_position = frame.to_lab(3.0, beam.positions[0])
    np.testing.assert_allclose(
        np.asarray(lab_position),
        lab[0] + beam_beta * (float(lab_time) - 1.5),
        rtol=1e-13,
        atol=1e-12,
    )


def test_boosted_undulator_field_is_magnetic_type_and_subluminal() -> None:
    gamma = 5.0
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    period = 1.0
    undulator = phx.applications.accelerator.InsertionDeviceField(
        np.pi, period, 6, polarization="planar", center=0.0, aperture=(0.3, 0.3)
    )
    frame = _frame(gamma, -10.0, 10.0)
    boosted = frame.boost_external_field(undulator)
    z = jnp.linspace(-0.3, 0.3, 61)
    points = jnp.stack((jnp.zeros_like(z), 0.05 * jnp.ones_like(z), z), -1)
    lab = undulator.external_fields(points, jnp.zeros_like(z))
    e, b = frame.boost_fields(lab.electric, lab.magnetic)
    # Field invariants: E·B = 0 and E² − c²B² = −c²B_lab² < 0 (magnetic type,
    # never the null field of radiation).
    np.testing.assert_allclose(np.sum(e * b, -1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.sum(e * e, -1) - np.sum(b * b, -1),
        -np.sum(np.asarray(lab.magnetic) ** 2, -1),
        rtol=1e-12,
        atol=1e-12,
    )
    # The boosted pattern translates rigidly at −βc: a subluminal (evanescent)
    # wave that no free vacuum mode can carry.
    shift = 0.01
    first = boosted.external_fields(points, jnp.zeros_like(z))
    later = boosted.external_fields(
        points.at[:, 2].add(-beta * shift), jnp.full_like(z, shift)
    )
    np.testing.assert_allclose(later.magnetic, first.magnetic, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(later.electric, first.electric, rtol=1e-10, atol=1e-12)
    assert bool(jnp.all(first.support))


# -- undulator: gather-only fields and lab-frame tracks ------------------------------------


@dataclass(frozen=True)
class _UndulatorRuns:
    frequencies: np.ndarray
    resonance: float
    lab_spectrum: np.ndarray
    boosted_spectrum: np.ndarray
    lab_status: int
    boosted_status: int
    boosted_grid_energy: float
    undulator_energy: float
    boosted_steps_accepted: bool


_K = 0.5
_UNDULATOR_GAMMA = 10.0


def _undulator() -> Any:
    return phx.applications.accelerator.InsertionDeviceField(
        2.0 * np.pi * _K,
        1.0,
        4,
        polarization="planar",
        center=8.0,
        aperture=(0.3, 0.3),
        ramp_periods=0.5,
    )


@pytest.fixture(scope="module")
def undulator_runs() -> _UndulatorRuns:
    """One electron (tiny macro weight) through a static planar undulator."""
    beta_e = np.sqrt(1.0 - 1.0 / _UNDULATOR_GAMMA**2)
    start, stop = 0.5, 15.5
    undulator = _undulator()
    scale = _scale()
    plan, charged = _species(0, 1, -1.0, "electron")
    mass = (np.asarray([1.0e-12]),)
    omega_u = 2.0 * np.pi
    resonance = 2.0 * _UNDULATOR_GAMMA**2 * omega_u / (1.0 + 0.5 * _K**2)
    frequencies = np.linspace(0.6, 1.3, 36) * resonance
    directions = np.asarray(
        [[0.0, 0.0, 1.0], [np.sin(0.02), 0.0, np.cos(0.02)], [0.0, 0.03, 1.0]]
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radiation = phx.electromagnetics.TrajectoryRadiationPlan(
        scale,
        phx.electromagnetics.RadiationObserverPlan(
            directions, np.asarray([1.0, 0.0, 0.0])
        ),
        frequencies,
        coherence="coherent",
        route="segment-exact",
    ).prepare()
    lab_dt = 0.0125
    lab_steps = int(np.ceil((stop - start) / beta_e / lab_dt))
    lab_recorder = _recorder((plan,), 1, lab_steps)
    lab_pic = S.ElectromagneticPICPlan(
        _solver(_bridge(16, 0.0, 16.0, center=0.0), (charged,)),
        species=(plan,),
        external_fields=(undulator,),
        recorders=(lab_recorder,),
    )
    lab_state = lab_pic.initialize(
        (np.asarray([[0.0, 0.0, start]]),),
        (np.asarray([[0.0, 0.0, beta_e]]),),
        lab_dt,
        masses=mass,
    )
    lab_final, lab_ok = _scan(lab_pic.step_detailed, lab_state, lab_dt, lab_steps)
    assert bool(jnp.all(lab_ok))
    lab = radiation.evaluate(
        lab_recorder.to_charged_trajectory(lab_final.recorders[0], scale)
    )

    frame = _frame(5.0, 0.0, 16.0, center=0.0)
    t0, _ = frame.from_lab(0.0, jnp.asarray([0.0, 0.0, start]))
    t1, _ = frame.from_lab((stop - start) / beta_e, jnp.asarray([0.0, 0.0, stop]))
    dt = 0.003
    steps = int(np.ceil((float(t1) - float(t0)) / dt))
    electron = frame.boost_particles(
        np.asarray([[0.0, 0.0, start]]),
        np.asarray([[0.0, 0.0, beta_e]]),
        boosted_time=t0,
    )
    solver = _solver(_bridge(16, 0.0, 3.2, center=0.0), (charged,), **_galilean(frame))
    boosted = frame.prepare(
        S.ElectromagneticPICPlan(
            solver,
            species=(plan,),
            external_fields=(frame.boost_external_field(undulator),),
            recorders=(_recorder((plan,), 1, steps),),
        )
    )
    state = boosted.initialize(
        (electron.positions,), (electron.velocities,), dt, time=t0, masses=mass
    )
    final, ok = _scan(boosted.step_detailed, state, dt, steps)
    result = radiation.evaluate(boosted.lab_trajectory(final, 0, scale))
    # Energy the boosted undulator would carry along the axis column of unit
    # cross-section of the final grid, a scale for the grid field energy.
    nodes = jnp.stack((jnp.zeros(16), jnp.zeros(16), 0.2 * jnp.arange(16)), axis=-1)
    sample = frame.boost_external_field(undulator).external_fields(
        nodes + final.pic.time * jnp.asarray(frame.galilean_velocity),
        jnp.full((16,), final.pic.time),
    )
    undulator_energy = 0.5 * float(jnp.sum(sample.electric**2 + sample.magnetic**2) * 0.2)
    return _UndulatorRuns(
        frequencies,
        resonance,
        np.asarray(lab.spectral_energy),
        np.asarray(result.spectral_energy),
        int(lab.evidence.status),
        int(result.evidence.status),
        float(solver.field_energy(final.pic.field)),
        undulator_energy,
        bool(jnp.all(ok)),
    )


def test_boosted_undulator_is_gathered_and_never_deposited(
    undulator_runs: _UndulatorRuns,
) -> None:
    assert undulator_runs.boosted_steps_accepted
    # Only the self-field of the 1e-12 test charge reaches the grid; the
    # boosted undulator, a prescribed field, contributes none of its energy.
    assert undulator_runs.undulator_energy > 1.0
    assert undulator_runs.boosted_grid_energy < 1.0e-20 * undulator_runs.undulator_energy


def test_boosted_undulator_tracks_radiate_like_lab_tracks(
    undulator_runs: _UndulatorRuns,
) -> None:
    lab, boosted = undulator_runs.lab_spectrum, undulator_runs.boosted_spectrum
    assert undulator_runs.lab_status == 0
    assert undulator_runs.boosted_status == 0
    # On-axis fundamental at the undulator resonance (independent reference).
    peak = undulator_runs.frequencies[np.argmax(lab[:, 0])]
    assert abs(peak / undulator_runs.resonance - 1.0) < 0.03
    # Boosted tracks, carried to the lab with per-lane lab times, radiate the
    # lab spectrum through the same trajectory-radiation route.
    assert np.linalg.norm(boosted - lab) / np.linalg.norm(lab) < 0.02


# -- back-transformed vacuum laser and restart ---------------------------------------------


_VACUUM_GAMMA = 2.0
_VACUUM_LAB_TIME = 12.0
_VACUUM_LASER = _PlaneLaser(1.0, 10.0, 5.0, 2.0 * np.pi)


def _vacuum_run(**prepare: Any) -> tuple[Any, Any, float]:
    """Boosted vacuum laser on a Galilean grid whose particle is inactive."""
    frame = _frame(_VACUUM_GAMMA, 0.0, 40.0)
    plan, charged = _species(0, 1, -1.0, "electron")
    solver = _solver(_bridge(429, -70.0, 30.0), (charged,), **_galilean(frame))
    boosted = frame.prepare(
        S.ElectromagneticPICPlan(solver, species=(plan,)),
        snapshots=S.BoostedSnapshotPlan((_VACUUM_LAB_TIME,)),
        **prepare,
    )
    dt = 0.05
    state = boosted.initialize(
        (np.asarray([[1.5, 1.5, -60.0]]),),
        (np.zeros((1, 3)),),
        dt,
        time=-46.0,
        active_masks=(np.asarray([False]),),
        masses=(np.zeros(1),),
        vacuum_fields=(_VACUUM_LASER,),
    )
    return boosted, state, dt


def test_back_transformed_vacuum_laser_equals_the_lab_field() -> None:
    boosted, state, dt = _vacuum_run()
    final, ok = _scan(boosted.step_detailed, state, dt, 1412)
    assert bool(jnp.all(ok))
    snapshot = boosted.lab_field_snapshot(final, 0)
    assert bool(jnp.all(snapshot.filled))
    z = snapshot.axis_coordinates
    lab = _laser_sample(_VACUUM_LASER, z, jnp.full_like(z, _VACUUM_LAB_TIME))
    electric = snapshot.electric[1, 1]
    magnetic = snapshot.magnetic[1, 1]
    # PSATD propagates vacuum exactly in the boosted frame; the lab field is
    # reconstructed plane by plane from boosted times γ(T − βz/c), linear in
    # time between steps: error ≤ (ωΔt′)²/8 ≈ 3e-3 at the grid-frame laser
    # frequency ω = k′(c + v_b).
    scale = float(jnp.max(jnp.abs(lab.electric)))
    np.testing.assert_allclose(electric, lab.electric, rtol=0, atol=5.0e-3 * scale)
    np.testing.assert_allclose(magnetic, lab.magnetic, rtol=0, atol=5.0e-3 * scale)
    assert float(final.evidence.nci_rejections) == 0


def test_restart_continues_bitwise_and_refuses_foreign_components() -> None:
    boosted, state, dt = _vacuum_run()
    first, _ = _scan(boosted.step_detailed, state, dt, 3)
    checkpoint = boosted.checkpoint(first)
    resumed, _ = _scan(boosted.step_detailed, boosted.restore(checkpoint), dt, 3)
    straight, _ = _scan(boosted.step_detailed, state, dt, 6)
    for left, right in zip(
        jax.tree.leaves(resumed), jax.tree.leaves(straight), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    other, _, _ = _vacuum_run(nci_growth_limit=50.0)
    with pytest.raises(ValueError, match="another plan"):
        other.restore(checkpoint)


# -- NCI guard -----------------------------------------------------------------------------


def test_nci_guard_rejects_high_k_growth_beyond_the_declared_limits() -> None:
    frame = _frame(5.0, 0.0, 16.0, center=0.0)
    plan, charged = _species(0, 1, -1.0, "electron")
    solver = _solver(_bridge(16, 0.0, 3.2, center=0.0), (charged,), **_galilean(frame))
    pic = S.ElectromagneticPICPlan(solver, species=(plan,))
    beam = frame.boost_particles(
        np.asarray([[0.0, 0.0, 4.0]]), np.asarray([[0.0, 0.0, 0.995]]), boosted_time=-2.0
    )
    guarded = frame.prepare(pic, nci_growth_limit=1.001, nci_energy_fraction=1.0e-9)
    lenient = frame.prepare(pic)
    for prepared, admitted in ((guarded, False), (lenient, True)):
        state = prepared.initialize(
            (beam.positions,),
            (beam.velocities,),
            0.01,
            time=-2.0,
            masses=(np.asarray([1.0e-12]),),
        )
        # The moving point charge's field changes its high-|k| content; the
        # tight guard reads that as growth and keeps the previous state.
        result = prepared.step_detailed(state, 0.01)
        assert bool(result.pic.successful)
        assert bool(result.nci_admissible) is admitted
        assert int(result.accepted_state.evidence.nci_rejections) == int(not admitted)
        expected = state.pic.time + (0.01 if admitted else 0.0)
        np.testing.assert_allclose(result.accepted_state.pic.time, expected)


# -- refusals ------------------------------------------------------------------------------


def _small(frame: Any, **options: Any) -> tuple[Any, Any, Any]:
    plan, charged = _species(0, 1, -1.0, "electron")
    count, lower, upper = options.pop("grid", (16, 0.0, 4.0))
    return plan, charged, _solver(_bridge(count, lower, upper), (charged,), **options)


def _refused_velocity() -> None:
    frame = _frame(2.0, 0.0, 8.0)
    velocity = (0.0, 0.0, -0.5)
    plan, _, solver = _small(
        frame,
        variant="galilean",
        galilean_velocity=velocity,
        charge_conservation="update-with-rho",
    )
    frame.prepare(S.ElectromagneticPICPlan(solver, species=(plan,)))


def _refused_coverage() -> None:
    frame = _frame(2.0, 0.0, 8.0)
    plan, _, solver = _small(frame, grid=(16, 0.0, 2.0), **_galilean(frame))
    frame.prepare(S.ElectromagneticPICPlan(solver, species=(plan,)))


def _refused_field() -> None:
    frame = _frame(2.0, 0.0, 8.0, center=0.0)
    plan, charged = _species(0, 1, -1.0, "electron")
    solver = _solver(_bridge(16, 0.0, 4.0, center=0.0), (charged,), **_galilean(frame))
    frame.prepare(
        S.ElectromagneticPICPlan(solver, species=(plan,), external_fields=(_undulator(),))
    )


def _refused_recorder() -> None:
    frame = _frame(2.0, 0.0, 8.0)
    plan, _, solver = _small(frame, **_galilean(frame))
    radiation = phx.electromagnetics.TrajectoryRadiationPlan(
        _scale(),
        phx.electromagnetics.RadiationObserverPlan(
            np.asarray([[0.0, 0.0, 1.0]]), np.asarray([1.0, 0.0, 0.0])
        ),
        np.asarray([1.0]),
        coherence="coherent",
        route="segment-exact",
    ).prepare()
    recorder = PICTrackRecorder(
        (plan,),
        [0],
        (np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint32)),
        relativity=PIC_CODE_RELATIVITY,
        radiation=radiation,
    )
    frame.prepare(
        S.ElectromagneticPICPlan(solver, species=(plan,), recorders=(recorder,))
    )


def _refused_medium() -> None:
    frame = _frame(2.0, 0.0, 8.0)
    plan, _, solver = _small(frame, permittivity=4.0)
    frame.prepare(S.ElectromagneticPICPlan(solver, species=(plan,)))


def _refused_ring() -> None:
    frame = _frame(2.0, 0.0, 8.0)
    plan, _, solver = _small(frame, **_galilean(frame))
    frame.prepare(
        S.ElectromagneticPICPlan(solver, species=(plan,)),
        snapshots=S.BoostedSnapshotPlan((0.0, 1.0), ring_capacity=1),
    )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(_refused_velocity, "comove", id="galilean-velocity"),
        pytest.param(_refused_coverage, "does not cover", id="lab-domain-coverage"),
        pytest.param(_refused_field, "boosted through this frame", id="lab-field"),
        pytest.param(_refused_recorder, "streams trajectory radiation", id="recorder"),
        pytest.param(_refused_medium, "vacuum", id="medium"),
        pytest.param(_refused_ring, "ring capacity", id="snapshot-ring"),
    ],
)
def test_prepare_refuses_inconsistent_boosted_runs(build: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


def test_standard_grid_refuses_boosted_plasma() -> None:
    frame = _frame(2.0, 0.0, 8.0)
    plan, _, solver = _small(frame)
    boosted = frame.prepare(S.ElectromagneticPICPlan(solver, species=(plan,)))
    plasma = frame.boost_particles(
        np.asarray([[1.5, 1.5, 4.0]]), np.zeros((1, 3)), boosted_time=0.0
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="Galilean grid"):
        boosted.initialize(
            (plasma.positions,),
            (plasma.velocities,),
            0.05,
            time=0.0,
            masses=(np.asarray([1.0e-12]),),
        )


def test_vacuum_fields_overlapping_particles_are_refused() -> None:
    frame = _frame(2.0, 0.0, 8.0)
    plan, _, solver = _small(frame, **_galilean(frame))
    boosted = frame.prepare(S.ElectromagneticPICPlan(solver, species=(plan,)))
    laser = _PlaneLaser(1.0, 4.0, 5.0, 2.0 * np.pi)
    time, _ = frame.from_lab(0.0, jnp.asarray([1.5, 1.5, 4.0]))
    plasma = frame.boost_particles(
        np.asarray([[1.5, 1.5, 4.0]]), np.zeros((1, 3)), boosted_time=time
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="must not overlap particles"):
        boosted.initialize(
            (plasma.positions,),
            (plasma.velocities,),
            0.05,
            time=time,
            masses=(np.asarray([1.0e-12]),),
            vacuum_fields=(laser,),
        )


def _far_field() -> Any:
    return phx.solver.maxwell.MaxwellFarFieldResult(
        angular_frequencies=jnp.ones((1,)),
        directions=jnp.asarray([[1.0, 0.0, 0.0]]),
        theta_basis=jnp.asarray([[0.0, 0.0, -1.0]]),
        phi_basis=jnp.asarray([[0.0, 1.0, 0.0]]),
        field_spectrum=jnp.ones((1, 1, 2), dtype=jnp.complex128),
        coherency=jnp.ones((1, 1, 2, 2), dtype=jnp.complex128),
        stokes=jnp.ones((1, 1, 4)),
        spectral_energy=jnp.ones((1, 1)),
    )


def test_boosted_huygens_requires_a_standard_grid_and_zero_surface_current() -> None:
    frame = _frame(2.0, 0.0, 8.0)
    plan, charged, solver = _small(frame, **_galilean(frame))
    galilean = frame.prepare(S.ElectromagneticPICPlan(solver, species=(plan,)))
    idle = galilean.initialize(
        (np.asarray([[1.5, 1.5, 2.0]]),),
        (np.zeros((1, 3)),),
        0.05,
        time=0.0,
        active_masks=(np.asarray([False]),),
        masses=(np.zeros(1),),
    )
    with pytest.raises(ValueError, match="standard spectral grid"):
        galilean.lab_far_field(idle, _far_field(), emission="complete")
    exterior = phx.solver.maxwell.HomogeneousMaxwellExterior()
    box = sp.SpectralHuygensBoxPlan(
        (0, 0, 4),
        (2, 2, 12),
        phx.solver.maxwell.MaxwellSpectralAcquisition(
            jnp.asarray([1.0]), sign="positive", measure="time-integral", stop_time=1.0
        ),
        exterior,
    )
    standard = frame.prepare(
        S.ElectromagneticPICPlan(
            _solver(
                _bridge(16, 0.0, 4.0), (charged,), grid="staggered", observers=(box,)
            ),
            species=(plan,),
        )
    )
    # A beam crossing the box drives current on its side faces.
    beam = frame.boost_particles(
        np.asarray([[1.5, 1.5, 3.0]]), np.asarray([[0.0, 0.0, 0.99]]), boosted_time=0.0
    )
    state = standard.initialize(
        (beam.positions,),
        (beam.velocities,),
        0.05,
        time=0.0,
        masses=(np.asarray([1.0e-12]),),
    )
    result = standard.step_detailed(state, 0.05)
    assert not bool(result.successful)
    with pytest.raises(ValueError, match="J = 0"):
        standard.lab_far_field(result.accepted_state, _far_field(), emission="complete")


def test_admitted_boosted_huygens_far_field_is_doppler_relabeled_into_the_lab() -> None:
    gamma = 2.0
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    frame = _frame(gamma, 0.0, 8.0)
    plan, charged = _species(0, 1, -1.0, "electron")
    box = sp.SpectralHuygensBoxPlan(
        (0, 0, 4),
        (2, 2, 12),
        phx.solver.maxwell.MaxwellSpectralAcquisition(
            jnp.asarray([1.0]), sign="positive", measure="time-integral", stop_time=0.1
        ),
        phx.solver.maxwell.HomogeneousMaxwellExterior(),
    )
    standard = frame.prepare(
        S.ElectromagneticPICPlan(
            _solver(
                _bridge(16, 0.0, 4.0), (charged,), grid="staggered", observers=(box,)
            ),
            species=(plan,),
        )
    )
    state = standard.initialize(
        (np.asarray([[1.5, 1.5, 2.0]]),),
        (np.zeros((1, 3)),),
        0.05,
        time=0.0,
        active_masks=(np.asarray([False]),),
        masses=(np.zeros(1),),
    )
    with pytest.raises(ValueError, match="window is still open"):
        standard.lab_far_field(state, _far_field(), emission="complete")
    final, ok = _scan(standard.step_detailed, state, 0.05, 3)
    assert bool(jnp.all(ok))
    directions = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.6, -0.8]])
    boosted = phx.solver.maxwell.MaxwellFarFieldResult(
        angular_frequencies=jnp.asarray([1.0, 3.0]),
        directions=jnp.asarray(directions),
        theta_basis=jnp.zeros((3, 3)),
        phi_basis=jnp.zeros((3, 3)),
        field_spectrum=jnp.zeros((2, 3, 2), dtype=jnp.complex128),
        coherency=jnp.zeros((2, 3, 2, 2), dtype=jnp.complex128),
        stokes=jnp.zeros((2, 3, 4)),
        spectral_energy=jnp.ones((2, 3)),
    )
    lab = standard.lab_far_field(final, boosted, emission="complete")
    # The boosted frame moves at +βc: a photon along n′ reaches the lab with
    # ω = γω′(1 + β n′_z), and ω⁻² d²W/(dω dΩ) is invariant.
    doppler = gamma * (1.0 + beta * directions[:, 2])
    np.testing.assert_allclose(
        lab.angular_frequencies, np.asarray([[1.0], [3.0]]) * doppler, rtol=1e-13
    )
    np.testing.assert_allclose(lab.spectral_energy, np.broadcast_to(doppler**2, (2, 3)))
    assert bool(jnp.all(lab.valid))
    with pytest.raises(ValueError, match="complete emission"):
        standard.lab_far_field(final, boosted, emission="truncated")


# -- antennas ------------------------------------------------------------------------------


def _antenna_bridge() -> Any:
    grid = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(4, periodic=True),
            D.UniformCellAxisSpec(4, periodic=True),
            D.UniformCellAxisSpec(24, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 6.0]]))
    return D.StructuredCochainBridge(grid)


def _lab_antenna(beta: float) -> Any:
    times = np.linspace(0.0, 4.0, 9)
    electric = np.zeros((2, 2, 9, 2), dtype=np.complex128)
    electric[..., 0] = np.exp(-((times - 2.0) ** 2))
    return phx.solver.maxwell.SampledPlaneCurrentAntennaPlan(
        _antenna_bridge(),
        2,
        2.5,
        [0.0, 1.0],
        [0.0, 1.0],
        times,
        electric,
        carrier_angular_frequency=6.0,
        beta=beta,
        scale=_scale(),
    )


@pytest.mark.parametrize("lab_beta", [0.0, 0.3], ids=["resting", "moving"])
def test_boosted_antenna_follows_the_lab_sheet_world_line(lab_beta: float) -> None:
    gamma = 2.0
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    frame = _frame(gamma, 0.0, 6.0)
    lab = _lab_antenna(lab_beta)
    boosted = frame.boost_antenna(lab, _antenna_bridge())
    assert boosted.beta == pytest.approx((lab_beta - beta) / (1.0 - lab_beta * beta))
    boosted_gamma = 1.0 / np.sqrt(1.0 - boosted.beta**2)
    lab_gamma = 1.0 / np.sqrt(1.0 - lab_beta**2)
    for time in (-3.0, 0.0, 2.5):
        position = jnp.asarray([0.0, 0.0, boosted.plane_coordinate + boosted.beta * time])
        lab_time, lab_position = frame.to_lab(time, position)
        # The boosted sheet is the lab sheet: z = z₀ + β_a c t in the lab ...
        assert float(lab_position[2]) == pytest.approx(
            lab.plane_coordinate + lab_beta * float(lab_time), abs=1e-12
        )
        # ... and both label each sheet event with the same rest-frame sample.
        np.testing.assert_allclose(
            np.asarray(boosted.times) - time / boosted_gamma,
            np.asarray(lab.times) - float(lab_time) / lab_gamma,
            rtol=0,
            atol=1e-12,
        )


def test_boosted_antenna_refuses_other_normals_and_missing_scales() -> None:
    lab = _lab_antenna(0.0)
    oblique = S.BoostedFramePlan(
        phx.LorentzFrame(phx.boost_matrix(jnp.asarray([0.5, 0.0, 0.0]))),
        phx.geometry.Box([0.5, 0.5, 3.0], [1.0, 1.0, 6.0]),
    )
    with pytest.raises(ValueError, match="normal axis"):
        oblique.boost_antenna(lab, _antenna_bridge())
    unscaled = phx.solver.maxwell.SampledPlaneCurrentAntennaPlan(
        _antenna_bridge(),
        2,
        2.5,
        [0.0, 1.0],
        [0.0, 1.0],
        lab.times,
        lab.electric,
    )
    with pytest.raises(ValueError, match="ElectromagneticScaleContract"):
        _frame(2.0, 0.0, 6.0).boost_antenna(unscaled, _antenna_bridge())


# -- laser wakefield stage: boosted versus lab ---------------------------------------------


_LWFA_WAVENUMBER = 2.0 * np.pi
_LWFA_PLASMA_FREQUENCY2 = 0.01 * _LWFA_WAVENUMBER**2
_LWFA_LASER = _PlaneLaser(_LWFA_WAVENUMBER, 20.0, 5.0, _LWFA_WAVENUMBER)
_LWFA_PLASMA = (26.0, 42.0)
_LWFA_WITNESSES = np.linspace(5.0, 14.5, 21)
_LWFA_WITNESS_GAMMA = 20.0
_LWFA_EXIT = 43.0
_LWFA_POINTS_PER_WAVELENGTH = 16


@dataclass(frozen=True)
class _LWFARuns:
    lab_gain: np.ndarray
    boosted_gain: np.ndarray
    snapshot_positions: np.ndarray
    snapshot_electric: np.ndarray
    lab_electric: np.ndarray
    particle_positions: np.ndarray
    particle_filled: np.ndarray
    lab_witness_positions: np.ndarray
    evidence: Any
    accepted: tuple[bool, bool]


def _witness_gain(trajectory: Any) -> np.ndarray:
    """Lorentz factor gained by each lane when it crosses the plasma exit plane."""
    positions = np.asarray(trajectory.positions)
    velocities = np.asarray(trajectory.proper_velocities)
    active = np.asarray(trajectory.active)
    gains = []
    for lane in range(positions.shape[1]):
        mask = active[:, lane]
        gamma = np.sqrt(1.0 + np.sum(velocities[mask, lane] ** 2, axis=-1))
        gains.append(np.interp(_LWFA_EXIT, positions[mask, lane, 2], gamma))
    return np.asarray(gains) - _LWFA_WITNESS_GAMMA


def _stage_species(
    plasma: np.ndarray, weight: float
) -> tuple[tuple[Any, Any], tuple[Any, Any], tuple[np.ndarray, np.ndarray]]:
    """Electrons (witness lanes first, macro weight 1e-9) and resting ions."""
    lanes, count = _LWFA_WITNESSES.size, plasma.shape[0]
    electrons, electron_charge = _species(0, lanes + count, -1.0, "electrons")
    ions, ion_charge = _species(10**6, count, 1.0 / 1836.0, "ions")
    masses = (
        np.concatenate((np.full(lanes, 1.0e-9 * weight), np.full(count, weight))),
        np.full(count, 1836.0 * weight),
    )
    return (electrons, ions), (electron_charge, ion_charge), masses


def _stage_electrons(plasma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lanes = _LWFA_WITNESSES.size
    witnesses = np.stack(
        (np.full(lanes, 1.5), np.full(lanes, 1.5), _LWFA_WITNESSES), axis=-1
    )
    velocities = np.zeros((lanes + plasma.shape[0], 3))
    velocities[:lanes, 2] = np.sqrt(1.0 - 1.0 / _LWFA_WITNESS_GAMMA**2)
    return np.concatenate((witnesses, plasma)), velocities


def _column(spacing: float, per_cell: int, lower: float, upper: float) -> np.ndarray:
    """Quiet-start plasma column, ``per_cell`` equally spaced particles per cell."""
    first, last = int(np.ceil(lower / spacing)), int(np.floor(upper / spacing))
    z = (np.arange(first * per_cell, last * per_cell) + 0.5) * spacing / per_cell
    return np.stack((np.full(z.size, 1.5), np.full(z.size, 1.5), z), axis=-1)


_LWFA_ANTENNA_PLANE = 22.0


def _lwfa_antenna(bridge: Any) -> Any:
    """Lab antenna at rest in vacuum at ``z_a`` launching ``_LWFA_LASER`` along +z.

    The sheet samples the laser's trace ``f(z_a − t − z₀)``, so the launched field
    ``Θ(z − z_a) f`` is the lab laser wherever the plasma (beyond ``z_a``) or the
    witnesses (trailing the pulse) meet it, and every sampled event lies after
    the boosted start slice through the plasma entrance.
    """
    laser = _LWFA_LASER
    times = (
        np.linspace(-laser.half_length, laser.half_length, 801)
        + _LWFA_ANTENNA_PLANE
        - laser.center
    )
    phase = _LWFA_ANTENNA_PLANE - times - laser.center
    electric = np.zeros((2, 2, times.size, 2), dtype=np.complex128)
    electric[..., 0] = (
        laser.amplitude
        * np.cos(0.5 * np.pi * phase / laser.half_length) ** 2
        * np.exp(1j * laser.wavenumber * (_LWFA_ANTENNA_PLANE - laser.center))
    )
    return phx.solver.maxwell.SampledPlaneCurrentAntennaPlan(
        bridge,
        2,
        _LWFA_ANTENNA_PLANE,
        [-1.0, 4.0],
        [-1.0, 4.0],
        times,
        electric,
        carrier_angular_frequency=laser.wavenumber,
        scale=_scale(),
    )


def _lwfa_travel() -> float:
    """Lab time for the first witness to pass the plasma exit plane."""
    beta = np.sqrt(1.0 - 1.0 / _LWFA_WITNESS_GAMMA**2)
    return float((_LWFA_EXIT + 0.5 - _LWFA_WITNESSES[0]) / beta)


def _boosted_stage(
    *, antenna: bool, snapshots: Any = None
) -> tuple[Any, Any, float, int]:
    """The stage in the γ_b = 2 frame on a Galilean grid, ready to run.

    The laser enters as the boosted lab vacuum field on the start slice, or,
    with ``antenna``, through the boosted `_lwfa_antenna` sheet.
    """
    lanes = _LWFA_WITNESSES.size
    # The slice t′₀ through the lab plasma entrance at t = 0 leaves the plasma
    # unperturbed and the laser in vacuum.
    gamma = 2.0
    frame = _frame(gamma, 0.0, 48.0)
    start, _ = frame.from_lab(0.0, jnp.asarray([1.5, 1.5, _LWFA_PLASMA[0]]))
    stop, _ = frame.from_lab(_lwfa_travel(), jnp.asarray([1.5, 1.5, _LWFA_EXIT + 0.5]))
    # About 16 cells per Doppler-stretched wavelength, with the contracted
    # plasma edges on node planes.
    entrance, exit_ = (value / gamma for value in _LWFA_PLASMA)
    wavelength = gamma * (1.0 + np.sqrt(1.0 - 1.0 / gamma**2))
    cells = round((exit_ - entrance) * _LWFA_POINTS_PER_WAVELENGTH / wavelength)
    spacing = (exit_ - entrance) / cells
    probe_electrons, probe_velocities = _stage_electrons(np.zeros((0, 3)))
    probe = frame.boost_particles(probe_electrons, probe_velocities, boosted_time=start)
    trailing = float(
        jnp.min(probe.positions[:, 2]) - float(start) * frame.galilean_velocity[2]
    )
    lower = entrance - np.ceil((entrance - trailing + 4.0) / spacing) * spacing
    count = int(np.ceil((30.0 - lower) / spacing))
    upper = lower + count * spacing
    column = entrance + (np.arange(4 * cells) + 0.5) * spacing / 4
    boosted_plasma = np.stack(
        (np.full(column.size, 1.5), np.full(column.size, 1.5), gamma * column), axis=-1
    )
    plans, charged, masses = _stage_species(
        boosted_plasma, _LWFA_PLASMA_FREQUENCY2 * 9.0 * gamma * spacing / 4
    )
    electrons, velocities = _stage_electrons(boosted_plasma)
    boosted_electrons = frame.boost_particles(electrons, velocities, boosted_time=start)
    ions = frame.boost_particles(
        boosted_plasma, np.zeros_like(boosted_plasma), boosted_time=start
    )
    dt = 0.45 * spacing / 1.85
    steps = int(np.ceil((float(stop) - float(start)) / dt))
    bridge = _bridge(count, lower, upper)
    antennas = (
        (frame.boost_antenna(_lwfa_antenna(_bridge(96, 0.0, 48.0)), bridge),)
        if antenna
        else ()
    )
    boosted = frame.prepare(
        S.ElectromagneticPICPlan(
            _solver(bridge, charged, antennas=antennas, **_galilean(frame)),
            species=plans,
            recorders=(_recorder(plans, lanes, steps),),
        ),
        snapshots=snapshots,
    )
    state = boosted.initialize(
        (boosted_electrons.positions, ions.positions),
        (boosted_electrons.velocities, ions.velocities),
        dt,
        time=start,
        masses=masses,
        vacuum_fields=() if antenna else (_LWFA_LASER,),
    )
    return boosted, state, dt, steps


@pytest.fixture(scope="module")
def lwfa_runs() -> _LWFARuns:
    """A one-dimensional LWFA stage: 21 test witnesses across one plasma period."""
    scale = _scale()
    lanes = _LWFA_WITNESSES.size
    travel = _lwfa_travel()

    # Lab frame: whole stage in one periodic box.
    dz = 1.0 / _LWFA_POINTS_PER_WAVELENGTH
    dt = 0.45 * dz
    plasma = _column(dz, 2, *_LWFA_PLASMA)
    plans, charged, masses = _stage_species(
        plasma, _LWFA_PLASMA_FREQUENCY2 * 9.0 * dz / 2
    )
    electrons, velocities = _stage_electrons(plasma)
    steps = int(np.ceil(travel / dt))
    snapshot_step = int(round(30.0 / dt))
    lab_time = snapshot_step * dt
    count = int(round(48.0 / dz))
    solver = _solver(_bridge(count, 0.0, 48.0), charged)
    lab_recorder = _recorder(plans, lanes, steps)
    pic = S.ElectromagneticPICPlan(solver, species=plans, recorders=(lab_recorder,))
    state = pic.initialize(
        (electrons, plasma), (velocities, np.zeros_like(plasma)), dt, masses=masses
    )
    z = jnp.arange(count) * dz
    laser = _laser_sample(_LWFA_LASER, z, jnp.zeros_like(z))
    state = eqx.tree_at(
        lambda value: (value.field.electric, value.field.magnetic),
        state,
        (
            state.field.electric + laser.electric[None, None],
            state.field.magnetic + laser.magnetic[None, None],
        ),
    )
    middle, lab_ok_first = _scan(pic.step_detailed, state, dt, snapshot_step)
    lab_field = np.asarray(middle.field.electric[1, 1])
    lab_final, lab_ok = _scan(pic.step_detailed, middle, dt, steps - snapshot_step)
    lab_tracks = lab_recorder.to_charged_trajectory(lab_final.recorders[0], scale)

    boosted, state, dt, steps = _boosted_stage(
        antenna=False, snapshots=S.BoostedSnapshotPlan((lab_time,), species=(0,))
    )
    final, boosted_ok = _scan(boosted.step_detailed, state, dt, steps)
    field = boosted.lab_field_snapshot(final, 0)
    particles = boosted.lab_particle_snapshot(final, 0, 0)
    tracks = lab_tracks
    witness_positions = np.asarray(
        [
            np.interp(
                lab_time,
                np.asarray(tracks.times[:, lane]),
                np.asarray(tracks.positions[:, lane, 2]),
            )
            for lane in range(lanes)
        ]
    )
    filled = np.asarray(field.filled)
    return _LWFARuns(
        _witness_gain(lab_tracks),
        _witness_gain(boosted.lab_trajectory(final, 0, scale)),
        np.asarray(field.axis_coordinates)[filled],
        np.asarray(field.electric[1, 1])[filled],
        np.stack(
            tuple(
                np.interp(
                    np.asarray(field.axis_coordinates)[filled],
                    np.arange(lab_field.shape[0]) * dz,
                    lab_field[:, component],
                )
                for component in range(3)
            ),
            axis=-1,
        ),
        np.asarray(particles.positions[:lanes, 2]),
        np.asarray(particles.filled[:lanes]),
        witness_positions,
        final.evidence,
        (
            bool(jnp.all(lab_ok_first) & jnp.all(lab_ok)),
            bool(jnp.all(boosted_ok)),
        ),
    )


def _fundamental(phases: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Amplitude and mean of the plasma-period harmonic fitted to ``values``."""
    wavenumber = np.sqrt(_LWFA_PLASMA_FREQUENCY2)
    basis = np.stack(
        (np.sin(wavenumber * phases), np.cos(wavenumber * phases), np.ones_like(phases)),
        axis=-1,
    )
    coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
    return float(np.hypot(coefficients[0], coefficients[1])), float(coefficients[2])


def test_boosted_lwfa_stage_gains_the_lab_energy(lwfa_runs: _LWFARuns) -> None:
    assert lwfa_runs.accepted == (True, True)
    assert int(lwfa_runs.evidence.nci_rejections) == 0
    lab, boosted = lwfa_runs.lab_gain, lwfa_runs.boosted_gain
    # Witnesses sample one plasma period behind the laser; the stage's energy
    # gain is the amplitude of that phase curve.
    lab_amplitude, lab_mean = _fundamental(_LWFA_WITNESSES, lab)
    boosted_amplitude, boosted_mean = _fundamental(_LWFA_WITNESSES, boosted)
    assert lab_amplitude > 2.0
    assert abs(boosted_amplitude / lab_amplitude - 1.0) < 0.05
    assert abs(boosted_mean - lab_mean) < 0.05 * lab_amplitude
    assert np.max(np.abs(boosted - lab)) < 0.15 * lab_amplitude


def test_boosted_antenna_seeds_the_lwfa_stage_like_the_lab_laser(
    lwfa_runs: _LWFARuns,
) -> None:
    # The laser now enters through the boosted lab antenna, a sheet receding at
    # −β_b c (stationary on the Galilean grid), instead of a loaded vacuum field.
    boosted, state, dt, steps = _boosted_stage(antenna=True)
    final, successful = _scan(boosted.step_detailed, state, dt, steps)
    gain = _witness_gain(boosted.lab_trajectory(final, 0, _scale()))
    lab_amplitude, lab_mean = _fundamental(_LWFA_WITNESSES, lwfa_runs.lab_gain)
    amplitude, mean = _fundamental(_LWFA_WITNESSES, gain)

    assert bool(jnp.all(successful))
    # The band-limited sheet leaves the high-|k| shells free of source energy.
    assert int(final.evidence.nci_rejections) == 0
    assert abs(amplitude / lab_amplitude - 1.0) < 0.05
    assert abs(mean - lab_mean) < 0.05 * lab_amplitude


def test_back_transformed_wake_snapshot_matches_the_lab_run(
    lwfa_runs: _LWFARuns,
) -> None:
    z = lwfa_runs.snapshot_positions
    boosted, lab = lwfa_runs.snapshot_electric, lwfa_runs.lab_electric
    assert z.size > 20
    wake = (z > _LWFA_PLASMA[0] + 1.0) & (z < _LWFA_PLASMA[1] - 1.0)
    # Longitudinal wake field: amplitude and shape within PIC noise.
    peak = np.max(np.abs(lab[wake, 2]))
    assert abs(np.max(np.abs(boosted[wake, 2])) / peak - 1.0) < 0.15
    assert np.linalg.norm(boosted[wake, 2] - lab[wake, 2]) < 0.3 * np.linalg.norm(
        lab[wake, 2]
    )


def test_back_transformed_particles_sit_on_the_lab_witnesses(
    lwfa_runs: _LWFARuns,
) -> None:
    filled = lwfa_runs.particle_filled
    assert np.count_nonzero(filled) == _LWFA_WITNESSES.size
    np.testing.assert_allclose(
        lwfa_runs.particle_positions[filled],
        lwfa_runs.lab_witness_positions[filled],
        rtol=0,
        atol=2.0e-3,
    )
