#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax.applications.numerical_relativity._dynamical_horizon import (
    DynamicalHorizonBalancePlan,
    DynamicalHorizonStatus,
    HorizonWorldtubeRegime,
    QuasilocalHorizonWorldtube,
)
from phydrax.applications.numerical_relativity._event_horizon import (
    CompletedSpacetimeHistory,
    EventHorizonStatus,
    OfflineEventHorizonTracingPlan,
)
from phydrax.applications.numerical_relativity._horizon_tracking import (
    HorizonGeometryEvidence,
)
from phydrax.applications.numerical_relativity._mots import (
    MOTSSolveResult,
    MOTSStabilityEvidence,
)
from phydrax.applications.numerical_relativity._surfaces import (
    SphericalSpectralSurface,
)
from phydrax.metrix._adm_exchange import ADMGridGeometry


def _completed_minkowski_history(times, *, transverse_rate, completed=True):
    axes = tuple(np.asarray((-2.0, 0.0, 2.0)) for _ in range(3))
    shape = (len(times), 3, 3, 3)
    alpha = np.ones(shape)
    beta = np.zeros(shape + (3,))
    beta[..., 0] = 1.0
    beta[..., 1] = -transverse_rate * axes[1][None, None, :, None]
    metric = np.broadcast_to(np.eye(3), shape + (3, 3)).copy()
    geometry = ADMGridGeometry(
        alpha,
        beta,
        metric,
        metric,
        np.ones(shape),
        np.zeros(shape + (3, 3)),
        np.ones(shape, dtype="bool"),
        np.ones(shape, dtype="bool"),
        snapshot_token=0,
        chart_id="minkowski-cartesian",
        convention_id="mostly-plus",
        scale_id="geometric",
        topology_id="fixed-cartesian-box",
        geometry_lineage_id=f"minkowski-flow-{transverse_rate}",
    )
    return CompletedSpacetimeHistory(
        times,
        *axes,
        geometry,
        completed=completed,
        completion_id=f"completed-minkowski-{transverse_rate}",
    )


def _trace_plan(times, *, terminal_positions, caustic_distance):
    surface = SphericalSpectralSurface(
        np.asarray(((1.0 + 0.0j,),)),
        np.zeros(3),
        "qualified-late-time-surface-plan",
    )
    terminal_covectors = np.broadcast_to(
        np.asarray((1.0, 0.0, 0.0)), terminal_positions.shape
    )
    return OfflineEventHorizonTracingPlan(
        surface,
        terminal_positions,
        terminal_covectors,
        np.ones((2,), dtype="bool"),
        np.asarray(((1,), (0,))),
        time_capacity=len(times),
        grid_shape=(3, 3, 3),
        terminal_surface_qualified=True,
        absolute_tolerance=2.0e-7,
        relative_tolerance=2.0e-6,
        null_tolerance=2.0e-6,
        caustic_distance=caustic_distance,
    )


def test_offline_generators_converge_backward_and_report_caustics():
    times = np.linspace(0.0, 1.0, 9)
    transverse_rate = 2.0
    history = _completed_minkowski_history(times, transverse_rate=transverse_rate)
    terminal = np.asarray(((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)))
    trace = _trace_plan(times, terminal_positions=terminal, caustic_distance=0.2).trace(
        history
    )

    expected_initial_y = terminal[:, 1] * np.exp(-transverse_rate)
    np.testing.assert_allclose(
        trace.generator_trajectories[0, :, 1],
        expected_initial_y,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        trace.generator_covectors,
        np.broadcast_to((1.0, 0.0, 0.0), trace.generator_covectors.shape),
        atol=2.0e-6,
    )
    assert float(np.max(np.asarray(trace.geodesic_residual))) < 3.0e-7
    assert bool(trace.converged)
    assert bool(trace.coverage_complete)
    assert bool(trace.caustic_detected)
    assert bool(trace.qualified)
    assert not bool(trace.derivative_valid)
    assert int(trace.status) & int(EventHorizonStatus.CAUSTIC_DETECTED)
    assert trace.generator_trajectories.shape == (len(times), 2, 3)


def test_event_horizon_history_must_be_completed_before_tracing():
    with pytest.raises(ValueError, match="completed"):
        _completed_minkowski_history(
            np.linspace(0.0, 1.0, 3),
            transverse_rate=0.0,
            completed=False,
        )


def test_offline_event_horizon_trace_fails_closed_outside_completed_coverage():
    times = np.linspace(0.0, 1.0, 5)
    history = _completed_minkowski_history(times, transverse_rate=0.0)
    terminal = np.asarray(((0.0, -2.5, 0.0), (0.0, 2.5, 0.0)))
    trace = _trace_plan(times, terminal_positions=terminal, caustic_distance=0.1).trace(
        history
    )
    assert not bool(trace.coverage_complete)
    assert not bool(trace.qualified)
    assert int(trace.status) & int(EventHorizonStatus.OUTSIDE_HISTORY_COVERAGE)
    assert bool(trace.global_history_complete)


def _quasilocal_slices(masses, *, mots_derivative_valid=True):
    mots = []
    geometries = []
    for index, mass in enumerate(masses):
        surface = SphericalSpectralSurface(
            np.asarray(((complex(mass),),)),
            np.zeros(3),
            "fixed-spherical-surface-plan",
        )
        stability = MOTSStabilityEvidence(
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1,)),
            np.asarray(0.0),
            np.asarray(0.0),
            np.asarray(True),
            np.asarray(True),
            np.asarray(True),
            np.asarray(True),
        )
        mots.append(
            MOTSSolveResult(
                surface,
                np.zeros((1,)),
                np.asarray(0.0),
                np.asarray(index + 1, dtype=np.int32),
                stability,
                np.asarray(True),
                np.asarray(True),
                np.asarray(True),
                np.asarray(True),
                np.asarray(mots_derivative_valid),
                np.asarray(0, dtype=np.int32),
                "fixed-mots-plan",
            )
        )
        area = 16.0 * np.pi * mass**2
        geometries.append(
            HorizonGeometryEvidence(
                np.asarray(area),
                np.asarray(2.0 * mass),
                np.asarray(mass),
                np.asarray(0.0),
                np.asarray(mass),
                np.asarray(0.0),
                np.asarray(0.0),
                np.asarray(True),
                np.asarray(True),
                np.asarray(True),
                np.asarray(True),
                "mostly-plus",
            )
        )
    return tuple(mots), tuple(geometries)


def test_dynamical_and_isolated_worldtube_flux_laws_are_quasilocal():
    times = np.linspace(0.0, 2.0, 5)
    energy_flux = 0.04
    masses = 1.0 + energy_flux * times
    mots, geometries = _quasilocal_slices(masses)
    dynamical_worldtube = QuasilocalHorizonWorldtube(
        times,
        mots,
        geometries,
        np.full(times.shape, energy_flux),
        np.zeros(times.shape),
        np.zeros(times.shape),
        np.zeros(times.shape),
        np.ones(times.shape),
        worldtube_name="analytic-dynamical-worldtube",
    )
    plan = DynamicalHorizonBalancePlan(
        len(times), absolute_tolerance=2.0e-7, relative_tolerance=2.0e-6
    )
    dynamical = plan.evaluate(dynamical_worldtube)
    np.testing.assert_allclose(dynamical.energy_balance_residual, 0.0, atol=2.0e-7)
    np.testing.assert_allclose(
        dynamical.angular_momentum_balance_residual, 0.0, atol=2.0e-7
    )
    assert np.all(np.asarray(dynamical.regime) == int(HorizonWorldtubeRegime.DYNAMICAL))
    assert bool(dynamical.qualified)

    isolated_mots, isolated_geometries = _quasilocal_slices(
        np.ones_like(times), mots_derivative_valid=False
    )
    isolated_worldtube = QuasilocalHorizonWorldtube(
        times,
        isolated_mots,
        isolated_geometries,
        np.zeros(times.shape),
        np.zeros(times.shape),
        np.zeros(times.shape),
        np.zeros(times.shape),
        np.zeros(times.shape),
        worldtube_name="analytic-isolated-worldtube",
    )
    isolated = plan.evaluate(isolated_worldtube)
    assert np.all(np.asarray(isolated.isolated))
    assert np.all(np.asarray(isolated.regime) == int(HorizonWorldtubeRegime.ISOLATED))
    assert bool(isolated.qualified)
    assert not bool(isolated.derivative_valid)


def test_area_law_checks_every_worldtube_interval():
    times = np.asarray((0.0, 1.0, 2.0))
    mots, geometries = _quasilocal_slices(np.asarray((2.0, 1.0, 2.0)))
    zeros = np.zeros(times.shape)
    worldtube = QuasilocalHorizonWorldtube(
        times,
        mots,
        geometries,
        zeros,
        zeros,
        zeros,
        zeros,
        np.ones(times.shape),
        worldtube_name="area-decrease-control",
    )
    result = DynamicalHorizonBalancePlan(
        3,
        absolute_tolerance=2.0,
        relative_tolerance=0.0,
    ).evaluate(worldtube)
    assert bool(result.converged)
    assert float(result.area_interval_rate[0]) < 0.0
    assert not bool(result.physically_valid)
    assert not bool(result.qualified)
    assert int(result.status) & int(DynamicalHorizonStatus.AREA_DECREASE)
