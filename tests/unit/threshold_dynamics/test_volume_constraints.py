#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import numpy as np
import pytest

import phydrax.threshold_dynamics as td
from phydrax.interfacial_transport import InterfaceMobilityMatrix, InterfaceTensionMatrix


LABELS = ("bubble", "air")


@eqx.filter_jit
def _run(
    prepared: td.PreparedThresholdDynamics, state: td.LabelFieldState, steps: int
) -> td.ThresholdDynamicsRunResult:
    return prepared.run(state, steps)


@eqx.filter_jit
def _step(
    prepared: td.PreparedThresholdDynamics, state: td.LabelFieldState
) -> td.ThresholdDynamicsStepResult:
    return prepared.step(state)


@eqx.filter_jit
def _coarsen(
    coarsening: td.GasDiffusionCoarsening, state: td.LabelFieldState, steps: int
) -> td.GasDiffusionRunResult:
    return coarsening.run(state, steps)


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(n) + 0.5) / n
    return np.meshgrid(x, x, indexing="ij")


def _prepared(
    n: int, dt: float, counts: np.ndarray, *, mobility: float = 1.0
) -> td.PreparedThresholdDynamics:
    plan = td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(LABELS, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(LABELS, mobility, structure="uniform"),
        dt,
        volume_constraint=td.LabelVolumeConstraint(counts),
        minimum_resolution_ratio=0.0,
    )
    return plan.prepare(td.PeriodicGridHeatKernel((n, n), (1.0, 1.0)))


def _ellipse(n: int) -> np.ndarray:
    x, y = _grid(n)
    return np.where(((x - 0.5) / 0.3) ** 2 + ((y - 0.5) / 0.15) ** 2 < 1.0, 0, 1)


def _aspect(field: np.ndarray) -> float:
    x, y = _grid(field.shape[0])
    inside = field == 0
    spread = [np.var(axis[inside]) for axis in (x, y)]
    return float(np.sqrt(max(spread) / min(spread)))


def test_exact_volumes_relax_an_ellipse_toward_a_circle() -> None:
    n = 128
    field = _ellipse(n)
    counts = np.bincount(field.reshape(-1))
    prepared = _prepared(n, 1e-3, counts)
    result = _run(prepared, prepared.initial_state(field), 12)
    evidence = result.evidence
    energies = np.concatenate(
        ([float(result.initial_energy)], np.asarray(evidence.energy_after))
    )

    assert np.all(np.asarray(evidence.status) == int(td.ThresholdDynamicsStatus.SUCCESS))
    np.testing.assert_array_equal(evidence.label_counts, np.broadcast_to(counts, (12, 2)))
    assert np.all(np.asarray(evidence.dissipation_admitted))
    assert np.all(np.diff(energies) <= np.asarray(evidence.energy_tolerance))
    assert _aspect(np.asarray(result.state.labels)) < 0.8 * _aspect(field)


def test_prescribed_counts_shrink_a_region_exactly() -> None:
    n = 64
    x, y = _grid(n)
    field = np.where((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.25**2, 0, 1)
    counts = np.bincount(field.reshape(-1))
    target = counts + np.array([-150, 150])
    prepared = _prepared(n, 1e-3, target)
    result = _step(prepared, prepared.initial_state(field))

    assert bool(result.committed)
    np.testing.assert_array_equal(result.evidence.label_counts, target)
    np.testing.assert_array_equal(
        np.bincount(np.asarray(result.state.labels).reshape(-1)), target
    )
    assert not bool(result.evidence.dissipation_admitted)


def test_lower_count_bound_stops_a_shrinking_region() -> None:
    n = 64
    x, y = _grid(n)
    field = np.where((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.15**2, 0, 1)
    counts = np.bincount(field.reshape(-1))
    floor = int(0.8 * counts[0])
    plan = td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(LABELS, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(LABELS, 1.0, structure="uniform"),
        1e-3,
        volume_constraint=td.LabelVolumeConstraint(
            np.array([floor, 0]), np.array([counts[0], n * n])
        ),
        minimum_resolution_ratio=0.0,
    )
    prepared = plan.prepare(td.PeriodicGridHeatKernel((n, n), (1.0, 1.0)))
    result = _run(prepared, prepared.initial_state(field), 16)
    inside = np.asarray(result.evidence.label_counts[:, 0])

    assert np.all(np.asarray(result.evidence.committed))
    assert np.all(inside >= floor)
    assert inside[-1] == floor
    assert np.all(np.diff(inside) <= 0)


def test_infeasible_counts_roll_back_without_partial_assignment() -> None:
    n = 32
    field = np.where(_ellipse(n) == 0, 0, 1)
    counts = np.bincount(field.reshape(-1))
    prepared = _prepared(n, 1e-3, counts + np.array([5, 0]))
    state = prepared.initial_state(field)
    result = _step(prepared, state)

    assert int(result.status) == int(td.ThresholdDynamicsStatus.VOLUME_CONSTRAINT_FAILED)
    assert not bool(result.committed)
    np.testing.assert_array_equal(result.state.labels, state.labels)
    volume = result.evidence.volume
    assert volume is not None
    assert not bool(volume.auction.bounds_consistent)


def test_prices_normalize_to_laplace_pressure_on_a_circle() -> None:
    # P = -sqrt(pi) p up to a gauge: sqrt(pi) (p_air - p_bubble) = sigma / R at rest.
    n = 256
    x, y = _grid(n)
    field = np.where((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.2**2, 0, 1)
    counts = np.bincount(field.reshape(-1))
    prepared = _prepared(n, 1e-3, counts)
    result = _run(prepared, prepared.initial_state(field), 2)
    volume = result.evidence.volume
    assert volume is not None
    prices = np.asarray(volume.prices)
    radius = np.sqrt(counts[0] / n**2 / np.pi)
    jump = np.sqrt(np.pi) * np.mean(prices[:, 1] - prices[:, 0])

    np.testing.assert_allclose(jump * radius, 1.0, rtol=0.1)


def test_gas_diffusion_shrinks_a_bubble_at_the_series_rate() -> None:
    # Isolated circle: dA/dt = -2 pi sigma k mu / (k + mu), total volume conserved.
    n, dt, permeance, mobility, steps = 256, 1e-3, 0.5, 1.0, 16
    x, y = _grid(n)
    field = np.where((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.2**2, 0, 1)
    prepared = _prepared(n, dt, np.bincount(field.reshape(-1)), mobility=mobility)
    coarsening = td.GasDiffusionCoarsening(prepared, permeance)
    result = _coarsen(coarsening, prepared.initial_state(field), steps)
    counts = np.asarray(result.evidence.step.label_counts)
    areas = counts[:, 0] / n**2
    slope = np.polyfit(dt * np.arange(1, steps + 1), areas, 1)[0]
    effective = permeance * mobility / (permeance + mobility)

    assert np.all(np.asarray(result.evidence.step.committed))
    np.testing.assert_array_equal(counts.sum(axis=1), n * n)
    np.testing.assert_allclose(slope, -2.0 * np.pi * effective, rtol=0.05)
    np.testing.assert_allclose(
        result.evidence.film_areas[:, 0], 2.0 * np.pi * np.sqrt(areas / np.pi), rtol=0.03
    )


def test_constraint_and_coarsening_refusals() -> None:
    with pytest.raises(ValueError):
        td.LabelVolumeConstraint(np.array([3, 1]), np.array([2, 1]))
    with pytest.raises(ValueError):
        td.LabelVolumeConstraint(np.array([-1, 1]))

    unconstrained = td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(LABELS, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(LABELS, 1.0, structure="uniform"),
        1e-3,
    ).prepare(td.PeriodicGridHeatKernel((16, 16), (1.0, 1.0)))
    with pytest.raises(ValueError, match="exact cell volumes"):
        td.GasDiffusionCoarsening(unconstrained, 1.0)
    bounded = td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(LABELS, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(LABELS, 1.0, structure="uniform"),
        1e-3,
        volume_constraint=td.LabelVolumeConstraint(
            np.array([10, 200]), np.array([60, 250])
        ),
    ).prepare(td.PeriodicGridHeatKernel((16, 16), (1.0, 1.0)))
    with pytest.raises(ValueError, match="exact cell volumes"):
        td.GasDiffusionCoarsening(bounded, 1.0)


@pytest.mark.parametrize(
    "counts",
    [
        np.asarray([2**31, 0], dtype=np.int64),
        np.asarray([2**32, 0], dtype=np.uint64),
    ],
    ids=["int32-max-plus-one", "uint32-wrap"],
)
def test_label_volume_constraint_refuses_counts_outside_int32(
    counts: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="signed int32"):
        td.LabelVolumeConstraint(counts)


def test_label_volume_constraint_accepts_int32_maximum_bound() -> None:
    maximum = np.iinfo(np.int32).max
    constraint = td.LabelVolumeConstraint(
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([maximum, maximum], dtype=np.int64),
    )

    assert constraint.lower_counts.dtype == np.int32
    assert constraint.upper_counts.dtype == np.int32
    np.testing.assert_array_equal(constraint.upper_counts, [maximum, maximum])


def test_label_volume_constraint_requires_integer_counts() -> None:
    with pytest.raises(TypeError):
        td.LabelVolumeConstraint(np.asarray([1.0, 1.0]))
