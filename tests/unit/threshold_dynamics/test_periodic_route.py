#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.threshold_dynamics as td
from phydrax.interfacial_transport import InterfaceMobilityMatrix, InterfaceTensionMatrix


ROOT2 = float(np.sqrt(2.0))


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


def _uniform(
    labels: Sequence[str],
) -> tuple[InterfaceTensionMatrix, InterfaceMobilityMatrix]:
    return (
        InterfaceTensionMatrix(labels, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(labels, 1.0, structure="uniform"),
    )


def _centers(n: int, dimension: int) -> list[np.ndarray]:
    x = (np.arange(n) + 0.5) / n
    return list(np.meshgrid(*([x] * dimension), indexing="ij"))


def _ball(n: int, dimension: int, radius: float, center: float = 0.5) -> np.ndarray:
    squared = sum((axis - center) ** 2 for axis in _centers(n, dimension))
    return np.where(squared < radius**2, 0, 1)


def _prepare(
    plan: td.ThresholdDynamicsPlan, n: int, dimension: int = 2
) -> td.PreparedThresholdDynamics:
    return plan.prepare(td.PeriodicGridHeatKernel((n,) * dimension, (1.0,) * dimension))


def _area_rate_error(n: int, dt: float, steps: int) -> float:
    prepared = _prepare(td.ThresholdDynamicsPlan(*_uniform(("in", "out")), dt), n)
    labels = _ball(n, 2, 0.3)
    result = _run(prepared, prepared.initial_state(labels), steps)
    areas = (
        np.concatenate(
            ([np.sum(labels == 0)], np.asarray(result.evidence.label_counts[:, 0]))
        )
        / n**2
    )
    slope = np.polyfit(dt * np.arange(steps + 1), areas, 1)[0]
    return abs(slope / (-2.0 * np.pi) - 1.0)


def test_shrinking_circle_area_rate_converges_under_joint_refinement() -> None:
    # Curve shortening: dA/dt = -2 pi mu sigma, independent of the radius.
    coarse = _area_rate_error(256, 2e-3, 12)
    fine = _area_rate_error(512, 1e-3, 24)

    assert coarse < 0.03
    assert fine < 0.015
    assert fine < coarse


def test_shrinking_sphere_follows_mean_curvature_flow() -> None:
    n, dt, steps = 64, 2e-3, 8
    prepared = _prepare(td.ThresholdDynamicsPlan(*_uniform(("in", "out")), dt), n, 3)
    labels = _ball(n, 3, 0.3)
    result = _run(prepared, prepared.initial_state(labels), steps)
    volumes = (
        np.concatenate(
            ([np.sum(labels == 0)], np.asarray(result.evidence.label_counts[:, 0]))
        )
        / n**3
    )
    squared_radius = (3.0 * volumes / (4.0 * np.pi)) ** (2.0 / 3.0)
    slope = np.polyfit(dt * np.arange(steps + 1), squared_radius, 1)[0]

    np.testing.assert_allclose(slope, -4.0, rtol=0.05)
    assert np.all(np.asarray(result.evidence.committed))


def _voronoi(n: int, seeds: np.ndarray) -> np.ndarray:
    points = np.stack(_centers(n, 2), axis=-1)
    offsets = np.abs(points[:, :, None, :] - seeds[None, None])
    offsets = np.minimum(offsets, 1.0 - offsets)
    return np.argmin(np.sum(offsets**2, axis=-1), axis=-1)


def test_energy_decreases_for_admitted_pairwise_tensions() -> None:
    labels = ("a", "b", "c")
    sigma = np.array([[0.0, 1.0, 1.2], [1.0, 0.0, 0.8], [1.2, 0.8, 0.0]])
    tension = InterfaceTensionMatrix(labels, sigma)
    reciprocal = np.where(sigma > 0.0, 1.0 / np.where(sigma > 0.0, sigma, 1.0), 0.0)
    mobility = InterfaceMobilityMatrix(labels, reciprocal)
    plan = td.ThresholdDynamicsPlan(tension, mobility, 1e-3)
    prepared = _prepare(plan, 128)
    seeds = np.random.default_rng(3).random((9, 2))
    field = _voronoi(128, seeds) % 3
    result = _run(prepared, prepared.initial_state(field), 12)
    evidence = result.evidence
    energies = np.concatenate(
        ([float(result.initial_energy)], np.asarray(evidence.energy_after))
    )

    assert bool(tension.admissibility().energy_dissipation_admitted)
    assert np.all(np.asarray(evidence.dissipation_admitted))
    assert np.all(np.asarray(evidence.status) == int(td.ThresholdDynamicsStatus.SUCCESS))
    assert np.all(np.diff(energies) <= np.asarray(evidence.energy_tolerance))
    assert energies[-1] < energies[0]
    assert int(np.sum(np.asarray(evidence.changed_sites))) > 0


def test_triple_junction_relaxes_to_herring_angles() -> None:
    # sigma_bc = sqrt(2) sigma_ab = sqrt(2) sigma_ac gives angles (90, 135, 135).
    labels = ("lens", "left", "right")
    sigma = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, ROOT2], [1.0, ROOT2, 0.0]])
    mobility = np.where(sigma > 0.0, 1.0 / np.where(sigma > 0.0, sigma, 1.0), 0.0)
    plan = td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(labels, sigma),
        InterfaceMobilityMatrix(labels, mobility),
        1e-3,
    )
    n = 256
    prepared = _prepare(plan, n)
    x, _ = _centers(n, 2)
    field = np.where(x < 0.5, 1, 2)
    field = np.where(_ball(n, 2, 0.2) == 0, 0, field)
    result = _run(prepared, prepared.initial_state(field), 10)
    final = np.asarray(result.state.labels)

    for center in ((0.5, 0.7), (0.5, 0.3)):
        angles = td.triple_junction_angles(
            final,
            (1.0, 1.0),
            (0, 1, 2),
            center=center,
            inner_radius=0.02,
            outer_radius=0.06,
        )
        np.testing.assert_allclose(angles, [90.0, 135.0, 135.0], atol=6.0)


def test_two_gaussian_kernel_realizes_pairwise_mobilities() -> None:
    labels = ("slow", "fast", "matrix")
    tension = InterfaceTensionMatrix(labels, 1.0, structure="uniform")
    mobility = InterfaceMobilityMatrix(
        labels, np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    )
    with pytest.raises(ValueError, match="two-gaussian"):
        td.ThresholdDynamicsPlan(tension, mobility, 1e-3)
    plan = td.ThresholdDynamicsPlan(tension, mobility, 5e-4, kernel_form="two-gaussian")
    n, steps = 256, 8
    prepared = _prepare(plan, n)
    x, y = _centers(n, 2)
    field = np.full((n, n), 2)
    field = np.where((x - 0.25) ** 2 + (y - 0.5) ** 2 < 0.18**2, 0, field)
    field = np.where((x - 0.75) ** 2 + (y - 0.5) ** 2 < 0.22**2, 1, field)
    result = _run(prepared, prepared.initial_state(field), steps)
    counts = (
        np.concatenate(
            (
                [np.bincount(field.reshape(-1), minlength=3)],
                np.asarray(result.evidence.label_counts),
            )
        )
        / n**2
    )
    time = 5e-4 * np.arange(steps + 1)
    slow = np.polyfit(time, counts[:, 0], 1)[0]
    fast = np.polyfit(time, counts[:, 1], 1)[0]

    np.testing.assert_allclose(slow, -2.0 * np.pi, rtol=0.1)
    np.testing.assert_allclose(fast / slow, 3.0, rtol=0.1)
    assert np.all(np.asarray(result.evidence.parameters_admissible))


def test_label_extinction_is_deterministic_and_final() -> None:
    labels = ("a", "b", "matrix")
    plan = td.ThresholdDynamicsPlan(*_uniform(labels), 2e-3)
    n = 64
    prepared = _prepare(plan, n)
    x, y = _centers(n, 2)
    field = np.full((n, n), 2)
    field = np.where((x - 0.25) ** 2 + (y - 0.5) ** 2 < 0.06**2, 0, field)
    field = np.where((x - 0.75) ** 2 + (y - 0.5) ** 2 < 0.06**2, 1, field)
    state = prepared.initial_state(field)
    result = _run(prepared, state, 10)
    extinct = np.asarray(result.evidence.extinct_labels)
    step_of = [int(np.flatnonzero(extinct[:, label])[0]) for label in (0, 1)]

    assert step_of[0] == step_of[1]
    assert np.all(np.sum(extinct, axis=0) == [1, 1, 0])
    np.testing.assert_array_equal(result.state.active_labels, [False, False, True])
    assert int(result.state.epoch) == 1
    assert np.all(np.asarray(result.state.labels) == 2)
    np.testing.assert_allclose(result.state.time, 10 * 2e-3)
    assert result.state.binding_id == state.binding_id


def test_failed_step_rolls_back_and_halts_the_run() -> None:
    prepared = _prepare(td.ThresholdDynamicsPlan(*_uniform(("in", "out")), 2e-3), 64)
    state = prepared.initial_state(_ball(64, 2, 0.3))
    broken = eqx.tree_at(lambda item: item.plan.time_step, prepared, jnp.asarray(jnp.nan))
    single = _step(broken, state)
    run = _run(broken, state, 3)

    assert int(single.status) == int(td.ThresholdDynamicsStatus.INADMISSIBLE_PARAMETERS)
    assert not bool(single.committed)
    np.testing.assert_array_equal(single.state.labels, state.labels)
    np.testing.assert_allclose(single.state.time, 0.0)
    assert int(run.committed_steps) == 0
    np.testing.assert_array_equal(run.state.labels, state.labels)
    assert single.state.binding_id == state.binding_id
    assert run.state.binding_id == state.binding_id


def test_under_resolved_kernel_commits_with_status() -> None:
    plan = td.ThresholdDynamicsPlan(*_uniform(("in", "out")), 1e-6)
    prepared = _prepare(plan, 64)
    result = _step(prepared, prepared.initial_state(_ball(64, 2, 0.3)))

    assert int(result.status) == int(td.ThresholdDynamicsStatus.UNDER_RESOLVED)
    assert bool(result.committed)
    assert float(result.evidence.resolution_ratio) < 1.0


@pytest.mark.parametrize(
    ("sigma", "message"),
    [
        (np.array([[0.0, 3.0, 1.0], [3.0, 0.0, 1.0], [1.0, 1.0, 0.0]]), "triangle"),
        (
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
            "positive tension",
        ),
    ],
)
def test_inadmissible_tensions_are_refused(sigma: np.ndarray, message: str) -> None:
    labels = ("a", "b", "c")
    with pytest.raises(ValueError, match=message):
        td.ThresholdDynamicsPlan(
            InterfaceTensionMatrix(labels, sigma),
            InterfaceMobilityMatrix(labels, 1.0, structure="uniform"),
            1e-3,
            kernel_form="two-gaussian",
        )


def test_invalid_plans_and_states_are_refused() -> None:
    tension, mobility = _uniform(("a", "b"))
    with pytest.raises(ValueError, match="same label_ids"):
        td.ThresholdDynamicsPlan(tension, _uniform(("a", "c"))[1], 1e-3)
    with pytest.raises(ValueError, match="positive"):
        td.ThresholdDynamicsPlan(tension, mobility, 0.0)
    plan = td.ThresholdDynamicsPlan(tension, mobility, 1e-3)
    with pytest.raises(MemoryError):
        td.ThresholdDynamicsPlan(
            tension,
            mobility,
            1e-3,
            resource_policy=td.ThresholdDynamicsResourcePolicy(
                maximum_working_bytes=1024
            ),
        ).prepare(td.PeriodicGridHeatKernel((64, 64), (1.0, 1.0)))
    prepared = _prepare(plan, 16)
    with pytest.raises(ValueError, match="index"):
        prepared.initial_state(np.full((16, 16), 2))
    with pytest.raises(ValueError, match="shape"):
        prepared.initial_state(np.zeros((8, 8), dtype=np.int32))
    with pytest.raises(ValueError, match="candidate_capacity|every label"):
        plan.prepare(
            td.PeriodicGridHeatKernel((16, 16), (1.0, 1.0)), candidate_capacity=2
        )


@pytest.mark.parametrize("boundary", ["potentials", "energy", "step", "run"])
def test_prepared_boundaries_refuse_reordered_label_identity(boundary: str) -> None:
    route = td.PeriodicGridHeatKernel((4,), (1.0,))
    source = td.ThresholdDynamicsPlan(*_uniform(("right", "left")), 1e-3).prepare(
        route
    )
    target = td.ThresholdDynamicsPlan(*_uniform(("left", "right")), 1e-3).prepare(
        route
    )
    state = source.initial_state(np.array([0, 0, 1, 1], dtype=np.int32))

    with pytest.raises(ValueError, match="label_ids"):
        if boundary == "run":
            target.run(state, 1)
        else:
            getattr(target, boundary)(state)


def test_equal_shaped_state_from_another_preparation_is_refused() -> None:
    route = td.PeriodicGridHeatKernel((4,), (1.0,))
    first = td.ThresholdDynamicsPlan(*_uniform(("a", "b")), 1e-3).prepare(route)
    second = td.ThresholdDynamicsPlan(
        *_uniform(("a", "b")), 1e-3, minimum_resolution_ratio=0.0
    ).prepare(route)
    state = first.initial_state(np.array([0, 0, 1, 1], dtype=np.int32))

    with pytest.raises(ValueError, match="prepared_id"):
        second.potentials(state)


def test_label_state_refuses_out_of_range_and_inactive_owners() -> None:
    prepared = _prepare(td.ThresholdDynamicsPlan(*_uniform(("a", "b")), 1e-3), 4, 1)
    state = prepared.initial_state(np.array([0, 1, 0, 1], dtype=np.int32))

    with pytest.raises(ValueError, match="index"):
        td.LabelFieldState(
            jnp.asarray([0, 2, 0, 1], dtype=jnp.int32),
            jnp.asarray([True, True]),
            state.epoch,
            state.time,
            label_ids=state.label_ids,
            route_id=state.route_id,
            prepared_id=state.prepared_id,
            site_id=state.site_id,
        )
    with pytest.raises(ValueError, match="owns sites"):
        td.LabelFieldState(
            state.labels,
            jnp.asarray([True, False]),
            state.epoch,
            state.time,
            label_ids=state.label_ids,
            route_id=state.route_id,
            prepared_id=state.prepared_id,
            site_id=state.site_id,
        )
    out_of_range = eqx.tree_at(
        lambda item: item.labels,
        state,
        jnp.asarray([0, 2, 0, 1], dtype=jnp.int32),
    )
    inactive_owner = eqx.tree_at(
        lambda item: item.active_labels,
        state,
        jnp.asarray([True, False]),
    )
    with pytest.raises(ValueError, match="index"):
        prepared.potentials(out_of_range)
    with pytest.raises(ValueError, match="owns sites"):
        prepared.step(inactive_owner)


def test_uniform_fifty_thousand_label_route_keeps_coefficients_structured() -> None:
    count = 50_000
    labels = tuple(f"label-{index}" for index in range(count))
    policy = td.ThresholdDynamicsResourcePolicy(maximum_working_bytes=16 * 1024**2)
    plan = td.ThresholdDynamicsPlan(
        *_uniform(labels),
        1e-3,
        resource_policy=policy,
        minimum_resolution_ratio=0.0,
    )
    prepared = plan.prepare(td.PeriodicGridHeatKernel((4,), (1.0,)))
    state = prepared.initial_state(
        np.array([0, count - 1, 0, count - 1], dtype=np.int32)
    )
    potentials = prepared.potentials(state)

    assert plan.decomposition().coefficients.shape == (2,)
    assert prepared.coefficient_bytes == 0
    assert prepared.working_bytes == 8 * 4 * count * np.dtype(np.float64).itemsize
    assert potentials.values.shape == (4, count)
    assert np.all(np.isfinite(np.asarray(potentials.own)))


def test_nonuniform_coefficient_storage_is_admitted_before_materialization() -> None:
    labels = ("a", "b", "c", "d")
    values = np.ones((4, 4), dtype=np.float64) - np.eye(4)
    expected = 3 * 4 * 4 * np.dtype(np.float64).itemsize
    with pytest.raises(MemoryError, match="coefficients"):
        td.ThresholdDynamicsPlan(
            InterfaceTensionMatrix(labels, values),
            InterfaceMobilityMatrix(labels, values),
            1e-3,
            resource_policy=td.ThresholdDynamicsResourcePolicy(
                maximum_working_bytes=expected - 1
            ),
        )

    plan = td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(labels, values),
        InterfaceMobilityMatrix(labels, values),
        1e-3,
        resource_policy=td.ThresholdDynamicsResourcePolicy(
            maximum_working_bytes=1024**2
        ),
    )
    prepared = plan.prepare(td.PeriodicGridHeatKernel((4,), (1.0,)))

    assert plan.coefficient_bytes == expected
    assert prepared.coefficient_bytes == expected
    assert prepared.working_bytes == expected + 8 * 4 * 4 * np.dtype(np.float64).itemsize


def test_energy_matches_interface_length() -> None:
    prepared = _prepare(td.ThresholdDynamicsPlan(*_uniform(("in", "out")), 1e-4), 256)
    energy = jax.jit(lambda item, state: item.energy(state))(
        prepared, prepared.initial_state(_ball(256, 2, 0.3))
    )

    np.testing.assert_allclose(energy, 2.0 * np.pi * 0.3, rtol=0.01)
