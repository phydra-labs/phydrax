#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg

import phydrax.threshold_dynamics as td
from phydrax.interfacial_transport import InterfaceMobilityMatrix, InterfaceTensionMatrix
from phydrax.linalg import (
    MatrixFunctionStatus,
    TaylorExponentialPolicy,
    TaylorExponentialResourcePolicy,
)


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


def _plan(
    dt: float, *, volume_constraint: td.LabelVolumeConstraint | None = None
) -> td.ThresholdDynamicsPlan:
    labels = ("out", "in")
    return td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(labels, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(labels, 1.0, structure="uniform"),
        dt,
        volume_constraint=volume_constraint,
    )


def _square_mesh(n: int, h: float) -> tuple[np.ndarray, np.ndarray]:
    """``n x n`` grid points ``(i h, j h)`` split into right triangles."""
    axis = np.arange(n) * h
    x, y = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack((x.ravel(), y.ravel()), axis=-1)
    i, j = (
        index.ravel()
        for index in np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    )
    a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
    triangles = np.concatenate((np.stack((a, b, c), -1), np.stack((a, c, d), -1)))
    return vertices, triangles.astype(np.int32)


def _disk(vertices: np.ndarray, center: float, radius: float) -> np.ndarray:
    squared = np.sum((vertices - center) ** 2, axis=-1)
    return (squared < radius**2).astype(np.int32)


def _p1_reference(
    vertices: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Independent dense P1 stiffness and lumped mass for simplices in any embedding."""
    count = vertices.shape[0]
    stiffness = np.zeros((count, count))
    lumped = np.zeros(count)
    order = cells.shape[1] - 1
    for cell in cells:
        edges = vertices[cell[1:]] - vertices[cell[0]]
        gram = edges @ edges.T
        volume = math.sqrt(np.linalg.det(gram)) / math.factorial(order)
        tail = np.linalg.solve(gram, edges)
        gradients = np.vstack((-tail.sum(axis=0), tail))
        stiffness[np.ix_(cell, cell)] += volume * gradients @ gradients.T
        lumped[cell] += volume / (order + 1)
    return stiffness, lumped


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array(
        [[0.0, s, t * phi] for s in (-1, 1) for t in (-1, 1)]
        + [[s, t * phi, 0.0] for s in (-1, 1) for t in (-1, 1)]
        + [[t * phi, 0.0, s] for s in (-1, 1) for t in (-1, 1)]
    )
    faces = []
    for a in range(12):
        for b in range(a + 1, 12):
            for c in range(b + 1, 12):
                corners = vertices[[a, b, c]]
                sides = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=-1)
                if np.allclose(sides, 2.0):
                    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
                    faces.append((a, b, c) if normal @ corners.sum(0) > 0 else (a, c, b))
    return vertices, np.asarray(faces, dtype=np.int32)


def _cube_tetrahedra() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float
    )
    cells = np.array(
        [
            [0, 1, 3, 7],
            [0, 1, 5, 7],
            [0, 2, 3, 7],
            [0, 2, 6, 7],
            [0, 4, 5, 7],
            [0, 4, 6, 7],
        ]
    )
    corners = vertices[cells]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0
    cells[negative] = cells[negative][:, [0, 2, 1, 3]]
    return vertices, cells.astype(np.int32)


def _mean_area_rate_error(n: int, dt: float, steps: int) -> float:
    # Curve shortening: dA/dt = -2 pi mu sigma for a circle of any radius.
    h = 1.0 / (n - 1)
    vertices, triangles = _square_mesh(n, h)
    prepared = _plan(dt).prepare(td.MeshHeatKernel(vertices, triangles))
    errors = []
    for offset in (0.0, 0.25, 0.5):
        labels = _disk(vertices, 0.5 + offset * h, 0.3)
        result = _run(prepared, prepared.initial_state(labels), steps)
        assert np.all(np.asarray(result.evidence.committed))
        # The disk stays in the interior, where every lumped mass is h^2.
        counts = np.concatenate(
            ([labels.sum()], np.asarray(result.evidence.label_counts[:, 1]))
        )
        slope = np.polyfit(dt * np.arange(steps + 1), counts * h**2, 1)[0]
        errors.append(abs(slope / (-2.0 * np.pi) - 1.0))
    return float(np.mean(errors))


def test_shrinking_circle_area_rate_converges_under_joint_refinement() -> None:
    coarse = _mean_area_rate_error(17, 0.016, 3)
    medium = _mean_area_rate_error(25, 0.012, 4)
    fine = _mean_area_rate_error(33, 0.008, 6)

    assert fine < medium < coarse
    assert fine < 0.01


def test_structured_mesh_agrees_with_periodic_grid_route() -> None:
    n, h, dt, steps = 33, 1.0 / 32.0, 0.008, 4
    vertices, triangles = _square_mesh(n, h)
    labels = _disk(vertices, 0.5, 0.3)
    mesh = _plan(dt).prepare(td.MeshHeatKernel(vertices, triangles))
    periodic = _plan(dt).prepare(td.PeriodicGridHeatKernel((n, n), (n * h, n * h)))
    mesh_state = mesh.initial_state(labels)
    periodic_state = periodic.initial_state(labels.reshape(n, n))

    mesh_psi = np.asarray(mesh.potentials(mesh_state).values).reshape(-1, 2)
    periodic_psi = np.asarray(periodic.potentials(periodic_state).values).reshape(-1, 2)
    scale = np.max(np.abs(periodic_psi))
    np.testing.assert_allclose(mesh_psi, periodic_psi, atol=0.03 * scale)

    first = np.asarray(_step(mesh, mesh_state).state.labels)
    reference = np.asarray(_step(periodic, periodic_state).state.labels).reshape(-1)
    ties = np.abs(periodic_psi[:, 0] - periodic_psi[:, 1]) < 0.05 * scale
    assert np.all((first == reference) | ties)

    mesh_run = _run(mesh, mesh_state, steps)
    periodic_run = _run(periodic, periodic_state, steps)
    np.testing.assert_array_less(
        np.abs(
            np.asarray(mesh_run.evidence.label_counts)
            - np.asarray(periodic_run.evidence.label_counts)
        ),
        5,
    )
    assert int(mesh_run.committed_steps) == steps


@pytest.mark.parametrize(
    "mesh", [_icosahedron, _cube_tetrahedra, lambda: _square_mesh(5, 0.25)]
)
def test_combination_matches_dense_lumped_heat_semigroup(
    mesh: Callable[[], tuple[np.ndarray, np.ndarray]],
) -> None:
    vertices, cells = mesh()
    kernel = td.MeshHeatKernel(vertices, cells)
    stiffness, lumped = _p1_reference(vertices, cells)
    labels = (vertices[:, 0] > vertices[:, 0].mean()).astype(np.int32)
    fields = np.eye(2)[labels]
    times = np.array([0.02, 0.05])
    matrices = np.array([[[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.5], [0.5, 0.0]]])

    potentials, evidence = kernel.combine(
        jnp.asarray(fields), jnp.asarray(times), jnp.asarray(matrices)
    )
    generator = -stiffness / lumped[:, None]
    expected = sum(
        scipy.sparse.linalg.expm_multiply(time * generator, fields) @ matrix
        for time, matrix in zip(times, matrices, strict=True)
    )

    np.testing.assert_allclose(potentials, expected, atol=1e-10)
    np.testing.assert_allclose(kernel.site_measures(), lumped, rtol=1e-12)
    assert kernel.site_shape == (vertices.shape[0],)
    assert bool(evidence.successful)
    assert int(evidence.native_status) == int(MatrixFunctionStatus.SUCCESS)
    assert 0.0 <= float(evidence.error_estimate) < 1e-8
    assert not evidence.exact
    assert bool(evidence.converged)
    assert bool(evidence.derivative_valid)
    assert int(evidence.iterations) > 0
    assert int(evidence.action_matvec_count) > 0
    assert evidence.retained_storage_bytes == kernel.action.plan.retained_storage_bytes
    assert evidence.workspace_bytes == kernel.action.plan.workspace_bytes
    assert evidence.operator_id == kernel.action.operator.operator_id
    assert evidence.prepared_id == kernel.action.prepared_id


def test_unequal_lumped_masses_refuse_volume_constraints() -> None:
    vertices, triangles = _square_mesh(9, 0.125)
    kernel = td.MeshHeatKernel(vertices, triangles)
    labels = _disk(vertices, 0.5, 0.3)
    counts = np.bincount(labels, minlength=2)
    plan = _plan(0.02, volume_constraint=td.LabelVolumeConstraint(counts))

    assert not kernel.equal_site_measure
    with pytest.raises(ValueError, match="equal site measure"):
        plan.prepare(kernel)


def test_taylor_resource_policy_refuses_at_construction() -> None:
    vertices, triangles = _square_mesh(9, 0.125)
    tight = TaylorExponentialResourcePolicy(max_workspace_bytes=1024)

    with pytest.raises(ValueError, match="resource policy refuses"):
        td.MeshHeatKernel(
            vertices, triangles, policy=TaylorExponentialPolicy(resources=tight)
        )
    with pytest.raises(TypeError, match="typed PRNG key"):
        td.MeshHeatKernel(vertices, triangles, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="tetrahedra in three dimensions"):
        td.MeshHeatKernel(vertices, np.array([[0, 1, 9, 10]]))


def test_native_action_refusal_fails_the_step_and_rolls_back() -> None:
    vertices, triangles = _square_mesh(9, 0.125)
    policy = TaylorExponentialPolicy(
        norm_mode="estimate",
        resources=TaylorExponentialResourcePolicy(max_action_matvec_count=8),
    )
    kernel = td.MeshHeatKernel(vertices, triangles, policy=policy)
    prepared = _plan(0.02).prepare(kernel)
    state = prepared.initial_state(_disk(vertices, 0.5, 0.3))
    result = _step(prepared, state)
    heat = result.evidence.heat

    # A refused native action returns NaN potentials; the heat failure is the
    # reported cause and the heat evidence carries the native status.
    assert int(result.status) == int(td.ThresholdDynamicsStatus.HEAT_ACTION_FAILED)
    assert not bool(result.committed)
    np.testing.assert_array_equal(result.state.labels, state.labels)
    assert not bool(heat.successful)
    assert int(heat.native_status) == int(MatrixFunctionStatus.RESOURCE_EXHAUSTED)
    assert np.isinf(float(heat.error_estimate))
    assert heat.method == "taylor-exponential-action"
    assert not bool(heat.converged)
    assert not bool(heat.derivative_valid)
    assert heat.operator_id == kernel.action.operator.operator_id
    assert heat.prepared_id == kernel.action.prepared_id
    assert heat.retained_storage_bytes == kernel.action.plan.retained_storage_bytes
    assert heat.workspace_bytes == kernel.action.plan.workspace_bytes


def test_stored_norm_bound_is_reported_and_refuses_stiff_times() -> None:
    vertices, triangles = _square_mesh(9, 0.125)
    kernel = td.MeshHeatKernel(
        vertices, triangles, policy=TaylorExponentialPolicy(norm_mode="exact")
    )
    fields = jnp.asarray(np.eye(2)[_disk(vertices, 0.5, 0.3)])
    matrices = jnp.asarray([[[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])

    _, short = kernel.combine(fields, jnp.asarray([1e-3, 1e-3]), matrices)
    potentials, stiff = kernel.combine(fields, jnp.asarray([0.1, 0.1]), matrices)

    assert bool(short.successful)
    assert 0.0 < float(short.error_estimate) < 1e-6
    assert not bool(stiff.successful)
    assert int(stiff.native_status) == int(MatrixFunctionStatus.RESOURCE_EXHAUSTED)
    assert np.all(np.isnan(np.asarray(potentials)))
