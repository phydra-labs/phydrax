#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _mesh() -> Any:
    return phx.geometry.TriangleMesh(
        jnp.asarray(
            (
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        ),
        jnp.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))),
        source_id="planar-square-film",
    )


def test_london_scenario_1() -> None:
    plan = phx.applications.superconductivity.ThinFilmLondonPlan(
        _mesh(), pearl_length=0.2, tolerance=1.0e-8
    )
    result = plan.solve(0.0)

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(result.stream_function, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.face_sheet_current, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.total_energy, 0.0, atol=1e-12)
    constraint = jnp.asarray(((1.0, -1.0, 0.0, 0.0, 0.0),))
    plan = phx.applications.superconductivity.ThinFilmLondonPlan(
        _mesh(),
        pearl_length=0.2,
        constraint_matrix=constraint,
        constraint_labels=("terminal-current",),
        tolerance=1.0e-7,
    )
    result = plan.solve(0.0, jnp.asarray((1.0,)))
    inductance, reciprocity = plan.inductance_matrix()

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(constraint @ result.stream_function, (1.0,), atol=1e-8)
    assert result.kinetic_energy >= 0.0
    assert inductance.shape == (1, 1)
    np.testing.assert_allclose(reciprocity, 0.0, atol=1e-12)


def _fan_mesh(count: int) -> phx.geometry.TriangleMesh:
    angles = np.arange(count, dtype=np.float64) * (2.0 * np.pi / count)
    vertices = np.concatenate(
        (
            np.column_stack((np.cos(angles), np.sin(angles), np.zeros(count))),
            np.zeros((1, 3)),
        )
    )
    faces = np.column_stack(
        (np.arange(count), (np.arange(count) + 1) % count, np.full(count, count))
    )
    return phx.geometry.TriangleMesh(
        vertices, faces, source_id=f"london-reference-{count}"
    )


def _dense_reference(
    plan: phx.applications.superconductivity.ThinFilmLondonPlan,
    applied: float,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    vertices, faces = np.asarray(plan.mesh.vertices), np.asarray(plan.mesh.faces)
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    area = np.linalg.norm(normals, axis=1) / 2.0
    normals /= 2.0 * area[:, None]
    basis = np.zeros((faces.shape[0], 3, vertices.shape[0]), dtype=np.float64)
    moment = np.zeros(vertices.shape[0], dtype=np.float64)
    centers = triangles.mean(axis=1)
    centered = centers - vertices.mean(axis=0)
    for f, triangle in enumerate(triangles):
        for local in range(3):
            gradient = np.cross(
                normals[f], triangle[(local + 2) % 3] - triangle[(local + 1) % 3]
            ) / (2.0 * area[f])
            current = np.cross(normals[f], gradient)
            basis[f, :, faces[f, local]] += current
            moment[faces[f, local]] += (
                0.5 * area[f] * np.dot(normals[f], np.cross(centered[f], current))
            )
    # Use the plan's material scale, not any prepared energy/action arrays.
    from phydrax._physical import ElectromagneticScaleContract

    mu = float(ElectromagneticScaleContract.si().vacuum_permeability)
    magnetic = np.zeros((vertices.shape[0], vertices.shape[0]), dtype=np.float64)
    kinetic = np.zeros_like(magnetic)
    for f in range(faces.shape[0]):
        kinetic += mu * plan.pearl_length * area[f] * (basis[f].T @ basis[f])
        for g in range(faces.shape[0]):
            radius = np.sqrt(
                np.sum((centers[f] - centers[g]) ** 2) + plan.softening_length**2
            )
            magnetic += (
                mu / (4.0 * np.pi) * area[f] * area[g] / radius * (basis[f].T @ basis[g])
            )
    hessian = magnetic + kinetic
    constraints = np.asarray(plan.constraint_matrix)
    kkt = np.block(
        [
            [hessian, constraints.T],
            [constraints, np.zeros((constraints.shape[0], constraints.shape[0]))],
        ]
    )
    solution = np.linalg.solve(kkt, np.concatenate((applied * moment, (0.0,), values)))
    stream, reaction = solution[: vertices.shape[0]], solution[vertices.shape[0] :]
    current = np.stack([b @ stream for b in basis])
    field = np.full(faces.shape[0], applied, dtype=np.float64)
    for f in range(faces.shape[0]):
        for g in range(faces.shape[0]):
            delta = centers[f] - centers[g]
            field[f] += (
                mu
                / (4.0 * np.pi)
                * area[g]
                * np.dot(normals[f], np.cross(current[g], delta))
                / (np.sum(delta**2) + plan.softening_length**2) ** 1.5
            )
    return (
        stream,
        reaction,
        field,
        float(0.5 * stream @ magnetic @ stream),
        float(0.5 * stream @ kinetic @ stream),
        float(np.linalg.eigvalsh(hessian + constraints.T @ constraints)[0]),
    )


@pytest.mark.parametrize(
    "count,route",
    (
        (4, "dense"),
        (20, "dense"),
        (80, "dense"),
        (4, "minres"),
        (20, "minres"),
        (4, "gmres"),
    ),
)
def test_london_routes_match_independent_dense_kkt(count: int, route: str) -> None:
    from phydrax.applications.superconductivity import LondonSolvePolicy, LondonSolveRoute
    from phydrax.typing import parse

    mesh = _fan_mesh(count)
    constraint = np.zeros((1, count + 1), dtype=np.float64)
    constraint[0, :2] = (1.0, -1.0)
    plan = phx.applications.superconductivity.ThinFilmLondonPlan(
        mesh,
        pearl_length=0.2,
        constraint_matrix=constraint,
        constraint_labels=("terminal",),
        solve_policy=LondonSolvePolicy(route=parse(route, LondonSolveRoute, "route")),
        tolerance=1.0e-8,
    )
    reference = _dense_reference(plan, 1.0e-7, np.asarray((0.3,)))
    result = plan.solve(1.0e-7, jnp.asarray((0.3,)))
    assert bool(result.evidence.successful)
    assert result.evidence.solve_route == route
    assert bool(result.evidence.curvature_successful)
    for actual, expected in zip(
        (
            result.stream_function,
            result.constraint_reaction,
            result.face_normal_field,
            result.magnetic_energy,
            result.kinetic_energy,
            result.evidence.minimum_energy_curvature,
        ),
        reference,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=2.0e-9, atol=2.0e-12)
    inductance, reciprocity = plan.inductance_matrix()
    zero_field = _dense_reference(plan, 0.0, np.asarray((1.0,)))
    np.testing.assert_allclose(
        inductance, -zero_field[1][1:].reshape((1, 1)), rtol=2.0e-9, atol=2.0e-12
    )
    np.testing.assert_allclose(reciprocity, 0.0, atol=1.0e-12)


def test_london_dense_byte_bound_refuses_and_auto_selects_matrix_free() -> None:
    from phydrax.applications.superconductivity import LondonSolvePolicy

    with pytest.raises(ValueError, match="byte/dof bound"):
        phx.applications.superconductivity.ThinFilmLondonPlan(
            _mesh(),
            pearl_length=0.2,
            solve_policy=LondonSolvePolicy(route="dense", dense_maximum_bytes=1),
        )
    plan = phx.applications.superconductivity.ThinFilmLondonPlan(
        _mesh(), pearl_length=0.2, solve_policy=LondonSolvePolicy(dense_maximum_bytes=1)
    )
    assert plan.solve_route == "minres"
    assert bool(plan.solve(0.0).evidence.successful)


_RESOURCE_CAPACITIES = (
    "dense_maximum_bytes",
    "dense_maximum_dofs",
    "face_tile_size",
    "krylov_restart",
    "maximum_steps",
    "workspace_bytes",
    "eigen_subspace_dimension",
)


@pytest.mark.parametrize("name", _RESOURCE_CAPACITIES)
@pytest.mark.parametrize("value", (True, False, 1.5, "8", None))
def test_london_resource_capacities_reject_nonintegral_kinds(
    name: str, value: Any
) -> None:
    with pytest.raises(TypeError, match=name):
        phx.applications.superconductivity.LondonSolvePolicy(**{name: value})


@pytest.mark.parametrize("name", _RESOURCE_CAPACITIES)
@pytest.mark.parametrize("value", (0, -1))
def test_london_resource_capacities_reject_nonpositive_values(
    name: str, value: int
) -> None:
    options: dict[str, Any] = {name: value}
    with pytest.raises(ValueError, match=name):
        phx.applications.superconductivity.LondonSolvePolicy(**options)


def test_london_eigen_subspace_requires_two_dimensions() -> None:
    with pytest.raises(ValueError, match="eigen_subspace_dimension"):
        phx.applications.superconductivity.LondonSolvePolicy(eigen_subspace_dimension=1)


@pytest.mark.parametrize("name", _RESOURCE_CAPACITIES)
def test_london_integral_capacity_normalization_preserves_identity(name: str) -> None:
    options: dict[str, Any] = {name: np.int64(8)}
    reference_options: dict[str, Any] = {name: 8}
    policy = phx.applications.superconductivity.LondonSolvePolicy(**options)
    reference = phx.applications.superconductivity.LondonSolvePolicy(**reference_options)
    assert policy.policy_id == reference.policy_id


def test_london_face_eigen_workspace_refuses_before_solve() -> None:
    with pytest.raises(ValueError, match="workspace.*byte bound"):
        phx.applications.superconductivity.ThinFilmLondonPlan(
            _mesh(),
            pearl_length=0.2,
            solve_policy=phx.applications.superconductivity.LondonSolvePolicy(
                route="minres", workspace_bytes=1
            ),
        )


def test_london_failed_multi_rhs_inductance_reports_native_status() -> None:
    plan = phx.applications.superconductivity.ThinFilmLondonPlan(
        _mesh(),
        pearl_length=0.2,
        constraint_matrix=jnp.asarray(
            ((1.0, -1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, -1.0, 0.0))
        ),
        constraint_labels=("first-winding", "second-winding"),
        solve_policy=phx.applications.superconductivity.LondonSolvePolicy(
            route="minres", maximum_steps=1
        ),
    )
    native = phx.linalg.solve(
        plan.prepared_solve,
        jnp.concatenate(
            (jnp.zeros((6, 2), dtype=jnp.float64), jnp.eye(2, dtype=jnp.float64)),
            axis=0,
        ),
    )
    assert not bool(jnp.all(native.successful))
    with pytest.raises(RuntimeError) as failure:
        plan.inductance_matrix()
    message = str(failure.value)
    assert f"status={np.asarray(native.status)!r}" in message
    assert "minres" in message
    for status in np.asarray(native.status).reshape((-1,)):
        assert phx.linalg.linear_status_message(int(status)) in message


def test_london_dense_dof_bound_refuses_preparation() -> None:
    with pytest.raises(ValueError, match="byte/dof bound"):
        phx.applications.superconductivity.ThinFilmLondonPlan(
            _mesh(),
            pearl_length=0.2,
            solve_policy=phx.applications.superconductivity.LondonSolvePolicy(
                route="dense", dense_maximum_dofs=1
            ),
        )
