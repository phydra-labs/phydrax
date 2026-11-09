from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax.discretization import CellGeometrySpec, CellMesh
from tools.meshing_qualification.scenarios import (
    native_hybrid_profiles as hybrid,
    reconstruction,
)


def _cube_laplacian(
    divisions: int, degree: int, /
) -> tuple[phx.linalg.LinearSystem, phx.equations.CompiledFiniteElementProblem]:
    """Dirichlet Laplacian on a Kuhn-split unit cube, declared SPD as the scenarios do."""
    axis = np.linspace(0.0, 1.0, divisions + 1, dtype=np.float64)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = grid.reshape((-1, 3))
    stride = np.asarray(((divisions + 1) ** 2, divisions + 1, 1), dtype=np.int64)
    kuhn = (
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
        ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)),
    )
    cells = np.asarray(
        [
            [int((np.asarray((i, j, k)) + np.asarray(corner)) @ stride) for corner in tet]
            for i in range(divisions)
            for j in range(divisions)
            for k in range(divisions)
            for tet in kuhn
        ],
        dtype=np.int32,
    )
    corners = points[cells]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
    cells[negative] = cells[negative][:, (0, 2, 1, 3)]
    mesh = CellMesh.from_tetrahedra(points, cells)
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("tetrahedron", degree)
        ),
        coordinate_spec=CellGeometrySpec.affine(mesh),
    ).prepare()
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "budget-cube-laplacian",
            "u",
            (phx.equations.DiffusionAction("u", 1.0),),
            properties=phx.linalg.OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={"positive_definite": "asserted"},
            ),
        ),
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=lambda values: values[..., 0] + 2.0 * values[..., 1],
        execution_policy=phx.equations.FiniteElementExecutionPolicy(realization="sparse"),
    )
    system, _ = problem.linear_system(storage="sparse")
    return system, problem


def _budgets(policy: phx.linalg.LinearSolvePolicy, /) -> set[int]:
    resources = policy.resources
    preconditioning = policy.preconditioning
    assert preconditioning is not None
    builder = preconditioning.builder
    assert isinstance(builder, phx.linalg.SparseFactorizationPreconditionerBuilder)
    factorization = builder.factorization_policy
    assert factorization.kind == "cholesky"
    assert factorization.max_factor_nnz == factorization.max_factor_bytes // 8
    return {
        resources.factorization_bytes,
        resources.workspace_bytes,
        resources.krylov_basis_bytes,
        resources.preconditioner_bytes,
        resources.recycling_state_bytes,
        factorization.max_factor_bytes,
    }


def test_every_reconstruction_profile_declares_one_solve_budget() -> None:
    profiles = {
        reconstruction._profile_id(kind, scale)
        for kind in ("lobed-sphere", "torus", "rotated-ellipsoid")
        for scale in (1, 2)
    } | {reconstruction._ENVELOPE_PROFILE}
    assert set(reconstruction._SOLVE_MEMORY_BUDGET_BYTES) == profiles
    for budget in reconstruction._SOLVE_MEMORY_BUDGET_BYTES.values():
        assert budget > 0
        assert _budgets(reconstruction._manufactured_policy(budget)) == {budget}


def test_every_layer_core_profile_declares_its_history_solve_budget() -> None:
    for controls in hybrid._LAYER_CORE_LIFECYCLES.values():
        budget = controls.solve_memory_bytes
        assert budget > 0
        assert _budgets(hybrid._harmonic_history_policy(budget)) == {budget}


def test_declared_budget_is_plan_identity_and_refuses_before_allocation() -> None:
    system, _ = _cube_laplacian(3, 1)
    budget = reconstruction._SOLVE_MEMORY_BUDGET_BYTES[reconstruction._ENVELOPE_PROFILE]
    declared = phx.linalg.plan(system, reconstruction._manufactured_policy(budget))
    assert declared.method == "pcg"
    assert (
        declared.plan_id
        == phx.linalg.plan(system, reconstruction._manufactured_policy(budget)).plan_id
    )
    assert (
        declared.plan_id
        != phx.linalg.plan(
            system, reconstruction._manufactured_policy(budget // 2)
        ).plan_id
    )
    with pytest.raises(ValueError, match="refused before allocation"):
        phx.linalg.plan(system, reconstruction._manufactured_policy(64))


def test_history_policy_solves_offset_columns_on_one_cholesky_factor() -> None:
    system, problem = _cube_laplacian(2, 2)
    rhs = jnp.asarray(problem.right_hand_side())
    prepared = phx.linalg.prepare(
        system,
        hybrid._harmonic_history_policy(1 << 20),
        rhs_layout=phx.linalg.RHSLayout((2,)),
    )
    state = prepared.preconditioning_state
    assert state is not None
    assert isinstance(state.action, phx.linalg.SparseFactorizationPreconditioner)
    assert state.action.factorization.plan.supernodal is not None
    solved = phx.linalg.solve(prepared, jnp.stack((rhs, 2.0 * rhs), axis=-1))
    assert bool(jnp.all(solved.successful))
    np.testing.assert_allclose(
        np.asarray(solved.value[:, 1]), 2.0 * np.asarray(solved.value[:, 0]), rtol=1e-12
    )


def _archive_budgets(limits: ArrayArchiveLimits, /) -> set[int]:
    return {
        limits.max_container_bytes,
        limits.max_aggregate_bytes,
        limits.max_member_bytes,
        limits.max_axis_length,
        limits.max_array_elements,
        limits.max_total_array_elements,
    }


def test_every_profile_declares_its_archive_budget() -> None:
    assert set(reconstruction._ARCHIVE_BUDGET_BYTES) == set(
        reconstruction._SOLVE_MEMORY_BUDGET_BYTES
    )
    assert set(reconstruction._ARCHIVE_MEMBER_LIMITS) == set(
        reconstruction._ARCHIVE_BUDGET_BYTES
    )
    for profile, budget in reconstruction._ARCHIVE_BUDGET_BYTES.items():
        assert _archive_budgets(reconstruction._archive_limits(profile)) == {budget}
        limits = reconstruction._archive_limits(profile)
        assert limits.max_members == reconstruction._ARCHIVE_MEMBER_LIMITS[profile]
        assert limits.max_manifest_bytes == budget // 64
        assert limits.max_central_directory_bytes == budget // 1024
    for controls in hybrid._LAYER_CORE_LIFECYCLES.values():
        assert _archive_budgets(
            hybrid._layer_core_archive_limits(controls.archive_bytes)
        ) == {controls.archive_bytes}


def test_archive_resource_plan_identity_covers_complete_policy() -> None:
    profile = reconstruction._profile_id("rotated-ellipsoid", 2)
    limits = reconstruction._archive_limits(profile)
    identity = reconstruction._resource_plan_id(profile, limits)
    assert identity == reconstruction._resource_plan_id(profile, limits)
    assert identity != reconstruction._resource_plan_id(
        profile, replace(limits, max_members=limits.max_members - 1)
    )
    assert identity != reconstruction._resource_plan_id(
        profile, replace(limits, max_manifest_bytes=limits.max_manifest_bytes - 1)
    )


def test_larger_torus_cold_reader_rejects_previous_archive_plan(
    tmp_path: Path,
) -> None:
    profile = reconstruction._profile_id("torus", 2)
    limits = reconstruction._archive_limits(profile)
    assert limits.max_members == 32769
    previous = replace(limits, max_members=8193)
    current_id = reconstruction._resource_plan_id(profile, limits)
    previous_id = reconstruction._resource_plan_id(profile, previous)
    assert current_id != previous_id
    expected: dict[str, object] = {
        "profile": profile,
        "archive_budget_bytes": reconstruction._ARCHIVE_BUDGET_BYTES[profile],
        "archive_member_limit": limits.max_members,
        "resource_plan_id": previous_id,
    }
    with pytest.raises(RuntimeError, match="resource plan"):
        reconstruction._resume_accepted_archive(
            str(tmp_path / "missing.phx"), "", expected
        )


@pytest.mark.parametrize(
    ("archive_bytes_delta", "members_delta", "message"),
    [(1, 1, "archive budget"), (0, 1, "archive member limit"), (0, 0, "resource plan")],
)
def test_cold_reader_refuses_resource_identity_before_io(
    tmp_path: Path, archive_bytes_delta: int, members_delta: int, message: str
) -> None:
    profile = reconstruction._profile_id("rotated-ellipsoid", 2)
    expected: dict[str, object] = {
        "profile": profile,
        "archive_budget_bytes": reconstruction._ARCHIVE_BUDGET_BYTES[profile]
        + archive_bytes_delta,
        "archive_member_limit": reconstruction._ARCHIVE_MEMBER_LIMITS[profile]
        + members_delta,
        "resource_plan_id": "incorrect",
    }
    with pytest.raises(RuntimeError, match=message):
        reconstruction._resume_accepted_archive(
            str(tmp_path / "missing.phx"), "", expected
        )
