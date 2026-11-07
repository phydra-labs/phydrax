#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded block-geometry worksets, deferred basis tables, and declared Hodge budgets."""

from __future__ import annotations

import itertools

import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.fem._de_rham import FiniteElementDeRhamComplex
from phydrax.exterior._form_type import FormProxy
from phydrax.linalg import LinearSolvePolicy, PCG, SolveResourcePolicy


def _kuhn_cube(count: int) -> phx.discretization.CellMesh:
    axis = np.linspace(0.0, 1.0, count + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    coordinates = grid.transpose(2, 1, 0, 3).reshape((-1, 3))

    def vertex(x: int, y: int, z: int) -> int:
        return (z * (count + 1) + y) * (count + 1) + x

    cells = []
    for x, y, z in itertools.product(range(count), repeat=3):
        for axes in itertools.permutations(range(3)):
            corner = [x, y, z]
            path = [vertex(*corner)]
            for axis_index in axes:
                corner[axis_index] += 1
                path.append(vertex(*corner))
            cells.append(path)
    tetrahedra = np.asarray(cells, dtype=np.int32)
    edges = coordinates[tetrahedra[:, 1:]] - coordinates[tetrahedra[:, :1]]
    negative = np.linalg.det(edges) < 0.0
    tetrahedra[negative, :2] = tetrahedra[negative, 1::-1]
    return phx.discretization.CellMesh.from_tetrahedra(coordinates, tetrahedra)


def _form_plan(mesh: phx.discretization.CellMesh) -> phx.discretization.FiniteElementPlan:
    proxies: tuple[FormProxy, ...] = ("scalar", "circulation", "flux", "density")
    kind = mesh.blocks[0].cell_kind
    return phx.discretization.FiniteElementPlan(
        mesh,
        tuple(
            phx.discretization.FiniteElementFieldSpec(
                f"form_{degree}",
                phx.discretization.form_element(
                    kind, degree, 2, family="trimmed", twist="untwisted", proxy=proxy
                ),
            )
            for degree, proxy in enumerate(proxies)
        ),
    )


def _geometry_values(
    discretization: phx.discretization.FiniteElementDiscretization,
) -> list[np.ndarray]:
    return [
        np.asarray(value)
        for level in discretization.block_geometries
        for geometry in level
        for value in (
            geometry.basis_values,
            geometry.physical_points,
            geometry.physical_weights,
            geometry.measure,
            *jax.tree_util.tree_leaves(geometry.gradient_table),
        )
    ]


def _mass_leaves(
    discretization: phx.discretization.FiniteElementDiscretization,
) -> list[np.ndarray]:
    return [
        np.asarray(leaf)
        for space in discretization.field_spaces
        for leaf in jax.tree_util.tree_leaves(
            discretization.assemble_field_operators(
                space.name, discretization.default_runtime
            )[0]
        )
        if isinstance(leaf, jax.Array | np.ndarray)
    ]


def test_chunked_geometry_and_deferred_basis_tables_are_bit_identical() -> None:
    # 162 tetrahedra exceed one Piola row chunk of the order-2 form tables, so a
    # one-byte workset forces per-cell geometry chunks and multi-block assembly.
    plan = _form_plan(_kuhn_cube(3))
    reference = plan.prepare()
    chunked = phx.discretization.FiniteElementDiscretization(
        plan, geometry_workset_bytes=1
    )
    deferred = phx.discretization.FiniteElementDiscretization(
        plan, basis_tables="deferred", geometry_workset_bytes=1
    )

    expected = _geometry_values(reference)
    for candidate in (chunked, deferred):
        actual = _geometry_values(candidate)
        assert len(actual) == len(expected)
        assert all(np.array_equal(left, right) for left, right in zip(actual, expected))
    expected_mass = _mass_leaves(reference)
    for candidate in (chunked, deferred):
        actual_mass = _mass_leaves(candidate)
        assert all(
            np.array_equal(left, right)
            for left, right in zip(actual_mass, expected_mass, strict=True)
        )


def test_deferred_basis_tables_retain_jacobians_not_mapped_tables() -> None:
    plan = _form_plan(_kuhn_cube(2))
    retained = plan.prepare()
    deferred = phx.discretization.FiniteElementDiscretization(
        plan, basis_tables="deferred"
    )

    def retained_bytes(
        discretization: phx.discretization.FiniteElementDiscretization,
    ) -> int:
        return sum(
            int(np.asarray(leaf).nbytes)
            for level in discretization.block_geometries[1:3]
            for geometry in level
            for leaf in jax.tree_util.tree_leaves(geometry.basis_table)
        )

    assert retained_bytes(deferred) < retained_bytes(retained)
    with pytest.raises(ValueError, match="basis_tables"):
        phx.discretization.FiniteElementDiscretization(plan, basis_tables="lazy")  # ty: ignore[invalid-argument-type]


def test_fe_de_rham_hodges_admit_with_the_declared_budget() -> None:
    mesh = _kuhn_cube(1)
    default = FiniteElementDeRhamComplex(mesh, family="trimmed", order=2)
    budget = 1 << 30
    declared = FiniteElementDeRhamComplex(
        mesh,
        family="trimmed",
        order=2,
        hodge_resources=SolveResourcePolicy(factorization_bytes=budget),
    )

    assert declared.realization_id == default.realization_id
    for left, right in zip(declared.hodges, default.hodges, strict=True):
        assert right.admission_bytes is None
        assert left.admission_bytes == budget
        assert np.array_equal(
            np.asarray(left.upper_values), np.asarray(right.upper_values)
        )
    with pytest.raises(ValueError, match="factor_bytes requires"):
        FiniteElementDeRhamComplex(
            mesh,
            family="trimmed",
            order=2,
            hodge_resources=SolveResourcePolicy(factorization_bytes=64),
        )
    with pytest.raises(ValueError, match="hodge_resources"):
        FiniteElementDeRhamComplex(
            mesh,
            family="trimmed",
            order=2,
            hodge_solve=LinearSolvePolicy(PCG()),
            hodge_resources=SolveResourcePolicy(factorization_bytes=budget),
        )
