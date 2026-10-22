# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Cylinder-source Q1 partitions with compatible circulation and flux."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization._cell_geometry_transition_types import (
    NestedReferenceWitnesses,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import evaluate
from phydrax.discretization._nested_reference import prepare_nested_reference
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import form_element, prepare_nested_field_transfer
from phydrax.discretization.fem._exact_form_moments import reference_arguments
from phydrax.discretization.fem._reference_rules import degree_aware_reference_rule
from phydrax.exterior._form_type import FormProxy, FormTwist
from phydrax.linalg import DenseLinearOperator, FactorizationPolicy, factorize, RHSLayout
from tests._support.rational_mapped_cells import cylinder_source, six_quad_source


# Unit-scale accuracy allocations are 2**22 binary64 eps for two moment
# solves and 2**15 eps for six-cell contractions/adjoints. Preserve the
# stricter decimal accuracy requirements of the analytic cylinder scenario.
_MOMENT_SOLVE_ACCURACY = 5e-10
_CONTRACTION_ACCURACY = 5e-12


def _density_content(
    space: FiniteElementDiscretization, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Independent exterior derivative integration, including every source cell."""
    content = np.zeros(2, dtype=np.float64)
    for element, routes, transforms in zip(
        space.elements[0],
        space.dof_maps[0].cell_dofs,
        space.dof_maps[0].cell_transforms,
        strict=True,
    ):
        basis = element.form_basis
        if basis is None:
            raise AssertionError("fixture.basis must be present; got None")
        points, weights = degree_aware_reference_rule(element.cell_kind, 10)
        _, gradient = basis.tabulate_components(points)
        density = np.asarray(gradient)[..., 1, 0] - np.asarray(gradient)[..., 0, 1]
        for route, transform in zip(
            np.asarray(routes), np.asarray(transforms), strict=True
        ):
            content += np.einsum(
                "q,qn,nk->k", np.asarray(weights), density, transform @ values[route]
            )
    return content


@pytest.mark.parametrize(
    ("twist", "proxy"), (("untwisted", "circulation"), ("twisted", "flux"))
)
def test_authored_six_quad_compatible_refine_complete_coarsen(
    twist: FormTwist,
    proxy: FormProxy,
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, geometry, root, controls = cylinder_source()
    fine_mesh, fine_geometry = six_quad_source(mesh, geometry, root, controls)
    parents = np.zeros(6, dtype=np.int64)
    pairs = prepare_nested_reference(
        mesh,
        geometry,
        fine_mesh,
        fine_geometry,
        parent_cells=parents,
        rooted=True,
        budget=budget,
    ).pairs
    vertices = reference_cell_topology("quadrilateral").vertices
    corners = np.asarray(
        [
            [
                [
                    float(
                        evaluate(
                            value,
                            tuple(Fraction(float(x)) for x in vertex),
                            budget=budget,
                        )
                    )
                    for value in reference_arguments(pair, budget=budget)
                ]
                for vertex in vertices
            ]
            for pair in pairs
        ],
        dtype=np.float64,
    )
    fine_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in fine_mesh.blocks]
    )
    witness = NestedReferenceWitnesses(fine_ids, np.zeros(6, dtype=np.int64), corners)
    element = form_element(
        "quadrilateral", 1, 2, family="tensor-trimmed", twist=twist, proxy=proxy
    )
    source = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", element), coordinate_spec=geometry
    ).prepare()
    fine = FiniteElementPlan(
        fine_mesh,
        FiniteElementFieldSpec("u", {block.name: element for block in fine_mesh.blocks}),
        coordinate_spec=fine_geometry,
    ).prepare()
    basis = element.form_basis
    if basis is None:
        raise AssertionError("fixture.basis must be present; got None")
    points = np.asarray(basis.functional_points)
    fields = np.stack(
        (
            np.stack((np.ones(len(points)), 2 * np.ones(len(points))), axis=1),
            np.stack((-points[:, 1], points[:, 0]), axis=1),
        ),
        axis=-1,
    )
    raw = np.asarray(basis.interpolate(fields))
    factor = factorize(
        DenseLinearOperator(np.asarray(source.dof_maps[0].cell_transforms[0])[0]),
        FactorizationPolicy("svd"),
    )
    local_values = np.asarray(factor.solve(raw, rhs_layout=RHSLayout((2,))).value)
    values = np.empty_like(local_values)
    values[np.asarray(source.dof_maps[0].cell_dofs[0])[0]] = local_values
    forward = prepare_nested_field_transfer(
        source,
        fine,
        parents,
        field_name="u",
        source_geometry=geometry,
        target_geometry=fine_geometry,
    )
    reverse = prepare_nested_field_transfer(
        fine,
        source,
        None,
        field_name="u",
        source_geometry=fine_geometry,
        target_geometry=geometry,
        coarsening_witnesses=witness,
    )
    fine_values = np.asarray(forward.transfer.apply(values))
    restored = np.asarray(reverse.transfer.apply(fine_values))
    np.testing.assert_allclose(
        restored, values, atol=_MOMENT_SOLVE_ACCURACY, rtol=_MOMENT_SOLVE_ACCURACY
    )
    np.testing.assert_allclose(
        _density_content(fine, fine_values), (0.0, 2.0), atol=_CONTRACTION_ACCURACY
    )
    np.testing.assert_allclose(
        _density_content(source, restored), (0.0, 2.0), atol=_CONTRACTION_ACCURACY
    )
    dual = np.random.default_rng(31).normal(size=fine_values.shape)
    np.testing.assert_allclose(
        np.vdot(fine_values, dual),
        np.vdot(values, forward.transfer.pullback(dual)),
        atol=_CONTRACTION_ACCURACY,
    )
    np.testing.assert_allclose(
        eqx.filter_jit(forward.transfer.apply)(jnp.asarray(values)),
        fine_values,
        atol=_CONTRACTION_ACCURACY,
    )
    assert forward.evidence.passed and reverse.evidence.passed
    before = np.asarray(fine_geometry.coordinates).copy()
    gap = NestedReferenceWitnesses(
        witness.fine_cell_ids[:-1],
        witness.coarse_cell_ids[:-1],
        witness.fine_reference_vertices[:-1],
    )
    with pytest.raises(ValueError, match="complete|uncovered|cover"):
        prepare_nested_field_transfer(
            fine,
            source,
            None,
            field_name="u",
            source_geometry=fine_geometry,
            target_geometry=geometry,
            coarsening_witnesses=gap,
        )
    np.testing.assert_array_equal(fine_geometry.coordinates, before)


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=256_000_000) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=100_000_000,
            maximum_memory_bytes=256_000_000,
        )
        with ledger.activate():
            yield ledger
