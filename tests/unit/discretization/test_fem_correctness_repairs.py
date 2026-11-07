#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._integration_domain import IntegrationDomain
from phydrax.discretization.fem._form_elements import form_element
from phydrax.discretization.fem._sbp import (
    MappedTensorMetricPlan,
    TensorGLLSBPPlan,
)
from phydrax.equations.fem._worksets import _validate_local_region, WorksetSignature


def _field() -> Any:
    return phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", 1),
    )


@pytest.mark.parametrize("form_basis", (False, True))
@pytest.mark.parametrize(
    ("selection", "expected_rows"),
    (
        ((0, 1, 2), (0, 1, 2)),
        ((2, 0, 1), (0, 1, 2)),
        ((2, 0), (0, 2)),
        ((), ()),
    ),
)
def test_prepared_local_cell_gathers_preserve_selection_order(
    selection: tuple[int, ...], expected_rows: tuple[int, ...], form_basis: bool
) -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.0))),
        jnp.asarray(((0, 1, 2), (1, 3, 2), (1, 4, 3)), dtype=jnp.int32),
    )
    field = (
        phx.discretization.FiniteElementFieldSpec(
            "u",
            form_element(
                "simplex:2", 0, 1, family="trimmed", twist="untwisted", proxy="scalar"
            ),
        )
        if form_basis
        else _field()
    )
    space = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    base = space.cell_domain
    domain = IntegrationDomain(
        "cell",
        np.asarray(selection, dtype=np.int32),
        base.support_id,
        base.entity_set_id,
    )
    regions = space.prepare_local_regions(
        domain, field_names=("u",), maximum_derivative_order=1, kernel_mode="dense"
    )
    assert len(regions) == 1
    source = np.asarray(space.dof_maps[0].cell_dofs[0])
    expected = source[np.asarray(expected_rows, dtype=np.int32)]
    actual = regions[0].field_gathers[0]
    assert np.array_equal(np.asarray(actual), expected)
    assert actual.dtype == space.dof_maps[0].cell_dofs[0].dtype
    assert actual.device == space.dof_maps[0].cell_dofs[0].device
    assert actual.committed == space.dof_maps[0].cell_dofs[0].committed
    if form_basis:
        transforms = np.asarray(space.dof_maps[0].cell_transforms[0])
        actual_transform = regions[0].reference_actions[0].coefficient_transform
        assert np.array_equal(
            np.asarray(actual_transform), transforms[np.asarray(expected_rows, dtype=np.int32)]
        )


def test_workset_region_requires_exact_order_and_action_ids() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.0))),
        jnp.asarray(((0, 1, 2), (1, 3, 2), (1, 4, 3)), dtype=jnp.int32),
    )
    space = phx.discretization.FiniteElementPlan(mesh, _field()).prepare()
    region = space.prepare_local_regions(
        space.cell_domain, field_names=("u",), maximum_derivative_order=1,
        kernel_mode="dense",
    )[0]

    def signature(geometry_id: str, reference_ids: tuple[str, ...]) -> WorksetSignature:
        return WorksetSignature(
            "cell", region.block_name, region.cell_kind, "test-rule",
            {"u": region.reference_actions[0].local_width},
            support_id=space.cell_domain.support_id,
            entity_set_id=space.cell_domain.entity_set_id,
            reference_action_ids=reference_ids,
            field_layout_ids=("test-layout",),
            geometry_action_id=geometry_id,
            precision_id="test-precision",
            ir_semantics_id="test-semantics",
            local_kernel="dense",
        )

    geometry_id = region.geometry_actions.action_id
    reference_ids = tuple(item.action_id for item in region.reference_actions)
    correct = signature(geometry_id, reference_ids)
    _validate_local_region(correct, region, np.asarray((0, 1, 2), dtype=np.int32))
    for values in ((2, 1, 0), (0, 1), (0, 1, 2, 3)):
        with pytest.raises(ValueError, match="Prepared local region does not match its workset"):
            _validate_local_region(correct, region, np.asarray(values, dtype=np.int32))
    for wrong in (
        signature("wrong-geometry", reference_ids),
        signature(geometry_id, ("wrong-reference",)),
    ):
        with pytest.raises(ValueError, match="Prepared local region does not match its workset"):
            _validate_local_region(wrong, region, np.asarray((0, 1, 2), dtype=np.int32))


def test_fem_correctness_repairs_scenario_1() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0))),
        jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, _field()).prepare()
    region = discretization.prepare_local_regions(
        discretization.cell_domain,
        field_names=("u",),
        maximum_derivative_order=1,
        kernel_mode="dense",
    )[0]

    metric = region.geometry_actions.realize(discretization.default_runtime)

    assert metric.jacobian.shape[-2:] == (3, 2)
    assert metric.inverse_jacobian.shape[-2:] == (2, 3)
    assert jnp.all(jnp.isfinite(metric.inverse_jacobian))
    assert jnp.all(metric.physical_weights > 0.0)
    target_mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
    )
    target = phx.discretization.FiniteElementPlan(target_mesh, _field()).prepare()
    foreign_mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        jnp.asarray(((0, 1, 2), (0, 2, 3)), dtype=jnp.int32),
    )
    foreign = phx.discretization.FiniteElementPlan(
        foreign_mesh,
        _field(),
    ).prepare()

    with pytest.raises(ValueError, match="another support"):
        target.prepare_local_regions(
            foreign.cell_domain,
            field_names=("u",),
            maximum_derivative_order=1,
            kernel_mode="dense",
        )
    sbp = TensorGLLSBPPlan(2).prepare()
    x, y = jnp.meshgrid(sbp.nodes, sbp.nodes, indexing="ij")
    coordinates = jnp.stack((x, y), axis=-1)[None, ...]
    stretched = coordinates.at[..., 0].multiply(2.0)
    plan = MappedTensorMetricPlan(sbp, 2)

    first = plan.prepare(coordinates)
    second = plan.prepare(stretched)

    assert first.metrics_id != second.metrics_id
