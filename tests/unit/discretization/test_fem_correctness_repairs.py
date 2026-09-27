#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.discretization.fem._sbp import (
    MappedTensorMetricPlan,
    TensorGLLSBPPlan,
)


def _field() -> Any:
    return phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", 1),
    )


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
