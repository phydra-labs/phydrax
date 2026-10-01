# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Real native coupled solve, analytic equilibrium, amounts and failure admission."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from examples.meshfree_bulk_surface_exchange import (
    prepare_workflow,
    PreparedBulkSurfaceWorkflow,
    run_workflow,
)
from phydrax.discretization import (
    prepare_point_cloud_field_reconstruction,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree._surface import SurfacePointCloudPlan
from phydrax.discretization.meshfree._surface_geometry import ImplicitSurfaceGeometry
from phydrax.discretization.meshfree._surface_quadrature import SurfaceQuadraturePolicy
from phydrax.metrix import RegularLevelSetManifold
from phydrax.solver.coupling._meshfree_components import MeshfreeComponent
from phydrax.solver.coupling._surface_exchange import MeshfreeBulkSurfaceMethod
from phydrax.sparse import SparseCoordinateOperator


def test_analytic_langmuir_equilibrium_and_moving_query_ledger() -> None:
    result = run_workflow(size=128, dimension=3, seed=4)
    assert result["accepted"] is True
    assert float(result["equilibrium_error"]) < 1e-8
    assert float(result["flux_error"]) < 1e-8
    assert float(result["conservation_error"]) < 1e-10
    assert float(result["nonlinear_residual"]) < 1e-10
    assert int(result["nonlinear_iterations"]) > 0
    assert float(result["query_displacement"]) > 0
    assert 0 < float(result["window_lag"]) < 0.1
    assert float(result["window_lag_error"]) < 1e-14
    assert float(result["query_constant_error"]) < 1e-10
    assert result["geometry_differentiated_within_window"] is False


@pytest.fixture(scope="module")
def prepared() -> PreparedBulkSurfaceWorkflow:
    return prepare_workflow(size=128, dimension=3, seed=0)


def test_surface_numeric_revision_refuses_stale_reconstruction(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    original = prepared.surface
    revised = SurfacePointCloudPlan(
        original.points,
        original.plan.geometry,
        original.plan.neighbors,
        quadrature=SurfaceQuadraturePolicy("supplied", measures=2 * original.measures),
        stencil_policy=original.plan.stencil_policy,
    ).prepare()
    laplace = revised.laplace_beltrami
    operator = SparseCoordinateOperator(
        laplace.relation,
        -revised.measures[:, None] * laplace.coefficients,
        source=laplace.source,
        target=laplace.target,
    )
    with pytest.raises(ValueError, match="owner revision"):
        MeshfreeComponent(
            revised,
            operator,
            prepared.surface_component.reconstruction,
            revised.measures,
            name="surface",
            owner_id=revised.prepared_id,
        )


def test_surface_source_program_rebinding_refuses_stale_reconstruction(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    original = prepared.surface
    geometry = original.plan.geometry
    if not isinstance(geometry, ImplicitSurfaceGeometry):
        raise TypeError("This workflow requires its declared implicit source.")
    source = geometry.source
    if not isinstance(source, RegularLevelSetManifold):
        raise TypeError("This workflow requires its native level-set manifold.")
    # Keep nominal labels, sampled geometry, measures and operators unchanged;
    # only the actual bound source program is replaced. A source rebind cannot
    # reuse the old reconstruction just because cached point arrays coincide.
    replacement_source = eqx.tree_at(
        lambda manifold: manifold.constraint,
        source,
        lambda point: 3.0 * source.constraint(point),
    )
    replacement_geometry = eqx.tree_at(
        lambda owner: owner.source, geometry, replacement_source
    )
    revised = eqx.tree_at(
        lambda owner: owner.plan.geometry, original, replacement_geometry
    )
    with pytest.raises(ValueError, match="owner revision"):
        MeshfreeComponent(
            revised,
            prepared.surface_component.native_operator,
            prepared.surface_component.reconstruction,
            revised.measures,
            name="surface",
            owner_id=revised.prepared_id,
        )


def test_overcapacity_native_step_is_rejected_without_spending_amounts(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    method = prepared.method
    bulk, _ = prepared.initial
    surface = 1.01 * method.surface_measures
    result = method.step(
        jnp.asarray(0), jnp.asarray(0.0), (bulk, surface), jnp.asarray(1.0), None
    )
    assert not bool(result.successful)
    np.testing.assert_array_equal(result.accepted_state[0], bulk)
    np.testing.assert_array_equal(result.accepted_state[1], surface)


def test_signed_polynomial_query_cannot_claim_positive_native_amount_partition(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    reconstruction = prepared.method.query.reconstruction
    cloud = prepared.bulk.owner
    if not isinstance(cloud, PreparedPointCloudDiscretization):
        raise TypeError(
            "The workflow bulk equation must use the native prepared point cloud."
        )
    kinetics = prepared.method.transport.structure.kinetics
    if kinetics is None:
        raise TypeError("Langmuir exchange requires its actual kinetic owner.")
    polynomial = prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=reconstruction.support_geometry,
        radius=1.5,
        capacity=cloud.points.shape[0],
        reconstruction="polynomial",
    ).prepare_query(prepared.graph.points)
    assert np.any(np.asarray(polynomial.route.weights) < -1e-8)
    np.testing.assert_allclose(
        polynomial.apply(jnp.ones(cloud.state_shape)), 1.0, atol=1e-12, rtol=0.0
    )
    with pytest.raises(ValueError, match="Signed query weights"):
        MeshfreeBulkSurfaceMethod(
            polynomial, prepared.graph, prepared.method.bulk_volumes, kinetics
        )


def test_workflow_refuses_wrong_ambient_geometry() -> None:
    with pytest.raises(ValueError, match="dimension=3"):
        prepare_workflow(size=128, dimension=2)
