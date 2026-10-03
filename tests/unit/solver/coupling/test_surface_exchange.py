# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Amount conservation is row reproduction, not shape or positivity inference."""

from __future__ import annotations

import functools
import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import (
    PointCloudPlan,
    prepare_point_cloud_field_reconstruction,
)
from phydrax.discretization._point_cloud_view import PointCloudReconstruction
from phydrax.discretization.meshfree._stencils import LocalStencilPolicy
from phydrax.discretization.meshfree._surface import SurfacePointCloudPlan
from phydrax.discretization.meshfree._surface_geometry import ImplicitSurfaceGeometry
from phydrax.discretization.meshfree._surface_quadrature import SurfaceQuadraturePolicy
from phydrax.geometry import Rectangle
from phydrax.interfacial_transport import AdsorptionKinetics
from phydrax.metrix import RegularLevelSetManifold
from phydrax.solver.coupling._components import VariationalComponent
from phydrax.solver.coupling._contributions import (
    ContributionEndpoint,
    ResidualContribution,
)
from phydrax.solver.coupling._laws import InterfaceConductance
from phydrax.solver.coupling._meshfree_components import MeshfreeComponent
from phydrax.solver.coupling._surface_exchange import (
    LangmuirAdsorptionFlux,
    SurfaceDeposition,
    SurfaceExchangeLaw,
)
from phydrax.sparse import SparseCoordinateOperator
from tests.unit.solver.coupling._cases import (
    build_region,
    QUADRATIC,
    Region,
    RegionSpec,
)
from tests.unit.solver.coupling.test_meshfree_components import prepared_cloud_component


BULK = ContributionEndpoint("bulk", "concentration")
SURFACE = ContributionEndpoint("surface", "concentration")
GRID = np.stack(
    [
        axis.ravel()
        for axis in np.meshgrid(*(np.linspace(0.0, 1.0, 7),) * 2, indexing="ij")
    ],
    axis=-1,
)


def grid_surface(
    points: np.ndarray, /, *, neighbors: int = 20, owner_id: str | None = None
) -> MeshfreeComponent:
    """Native point-cloud surface endpoint over ``points`` of the unit square."""
    measures = np.linspace(0.5, 1.5, points.shape[0]) / points.shape[0]
    cloud = PointCloudPlan(points, measures, neighbors=neighbors).prepare()
    view = prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=Rectangle((0.5, 0.5), (1.0, 1.0)).compile(),
        radius=0.65,
        capacity=points.shape[0],
        reconstruction="polynomial",
    )
    full = cloud.field_spaces[0].vector_space
    weights = sum(pair[1] for pair in cloud.derivative_weights)
    native = SparseCoordinateOperator(
        cloud.relation,
        -jnp.asarray(measures)[:, None] * weights,
        source=full,
        target=full,
    )
    return MeshfreeComponent(
        cloud,
        native,
        view,
        measures,
        name="surface",
        owner_id=cloud.prepared_id if owner_id is None else owner_id,
    )


def exchange_case(
    *,
    langmuir: bool = False,
    reconstruction: PointCloudReconstruction = "polynomial",
    deposition: SurfaceDeposition = "signed",
) -> tuple[SurfaceExchangeLaw, dict[str, MeshfreeComponent]]:
    bulk = prepared_cloud_component(name="bulk", reconstruction=reconstruction)
    surface = grid_surface(GRID)
    query = bulk.reconstruction.prepare_query(surface.owner.points)
    normals = np.tile([0.0, 1.0], (query.admitted_count, 1))
    flux = (
        LangmuirAdsorptionFlux(AdsorptionKinetics(2.0, 0.5, 1.0))
        if langmuir
        else InterfaceConductance(3.0)
    )
    law = SurfaceExchangeLaw(
        BULK, SURFACE, query, surface, normals, flux, deposition=deposition
    )
    return law, {"bulk": bulk, "surface": surface}


def states(
    law: SurfaceExchangeLaw, bulk: Array, surface: Array
) -> dict[tuple[str, str], Array]:
    return {
        (law.bulk.owner, law.bulk.block): bulk,
        (law.surface.owner, law.surface.block): surface,
    }


def test_signed_polynomial_query_exchange_conserves_amount_and_duality() -> None:
    law, components = exchange_case()
    # This is the real canonical polynomial fit, not an identity interpolation.
    assert np.any(np.asarray(law.query.route.weights) < -1e-8)
    assert law.evidence.deposition == "signed"
    minimum = law.evidence.minimum_query_weight
    assert minimum is not None and minimum < 0.0
    prepared = law.prepare(components, ())
    contribution = prepared.contributions[0]
    assert isinstance(contribution, ResidualContribution)
    bulk = jnp.linspace(0.2, 1.1, law.query.coefficient_shape[0])
    surface = jnp.linspace(0.6, 0.1, law.query.admitted_count)
    bulk_rows, surface_rows = contribution.residual.evaluate((bulk, surface), None)
    density = law.flux.evaluate(
        law.query.apply(bulk), surface, law.query.points, law.normals, None
    )
    np.testing.assert_allclose(
        bulk_rows, -law.query.transpose(law.measures * density), atol=1e-13
    )
    np.testing.assert_allclose(surface_rows, law.measures * density, atol=1e-13)
    np.testing.assert_allclose(
        jnp.sum(bulk_rows) + jnp.sum(surface_rows), 0.0, atol=1e-13
    )
    assert bool(law.query.duality_evidence(bulk, surface).valid)
    defects = prepared.certificate.defects(states(law, bulk, surface), (), {})
    assert defects.names == ("amount_balance", "admissibility")
    assert bool(defects.accepted(1e-10))


@functools.cache
def circle_surface(count: int, /) -> MeshfreeComponent:
    """Intrinsic meshfree curve r=0.3 about (0.5, 0.5), strictly inside the FE square."""
    center, radius = np.asarray([0.5, 0.5]), 0.3
    angle = 2 * math.pi * (np.arange(count, dtype=np.float64) + 0.25) / count
    points = center + radius * np.column_stack((np.cos(angle), np.sin(angle)))

    def constraint(point: Array) -> Array:
        offset = point - jnp.asarray(center)
        return jnp.asarray([jnp.dot(offset, offset) - radius**2])

    source = RegularLevelSetManifold(
        constraint, ambient_dimension=2, codimension=1, manifold_id="exchange-circle"
    )
    surface = SurfacePointCloudPlan(
        jnp.asarray(points),
        ImplicitSurfaceGeometry(
            source, certified_tube_radius=0.1, geometry_id="exchange-circle"
        ),
        8,
        quadrature=SurfaceQuadraturePolicy(
            "normalized-density",
            density=jnp.ones(count, dtype=jnp.float64),
            total_area=2 * math.pi * radius,
        ),
        stencil_policy=LocalStencilPolicy(polynomial_degree=2),
        require_tube=True,
    ).prepare()
    laplace = surface.laplace_beltrami
    native = SparseCoordinateOperator(
        laplace.relation,
        -surface.measures[:, None] * laplace.coefficients,
        source=laplace.source,
        target=laplace.target,
    )
    return MeshfreeComponent(
        surface,
        native,
        surface.prepare_field_reconstruction(
            support_geometry=Rectangle((0.5, 0.5), (1.0, 1.0)).compile()
        ),
        surface.measures,
        name="surface",
        owner_id=surface.prepared_id,
    )


@pytest.fixture(scope="module")
def finite_element_bulk() -> Region:
    """Native P1 triangle owner of [0, 1]^2 (no strong data), field ``u``."""
    spec = RegionSpec("bulk", "fe", 0.0, 1.0, 6, 1, dirichlet="none")
    return build_region(spec, QUADRATIC)


def finite_element_law(
    bulk: VariationalComponent, deposition: SurfaceDeposition = "signed"
) -> SurfaceExchangeLaw:
    surface = circle_surface(24)
    query = bulk.prepare_field_reconstruction("u").prepare_query(surface.owner.points)
    owner = surface.owner
    normals = np.asarray(owner.points) - 0.5
    return SurfaceExchangeLaw(
        ContributionEndpoint("bulk", "u"),
        SURFACE,
        query,
        surface,
        normals / np.linalg.norm(normals, axis=1, keepdims=True),
        InterfaceConductance(2.0),
        deposition=deposition,
    )


def test_finite_element_bulk_exchanges_conservatively_with_meshfree_curve(
    finite_element_bulk: Region,
) -> None:
    component = finite_element_bulk.component
    law = finite_element_law(component)
    surface = circle_surface(24)
    # The FE cell route publishes no gather partition to inspect.
    assert law.evidence.minimum_query_weight is None
    assert law.evidence.constant_reproduction_error < 1e-12
    prepared = law.prepare({"bulk": component, "surface": surface}, ())
    contribution = prepared.contributions[0]
    assert isinstance(contribution, ResidualContribution)
    dofs = law.query.coefficient_shape[0]
    assert component.field("u").full_space.shape == (dofs,)
    coordinates = finite_element_bulk.dof_points
    bulk = jnp.asarray(1.0 + coordinates[:, 0] - 0.5 * coordinates[:, 1] ** 2)
    curve = jnp.linspace(0.2, 0.9, law.query.admitted_count)
    bulk_rows, surface_rows = contribution.residual.evaluate((bulk, curve), None)
    assert bulk_rows.shape == (dofs,) and surface_rows.shape == (24,)
    # Bulk loss pairs with surface gain through the exact FE transpose.
    density = law.flux.evaluate(
        law.query.apply(bulk), curve, law.query.points, law.normals, None
    )
    np.testing.assert_allclose(surface_rows, law.measures * density, atol=1e-14)
    np.testing.assert_allclose(
        bulk_rows, -law.query.transpose(law.measures * density), atol=1e-14
    )
    assert abs(float(jnp.sum(surface_rows))) > 1e-3
    np.testing.assert_allclose(
        jnp.sum(bulk_rows) + jnp.sum(surface_rows), 0.0, atol=1e-14
    )
    assert bool(law.query.duality_evidence(bulk, curve).valid)
    defects = prepared.certificate.defects(
        {("bulk", "u"): bulk, ("surface", "concentration"): curve}, (), {}
    )
    assert bool(defects.accepted(1e-12))


def test_finite_element_cell_route_cannot_claim_positive_deposition(
    finite_element_bulk: Region,
) -> None:
    with pytest.raises(ValueError, match="GatherStencil"):
        finite_element_law(finite_element_bulk.component, deposition="positive")


def test_positive_deposition_needs_a_nonnegative_gather_partition() -> None:
    law, components = exchange_case(reconstruction="shepard", deposition="positive")
    minimum = law.evidence.minimum_query_weight
    assert law.evidence.deposition == "positive" and law.deposition == "positive"
    assert minimum is not None and minimum >= 0.0
    prepared = law.prepare(components, ())
    bulk = jnp.linspace(0.2, 1.1, law.query.coefficient_shape[0])
    surface = jnp.linspace(0.6, 0.1, law.query.admitted_count)
    defects = prepared.certificate.defects(states(law, bulk, surface), (), {})
    assert defects.names == ("amount_balance", "admissibility", "deposition_sign")
    assert bool(defects.accepted(1e-12))
    signed, _ = exchange_case()
    with pytest.raises(ValueError, match="Signed query weights"):
        SurfaceExchangeLaw(
            BULK,
            SURFACE,
            signed.query,
            components["surface"],
            signed.normals,
            signed.flux,
            deposition="positive",
        )
    with pytest.raises(ValueError, match="deposition"):
        SurfaceExchangeLaw(
            BULK,
            SURFACE,
            law.query,
            components["surface"],
            law.normals,
            law.flux,
            deposition="monotone",  # ty: ignore[invalid-argument-type]
        )


def test_positive_certificate_refuses_bulk_gain_while_surface_adsorbs() -> None:
    law, components = exchange_case(reconstruction="shepard", deposition="positive")
    signed, signed_components = exchange_case()
    route = np.asarray(signed.query.route.weights)
    site = int(np.argwhere(route < -1e-8)[0][0])
    # Constant bulk: only `site` exchanges, so the deposit is one-signed.
    bulk = jnp.ones(law.query.coefficient_shape)
    surface = jnp.ones(law.query.output_shape).at[site].set(0.25)
    honest = law.prepare(components, ()).certificate.defects(
        states(law, bulk, surface), (), {}
    )
    assert float(honest.value("deposition_sign")) == 0.0
    assert bool(honest.accepted(1e-12))
    # A positive declaration over a corrupted signed constant-reproducing route:
    # the amount still balances, but a bulk node gains while `site` deposits.
    forged = eqx.tree_at(lambda item: item.query, law, signed.query)
    report = forged.prepare(signed_components, ()).certificate.defects(
        states(law, bulk, surface), (), {}
    )
    assert float(report.value("amount_balance")) < 1e-13
    assert float(report.value("deposition_sign")) > 1e-6
    assert not bool(report.accepted(1e-10))
    # A signed declaration carries no such gate and accepts the same state.
    accepted = signed.prepare(signed_components, ()).certificate.defects(
        states(law, bulk, surface), (), {}
    )
    assert "deposition_sign" not in accepted.names
    assert bool(accepted.accepted(1e-10))


def test_nonconstant_and_incomplete_queries_are_refused() -> None:
    law, components = exchange_case()
    nonconstant = eqx.tree_at(
        lambda query: query.route.weights, law.query, law.query.route.weights * 0.5
    )
    with pytest.raises(ValueError, match="reproduce constants"):
        SurfaceExchangeLaw(
            BULK, SURFACE, nonconstant, components["surface"], law.normals, law.flux
        )
    incomplete = law.query.reconstruction.prepare_query(
        np.asarray([[0.4, 0.4], [5.0, 5.0]]), coverage="masked"
    )
    with pytest.raises(ValueError, match="complete"):
        SurfaceExchangeLaw(
            BULK,
            SURFACE,
            incomplete,
            components["surface"],
            np.asarray([[0.0, 1.0]]),
            law.flux,
        )


def test_stale_same_shaped_surface_source_is_refused() -> None:
    law, components = exchange_case()
    # Same points and capacity, another native owner revision.
    revised = grid_surface(GRID, neighbors=12)
    np.testing.assert_array_equal(
        revised.owner.points, components["surface"].owner.points
    )
    np.testing.assert_array_equal(revised.mass_diagonal, law.measures)
    with pytest.raises(ValueError, match="owner revision"):
        law.prepare(components | {"surface": revised}, ())
    # Equal measures and points, but declared by another owner.
    foreign = grid_surface(GRID, owner_id="another-surface-owner")
    with pytest.raises(ValueError, match="owner revision"):
        law.prepare(components | {"surface": foreign}, ())
    points = np.asarray(law.query.points).copy()
    points[24, 0] += 0.01
    moved_query = law.query.reconstruction.prepare_query(points)
    with pytest.raises(ValueError, match="point enumeration"):
        SurfaceExchangeLaw(
            BULK, SURFACE, moved_query, components["surface"], law.normals, law.flux
        )


def test_langmuir_flux_uses_native_kinetics_and_capacity_acceptance() -> None:
    law, components = exchange_case(langmuir=True)
    prepared = law.prepare(components, ())
    contribution = prepared.contributions[0]
    assert isinstance(contribution, ResidualContribution)
    bulk = jnp.full(law.query.coefficient_shape, 0.5)
    equilibrium_surface = 2.0 * 0.5 / (2.0 * 0.5 + 0.5)
    surface = jnp.full(law.query.output_shape, equilibrium_surface)
    rows = contribution.residual.evaluate((bulk, surface), None)
    np.testing.assert_allclose(rows[0], 0.0, atol=1e-13)
    np.testing.assert_allclose(rows[1], 0.0, atol=1e-13)
    invalid_surface = jnp.full(law.query.output_shape, 1.01)
    defects = prepared.certificate.defects(states(law, bulk, invalid_surface), (), {})
    assert not bool(defects.accepted(1e-10))
    saturated = prepared.certificate.defects(
        states(law, bulk, jnp.ones(law.query.output_shape)), (), {}
    )
    assert bool(saturated.accepted(1e-10))


def test_signed_reconstruction_negative_sample_refuses_langmuir_acceptance() -> None:
    law, components = exchange_case(langmuir=True)
    route = law.query.route
    row, slot = np.argwhere(np.asarray(route.weights) < -1e-8)[0]
    source = int(np.asarray(route.indices)[row, slot])
    bulk = jnp.zeros(law.query.coefficient_shape).at[source].set(1.0)
    surface = jnp.zeros(law.query.output_shape)
    assert float(jnp.min(law.query.apply(bulk))) < 0.0
    prepared = law.prepare(components, ())
    defects = prepared.certificate.defects(states(law, bulk, surface), (), {})
    assert not bool(defects.accepted(1e-10))


def test_host_refresh_keeps_deposition_rebinds_identity_and_refuses_excess_lag() -> None:
    law, components = exchange_case(reconstruction="shepard", deposition="positive")
    points = GRID.copy()
    points[:, 0] += 0.02 * points[:, 0] * (1.0 - points[:, 0])
    moved = components | {"surface": grid_surface(points)}
    refreshed = law.refresh_at_window(
        moved,
        law.normals,
        window_start=2.0,
        geometry_time=1.9,
        geometry_epoch=1,
        maximum_displacement=0.01,
        maximum_lag=0.2,
    )
    np.testing.assert_allclose(
        refreshed.evidence.displacement,
        np.max(np.linalg.norm(points - GRID, axis=1)),
        atol=1e-15,
    )
    np.testing.assert_allclose(refreshed.evidence.lag, 0.1, atol=1e-15)
    assert not refreshed.evidence.geometry_differentiated_within_window
    assert refreshed.query.query_id != law.query.query_id
    assert refreshed.deposition == refreshed.evidence.deposition == "positive"
    assert refreshed.evidence.surface_id != law.evidence.surface_id
    refreshed.prepare(moved, ())
    with pytest.raises(ValueError, match="owner revision"):
        refreshed.prepare(components, ())
    with pytest.raises(ValueError, match="owner revision"):
        law.prepare(moved, ())
    with pytest.raises(ValueError, match="lag"):
        law.refresh_at_window(
            moved,
            law.normals,
            window_start=2.0,
            geometry_time=1.0,
            geometry_epoch=1,
            maximum_displacement=0.01,
            maximum_lag=0.2,
        )
    with pytest.raises(ValueError, match="newer geometry epoch"):
        law.refresh_at_window(
            moved,
            law.normals,
            window_start=2.0,
            geometry_time=2.0,
            geometry_epoch=0,
            maximum_displacement=0.01,
            maximum_lag=0.2,
        )
    with pytest.raises(ValueError, match="epoch relocation"):
        law.refresh_at_window(
            components | {"surface": grid_surface(GRID[:-1])},
            law.normals[:-1],
            window_start=2.0,
            geometry_time=2.0,
            geometry_epoch=1,
            maximum_displacement=1.0,
            maximum_lag=0.2,
        )
