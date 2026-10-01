# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Amount conservation is row reproduction, not shape or positivity inference."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.interfacial_transport import AdsorptionKinetics
from phydrax.solver.coupling._contributions import (
    ContributionEndpoint,
    ResidualContribution,
)
from phydrax.solver.coupling._laws import InterfaceConductance
from phydrax.solver.coupling._meshfree_components import MeshfreeComponent
from phydrax.solver.coupling._surface_exchange import (
    LangmuirAdsorptionFlux,
    SurfaceExchangeLaw,
)
from tests.unit.solver.coupling.test_meshfree_components import prepared_cloud_component


def exchange_case(
    *, langmuir: bool = False
) -> tuple[SurfaceExchangeLaw, dict[str, MeshfreeComponent]]:
    bulk = prepared_cloud_component(name="bulk")
    surface = prepared_cloud_component(name="surface")
    query = bulk.reconstruction.prepare_query(bulk.owner.points)
    measures = np.asarray(surface.mass_diagonal)
    normals = np.tile([0.0, 1.0], (query.admitted_count, 1))
    flux = (
        LangmuirAdsorptionFlux(AdsorptionKinetics(2.0, 0.5, 1.0))
        if langmuir
        else InterfaceConductance(3.0)
    )
    law = SurfaceExchangeLaw(
        ContributionEndpoint("bulk", "concentration"),
        ContributionEndpoint("surface", "concentration"),
        query,
        measures,
        normals,
        flux,
    )
    return law, {"bulk": bulk, "surface": surface}


def test_signed_polynomial_query_exchange_conserves_amount_and_duality() -> None:
    law, components = exchange_case()
    # This is the real canonical polynomial fit, not an identity interpolation.
    assert np.any(np.asarray(law.query.route.weights) < -1e-8)
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
    defects = prepared.certificate.defects(
        {("bulk", "concentration"): bulk, ("surface", "concentration"): surface}, (), {}
    )
    assert bool(defects.accepted(1e-10))


def test_nonconstant_and_incomplete_queries_are_refused() -> None:
    law, _ = exchange_case()
    nonconstant = eqx.tree_at(
        lambda query: query.route.weights, law.query, law.query.route.weights * 0.5
    )
    with pytest.raises(ValueError, match="reproduce constants"):
        SurfaceExchangeLaw(
            law.bulk, law.surface, nonconstant, law.measures, law.normals, law.flux
        )
    incomplete = law.query.reconstruction.prepare_query(
        np.asarray([[0.4, 0.4], [5.0, 5.0]]), coverage="masked"
    )
    with pytest.raises(ValueError, match="complete"):
        SurfaceExchangeLaw(
            law.bulk,
            law.surface,
            incomplete,
            np.asarray([1.0]),
            np.asarray([[0.0, 1.0]]),
            law.flux,
        )


def test_equal_size_surface_endpoint_cannot_hide_geometry_or_capacity_mismatch() -> None:
    law, components = exchange_case()
    wrong_measures = SurfaceExchangeLaw(
        law.bulk, law.surface, law.query, 2 * law.measures, law.normals, law.flux
    )
    with pytest.raises(ValueError, match="native capacity"):
        wrong_measures.prepare(components, ())
    points = np.asarray(law.query.points).copy()
    points[24, 0] += 0.01
    moved_query = law.query.reconstruction.prepare_query(points)
    wrong_points = SurfaceExchangeLaw(
        law.bulk, law.surface, moved_query, law.measures, law.normals, law.flux
    )
    with pytest.raises(ValueError, match="point enumeration"):
        wrong_points.prepare(components, ())


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
    defects = prepared.certificate.defects(
        {("bulk", "concentration"): bulk, ("surface", "concentration"): invalid_surface},
        (),
        {},
    )
    assert not bool(defects.accepted(1e-10))
    saturated = prepared.certificate.defects(
        {
            ("bulk", "concentration"): bulk,
            ("surface", "concentration"): jnp.ones(law.query.output_shape),
        },
        (),
        {},
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
    defects = prepared.certificate.defects(
        {("bulk", "concentration"): bulk, ("surface", "concentration"): surface}, (), {}
    )
    assert not bool(defects.accepted(1e-10))


def test_host_refresh_records_motion_and_refuses_excess_lag() -> None:
    law, components = exchange_case()
    points = np.asarray(law.query.points).copy()
    points[:, 0] += 0.02 * points[:, 0] * (1.0 - points[:, 0])
    refreshed = law.refresh_at_window(
        components,
        points,
        law.measures,
        law.normals,
        window_start=2.0,
        geometry_time=1.9,
        geometry_epoch=1,
        maximum_displacement=0.01,
        maximum_lag=0.2,
    )
    np.testing.assert_allclose(
        refreshed.evidence.displacement,
        np.max(np.linalg.norm(points - np.asarray(law.query.points), axis=1)),
        atol=1e-15,
    )
    np.testing.assert_allclose(refreshed.evidence.lag, 0.1, atol=1e-15)
    assert not refreshed.evidence.geometry_differentiated_within_window
    assert refreshed.query.query_id != law.query.query_id
    with pytest.raises(ValueError, match="lag"):
        law.refresh_at_window(
            components,
            points,
            law.measures,
            law.normals,
            window_start=2.0,
            geometry_time=1.0,
            geometry_epoch=1,
            maximum_displacement=0.01,
            maximum_lag=0.2,
        )
    with pytest.raises(ValueError, match="newer geometry epoch"):
        law.refresh_at_window(
            components,
            points,
            law.measures,
            law.normals,
            window_start=2.0,
            geometry_time=2.0,
            geometry_epoch=0,
            maximum_displacement=0.01,
            maximum_lag=0.2,
        )
