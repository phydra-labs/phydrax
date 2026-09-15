#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.applications.lattice_field._gauge_fixing import (
    CoulombGaugeFixingPlan,
    LandauGaugeFixingPlan,
)
from phydrax.discretization import polygonal_cell_complex
from phydrax.graph import gauge_transform_links, MatrixGaugeLinkSpace
from phydrax.metrix import SpecialUnitaryGroup


def _pure_gauge_links():
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    space = MatrixGaugeLinkSpace(topology, SpecialUnitaryGroup(2))
    coordinates = jnp.asarray(
        [[0.35, -0.12, 0.09], [-0.21, 0.18, 0.06], [0.08, 0.04, -0.27]]
    )
    vertices = jax.vmap(lambda value: space.group.exp(space.group.hat(value)))(
        coordinates
    )
    return space, gauge_transform_links(space, space.identity(), vertices)


def test_landau_fixing_reduces_residual_and_retains_transformation_evidence():
    space, links = _pure_gauge_links()
    plan = LandauGaugeFixingPlan(
        space,
        maximum_iterations=80,
        maximum_backtracks=10,
        step_size=0.4,
        residual_tolerance=2e-5,
    )
    prepared = plan.prepare()
    initial_residual = prepared.residual(links)
    result = prepared.fix(links)

    assert result.evidence.transformation.residual < initial_residual
    assert result.evidence.transformation.reconstruction_residual < 2e-6
    assert space.contains(result.links)
    assert space.group.contains(result.transformation)
    direction = jnp.ones(result.faddeev_popov.source.shape)
    image = jax.jit(lambda value: result.faddeev_popov.mv(value))(direction)
    assert image.shape == direction.shape
    assert jnp.all(jnp.isfinite(image))
    assert jnp.allclose(image[plan.anchor_vertex], 0.0)


def test_coulomb_fixing_uses_only_explicit_spatial_edges_and_reports_gribov_copies():
    space, links = _pure_gauge_links()
    spatial = jnp.asarray([True, True, False])
    plan = CoulombGaugeFixingPlan(
        space,
        spatial,
        maximum_iterations=40,
        maximum_gribov_copies=2,
        residual_tolerance=1e-4,
    )
    identity = space.group.identity()
    alternate_coordinates = jnp.asarray(
        [[0.04, 0.02, -0.01], [-0.02, 0.03, 0.01], [0.01, -0.02, 0.02]]
    )
    alternate = jax.vmap(lambda value: space.group.exp(space.group.hat(value)))(
        alternate_coordinates
    )
    starts = jnp.stack(
        (jnp.broadcast_to(identity, (space.num_vertices, 2, 2)), alternate)
    )
    result = plan.prepare(starts).fix(links)

    assert result.evidence.gribov.functionals.shape == (2,)
    assert result.evidence.gribov.residuals.shape == (2,)
    assert result.evidence.gribov.selected_copy.shape == ()
    assert (
        result.evidence.transformation.final_functional
        >= result.evidence.transformation.initial_functional
    )
