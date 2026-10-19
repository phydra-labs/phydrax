# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Constructor revalidation and source rediscovery at adaptive restore boundaries."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.implicit._adaptive_discovery import (
    AdaptiveImplicitBoundarySource,
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfaceEvidence,
    ImplicitVolumeQuery,
)
from phydrax.units import METER
from tests._support.assertions import assert_tree_equal


@pytest.fixture(scope="module")
def discovered_surface() -> AdaptiveImplicitSurface:
    geometry = phx.geometry.Sphere(
        (0.013, -0.021, 0.007), 0.4, feature_id="stored-sphere"
    ).compile(coordinate_unit=METER)
    return phx.geometry.discover_adaptive_implicit_surface(
        geometry,
        domain=np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=np.float64),
        policy=phx.geometry.AdaptiveImplicitSurfacePolicy(maximum_level=5),
        source_id="stored-adaptive-source",
        maximum_root_solves=4096,
    )


@pytest.fixture(scope="module")
def certified_boundary_surface() -> AdaptiveImplicitSurface:
    """Use the established tiny-sphere certified-discovery scenario.

    Provenance: test_adaptive_implicit.test_certified_discovery_finds_component_inside_one_coarse_cell.
    Its source, domain and three refinement levels are retained here; the
    max-level-five restore fixture remains unchanged for six tests.
    """
    geometry = phx.geometry.Sphere(
        (0.37, 0.13, -0.36), 0.04, feature_id="stored-sphere"
    ).compile(coordinate_unit=METER)
    surface = phx.geometry.discover_adaptive_implicit_surface(
        geometry,
        domain=np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=np.float64),
        policy=phx.geometry.AdaptiveImplicitSurfacePolicy(
            initial_level=2, minimum_surface_level=2, maximum_level=8
        ),
        source_id="stored-certified-boundary",
        maximum_root_solves=4096,
    )
    assert surface.evidence.certified
    assert surface.topology is not None and surface.topology.established
    assert surface.cover is not None and surface.cover.established
    return surface


def test_volume_restore_recomputes_query_identity(
    discovered_surface: AdaptiveImplicitSurface,
) -> None:
    value = discovered_surface.volume
    assert_tree_equal(ImplicitVolumeQuery.from_stored_fields(value), value)


def test_volume_restore_refuses_incompatible_leaf_extents(
    discovered_surface: AdaptiveImplicitSurface,
) -> None:
    value = discovered_surface.volume
    corrupted = eqx.tree_at(lambda query: query.leaf_upper, value, value.leaf_upper[:-1])
    with pytest.raises(ValueError, match="leaf_upper"):
        ImplicitVolumeQuery.from_stored_fields(corrupted)


def test_evidence_restore_recomputes_evidence_identity(
    discovered_surface: AdaptiveImplicitSurface,
) -> None:
    value = discovered_surface.evidence
    assert_tree_equal(AdaptiveImplicitSurfaceEvidence.from_stored_fields(value), value)


def test_evidence_restore_refuses_incompatible_unresolved_extents(
    discovered_surface: AdaptiveImplicitSurface,
) -> None:
    value = discovered_surface.evidence
    boxes = np.zeros((value.unresolved_issues.shape[0] + 1, 2, 3), dtype=np.float64)
    corrupted = eqx.tree_at(lambda evidence: evidence.unresolved_boxes, value, boxes)
    with pytest.raises(ValueError, match="unresolved_issues"):
        AdaptiveImplicitSurfaceEvidence.from_stored_fields(corrupted)


def test_surface_restore_rediscovers_with_retained_root_capacity(
    discovered_surface: AdaptiveImplicitSurface,
) -> None:
    rebuilt = AdaptiveImplicitSurface.from_stored_fields(discovered_surface)
    assert rebuilt.root_solve_capacity == 4096
    assert_tree_equal(rebuilt, discovered_surface)


def test_surface_restore_refuses_invalid_retained_domain(
    discovered_surface: AdaptiveImplicitSurface,
) -> None:
    invalid_domain = np.asarray(((1.0, -1.0, -1.0), (-1.0, 1.0, 1.0)), dtype=np.float64)
    corrupted = eqx.tree_at(
        lambda surface: surface.volume.domain, discovered_surface, invalid_domain
    )
    with pytest.raises(ValueError, match=r"AdaptiveImplicitDiscovery\.domain"):
        AdaptiveImplicitSurface.from_stored_fields(corrupted)


def test_boundary_restore_rebinds_source_revision_and_query_limits(
    certified_boundary_surface: AdaptiveImplicitSurface,
) -> None:
    value = AdaptiveImplicitBoundarySource(
        certified_boundary_surface.geometry,
        certified_boundary_surface,
        "retained-revision",
        maximum_distance_pairs=8192,
        maximum_scratch_bytes=8192,
    )
    rebuilt = AdaptiveImplicitBoundarySource.from_stored_fields(value)
    assert_tree_equal(rebuilt, value)
    assert rebuilt.source_revision == "retained-revision"
    assert rebuilt.maximum_distance_pairs == 8192
    assert rebuilt.maximum_scratch_bytes == 8192


def test_boundary_restore_refuses_a_changed_source_state(
    certified_boundary_surface: AdaptiveImplicitSurface,
) -> None:
    value = AdaptiveImplicitBoundarySource(
        certified_boundary_surface.geometry,
        certified_boundary_surface,
        "retained-revision",
    )
    radius_index = value.geometry.schema.index(
        phx.geometry.ParameterId("stored-sphere", "radius")
    )
    changed_state = value.geometry.state.replace_at(
        radius_index,
        value.geometry.state.values[radius_index] + jnp.asarray(0.01, dtype=jnp.float64),
    )
    changed_geometry = eqx.tree_at(
        lambda geometry: geometry.state, value.geometry, changed_state
    )
    corrupted = eqx.tree_at(lambda source: source.geometry, value, changed_geometry)
    with pytest.raises(ValueError, match="state"):
        AdaptiveImplicitBoundarySource.from_stored_fields(corrupted)
