#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.geometry._contracts import CompiledGeometry
from phydrax.geometry._mesh_certificates import MeshCertificateBinding
from phydrax.geometry.implicit._adaptive_discovery import (
    AdaptiveImplicitBoxIssue,
    AdaptiveImplicitScaffold,
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfaceArchiveRecipe,
    AdaptiveImplicitSurfaceStatus,
    discover_adaptive_implicit_surface,
    ImplicitVolumeClass,
)
from phydrax.geometry.implicit._policy import AdaptiveImplicitSurfacePolicy


_G = phx.geometry


def _discover(
    geometry: CompiledGeometry,
    domain: np.ndarray,
    /,
    *,
    relative_cell_diameter: float,
    allow_nonsmooth_field: bool = False,
) -> AdaptiveImplicitSurface:
    return discover_adaptive_implicit_surface(
        geometry,
        domain=domain,
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=relative_cell_diameter,
            allow_nonsmooth_field=allow_nonsmooth_field,
        ),
        source_id="adaptive-implicit-test",
    )


def _contains(boxes: np.ndarray, point: np.ndarray) -> bool:
    return bool(np.any(np.all((boxes[:, 0] <= point) & (point <= boxes[:, 1]), axis=1)))


def test_adaptive_enclosure_finds_hidden_component_and_certifies_source_cover() -> None:
    center = np.asarray((0.37, 0.13, -0.36))
    radius = 0.04
    geometry = _G.Sphere(center, radius).compile()
    domain = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))

    coarse = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(domain))
    with pytest.raises(ValueError, match="no surface crossings"):
        _G.discover_implicit_surface(geometry, coarse, source_id="fixed-grid-test")

    result = _discover(geometry, domain, relative_cell_diameter=0.005)
    scaffold = result.scaffold
    diagnostic = {
        "status_flags": result.evidence.status_flags,
        "unresolved_count": result.evidence.unresolved_count,
        "unresolved_issue_counts": np.unique(
            result.evidence.unresolved_issues, return_counts=True
        ),
        "cell_issue_counts": np.unique(scaffold.cell_issues, return_counts=True),
        "root_issue_counts": np.unique(scaffold.root_issues, return_counts=True),
    }
    mesh = result.mesh
    assert mesh is not None, diagnostic
    assert mesh.topology.num_face_components == 1, diagnostic
    assert result.region_ids == ("interior",), diagnostic
    assert result.domain_id

    assert result.cover is not None and result.cover.complete and result.cover.certified
    assert result.evidence.accuracy == "enclosed"
    assert result.topology is None
    assert result.policy.relative_cell_diameter == 0.005
    assert isinstance(result.scaffold, AdaptiveImplicitScaffold)
    assert scaffold.corner_coordinates.shape == scaffold.corner_ids.shape
    assert scaffold.corner_value_lower.shape == scaffold.corner_signs.shape
    assert scaffold.tetrahedra.shape[0] == 6 * scaffold.cell_indices.shape[0]
    assert scaffold.triangle_cell_rows.shape == (mesh.faces.shape[0],)
    assert scaffold.triangle_root_edge_rows.shape == mesh.faces.shape
    assert np.all(scaffold.corner_signs != 0)
    endpoint_negative_positive = (
        (scaffold.root_endpoint_value_upper[:, 0] < 0.0)
        & (scaffold.root_endpoint_value_lower[:, 1] > 0.0)
    ) | (
        (scaffold.root_endpoint_value_upper[:, 1] < 0.0)
        & (scaffold.root_endpoint_value_lower[:, 0] > 0.0)
    )
    assert np.all(endpoint_negative_positive)
    assert np.all(scaffold.root_issues == 0)
    assert any(leaf is scaffold.root_edge_ids for leaf in jax.tree_util.tree_leaves(scaffold))
    recipe = result.archive_recipe
    assert isinstance(recipe, AdaptiveImplicitSurfaceArchiveRecipe)
    assert recipe.geometry is result.geometry
    assert np.array_equal(recipe.domain, result.volume.domain)
    assert recipe.policy is result.policy
    assert recipe.source_id == result.source_id
    source = result.boundary_source
    binding_mesh = CellMesh.from_triangles(
        mesh.vertices, mesh.faces.astype(np.int32)
    )
    binding_geometry = CellGeometrySpec.affine(binding_mesh)
    binding = MeshCertificateBinding(
        binding_mesh,
        binding_geometry,
        "affine",
        source=source,
    )
    binding.require(binding_mesh, binding_geometry, source=source)
    samples = source.boundary_cover(0.02)
    assert samples.semantics == "certified" and samples.complete
    assert samples.points.shape[0] > 0
    distance = np.abs(np.linalg.norm(samples.points - center, axis=1) - radius)
    assert np.max(distance - samples.point_error) <= 1.0e-12
    assert np.all(np.isfinite(samples.covering_radius))


def test_truncated_discovery_domain_does_not_expose_a_volume_region() -> None:
    geometry = _G.Sphere((0.0, 0.0, 0.0), 0.8).compile()
    domain = np.asarray(((-0.4, -0.4, -0.4), (0.4, 0.4, 0.4)))

    result = _discover(geometry, domain, relative_cell_diameter=0.02)

    assert AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING in result.evidence.status_flags
    assert result.region_ids == ()


def test_adaptive_tangent_singularity_remains_an_explicit_unresolved_box() -> None:
    geometry = (
        _G.Sphere((-0.3, 0.0, 0.0), 0.3)
        & _G.Sphere((0.3, 0.0, 0.0), 0.3)
    ).compile()
    domain = np.asarray(((-1.013, -1.013, -1.013), (1.013, 1.013, 1.013)))

    result = _discover(
        geometry,
        domain,
        relative_cell_diameter=0.015,
        allow_nonsmooth_field=True,
    )

    assert result.mesh is None
    assert result.region_ids == ()
    assert result.cover is not None and result.cover.certified
    assert AdaptiveImplicitSurfaceStatus.UNRESOLVED_BOXES in result.evidence.status_flags
    assert np.any(
        result.evidence.unresolved_issues & int(AdaptiveImplicitBoxIssue.SINGULAR)
    )
    assert _contains(result.evidence.unresolved_boxes, np.zeros(3))


def test_adaptive_resolution_separates_a_thin_gap_between_components() -> None:
    geometry = (
        _G.Sphere((-0.41, 0.013, 0.007), 0.4)
        | _G.Sphere((0.41, 0.013, 0.007), 0.4)
    ).compile()
    domain = np.asarray(((-1.1, -1.1, -1.1), (1.1, 1.1, 1.1)))

    result = _discover(
        geometry,
        domain,
        relative_cell_diameter=0.004,
        allow_nonsmooth_field=True,
    )

    diagnostic = {
        "status_flags": result.evidence.status_flags,
        "unresolved_issue_counts": np.unique(
            result.evidence.unresolved_issues, return_counts=True
        ),
        "unresolved_boxes": result.evidence.unresolved_boxes,
        "cell_issue_counts": np.unique(result.scaffold.cell_issues, return_counts=True),
        "root_issue_counts": np.unique(result.scaffold.root_issues, return_counts=True),
    }
    assert result.cover is not None and result.cover.complete and result.cover.certified
    assert result.policy.relative_cell_diameter == 0.004
    assert result.evidence.unresolved_count == 0, diagnostic
    assert result.mesh is not None, diagnostic
    gap_midpoint = np.asarray((0.0, 0.013, 0.007))
    assert result.volume.classify_points(gap_midpoint[None, :]).classes.tolist() == [
        int(ImplicitVolumeClass.OUTSIDE)
    ]
    samples = result.boundary_source.boundary_cover(0.01)
    assert samples.complete and samples.semantics == "certified"
