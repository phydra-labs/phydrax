"""Boundary trace-space capabilities of native boundary-integral owners.

Gram pairings are checked against analytic boundary integrals and independent
collapsed Gauss--Legendre quadrature; Cauchy dualities against the divergence
theorem `∫_Γ x n_x ds = |Ω|`.
"""

from __future__ import annotations

from functools import cache

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

import phydrax as phx
from phydrax.discretization import (
    BoundaryTraceSpaceCapability,
    CauchyTraceCapability,
)
from phydrax.discretization.bem import (
    BuffaChristiansenDualSpace3D,
    OrientedTriangleSurfaceComplex3D,
    prepare_buffa_christiansen_dual_3d,
    RWGSurfaceCurrentSpace3D,
)
from phydrax.operators import (
    ClosedPolygonalCurve2D,
    LaplaceSingleLayerDP0GalerkinPolicy3D,
    prepare_scalar_calderon_3d,
    prepare_scalar_laplace_galerkin_2d,
    ScalarBoundarySpaces3D,
    ScalarKernelFamily3D,
)


_Floats = npt.NDArray[np.float64]
_Ints = npt.NDArray[np.int64]

_TETRAHEDRON_VERTICES = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)
_TETRAHEDRON_FACES = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
_TETRAHEDRON_AREA = 1.5 + 0.5 * np.sqrt(3.0)
_TETRAHEDRON_VOLUME = 1.0 / 6.0


def _square(panels_per_side: int) -> _Floats:
    steps = np.linspace(-1.0, 1.0, panels_per_side + 1)[:-1]
    ones = np.ones(panels_per_side)
    return np.concatenate(
        (
            np.stack((steps, -ones), axis=1),
            np.stack((ones, steps), axis=1),
            np.stack((-steps, ones), axis=1),
            np.stack((-ones, -steps), axis=1),
        )
    )


def _polygon_cauchy(vertices: _Floats, source_id: str) -> CauchyTraceCapability:
    galerkin = prepare_scalar_laplace_galerkin_2d(
        ClosedPolygonalCurve2D(vertices, source_id=source_id)
    )
    return galerkin.spaces.cauchy_trace_capability()


def _panel_normals(vertices: _Floats) -> _Floats:
    """Unit normals of the closed polygon pointing away from its bounded side."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    signed_area = 0.5 * np.sum(
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - np.roll(vertices[:, 0], -1) * vertices[:, 1]
    )
    rotated = np.stack((edges[:, 1], -edges[:, 0]), axis=1) * np.sign(signed_area)
    return rotated / np.linalg.norm(rotated, axis=1)[:, None]


def _face_normals(vertices: _Floats, faces: _Ints) -> _Floats:
    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def _triangle_rule(corners: _Floats, order: int) -> tuple[_Floats, _Floats]:
    """Collapsed Gauss--Legendre points and area weights on one flat triangle."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    unit = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    s, v = np.meshgrid(unit, unit, indexing="ij")
    ws, wv = np.meshgrid(unit_weights, unit_weights, indexing="ij")
    t = v * (1.0 - s)
    jacobian = np.linalg.norm(np.cross(corners[1] - corners[0], corners[2] - corners[0]))
    points = (
        corners[0]
        + s.reshape((-1, 1)) * (corners[1] - corners[0])
        + t.reshape((-1, 1)) * (corners[2] - corners[0])
    )
    return points, (ws * wv * (1.0 - s)).reshape((-1,)) * jacobian


def _rwg_current_energy(
    surface: OrientedTriangleSurfaceComplex3D, coefficients: _Floats
) -> float:
    """Quadrature of `∫ |Σ c_e f_e|² dA` from the RWG definition on each face."""
    vertices = np.asarray(surface.vertices)
    faces = np.asarray(surface.triangles)
    edges = np.asarray(surface.face_edges)
    signs = np.asarray(surface.face_edge_signs, dtype=np.float64)
    opposite = np.asarray(surface.opposite_vertices)
    lengths = np.asarray(surface.edge_lengths)
    total = 0.0
    for face in range(faces.shape[0]):
        corners = vertices[faces[face]]
        points, weights = _triangle_rule(corners, 4)
        area = 0.5 * np.linalg.norm(
            np.cross(corners[1] - corners[0], corners[2] - corners[0])
        )
        current = np.zeros_like(points)
        for local in range(3):
            edge = edges[face, local]
            scale = signs[face, local] * lengths[edge] / (2.0 * area)
            current += (
                coefficients[edge] * scale * (points - vertices[opposite[face, local]])
            )
        total += float(np.sum(weights * np.sum(current * current, axis=1)))
    return total


def _policy_3d() -> LaplaceSingleLayerDP0GalerkinPolicy3D:
    return LaplaceSingleLayerDP0GalerkinPolicy3D(
        regular_order=3,
        singular_order=3,
        near_order=3,
        near_ratio=1.0,
        absolute_tolerance=5.0e-2,
        relative_tolerance=5.0e-2,
        target_block_size=3,
        source_block_size=2,
    )


@cache
def _tetrahedron_spaces(kernel: str) -> ScalarBoundarySpaces3D:
    family = (
        ScalarKernelFamily3D.laplace()
        if kernel == "laplace"
        else ScalarKernelFamily3D.outgoing_helmholtz(1.0)
    )
    region = phx.geometry.MeshRegion(
        jnp.asarray(_TETRAHEDRON_VERTICES),
        jnp.asarray(_TETRAHEDRON_FACES, dtype=jnp.int32),
    )
    return prepare_scalar_calderon_3d(region, kernel=family, policy=_policy_3d()).spaces


@cache
def _tetrahedron_rwg() -> RWGSurfaceCurrentSpace3D:
    return RWGSurfaceCurrentSpace3D(
        OrientedTriangleSurfaceComplex3D(_TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES)
    )


def test_polygon_cauchy_traces_pair_by_arc_length_and_divergence_theorem() -> None:
    vertices = _square(3)
    cauchy = _polygon_cauchy(vertices, "square-3")
    dirichlet, neumann = cauchy.dirichlet, cauchy.neumann
    assert (
        dirichlet.quantity,
        dirichlet.representation,
        dirichlet.conformity,
        dirichlet.orientation,
        dirichlet.entity_kind,
        dirichlet.entity_count,
    ) == ("dirichlet", "continuous-p1", "H^(1/2)(Gamma)", "unoriented", "vertex", 12)
    assert (
        neumann.quantity,
        neumann.representation,
        neumann.conformity,
        neumann.orientation,
        neumann.entity_kind,
        neumann.entity_count,
    ) == ("neumann", "dp0", "H^(-1/2)(Gamma)", "outward", "panel", 12)
    assert cauchy.interior == "bounded-side"
    assert (dirichlet.ambient_dimension, dirichlet.boundary_dimension) == (2, 1)

    # ∫_Γ (x + 2)(y + 3) ds = 6 |Γ| on the centered square of perimeter 8.
    shifted_x = jnp.asarray(vertices[:, 0] + 2.0)
    shifted_y = jnp.asarray(vertices[:, 1] + 3.0)
    assert float(dirichlet.gram_space.inner(shifted_x, shifted_y)) == pytest.approx(
        48.0, rel=1.0e-13
    )
    assert float(jnp.dot(shifted_x, shifted_y)) != pytest.approx(48.0)
    assert float(jnp.dot(dirichlet.mass.mv(shifted_x), shifted_y)) == pytest.approx(
        48.0, rel=1.0e-13
    )
    np.testing.assert_allclose(
        dirichlet.gram_space.inverse_riesz(dirichlet.mass.mv(shifted_x)),
        shifted_x,
        rtol=1.0e-10,
    )

    normals = _panel_normals(vertices)
    normal_x = jnp.asarray(normals[:, 0])
    normal_y = jnp.asarray(normals[:, 1])
    # ∫ n_x² ds = 4 (left and right sides), ∫ n_x n_y ds = 0.
    assert float(neumann.gram_space.inner(normal_x, normal_x)) == pytest.approx(4.0)
    assert float(neumann.gram_space.inner(normal_x, normal_y)) == pytest.approx(
        0.0, abs=1.0e-14
    )
    # ∫_Γ x ∂x/∂n ds = |Ω| = 4 for the outward normal.
    x = jnp.asarray(vertices[:, 0])
    assert float(cauchy.pair(normal_x, x)) == pytest.approx(4.0, rel=1.0e-13)
    assert float(jnp.dot(cauchy.duality.transpose_mv(normal_x), x)) == pytest.approx(
        4.0, rel=1.0e-13
    )


def test_polygon_traversal_reversal_keeps_the_outward_neumann_trace() -> None:
    counterclockwise = _square(2)
    clockwise = counterclockwise[::-1].copy()
    forward = _polygon_cauchy(counterclockwise, "square")
    reversed_ = _polygon_cauchy(clockwise, "square")
    assert reversed_.neumann.orientation == forward.neumann.orientation == "outward"
    assert reversed_.interior == forward.interior
    assert reversed_.convention_id == forward.convention_id
    assert reversed_.dirichlet.revision_id != forward.dirichlet.revision_id
    for vertices, cauchy in ((counterclockwise, forward), (clockwise, reversed_)):
        flux = jnp.asarray(_panel_normals(vertices)[:, 1])
        y = jnp.asarray(vertices[:, 1])
        # ∫_Γ y ∂y/∂n ds = |Ω| regardless of the declared traversal.
        assert float(cauchy.pair(flux, y)) == pytest.approx(4.0, rel=1.0e-13)


def test_polygon_revision_follows_moved_vertices() -> None:
    vertices = _square(2)
    original = _polygon_cauchy(vertices, "square")
    again = _polygon_cauchy(vertices.copy(), "square")
    moved_vertices = vertices.copy()
    moved_vertices[0] = (-1.25, -1.25)
    moved = _polygon_cauchy(moved_vertices, "square")
    assert again.capability_id == original.capability_id
    assert moved.dirichlet.revision_id != original.dirichlet.revision_id
    assert moved.neumann.revision_id == moved.dirichlet.revision_id
    assert moved.capability_id != original.capability_id
    ones = jnp.ones((8,))
    perimeter = float(
        np.sum(np.linalg.norm(np.roll(moved_vertices, -1, 0) - moved_vertices, axis=1))
    )
    assert float(moved.neumann.gram_space.inner(ones, ones)) == pytest.approx(
        perimeter, rel=1.0e-13
    )


def test_closed_surface_cauchy_traces_pair_by_area_and_divergence_theorem() -> None:
    spaces = _tetrahedron_spaces("laplace")
    cauchy = spaces.cauchy_trace_capability()
    dirichlet, neumann = cauchy.dirichlet, cauchy.neumann
    assert (dirichlet.entity_kind, dirichlet.entity_count) == ("vertex", 4)
    assert (neumann.entity_kind, neumann.entity_count) == ("face", 4)
    assert (dirichlet.ambient_dimension, dirichlet.boundary_dimension) == (3, 2)
    assert neumann.orientation == "outward"
    assert cauchy.interior == "bounded-side"
    assert dirichlet.coefficient_space is spaces.dirichlet_space
    assert neumann.coefficient_space is spaces.neumann_space
    assert not dirichlet.coefficient_space.compatible(neumann.coefficient_space)

    x = jnp.asarray(_TETRAHEDRON_VERTICES[:, 0])
    y = jnp.asarray(_TETRAHEDRON_VERTICES[:, 1])
    # ∫_Γ x y dA: only the z = 0 face (1/24) and the slanted face (√3/24).
    expected = (1.0 + np.sqrt(3.0)) / 24.0
    assert float(dirichlet.gram_space.inner(x, y)) == pytest.approx(expected)
    np.testing.assert_allclose(
        dirichlet.gram_space.inverse_riesz(dirichlet.mass.mv(y)),
        y,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    ones = jnp.ones((4,))
    assert float(neumann.gram_space.inner(ones, ones)) == pytest.approx(_TETRAHEDRON_AREA)
    normals = _face_normals(_TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES)
    # ∫_Γ x n_x dA = |Ω| for the outward normal of the closed surface.
    assert float(cauchy.pair(jnp.asarray(normals[:, 0]), x)) == pytest.approx(
        _TETRAHEDRON_VOLUME
    )


def test_closed_surface_cauchy_traces_keep_the_complex_kernel_dtype() -> None:
    cauchy = _tetrahedron_spaces("helmholtz").cauchy_trace_capability()
    assert cauchy.dirichlet.coefficient_space.dtype == np.complex128
    assert cauchy.neumann.gram_space.dtype == np.complex128
    normals = _face_normals(_TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES)
    x = jnp.asarray(_TETRAHEDRON_VERTICES[:, 0], dtype=jnp.complex128)
    flux = jnp.asarray((1.0 + 2.0j) * normals[:, 0])
    assert complex(cauchy.pair(flux, x)) == pytest.approx(
        (1.0 + 2.0j) * _TETRAHEDRON_VOLUME
    )


def test_closed_surface_reversed_winding_keeps_the_outward_neumann_trace() -> None:
    region = phx.geometry.MeshRegion(
        jnp.asarray(_TETRAHEDRON_VERTICES),
        jnp.asarray(_TETRAHEDRON_FACES[:, ::-1], dtype=jnp.int32),
    )
    cauchy = prepare_scalar_calderon_3d(
        region, policy=_policy_3d()
    ).spaces.cauchy_trace_capability()
    assert cauchy.neumann.orientation == "outward"
    assert cauchy.interior == "bounded-side"
    # Face order is kept, so the declared outward normals still index the panels.
    normals = _face_normals(_TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES)
    z = jnp.asarray(_TETRAHEDRON_VERTICES[:, 2])
    # ∫_Γ z ∂z/∂n dA = |Ω| although every triangle was declared inward.
    assert float(cauchy.pair(jnp.asarray(normals[:, 2]), z)) == pytest.approx(
        _TETRAHEDRON_VOLUME
    )


def test_rwg_current_space_pairs_by_area_and_is_a_distinct_representation() -> None:
    rwg = _tetrahedron_rwg()
    capability = rwg.trace_capability()
    assert (
        capability.quantity,
        capability.representation,
        capability.conformity,
        capability.orientation,
        capability.entity_kind,
        capability.entity_count,
    ) == (
        "surface-current",
        "rwg",
        "H^(-1/2)(div_Gamma)",
        "surface-oriented",
        "edge",
        6,
    )
    assert capability.coefficient_space is rwg.vector_space
    coefficients = np.asarray((0.7, -1.1, 0.4, 2.0, -0.3, 0.9))
    expected = _rwg_current_energy(rwg.surface, coefficients)
    values = jnp.asarray(coefficients, dtype=jnp.complex128)
    assert complex(capability.gram_space.inner(values, values)) == pytest.approx(
        expected, rel=1.0e-12
    )
    np.testing.assert_allclose(
        capability.gram_space.inverse_riesz(capability.mass.mv(values)),
        values,
        rtol=1.0e-10,
    )

    scalar = _tetrahedron_spaces("laplace").cauchy_trace_capability()
    assert not capability.gram_space.compatible(scalar.neumann.gram_space)
    with pytest.raises(ValueError, match="surface-current trace"):
        CauchyTraceCapability(
            scalar.dirichlet,
            capability,
            scalar.duality,
            interior="bounded-side",
            convention_id=scalar.convention_id,
        )
    with pytest.raises(ValueError, match="represents the surface-current trace"):
        BoundaryTraceSpaceCapability(
            owner_id=capability.owner_id,
            quantity="neumann",
            representation="rwg",
            coefficient_space=capability.coefficient_space,
            gram_space=capability.gram_space,
            mass=capability.mass,
            ambient_dimension=3,
            revision_id=capability.revision_id,
        )
    with pytest.raises(ValueError, match="represents the neumann trace"):
        BoundaryTraceSpaceCapability(
            owner_id=scalar.neumann.owner_id,
            quantity="surface-current",
            representation="dp0",
            coefficient_space=scalar.neumann.coefficient_space,
            gram_space=scalar.neumann.gram_space,
            mass=scalar.neumann.mass,
            ambient_dimension=3,
            revision_id=scalar.neumann.revision_id,
        )


def test_rwg_orientation_reversal_negates_currents_and_moves_revision() -> None:
    forward = _tetrahedron_rwg()
    reversed_ = RWGSurfaceCurrentSpace3D(
        OrientedTriangleSurfaceComplex3D(
            _TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES[:, ::-1].copy()
        )
    )
    forward_edges = [tuple(edge) for edge in np.asarray(forward.surface.edge_vertices)]
    reversed_edges = [tuple(edge) for edge in np.asarray(reversed_.surface.edge_vertices)]
    permutation = np.asarray([reversed_edges.index(edge) for edge in forward_edges])
    coefficients = np.asarray((0.7, -1.1, 0.4, 2.0, -0.3, 0.9))
    matched = coefficients[np.argsort(permutation)]
    forward_values = jnp.asarray(coefficients, dtype=jnp.complex128)
    reversed_values = jnp.asarray(matched, dtype=jnp.complex128)
    np.testing.assert_allclose(
        reversed_.current_at_centroids(reversed_values),
        -forward.current_at_centroids(forward_values),
        atol=1.0e-14,
    )
    forward_capability = forward.trace_capability()
    reversed_capability = reversed_.trace_capability()
    assert complex(
        reversed_capability.gram_space.inner(reversed_values, reversed_values)
    ) == pytest.approx(
        complex(forward_capability.gram_space.inner(forward_values, forward_values))
    )
    assert reversed_capability.revision_id != forward_capability.revision_id

    moved = _TETRAHEDRON_VERTICES.copy()
    moved[3] = (0.1, 0.1, 1.4)
    moved_rwg = RWGSurfaceCurrentSpace3D(
        OrientedTriangleSurfaceComplex3D(moved, _TETRAHEDRON_FACES)
    )
    moved_capability = moved_rwg.trace_capability()
    assert moved_capability.revision_id != forward_capability.revision_id
    assert complex(
        moved_capability.gram_space.inner(forward_values, forward_values)
    ) == pytest.approx(_rwg_current_energy(moved_rwg.surface, coefficients))


_OCTAHEDRON_VERTICES = np.asarray(
    (
        (1.8, -0.2, 0.3),
        (-1.6, -0.2, 0.3),
        (0.1, 0.6, 0.3),
        (0.1, -1.0, 0.3),
        (0.1, -0.2, 1.5),
        (0.1, -0.2, -0.9),
    )
)
_OCTAHEDRON_FACES = np.asarray(
    (
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    )
)


@cache
def _octahedron_rwg() -> RWGSurfaceCurrentSpace3D:
    return RWGSurfaceCurrentSpace3D(
        OrientedTriangleSurfaceComplex3D(_OCTAHEDRON_VERTICES, _OCTAHEDRON_FACES)
    )


def _bc_dual() -> BuffaChristiansenDualSpace3D:
    return prepare_buffa_christiansen_dual_3d(_octahedron_rwg())


def test_buffa_christiansen_dual_pairs_through_its_barycentric_currents() -> None:
    dual = _bc_dual()
    capability = dual.trace_capability()
    assert (
        capability.quantity,
        capability.representation,
        capability.orientation,
        capability.entity_kind,
    ) == ("surface-current-dual", "buffa-christiansen", "surface-oriented", "edge")
    primal = _octahedron_rwg().trace_capability()
    assert capability.revision_id == primal.revision_id
    coefficients = np.linspace(-1.2, 1.7, 12) ** 2 - 0.8
    values = jnp.asarray(coefficients, dtype=jnp.complex128)
    refined = np.real(np.asarray(dual.barycentric_rwg_coefficients(values)))
    expected = _rwg_current_energy(dual.barycentric_surface, refined)
    assert complex(capability.gram_space.inner(values, values)) == pytest.approx(
        expected, rel=1.0e-11
    )
    assert not capability.gram_space.compatible(primal.gram_space)
    scalar = _tetrahedron_spaces("laplace").cauchy_trace_capability()
    with pytest.raises(ValueError, match="surface-current-dual trace"):
        CauchyTraceCapability(
            capability,
            scalar.neumann,
            scalar.duality,
            interior="bounded-side",
            convention_id=scalar.convention_id,
        )
